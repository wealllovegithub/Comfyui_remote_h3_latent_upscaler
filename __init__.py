"""Offload the MiniMax H3 3D latent upscaler to a slave ComfyUI.

The slave must also have https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
and the checkpoint under models/latent_upscale_models/. This package only ships the LAN worker.
"""

import gc
import hmac
import os
import socket
import threading
import time

import folder_paths
import torch

import comfy.model_management
from .protocol import (
    CONNECT_BACKOFF,
    CONNECT_RETRIES,
    DEFAULT_PORT,
    IDLE_TIMEOUT,
    PROTOCOL_VERSION,
    SOCKET_TIMEOUT,
    UPSCALE_TIMEOUT,
    log,
    pack_tensors,
    recv_blob,
    recv_header,
    recv_packet,
    resolve_transport_dtype,
    send_packet,
    set_socket_opts,
    unpack_tensors,
)

MODES = ["megapixels", "scale by multiplier", "target dimensions"]
PRECISIONS = ["fp16", "fp32", "bf16"]
CLIP_PROTOCOL_VERSION = 2


def _model_list():
    try:
        names = folder_paths.get_filename_list("latent_upscale_models")
    except Exception:
        names = []
    names = [n for n in names if n.lower().endswith((".safetensors", ".pth", ".pt"))]
    return names or ["(put the upscaler in models/latent_upscale_models)"]


def _upscaler_module():
    """Return the LBH 3D upscaler module.

    Do not probe arbitrary modules with hasattr(). torch.ops answers yes for
    every name, and the last run called execute on that registry.
    """
    import sys
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.endswith("minimax_h3_latent_upscaler_3d"):
            continue
        cls = getattr(mod, "MinimaxH3LatentUpscaler3D", None)
        if not isinstance(cls, type) or not callable(getattr(cls, "execute", None)):
            continue
        if not callable(getattr(mod, "_resolve_device", None)):
            continue
        if not callable(getattr(mod, "load_model", None)):
            continue
        return mod
    raise RuntimeError(
        "Comfyui_Minimax_h3_latent_Upscaler is not loaded. Clone "
        "https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler "
        "into custom_nodes and restart ComfyUI."
    )


def _pin_cuda_device(mod, index):
    """LBH resolves 'cuda' to the current CUDA device. Pin it to cuda:N."""
    if getattr(mod, "_remote_h3_pin", None) == index:
        return

    def _resolve_device(backend):
        if backend == "cpu":
            return torch.device("cpu")
        if backend == "rocm":
            if getattr(torch.version, "hip", None) is None:
                raise RuntimeError("ROCm was selected, but this PyTorch build has no HIP support.")
            return torch.device("cuda")
        if backend == "cuda":
            if not torch.cuda.is_available():
                return torch.device("cpu")
            count = torch.cuda.device_count()
            if index < 0 or index >= count:
                raise RuntimeError(f"cuda:{index} is not available (device count {count})")
            return torch.device(f"cuda:{index}")
        raise ValueError(f"Unsupported device backend: {backend}")

    mod._resolve_device = _resolve_device
    mod._remote_h3_pin = index


def _empty_cuda():
    gc.collect()
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class _Worker:
    def __init__(self, model_name, cuda_device, precision, force_unload, auth_token=""):
        self.model_name = model_name
        self.cuda_device = int(cuda_device)
        self.precision = precision
        self.force_unload = bool(force_unload)
        self.auth_token = auth_token
        self._infer_lock = threading.Lock()
        self.mod = _upscaler_module()
        _pin_cuda_device(self.mod, self.cuda_device)
        if torch.cuda.is_available() and 0 <= self.cuda_device < torch.cuda.device_count():
            gpu_name = torch.cuda.get_device_name(self.cuda_device)
        else:
            gpu_name = "cpu"
        self.meta = {
            "model_name": model_name,
            "cuda_device": self.cuda_device,
            "gpu_name": gpu_name,
            "precision": precision,
            "force_unload": self.force_unload,
        }

    def upscale(self, samples, settings):
        mode_name = settings.get("mode") or "megapixels"
        if mode_name not in MODES:
            raise ValueError(f"Unknown upscale mode: {mode_name}")
        mode = {
            "mode": mode_name,
            "scale": float(settings.get("scale", 2.0)),
            "width": int(settings.get("width", 1280)),
            "height": int(settings.get("height", 704)),
            "megapixels": float(settings.get("megapixels", 1.0)),
        }
        device = "cuda" if self.cuda_device >= 0 and torch.cuda.is_available() else "cpu"
        node = self.mod.MinimaxH3LatentUpscaler3D
        with self._infer_lock:
            with torch.inference_mode():
                result = node.execute(
                    latent={"samples": samples},
                    model_name=self.model_name,
                    mode=mode,
                    align=int(settings.get("align", 32)),
                    enable_temporal_chunking=bool(settings.get("enable_temporal_chunking", True)),
                    force_unload=self.force_unload,
                    device=device,
                    precision=self.precision,
                )
        latent = result.args[0] if hasattr(result, "args") else result[0]
        return latent["samples"]

    def cleanup(self):
        with self._infer_lock:
            cache = getattr(self.mod, "MODEL_CACHE", None)
            if isinstance(cache, dict):
                for model in list(cache.values()):
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
            _empty_cuda()
        log("Upscaler worker: model off GPU")


class _Connection:
    def __init__(self, ip, port, auth_token=""):
        self.ip = ip
        self.port = int(port)
        self.auth_token = auth_token
        self._sock = None
        self._lock = threading.Lock()

    def _connect(self):
        last_err = None
        backoff = CONNECT_BACKOFF
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                sock = socket.create_connection((self.ip, self.port), timeout=SOCKET_TIMEOUT)
                set_socket_opts(sock)
                self._sock = sock
                log(f"Connected to worker {self.ip}:{self.port}")
                return
            except OSError as exc:
                last_err = exc
                log(f"Connect attempt {attempt}/{CONNECT_RETRIES} failed: {exc}")
                if attempt < CONNECT_RETRIES:
                    time.sleep(backoff)
                    backoff *= 2
        raise ConnectionError(f"Could not connect to H3 upscaler worker {self.ip}:{self.port}: {last_err}")

    def _close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def request(self, header, blob=b"", response_timeout=None, retries=2):
        if self.auth_token:
            header = {**header, "auth": self.auth_token}
        with self._lock:
            for attempt in range(1, retries + 1):
                try:
                    if self._sock is None:
                        self._connect()
                    if response_timeout is not None:
                        self._sock.settimeout(response_timeout)
                    send_packet(self._sock, header, blob)
                    resp, out_blob = recv_packet(self._sock)
                    self._sock.settimeout(IDLE_TIMEOUT)
                    return resp, out_blob
                except (ConnectionError, OSError, ValueError) as exc:
                    log(f"Request failed ({exc}); reconnecting (attempt {attempt}/{retries})")
                    self._close()
                    if attempt == retries:
                        raise
        raise ConnectionError("Remote H3 upscaler request failed")


def _release_remote_clip(ip, port, auth_token=""):
    """Ask the Remote CLIP worker to drop GPU weights before the upscale."""
    header = {"cmd": "cleanup", "proto": CLIP_PROTOCOL_VERSION, "blob_size": 0}
    if auth_token:
        header["auth"] = auth_token
    sock = socket.create_connection((ip, int(port)), timeout=30)
    try:
        set_socket_opts(sock, timeout=120)
        send_packet(sock, header)
        resp = recv_header(sock)
        recv_blob(sock, resp)
        if resp.get("error"):
            log(f"Remote CLIP cleanup {ip}:{port}: {resp['error']}")
        else:
            log(f"Remote CLIP worker {ip}:{port} released GPU")
    finally:
        try:
            sock.close()
        except OSError:
            pass


_MASTER_WORKERS = {}
_CLEANUP_FNS_ATTR = "_remote_av_cleanup_fns"


def _install_master_cleanup_hook(flush_fn):
    try:
        import execution
    except Exception:
        return
    fns = getattr(execution.PromptExecutor, _CLEANUP_FNS_ATTR, None)
    if fns is None:
        fns = []
        setattr(execution.PromptExecutor, _CLEANUP_FNS_ATTR, fns)
        orig = execution.PromptExecutor.execute

        def wrapped(self, *args, **kwargs):
            try:
                return orig(self, *args, **kwargs)
            finally:
                for fn in list(getattr(execution.PromptExecutor, _CLEANUP_FNS_ATTR, [])):
                    try:
                        fn()
                    except Exception as exc:
                        log(f"prompt-end cleanup: {exc}")

        execution.PromptExecutor.execute = wrapped
    if flush_fn not in fns:
        fns.append(flush_fn)


def _flush_workers():
    for (ip, port), conn in list(_MASTER_WORKERS.items()):
        try:
            log(f"Requesting upscaler cleanup {ip}:{port}")
            resp, _ = conn.request(
                {"cmd": "cleanup", "proto": PROTOCOL_VERSION, "blob_size": 0},
                retries=1,
            )
            if isinstance(resp, dict) and resp.get("error"):
                log(f"Upscaler cleanup error from {ip}:{port}: {resp['error']}")
            else:
                log(f"Upscaler worker {ip}:{port} released GPU")
        except Exception as exc:
            log(f"Upscaler cleanup failed {ip}:{port}: {exc}")


class SendRemoteH3LatentUpscaler:
    _servers = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_list(),),
                "cuda_device": ("INT", {"default": 0, "min": 0, "max": 7}),
                "precision": (PRECISIONS, {"default": "fp16"}),
                "listen_port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
            },
            "optional": {
                "bind_host": ("STRING", {"default": "0.0.0.0"}),
                "auth_token": ("STRING", {"default": ""}),
                "force_unload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "start_worker"
    OUTPUT_NODE = True
    CATEGORY = "Remote H3 Latent Upscaler"

    def start_worker(self, model_name, cuda_device, precision, listen_port, bind_host="0.0.0.0", auth_token="", force_unload=True):
        token = auth_token or os.environ.get("REMOTE_H3_UPSCALE_TOKEN", "")
        if model_name.startswith("("):
            raise RuntimeError("Place the MiniMax H3 latent upscaler checkpoint in models/latent_upscale_models/")
        existing = SendRemoteH3LatentUpscaler._servers.pop(listen_port, None)
        if existing is not None:
            try:
                existing.close()
            except OSError:
                pass
        worker = _Worker(model_name, cuda_device, precision, force_unload, token)
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((bind_host, int(listen_port)))
        server.listen(4)
        SendRemoteH3LatentUpscaler._servers[listen_port] = server
        if bind_host == "0.0.0.0" and not token:
            log("WARNING: worker bound to 0.0.0.0 with no auth token. Anyone on the LAN can use this upscaler.")
        log(
            f"Worker listening on {bind_host}:{listen_port} "
            f"model={worker.model_name} cuda:{worker.cuda_device} ({worker.meta['gpu_name']}) {worker.precision}"
        )

        def handle(conn, addr):
            log(f"Client connected: {addr}")
            try:
                set_socket_opts(conn, timeout=IDLE_TIMEOUT)
                while True:
                    header = recv_header(conn)
                    if token and not hmac.compare_digest(header.get("auth", ""), token):
                        send_packet(conn, {"error": "unauthorized", "blob_size": 0})
                        break
                    if header.get("proto", 0) != PROTOCOL_VERSION:
                        send_packet(conn, {
                            "error": (
                                f"protocol mismatch: worker speaks v{PROTOCOL_VERSION}, "
                                f"client sent v{header.get('proto')}"
                            ),
                            "blob_size": 0,
                        })
                        break
                    blob_data = recv_blob(conn, header)
                    cmd = header.get("cmd")
                    if cmd == "meta":
                        send_packet(conn, {"meta": worker.meta, "blob_size": 0})
                        continue
                    if cmd == "cleanup":
                        try:
                            worker.cleanup()
                            send_packet(conn, {"ok": True, "blob_size": 0})
                        except Exception as exc:
                            log(f"Cleanup failed: {exc}")
                            send_packet(conn, {"error": str(exc), "blob_size": 0})
                        continue
                    if cmd != "upscale":
                        send_packet(conn, {"error": "bad request", "blob_size": 0})
                        continue
                    try:
                        tensors = unpack_tensors(header.get("tensors", {}), blob_data)
                        out = worker.upscale(tensors["samples"], header.get("settings") or {})
                        meta, blob = pack_tensors({"samples": out}, None)
                        send_packet(conn, {"tensors": meta, "blob_size": len(blob)}, blob)
                        log(f"Sent upscale result {tuple(out.shape)} ({len(blob)} bytes)")
                        del tensors, out, blob, blob_data
                    except Exception as exc:
                        log(f"upscale failed: {exc}")
                        send_packet(conn, {"error": str(exc), "blob_size": 0})
            except (ConnectionError, OSError) as exc:
                log(f"Client {addr} disconnected: {exc}")
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

        def accept_loop():
            while True:
                try:
                    conn, addr = server.accept()
                except OSError:
                    log("Server socket closed; stopping accept loop")
                    break
                threading.Thread(target=handle, args=(conn, addr), daemon=True).start()

        threading.Thread(target=accept_loop, daemon=True).start()
        text = (
            f"Remote H3 upscaler on {bind_host}:{listen_port} "
            f"cuda:{worker.cuda_device} {worker.meta['gpu_name']}"
        )
        return {"ui": {"text": [text]}}


class RemoteH3LatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "worker_ip": ("STRING", {"default": "192.168.1.37"}),
                "port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
                "mode": (MODES, {"default": "megapixels"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "align": ("INT", {"default": 32, "min": 1, "max": 512}),
            },
            "optional": {
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05}),
                "width": ("INT", {"default": 1280, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 704, "min": 64, "max": 8192, "step": 8}),
                "enable_temporal_chunking": ("BOOLEAN", {"default": True}),
                "auth_token": ("STRING", {"default": ""}),
                "transport_precision": (["auto", "fp16", "fp32"], {"default": "auto"}),
                "release_remote_clip": ("BOOLEAN", {"default": True}),
                "clip_port": ("INT", {"default": 8181, "min": 1, "max": 65535}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "Remote H3 Latent Upscaler"

    def upscale(
        self,
        latent,
        worker_ip,
        port,
        mode,
        megapixels,
        align,
        scale=2.0,
        width=1280,
        height=704,
        enable_temporal_chunking=True,
        auth_token="",
        transport_precision="auto",
        release_remote_clip=True,
        clip_port=8181,
    ):
        samples = latent["samples"]
        if not isinstance(samples, torch.Tensor):
            raise RuntimeError("Remote H3 upscale expects a video latent tensor in latent['samples']")
        token = auth_token or os.environ.get("REMOTE_H3_UPSCALE_TOKEN", "")
        if release_remote_clip:
            try:
                _release_remote_clip(worker_ip, clip_port, os.environ.get("REMOTE_CLIP_TOKEN", ""))
            except Exception as exc:
                log(f"Remote CLIP was not released ({exc}). Upscale will still run.")
        conn = _Connection(worker_ip, port, token)
        _MASTER_WORKERS[(conn.ip, conn.port)] = conn
        from .protocol import ALLOWED_DTYPES
        transport = resolve_transport_dtype(transport_precision, worker_ip)
        transport_dtype = ALLOWED_DTYPES.get(transport) if transport else None
        meta, blob = pack_tensors({"samples": samples}, transport_dtype)
        header = {
            "cmd": "upscale",
            "proto": PROTOCOL_VERSION,
            "tensors": meta,
            "settings": {
                "mode": mode,
                "scale": scale,
                "width": width,
                "height": height,
                "megapixels": megapixels,
                "align": align,
                "enable_temporal_chunking": enable_temporal_chunking,
            },
            "blob_size": len(blob),
        }
        log(f"Sending upscale {tuple(samples.shape)} mode={mode} ({len(blob)} bytes)")
        resp, out_blob = conn.request(header, blob, response_timeout=UPSCALE_TIMEOUT, retries=2)
        if resp.get("error"):
            raise RuntimeError(f"Remote H3 upscaler worker error: {resp['error']}")
        out = unpack_tensors(resp["tensors"], out_blob)["samples"]
        device = comfy.model_management.intermediate_device()
        out = out.to(device)
        log(f"Received upscale {tuple(out.shape)} ({len(out_blob)} bytes)")
        updated = dict(latent)
        updated["samples"] = out
        mask = updated.get("noise_mask")
        if isinstance(mask, torch.Tensor) and mask.shape[-2:] != out.shape[-2:]:
            updated.pop("noise_mask", None)
        return (updated,)


NODE_CLASS_MAPPINGS = {
    "SendRemoteH3LatentUpscaler": SendRemoteH3LatentUpscaler,
    "RemoteH3LatentUpscale": RemoteH3LatentUpscale,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SendRemoteH3LatentUpscaler": "Send Remote H3 Latent Upscaler",
    "RemoteH3LatentUpscale": "Remote H3 Latent Upscale",
}

try:
    _install_master_cleanup_hook(_flush_workers)
except Exception as exc:
    log(f"Could not install prompt-end upscaler cleanup hook: {exc}")
