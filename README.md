# ComfyUI Remote H3 Latent Upscaler

Run the **MiniMax H3 3D latent upscaler on a slave machine**. The master keeps the diffusion model and only sends the video latent over the LAN.

This follows the same split as **[ComfyUI Remote VAE](https://github.com/wealllovegithub/Comfyui_remote_vae)**. The wire protocol is the same idea (JSON header + tensor blob, optional shared `auth_token`) on its own port, so it can run next to Remote CLIP and Remote VAE.

The math is **[LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)**. That node only offers CUDA, ROCm, and CPU. On an Intel Arc master, its CUDA setting falls through to CPU. This plugin keeps that network on an NVIDIA slave instead.

## Roles

| Role | Machine | Nodes |
|---|---|---|
| **Slave (sender)** | Holds the upscaler checkpoint and does the 3D convolution | **Send Remote H3 Latent Upscaler** |
| **Master** | Runs sampling | **Remote H3 Latent Upscale** in place of Minimax H3 Latent Upscaler (3D) |

Install **this** plugin on both machines. Install the LBH upscaler node and the checkpoint on the **slave only**.

## Installation

On **each** ComfyUI (`custom_nodes`):

```bash
git clone https://github.com/wealllovegithub/Comfyui_remote_h3_latent_upscaler.git
```

On the **slave** only:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
```

Download one checkpoint into `ComfyUI/models/latent_upscale_models/` on the slave. The fp16 file is the one the example graph uses:

`minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`

from [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler).

Restart ComfyUI on both machines.

## Usage

### Slave

1. Add **Send Remote H3 Latent Upscaler**.
2. Pick the checkpoint. Set `cuda_device` to the NVIDIA GPU index (`0` is the first card).
3. Leave `precision` at `fp16` and `listen_port` at **8184**.
4. **Queue the prompt once.** The node keeps listening until ComfyUI restarts. Re-queue after a restart.

`force_unload` (default on) moves the network back to CPU after each clip so the card is free for CLIP or the VAE.

Open the port on the slave firewall, **LAN only**:

```bash
sudo ufw allow from 192.168.1.0/24 to any port 8184 proto tcp
```

### Master

1. Add **Remote H3 Latent Upscale** where the local 3D upscaler node was.
2. Connect the video latent from **Separate AV Latent**. Connect the output back into **Concat AV Latent**.
3. Set `worker_ip` to the slave and `port` to **8184**.
4. `mode` / `megapixels` / `align` match the local 3D node. The example graph uses megapixels `1` and align `32`.

`release_remote_clip` (default on) asks the Remote CLIP worker on the same host, port **8181**, to leave the GPU before the upscale. CLIP encoding for that prompt has already finished by the time this node runs. The second sampling pass still runs on the master.

## Together with Remote CLIP and Remote VAE

| Plugin | Default port | Offloads |
|---|---|---|
| [ComfyUI-RemoteCLIPLoader](https://github.com/nyueki/ComfyUI-RemoteCLIPLoader) | 8181 | CLIP |
| [ComfyUI Remote VAE](https://github.com/wealllovegithub/Comfyui_remote_vae) | 8182 video, 8183 audio | VAE encode / decode |
| This repo | 8184 | H3 3D latent upscaler |

Do not share a port between them.

## Notes

- Traffic is **not encrypted**. Use a LAN or a VPN. Do not expose the port to the internet.
- Protocol v1 must match on both sides.
- A 124-frame latent at this graph's sizes is a few megabytes in and a few tens of megabytes back. Gigabit Ethernet adds about a second.
- The refinement pass after the upscale still runs on the master at the larger resolution. This plugin only moves the upscaler network.
- Re-queue **Send Remote H3 Latent Upscaler** after every ComfyUI restart on the slave.

Node category: **Remote H3 Latent Upscaler**

## License

MIT. Protocol layout follows [Comfyui_remote_vae](https://github.com/wealllovegithub/Comfyui_remote_vae).
