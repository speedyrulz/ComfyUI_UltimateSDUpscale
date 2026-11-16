# ComfyUI_UltimateSDUpscale

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) nodes for performing the image-to-image diffusion process on large images in tiles. This approach improves the details that is commonly found on upscaled images while reducing hardware requirements and maintaining an image size that the diffusion model is trained on.

## Installation


### Using Git
1. Git must be installed on your system. Verify by running `git -v` in a terminal.
2. Enter the following command from the terminal starting in ComfyUI/custom_nodes/
    ```
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale
    ```

### ComfyUI Manager
1. [ComfyUI Manager](https://github.com/Comfy-Org/ComfyUI-Manager) must be installed.
2. After launching ComfyUI, open ComfyUI Manager and select the "Custom Nodes Manager" option.
3. Search for "UltimateSDUpscale" and install the node. Select latest for the most up-to-date version.
4. Follow any prompts to restart ComfyUI.

### comfy-cli

1. [comfy-cli](https://github.com/Comfy-Org/comfy-cli) must be installed.
2. Run this command from the terminal: `comfy node install comfyui_ultimatesdupscale`

### Manual Download
1. Download the zip file from https://registry.comfy.org/nodes/comfyui_ultimatesdupscale to select the version you want, or obtain the current nightly version by clicking the green "Code" button on the GitHub repository page and selecting "Download ZIP".
2. Create a new folder in the `ComfyUI/custom_nodes/` directory to hold the extracted files (e.g. `ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale`).
3. Extract the contents of the zip file into the `ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale` folder.


## Usage

Nodes can be found in the node menu under `image/upscaling`.

Documentation for the nodes can be found in the [`js/docs/`](js/docs/) folder, or viewed within the application by right-clicking the relevant node and selecting the info icon.

Details about most of the parameters can be found [here](https://github.com/Coyote-A/ultimate-upscale-for-automatic1111/wiki/FAQ#parameters-descriptions).

Example workflows can be found in the [`example_workflows/`](example_workflows/) folder. You can also find them in the ComfyUI application under the Templates menu, scroll down the left sidebar to find the Extensions section, then selecting this repository.

### MiniMax H3 tiled sampling

Two nodes (`image/upscaling`) refine large MiniMax H3 audio+video latents tile by tile inside the sampler:

* **Ultimate SD Upscale: MiniMax H3 Tiled Sampling** takes a `MODEL` and returns it with a wrapper that, on every
  sampler step, splits the video latent into overlapping spatial tiles (full time length each), runs H3 once per tile
  and blends the tile outputs with a linear feather (MultiDiffusion style). The audio latent goes whole into every tile
  call. Keyframe guides and the denoise mask are cropped per tile; reference images are left whole. With
  `true_positions` each tile gets the rotary positions it has in the full frame. Use it in place of the plain model:
  upscale the video pixels, VAE encode, then sample with `denoise` below 1. Not compatible with FunControl or Spectrum.
* **Ultimate SD Upscale: MiniMax H3 Hold Audio** sets the latent's `noise_mask` so the video is denoised and the audio
  is kept as it is.

#### Two-GPU variant

**Ultimate SD Upscale: MiniMax H3 Tiled Sampling (2 GPU)** is the same node plus a `worker_url` input (default
`http://127.0.0.1:8189`). It splits the tiles of every model call between this GPU and a second ComfyUI server (the
worker, for example the same install started with `--cuda-device 1 --port 8189`) and blends all tile outputs on this
side. The single-GPU node is unchanged, and an empty `worker_url` runs everything locally.

How it works: when sampling starts, the node queues a small job on the worker (the loaders of the model's subgraph plus
an internal `Tile Worker` node). The worker loads the model, checks that it matches this server's model, and then serves
tile requests for the length of the run. Model calls run on this GPU alone until the worker reports that its model is
loaded; the first call after that splits the tiles by pulling and measures both GPUs, and from then on a fixed share
goes to the worker. If anything
goes wrong with the worker, the remaining tiles run locally and the run continues.

Requirements and limits:

* The worker must be the same ComfyUI install (or an identical one) with the same custom nodes and model files as this
  server, and its model subgraph must replay from the prompt: only `MODEL` links are followed, so a `MODEL` that
  depends on a latent, image, conditioning, clip or vae input, an output node in the chain, or an expanded
  (subgraph or loop) node makes the node run locally and print one warning.
* The worker's queue is busy for the whole run (one queued job). A job that is already running on the worker delays its
  start; this server samples alone until then. Stopping the run here removes or interrupts that one worker job and
  nothing else.
* Both servers keep their own copy of the H3 model in RAM (for the int8 model roughly twice the system memory of one
  server) and each GPU needs room for its model and a tile.
* Needs at least two tiles per frame; the speed-up is below 2x and depends on the two GPUs. Model calls that use
  ControlNet-style controls, conditioning hooks, extra patches or unknown model arguments run locally.
* Tile data (about 4 MB per 768 px tile, plus about 6 MB of text context once per run) travels as safetensors over
  HTTP, tagged with a random per-run token. Use it only between machines you trust; the worker needs no extra open
  port beyond its normal ComfyUI one, and request bodies stay under ComfyUI's `--max-upload-size`.
* Different LoRA files with the same structure and strengths on the two servers are not detected; keep the model
  setup identical.

Tests (CPU only, need the WSL ComfyUI install): `test/h3/run_tests.sh`.

## References
* Ultimate Stable Diffusion Upscale script for the Automatic1111 Web UI: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111
* ComfyUI: https://github.com/comfyanonymous/ComfyUI