`Ultimate SD Upscale Tile Prompts` loads an image, shows it split into the same 4 overlapping tiles that `Ultimate SD Upscale (No Upscale - Batch Latent Overlap)` samples, and gives each tile its own prompt box.

It exists so that per tile prompting can be done by looking at the tiles rather than by guessing at them. The node draws each tile exactly as the upscale node will crop it, with a dashed outline marking that tile's own quarter of the image: everything outside the outline is the padding it shares with its neighbours, which is the part that gets averaged with them during sampling.

Its four conditioning outputs are named after the upscale node's four tile inputs, so they connect one to one.

## Inputs

| Parameter | Data Type | Input Method | Default | Range | Description |
|-----------|-----------|--------------|---------|--------|-------------|
| `image` | COMBO | Dropdown + upload | - | Files in the input folder | The image to prompt the tiles of, picked or uploaded the same way as the core Load Image node. |
| `clip` | CLIP | Model Selection | None | - | The CLIP model used to encode the prompts. |
| `tile_padding` | INT | Number Input | 32 | 8-8192 (step 8) | How far each tile reaches into its neighbours. This only affects the tiles shown here, so set it to the same value as the upscale node to see what will actually be sampled. |
| `prompt` | STRING | Multiline Text | "" | - | The prompt every tile gets. It is put in front of each tile's own prompt, separated by a full stop, so what the whole image has in common goes here and each tile box only has to say what is different about that quarter. |
| `prompt_top_left` | STRING | Multiline Text | "" | - | The prompt for the top left tile, shown under that tile's preview. |
| `prompt_top_right` | STRING | Multiline Text | "" | - | The prompt for the top right tile. |
| `prompt_bottom_left` | STRING | Multiline Text | "" | - | The prompt for the bottom left tile. |
| `prompt_bottom_right` | STRING | Multiline Text | "" | - | The prompt for the bottom right tile. |

The four tile prompt boxes are drawn under their tile previews rather than as ordinary widgets, but they are still ordinary inputs underneath: they are saved into the workflow and restored from it like any other text widget.

## Outputs

| Output | Data Type | Description |
|--------|-----------|-------------|
| `positive_top_left` | CONDITIONING | The encoded prompt for the top left tile, for the upscale node's `positive_top_left` input. |
| `positive_top_right` | CONDITIONING | The encoded prompt for the top right tile. |
| `positive_bottom_left` | CONDITIONING | The encoded prompt for the bottom left tile. |
| `positive_bottom_right` | CONDITIONING | The encoded prompt for the bottom right tile. |
| `filename` | STRING | The name of the loaded file with its folder and extension removed, so it can be handed to a save node's filename prefix. |
| `image` | IMAGE | The whole loaded image, for the upscale node's `upscaled_image` input. |

## How the prompts are combined

Each tile is encoded as the shared `prompt`, then that tile's own prompt, joined with a full stop:

- `prompt` = "a photo of a woman on a beach", `prompt_top_left` = "her face, freckles" gives *a photo of a woman on a beach. her face, freckles*.
- A box left empty contributes nothing and leaves no stray separator behind, so a tile with no prompt of its own is encoded as just the shared prompt.
- A trailing full stop on either half is not doubled up.

## Usage Tips

1. **Wiring it up**
	- `image` goes to the upscale node's `upscaled_image`, and each `positive_<corner>` goes to the matching `positive_<corner>` input.
	- The upscale node still needs its main `positive` connected: it is what the seam fix uses, and what any tile whose input is left disconnected falls back to.
	- Set `tile_padding` to the same value on both nodes, otherwise the previews show a different split from the one being sampled.

2. **What to write where**
	- Put everything the whole image shares in the main `prompt`: subject, style, lighting, camera. Put only what is actually in that quarter in each tile box.
	- Resist describing a tile as if it were a whole picture. The tiles are averaged together in their overlaps every step, so four full scene descriptions fight each other, while one shared description plus four local details reinforces.
	- A tile whose quarter has nothing worth calling out can be left empty and will simply follow the shared prompt.

3. **Reading the previews**
	- The dashed outline is that tile's own quarter. What lies outside it also appears in a neighbouring tile, so anything you describe there should be described compatibly in both, or left to the shared prompt.
	- Raising `tile_padding` grows the region outside the outline, which is what gives neighbouring tiles more common ground to agree on.

4. **The image the previews come from**
	- The previews are drawn in the browser from the picked file, so they show the image as loaded, before any upscaling. If you feed the upscale node a different or already upscaled image, the split shown here is still correct in proportion, since the tiles are always the four quadrants plus padding.
