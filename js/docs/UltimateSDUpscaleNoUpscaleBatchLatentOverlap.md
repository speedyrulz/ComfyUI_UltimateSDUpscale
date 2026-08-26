`Ultimate SD Upscale (No Upscale - Batch Latent Overlap)` refines an already upscaled image by splitting it into exactly 4 overlapping tiles that are sampled together as one batch, sharing a single latent.

This variant exists for the case where tile-by-tile refinement produces tiles that do not agree with each other. The regular nodes sample each tile on its own and blend the finished tiles together at the end, so a tile only ever sees its own crop and neighbouring tiles can drift apart in style, lighting or content, leaving seams for the seam fix step to repair. This node keeps all 4 tiles in one denoising process instead: the tiles overlap, and the overlapping parts are averaged after every sampling step, so neighbouring tiles are corrected towards each other while they are still being generated rather than after the fact.

## How the redraw works

1. The image is encoded once into a single canvas latent.
2. The canvas is split into 4 evenly sized tiles, a 2x2 grid. Every tile edge that touches another tile is extended into that tile by `tile_padding` pixels, so neighbours overlap by twice the padding. Edges on the outside of the image are not extended.
3. Every step of sampling evaluates the model on all 4 tiles as a single batch of 4.
4. The 4 predictions are merged back onto the canvas. In the overlaps the tiles are averaged, not added: the blending weights fade out towards each tile's inner edges and always sum to exactly 1, so an overlap is a true average of the tiles covering it and no part of the image comes out brighter than the rest.
5. The merged canvas is what the sampler steps, so the next step starts from a canvas the tiles already agree on. This repeats until sampling is finished.
6. The finished canvas is decoded in one piece, honouring `tiled_decode`, and the seam fix step then runs with the usual settings.

Because the tiles share one latent and one noise field, they never diverge in the overlaps in the first place, and the redraw does not produce seams of its own. The seam fix step is therefore usually unnecessary and defaults to `None`.

## Inputs

| Parameter | Data Type | Input Method | Default | Range | Description |
|-----------|-----------|--------------|---------|--------|-------------|
| `input_mode` | COMBO | Dropdown | image | image, latent | Whether to take the image to refine from the `upscaled_image` input or from the `latent` input. A latent is decoded once with the given VAE before the redraw starts. |
| `upscaled_image` | IMAGE | Image Input | None | - | The already upscaled image to refine. Used when `input_mode` is set to image. One image at a time: a batch is rejected. |
| `latent` | LATENT | Latent Input | None | - | The latent to refine. Used when `input_mode` is set to latent. |
| `model` | MODEL | Model Selection | None | - | The model to use for the redraw. |
| `positive` | CONDITIONING | Conditioning Input | None | - | The positive conditioning. Cropped to each tile the same way the other nodes crop it. |
| `negative` | CONDITIONING | Conditioning Input | None | - | The negative conditioning. |
| `vae` | VAE | Model Selection | None | - | The VAE used to encode the image and decode the result. |
| `seed` | INT | Number Input | 0 | 0-18446744073709551615 | The seed for the redraw. One noise field is drawn for the whole canvas, so the tiles get consistent noise. |
| `steps` | INT | Number Input | 20 | 1-10000 | The number of sampling steps. The overlaps are averaged after every one of them. |
| `cfg` | FLOAT | Slider | 8.0 | 0.0-100.0 | The CFG scale. Above 1.0 the model runs on 8 tiles per step (4 positive, 4 negative) instead of 4. |
| `sampler_name` | COMBO | Dropdown | - | Available samplers | The sampler to use. |
| `scheduler` | COMBO | Dropdown | - | Available schedulers | The scheduler to use. |
| `denoise` | FLOAT | Slider | 0.2 | 0.0-1.0 (step 0.01) | The denoising strength for the redraw. |
| `tile_padding` | INT | Number Input | 32 | 8-8192 (step 8) | How far each tile reaches into its neighbours. Neighbouring tiles overlap by twice this amount, and that overlap is what gets averaged after every step. Larger values give the tiles more shared context to agree on, at the cost of a larger tile and so more VRAM and time. |
| `mask_blur` | INT | Number Input | 8 | 0-64 | The blur radius for the mask used by the seam fix step. Has no effect on the redraw, which has no per-tile masks to blend. |
| `seam_fix_mode` | COMBO | Dropdown | None | None, Band Pass, Half Tile, Half Tile + Intersections | The seam fix mode. The redraw does not create seams, so this is normally left at `None`. When it is set, the redrawn image is passed to the regular pipeline with the redraw step disabled, and only the seam fix runs, over the same 2x2 tile grid. |
| `seam_fix_denoise` | FLOAT | Slider | 0.35 | 0.0-1.0 (step 0.01) | The denoising strength for the seam fix step. It applies on top of the finished redraw, and with this node's 2x2 grid the seam fix bands cover most of the image, so high values largely repaint the redraw and shift its colours: keep this low. |
| `seam_fix_width` | INT | Number Input | 64 | 0-8192 (step 8) | The width of the bands used for the Band Pass seam fix mode. |
| `seam_fix_mask_blur` | INT | Number Input | 8 | 0-64 | The blur radius for the seam fix mask. |
| `seam_fix_padding` | INT | Number Input | 16 | 0-8192 (step 8) | The padding to apply for the seam fix tiles. |
| `tiled_decode` | BOOLEAN | Toggle | False | True/False | Whether to use tiled decoding when decoding the finished canvas. |
| `tile_reference_latent` | BOOLEAN | Toggle | False | True/False | Use the region of the image matching each tile as that tile's reference latent, for reference latent models such as Flux.2 Klein. Each of the 4 tiles references its own region of the canvas. |
| `reference_strength` | FLOAT | Number Input | 1.00 | 0.0-5.0 (step 0.05) | How strongly the reference latent influences the tiles, by scaling the attention keys and values of the reference image tokens. |
| `reference_image` | IMAGE | Image Input | None | - | Optional image to take the reference tiles from. Defaults to the image being refined. |
| `use_mask` | BOOLEAN | Toggle | False | True/False | Only refine the area covered by the `mask` input. The mask limits the denoising of the redraw and limits what is written back over the original image. |
| `mask` | MASK | Mask Input | None | - | The area to refine: white is refined, black is left untouched. Only used when `use_mask` is enabled. Resized to the size of the image if it does not already match. |
| `positive_top_left` | CONDITIONING | Conditioning Input | None | - | Optional prompt for the top left tile. Tiles without their own prompt use the main `positive`. Only the text is taken from these inputs; guidance, controlnet and reference latents still come from the main `positive`. |
| `positive_top_right` | CONDITIONING | Conditioning Input | None | - | Optional prompt for the top right tile. |
| `positive_bottom_left` | CONDITIONING | Conditioning Input | None | - | Optional prompt for the bottom left tile. |
| `positive_bottom_right` | CONDITIONING | Conditioning Input | None | - | Optional prompt for the bottom right tile. |
| `color_match` | FLOAT | Number Input | 0.00 | 0.0-1.0 (step 0.05) | Pull the colours of the redrawn area back towards the input image. Each channel is both shifted and rescaled, so tint and saturation are put back together: edit models tend to return an image that is not only warmer but more saturated and contrasty than what went in. Both measurements are quantiles taken over the area that was actually redrawn, the masked area when a mask is used, so something deliberately recoloured barely moves them and keeps its new colour. 0.00 leaves the model's colours alone, 1.00 matches the input fully. Applied to the image that leaves the node, after the seam fix, and skipped when the colours already agree. |

## Outputs

| Output | Data Type | Description |
|--------|-----------|-------------|
| `IMAGE` | IMAGE | The refined image, at the size of the input image. |

## Usage Tips

1. **When to use this instead of the regular node**
	- Use it when tiles disagree with each other: shifts in colour, lighting or style between tiles, or content that does not line up across a seam.
	- Use it at higher denoise values, where independent tiles drift the most.
	- The regular node is still the better choice for large upscales with many tiles, since this node always uses 4 tiles and each of them is a quarter of the image, so the tile size grows with the image.

2. **Tile size and VRAM**
	- Each tile is about a quarter of the image plus the padding, and all 4 are in flight at once, so peak VRAM is roughly the whole image plus the overlap, not a single tile. Above CFG 1.0 the batch is 8 tiles.
	- For a 2048x2048 image the tiles are about 1024x1024 plus padding, which is a size most models are comfortable with. Much larger images push each tile past what the model was trained on, which is where the regular tiled node is the better tool.

3. **Padding**
	- The padding is the whole mechanism here: it is the only place where tiles can agree with each other. Very small values leave a thin band to blend over, larger values give the model more shared context.
	- The padding is applied in latent units internally, so it is rounded to the nearest multiple of the VAE's compression factor (8 pixels for most models, 16 for Flux.2).

4. **Edges of the image**
	- The VAE can only encode whole blocks, so if the image size is not a multiple of the VAE compression factor the outermost few pixels are left exactly as they came in. Feed it an image whose size is a multiple of 8 (or 16 for Flux.2) to refine every pixel.

5. **Colour shifts**
	- Edit models drift: what comes back is tinted, warmer or flatter than what went in. With a mask this is at its most obvious, because the redrawn area sits directly against untouched pixels that did not drift with it.
	- `color_match` corrects that drift, and for a masked edit it is measured inside the mask, which is the only place the drift exists. Turn it up to 1.00 for a masked edit where the result should be indistinguishable in colour from its surroundings; 0.50 to 0.80 leaves the edit some colour of its own.
	- It corrects saturation as well as tint. Klein and models like it usually come back both warmer and more saturated, and a shift alone cannot put that back, so each channel is rescaled as well as shifted.
	- It uses quantiles rather than averages, so an edit that deliberately recolours part of the area keeps its new colour: a recoloured object is a minority of the pixels and sits out in the tails, where it barely moves the measurement. No pixel is moved further than a fixed cap either, so such a colour cannot be pulled all the way back.
	- Judge the result from the decoded image, not from the preview during sampling. That preview is a crude linear projection of the latent that clips at both ends, so it consistently looks less saturated and better behaved than what the VAE actually produces. The VAE itself is colour neutral: its round trip moves channels by under 0.004.
	- At denoise 1.00 the reference latents are all that anchors the image, and the quarter size tiles drift more than a whole canvas sample does. Raising `reference_strength` to about 1.15 closes that gap: measured against a plain whole canvas sample of the same photo at denoise 1.00, the node drifts 1.7x as far at strength 1.00 and exactly as far at 1.15.

6. **Per tile prompts**
	- The four `positive_<corner>` inputs give each tile its own text prompt: the four tiles are sampled as one batch, so each tile attends to its own prompt row while still being averaged with its neighbours in the overlaps. Describe what actually sits in that quarter of the image and the shared overlap keeps the quarters consistent with each other.
	- Any tile left unconnected uses the main `positive`. Prompts of different lengths are aligned by padding the shorter ones with empty tokens, the same way Flux.2 pads short prompts itself.
	- The per tile inputs are text only: guidance, controlnet hints and reference latents always come from the main `positive`, and the negative applies to all tiles. The seam fix step, if enabled, also uses the main `positive`.

7. **Conditioning that cannot be split**
	- Conditioning that is built for the whole image is rejected up front rather than failing part way through sampling: inpainting conditioning carrying a whole-image latent (`concat_latent_image`), and area or mask conditioning (`ConditioningSetArea`, `ConditioningSetMask`), which ComfyUI resizes to the latent being sampled and so cannot be cropped per tile here.
	- Models that build an image-sized conditioning of their own, such as inpainting and instruct-style models, are rejected for the same reason.
	- GLIGEN conditioning can only be cropped for one region by this pack, so every tile is given the first tile's positions. A warning is logged when that happens.
	- Use the regular Ultimate SD Upscale (No Upscale) node, which samples one tile at a time, for any of those.
