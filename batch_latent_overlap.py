"""Redraw an image as four overlapping tiles that are sampled as a single batch.

The image is encoded once into a canvas latent that is sampled as one latent. Every model
evaluation crops that canvas into the four overlapping tiles, evaluates all four as one batch,
and merges the predictions back onto the canvas, averaging the overlaps with complementary
weights. The canvas therefore carries a single denoising trajectory: after every step the
overlapping parts of the tiles are one and the same latent, which is what the per step
averaging is for, without the tiles ever drifting apart in between.

This is done by hooking sampler_calc_cond_batch_function, which sits above the conditioning
batch: the tiles become the real sampling batch for the duration of the model call, so the
per tile conditioning that this pack already builds (cropped controlnet hints, per tile
reference latents) lines up with the tiles by construction.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

import comfy.model_base
import comfy.model_patcher
import comfy.samplers
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled

from contextlib import contextmanager

from crop_model_patch import crop_model_cond
from modules.processing import (
    get_vae_latent_scale,
    reference_strength_patch,
    sample,
)
from usdu_utils import crop_cond, set_reference_latents

logger = logging.getLogger(__name__)

# The tiles are always the four quadrants, extended into their neighbours by the padding
TILE_COUNT = 4

# Conditioning that is baked to the size of the whole canvas before the tiles are cut out of it.
# ComfyUI resizes a conditioning mask to the size of the latent being sampled, which here is the
# whole canvas, so a per tile crop of it cannot survive to the tile evaluation.
UNSUPPORTED_COND_KEYS = ("concat_latent_image", "concat_mask", "mask", "area")
# Conditioning this pack can only crop for a single region, so tiles 2 to 4 get tile 1's crop
SINGLE_REGION_COND_KEYS = ("gligen",)

Region = Tuple[int, int, int, int]


def latent_tile_regions(height: int, width: int, pad_h: int, pad_w: int) -> Tuple[List[Region], Tuple[int, int]]:
    """
    The four latent space tile regions and the size of the overlap between neighbours.

    The canvas is split into four evenly sized quadrants, and every edge that touches another
    tile is extended into that tile by the padding. All four regions have the same size, which
    is what lets them be sampled as one batch, and together they cover the whole canvas.

    :return: ([(x1, y1, x2, y2) for each tile], (overlap_height, overlap_width))
    """
    half_h, half_w = height // 2, width // 2
    pad_h = max(1, min(pad_h, half_h))
    pad_w = max(1, min(pad_w, half_w))

    tile_h = min(half_h + pad_h, height)
    tile_w = min(half_w + pad_w, width)

    overlap_h = 2 * tile_h - height
    overlap_w = 2 * tile_w - width
    assert overlap_h > 0 and overlap_w > 0, (
        f"The tiles do not overlap for a {width}x{height} latent with padding {pad_w}x{pad_h}"
    )

    # Top left, top right, bottom left, bottom right
    regions = [
        (x, y, x + tile_w, y + tile_h)
        for y in (0, height - tile_h)
        for x in (0, width - tile_w)
    ]
    return regions, (overlap_h, overlap_w)


def _ramp(length: int, device, dtype) -> torch.Tensor:
    """A rising ramp of *length* values in (0, 1), whose mirror image sums with it to exactly 1."""
    return (torch.arange(length, device=device, dtype=dtype) + 0.5) / length


def tile_weight(region: Region, canvas_size: Tuple[int, int], overlap: Tuple[int, int],
                device, dtype) -> torch.Tensor:
    """
    The blending weight of one tile, as a (1, 1, tile_height, tile_width) tensor.

    The weight fades out towards every edge that overlaps a neighbouring tile and stays at 1
    along the outer edges of the canvas. Two neighbours therefore always sum to exactly 1 over
    their overlap, which is what makes the merge an average of the tiles instead of a sum.
    """
    height, width = canvas_size
    overlap_h, overlap_w = overlap
    x1, y1, x2, y2 = region

    weight_y = torch.ones(y2 - y1, device=device, dtype=dtype)
    if y1 > 0:
        weight_y[:overlap_h] = _ramp(overlap_h, device, dtype)
    if y2 < height:
        weight_y[-overlap_h:] = _ramp(overlap_h, device, dtype).flip(0)

    weight_x = torch.ones(x2 - x1, device=device, dtype=dtype)
    if x1 > 0:
        weight_x[:overlap_w] = _ramp(overlap_w, device, dtype)
    if x2 < width:
        weight_x[-overlap_w:] = _ramp(overlap_w, device, dtype).flip(0)

    return (weight_y[:, None] * weight_x[None, :])[None, None]


class TiledCondBatch:
    """
    Evaluates the model on the four overlapping tiles of the canvas latent as one batch.

    Installed as sampler_calc_cond_batch_function, so it runs for every model evaluation of
    every sampling step, and returns predictions shaped like the canvas that the sampler then
    steps as usual.
    """

    def __init__(self, regions: List[Region], overlap: Tuple[int, int], canvas_size: Tuple[int, int]):
        self.regions = regions
        self.overlap = overlap
        self.canvas_size = canvas_size  # (height, width) in latent units
        # A calc cond batch function that was already installed, if any, so this one composes
        # with it instead of replacing it
        self.previous = None
        self.calls = 0
        self._weights: Optional[List[torch.Tensor]] = None
        self._weights_key = None

    def weights(self, device, dtype) -> List[torch.Tensor]:
        key = (device, dtype)
        if self._weights is None or self._weights_key != key:
            self._weights = [
                tile_weight(region, self.canvas_size, self.overlap, device, dtype)
                for region in self.regions
            ]
            self._weights_key = key
        return self._weights

    def merge(self, tiles: torch.Tensor, canvas_shape, dtype) -> torch.Tensor:
        """Average the per tile predictions back onto the canvas."""
        batch, channels, height, width = canvas_shape
        # Accumulate in float32: four bf16 predictions summed and divided lose exactly the
        # precision that the overlap is being blended with
        accumulator = torch.zeros((batch, channels, height, width), device=tiles.device, dtype=torch.float32)
        weight_sum = torch.zeros((batch, 1, height, width), device=tiles.device, dtype=torch.float32)

        for i, (x1, y1, x2, y2) in enumerate(self.regions):
            weight = self.weights(tiles.device, torch.float32)[i]
            accumulator[:, :, y1:y2, x1:x2] += tiles[i:i + 1].float() * weight
            weight_sum[:, :, y1:y2, x1:x2] += weight

        assert float(weight_sum.min()) > 1e-6, "The tiles do not cover the whole canvas"
        return (accumulator / weight_sum).to(dtype)

    def inner_call(self, model, conds, x, sigma, model_options):
        """The conditioning batch this hook replaced, or comfy's own."""
        if self.previous is not None:
            return self.previous({"conds": conds, "input": x, "sigma": sigma,
                                  "model": model, "model_options": model_options})
        return comfy.samplers.calc_cond_batch(model, conds, x, sigma, model_options)

    def release(self):
        """Drop the cached weights so they do not hold on to device memory."""
        self._weights = None
        self._weights_key = None

    def __call__(self, args):
        conds = args["conds"]
        x = args["input"]
        sigma = args["sigma"]
        model = args["model"]
        model_options = args["model_options"]

        # Sampling the tiles goes through the normal conditioning batch, without this hook
        inner_options = model_options.copy()
        inner_options.pop("sampler_calc_cond_batch_function", None)

        if x.shape[0] != 1 or tuple(x.shape[-2:]) != self.canvas_size:
            # Not the canvas this was built for: an area conditioning narrowed the input, or
            # something else is sampling with this model. Leave it alone.
            logger.debug("Not tiling a %s input, expected a single %s canvas", tuple(x.shape), self.canvas_size)
            return self.inner_call(model, conds, x, sigma, inner_options)

        tiles = torch.cat([x[:, :, y1:y2, x1:x2] for x1, y1, x2, y2 in self.regions], dim=0)
        tile_sigma = sigma.repeat_interleave(len(self.regions), dim=0)
        assert tile_sigma.shape[0] == tiles.shape[0]

        out = self.inner_call(model, conds, tiles, tile_sigma, inner_options)
        self.calls += 1

        return [self.merge(cond_out, x.shape, x.dtype) for cond_out in out]


def check_conditioning(conds, name: str):
    """Warn or fail early on conditioning that cannot be cropped to four separate tiles."""
    for cond in conds or []:
        cond_dict = cond[1]
        for key in UNSUPPORTED_COND_KEYS:
            assert key not in cond_dict, (
                f"The {name} conditioning carries '{key}', which is built for the whole image and "
                f"cannot be split between the tiles. Use the regular Ultimate SD Upscale (No Upscale) "
                f"node for it, which samples one tile at a time."
            )
        for key in SINGLE_REGION_COND_KEYS:
            if key in cond_dict:
                logger.warning(
                    "The %s conditioning carries '%s', which can only be cropped for one region, "
                    "so every tile is given the crop of the first tile.", name, key
                )


def check_model(model):
    """Fail early on models whose conditioning is built for the whole image."""
    inner = getattr(model, "model", None)
    concat_cond = getattr(type(inner), "concat_cond", None)
    if concat_cond is not None and concat_cond is not comfy.model_base.BaseModel.concat_cond:
        # Model families override concat_cond for their fill / inpaint / control lora editions, but
        # the plain editions return None from the same method (Flux and Flux2 decide by the channel
        # count of their input layer). Probe it with an empty conditioning: only a model that builds
        # an image sized concat on its own is a problem for the tiles.
        try:
            channels = getattr(getattr(inner, "latent_format", None), "latent_channels", 16)
            probe = torch.zeros(1, channels, 8, 8)
            built = inner.concat_cond(noise=probe, device=torch.device("cpu"))
        except Exception:
            # Cannot tell from here; the conditioning checks still refuse concat_latent_image
            logger.debug("Could not probe concat_cond, relying on the conditioning checks", exc_info=True)
            built = None
        assert built is None, (
            f"{type(inner).__name__} builds an image sized extra conditioning of its own (inpainting "
            f"and fill style models do this), which cannot be split between the tiles. Use the regular "
            f"Ultimate SD Upscale (No Upscale) node for it."
        )

    if model.model_options.get("sampler_post_cfg_function"):
        logger.warning(
            "A post CFG function is attached to the model (PAG, SAG, SLG and similar). Those run "
            "their own model evaluation on the whole canvas instead of the tiles, which is slow and "
            "uses the first tile's conditioning for the whole image."
        )


@contextmanager
def sampling_model_options(model, reference_strength: float, tiled_cond_batch: "TiledCondBatch"):
    """
    Give the model the tiling hook and the reference strength for the length of the redraw.

    The options are swapped on the model patcher the graph owns rather than on a throwaway clone.
    ComfyUI tracks a loaded model through a weak reference to the patcher it was loaded with, so a
    clone that is dropped the moment the node returns leaves that bookkeeping pointing at an object
    that no longer exists, which is what its memory leak warning is about.
    """
    original_options = model.model_options
    # A nested copy: patch lists are copied, the patch objects inside them stay shared
    model.model_options = comfy.model_patcher.create_model_options_clone(original_options)
    try:
        if reference_strength != 1.0:
            model.set_model_attn1_patch(reference_strength_patch(reference_strength))
        tiled_cond_batch.previous = model.model_options.get("sampler_calc_cond_batch_function")
        model.set_model_sampler_calc_cond_batch_function(tiled_cond_batch)
        yield model
    finally:
        model.model_options = original_options


def center_crop_to_multiple(image: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Crop an image tensor to a multiple of *multiple* the same way the VAE does when encoding.

    Returns the cropped image and the (x, y) offset it was taken from, so that the result can
    be composited back onto the original image afterwards.
    """
    _, height, width, _ = image.shape
    new_height = (height // multiple) * multiple
    new_width = (width // multiple) * multiple
    assert new_height > 0 and new_width > 0, (
        f"The image is smaller than the {multiple} pixel block size of the VAE"
    )

    y_offset = (height % multiple) // 2
    x_offset = (width % multiple) // 2
    cropped = image[:, y_offset:y_offset + new_height, x_offset:x_offset + new_width, :]
    return cropped, (x_offset, y_offset)


def merge_tile_prompts(positive_cropped, tile_prompts, fallback):
    """
    Give each tile its own text prompt.

    The four tiles are sampled as one batch, so the text embeddings can be stacked along the
    batch dimension the same way the per tile reference latents are: tile i then attends to
    row i of the stack. Tiles without their own prompt fall back to the main positive.

    Only the text embedding is taken from the per tile inputs. Everything else on the
    conditioning (guidance, controlnet, reference latents) comes from the main positive, so
    the per tile prompts stay simple text encodes.
    """
    if not any(p is not None for p in tile_prompts):
        return positive_cropped

    names = ("top left", "top right", "bottom left", "bottom right")
    embeddings = []
    for name, prompt in zip(names, tile_prompts):
        source = prompt if prompt is not None else fallback
        if len(source) > 1:
            logger.warning("The %s tile prompt has %d conditioning entries, only the first is used",
                           name, len(source))
        embeddings.append(source[0][0][:1])

    # Prompts of different lengths are stacked by padding the shorter ones with empty tokens at
    # the front, which is how Flux.2 pads short prompts itself
    max_tokens = max(e.shape[1] for e in embeddings)
    embeddings = [
        F.pad(e, (0, 0, max_tokens - e.shape[1], 0)) if e.shape[1] < max_tokens else e
        for e in embeddings
    ]
    stacked = torch.cat(embeddings, dim=0)

    if len(positive_cropped) > 1:
        logger.warning("The positive conditioning has %d entries, the per tile prompts only replace "
                       "the text of the first", len(positive_cropped))
    return [[stacked if i == 0 else emb, cond_dict]
            for i, (emb, cond_dict) in enumerate(positive_cropped)]


def build_reference_latent(vae, reference_image: Optional[torch.Tensor],
                           canvas_latent: torch.Tensor, canvas_pixels: torch.Tensor) -> torch.Tensor:
    """
    The canvas sized reference latent that the per tile references are cut out of.

    Without a reference image the canvas latent is the reference: it is already the encoding of
    the pixels being refined, so every tile ends up referencing its own region.
    """
    if reference_image is None:
        return canvas_latent

    reference = reference_image[:1]
    _, height, width, _ = canvas_pixels.shape
    if (reference.shape[1], reference.shape[2]) != (height, width):
        reference = F.interpolate(
            reference.movedim(-1, 1), size=(height, width), mode="bilinear", align_corners=False
        ).movedim(1, -1)

    (latent,) = VAEEncode().encode(vae, reference)
    return latent["samples"]


def redraw(
    image: torch.Tensor,
    model,
    positive,
    negative,
    vae,
    seed: int,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    denoise: float,
    tile_padding: int,
    tiled_decode: bool,
    tile_reference_latent: bool = False,
    reference_strength: float = 1.0,
    reference_image: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    use_mask: bool = False,
    color_match: float = 0.0,
    tile_prompts: Optional[List] = None,
) -> torch.Tensor:
    """
    Refine an image as four overlapping tiles sampled as one batch, and return the result.

    :param image: The image to refine, as a (1, height, width, 3) tensor.
    """
    assert image.shape[0] == 1, (
        f"This node refines one image at a time, got a batch of {image.shape[0]}."
    )
    if denoise <= 0.0:
        # Nothing to redraw, and sampling would return before the model is ever evaluated
        logger.info("Skipping the redraw, denoise is 0")
        return image

    check_conditioning(positive, "positive")
    check_conditioning(negative, "negative")
    check_model(model)

    vae_encoder = VAEEncode()

    latent_scale = get_vae_latent_scale(vae)

    # The canvas is the part of the image the VAE can encode without cropping it itself
    canvas_pixels, (x_offset, y_offset) = center_crop_to_multiple(image, latent_scale)
    canvas_width, canvas_height = canvas_pixels.shape[2], canvas_pixels.shape[1]

    (encoded,) = vae_encoder.encode(vae, canvas_pixels)
    canvas_latent = encoded["samples"]
    latent_height, latent_width = canvas_latent.shape[2], canvas_latent.shape[3]

    pad_latent = int(round(tile_padding / latent_scale))
    regions, overlap = latent_tile_regions(latent_height, latent_width, pad_latent, pad_latent)
    tile_latent_size = (regions[0][2] - regions[0][0], regions[0][3] - regions[0][1])
    logger.info(
        "Redrawing %sx%s as 4 tiles of %sx%s latents overlapping by %sx%s",
        canvas_width, canvas_height, tile_latent_size[0], tile_latent_size[1], overlap[1], overlap[0]
    )

    # The same regions in pixel space, for cropping the conditioning to each tile
    pixel_regions = [tuple(v * latent_scale for v in region) for region in regions]
    tile_pixel_size = (tile_latent_size[0] * latent_scale, tile_latent_size[1] * latent_scale)
    canvas_size = (canvas_width, canvas_height)

    positive_cropped = crop_cond(positive, pixel_regions, canvas_size, canvas_size, tile_pixel_size,
                                 latent_scale=latent_scale)
    negative_cropped = crop_cond(negative, pixel_regions, canvas_size, canvas_size, tile_pixel_size,
                                 latent_scale=latent_scale)

    if tile_prompts is not None:
        positive_cropped = merge_tile_prompts(positive_cropped, tile_prompts, positive)

    if tile_reference_latent:
        reference_latent = build_reference_latent(vae, reference_image, canvas_latent, canvas_pixels)
        if reference_latent.shape[2:] != canvas_latent.shape[2:]:
            reference_latent = F.interpolate(
                reference_latent, size=canvas_latent.shape[2:], mode="bilinear", align_corners=False
            )
        # One entry whose batch matches the tiles, so tile i references its own region
        reference_tiles = torch.cat(
            [reference_latent[:, :, y1:y2, x1:x2] for x1, y1, x2, y2 in regions], dim=0
        )
        positive_cropped = set_reference_latents(positive_cropped, reference_tiles)

    latent = {"samples": canvas_latent}
    canvas_mask = None
    if use_mask and mask is not None:
        canvas_mask = _mask_for_canvas(mask, image.shape, (x_offset, y_offset), (canvas_height, canvas_width))
        latent["noise_mask"] = canvas_mask

    tiled_cond_batch = TiledCondBatch(regions, overlap, (latent_height, latent_width))

    # crop_model_cond crops the model patches in place, so the patcher it hands back is only a
    # vehicle for the restore and the redraw can go on sampling with the graph's own patcher
    try:
        with crop_model_cond(model, list(pixel_regions), canvas_size, canvas_size, tile_pixel_size):
            with sampling_model_options(model, reference_strength, tiled_cond_batch) as sampling_model:
                samples = sample(sampling_model, seed, steps, cfg, sampler_name, scheduler,
                                 positive_cropped, negative_cropped, latent, denoise, None, None)
    finally:
        tiled_cond_batch.release()

    assert tiled_cond_batch.calls > 0, (
        "The tiles were never evaluated, the canvas was sampled in one piece instead."
    )
    logger.debug("Merged the tile predictions %s times", tiled_cond_batch.calls)

    if not tiled_decode:
        (decoded,) = VAEDecode().decode(vae, samples)
    else:
        (decoded,) = VAEDecodeTiled().decode(vae, samples, 512)

    decoded = decoded.to(device=canvas_pixels.device, dtype=canvas_pixels.dtype)
    if color_match > 0.0:
        decoded = match_colors(decoded, canvas_pixels, color_match, canvas_mask)

    return composite(image, decoded, (x_offset, y_offset), canvas_mask)


def _mask_for_canvas(mask: torch.Tensor, image_shape, offset: Tuple[int, int],
                     canvas_size: Tuple[int, int]) -> torch.Tensor:
    """The part of the mask that covers the canvas, as a (1, canvas_height, canvas_width) tensor."""
    if mask.ndim == 4:
        mask = mask[..., 0] if mask.shape[-1] == 1 else mask[:, 0]
    elif mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask[:1].clamp(0.0, 1.0)

    _, image_height, image_width, _ = image_shape
    if (mask.shape[1], mask.shape[2]) != (image_height, image_width):
        mask = F.interpolate(mask.unsqueeze(1), size=(image_height, image_width),
                             mode="bilinear", align_corners=False).squeeze(1)

    x_offset, y_offset = offset
    canvas_height, canvas_width = canvas_size
    return mask[:, y_offset:y_offset + canvas_height, x_offset:x_offset + canvas_width]


def changed_region_pixels(refined: torch.Tensor, source: torch.Tensor,
                          mask: Optional[torch.Tensor], limit: int = 200000):
    """
    The pixels of the part of the image the redraw actually rewrote, as two (N, C) tensors.

    That is the masked area when there is a mask, since everything outside it is kept as it was and
    so says nothing about how the model's colours drifted, and the whole canvas otherwise. Large
    images are sampled rather than measured in full, which changes the statistics by nothing that
    matters and keeps this off the critical path.
    """
    flat_refined = refined.reshape(-1, refined.shape[-1])
    flat_source = source.reshape(-1, source.shape[-1])

    if mask is not None:
        selected = (mask.reshape(-1) > 0.5).nonzero(as_tuple=True)[0]
        if selected.numel() >= 64:
            flat_refined = flat_refined[selected]
            flat_source = flat_source[selected]

    count = flat_refined.shape[0]
    if count > limit:
        step = count // limit + 1
        flat_refined = flat_refined[::step]
        flat_source = flat_source[::step]

    return flat_refined, flat_source


def match_colors(refined: torch.Tensor, source: torch.Tensor, strength: float,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Pull the colours of the redrawn area back towards the image it came from.

    Edit models drift: what comes back is tinted or lifted compared to what went in, which shows up
    against the parts of the picture that were not redrawn. Each channel is shifted and rescaled so
    that the redrawn area sits where the original did.

    The measurement is a median, not a mean, and it is taken over the area that was actually
    redrawn. Using the median is what lets this coexist with an edit that deliberately changes
    something: recoloured clothing is a minority of the pixels, so it barely moves the median, and
    the correction that comes out is the drift the rest of the area shares rather than the colour of
    the thing that was changed. The correction is also capped, so an area that really was rewritten
    from top to bottom cannot pull the picture far.

    *strength* blends between the model's own colours (0.0) and a full match (1.0).
    """
    if strength <= 0.0:
        return refined

    refined = refined.float()
    source = source.float()

    sampled_refined, sampled_source = changed_region_pixels(refined, source, mask)
    if sampled_refined.shape[0] < 64:
        logger.warning("Skipping the colour match, there is too little of the image to measure")
        return refined

    refined_middle = sampled_refined.median(dim=0).values
    source_middle = sampled_source.median(dim=0).values
    refined_spread = (sampled_refined - refined_middle).abs().median(dim=0).values
    source_spread = (sampled_source - source_middle).abs().median(dim=0).values

    # A cap on both halves of the correction: a drift is a nudge, and anything bigger than this is
    # the measurement being wrong rather than the model being that far off
    offset = (source_middle - refined_middle).clamp(-0.15, 0.15)
    measurable = (refined_spread > 1e-3) & (source_spread > 1e-3)
    scale = torch.where(measurable, source_spread / refined_spread.clamp(min=1e-3),
                        torch.ones_like(source_spread)).clamp(0.75, 1.333)

    shape = (1, 1, 1, refined.shape[-1])
    matched = (refined - refined_middle.reshape(shape)) * scale.reshape(shape)         + (refined_middle + offset).reshape(shape)

    logger.debug("Colour match over %d pixels: offset %s, scale %s",
                 sampled_refined.shape[0], offset.tolist(), scale.tolist())
    return torch.lerp(refined, matched.clamp(0.0, 1.0), strength)


def composite(image: torch.Tensor, decoded: torch.Tensor, offset: Tuple[int, int],
              canvas_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Put the refined canvas back onto the image, only where the mask covers it."""
    x_offset, y_offset = offset
    height, width = decoded.shape[1], decoded.shape[2]

    result = image.clone()
    region = result[:, y_offset:y_offset + height, x_offset:x_offset + width, :]
    refined = decoded.to(device=region.device, dtype=region.dtype)

    if canvas_mask is not None:
        alpha = canvas_mask.to(device=region.device, dtype=region.dtype).unsqueeze(-1)
        refined = region * (1.0 - alpha) + refined * alpha

    result[:, y_offset:y_offset + height, x_offset:x_offset + width, :] = refined
    return result
