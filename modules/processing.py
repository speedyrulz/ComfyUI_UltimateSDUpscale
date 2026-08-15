from PIL import Image, ImageFilter, ImageDraw
import logging
import torch
import math
from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import SamplerCustom
import comfy.sample
import comfy.model_management
import latent_preview
from usdu_utils import (
    pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond,
    crop_reference_latents, resize_region, resize_tensor, set_reference_latents,
)
from contextlib import contextmanager
from modules import shared
from tqdm import tqdm
import comfy.utils as comfy_utils
from enum import Enum
import json
import os
from typing import Callable, List, Optional, Tuple
from crop_model_patch import crop_model_cond

logger = logging.getLogger(__name__)

if (not hasattr(Image, 'Resampling')):  # For older versions of Pillow
    Image.Resampling = Image

# Taken from the USDU script
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class StableDiffusionProcessing:

    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        uniform_tile_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        custom_sampler=None,
        custom_sigmas=None,
        batch_size=1,
        guider=None,
        tile_reference_latent=False,
        reference_image=None,
        reference_strength=1.0,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = Image.new('L', init_img.size, 0)  # Placeholder mask
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width * upscale_by
        self.height = init_img.height * upscale_by
        self.rows = round(self.height / tile_height)
        self.cols = round(self.width / tile_width)

        # Guider-based sampling (guider encapsulates model + conditioning + cfg)
        self.guider = guider

        # ComfyUI Sampler inputs
        self.model = guider.model_patcher if guider is not None else model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Per-tile reference latents (Flux.2 Klein, Kontext, Qwen Edit, ...)
        self.tile_reference_latent = tile_reference_latent
        self.reference_image = reference_image
        self.reference_strength = reference_strength

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        if guider is None and (custom_sampler is not None) ^ (custom_sigmas is not None):
            logger.warning("Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        self.uniform_tile_mode = uniform_tile_mode
        self.tiled_decode = tiled_decode
        self.batch_size = batch_size
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            logger.info("Using tiled decode")

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}

        # Load config file for USDU
        config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Progress bar for the entire process instead of per tile
        self.progress_bar_enabled = False
        if comfy_utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy_utils.PROGRESS_BAR_ENABLED = config.get('per_tile_progress', True)
            self.tiles = 0
            if redraw_mode.value != USDUMode.NONE.value:
                self.tiles += self.rows * self.cols
            if seam_fix_mode.value == USDUSFMode.BAND_PASS.value:
                self.tiles += (self.rows - 1) + (self.cols - 1)
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows + (self.rows - 1) * (self.cols - 1)
            self.pbar: Optional[tqdm] = None
            # self.pbar = tqdm(total=self.tiles, desc='USDU') # Creating the pbar here will cause an empty progress bar to be displayed

    def __del__(self):
        # Undo changes to progress bar flag when node is done or cancelled
        if self.progress_bar_enabled:
            comfy_utils.PROGRESS_BAR_ENABLED = True
    
class Processed:

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    """Choose the way to sample based on given inputs"""

    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        kwargs = dict(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        if "execute" in dir(SamplerCustom):
            (samples, _) = SamplerCustom.execute(**kwargs)
        else:
            custom_sample = SamplerCustom()
            (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(**kwargs)
        return samples

    # Default
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


def sample_with_guider(guider, sampler, sigmas, seed, latent):
    """Sample using a guider (which encapsulates model, conditioning, and cfg)."""
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)

    noise = comfy.sample.prepare_noise(latent_image, seed)

    callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(noise, latent_image, sampler, sigmas,
                            denoise_mask=latent.get("noise_mask", None),
                            callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())
    return {"samples": samples}


def get_latent_scale(p: StableDiffusionProcessing) -> int:
    """The pixel -> latent downscale factor of the VAE (8 for SD/Flux.1, 16 for Flux.2)."""
    try:
        return int(p.vae.spacial_compression_encode())
    except Exception:
        logger.debug("Could not query the VAE compression ratio, assuming 8")
        return 8


def scale_reference_latent(p: StableDiffusionProcessing, latent):
    """
    Scale a reference latent by p.reference_strength.

    The models have no native strength for reference latents, so the latent itself is
    scaled towards the neutral point of the model's latent format: 1.0 leaves it untouched,
    0.0 makes it a blank reference and values above 1.0 exaggerate it.
    """
    strength = p.reference_strength
    if latent is None or strength == 1.0:
        return latent

    shift = 0.0
    latent_format = getattr(getattr(p.model, "model", None), "latent_format", None)
    if latent_format is not None:
        shift = getattr(latent_format, "shift_factor", 0.0)
        if shift is None:
            shift = 0.0
    if torch.is_tensor(shift):
        # Per channel shift, broadcast over the channel dimension of a (B, C, ...) latent
        shift = shift.to(device=latent.device, dtype=latent.dtype)
        shift = shift.view(1, -1, *([1] * (latent.ndim - 2)))

    return shift + (latent - shift) * strength


def scale_cond_reference_latents(p: StableDiffusionProcessing, cond):
    """Apply the reference strength to the reference latents of an already cropped conditioning."""
    if p.reference_strength == 1.0:
        return cond
    out = []
    for emb, cond_dict in cond:
        latents = cond_dict.get("reference_latents")
        if isinstance(latents, list):
            cond_dict = cond_dict.copy()
            cond_dict["reference_latents"] = [scale_reference_latent(p, lat) for lat in latents]
        out.append([emb, cond_dict])
    return out


def encode_reference_tiles(p: StableDiffusionProcessing, crop_regions, canvas_size, tile_size, image_index=0):
    """Crop the given regions out of the reference image and VAE encode them as a latent batch."""
    ref = p.reference_image
    if ref.shape[0] > 1:
        ref = ref[min(image_index, ref.shape[0] - 1):][:1]
    else:
        ref = ref[:1]
    ref_size = (ref.shape[2], ref.shape[1])  # (w, h)

    crops = []
    for region in crop_regions:
        x1, y1, x2, y2 = resize_region(region, canvas_size, ref_size)
        crop = ref[:, y1:y2, x1:x2, :]
        if (crop.shape[2], crop.shape[1]) != tuple(tile_size):
            crop = resize_tensor(crop.movedim(-1, 1), (tile_size[1], tile_size[0]), mode="bilinear").movedim(1, -1)
        crops.append(crop)

    (latent,) = p.vae_encoder.encode(p.vae, torch.cat(crops, dim=0))
    return latent["samples"]


def get_tile_reference_latent(p: StableDiffusionProcessing, crop_regions, canvas_size, tile_size,
                              tile_latent, image_index=0):
    """
    Build the reference latent for the tile(s) currently being generated.

    Returns None when the per-tile reference is disabled, in which case any reference
    latents already present on the conditioning are cropped to the tile instead.
    """
    if not p.tile_reference_latent:
        return None
    if not isinstance(crop_regions, list):
        crop_regions = [crop_regions]
    if p.reference_image is None:
        # The tile of the image being upscaled has already been encoded for img2img,
        # reuse it instead of encoding the same pixels twice.
        latent = tile_latent["samples"]
    else:
        latent = encode_reference_tiles(p, crop_regions, canvas_size, tile_size, image_index)
    return scale_reference_latent(p, latent)


@contextmanager
def guider_reference_latents(p: StableDiffusionProcessing, guider, reference_latent, crop_regions,
                             canvas_size, tile_size):
    """
    Temporarily replace the reference latents held by a guider's conditioning with the
    per-tile reference latent (or with the tile-cropped version of the latents it already has).
    """
    conds = getattr(guider, "original_conds", None)
    if conds is None:
        yield
        return

    latent_scale = get_latent_scale(p)
    originals = dict(conds)
    try:
        for key, cond_list in originals.items():
            patched = []
            for cond_dict in cond_list:
                cond_dict = cond_dict.copy()
                if reference_latent is not None and key == "positive":
                    cond_dict["reference_latents"] = [reference_latent]
                else:
                    crop_reference_latents(cond_dict, crop_regions, p.init_size, canvas_size,
                                           tile_size, 0, 0, latent_scale)
                    latents = cond_dict.get("reference_latents")
                    if isinstance(latents, list):
                        cond_dict["reference_latents"] = [scale_reference_latent(p, lat) for lat in latents]
                patched.append(cond_dict)
            conds[key] = patched
        yield
    finally:
        for key, cond_list in originals.items():
            conds[key] = cond_list


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Show the progress bar
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='USDU', unit='tile')

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    if p.uniform_tile_mode:
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height
    else:
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    tiles = [img.crop(crop_region) for img in shared.batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    # Reference latent for the tile currently being generated
    reference_latent = get_tile_reference_latent(p, crop_region, init_image.size, tile_size, latent)

    if p.guider is not None:
        # Guider-based sampling
        with guider_reference_latents(p, p.guider, reference_latent, crop_region, init_image.size, tile_size), \
                crop_model_cond(p.model, crop_region, p.init_size, init_image.size, tile_size) as model:
            samples = sample_with_guider(p.guider, p.custom_sampler, p.custom_sigmas, p.seed, latent)
    else:
        # Crop conditioning
        latent_scale = get_latent_scale(p)
        positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size,
                                     latent_scale=latent_scale)
        negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size,
                                     latent_scale=latent_scale)

        positive_cropped = scale_cond_reference_latents(p, positive_cropped)
        negative_cropped = scale_cond_reference_latents(p, negative_cropped)

        if reference_latent is not None:
            positive_cropped = set_reference_latents(positive_cropped, reference_latent)

        with crop_model_cond(p.model, crop_region, p.init_size, init_image.size, tile_size) as model:
            # Generate samples
            samples = sample(model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
                            negative_cropped, latent, p.denoise, p.custom_sampler, p.custom_sigmas)

    # Update the progress bar
    if p.progress_bar_enabled:
        assert p.pbar is not None
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = shared.batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')

        shared.batch[i] = result

    processed = Processed(p, [shared.batch[0]], p.seed, "")
    return processed


def process_batch_tiles(
    p: StableDiffusionProcessing,
    tiles_coords: List[Tuple[int, int]],
    images: List[Image.Image],
    calc_rectangle_fn: Callable,
) -> List[Image.Image]:
    """Encode, sample and decode a batch of tiles and composite them back into *images*.

    Unlike process_images() which operates on a single pre-built mask, this function
    builds per-tile masks from *calc_rectangle_fn* and handles every (tile, image)
    combination in one batched encode → sample → decode pass.
    """
    if not tiles_coords or not images:
        return images

    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=getattr(p, "tiles", 0), desc='USDU', unit='tile')

    batch_tiles: List[Tuple[Image.Image, Tuple[int, int]]] = []
    batch_masks: List[Image.Image] = []
    batch_crop_regions: List[Tuple[int, int, int, int]] = []
    batch_tile_sizes: List[Tuple[int, int]] = []

    for image in images:
        for tx, ty in tiles_coords:
            tile_mask = Image.new("L", (image.width, image.height), "black")
            tile_draw = ImageDraw.Draw(tile_mask)
            tile_draw.rectangle(calc_rectangle_fn(tx, ty), fill="white")

            crop_region = get_crop_region(tile_mask, p.inpaint_full_res_padding)

            if p.uniform_tile_mode:
                x1, y1, x2, y2 = crop_region
                crop_w = x2 - x1
                crop_h = y2 - y1
                crop_ratio = crop_w / crop_h if crop_h != 0 else 1.0
                p_ratio = p.width / p.height if p.height != 0 else 1.0
                if crop_ratio > p_ratio:
                    target_w = crop_w
                    target_h = round(crop_w / p_ratio)
                else:
                    target_w = round(crop_h * p_ratio)
                    target_h = crop_h
                crop_region, _ = expand_crop(crop_region, tile_mask.width, tile_mask.height, target_w, target_h)
                tile_size: Tuple[int, int] = (p.width, p.height)
            else:
                x1, y1, x2, y2 = crop_region
                crop_w = x2 - x1
                crop_h = y2 - y1
                target_w = math.ceil(crop_w / 8) * 8
                target_h = math.ceil(crop_h / 8) * 8
                crop_region, tile_size = expand_crop(crop_region, tile_mask.width, tile_mask.height, target_w, target_h)

            if p.mask_blur > 0:
                tile_mask = tile_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

            cropped_tile = image.crop(crop_region)
            initial_tile_size = cropped_tile.size
            if cropped_tile.size != tile_size:
                cropped_tile = cropped_tile.resize(tile_size, Image.Resampling.LANCZOS)

            batch_tiles.append((cropped_tile, initial_tile_size))
            batch_masks.append(tile_mask)
            batch_crop_regions.append(crop_region)
            batch_tile_sizes.append(tile_size)

    # Encode all tiles into a single latent batch
    batched_tensors = torch.cat([pil_to_tensor(tile) for tile, _ in batch_tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tensors)

    first_tile_size = batch_tile_sizes[0]

    # Reference latents for every tile in the batch, in the same order as the encoded tiles
    reference_latent = get_tile_reference_latent(p, batch_crop_regions, images[0].size, first_tile_size, latent)

    if p.guider is not None:
        # Guider-based sampling
        with guider_reference_latents(p, p.guider, reference_latent, batch_crop_regions, images[0].size, first_tile_size), \
                crop_model_cond(p.model, batch_crop_regions, p.init_size, images[0].size, first_tile_size) as model:
            samples = sample_with_guider(p.guider, p.custom_sampler, p.custom_sigmas, p.seed, latent)
    else:
        # Crop conditioning using the full list of regions (first tile size assumed uniform)
        latent_scale = get_latent_scale(p)
        positive_cropped = crop_cond(p.positive, batch_crop_regions, p.init_size, images[0].size, first_tile_size,
                                     latent_scale=latent_scale)
        negative_cropped = crop_cond(p.negative, batch_crop_regions, p.init_size, images[0].size, first_tile_size,
                                     latent_scale=latent_scale)

        positive_cropped = scale_cond_reference_latents(p, positive_cropped)
        negative_cropped = scale_cond_reference_latents(p, negative_cropped)

        if reference_latent is not None:
            positive_cropped = set_reference_latents(positive_cropped, reference_latent)

        with crop_model_cond(p.model, batch_crop_regions, p.init_size, images[0].size, first_tile_size) as model:
            samples = sample(model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler,
                             positive_cropped, negative_cropped, latent, p.denoise,
                             p.custom_sampler, p.custom_sigmas)

    # Update progress bar once per batch call (one step per tile coord)
    if p.progress_bar_enabled:
        assert p.pbar is not None
        p.pbar.update(len(tiles_coords))

    # Decode
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)

    # Composite each decoded tile back onto its source image
    result_imgs = list(images)
    for i, result_img in enumerate(result_imgs):
        for j in range(len(tiles_coords)):
            idx = i * len(tiles_coords) + j
            tile_sampled = tensor_to_pil(decoded, idx)
            initial_tile_size = batch_tiles[idx][1]
            crop_region = batch_crop_regions[idx]
            tile_mask = batch_masks[idx]

            if tile_sampled.size != initial_tile_size:
                tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

            image_tile_only = Image.new('RGBA', result_img.size)
            image_tile_only.paste(tile_sampled, crop_region[:2])

            temp = image_tile_only.copy()
            temp.putalpha(tile_mask)
            image_tile_only.paste(temp, image_tile_only)

            result = result_img.convert('RGBA')
            result.alpha_composite(image_tile_only)
            result_img = result.convert('RGB')
            result_imgs[i] = result_img

    return result_imgs
