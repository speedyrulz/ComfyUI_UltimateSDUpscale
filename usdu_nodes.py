# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

import logging
from contextlib import contextmanager
import torch
import comfy
from usdu_patch import usdu
from usdu_utils import tensor_to_pil, pil_to_tensor
from modules.processing import StableDiffusionProcessing, get_vae_latent_scale
from batch_latent_overlap import redraw as batch_latent_overlap_redraw
from nodes import VAEDecode, VAEDecodeTiled
import modules.shared as shared
from modules.upscaler import UpscalerData

logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    """Context manager to temporarily suppress logging output."""
    root_logger = logging.getLogger()
    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(old_level)

MAX_RESOLUTION = 8192
# The image sources for the node variant that does not upscale
INPUT_MODES = ["image", "latent"]
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}


def USDU_base_inputs():
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        # Sampling Params
        ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
        ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
        ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed to use for image-to-image."})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
        ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
        ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. Yields different results than individual tiles. Only affects the main redraw step, not the seam fix step."})),
    ]

    optional = list(USDU_reference_inputs())

    return required, optional


def USDU_reference_inputs():
    """Inputs for per-tile reference latents (Flux.2 Klein, Kontext, Qwen Edit, ...)."""
    return [
        ("tile_reference_latent", ("BOOLEAN", {"default": False, "tooltip": "Use the region of the image that matches the tile being generated as the reference latent for that tile. Required for reference latent models such as Flux.2 Klein: feeding a reference latent through the conditioning instead uses the whole image as the reference for every tile. Replaces any reference latents already present on the positive conditioning."})),
        ("reference_strength", ("FLOAT", {"default": 1.00, "min": 0.0, "max": 5.0, "step": 0.05, "tooltip": "How strongly the reference latent influences each tile. Scales the attention keys and values of the reference image tokens in every block, the same way the FLUX.2 Klein reference latent controller nodes do. 1.00 leaves the reference as is, lower values weaken it, 0.00 ignores it, and higher values strengthen it. Also applies to reference latents supplied through the conditioning."})),
        ("reference_image", ("IMAGE", {"tooltip": "Optional image to take the reference tiles from when tile_reference_latent is enabled. Defaults to the image being upscaled. Any resolution: the region matching the current tile is cropped and resized to the tile size before encoding."})),
    ]


def insert_input(inputs: list, before_name: str, entry):
    for i, (n, _) in enumerate(inputs):
        if n == before_name:
            inputs.insert(i, entry)
            return
    inputs.append(entry)


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Upscales an image and runs image-to-image on tiles from the input image."

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                custom_sampler=None, custom_sigmas=None,
                tile_reference_latent=False, reference_strength=1.0, reference_image=None,
                mask=None, use_mask=False):
        redraw_mode = MODES[mode_type]
        seam_fix_mode = SEAM_FIX_MODES[seam_fix_mode]

        #
        # Set up A1111 patches
        #

        # Upscaler
        # An object that the script works with
        shared.sd_upscalers[0] = UpscalerData()
        # Where the actual upscaler is stored, will be used when the script upscales using the Upscaler in UpscalerData
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        logger.debug("UltimateSDUpscale.upscale() using batch_size=%s", batch_size)
        assert batch_size == 1 or force_uniform_tiles, "batch_size greater than 1 requires force_uniform_tiles to be True; all tiles in the batch must be the same size."

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, redraw_mode, seam_fix_mode,
            custom_sampler, custom_sigmas, batch_size,
            tile_reference_latent=tile_reference_latent, reference_strength=reference_strength,
            reference_image=reference_image, mask=mask, use_mask=use_mask,
        )
        logger.debug("StableDiffusionProcessing created with batch_size=%s", sdprocessing.batch_size)

        # Suppress logging to prevent duplicate tqdm progress bars
        with suppress_logging():
            #
            # Running the script
            #
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=redraw_mode,
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=seam_fix_mode, target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=upscale_by)

        # Return the resulting images
        images = [pil_to_tensor(img) for img in shared.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)

class UltimateSDUpscaleNoUpscale(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        remove_input(required, "image")
        required.insert(0, ("input_mode", (list(INPUT_MODES), {"tooltip": "Whether to take the image to refine from the upscaled_image input or from the latent input. A latent is decoded once with the given VAE before the tiling starts."})))
        optional.insert(0, ("upscaled_image", ("IMAGE", {"tooltip": "The image to refine. Used when input_mode is set to image."})))
        optional.insert(1, ("latent", ("LATENT", {"tooltip": "The latent to refine. Used when input_mode is set to latent, and decoded to an image before the tiling starts."})))
        optional.append(("use_mask", ("BOOLEAN", {"default": False, "tooltip": "Only refine the area covered by the mask input. The mask is split between the tiles so that each tile is masked by the part of the mask that lines up with it, and tiles the mask does not cover at all are skipped."})))
        optional.append(("mask", ("MASK", {"tooltip": "The area to refine, white is refined and black is left untouched. Only used when use_mask is enabled. Resized to the size of the image if it does not already match."})))
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final refined image.",)
    DESCRIPTION = "Runs image-to-image on tiles from the input image or latent."

    def upscale(self, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                input_mode="image", upscaled_image=None, latent=None,
                tile_reference_latent=False, reference_strength=1.0, reference_image=None,
                use_mask=False, mask=None):
        upscale_by = 1.0

        logger.debug("UltimateSDUpscaleNoUpscale.upscale() received batch_size=%s", batch_size)

        image = self.get_input_image(input_mode, upscaled_image, latent, vae, tiled_decode)

        return super().upscale(image, model, positive, negative, vae, upscale_by, seed,
                               steps, cfg, sampler_name, scheduler, denoise, None,
                               mode_type, tile_width, tile_height, mask_blur, tile_padding,
                               seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                               seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size,
                               tile_reference_latent=tile_reference_latent, reference_strength=reference_strength,
                               reference_image=reference_image, mask=mask, use_mask=use_mask)

    @staticmethod
    def get_input_image(input_mode, upscaled_image, latent, vae, tiled_decode):
        """The image to refine, decoding the latent input first when that is the chosen input."""
        if input_mode == "latent":
            assert latent is not None, "input_mode is set to latent, but no latent was given."
            if tiled_decode:
                (decoded,) = VAEDecodeTiled().decode(vae, latent, 512)
            else:
                (decoded,) = VAEDecode().decode(vae, latent)
            return decoded

        assert upscaled_image is not None, "input_mode is set to image, but no upscaled_image was given."
        return upscaled_image
    
class UltimateSDUpscaleCustomSample(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        optional.append(("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The model to use for upscaling the image. If not provided, a simple Lanczos scaling will be used instead."})))
        optional.append(("custom_sampler", ("SAMPLER", {"tooltip": "A custom sampler to use instead of the built-in ComfyUI sampler specified by sampler_name. Only used if both custom_sampler and custom_sigmas are provided."})))
        optional.append(("custom_sigmas", ("SIGMAS", {"tooltip": "A custom noise schedule to use during sampling. Only used if both custom_sampler and custom_sigmas are provided."})))
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Runs image-to-image on tiles from the input image."

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                upscale_model=None,
                custom_sampler=None, custom_sigmas=None,
                tile_reference_latent=False, reference_strength=1.0, reference_image=None):
        return super().upscale(image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size,
                custom_sampler, custom_sigmas,
                tile_reference_latent=tile_reference_latent, reference_strength=reference_strength,
                reference_image=reference_image)

class UltimateSDUpscaleNoUpscaleBatchLatentOverlap(UltimateSDUpscaleNoUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        for name in ("upscale_model", "upscale_by", "image", "mode_type",
                     "tile_width", "tile_height", "force_uniform_tiles", "batch_size"):
            remove_input(required, name)
        remove_input(required, "tile_padding")

        required.insert(0, ("input_mode", (list(INPUT_MODES), {"tooltip": "Whether to take the image to refine from the upscaled_image input or from the latent input. A latent is decoded once with the given VAE before the redraw starts."})))
        # The tiling is fixed at 2x2, so the padding is the only thing left to set about it
        insert_input(required, "mask_blur", ("tile_padding", ("INT", {"default": 32, "min": 8, "max": MAX_RESOLUTION, "step": 8, "tooltip": "How far each tile reaches into its neighbours. The image is always split into 4 evenly sized tiles, and every edge that touches another tile is extended by this many pixels, so neighbouring tiles overlap by twice this amount. That overlap is what is averaged together after every step."})))

        optional.insert(0, ("upscaled_image", ("IMAGE", {"tooltip": "The image to refine. Used when input_mode is set to image."})))
        optional.insert(1, ("latent", ("LATENT", {"tooltip": "The latent to refine. Used when input_mode is set to latent, and decoded to an image before the redraw starts."})))
        optional.append(("use_mask", ("BOOLEAN", {"default": False, "tooltip": "Only refine the area covered by the mask input. The mask limits the denoising of the redraw and limits what is pasted back over the original image."})))
        optional.append(("mask", ("MASK", {"tooltip": "The area to refine, white is refined and black is left untouched. Only used when use_mask is enabled. Resized to the size of the image if it does not already match."})))
        for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
            label = corner.replace("_", " ")
            optional.append((f"positive_{corner}", ("CONDITIONING", {"tooltip": f"Optional prompt for the {label} tile. Tiles without their own prompt use the main positive. Only the text is taken from this input; guidance, controlnet and reference latents still come from the main positive."})))
        optional.append(("color_match", ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Pull the colours of the result back towards the input image. Edit models tend to shift the whole image in tint and contrast, which shows against the rest of the picture. Each channel is rescaled to the mean and standard deviation of the input, correcting an overall shift without flattening the detail that was generated. The shift is measured only over the parts of the image the edit left alone, so recolouring something in the picture does not drag everything else towards its old colour. 0.00 leaves the model's colours alone, 0.50 to 0.80 corrects a drift while leaving the edit its own look, 1.00 matches the input fully."})))
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final refined image.",)
    DESCRIPTION = ("Refines an image as 4 overlapping tiles that are sampled as a single batch of 4. "
                   "The tiles share one canvas latent, so the overlapping parts of the tiles are averaged "
                   "together after every sampling step instead of being blended only at the end.")

    def upscale(self, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler, denoise,
                tile_padding, mask_blur, seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, tiled_decode,
                input_mode="image", upscaled_image=None, latent=None,
                tile_reference_latent=False, reference_strength=1.0, reference_image=None,
                use_mask=False, mask=None, color_match=0.0,
                positive_top_left=None, positive_top_right=None,
                positive_bottom_left=None, positive_bottom_right=None):
        image = self.get_input_image(input_mode, upscaled_image, latent, vae, tiled_decode)

        redrawn = batch_latent_overlap_redraw(
            image, model, positive, negative, vae, seed, steps, cfg, sampler_name, scheduler,
            denoise, tile_padding, tiled_decode,
            tile_reference_latent=tile_reference_latent, reference_strength=reference_strength,
            reference_image=reference_image, mask=mask, use_mask=use_mask, color_match=color_match,
            tile_prompts=[positive_top_left, positive_top_right,
                          positive_bottom_left, positive_bottom_right],
        )

        if SEAM_FIX_MODES[seam_fix_mode].value == usdu.USDUSFMode.NONE.value:
            return (redrawn,)

        # Hand the redrawn image to the normal pipeline, with the redraw step turned off, so that
        # the seam fix runs with the usual settings. The tiles it works from are the same 2x2 split.
        latent_scale = get_vae_latent_scale(vae)
        tile_width = max(latent_scale, round(redrawn.shape[2] / 2 / latent_scale) * latent_scale)
        tile_height = max(latent_scale, round(redrawn.shape[1] / 2 / latent_scale) * latent_scale)
        return UltimateSDUpscale.upscale(
            self, redrawn, model, positive, negative, vae, 1.0, seed,
            steps, cfg, sampler_name, scheduler, denoise, None,
            "None", tile_width, tile_height, mask_blur, tile_padding,
            seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
            seam_fix_width, seam_fix_padding, True, tiled_decode, 1,
            tile_reference_latent=tile_reference_latent, reference_strength=reference_strength,
            reference_image=reference_image, mask=mask, use_mask=use_mask,
        )


def USDU_guider_inputs():
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        # Sampling Params (guider encapsulates model + conditioning + cfg)
        ("guider", ("GUIDER", {"tooltip": "The guider to use for sampling. Encapsulates the model, conditioning, and CFG scale."})),
        ("sampler", ("SAMPLER", {"tooltip": "The sampler to use for each tile."})),
        ("sigmas", ("SIGMAS", {"tooltip": "The noise schedule (sigmas) to use for sampling."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed to use for noise generation."})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Unused by this node: the guider samples with the given sigmas, which determine the denoising strength for both the redraw and seam fix steps."})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
        ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
        ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. Yields different results than individual tiles. Only affects the main redraw step, not the seam fix step."})),
    ]

    optional = list(USDU_reference_inputs())

    return required, optional


class UltimateSDUpscaleGuider:
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_guider_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Upscales an image using a guider for sampling. Use this with custom sampling nodes (BasicGuider, CFGGuider, etc.) for full control over the sampling pipeline."

    def upscale(self, image, guider, sampler, sigmas, vae, upscale_by, seed,
                upscale_model, mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                tile_reference_latent=False, reference_strength=1.0, reference_image=None):
        redraw_mode = MODES[mode_type]
        seam_fix_mode = SEAM_FIX_MODES[seam_fix_mode]

        # Upscaler
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        logger.debug("UltimateSDUpscaleGuider.upscale() using batch_size=%s", batch_size)
        assert batch_size == 1 or force_uniform_tiles, "batch_size greater than 1 requires force_uniform_tiles to be True; all tiles in the batch must be the same size."

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], None, None, None, vae,
            seed, 0, 0, None, None, 0, upscale_by, force_uniform_tiles, tiled_decode,
            tile_width, tile_height, redraw_mode, seam_fix_mode,
            custom_sampler=sampler, custom_sigmas=sigmas, batch_size=batch_size,
            guider=guider,
            tile_reference_latent=tile_reference_latent, reference_strength=reference_strength,
            reference_image=reference_image,
        )
        logger.debug("StableDiffusionProcessing created with guider, batch_size=%s", sdprocessing.batch_size)

        # Suppress logging to prevent duplicate tqdm progress bars
        with suppress_logging():
            script = usdu.Script()
            processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                               mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                               seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                               upscaler_index=0, save_upscaled_image=False, redraw_mode=redraw_mode,
                               save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                               seams_fix_type=seam_fix_mode, target_size_type=2,
                               custom_width=None, custom_height=None, custom_scale=upscale_by)

        # Return the resulting images
        images = [pil_to_tensor(img) for img in shared.batch]
        tensor = torch.cat(images, dim=0)
        return (tensor,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscale": UltimateSDUpscale,
    "UltimateSDUpscaleNoUpscale": UltimateSDUpscaleNoUpscale,
    "UltimateSDUpscaleNoUpscaleBatchLatentOverlap": UltimateSDUpscaleNoUpscaleBatchLatentOverlap,
    "UltimateSDUpscaleCustomSample": UltimateSDUpscaleCustomSample,
    "UltimateSDUpscaleGuider": UltimateSDUpscaleGuider,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscale": "Ultimate SD Upscale",
    "UltimateSDUpscaleNoUpscale": "Ultimate SD Upscale (No Upscale)",
    "UltimateSDUpscaleNoUpscaleBatchLatentOverlap": "Ultimate SD Upscale (No Upscale - Batch Latent Overlap)",
    "UltimateSDUpscaleCustomSample": "Ultimate SD Upscale (Custom Sample)",
    "UltimateSDUpscaleGuider": "Ultimate SD Upscale (Guider)",
}
