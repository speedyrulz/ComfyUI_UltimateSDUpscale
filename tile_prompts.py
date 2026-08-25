"""Load an image and prompt each of its four tiles separately.

A companion to Ultimate SD Upscale (No Upscale - Batch Latent Overlap): it loads an image,
shows it split into the same four overlapping tiles that node samples, and gives each tile its
own prompt box. The four conditioning outputs are named after that node's four tile inputs, so
they line up one to one.

The splitting shown in the node is done in the browser, by js/tile_prompts.js. This module only
loads the image and encodes the four prompts.
"""

import logging
import os

import folder_paths
from nodes import LoadImage

logger = logging.getLogger(__name__)

# The names the four tiles have on the upscale node, in the order the tiles are laid out
CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")

MAX_RESOLUTION = 8192


def combine_prompts(shared: str, tile: str) -> str:
    """
    The prompt one tile is encoded with: what every tile has in common, then what that tile adds.

    The two are joined with a full stop, and either half on its own is used as it is, so an empty
    box never leaves a stray separator behind.
    """
    parts = []
    for part in (shared, tile):
        part = (part or "").strip()
        while part.endswith("."):
            part = part[:-1].rstrip()
        if part:
            parts.append(part)
    return ". ".join(parts)


class UltimateSDUpscaleTilePrompts:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True, "tooltip": "The image to prompt the tiles of."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model used to encode the four prompts."}),
                "tile_padding": ("INT", {"default": 32, "min": 8, "max": MAX_RESOLUTION, "step": 8, "tooltip": "How far each tile reaches into its neighbours. Only affects the tiles shown here: set it to the same value as the upscale node so that the preview matches what will actually be sampled."}),
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt every tile gets. It is put in front of each tile's own prompt, separated by a full stop, so this is where what the whole image has in common goes and each tile box only has to say what is different about that quarter."}),
                "prompt_top_left": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt for the top left tile."}),
                "prompt_top_right": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt for the top right tile."}),
                "prompt_bottom_left": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt for the bottom left tile."}),
                "prompt_bottom_right": ("STRING", {"multiline": True, "default": "", "tooltip": "The prompt for the bottom right tile."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "STRING", "IMAGE")
    RETURN_NAMES = ("positive_top_left", "positive_top_right", "positive_bottom_left", "positive_bottom_right",
                    "filename", "image")
    OUTPUT_TOOLTIPS = (
        "The prompt for the top left tile, for the upscale node's positive_top_left input.",
        "The prompt for the top right tile, for the upscale node's positive_top_right input.",
        "The prompt for the bottom left tile, for the upscale node's positive_bottom_left input.",
        "The prompt for the bottom right tile, for the upscale node's positive_bottom_right input.",
        "The name of the loaded file without its extension, for naming what is saved from it.",
        "The whole loaded image, for the upscale node's image input.",
    )
    FUNCTION = "encode_tiles"
    CATEGORY = "image/upscaling"
    DESCRIPTION = ("Loads an image, shows it split into the same 4 overlapping tiles that Ultimate SD Upscale "
                   "(No Upscale - Batch Latent Overlap) samples, and encodes a separate prompt for each tile.")

    def encode_tiles(self, image, clip, tile_padding, prompt,
                     prompt_top_left, prompt_top_right, prompt_bottom_left, prompt_bottom_right):
        assert clip is not None, (
            "The clip input is invalid: None. If it comes from a checkpoint loader, that checkpoint "
            "does not contain a text encoder."
        )

        (loaded_image, _) = LoadImage().load_image(image)

        tile_prompts = (prompt_top_left, prompt_top_right, prompt_bottom_left, prompt_bottom_right)
        texts = [combine_prompts(prompt, tile_prompt) for tile_prompt in tile_prompts]
        for corner, text in zip(CORNERS, texts):
            logger.debug("Encoding the %s tile as %r", corner, text)
        conditioning = tuple(clip.encode_from_tokens_scheduled(clip.tokenize(text)) for text in texts)

        # The name on its own, so it can be handed straight to a save node's filename prefix
        filename = os.path.splitext(os.path.basename(image.split(" [")[0]))[0]

        return conditioning + (filename, loaded_image)

    @classmethod
    def IS_CHANGED(s, image, **kwargs):
        return LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(s, image, **kwargs):
        return LoadImage.VALIDATE_INPUTS(image)
