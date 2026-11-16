# MiniMax H3 (audio + video) latent tile blending for Ultimate SD Upscale.
#
# MultiDiffusion-style: on every sampler model call the video latent is cut into overlapping spatial tiles
# (every tile keeps the full time length), the H3 diffusion model runs once per tile, and the per-tile video
# velocities are blended back into one full-size output with feathered weights. The audio latent is handed whole
# to every tile call (H3 attends over video and audio jointly) and is never tiled.

import functools
import itertools
import logging
import math
import threading
from collections import OrderedDict
from contextlib import nullcontext

import torch

import comfy
import comfy.patcher_extension

try:
    import comfy.ldm.common_dit
    import comfy.ldm.minimax.model as h3
    import comfy.model_prefetch as model_prefetch
    import comfy.nested_tensor
except Exception as e:  # ComfyUI without MiniMax H3 support: the nodes load but refuse to run
    h3 = None
    model_prefetch = None
    logging.getLogger(__name__).info("MiniMax H3 not available in this ComfyUI (%s); H3 tiling nodes disabled", e)

logger = logging.getLogger(__name__)

LATENT_PX = 16  # one latent unit is 16 px (VAE spatial compression)
PATCH = 2  # DiT patch is 2x2 latent units


def _even(n):
    return n + (n % 2)


def _px_to_even_latent(px, minimum):
    # px -> latent units, rounded to the nearest even count
    return max(minimum, int(round(px / (LATENT_PX * PATCH))) * PATCH)


def axis_tiles(length, tile, overlap):
    """Tile starts along one axis (latent units, `length` and `tile` even, starts even).

    All tiles have the same size `tile`; the last one is flush with the far edge. A frame that fits in one
    tile gives a single full-length tile. Returns (tile_size, [starts]).
    """
    if length <= tile:
        return length, [0]
    overlap = max(0, min(overlap, tile - PATCH))
    stride = tile - overlap
    n = math.ceil((length - tile) / stride) + 1
    starts = [i * stride for i in range(n - 1)] + [length - tile]
    return tile, starts


def axis_weights(starts, size, i):
    """Feather weights of tile i along one axis: linear ramps across the overlap on interior edges only."""
    w = torch.ones(size, dtype=torch.float32)
    if i > 0:
        n = starts[i - 1] + size - starts[i]
        if n > 0:
            w[:n] *= torch.arange(1, n + 1, dtype=torch.float32) / (n + 1)
    if i < len(starts) - 1:
        n = starts[i] + size - starts[i + 1]
        if n > 0:
            w[size - n:] *= torch.arange(n, 0, -1, dtype=torch.float32) / (n + 1)
    return w


def _kf_signature(keyframes):
    return tuple((kf.get("resolved_frame_index"),
                  None if kf.get("latent") is None else int(kf["latent"].shape[2]),
                  None if kf.get("audio_latent") is None else int(kf["audio_latent"].shape[-1]))
                 for kf in keyframes or ())


def _ref_signature(refs):
    return tuple((blk.get("kind"), blk.get("latent_t"), blk.get("latent_h"), blk.get("latent_w"), blk.get("ref_audio_t"))
                 for blk in refs or ())


def _crop_hw(t, y0, y1, x0, x1):
    return t[..., y0:y1, x0:x1]


def _crop_pad_hw(t, y0, y1, x0, x1):
    # crop, then pad to the DiT patch size like _forward does for the video (patchify_video does not pad)
    return comfy.ldm.common_dit.pad_to_patch_size(_crop_hw(t, y0, y1, x0, x1), (1, PATCH, PATCH))


class _LRU(OrderedDict):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def fetch(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None

    def put(self, key, value):
        self[key] = value
        self.move_to_end(key)
        while len(self) > self.size:
            self.popitem(last=False)


def crop_payload(payload, y0, y1, x0, x1):
    """Shallow copy of an H3 payload with the keyframe guides cropped to the tile.

    Keyframe latents live at the target canvas, so they are cropped (also inside cond_video_latents, where they
    come first, before the references). References keep their own resolution and the audio guides stay whole.
    The prebuilt layout is dropped: the caller installs the tile's layout.
    """
    out = dict(payload)
    out.pop("layout", None)
    keyframes = payload.get("keyframes")
    if keyframes:
        new_kfs = []
        for kf in keyframes:
            kf = dict(kf)
            if kf.get("latent") is not None:
                kf["latent"] = _crop_pad_hw(kf["latent"], y0, y1, x0, x1)
            new_kfs.append(kf)
        out["keyframes"] = new_kfs
        n_kf = sum(1 for kf in keyframes if kf.get("latent") is not None)
        cond = payload.get("cond_video_latents")
        if cond is not None and n_kf:
            out["cond_video_latents"] = [_crop_pad_hw(z, y0, y1, x0, x1) for z in cond[:n_kf]] + list(cond[n_kf:])
    return out


def tile_layout(full_layout, text_len, latent_t, tile_h, tile_w, audio_t, y0, x0, full_h, full_w,
                keyframes, refs, true_positions):
    """PackedLayout of one tile; with true_positions its rope positions are those of the full frame.

    A cropped tile would otherwise get tile-local h/w positions (a centred grid normalised by the tile's own
    sqrt-area). With true_positions the target video rows and the keyframe cond rows take the full-frame
    grid cropped to the tile's patch window, and the audio rows take the full frame's audio positions (the audio is
    shared by all tiles). Text and reference rows are the same in both layouts already.
    """
    layout = h3.PackedLayout(text_len, latent_t, tile_h, tile_w, audio_t, keyframes=keyframes, refs=refs)
    if not true_positions:
        return layout
    kinds = [k for _, _, k in layout.segments]
    if kinds != [k for _, _, k in full_layout.segments]:
        raise RuntimeError("H3 tiling: tile and full-frame layouts have different segments")
    frame, _ = h3._frame_grid(full_h, full_w)
    window = frame.reshape(full_h // PATCH, full_w // PATCH, 2)[y0 // PATCH:(y0 + tile_h) // PATCH,
                                                                 x0 // PATCH:(x0 + tile_w) // PATCH].reshape(-1, 2)
    pos = layout.position_ids.clone()
    full_pos = full_layout.position_ids
    for (a, b, kind), (fa, fb, _) in zip(layout.segments, full_layout.segments):
        if kind in ("video", "cond"):
            frames = (b - a) // window.shape[0]
            pos[a:b, 1:3] = window.repeat(frames, 1)
        elif kind in ("audio", "cond_audio"):
            pos[a:b] = full_pos[fa:fb]
    layout.position_ids = pos
    return layout


def check_guide_sizes(payload, height, width):
    """Keyframe guides must be encoded at the latent's own size; H3 cannot use a guide of another size."""
    for kf in (payload or {}).get("keyframes") or ():
        z = kf.get("latent")
        if z is None:
            continue
        gh, gw = int(z.shape[-2]), int(z.shape[-1])
        if (_even(gh), _even(gw)) != (_even(height), _even(width)):
            raise RuntimeError(
                "H3 tiled sampling: a keyframe guide (first/last frame or Add Guide) was encoded at %dx%d px, but the "
                "latent being sampled is %dx%d px. This happens when the conditioning from the first pass is reused "
                "after the latent was upscaled. Build the guide again at the upscaled size (for example a second "
                "MiniMax H3 Image to Video with the upscaled width/height feeding this sampler), or use conditioning "
                "without keyframe guides for the refine pass."
                % (gw * LATENT_PX, gh * LATENT_PX, width * LATENT_PX, height * LATENT_PX))


class H3TiledForward:
    """DIFFUSION_MODEL wrapper that runs the H3 model per spatial tile and blends the video outputs."""

    MAX_CACHE = 256
    MAX_FULL_CACHE = 8

    def __init__(self, tile_width, tile_height, tile_overlap, true_positions=True):
        self.tile_width = int(tile_width)
        self.tile_height = int(tile_height)
        self.tile_overlap = int(tile_overlap)
        self.true_positions = bool(true_positions)
        self._layouts = _LRU(self.MAX_CACHE)
        self._full_layouts = _LRU(self.MAX_FULL_CACHE)

    # -- geometry ---------------------------------------------------------------------------------------------
    def grid(self, height, width):
        """(tile_h, ys, tile_w, xs) in latent units for a latent of the given size (padded to even)."""
        hp, wp = _even(height), _even(width)
        th, ys = axis_tiles(hp, _px_to_even_latent(self.tile_height, PATCH), _px_to_even_latent(self.tile_overlap, 0))
        tw, xs = axis_tiles(wp, _px_to_even_latent(self.tile_width, PATCH), _px_to_even_latent(self.tile_overlap, 0))
        return th, ys, tw, xs

    # -- guards -----------------------------------------------------------------------------------------------
    @staticmethod
    def check_compatible(wrappers, patches_replace=None):
        tiled = 0
        for wtype, per_key in (wrappers or {}).items():
            for fns in per_key.values():
                for fn in fns:
                    while isinstance(fn, functools.partial):
                        fn = fn.func
                    if isinstance(fn, H3TiledForward):
                        tiled += 1
                        if tiled > 1:
                            raise RuntimeError("H3 tiled sampling is applied twice; use only one tiled-sampling node.")
                        continue
                    owner = type(getattr(fn, "__self__", None))
                    ident = "%s %s %s %s %s" % (getattr(fn, "__module__", ""), getattr(fn, "__qualname__", ""),
                                                owner.__name__, type(fn).__qualname__, type(fn).__module__)
                    low = ident.lower()
                    if "funcontrol" in low:
                        raise RuntimeError("H3 tiled sampling cannot be combined with MiniMax H3 FunControl: its control "
                                           "video is encoded for the full frame, not per tile.")
                    if "spectrum" in low:
                        raise RuntimeError("H3 tiled sampling cannot be combined with Spectrum: it forecasts one hidden "
                                           "state per model call, and tiling makes several calls per step.")
        for blk in ((patches_replace or {}).get("dit") or {}).values():
            if "funcontrol" in type(blk).__name__.lower():
                raise RuntimeError("H3 tiled sampling cannot be combined with MiniMax H3 FunControl.")

    # -- layouts ----------------------------------------------------------------------------------------------
    def _full_layout(self, payload, text_len, t, hp, wp, audio_t):
        layout = payload.get("layout")
        if layout is not None and layout.signature == (text_len, t, hp, wp, audio_t):
            return layout
        key = (text_len, t, hp, wp, audio_t, _kf_signature(payload.get("keyframes")), _ref_signature(payload.get("refs")))
        layout = self._full_layouts.fetch(key)
        if layout is None:
            layout = h3.PackedLayout(text_len, t, hp, wp, audio_t, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
            self._full_layouts.put(key, layout)
        return layout

    def _tile_layout(self, payload, tile_payload, text_len, t, th, tw, y0, x0, hp, wp, audio_t):
        key = (text_len, t, th, tw, y0, x0, hp, wp, audio_t, self.true_positions,
               _kf_signature(payload.get("keyframes")), _ref_signature(payload.get("refs")))
        layout = self._layouts.fetch(key)
        if layout is None:
            full = self._full_layout(payload, text_len, t, hp, wp, audio_t) if self.true_positions else None
            layout = tile_layout(full, text_len, t, th, tw, audio_t, y0, x0, hp, wp,
                                 tile_payload.get("keyframes"), tile_payload.get("refs"), self.true_positions)
            self._layouts.put(key, layout)
        return layout

    # -- the wrapper ------------------------------------------------------------------------------------------
    def __call__(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        is_list = isinstance(x, (list, tuple))
        video = x[0] if is_list else x
        if video.ndim != 5:
            return executor(x, timestep, context, transformer_options, **kwargs)
        height, width = int(video.shape[-2]), int(video.shape[-1])
        check_guide_sizes(kwargs.get("minimax_payload"), height, width)
        th, ys, tw, xs = self.grid(height, width)
        if len(ys) == 1 and len(xs) == 1:
            return executor(x, timestep, context, transformer_options, **kwargs)
        if h3 is None:
            raise RuntimeError("MiniMax H3 is not available in this ComfyUI")
        self.check_compatible((transformer_options or {}).get("wrappers"),
                              (transformer_options or {}).get("patches_replace"))

        payload = kwargs.get("minimax_payload") or {}
        hp, wp = _even(height), _even(width)
        t = int(video.shape[2])
        text_len = int(context.shape[1])
        audio_t = int(x[1].shape[-1]) if is_list and len(x) > 1 else 0
        denoise_mask = kwargs.get("denoise_mask")

        acc = torch.zeros(video.shape, dtype=torch.float32, device=video.device)
        wsum = torch.zeros((1, 1, 1, height, width), dtype=torch.float32, device=video.device)
        extra_acc = None  # audio outputs: plain mean over tiles (every tile saw the same audio)
        count = 0

        # Every tile call has to look like one ordinary model call to the allocation-graph recorder. forward() opened
        # a recording before running the wrappers: close it and give each tile its own scope.
        graph = model_prefetch.MALLOC_GRAPHS.get(threading.get_ident()) if model_prefetch is not None else None
        scoped = graph is not None and getattr(graph, "_comfy_active", False)
        if scoped:
            model_prefetch.malloc_graph_end()

        for (iy, y0), (ix, x0) in itertools.product(enumerate(ys), enumerate(xs)):
            y1, x1 = y0 + th, x0 + tw
            ye, xe = min(y1, height), min(x1, width)
            tile_kwargs = dict(kwargs)
            tile_payload = crop_payload(payload, y0, ye, x0, xe)
            tile_payload["layout"] = self._tile_layout(payload, tile_payload, text_len, t, th, tw, y0, x0,
                                                       hp, wp, audio_t)
            tile_kwargs["minimax_payload"] = tile_payload
            if denoise_mask is not None:
                tile_kwargs["denoise_mask"] = _crop_hw(denoise_mask, y0, ye, x0, xe)
            tile_video = video[..., y0:ye, x0:xe]
            tile_x = [tile_video] + list(x[1:]) if is_list else tile_video
            scope = model_prefetch.malloc_graph_scope(video.device) if scoped else nullcontext()
            with scope:
                out = executor(tile_x, timestep, context, transformer_options, **tile_kwargs)
            out_video = out[0] if is_list else out
            w = (axis_weights(ys, th, iy)[:ye - y0, None] * axis_weights(xs, tw, ix)[None, :xe - x0]).to(video.device)
            acc[..., y0:ye, x0:xe] += out_video.to(torch.float32) * w
            wsum[..., y0:ye, x0:xe] += w
            if is_list and len(out) > 1:
                if extra_acc is None:
                    extra_acc = [o.to(torch.float32).clone() for o in out[1:]]
                else:
                    for a, o in zip(extra_acc, out[1:]):
                        a += o.to(torch.float32)
            del out, out_video
            count += 1

        blended = (acc / wsum).to(video.dtype)
        if not is_list:
            return blended
        result = [blended]
        if extra_acc is not None:
            result += [(a / count).to(o.dtype) for a, o in zip(extra_acc, x[1:])]
        return result


class UltimateSDUpscaleH3TiledSampling:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "A MiniMax H3 model (video+audio, or the image model)."}),
                "tile_width": ("INT", {"default": 768, "min": 64, "max": 16384, "step": 32,
                                       "tooltip": "Tile width in pixels. Rounded to an even number of latent units (32 px)."}),
                "tile_height": ("INT", {"default": 768, "min": 64, "max": 16384, "step": 32,
                                        "tooltip": "Tile height in pixels. A frame smaller than a tile in one direction uses the full size there."}),
                "tile_overlap": ("INT", {"default": 128, "min": 0, "max": 4096, "step": 32,
                                         "tooltip": "Overlap between neighbouring tiles in pixels. Overlapping outputs are blended with a linear feather."}),
                "true_positions": ("BOOLEAN", {"default": True,
                                               "tooltip": "Give every tile the rotary positions it has in the full frame instead of tile-local ones, so the model sees the tile at its real place and scale. Turn off only to compare."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "image/upscaling"
    DESCRIPTION = ("MultiDiffusion-style tiling for MiniMax H3 sampling. On every sampler step the video latent is split "
                   "into overlapping spatial tiles (full time length each), the model runs once per tile, and the tile "
                   "outputs are blended with feathered weights. The audio is passed whole to every tile. Use it after "
                   "upscaling the video pixels -> VAE encode -> sampler with denoise < 1, in place of the plain model, "
                   "to refine a large video that would not fit in memory in one piece. Not compatible with FunControl or Spectrum.")

    def apply(self, model, tile_width, tile_height, tile_overlap, true_positions):
        if h3 is None:
            raise RuntimeError("This ComfyUI has no MiniMax H3 support.")
        H3TiledForward.check_compatible(model.wrappers, model.model_options.get("transformer_options", {}).get("patches_replace"))
        for fns in model.wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {}).values():
            for fn in fns:
                while isinstance(fn, functools.partial):
                    fn = fn.func
                if isinstance(fn, H3TiledForward):
                    raise RuntimeError("This model already has H3 tiled sampling applied; stacking two tilings is not supported.")
        m = model.clone()
        m.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                      H3TiledForward(tile_width, tile_height, tile_overlap, true_positions))
        return (m,)


class UltimateSDUpscaleH3HoldAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "A MiniMax H3 audio+video latent (e.g. from a VAE encode of the upscaled video plus the original audio)."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "apply"
    CATEGORY = "image/upscaling"
    DESCRIPTION = ("Marks the audio part of an H3 audio+video latent as fixed: the sampler denoises the video and keeps the "
                   "audio. Combines with an existing video noise_mask. A latent that is not audio+video passes through unchanged.")

    def apply(self, latent):
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False) or len(samples.unbind()) < 2:
            return (latent,)
        video, audio = samples.unbind()[0], samples.unbind()[1]
        existing = latent.get("noise_mask")
        if existing is not None and getattr(existing, "is_nested", False):
            video_mask = existing.unbind()[0]
        elif existing is not None:
            video_mask = existing
        else:
            video_mask = torch.ones((video.shape[0], 1) + tuple(video.shape[2:]), dtype=torch.float32, device=video.device)
        audio_mask = torch.zeros((audio.shape[0], 1) + tuple(audio.shape[2:]), dtype=torch.float32, device=audio.device)
        out = latent.copy()
        out["noise_mask"] = comfy.nested_tensor.NestedTensor([video_mask, audio_mask])
        return (out,)


NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleH3TiledSampling": UltimateSDUpscaleH3TiledSampling,
    "UltimateSDUpscaleH3HoldAudio": UltimateSDUpscaleH3HoldAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleH3TiledSampling": "Ultimate SD Upscale: MiniMax H3 Tiled Sampling",
    "UltimateSDUpscaleH3HoldAudio": "Ultimate SD Upscale: MiniMax H3 Hold Audio",
}
