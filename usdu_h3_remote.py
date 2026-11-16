# Two-GPU tile dispatch for the MiniMax H3 tiled sampler.
#
# The "(2 GPU)" node tiles exactly like the single-GPU node (usdu_h3_tiled.py), but hands some of the tiles of every
# model call to a second ComfyUI server (the worker, e.g. started with --cuda-device 1 on port 8189) and blends all
# tile outputs on this side. How it works:
#
#   main (this server)                                  worker (second ComfyUI server)
#   ------------------                                  ------------------------------
#   OUTER_SAMPLE wrapper opens a session:               runs a normal prompt: the loaders of the model subgraph
#     POST /prompt  (model subgraph + sink node)  --->  followed by the sink node UltimateSDUpscaleH3TileWorker
#     heartbeat every ~10 s                             The sink checks the model signature, waits for the plan,
#   per model call: tiles split between the two  --->   loads the model through the sampler lifecycle and then
#     POST /usdu_h3/<session>/tile (safetensors)        serves tile requests from the routes in its own prompt thread
#   on close: done / queue delete / interrupt           (the routes only decode and enqueue).
#
# Only safetensors and JSON cross the wire, every request carries the session token, and declared shapes are checked
# against the tensors. Any remote problem makes the remaining tiles run locally; the run never fails because of it.

import asyncio
import collections
import concurrent.futures
import contextlib
import copy
import functools
import hashlib
import hmac
import json
import logging
import queue
import secrets
import statistics
import struct
import threading
import time
import types
import urllib.error
import urllib.parse
import urllib.request
import uuid

import torch

import comfy
import comfy.patcher_extension

import usdu_h3_tiled as base
from usdu_h3_tiled import H3TiledForward, UltimateSDUpscaleH3TiledSampling, TileSpec

try:
    import safetensors.torch as st
except Exception as e:  # pragma: no cover - ComfyUI always ships safetensors
    st = None
    logging.getLogger(__name__).info("safetensors not available (%s); the 2-GPU H3 node is disabled", e)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
ROUTE_PREFIX = "/usdu_h3/session"
SINK_NODE_ID = "usdu_h3_tile_worker"
SINK_CLASS = "UltimateSDUpscaleH3TileWorker"
TOKEN_HEADER = "X-USDU-Token"
KEY_HEADER = "X-USDU-Key"

HEARTBEAT_S = 10.0  # main -> worker keep-alive interval
IDLE_TIMEOUT_S = 60.0  # the sink leaves when nothing (request or heartbeat) arrived for this long
POLL_S = 0.2  # sink queue poll interval
MAX_BODY_BYTES = 96 * 1024 * 1024  # stay under ComfyUI's default --max-upload-size of 100 MB
MAX_HEADER_BYTES = 1 << 20
MAX_TILE_ELEMENTS = 1 << 28
MIN_TILE_TIMEOUT_S = 30.0
TILE_TIMEOUT_FACTOR = 5.0
FIRST_TILE_TIMEOUT_S = 600.0  # until the worker has delivered a tile: its first one loads the model onto its GPU
MAX_TREE_DEPTH = 32
MAX_TREE_NODES = 100000
MAX_SETUPS = 4  # worker keeps this many per-session setups (context + payload)

KNOWN_KWARGS = frozenset({"minimax_payload", "denoise_mask", "audio_denoise_mask", "control"})
ALLOWED_OPTION_KEYS = frozenset({"sigmas", "sample_sigmas", "cond_or_uncond", "uuids", "prefetch_dynamic_vbars", "wrappers",
                                 "callbacks", "patches", "patches_replace", "minimax_h3_layout", "block_index"})
STEP_TENSOR_KEYS = ("sigmas", "sample_sigmas")


class RemoteError(Exception):
    """A problem with the worker or the wire; the caller falls back to local tiles."""


class CodecError(ValueError):
    """A wire blob that is malformed or does not match what it declares."""


# ---------------------------------------------------------------------------------------------------------------------
# Codec: safetensors bytes + a JSON tree. No pickle anywhere.
# ---------------------------------------------------------------------------------------------------------------------

_DTYPE_NAMES = {"F64", "F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8", "BOOL"}


def encode(tensors, meta=None):
    """tensors: {name: Tensor}; meta: JSON-able dict. Returns safetensors bytes (tensors keep their dtype)."""
    out = {}
    for name, t in tensors.items():
        if not torch.is_tensor(t):
            raise CodecError("tensor '%s' is not a tensor" % name)
        # own storage for every tensor: safetensors refuses tensors that share memory (views of one packed latent,
        # the timestep that is also transformer_options["sigmas"], ...)
        c = t.detach()
        out[name] = (c.clone() if c.device.type == "cpu" else c.cpu()).contiguous()
    return st.save(out, metadata={"json": json.dumps(meta or {}, sort_keys=True, separators=(",", ":"))})


def decode(data, max_bytes=MAX_BODY_BYTES):
    """Inverse of encode. Validates the header before touching the tensors. Returns (tensors, meta)."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CodecError("blob is not bytes")
    data = bytes(data)
    if len(data) < 8 or len(data) > max_bytes:
        raise CodecError("blob size %d out of range" % len(data))
    (hlen,) = struct.unpack("<Q", data[:8])
    if hlen <= 0 or hlen > MAX_HEADER_BYTES or 8 + hlen > len(data):
        raise CodecError("bad safetensors header length")
    try:
        header = json.loads(data[8:8 + hlen].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise CodecError("bad safetensors header: %s" % e)
    if not isinstance(header, dict):
        raise CodecError("bad safetensors header")
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or info.get("dtype") not in _DTYPE_NAMES:
            raise CodecError("tensor '%s' has an unsupported dtype" % name)
    try:
        tensors = st.load(data)
    except Exception as e:
        raise CodecError("bad safetensors data: %s" % e)
    meta_text = (header.get("__metadata__") or {}).get("json", "{}")
    try:
        meta = json.loads(meta_text)
    except ValueError as e:
        raise CodecError("bad metadata: %s" % e)
    if not isinstance(meta, dict):
        raise CodecError("metadata is not an object")
    return tensors, meta


def pack_tree(obj, tensors, ids=None):
    """Turn a nested dict/list of tensors and JSON scalars into a JSON tree; tensors go to `tensors` (shared ones once)."""
    ids = {} if ids is None else ids
    if torch.is_tensor(obj):
        key = id(obj)
        if key not in ids:
            name = "t%d" % len(tensors)
            tensors[name] = obj
            ids[key] = name
        return {"t": ids[key]}
    if isinstance(obj, dict):
        packed = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise CodecError("dict key %r is not a string" % (k,))
            packed[k] = pack_tree(v, tensors, ids)
        return {"d": packed}
    if isinstance(obj, (list, tuple)):
        return {"l": [pack_tree(v, tensors, ids) for v in obj]}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise CodecError("cannot send a value of type %s" % type(obj).__name__)


def unpack_tree(node, tensors, _depth=0, _budget=None):
    """Inverse of pack_tree. Depth and node count are limited so a hostile blob can not exhaust the stack or memory."""
    if _budget is None:
        _budget = [MAX_TREE_NODES]
    _budget[0] -= 1
    if _depth > MAX_TREE_DEPTH or _budget[0] < 0:
        raise CodecError("payload tree is too deep or too large")
    if isinstance(node, dict):
        if set(node) == {"t"}:
            if node["t"] not in tensors:
                raise CodecError("tree names a missing tensor")
            return tensors[node["t"]]
        if set(node) == {"d"} and isinstance(node["d"], dict):
            return {k: unpack_tree(v, tensors, _depth + 1, _budget) for k, v in node["d"].items()}
        if set(node) == {"l"} and isinstance(node["l"], list):
            return [unpack_tree(v, tensors, _depth + 1, _budget) for v in node["l"]]
        raise CodecError("malformed tree node")
    if node is None or isinstance(node, (bool, int, float, str)):
        return node
    raise CodecError("malformed tree value")


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_device(v, device) for v in obj]
    return obj


# ---------------------------------------------------------------------------------------------------------------------
# Model signature: main and worker must be running the same model with the same patches.
# ---------------------------------------------------------------------------------------------------------------------

def _simple(v):
    """repr of a plain value, the type name of anything else (its state can not be compared)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return repr(v)
    return "<%s>" % type(v).__qualname__


def _callable_name(fn):
    """Identity of a wrapper or callback: module.qualname|owner type, plus simple partial arguments.
    None for our own tiling wrapper (it only exists on main). Instance state of other objects is not compared."""
    partials = []
    while isinstance(fn, functools.partial):
        partials.append("partial(%s;%s)" % (",".join(_simple(a) for a in fn.args),
                                            ",".join("%s=%s" % (k, _simple(v)) for k, v in sorted((fn.keywords or {}).items()))))
        fn = fn.func
    if isinstance(fn, H3TiledForward):
        return None
    owner = getattr(fn, "__self__", None)
    name = "%s.%s|%s" % (getattr(fn, "__module__", type(fn).__module__),
                         getattr(fn, "__qualname__", type(fn).__qualname__), type(owner).__name__ if owner is not None else "")
    return name + "".join("|" + p for p in partials)


def _fn_id(fn):
    """Like _callable_name but never None; used for patch functions and patched objects."""
    if fn is None:
        return None
    if isinstance(fn, (functools.partial,)) or callable(fn) and not isinstance(fn, torch.nn.Module):
        name = _callable_name(fn)
        return name if name is not None else "H3TiledForward"
    return "%s.%s" % (type(fn).__module__, type(fn).__qualname__)


def _patch_fingerprint(entry):
    # entry: (strength_patch, patch, strength_model, offset, function). Any failure propagates: the signature then
    # becomes unmatchable (model_signature) instead of trusting a partial fingerprint.
    out = [float(entry[0]), float(entry[2])]
    patch = entry[1]
    out.append(type(patch).__name__)
    for item in (patch if isinstance(patch, (tuple, list)) else [patch]):
        if torch.is_tensor(item):
            out.append([str(item.dtype), list(item.shape)])
        else:
            out.append(type(item).__name__)
    out.append(_simple(entry[3]) if len(entry) > 3 else None)  # offset
    out.append(_fn_id(entry[4]) if len(entry) > 4 else None)  # function
    return out


def model_signature(patcher):
    """Hash of everything that makes two ModelPatchers produce the same tile output (contents of weights excluded:
    both servers load the same files, only the structure, strengths and wrapper stack are compared)."""
    try:
        info = _signature_info(patcher)
    except Exception as e:
        # a part that can not be fingerprinted must never be trusted: an unmatchable value sends the run local
        logger.warning("USDU H3 2-GPU: cannot fingerprint the model (%s: %s); the worker will not be used",
                       type(e).__name__, e)
        info = {"unmatchable": secrets.token_hex(16)}
    return hashlib.sha256(json.dumps(info, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _signature_info(patcher):
    inner = patcher.model
    sampling = getattr(inner, "model_sampling", None)
    sampling_params = {}
    for name in ("shift", "audio_shift", "multiplier", "audio_scale"):
        v = getattr(sampling, name, None)
        if isinstance(v, (int, float)):
            sampling_params[name] = float(v)
    to = (getattr(patcher, "model_options", None) or {}).get("transformer_options", {}) or {}
    dit_wrappers = list(comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, to))
    for fns in ((getattr(patcher, "wrappers", None) or {}).get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL) or {}).values():
        dit_wrappers.extend(fns)
    # order matters (main and worker build the stack the same way); the state inside a wrapper instance is not compared
    wrappers = [n for n in (_callable_name(f) for f in dit_wrappers) if n is not None]
    dtype = str(inner.get_dtype_inference())
    return {
        "class": "%s.%s" % (type(inner).__module__, type(inner).__name__),
        "diffusion_model": type(getattr(inner, "diffusion_model", None)).__name__,
        "sampling": [c.__name__ for c in type(sampling).__mro__[:4]] if sampling is not None else None,
        "sampling_params": sampling_params,
        "dtype": dtype,
        "patches": sorted((str(k), [_patch_fingerprint(e) for e in v]) for k, v in (getattr(patcher, "patches", None) or {}).items()),
        "object_patches": sorted((str(k), _fn_id(v) if callable(v) else "%s.%s" % (type(v).__module__, type(v).__qualname__))
                                 for k, v in (getattr(patcher, "object_patches", None) or {}).items()),
        "wrappers": wrappers,
        "options": sorted((str(k), v) for k, v in to.items() if isinstance(v, (str, int, float, bool))),
    }


# ---------------------------------------------------------------------------------------------------------------------
# Subgraph closure (R3): the part of the prompt that produces our MODEL input, as a worker prompt.
# ---------------------------------------------------------------------------------------------------------------------

class ClosureError(Exception):
    pass


def _is_link(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (str, int)) \
        and not isinstance(value[0], bool) and isinstance(value[1], int)


def _node_classes():
    import nodes
    return nodes.NODE_CLASS_MAPPINGS


def _input_types(cls):
    declared = {}
    try:
        spec = cls.INPUT_TYPES()
    except Exception as e:
        raise ClosureError("cannot read the inputs of %s: %s" % (getattr(cls, "__name__", cls), e))
    for section in ("required", "optional"):
        for name, v in (spec.get(section) or {}).items():
            declared[name] = v[0] if isinstance(v, (tuple, list)) and v else v
    return declared


def _return_types(cls):
    rt = getattr(cls, "RETURN_TYPES", None)
    if rt is None:
        raise ClosureError("%s has no RETURN_TYPES" % getattr(cls, "__name__", cls))
    return tuple(rt)


def build_closure(prompt, node_id, input_name="model", classes=None):
    """Walk upstream from node_id's `input_name` link. Returns (closure {id: node}, (source id, slot)).

    Only MODEL links are followed and widget values are copied. Refuses (ClosureError) when any node in the closure
    has a non-MODEL link input, is an output node, or a referenced id is missing from the prompt.
    """
    classes = _node_classes() if classes is None else classes
    node = prompt.get(str(node_id))
    if node is None:
        raise ClosureError("this node (%s) is not in the prompt (expanded or ephemeral graph)" % node_id)
    start = (node.get("inputs") or {}).get(input_name)
    if not _is_link(start):
        raise ClosureError("the '%s' input is not connected to another node" % input_name)
    closure = {}
    todo = [(str(start[0]), int(start[1]))]
    first = todo[0]
    while todo:
        nid, slot = todo.pop()
        src = prompt.get(nid)
        if src is None:
            raise ClosureError("node %s referenced by the model subgraph is missing from the prompt" % nid)
        cls = classes.get(src.get("class_type"))
        if cls is None:
            raise ClosureError("node %s has unknown class %s" % (nid, src.get("class_type")))
        rtypes = _return_types(cls)
        if slot >= len(rtypes) or rtypes[slot] not in ("MODEL", "*"):
            raise ClosureError("node %s output %d is not a MODEL" % (nid, slot))
        if rtypes[slot] == "*" and not any(_is_link(v) for v in (src.get("inputs") or {}).values()):
            raise ClosureError("node %s has an any-type output and no linked input to tell its type" % nid)
        if nid in closure:
            continue
        if getattr(cls, "OUTPUT_NODE", False):
            raise ClosureError("node %s (%s) is an output node" % (nid, src.get("class_type")))
        declared = _input_types(cls)
        inputs = {}
        for name, value in (src.get("inputs") or {}).items():
            if _is_link(value):
                if declared.get(name) not in ("MODEL", "*"):  # "*" inputs are fine when the source resolves to MODEL
                    raise ClosureError("node %s (%s) has a %s link input '%s'; only MODEL links can be replayed on the worker"
                                       % (nid, src.get("class_type"), declared.get(name), name))
                todo.append((str(value[0]), int(value[1])))
                inputs[name] = [str(value[0]), int(value[1])]
            else:
                inputs[name] = copy.deepcopy(value)
        closure[nid] = {"class_type": src["class_type"], "inputs": inputs}
    return closure, (first[0], first[1])


def worker_prompt(closure, source, session_id, token, signature):
    prompt = copy.deepcopy(closure)
    prompt[SINK_NODE_ID] = {"class_type": SINK_CLASS, "inputs": {
        "model": [source[0], source[1]], "session_id": session_id, "token": token, "main_signature": signature}}
    return prompt


# ---------------------------------------------------------------------------------------------------------------------
# Worker side: sessions, routes and the sink node.
# ---------------------------------------------------------------------------------------------------------------------

def _digest(token):
    return hashlib.sha256(str(token).encode("utf-8")).digest()


def _token_ok(expected, given):
    return given is not None and hmac.compare_digest(_digest(expected), _digest(given))


class TileRequest:
    def __init__(self, tensors, meta, setup):
        self.tensors = tensors
        self.meta = meta
        self.setup = setup
        self.future = concurrent.futures.Future()


class WorkerSession:
    """State of one run on the worker, shared between the routes (event loop) and the sink (prompt thread)."""

    def __init__(self, session_id, token, clock=time.monotonic):
        self.id = session_id
        self.token = token
        self.clock = clock
        self.requests = queue.Queue()
        self.setups = base._LRU(MAX_SETUPS)
        self.lock = threading.Lock()
        self.plan = None
        self.plan_event = threading.Event()
        self.closed = threading.Event()
        self.state = "waiting_plan"
        self.last_seen = clock()

    def touch(self):
        self.last_seen = self.clock()

    def idle_expired(self, timeout):
        return self.clock() - self.last_seen > timeout


_REG_LOCK = threading.Lock()
SESSIONS = {}  # session id -> WorkerSession of a sink that is running
TOMBSTONES = collections.OrderedDict()  # session id -> token digest of sessions that were closed
REFUSED = collections.OrderedDict()  # session id -> (token digest, reason) of sinks that refused the session
_KEEP = 256


def reset_worker_state():
    with _REG_LOCK:
        SESSIONS.clear()
        TOMBSTONES.clear()
        REFUSED.clear()


def _remember(table, key, value):
    table[key] = value
    table.move_to_end(key)
    while len(table) > _KEEP:
        table.popitem(last=False)


def core_status(session_id, token):
    with _REG_LOCK:
        s = SESSIONS.get(session_id)
        if s is not None:
            if not _token_ok(s.token, token):
                return 403, {"error": "bad token"}
            s.touch()  # main polls the status while it waits for the worker: that is a sign of life too
            return 200, {"state": s.state}
        refused = REFUSED.get(session_id)
        if refused is not None and token is not None and hmac.compare_digest(refused[0], _digest(token)):
            return 200, {"state": "refused", "reason": refused[1]}
        tomb = TOMBSTONES.get(session_id)
        if tomb is not None and token is not None and hmac.compare_digest(tomb, _digest(token)):
            return 200, {"state": "closed"}
    return 200, {"state": "pending"}


def _find(session_id, token):
    with _REG_LOCK:
        s = SESSIONS.get(session_id)
    if s is None:
        return None, (409, {"error": "session is not running on this worker"})
    if not _token_ok(s.token, token):
        return None, (403, {"error": "bad token"})
    return s, None


def core_heartbeat(session_id, token):
    s, err = _find(session_id, token)
    if err:
        return err
    s.touch()
    return 200, {"state": s.state}


def core_done(session_id, token):
    with _REG_LOCK:
        s = SESSIONS.get(session_id)
        if s is not None and not _token_ok(s.token, token):
            return 403, {"error": "bad token"}
        if token is not None:
            _remember(TOMBSTONES, session_id, _digest(token))
    if s is not None:
        s.closed.set()
    return 200, {"ok": True}


def core_plan(session_id, token, body):
    s, err = _find(session_id, token)
    if err:
        return err
    try:
        plan = json.loads(bytes(body).decode("utf-8"))
        shape = [int(v) for v in plan["tile_shape"]]
        if len(shape) != 5 or shape[0] != 1 or any(v < 1 or v > 1 << 16 for v in shape):
            raise ValueError("bad tile_shape")
    except Exception as e:
        return 400, {"error": "bad plan: %s" % e}
    s.touch()
    if s.plan is None:
        s.plan = {"tile_shape": shape}
        s.plan_event.set()
    return 200, {"state": s.state}


def core_setup(session_id, token, key, body):
    s, err = _find(session_id, token)
    if err:
        return err
    if key is None or hashlib.sha256(bytes(body)).hexdigest() != key:
        return 400, {"error": "setup key does not match its content"}
    try:
        tensors, meta = decode(body)
        if "context" not in tensors or tensors["context"].ndim != 3:
            raise CodecError("setup needs a [B, tokens, dim] context")
        payload = unpack_tree(meta.get("payload"), {k: v for k, v in tensors.items() if k != "context"})
        if not isinstance(payload, dict):
            raise CodecError("setup payload is not a dict")
        true_positions = bool(meta.get("true_positions", True))
    except CodecError as e:
        return 400, {"error": str(e)}
    s.touch()
    entry = types.SimpleNamespace(context=tensors["context"], payload=payload,
                            fwd=H3TiledForward(1, 1, 0, true_positions), on_device=None)
    with s.lock:
        s.setups.put(key, entry)
    return 200, {"ok": True}


def _int_field(meta, name, lo, hi):
    v = meta.get(name)
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise CodecError("tile field '%s' is missing or out of range" % name)
    return v


def validate_tile_request(tensors, meta, setup):
    """Check declared geometry against the tensors and the session's setup. Raises CodecError."""
    tile = meta.get("tile")
    if not isinstance(tile, dict):
        raise CodecError("missing tile geometry")
    g = {n: _int_field(tile, n, 0, 1 << 16) for n in ("y0", "x0", "ye", "xe", "th", "tw", "hp", "wp", "height", "width",
                                                       "t", "text_len", "audio_t", "index")}
    video = tensors.get("video")
    if video is None or video.ndim != 5 or video.shape[0] != 1:
        raise CodecError("video tile must be [1, C, T, H, W]")
    if g["ye"] <= g["y0"] or g["xe"] <= g["x0"] or g["ye"] > g["height"] or g["xe"] > g["width"]:
        raise CodecError("tile window outside the frame")
    if (g["ye"] - g["y0"], g["xe"] - g["x0"]) != tuple(video.shape[3:]):
        raise CodecError("tile window does not match the video tile shape")
    if g["t"] != video.shape[2]:
        raise CodecError("tile time length does not match")
    if g["th"] < 2 or g["tw"] < 2 or g["th"] % 2 or g["tw"] % 2 or g["hp"] % 2 or g["wp"] % 2:
        raise CodecError("tile size must be even")
    if g["y0"] + g["th"] < g["ye"] or g["x0"] + g["tw"] < g["xe"] or g["hp"] < g["height"] or g["wp"] < g["width"]:
        raise CodecError("inconsistent padded sizes")
    if video.numel() > MAX_TILE_ELEMENTS or g["th"] * g["tw"] * g["t"] > MAX_TILE_ELEMENTS:
        raise CodecError("tile too large")
    if g["text_len"] != setup.context.shape[1]:
        raise CodecError("text length does not match the setup context")
    audio = tensors.get("audio")
    if audio is None:
        if g["audio_t"] != 0:
            raise CodecError("audio length declared without audio")
    elif audio.ndim < 2 or audio.shape[0] != 1 or audio.shape[-1] != g["audio_t"]:
        raise CodecError("audio shape does not match")
    dm = tensors.get("denoise_mask")
    if dm is not None and (dm.ndim != 5 or tuple(dm.shape[-2:]) != tuple(video.shape[3:])):
        raise CodecError("denoise mask does not match the tile")
    adm = tensors.get("audio_denoise_mask")
    if adm is not None and adm.numel() > MAX_TILE_ELEMENTS:
        raise CodecError("audio denoise mask too large")
    ts = tensors.get("timestep")
    if ts is None or ts.numel() < 1 or ts.numel() > 16:
        raise CodecError("missing timestep")
    for key in tensors:
        if key not in ("video", "audio", "denoise_mask", "audio_denoise_mask", "timestep") + STEP_TENSOR_KEYS:
            raise CodecError("unexpected tensor '%s'" % key)
    for name in ("sigmas", "sample_sigmas"):
        t = tensors.get(name)
        if t is not None and t.numel() > 1 << 16:
            raise CodecError("%s too large" % name)
    cu = meta.get("cond_or_uncond", [])
    uu = meta.get("uuids", [])
    if not isinstance(cu, list) or not all(isinstance(v, int) and not isinstance(v, bool) for v in cu) or len(cu) > 64:
        raise CodecError("bad cond_or_uncond")
    if not isinstance(uu, list) or not all(isinstance(v, str) for v in uu) or len(uu) > 64:
        raise CodecError("bad uuids")
    return g


def core_tile_submit(session_id, token, body):
    """Validate and enqueue one tile request. Returns (None, TileRequest) or (status, error payload)."""
    s, err = _find(session_id, token)
    if err:
        return err[0], err[1]
    if s.state != "ready":
        return 409, {"error": "worker is not ready (%s)" % s.state}
    try:
        tensors, meta = decode(body)
        key = meta.get("setup")
        with s.lock:
            setup = s.setups.fetch(key)
        if setup is None:
            return 410, {"error": "setup unknown"}
        validate_tile_request(tensors, meta, setup)
    except CodecError as e:
        return 400, {"error": str(e)}
    s.touch()
    req = TileRequest(tensors, meta, setup)
    with s.lock:  # same lock as the teardown's close + drain, so nothing is queued after the last drain
        if s.closed.is_set():
            return 409, {"error": "session closed"}
        s.requests.put(req)
    return None, req


# -- worker: running a tile in the sampler lifecycle ------------------------------------------------------------------

class WorkerContext:
    def __init__(self, patcher, device, model_options):
        self.patcher = patcher
        self.device = device
        self.model_options = model_options
        self.transformer_options = model_options["transformer_options"]


@contextlib.contextmanager
def worker_lifecycle(patcher, noise_shape):
    """The part of CFGGuider.outer_sample that matters for running the diffusion model, around a serve loop:
    prepare_sampling (load with a tile-shaped memory estimate), prepare_model_patcher (merge wrappers and callbacks),
    cuda_device_context, cast_to_load_options, pre_run, writable models; cleanup in the finally."""
    import comfy.model_management
    import comfy.model_patcher
    import comfy.sampler_helpers
    import comfy.samplers

    device = patcher.load_device
    model_options = comfy.model_patcher.create_model_options_clone(patcher.model_options)
    model_options.setdefault("transformer_options", {})
    _, conds, loaded = comfy.sampler_helpers.prepare_sampling(patcher, tuple(noise_shape), {}, model_options)
    try:
        comfy.sampler_helpers.prepare_model_patcher(patcher, {}, model_options)
        with comfy.model_management.cuda_device_context(device):
            try:
                comfy.samplers.cast_to_load_options(model_options, device=device, dtype=patcher.model_dtype())
                patcher.pre_run()
                with patcher.writable_models({}, tuple(noise_shape), model_options):
                    yield WorkerContext(patcher, device, model_options)
            finally:
                patcher.cleanup()
    finally:
        try:
            comfy.samplers.cast_to_load_options(model_options, device=patcher.offload_device)
        finally:
            comfy.sampler_helpers.cleanup_models(conds, loaded)


def run_worker_tile(ctx, req):
    """Run one validated tile request on the worker and return the response bytes.

    Mirrors what main does for a local tile: tile inputs from the same tile_inputs(), the DIFFUSION_MODEL wrappers
    around the model's _forward, one allocation-graph scope per tile. Not done here, as on main: the audio-scale
    correction and the denoise-mask multiply (diffusion model forward applies them outside the wrappers, and main
    applies them after blending).
    """
    import comfy.model_prefetch as model_prefetch

    t, meta, setup = req.tensors, req.meta, req.setup
    g = meta["tile"]
    patcher, device = ctx.patcher, ctx.device
    dm = patcher.model.diffusion_model
    with torch.inference_mode():
        if setup.on_device is None or setup.on_device[0] != device:
            setup.on_device = (device, setup.context.to(device), _to_device(setup.payload, device))
        _, context, payload = setup.on_device
        tile_payload = setup.fwd.tile_inputs(payload, g["text_len"], g["t"], g["th"], g["tw"], g["y0"], g["x0"],
                                             g["ye"], g["xe"], g["hp"], g["wp"], g["audio_t"])
        video = t["video"].to(device)
        has_audio = "audio" in t
        x = [video, t["audio"].to(device)] if has_audio else video
        timestep = t["timestep"].to(device)
        to = dict(ctx.transformer_options)
        to["prefetch_dynamic_vbars"] = bool(patcher.is_dynamic())
        for key in STEP_TENSOR_KEYS:
            if key in t:
                to[key] = t[key].to(device)
        to["cond_or_uncond"] = list(meta.get("cond_or_uncond", []))
        to["uuids"] = list(meta.get("uuids", []))
        kwargs = {"minimax_payload": tile_payload, "control": None}
        if "denoise_mask" in t:
            kwargs["denoise_mask"] = t["denoise_mask"].to(device)
        if "audio_denoise_mask" in t:
            kwargs["audio_denoise_mask"] = t["audio_denoise_mask"].to(device)
        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            dm._forward, dm, comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, to))
        compile_allocations = model_prefetch.malloc_graph_enabled(device)
        if compile_allocations:  # same shape as DiT.forward(): outputs allocated outside the recording, copied out of it
            out = [torch.empty_like(v) for v in (x if has_audio else [x])]
        with model_prefetch.malloc_graph_scope(device) if compile_allocations else contextlib.nullcontext():
            graph_out = executor.execute(x, timestep, context, to, **kwargs)
            graph_list = list(graph_out) if has_audio else [graph_out]
            if compile_allocations:
                for dst, src in zip(out, graph_list):
                    dst.copy_(src)
                result = out
            else:
                result = graph_list
            del graph_out, graph_list
    tensors = {"video": result[0].cpu()}
    for i, extra in enumerate(result[1:]):
        tensors["extra%d" % i] = extra.cpu()
    return encode(tensors, {"index": g["index"]})


def _settle(future, result=None, exc=None):
    """Set a future's outcome unless the route already cancelled it (set_* would raise InvalidStateError)."""
    if future.done():
        return
    try:
        if exc is not None:
            future.set_exception(exc)
        else:
            future.set_result(result)
    except concurrent.futures.InvalidStateError:
        pass


def worker_serve(patcher, session_id, token, main_signature, *, clock=time.monotonic, idle_timeout=IDLE_TIMEOUT_S,
                 poll=POLL_S, check_interrupt=None):
    """Body of the sink node: refuse a mismatching model, wait for the plan, load, then serve tiles until done,
    interrupt, tombstone or idle timeout. Returns a short summary string."""
    if check_interrupt is None:
        import comfy.model_management
        check_interrupt = comfy.model_management.throw_exception_if_processing_interrupted
    token_digest = _digest(token)
    session = WorkerSession(session_id, token, clock)
    with _REG_LOCK:  # tombstone check and registration are one step, so a `done` can not fall in between
        tomb = TOMBSTONES.get(session_id)
        if tomb is not None and hmac.compare_digest(tomb, token_digest):
            return "session %s was closed before it started" % session_id[:8]
        SESSIONS[session_id] = session
    served = 0
    reason = "done"
    try:
        if model_signature(patcher) != main_signature:
            why = "the worker's model differs from main's (class, sampling, patches or wrappers); refusing to serve tiles"
            with _REG_LOCK:
                _remember(REFUSED, session_id, (token_digest, why))
            raise RuntimeError("USDU H3 worker: " + why)
        while not session.plan_event.wait(poll):
            check_interrupt()
            if session.closed.is_set():
                return "session closed before the plan arrived"
            if session.idle_expired(idle_timeout):
                return "idle before the plan arrived"
        check_interrupt()
        session.state = "loading"
        with worker_lifecycle(patcher, session.plan["tile_shape"]) as ctx:
            session.touch()
            session.state = "ready"
            logger.info("USDU H3 worker: session %s ready", session_id[:8])
            while True:
                check_interrupt()
                if session.closed.is_set():
                    reason = "done"
                    break
                if session.idle_expired(idle_timeout):
                    reason = "idle timeout"
                    break
                try:
                    req = session.requests.get(timeout=poll)
                except queue.Empty:
                    continue
                session.touch()
                if not req.future.set_running_or_notify_cancel():
                    continue
                try:
                    _settle(req.future, result=run_worker_tile(ctx, req))
                    served += 1
                except BaseException as e:
                    _settle(req.future, exc=RuntimeError("tile failed: %s: %s" % (type(e).__name__, e)))
                    if not isinstance(e, Exception):
                        raise
                    logger.warning("USDU H3 worker: tile failed: %s", e, exc_info=True)
                session.touch()
    finally:
        session.state = "closed"
        with session.lock:
            session.closed.set()
            pending = []
            while True:
                try:
                    pending.append(session.requests.get_nowait())
                except queue.Empty:
                    break
        with _REG_LOCK:
            SESSIONS.pop(session_id, None)
            _remember(TOMBSTONES, session_id, token_digest)
        for req in pending:  # fail anything still queued so no route waits for it
            _settle(req.future, exc=RuntimeError("session closed"))
    return "served %d tiles (%s)" % (served, reason)


class UltimateSDUpscaleH3TileWorker:
    """Sink node queued on the worker by the 2-GPU node. Not meant to be placed by hand."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL",),
            "session_id": ("STRING", {"default": ""}),
            "token": ("STRING", {"default": ""}),
            "main_signature": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "serve"
    OUTPUT_NODE = True
    CATEGORY = "image/upscaling"
    DESCRIPTION = ("Internal: serves tile requests from the 'MiniMax H3 Tiled Sampling (2 GPU)' node of another ComfyUI "
                   "server for the length of its sampling run. The 2-GPU node queues it on its worker automatically.")

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def serve(self, model, session_id, token, main_signature):
        if base.h3 is None:
            raise RuntimeError("This ComfyUI has no MiniMax H3 support.")
        summary = worker_serve(model, session_id, token, main_signature)
        return {"ui": {"text": [summary]}, "result": ()}


# -- worker: aiohttp routes ----------------------------------------------------------------------------------------

def _header(request, name):
    return request.headers.get(name)


def register_routes(routes):
    """Register the session endpoints on an aiohttp RouteTableDef. They only validate, decode and enqueue."""
    from aiohttp import web

    def reply(status, payload):
        return web.json_response(payload, status=status)

    @routes.get(ROUTE_PREFIX + "/{sid}/status")
    async def status(request):
        return reply(*core_status(request.match_info["sid"], _header(request, TOKEN_HEADER)))

    @routes.post(ROUTE_PREFIX + "/{sid}/heartbeat")
    async def heartbeat(request):
        return reply(*core_heartbeat(request.match_info["sid"], _header(request, TOKEN_HEADER)))

    @routes.post(ROUTE_PREFIX + "/{sid}/done")
    async def done(request):
        return reply(*core_done(request.match_info["sid"], _header(request, TOKEN_HEADER)))

    async def read_authorized(request):
        """The body of an authorized request, or an error response. Nothing is read before session and token check out."""
        _, err = _find(request.match_info["sid"], _header(request, TOKEN_HEADER))
        if err:
            return None, reply(*err)
        length = request.content_length
        if length is not None and length > MAX_BODY_BYTES:
            return None, reply(413, {"error": "body too large"})
        chunks, total = [], 0
        async for chunk in request.content.iter_chunked(1 << 16):  # chunked uploads have no Content-Length: cap as we read
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                return None, reply(413, {"error": "body too large"})
            chunks.append(chunk)
        return b"".join(chunks), None

    @routes.post(ROUTE_PREFIX + "/{sid}/plan")
    async def plan(request):
        body, err = await read_authorized(request)
        if err is not None:
            return err
        return reply(*core_plan(request.match_info["sid"], _header(request, TOKEN_HEADER), body))

    @routes.post(ROUTE_PREFIX + "/{sid}/setup")
    async def setup(request):
        body, err = await read_authorized(request)
        if err is not None:
            return err
        return reply(*core_setup(request.match_info["sid"], _header(request, TOKEN_HEADER), _header(request, KEY_HEADER), body))

    @routes.post(ROUTE_PREFIX + "/{sid}/tile")
    async def tile(request):
        body, err = await read_authorized(request)
        if err is not None:
            return err
        status_code, result = core_tile_submit(request.match_info["sid"], _header(request, TOKEN_HEADER), body)
        if status_code is not None:
            return reply(status_code, result)
        try:
            data = await asyncio.wait_for(asyncio.wrap_future(result.future), timeout=3600)
        except asyncio.CancelledError:
            result.future.cancel()
            raise
        except Exception as e:
            return reply(500, {"error": str(e)})
        return web.Response(body=data, content_type="application/octet-stream")

    return routes


_ROUTES_REGISTERED = False


def _register_on_server():
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        import server
        routes = server.PromptServer.instance.routes
    except Exception:
        return  # no running server (tests, import without ComfyUI's server)
    register_routes(routes)
    _ROUTES_REGISTERED = True


# ---------------------------------------------------------------------------------------------------------------------
# Main side: HTTP client, session with heartbeat, scheduler.
# ---------------------------------------------------------------------------------------------------------------------

class _Http:
    """urllib client that ignores proxy settings (the worker is on the LAN) and never raises on HTTP status codes."""

    def __init__(self):
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @staticmethod
    def _read(resp):
        data = resp.read(MAX_BODY_BYTES + 1)
        if len(data) > MAX_BODY_BYTES:
            raise RemoteError("the worker's response is larger than %d bytes" % MAX_BODY_BYTES)
        return data

    def request(self, method, url, data=None, headers=None, timeout=10.0):
        req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        try:
            with self._opener.open(req, timeout=timeout) as resp:
                return resp.status, self._read(resp)
        except urllib.error.HTTPError as e:
            try:
                return e.code, self._read(e)
            finally:
                e.close()
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise RemoteError("%s %s failed: %s" % (method, url, e))


def _json_body(data):
    try:
        out = json.loads(bytes(data).decode("utf-8"))
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _clip(data, n=300):
    return bytes(data)[:n].decode("utf-8", "replace")


class RemoteSession:
    """One worker session as seen from main: queues the worker prompt, keeps it alive and sends tile requests."""

    def __init__(self, worker_url, signature, closure, source, *, http=None, clock=time.monotonic,
                 heartbeat_s=HEARTBEAT_S, poll_s=1.0):
        self.base_url = worker_url.rstrip("/")
        self.signature = signature
        self.closure = closure
        self.source = source
        self.http = http or _Http()
        self.clock = clock
        self.heartbeat_s = heartbeat_s
        self.poll_s = poll_s
        self.session_id = uuid.uuid4().hex
        self.token = secrets.token_hex(24)
        self.prompt_id = str(uuid.uuid4())
        self.ready = False
        self.dead = False
        self.reason = None
        self.queued = False
        self._plan = None
        self._plan_sent = False
        self._sent = set()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._last_hb = 0.0
        self._last_history = -1e9
        self._failures = 0
        self._lock = threading.Lock()

    # -- plumbing ---------------------------------------------------------------------------------------------
    def _url(self, action):
        return "%s%s/%s/%s" % (self.base_url, ROUTE_PREFIX, self.session_id, action)

    def _call(self, method, action, data=None, headers=None, timeout=10.0):
        h = {TOKEN_HEADER: self.token}
        h.update(headers or {})
        return self.http.request(method, self._url(action), data, h, timeout)

    def fail(self, reason):
        with self._lock:
            if self.dead:
                return
            self.dead = True
            self.reason = reason
        logger.warning("USDU H3 2-GPU: worker disabled for the rest of this run, the remaining tiles run locally (%s)", reason)

    # -- lifecycle --------------------------------------------------------------------------------------------
    def open(self):
        body = json.dumps({
            "prompt": worker_prompt(self.closure, self.source, self.session_id, self.token, self.signature),
            "prompt_id": self.prompt_id,
            "client_id": "usdu-h3-" + self.session_id[:8],
        }).encode("utf-8")
        status, data = self.http.request("POST", self.base_url + "/prompt", body, {"Content-Type": "application/json"}, 15.0)
        if status != 200:
            raise RemoteError("the worker refused the prompt (HTTP %d): %s" % (status, _clip(data)))
        self.queued = True
        self._thread = threading.Thread(target=self._loop, name="usdu-h3-heartbeat", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            self.tick()

    def tick(self):
        """One heartbeat-thread step: poll until the worker is ready, then keep it alive."""
        if self.dead or self._stop.is_set():
            return
        now = self.clock()
        try:
            if not self.ready:
                status, data = self._call("GET", "status", timeout=10.0)
                state = _json_body(data).get("state") if status == 200 else None
                if state == "waiting_plan" and self._plan is not None and not self._plan_sent:
                    pstatus, pdata = self._call("POST", "plan", json.dumps({"tile_shape": self._plan}).encode("utf-8"))
                    if pstatus != 200:
                        raise RemoteError("plan rejected (HTTP %d): %s" % (pstatus, _clip(pdata)))
                    self._plan_sent = True
                elif state == "ready":
                    self.ready = True
                    self._last_hb = now
                    logger.info("USDU H3 2-GPU: worker is ready")
                elif state in ("refused", "closed"):
                    self.fail("worker %s the session: %s" % (state, _json_body(data).get("reason", "")))
                elif state == "pending" and now - self._last_history >= 5.0:
                    self._last_history = now
                    self._check_history()
            elif now - self._last_hb >= self.heartbeat_s:
                self._last_hb = now
                status, data = self._call("POST", "heartbeat", timeout=10.0)
                if status != 200:
                    raise RemoteError("heartbeat answered HTTP %d" % status)
            self._failures = 0
        except RemoteError as e:
            self._failures += 1
            if self._failures >= 3:
                self.fail("worker unreachable: %s" % e)

    def _check_history(self):
        status, data = self.http.request("GET", "%s/history/%s" % (self.base_url, self.prompt_id), timeout=10.0)
        if status != 200:
            return
        entry = _json_body(data).get(self.prompt_id)
        if not isinstance(entry, dict):
            return
        info = entry.get("status") or {}
        if info.get("status_str") == "error":
            self.fail("the worker prompt failed (see the worker's console)")
        else:
            self.fail("the worker prompt ended before it was ready")

    def set_plan(self, tile_shape):
        if self._plan is None:
            self._plan = [int(v) for v in tile_shape]

    def post_tile(self, key, setup_blob, body, timeout):
        """Send one tile (and the setup first if the worker has not seen it); returns the response bytes."""
        if self.dead:
            raise RemoteError("worker disabled: %s" % self.reason)
        for attempt in (0, 1):
            if key not in self._sent:
                status, data = self._call("POST", "setup", setup_blob, {KEY_HEADER: key}, timeout)
                if status != 200:
                    raise RemoteError("setup rejected (HTTP %d): %s" % (status, _clip(data)))
                self._sent.add(key)
            status, data = self._call("POST", "tile", body, {"Content-Type": "application/octet-stream"}, timeout)
            if status == 410 and attempt == 0:
                self._sent.discard(key)  # the worker dropped it from its small cache: send it again
                continue
            if status != 200:
                raise RemoteError("tile rejected (HTTP %d): %s" % (status, _clip(data)))
            return data
        raise RemoteError("setup could not be restored on the worker")

    def close(self, interrupted=False):
        """Stop the heartbeat, tell the worker we are done, drop its prompt if it never started, and (only after an
        interrupt on main) interrupt that one worker prompt. Never a global interrupt or queue clear."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.0)
        steps = [("done", lambda: self._call("POST", "done", timeout=5.0))]
        if self.queued and not self.ready:
            steps.append(("queue delete", lambda: self.http.request(
                "POST", self.base_url + "/queue", json.dumps({"delete": [self.prompt_id]}).encode("utf-8"),
                {"Content-Type": "application/json"}, 5.0)))
        if interrupted and self.queued:
            steps.append(("interrupt", lambda: self.http.request(
                "POST", self.base_url + "/interrupt", json.dumps({"prompt_id": self.prompt_id}).encode("utf-8"),
                {"Content-Type": "application/json"}, 5.0)))
        for name, fn in steps:
            try:
                fn()
            except Exception as e:
                logger.info("USDU H3 2-GPU: closing the worker session (%s) failed: %s", name, e)


# -- scheduling ------------------------------------------------------------------------------------------------------

def assign_tiles(n, ratio):
    """Tile indices that go to the worker for n tiles when the worker is `ratio` times as fast as main.

    Deterministic: the same (n, ratio) always gives the same set. Main always keeps at least one tile, and the worker's
    tiles are spread evenly over the index range.
    """
    if n < 2:
        return []
    k = int(round(n * ratio / (1.0 + ratio)))
    k = max(0, min(k, n - 1))
    return sorted({int((j + 0.5) * n / k) for j in range(k)}) if k else []


class _TileQueue:
    """Thread-safe FIFO of tile indices; pop() returns None when empty."""

    def __init__(self, items):
        self._items = collections.deque(items)
        self._lock = threading.Lock()

    def pop(self):
        with self._lock:
            return self._items.popleft() if self._items else None

    def drain(self):
        with self._lock:
            out = list(self._items)
            self._items.clear()
            return out


def check_interrupted():
    try:
        import comfy.model_management
        comfy.model_management.throw_exception_if_processing_interrupted()
    except ImportError:
        pass


def _sync():
    try:
        import comfy.model_management
        comfy.model_management.synchronize()
    except Exception:
        pass


def _same(a, b):
    """Structural equality that falls back to identity for objects (patch closures, modules, tensors)."""
    if a is b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (str, int, float, bool)) and isinstance(b, (str, int, float, bool)):
        return a == b
    return False


def _wrapper_names(options):
    """DIFFUSION_MODEL wrapper identities in call order, without our own tiling wrappers."""
    fns = comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, options or {})
    return [n for n in (_callable_name(f) for f in fns) if n is not None]


def _callback_names(options):
    out = []
    for call_type, per_key in ((options or {}).get("callbacks") or {}).items():
        for key, fns in per_key.items():
            out.extend((call_type, str(key), _callable_name(f)) for f in fns)
    return out


def remote_ineligible_reason(kwargs, options, base_options, hooks_present=False):
    """Why this model call must run all-local (None when it may use the worker). R2."""
    if kwargs.get("control") is not None:
        return "a ControlNet-style control is attached to this call"
    extra = sorted(set(kwargs) - KNOWN_KWARGS)
    if extra:
        return "unsupported model kwargs: %s" % ", ".join(extra)
    if hooks_present:
        return "conditioning hooks are in use"
    base_options = base_options or {}
    if _wrapper_names(options) != _wrapper_names(base_options):
        return "the DIFFUSION_MODEL wrapper stack differs from the model's"
    if _callback_names(options) != _callback_names(base_options):
        return "the callbacks differ from the model's"
    for key, value in (options or {}).items():
        if key in ("patches", "patches_replace"):
            if not _same(value or {}, base_options.get(key) or {}):
                return "transformer_options['%s'] differs from the model's" % key
        elif key in ("wrappers", "callbacks"):
            continue  # compared by name above
        elif key in ALLOWED_OPTION_KEYS:
            continue
        elif key in base_options and _same(value, base_options[key]):
            continue
        else:
            return "transformer_options has the extra key '%s'" % key
    return None


class RemoteRun:
    """Everything one sampling run needs on main: the session, the speed measurements and the tile scheduler."""

    def __init__(self, worker_url, signature, closure, source, *, session_factory=None):
        self.worker_url = worker_url
        self.signature = signature
        self.closure = closure
        self.source = source
        self._session_factory = session_factory or self._default_session
        self.session = None
        self._reset()

    def _reset(self):
        self.ratio = None
        self.local_times = collections.deque(maxlen=64)
        self.remote_times = collections.deque(maxlen=64)  # worker tiles after its warm-up tile
        self.worker_warm = False
        self.base_options = {}
        self.hooks_present = False
        self._warned = set()
        self.last_assignment = None

    def _default_session(self):
        session = RemoteSession(self.worker_url, self.signature, self.closure, self.source)
        session.open()
        return session

    def _warn_once(self, key, message, *args):
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("USDU H3 2-GPU: " + message, *args)

    # -- run bracket ------------------------------------------------------------------------------------------
    def begin(self, guider=None):
        self._reset()
        options = getattr(guider, "model_options", None) or {}
        self.base_options = options.get("transformer_options") or {}
        conds = getattr(guider, "conds", None) or {}
        self.hooks_present = bool(options.get("registered_hooks")) or any(
            ("hooks" in c) or (c.get("control") is not None) for cs in conds.values() for c in cs)
        try:
            self.session = self._session_factory()
        except Exception as e:  # never fail the run because the worker cannot be reached
            self.session = None
            logger.warning("USDU H3 2-GPU: could not start the worker session, running all tiles locally (%s)", e)

    def finish(self, interrupted=False):
        session, self.session = self.session, None
        if session is not None:
            session.close(interrupted)

    # -- per model call -----------------------------------------------------------------------------------------
    def _timed_local(self, local_fn, call, tile):
        t0 = time.perf_counter()
        out = local_fn(call, tile)
        _sync()
        self.local_times.append(time.perf_counter() - t0)
        return out

    def _all_local(self, local_fn, call, tiles):
        for tile in tiles:
            yield tile, self._timed_local(local_fn, call, tile)

    def _step_meta(self, call):
        to = call.transformer_options or {}
        tensors_cpu = {}
        for key in STEP_TENSOR_KEYS:
            v = to.get(key)
            if torch.is_tensor(v):
                tensors_cpu[key] = v.detach().cpu()
        return tensors_cpu, {"cond_or_uncond": [int(v) for v in to.get("cond_or_uncond", [])],
                             "uuids": [str(v) for v in to.get("uuids", [])]}

    def _setup_blob(self, call, owner):
        tensors = {}
        payload = {k: v for k, v in call.payload.items() if k != "layout"}
        tree = pack_tree(payload, tensors)
        out = {k: v for k, v in tensors.items()}
        out["context"] = call.context
        blob = encode(out, {"payload": tree, "true_positions": bool(owner.true_positions)})
        if len(blob) > MAX_BODY_BYTES:
            raise CodecError("setup is %d bytes, over the worker's upload limit" % len(blob))
        return hashlib.sha256(blob).hexdigest(), blob

    def _tile_blob(self, call, tile, step_tensors, step_meta, setup_key):
        tensors = {"video": call.video[..., tile.y0:tile.ye, tile.x0:tile.xe], "timestep": call.timestep}
        tensors.update(step_tensors)
        if call.is_list and len(call.x) > 1:
            tensors["audio"] = call.x[1]
        if call.denoise_mask is not None:
            tensors["denoise_mask"] = base._crop_hw(call.denoise_mask, tile.y0, tile.ye, tile.x0, tile.xe)
        if call.kwargs.get("audio_denoise_mask") is not None:
            tensors["audio_denoise_mask"] = call.kwargs["audio_denoise_mask"]
        meta = dict(step_meta)
        meta.update({"v": PROTOCOL_VERSION, "setup": setup_key, "tile": {
            "index": tile.index, "y0": tile.y0, "x0": tile.x0, "ye": tile.ye, "xe": tile.xe, "th": call.th, "tw": call.tw,
            "hp": call.hp, "wp": call.wp, "height": call.height, "width": call.width, "t": call.t,
            "text_len": call.text_len, "audio_t": call.audio_t}})
        blob = encode(tensors, meta)
        if len(blob) > MAX_BODY_BYTES:
            raise CodecError("tile request is %d bytes, over the worker's upload limit" % len(blob))
        return blob

    def _parse_result(self, data, call, tile):
        tensors, _ = decode(data)
        video = tensors.get("video")
        want = tuple(call.video.shape[:3]) + (tile.ye - tile.y0, tile.xe - tile.x0)
        if video is None or tuple(video.shape) != want:
            raise CodecError("worker returned a video tile of the wrong shape")
        if not call.is_list:
            return video
        out = [video]
        for i, ref in enumerate(call.x[1:]):
            extra = tensors.get("extra%d" % i)
            if extra is None or tuple(extra.shape) != tuple(ref.shape):
                raise CodecError("worker returned an audio output of the wrong shape")
            out.append(extra)
        return out  # CPU tensors: the remote thread does no device work

    @staticmethod
    def _to_device(out, device):
        return [o.to(device) for o in out] if isinstance(out, list) else out.to(device)

    def tile_timeout(self):
        """Timeout for a worker tile once the worker is warm: a multiple of the slower side's median tile time."""
        medians = [statistics.median(t) for t in (self.local_times, self.remote_times) if t]
        if not medians:
            return FIRST_TILE_TIMEOUT_S
        return max(MIN_TILE_TIMEOUT_S, TILE_TIMEOUT_FACTOR * max(medians))

    def tile_outputs(self, owner, call, tiles, local_fn):
        """Yield (tile, output) for every tile in tile order, some computed on the worker (R5, R6)."""
        session = self.session
        n = len(tiles)
        if session is None or session.dead or n < 2:
            yield from self._all_local(local_fn, call, tiles)
            return
        reason = remote_ineligible_reason(call.kwargs, call.transformer_options, self.base_options, self.hooks_present)
        if reason is None and call.is_list and len(call.x) != 2:
            reason = "more than two latent streams"
        if reason is not None:
            self._warn_once(("eligible", reason), "this model call runs all-local (%s)", reason)
            yield from self._all_local(local_fn, call, tiles)
            return
        session.set_plan((1, call.video.shape[1], call.t, call.th, call.tw))
        if not session.ready:
            yield from self._all_local(local_fn, call, tiles)
            return
        calibrating = self.ratio is None
        if calibrating:
            local_q = remote_q = _TileQueue(range(n))
            remote_ids = list(range(n))
        else:
            remote_ids = assign_tiles(n, self.ratio)
            if not remote_ids:
                yield from self._all_local(local_fn, call, tiles)
                return
            rset = set(remote_ids)
            remote_q = _TileQueue(remote_ids)
            local_q = _TileQueue(i for i in range(n) if i not in rset)
        # CPU copies of every tile the worker may take, built here on the prompt thread (R6)
        try:
            setup_key, setup_blob = self._setup_blob(call, owner)
            step_tensors, step_meta = self._step_meta(call)
            blobs = {i: self._tile_blob(call, tiles[i], step_tensors, step_meta, setup_key) for i in remote_ids}
        except Exception as e:
            session.fail("could not build the tile requests: %s" % e)
            yield from self._all_local(local_fn, call, tiles)
            return

        results = {}
        leftover = []
        batch_local, batch_remote = [], []
        done = threading.Event()
        stop = threading.Event()

        def remote_main():
            current = None
            try:
                while not session.dead and not stop.is_set():
                    current = remote_q.pop()
                    if current is None:
                        break
                    warm = self.worker_warm
                    timeout = self.tile_timeout() if warm else FIRST_TILE_TIMEOUT_S
                    t0 = time.perf_counter()
                    data = session.post_tile(setup_key, setup_blob, blobs[current], timeout)
                    results[current] = self._parse_result(data, call, tiles[current])
                    if warm:
                        dt = time.perf_counter() - t0
                        batch_remote.append(dt)
                        self.remote_times.append(dt)
                    else:  # the warm-up tile (model load on the worker) says nothing about its speed
                        self.worker_warm = True
                    current = None
            except Exception as e:
                session.fail("%s: %s" % (type(e).__name__, e))
                if current is not None:
                    leftover.append(current)
            finally:
                if remote_q is not local_q:  # every exit path (failure, dead flag, stop): hand the rest back to main
                    leftover.extend(remote_q.drain())
                done.set()

        thread = threading.Thread(target=remote_main, name="usdu-h3-remote-tiles", daemon=True)
        thread.start()
        try:
            while True:
                i = local_q.pop()
                if i is None:
                    break
                t0 = time.perf_counter()
                results[i] = local_fn(call, tiles[i])
                _sync()
                dt = time.perf_counter() - t0
                batch_local.append(dt)
                self.local_times.append(dt)
                check_interrupted()
            while not done.wait(0.05):
                check_interrupted()
        finally:
            # a local error, an interrupt or an abandoned generator must not leave the remote thread working
            stop.set()
            thread.join(timeout=5.0)
        for i in sorted(leftover):  # tiles the worker did not deliver
            if i not in results:
                results[i] = self._timed_local(local_fn, call, tiles[i])
        for i in list(results):  # worker results are CPU tensors until here: the device move happens on this thread
            results[i] = self._to_device(results[i], call.video.device)

        if calibrating and batch_local and batch_remote and not session.dead:
            self.ratio = max(0.05, min(20.0, statistics.median(batch_local) / statistics.median(batch_remote)))
            self.last_assignment = assign_tiles(n, self.ratio)
            logger.info("USDU H3 2-GPU: the worker runs at %.2fx the speed of this GPU; from now on %d of %d tiles per "
                        "call go to the worker (tiles %s)", self.ratio, len(self.last_assignment), n, self.last_assignment)
        for tile in tiles:
            yield tile, results.pop(tile.index)


class H3TiledForward2GPU(H3TiledForward):
    """H3TiledForward whose tile loop may hand tiles to the worker; blending is unchanged and stays on main."""

    def __init__(self, tile_width, tile_height, tile_overlap, true_positions=True, run=None):
        super().__init__(tile_width, tile_height, tile_overlap, true_positions)
        self.run = run

    def _tile_outputs(self, call, tiles):
        if self.run is None:
            yield from super()._tile_outputs(call, tiles)
            return
        yield from self.run.tile_outputs(self, call, tiles, self._local_tile)


class RemoteBracket:
    """OUTER_SAMPLE wrapper: opens the worker session before sampling and closes it afterwards (R7)."""

    def __init__(self, run):
        self.run = run

    def __call__(self, executor, *args, **kwargs):
        self.run.begin(getattr(executor, "class_obj", None))
        interrupted = False
        try:
            return executor(*args, **kwargs)
        except BaseException as e:
            try:
                import comfy.model_management
                interrupted = isinstance(e, comfy.model_management.InterruptProcessingException)
            except ImportError:
                pass
            raise
        finally:
            self.run.finish(interrupted)


def check_worker_url(url):
    parsed = urllib.parse.urlparse((url or "").strip())
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("worker_url must look like http://host:port")
    return "%s://%s%s" % (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"))


class UltimateSDUpscaleH3TiledSampling2GPU(UltimateSDUpscaleH3TiledSampling):
    @classmethod
    def INPUT_TYPES(s):
        spec = UltimateSDUpscaleH3TiledSampling.INPUT_TYPES()
        required = dict(spec["required"])
        required["worker_url"] = ("STRING", {"default": "http://127.0.0.1:8189", "multiline": False,
                                             "tooltip": "Address of the second ComfyUI server that runs half of the tiles (same ComfyUI install, custom nodes and models as this one, started on another GPU). Leave empty to run all tiles on this GPU."})
        return {"required": required, "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    DESCRIPTION = ("Same tiling as 'MiniMax H3 Tiled Sampling', but the tiles of every model call are split between this "
                   "GPU and a second ComfyUI server (worker_url). The worker loads the same model from the same node "
                   "subgraph, is kept busy by one queued job for the length of the run, and its share is balanced from the "
                   "measured speed of both GPUs. If the worker cannot be used, every tile runs on this GPU instead.")

    def apply(self, model, tile_width, tile_height, tile_overlap, true_positions, worker_url="", prompt=None, unique_id=None):
        if base.h3 is None:
            raise RuntimeError("This ComfyUI has no MiniMax H3 support.")
        H3TiledForward.check_compatible(model.wrappers, model.model_options.get("transformer_options", {}).get("patches_replace"))
        for fns in model.wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {}).values():
            for fn in fns:
                while isinstance(fn, functools.partial):
                    fn = fn.func
                if isinstance(fn, H3TiledForward):
                    raise RuntimeError("This model already has H3 tiled sampling applied; stacking two tilings is not supported.")
        run = None
        if (worker_url or "").strip():
            try:
                url = check_worker_url(worker_url)
                signature = model_signature(model)  # before the clone adds the tiling wrapper
                closure, source = build_closure(prompt or {}, unique_id)
                run = RemoteRun(url, signature, closure, source)
            except Exception as e:
                logger.warning("USDU H3 2-GPU: not using the worker, all tiles run on this GPU (%s)", e)
        m = model.clone()
        m.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                      H3TiledForward2GPU(tile_width, tile_height, tile_overlap, true_positions, run))
        if run is not None:
            m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "usdu_h3_remote", RemoteBracket(run))
        return (m,)


NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleH3TiledSampling2GPU": UltimateSDUpscaleH3TiledSampling2GPU,
    SINK_CLASS: UltimateSDUpscaleH3TileWorker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleH3TiledSampling2GPU": "Ultimate SD Upscale: MiniMax H3 Tiled Sampling (2 GPU)",
    SINK_CLASS: "Ultimate SD Upscale: MiniMax H3 Tile Worker (internal)",
}

_register_on_server()
