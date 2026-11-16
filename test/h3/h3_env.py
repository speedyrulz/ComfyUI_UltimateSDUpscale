"""CPU-only bootstrap for the H3 tiling tests: real ComfyUI code from /home/jeff/ComfyUI, no GPU, no server."""
import importlib.util
import os
import sys
import types

COMFY_DIR = os.environ.get("COMFY_DIR", "/home/jeff/ComfyUI")
PACK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.argv = ["main.py", "--cpu"]
if COMFY_DIR not in sys.path:
    sys.path.insert(0, COMFY_DIR)

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

if "server" not in sys.modules:  # nodes / folder_paths may want server.PromptServer; not needed here
    stub = types.ModuleType("server")
    stub.PromptServer = type("PromptServer", (), {"instance": None})
    sys.modules["server"] = stub

import torch  # noqa: E402

import comfy.ops  # noqa: E402
import comfy.ldm.minimax.model as h3  # noqa: E402
import comfy.patcher_extension as pe  # noqa: E402


def load_pack():
    spec = importlib.util.spec_from_file_location("usdu_h3_tiled", os.path.join(PACK_DIR, "usdu_h3_tiled.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["usdu_h3_tiled"] = mod
    spec.loader.exec_module(mod)
    return mod


def tiny_model(image=False, seed=0):
    """Smallest MiniMaxH3Model the constructor accepts (head dim 96 is fixed by the 3x16 rope table)."""
    torch.manual_seed(seed)
    kw = dict(hidden_size=96, num_layers=2, token_refiner_num_layers=1, num_attention_heads=1, attention_head_dim=96,
              ffn_hidden_size=64, text_dim=96, timestep_input_dim=16, time_embed_hidden_size=32, time_embed_dim=32,
              dtype=torch.float32, device="cpu", operations=comfy.ops.disable_weight_init)
    if image:
        model = h3.MiniMaxH3ImageModel(pdd_sigmas=[1.0, 0.0], **kw)
    else:
        model = h3.MiniMaxH3Model(**kw)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim > 1:
                torch.nn.init.normal_(p, std=0.2)
            elif name.endswith("weight"):
                torch.nn.init.ones_(p)  # RMSNorm scales: zero would silence q/k and with them the rope
            else:
                torch.nn.init.normal_(p, std=0.02)
        model.rope.inv_freq.copy_(torch.logspace(0, -2, 16))
    model.requires_grad_(False)  # the in-place rope kernel refuses tensors that require grad
    return model


def wrapper_options(*wrappers):
    return {"wrappers": {pe.WrappersMP.DIFFUSION_MODEL: {None: list(wrappers)}}}
