"""CPU tests for the 2-GPU H3 tile dispatch (usdu_h3_remote.py): real ComfyUI code, tiny random H3 model, no GPU.

The worker is run in-process: its routes on a real aiohttp server in a thread, its sink loop on a thread, or (for the
scheduler tests) a fake session that goes through the same validation and tile code without HTTP.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import h3_env  # noqa: E402,F401  (must come first: sets up the CPU-only ComfyUI import)
import h3_tiled_tests  # noqa: E402  (loads the pack once; reuse its payload builder)
from h3_env import PACK_DIR, h3, tiny_model, wrapper_options  # noqa: E402

import asyncio  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import unittest.mock  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

import torch  # noqa: E402

import comfy.model_base as mb  # noqa: E402
import comfy.model_patcher as mp  # noqa: E402
import comfy.patcher_extension as pe  # noqa: E402
import comfy.supported_models as sm  # noqa: E402

pack = sys.modules["usdu_h3_tiled"]
make_payload = h3_tiled_tests.make_payload

_spec = importlib.util.spec_from_file_location("usdu_h3_remote", os.path.join(PACK_DIR, "usdu_h3_remote.py"))
remote = importlib.util.module_from_spec(_spec)
sys.modules["usdu_h3_remote"] = remote
_spec.loader.exec_module(remote)

TILE_SHAPE = (1, 24, 3, 16, 16)


def types_ns(**kw):
    return types.SimpleNamespace(**kw)


# -- fixtures ---------------------------------------------------------------------------------------------------------

def make_patcher(seed=0):
    cfg = sm.MiniMaxH3({"image_model": "minimax_h3", "disable_unet_model_creation": True})
    cfg.set_inference_dtype(torch.float32, None)
    bm = mb.MiniMaxH3(cfg, device=torch.device("cpu"))
    bm.diffusion_model = tiny_model(seed=seed)
    return mp.ModelPatcher(bm, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def make_inputs(h=20, w=30, seed=3):
    g = torch.Generator().manual_seed(seed)
    x = [torch.randn(1, 24, 3, h, w, generator=g), torch.randn(1, 32, 2, 6, generator=g)]
    return x, torch.randn(1, 5, 96, generator=g), torch.tensor([500.0]), make_payload(h, w, t=3, audio_t=6, text_len=5)


def local_reference(dm, x, ctx, ts, payload, wrapper_cls=None, **kwargs):
    fwd = (wrapper_cls or pack.H3TiledForward)(256, 256, 64)
    with torch.inference_mode():
        return dm(x, ts, ctx, transformer_options=wrapper_options(fwd), minimax_payload=payload, **kwargs)


class InProcWorker:
    """The worker without HTTP: a registered ready session plus the real sampler lifecycle on a tiny patcher.

    It has the same interface as RemoteSession for the scheduler (ready, dead, set_plan, post_tile, fail, close)."""

    def __init__(self, delay=0.0, fail_on=None, patcher=None):
        self.patcher = patcher or make_patcher()
        self.session = remote.WorkerSession("s" + os.urandom(6).hex(), "tok" + os.urandom(6).hex())
        self.session.state = "ready"
        remote.SESSIONS[self.session.id] = self.session
        self.cm = remote.worker_lifecycle(self.patcher, TILE_SHAPE)
        self.ctx = self.cm.__enter__()
        self.delay, self.fail_on = delay, fail_on
        self.ready, self.dead, self.reason = True, False, None
        self.tiles = []
        self.timeouts = []
        self.plan = None
        self.closed = None

    def set_plan(self, shape):
        self.plan = tuple(shape)

    def fail(self, reason):
        self.dead, self.reason = True, reason

    def post_tile(self, key, setup_blob, body, timeout):
        self.timeouts.append(timeout)
        if self.dead:
            raise remote.RemoteError("dead")
        if self.fail_on is not None and len(self.tiles) >= self.fail_on:
            raise remote.RemoteError("boom")
        status, payload = remote.core_setup(self.session.id, self.session.token, key, setup_blob)
        assert status == 200, payload
        status, req = remote.core_tile_submit(self.session.id, self.session.token, body)
        assert status is None, req
        req = self.session.requests.get_nowait()
        time.sleep(self.delay)
        self.tiles.append(req.meta["tile"]["index"])
        return remote.run_worker_tile(self.ctx, req)

    def close(self, interrupted=False):
        self.closed = interrupted

    def shutdown(self):
        self.cm.__exit__(None, None, None)
        remote.SESSIONS.pop(self.session.id, None)


def make_run(session):
    return remote.RemoteRun("http://127.0.0.1:1", "sig", {}, ("1", 0), session_factory=lambda: session)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeHttp:
    """Scripted HTTP for RemoteSession: answers by URL suffix, records every request."""

    def __init__(self):
        self.log = []
        self.state = "pending"
        self.fail = False
        self.history = {}

    def request(self, method, url, data=None, headers=None, timeout=10.0):
        self.log.append((method, url.split("127.0.0.1:9", 1)[-1], data))
        if self.fail:
            raise remote.RemoteError("down")
        if url.endswith("/status"):
            return 200, json.dumps({"state": self.state}).encode()
        if "/history/" in url:
            return 200, json.dumps(self.history).encode()
        return 200, b"{}"

    def paths(self):
        return [(m, p.rsplit("/", 1)[-1]) for m, p, _ in self.log]


def make_session(http, clock=None, **kw):
    return remote.RemoteSession("http://127.0.0.1:9", "sig", {}, ("1", 0), http=http, clock=clock or time.monotonic, **kw)


# -- codec ------------------------------------------------------------------------------------------------------------

class A_Codec(unittest.TestCase):
    def test_roundtrip_dtypes(self):
        t = {"bf": torch.randn(2, 3).to(torch.bfloat16), "f32": torch.randn(4), "i64": torch.arange(5, dtype=torch.long) * (1 << 40),
             "f16": torch.randn(3).to(torch.float16), "b": torch.tensor([True, False]), "empty": torch.zeros(0, 3), "s": torch.tensor(2.5)}
        out, meta = remote.decode(remote.encode(t, {"a": [1, 2], "b": "x"}))
        self.assertEqual(meta, {"a": [1, 2], "b": "x"})
        for k, v in t.items():
            self.assertEqual(out[k].dtype, v.dtype, k)
            self.assertTrue(torch.equal(out[k], v), k)

    def test_noncontiguous_and_grad_tensors(self):
        base_t = torch.randn(4, 6)
        out, _ = remote.decode(remote.encode({"v": base_t.t()}))
        self.assertTrue(torch.equal(out["v"], base_t.t()))

    def test_tree_roundtrip_with_shared_tensor(self):
        shared = torch.randn(2, 2)
        obj = {"keyframes": [{"latent": shared, "resolved_frame_index": 0}, {"audio_latent": None, "n": 1.5}],
               "cond": [shared, torch.ones(1)], "seed": 3, "flag": True, "name": "x", "tup": (1, 2)}
        tensors = {}
        tree = remote.pack_tree(obj, tensors)
        self.assertEqual(len(tensors), 2)  # the shared tensor is sent once
        back = remote.unpack_tree(json.loads(json.dumps(tree)), tensors)
        self.assertIs(back["keyframes"][0]["latent"], back["cond"][0])
        self.assertEqual(back["tup"], [1, 2])
        self.assertEqual((back["seed"], back["flag"], back["name"], back["keyframes"][1]["n"]), (3, True, "x", 1.5))

    def test_tree_depth_and_size_limits(self):
        node = 1
        for _ in range(remote.MAX_TREE_DEPTH + 5):
            node = {"l": [node]}
        with self.assertRaises(remote.CodecError):
            remote.unpack_tree(node, {})
        ok = 1
        for _ in range(remote.MAX_TREE_DEPTH - 2):
            ok = {"l": [ok]}
        self.assertIsNotNone(remote.unpack_tree(ok, {}))
        wide = {"l": [0] * (remote.MAX_TREE_NODES + 5)}
        with self.assertRaises(remote.CodecError):
            remote.unpack_tree(wide, {})

    def test_rejects_bad_input(self):
        good = remote.encode({"a": torch.ones(3)})
        for bad in (b"", b"x" * 20, good[:20], b"\xff" * 8 + good[8:], (len(good) + 1).to_bytes(8, "little") + good[8:]):
            with self.assertRaises(remote.CodecError):
                remote.decode(bad)
        with self.assertRaises(remote.CodecError):
            remote.decode(good, max_bytes=10)
        with self.assertRaises(remote.CodecError):
            remote.pack_tree({"x": object()}, {})
        with self.assertRaises(remote.CodecError):
            remote.pack_tree({1: torch.ones(1)}, {})
        with self.assertRaises(remote.CodecError):
            remote.unpack_tree({"t": "missing"}, {})
        with self.assertRaises(remote.CodecError):
            remote.unpack_tree({"zzz": 1}, {})

    def test_no_pickle_surface(self):
        src = open(os.path.join(PACK_DIR, "usdu_h3_remote.py"), encoding="utf-8").read()
        for word in ("import pickle", "torch.load(", "import marshal", "pickle.loads"):
            self.assertFalse(word in src, word)


# -- worker tile == local tile ------------------------------------------------------------------------------------------

class B_WorkerTile(unittest.TestCase):
    def setUp(self):
        remote.reset_worker_state()
        self.worker = InProcWorker()

    def tearDown(self):
        self.worker.shutdown()

    def test_worker_tile_equals_local_tile(self):
        dm = tiny_model()
        x, ctx, ts, payload = make_inputs()
        captured = {}

        class Spy(remote.H3TiledForward2GPU):
            def _tile_outputs(self, call, tiles):
                captured["call"], captured["tiles"] = call, tiles
                yield from super()._tile_outputs(call, tiles)

        spy = Spy(256, 256, 64, True, None)
        with torch.inference_mode():
            dm(x, ts, ctx, transformer_options=wrapper_options(spy), minimax_payload=payload)
        call, tiles = captured["call"], captured["tiles"]
        run = make_run(self.worker)
        key, setup_blob = run._setup_blob(call, spy)
        step_tensors, step_meta = run._step_meta(call)
        self.assertGreater(len(tiles), 3)
        for tile in tiles:
            body = run._tile_blob(call, tile, step_tensors, step_meta, key)
            data = self.worker.post_tile(key, setup_blob, body, 10)
            got = run._parse_result(data, call, tile)
            with torch.inference_mode():
                want = spy._local_tile(call, tile)
            self.assertTrue(torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), tile)

    def test_full_call_matches_all_local_incl_masks_and_sigmas(self):
        dm = tiny_model()
        x, ctx, ts, payload = make_inputs(h=22, w=34)
        mask = torch.ones(1, 1, 3, 22, 34)
        mask[..., :8, :] = 0.3
        amask = torch.ones(1, 1, 2, 6)
        ref = local_reference(dm, x, ctx, ts, payload, denoise_mask=mask, audio_denoise_mask=amask)
        run = make_run(self.worker)
        run.begin()
        run.ratio = 1.0  # fixed assignment: half of the tiles on the worker
        fwd = remote.H3TiledForward2GPU(256, 256, 64, True, run)
        with torch.inference_mode():
            out = dm(x, ts, ctx, transformer_options=wrapper_options(fwd), minimax_payload=payload,
                     denoise_mask=mask, audio_denoise_mask=amask)
        self.assertGreater(len(self.worker.tiles), 0)
        self.assertTrue(torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]))

    def test_validation_rejects_bad_requests(self):
        dm = tiny_model()
        x, ctx, ts, payload = make_inputs()
        captured = {}

        class Spy(remote.H3TiledForward2GPU):
            def _tile_outputs(self, call, tiles):
                captured["call"], captured["tiles"] = call, tiles
                yield from super()._tile_outputs(call, tiles)

        spy = Spy(256, 256, 64, True, None)
        with torch.inference_mode():
            dm(x, ts, ctx, transformer_options=wrapper_options(spy), minimax_payload=payload)
        call, tiles = captured["call"], captured["tiles"]
        run = make_run(self.worker)
        key, setup_blob = run._setup_blob(call, spy)
        sid, tok = self.worker.session.id, self.worker.session.token
        self.assertEqual(remote.core_setup(sid, tok, "0" * 64, setup_blob)[0], 400)  # key must match the content
        self.assertEqual(remote.core_setup(sid, tok, key, b"junk")[0], 400)
        self.assertEqual(remote.core_setup(sid, tok, key, setup_blob)[0], 200)
        step_tensors, step_meta = run._step_meta(call)
        good_t = {"video": call.video[..., tiles[0].y0:tiles[0].ye, tiles[0].x0:tiles[0].xe], "timestep": ts}

        def meta_for(tile, **over):
            m = {"setup": key, "tile": {"index": 0, "y0": tile.y0, "x0": tile.x0, "ye": tile.ye, "xe": tile.xe, "th": call.th,
                                       "tw": call.tw, "hp": call.hp, "wp": call.wp, "height": call.height, "width": call.width,
                                       "t": call.t, "text_len": call.text_len, "audio_t": 0}}
            m["tile"].update(over)
            return m

        def submit(tensors, meta):
            return remote.core_tile_submit(sid, tok, remote.encode(tensors, meta))[0]

        tile = tiles[0]
        self.assertEqual(submit(good_t, meta_for(tile, ye=tile.ye + 1)), 400)  # window vs tensor shape
        self.assertEqual(submit(good_t, meta_for(tile, text_len=99)), 400)  # context length
        self.assertEqual(submit(good_t, meta_for(tile, th=7)), 400)  # odd tile
        self.assertEqual(submit(good_t, meta_for(tile, audio_t=6)), 400)  # audio declared, none sent
        self.assertEqual(submit(dict(good_t, extra=torch.ones(1)), meta_for(tile)), 400)  # unknown tensor
        self.assertEqual(submit({"timestep": ts}, meta_for(tile)), 400)  # no video
        bad_meta = meta_for(tile)
        bad_meta["setup"] = "f" * 64
        self.assertEqual(submit(good_t, bad_meta), 410)  # unknown setup
        self.assertEqual(submit(good_t, meta_for(tile)), None)  # a valid one is accepted


# -- scheduler ---------------------------------------------------------------------------------------------------------

class C_Scheduler(unittest.TestCase):
    def setUp(self):
        remote.reset_worker_state()
        self.dm = tiny_model()
        self.x, self.ctx, self.ts, self.payload = make_inputs(h=22, w=34)
        self.ref = local_reference(self.dm, self.x, self.ctx, self.ts, self.payload)
        self.workers = []

    def tearDown(self):
        for w in self.workers:
            w.shutdown()

    def worker(self, **kw):
        w = InProcWorker(**kw)
        self.workers.append(w)
        return w

    def call(self, run):
        fwd = remote.H3TiledForward2GPU(256, 256, 64, True, run)
        with torch.inference_mode():
            return self.dm(self.x, self.ts, self.ctx, transformer_options=wrapper_options(fwd), minimax_payload=self.payload)

    def assertSame(self, out):
        self.assertTrue(torch.equal(out[0], self.ref[0]) and torch.equal(out[1], self.ref[1]))

    def test_assign_tiles_is_deterministic_and_bounded(self):
        for n in range(0, 20):
            for ratio in (0.05, 0.3, 1.0, 2.5, 20.0):
                a = remote.assign_tiles(n, ratio)
                self.assertEqual(a, remote.assign_tiles(n, ratio))
                self.assertEqual(a, sorted(set(a)))
                self.assertTrue(all(0 <= i < n for i in a))
                self.assertLessEqual(len(a), max(0, n - 1))  # main always keeps a tile
        self.assertEqual(len(remote.assign_tiles(10, 1.0)), 5)
        self.assertEqual(len(remote.assign_tiles(10, 3.0)), 8)
        self.assertEqual(remote.assign_tiles(10, 0.05), [])
        self.assertEqual(remote.assign_tiles(1, 5.0), [])
        self.assertLessEqual(len(remote.assign_tiles(9, 1.0)), len(remote.assign_tiles(9, 2.0)))

    def test_calibration_then_fixed_assignment(self):
        w = self.worker(delay=0.01)
        run = make_run(w)
        run.begin()
        self.assertIsNone(run.ratio)
        self.assertSame(self.call(run))  # calibration call: tiles pulled by both sides
        self.assertIsNotNone(run.ratio)
        self.assertGreater(len(w.tiles), 0)
        w.tiles.clear()
        for _ in range(3):
            self.assertSame(self.call(run))
            self.assertEqual(sorted(w.tiles), run.last_assignment)  # identical assignment every call
            w.tiles.clear()
        self.assertEqual(w.plan[0], 1)

    def test_worker_joining_late_gets_the_warm_up_timeout(self):
        # regression (first real run): the worker became ready after main had timed local tiles, so its first tile,
        # which loads the model onto its GPU, got the short steady-state timeout and the worker was dropped
        w = self.worker(delay=0.01)
        w.ready = False
        run = make_run(w)
        run.begin()
        self.assertSame(self.call(run))  # not ready: all local, local tile times recorded
        self.assertTrue(run.local_times)
        self.assertEqual(w.timeouts, [])
        w.ready = True
        self.assertSame(self.call(run))  # the worker's first tile: warm-up timeout, and no calibration from it
        self.assertEqual(w.timeouts[0], remote.FIRST_TILE_TIMEOUT_S)
        self.assertTrue(run.worker_warm)
        if len(w.tiles) == 1:
            self.assertIsNone(run.ratio)  # only the warm-up tile came back: calibrate on the next call
            self.assertSame(self.call(run))
        self.assertIsNotNone(run.ratio)
        self.assertEqual(w.timeouts.count(remote.FIRST_TILE_TIMEOUT_S), 1)
        self.assertEqual(len(run.remote_times), len(w.tiles) - 1)  # the warm-up tile is never timed
        n = len(w.timeouts)
        self.assertSame(self.call(run))
        self.assertTrue(all(t == remote.MIN_TILE_TIMEOUT_S for t in w.timeouts[n:]))  # tiny tiles: the floor

    def test_tile_timeout_uses_the_slower_side(self):
        run = make_run(self.worker())
        self.assertEqual(run.tile_timeout(), remote.FIRST_TILE_TIMEOUT_S)
        run.local_times.extend([2.0, 2.0, 2.0])
        self.assertEqual(run.tile_timeout(), remote.MIN_TILE_TIMEOUT_S)
        run.local_times.extend([10.0] * 5)
        self.assertEqual(run.tile_timeout(), 50.0)
        run.remote_times.extend([20.0] * 3)  # a slower worker raises the timeout
        self.assertEqual(run.tile_timeout(), 100.0)

    def test_same_ratio_same_assignment_across_runs(self):
        seen = []
        for _ in range(2):
            w = self.worker()
            run = make_run(w)
            run.begin()
            run.ratio = 1.7
            self.assertSame(self.call(run))
            seen.append(sorted(w.tiles))
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(seen[0], remote.assign_tiles(len(self.call_tiles()), 1.7))

    def call_tiles(self):
        _, ys, _, xs = pack.H3TiledForward(256, 256, 64).grid(22, 34)
        return [0] * (len(ys) * len(xs))

    def test_remote_failure_runs_remaining_tiles_locally(self):
        w = self.worker(fail_on=1)
        run = make_run(w)
        run.begin()
        run.ratio = 1.0
        self.assertSame(self.call(run))
        self.assertTrue(w.dead)
        self.assertEqual(len(w.tiles), 1)
        self.assertSame(self.call(run))  # later calls are all-local, the dead worker is not asked again
        self.assertEqual(len(w.tiles), 1)

    def test_failure_during_calibration(self):
        w = self.worker(fail_on=0)
        run = make_run(w)
        run.begin()
        self.assertSame(self.call(run))
        self.assertTrue(w.dead)
        self.assertIsNone(run.ratio)

    def test_not_ready_runs_local_and_sends_plan(self):
        w = self.worker()
        w.ready = False
        run = make_run(w)
        run.begin()
        self.assertSame(self.call(run))
        self.assertEqual(w.tiles, [])
        self.assertEqual(w.plan, (1, 24, 3, 16, 16))

    def test_single_tile_frames_never_use_the_worker(self):
        w = self.worker()
        run = make_run(w)
        run.begin()
        x = [torch.randn(1, 24, 3, 8, 8), torch.randn(1, 32, 2, 6)]
        fwd = remote.H3TiledForward2GPU(2048, 2048, 64, True, run)
        with torch.inference_mode():
            self.dm(x, self.ts, self.ctx, transformer_options=wrapper_options(fwd), minimax_payload=make_payload(8, 8))
        self.assertEqual(w.tiles, [])

    def test_dead_flag_flipping_between_tiles_hands_the_rest_back(self):
        class Flipper(InProcWorker):
            def post_tile(self, *a):
                out = super().post_tile(*a)
                self.dead = True  # the heartbeat thread noticed the worker is gone
                return out

        w = Flipper()
        self.workers.append(w)
        run = make_run(w)
        run.begin()
        run.ratio = 4.0  # several remote tiles in a fixed assignment
        self.assertGreater(len(remote.assign_tiles(len(self.call_tiles()), 4.0)), 1)
        self.assertSame(self.call(run))  # used to end in a KeyError for the undelivered tiles
        self.assertEqual(len(w.tiles), 1)

    def test_local_error_stops_the_remote_thread(self):
        w = self.worker(delay=0.3)
        run = make_run(w)
        run.begin()
        run.ratio = 1.0

        class Boom(remote.H3TiledForward2GPU):
            def _local_tile(self, call, tile):
                raise ValueError("local failure")

        fwd = Boom(256, 256, 64, True, run)
        with self.assertRaises(ValueError):
            with torch.inference_mode():
                self.dm(self.x, self.ts, self.ctx, transformer_options=wrapper_options(fwd), minimax_payload=self.payload)
        self.assertFalse([t for t in threading.enumerate() if t.name == "usdu-h3-remote-tiles"])
        done = len(w.tiles)
        time.sleep(0.8)
        self.assertEqual(len(w.tiles), done)  # nothing is sent after the failure
        self.assertLessEqual(done, 1)

    def test_remote_results_stay_on_cpu_until_collected(self):
        w = self.worker()
        run = make_run(w)
        run.begin()
        run.ratio = 1.0
        seen = []
        orig = run._to_device
        run._to_device = lambda out, device: (seen.append(threading.current_thread().name), orig(out, device))[1]
        self.assertSame(self.call(run))
        self.assertTrue(seen)
        self.assertEqual(set(seen), {threading.current_thread().name})  # device moves only on the prompt thread

    def test_blending_order_is_by_tile_index(self):
        # a slow worker finishes after main: the blend must not depend on arrival order
        w = self.worker(delay=0.05)
        run = make_run(w)
        run.begin()
        run.ratio = 2.0
        self.assertSame(self.call(run))


class D_Eligibility(unittest.TestCase):
    def test_reasons(self):
        r = remote.remote_ineligible_reason
        base_opts = {"patches": {"a": [1]}, "minimax_h3_sigma_shift_video": 3.0}
        ok_opts = {"sigmas": 1, "sample_sigmas": 2, "cond_or_uncond": [0], "uuids": [], "wrappers": {}, "callbacks": {},
                   "minimax_h3_layout": object(), "block_index": 1, "prefetch_dynamic_vbars": False,
                   "patches": {"a": [1]}, "minimax_h3_sigma_shift_video": 3.0}
        known = {"minimax_payload": {}, "denoise_mask": None, "audio_denoise_mask": None, "control": None}
        self.assertIsNone(r(known, ok_opts, base_opts))
        self.assertIn("control", r(dict(known, control=object()), ok_opts, base_opts))
        self.assertIn("kwargs", r(dict(known, guidance=torch.ones(1)), ok_opts, base_opts))
        self.assertIn("hooks", r(known, ok_opts, base_opts, hooks_present=True))
        self.assertIn("extra key", r(known, dict(ok_opts, funny=1), base_opts))
        self.assertIn("patches", r(known, dict(ok_opts, patches={"a": [1], "b": [2]}), base_opts))
        self.assertIn("patches_replace", r(known, dict(ok_opts, patches_replace={"dit": {}} ), base_opts))
        self.assertIn("extra key", r(known, dict(ok_opts, minimax_h3_sigma_shift_video=4.0), base_opts))
        self.assertIsNone(r(known, dict(ok_opts, patches_replace={}), base_opts))

    def test_wrapper_and_callback_stacks_must_match_the_base(self):
        r = remote.remote_ineligible_reason
        known = {"minimax_payload": {}, "denoise_mask": None, "audio_denoise_mask": None, "control": None}

        def w1(executor, *a, **k):
            return executor(*a, **k)

        def w2(executor, *a, **k):
            return executor(*a, **k)

        def cb(*a):
            return None

        def opts(*fns, callbacks=None):
            o = wrapper_options(*fns)
            o["callbacks"] = callbacks or {}
            return o

        own = remote.H3TiledForward2GPU(256, 256, 64)
        base_o = opts(w1, w2)
        self.assertIsNone(r(known, opts(w1, own, w2), base_o))  # our own tiling wrapper is ignored
        self.assertIn("wrapper", r(known, opts(w1, w2, w1), base_o))
        self.assertIn("wrapper", r(known, opts(w2, w1), base_o))  # order counts
        self.assertIn("wrapper", r(known, opts(w1), base_o))
        self.assertIsNone(r(known, opts(w1, w2, callbacks={"on_pre_run": {None: [cb]}}),
                            opts(w1, w2, callbacks={"on_pre_run": {None: [cb]}})))
        self.assertIn("callbacks", r(known, opts(w1, w2, callbacks={"on_pre_run": {None: [cb]}}), base_o))
        self.assertIn("callbacks", r(known, base_o, opts(w1, w2, callbacks={"on_pre_run": {None: [cb]}})))

    def test_sigma_shifts_are_compared_not_whitelisted(self):
        r = remote.remote_ineligible_reason
        known = {"minimax_payload": {}, "denoise_mask": None, "audio_denoise_mask": None, "control": None}
        base_o = {"minimax_h3_sigma_shift_video": 3.0, "minimax_h3_sigma_shift_audio": 1.5}
        self.assertIsNone(r(known, dict(base_o), base_o))
        self.assertIn("extra key", r(known, dict(base_o, minimax_h3_sigma_shift_audio=2.0), base_o))
        self.assertIn("extra key", r(known, {"minimax_h3_sigma_shift_video": 3.0}, {}))  # shift the base does not have
        # and the worker reproduces them: a different shift in the model options changes the signature
        a, b = make_patcher(), make_patcher()
        b.model_options.setdefault("transformer_options", {})["minimax_h3_sigma_shift_video"] = 2.0
        self.assertNotEqual(remote.model_signature(a), remote.model_signature(b))

    def test_ineligible_call_runs_local_and_never_asks_the_worker(self):
        remote.reset_worker_state()
        w = InProcWorker()
        try:
            dm = tiny_model()
            x, ctx, ts, payload = make_inputs()
            ref = local_reference(dm, x, ctx, ts, payload)
            run = make_run(w)
            run.begin()
            run.ratio = 1.0
            fwd = remote.H3TiledForward2GPU(256, 256, 64, True, run)
            opts = wrapper_options(fwd)
            opts["funny_option"] = 1
            with torch.inference_mode():
                out = dm(x, ts, ctx, transformer_options=opts, minimax_payload=payload)
            self.assertEqual(w.tiles, [])
            self.assertTrue(torch.equal(out[0], ref[0]))
            run.hooks_present = True
            with torch.inference_mode():
                dm(x, ts, ctx, transformer_options=wrapper_options(fwd), minimax_payload=payload)
            self.assertEqual(w.tiles, [])
        finally:
            w.shutdown()


# -- closure -----------------------------------------------------------------------------------------------------------

def _cls(name, inputs, returns, output=False):
    return type(name, (), {"INPUT_TYPES": classmethod(lambda c: {"required": inputs}), "RETURN_TYPES": tuple(returns),
                           "OUTPUT_NODE": output})


FAKE_CLASSES = {
    "Loader": _cls("Loader", {"ckpt": (["a.safetensors"],)}, ["MODEL", "CLIP", "VAE"]),
    "Lora": _cls("Lora", {"model": ("MODEL",), "name": ("STRING",), "strength": ("FLOAT",)}, ["MODEL"]),
    "BadLatent": _cls("BadLatent", {"model": ("MODEL",), "latent": ("LATENT",)}, ["MODEL"]),
    "SaveModel": _cls("SaveModel", {"model": ("MODEL",)}, ["MODEL"], output=True),
    "Empty": _cls("Empty", {}, ["LATENT"]),
    "Ours": _cls("Ours", {"model": ("MODEL",)}, ["MODEL"]),
    "Pass": _cls("Pass", {"value": ("*",)}, ["*"]),
    "PassWidget": _cls("PassWidget", {"value": ("*",), "n": ("INT",)}, ["*"]),
}


class E_Closure(unittest.TestCase):
    def test_walks_model_links_and_copies_widgets(self):
        prompt = {"1": {"class_type": "Loader", "inputs": {"ckpt": "a.safetensors"}},
                  "2": {"class_type": "Lora", "inputs": {"model": ["1", 0], "name": "x", "strength": 0.5}},
                  "3": {"class_type": "Lora", "inputs": {"model": ["2", 0], "name": "y", "strength": 1.0}},
                  "9": {"class_type": "Empty", "inputs": {}},
                  "5": {"class_type": "Ours", "inputs": {"model": ["3", 0]}}}
        closure, source = remote.build_closure(prompt, "5", classes=FAKE_CLASSES)
        self.assertEqual(sorted(closure), ["1", "2", "3"])
        self.assertEqual(source, ("3", 0))
        self.assertEqual(closure["2"]["inputs"], {"model": ["1", 0], "name": "x", "strength": 0.5})
        closure["2"]["inputs"]["name"] = "changed"
        self.assertEqual(prompt["2"]["inputs"]["name"], "x")  # a copy
        wp = remote.worker_prompt(closure, source, "sid", "tok", "sig")
        self.assertEqual(wp[remote.SINK_NODE_ID]["inputs"]["model"], ["3", 0])
        self.assertEqual(wp[remote.SINK_NODE_ID]["class_type"], remote.SINK_CLASS)
        self.assertNotIn("5", wp)

    def test_any_type_passthrough_nodes(self):
        loader = {"class_type": "Loader", "inputs": {"ckpt": "a"}}
        ours = {"class_type": "Ours", "inputs": {"model": ["3", 0]}}
        passthrough = {"1": loader, "2": {"class_type": "Pass", "inputs": {"value": ["1", 0]}},
                       "3": {"class_type": "Pass", "inputs": {"value": ["2", 0]}}, "5": ours}
        closure, source = remote.build_closure(passthrough, "5", classes=FAKE_CLASSES)
        self.assertEqual(sorted(closure), ["1", "2", "3"])
        self.assertEqual(source, ("3", 0))
        # a "*" chain that ends in something that is not a MODEL is still refused
        self.refuse({"1": {"class_type": "Empty", "inputs": {}}, "2": {"class_type": "Pass", "inputs": {"value": ["1", 0]}},
                     "3": {"class_type": "Pass", "inputs": {"value": ["2", 0]}}, "5": ours}, needle="not a MODEL")
        self.refuse({"1": {"class_type": "Loader", "inputs": {"ckpt": "a"}},
                     "3": {"class_type": "Pass", "inputs": {"value": ["1", 1]}}, "5": ours}, needle="not a MODEL")  # CLIP out
        # an any-type output with nothing linked has no type to resolve
        self.refuse({"3": {"class_type": "PassWidget", "inputs": {"value": 4, "n": 1}}, "5": ours}, needle="any-type")
        # a "*" node that also takes a non-MODEL link
        self.refuse({"1": loader, "9": {"class_type": "Empty", "inputs": {}},
                     "3": {"class_type": "PassWidget", "inputs": {"value": ["1", 0], "n": ["9", 0]}}, "5": ours}, needle="INT")

    def refuse(self, prompt, node="5", needle=""):
        with self.assertRaises(remote.ClosureError) as cm:
            remote.build_closure(prompt, node, classes=FAKE_CLASSES)
        self.assertIn(needle, str(cm.exception))

    def test_refusals(self):
        loader = {"class_type": "Loader", "inputs": {"ckpt": "a"}}
        ours = {"class_type": "Ours", "inputs": {"model": ["2", 0]}}
        self.refuse({"1": loader, "2": {"class_type": "BadLatent", "inputs": {"model": ["1", 0], "latent": ["9", 0]}},
                     "9": {"class_type": "Empty", "inputs": {}}, "5": ours}, needle="LATENT")
        self.refuse({"1": loader, "2": {"class_type": "SaveModel", "inputs": {"model": ["1", 0]}}, "5": ours}, needle="output node")
        self.refuse({"2": {"class_type": "Lora", "inputs": {"model": ["1", 0], "name": "x", "strength": 1}}, "5": ours}, needle="missing")
        self.refuse({"1": loader, "2": {"class_type": "Lora", "inputs": {"model": ["1", 1], "name": "x", "strength": 1}}, "5": ours},
                    needle="not a MODEL")  # CLIP output wired in
        self.refuse({"1": loader, "5": {"class_type": "Ours", "inputs": {"model": "text"}}}, needle="not connected")
        self.refuse({"1": loader}, node="7", needle="not in the prompt")
        self.refuse({"1": {"class_type": "Nope", "inputs": {}}, "2": {"class_type": "Ours", "inputs": {"model": ["1", 0]}}}, node="2", needle="unknown class")


# -- heartbeat, idle, tokens, stopping -------------------------------------------------------------------------------

class F_Heartbeat(unittest.TestCase):
    def test_main_side_state_machine(self):
        clock, http = FakeClock(), FakeHttp()
        s = make_session(http, clock, heartbeat_s=10.0)
        s.queued = True
        s.tick()
        self.assertFalse(s.ready)
        http.state = "waiting_plan"
        s.tick()
        self.assertNotIn(("POST", "plan"), http.paths())  # no plan yet
        s.set_plan((1, 24, 3, 16, 16))
        s.tick()
        self.assertEqual(http.paths().count(("POST", "plan")), 1)
        s.tick()
        self.assertEqual(http.paths().count(("POST", "plan")), 1)  # sent once
        http.state = "ready"
        s.tick()
        self.assertTrue(s.ready)
        n = len(http.log)
        clock.now += 5
        s.tick()
        self.assertEqual(len(http.log), n)  # not due yet
        clock.now += 5.1
        s.tick()
        self.assertEqual(http.paths()[-1], ("POST", "heartbeat"))
        n = len(http.log)
        clock.now += 3
        s.tick()
        self.assertEqual(len(http.log), n)
        http.fail = True
        for _ in range(3):
            clock.now += 11
            s.tick()
        self.assertTrue(s.dead)
        self.assertIn("unreachable", s.reason)

    def test_main_side_notices_refusal_and_dead_prompt(self):
        http = FakeHttp()
        s = make_session(http)
        http.state = "refused"
        s.tick()
        self.assertTrue(s.dead)
        clock, http = FakeClock(), FakeHttp()
        s = make_session(http, clock)
        s.prompt_id = "pid"
        http.history = {"pid": {"status": {"status_str": "error"}}}
        s.tick()
        self.assertTrue(s.dead)
        self.assertIn("failed", s.reason)

    def test_worker_session_idle_clock(self):
        clock = FakeClock()
        s = remote.WorkerSession("s", "t", clock)
        clock.now += 59
        self.assertFalse(s.idle_expired(60))
        clock.now += 2
        self.assertTrue(s.idle_expired(60))
        s.touch()
        self.assertFalse(s.idle_expired(60))

    def test_sink_serve_loop_idle_with_fake_clock(self):
        remote.reset_worker_state()
        clock = FakeClock()
        patcher = make_patcher()
        sig = remote.model_signature(patcher)
        out = {}

        def run():
            out["r"] = remote.worker_serve(patcher, "sidA", "tokA", sig, clock=clock, idle_timeout=60.0, poll=0.01,
                                           check_interrupt=lambda: None)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        self.wait_for(lambda: "sidA" in remote.SESSIONS)
        self.assertEqual(remote.core_status("sidA", "tokA"), (200, {"state": "waiting_plan"}))
        plan = json.dumps({"tile_shape": list(TILE_SHAPE)}).encode()
        self.assertEqual(remote.core_plan("sidA", "tokB", plan)[0], 403)
        self.assertEqual(remote.core_plan("sidA", "tokA", plan)[0], 200)
        self.wait_for(lambda: remote.core_status("sidA", "tokA")[1]["state"] == "ready")
        clock.now += 59
        time.sleep(0.15)
        self.assertTrue(th.is_alive())
        self.assertEqual(remote.core_heartbeat("sidA", "tokA")[0], 200)  # touch at +59
        clock.now += 59
        time.sleep(0.15)
        self.assertTrue(th.is_alive())  # 59 s after the heartbeat: still fine
        clock.now += 2
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertIn("idle", out["r"])
        self.assertNotIn("sidA", remote.SESSIONS)
        self.assertEqual(remote.core_status("sidA", "tokA"), (200, {"state": "closed"}))

    def wait_for(self, cond, timeout=20.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.01)
        self.fail("timed out")

    def test_tombstone_exits_a_late_sink_immediately(self):
        remote.reset_worker_state()
        patcher = make_patcher()
        sig = remote.model_signature(patcher)
        self.assertEqual(remote.core_done("late", "tok")[0], 200)  # done arrives before the sink ever started
        t0 = time.time()
        r = remote.worker_serve(patcher, "late", "tok", sig, poll=0.01, check_interrupt=lambda: None)
        self.assertLess(time.time() - t0, 2)
        self.assertIn("closed before", r)
        # a different token cannot cancel somebody else's session
        remote.core_done("other", "attacker")
        r2 = {}
        th = threading.Thread(target=lambda: r2.setdefault("r", remote.worker_serve(
            patcher, "other", "real", sig, idle_timeout=0.2, poll=0.01, check_interrupt=lambda: None)), daemon=True)
        th.start()
        th.join(10)
        self.assertIn("idle", r2["r"])

    def test_signature_refusal(self):
        remote.reset_worker_state()
        patcher = make_patcher()
        with self.assertRaises(RuntimeError):
            remote.worker_serve(patcher, "sigx", "tok", "not-the-signature", poll=0.01, check_interrupt=lambda: None)
        self.assertEqual(remote.core_status("sigx", "tok")[1]["state"], "refused")
        self.assertEqual(remote.core_status("sigx", "wrong")[1]["state"], "pending")  # nothing leaks without the token
        self.assertNotIn("sigx", remote.SESSIONS)

    def test_signature_reacts_to_model_changes(self):
        a, b = make_patcher(), make_patcher()
        self.assertEqual(remote.model_signature(a), remote.model_signature(b))
        b.add_wrapper(pe.WrappersMP.DIFFUSION_MODEL, lambda executor, *args, **kw: executor(*args, **kw))
        self.assertNotEqual(remote.model_signature(a), remote.model_signature(b))
        c = make_patcher()
        c.add_wrapper(pe.WrappersMP.DIFFUSION_MODEL, pack.H3TiledForward(256, 256, 64))  # our own wrapper does not count
        self.assertEqual(remote.model_signature(a), remote.model_signature(c))
        d = make_patcher()
        sd = d.model.state_dict()
        key = next(k for k, v in sd.items() if k.startswith("diffusion_model.") and k.endswith("weight") and v.ndim == 2)
        self.assertEqual(len(d.add_patches({key: ("diff", (torch.zeros_like(sd[key]),))}, 0.5)), 1)
        self.assertNotEqual(remote.model_signature(a), remote.model_signature(d))

    def test_done_between_registration_and_signature_is_not_lost(self):
        remote.reset_worker_state()
        patcher = make_patcher()
        sig = remote.model_signature(patcher)
        real = remote.model_signature

        def sig_with_done(p):
            remote.core_done("gap", "tok")  # arrives after the sink registered, before the plan wait
            return real(p)

        remote.model_signature = sig_with_done
        try:
            t0 = time.time()
            r = remote.worker_serve(patcher, "gap", "tok", sig, poll=0.01, idle_timeout=30.0, check_interrupt=lambda: None)
        finally:
            remote.model_signature = real
        self.assertLess(time.time() - t0, 5)
        self.assertIn("closed", r)
        self.assertNotIn("gap", remote.SESSIONS)

    def test_lifecycle_cleans_up_when_prepare_model_patcher_fails(self):
        import comfy.sampler_helpers as sh
        calls = []
        orig_prepare, orig_cleanup = sh.prepare_model_patcher, sh.cleanup_models
        sh.prepare_model_patcher = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("prepare failed"))
        sh.cleanup_models = lambda conds, models: (calls.append("cleanup"), orig_cleanup(conds, models))[1]
        try:
            with self.assertRaises(RuntimeError):
                with remote.worker_lifecycle(make_patcher(), TILE_SHAPE):
                    self.fail("must not get here")
        finally:
            sh.prepare_model_patcher, sh.cleanup_models = orig_prepare, orig_cleanup
        self.assertEqual(calls, ["cleanup"])

    def test_cancelled_future_does_not_break_settling(self):
        import concurrent.futures
        f = concurrent.futures.Future()
        f.cancel()
        remote._settle(f, result=b"x")
        remote._settle(f, exc=RuntimeError("late"))
        g = concurrent.futures.Future()
        remote._settle(g, result=b"ok")
        remote._settle(g, exc=RuntimeError("second"))  # already done: ignored
        self.assertEqual(g.result(), b"ok")
        # a request cancelled by its route and still queued when the session ends must not abort the drain
        remote.reset_worker_state()
        patcher = make_patcher()
        sig = remote.model_signature(patcher)
        cancelled = remote.TileRequest({}, {}, None)
        cancelled.future.cancel()
        waiting = remote.TileRequest({}, {}, None)
        out = {}

        def run():
            out["r"] = remote.worker_serve(patcher, "cx", "tok", sig, poll=0.01, check_interrupt=lambda: None)

        th = threading.Thread(target=run, daemon=True)
        th.start()
        self.wait_for(lambda: "cx" in remote.SESSIONS)
        s = remote.SESSIONS["cx"]
        s.requests.put(cancelled)
        s.requests.put(waiting)
        remote.core_done("cx", "tok")
        th.join(10)
        self.assertFalse(th.is_alive())
        self.assertTrue(waiting.future.done())  # failed or served, but settled

    def test_http_response_reads_are_bounded(self):
        class Resp:
            def read(self, n=-1):
                return b"x" * (n if n >= 0 else 10 ** 6)

        old = remote.MAX_BODY_BYTES
        remote.MAX_BODY_BYTES = 100
        try:
            with self.assertRaises(remote.RemoteError):
                remote._Http._read(Resp())
        finally:
            remote.MAX_BODY_BYTES = old

    def test_signature_covers_patch_offsets_functions_objects_and_partials(self):
        import functools
        a = make_patcher()
        sd = a.model.state_dict()
        key = next(k for k, v in sd.items() if k.startswith("diffusion_model.") and k.endswith("weight") and v.ndim == 2)

        def with_patch(offset=None, function=None):
            p = make_patcher()
            p.add_patches({key: ("diff", (torch.zeros_like(sd[key]),))}, 0.5)
            entry = list(p.patches[key][0])
            entry[3], entry[4] = offset, function
            p.patches[key][0] = tuple(entry)
            return p

        def f1(x):
            return x

        def f2(x):
            return x

        sigs = {remote.model_signature(with_patch()), remote.model_signature(with_patch(offset=(0, 0, 4))),
                remote.model_signature(with_patch(function=f1)), remote.model_signature(with_patch(function=f2))}
        self.assertEqual(len(sigs), 4)
        o1, o2, o3 = make_patcher(), make_patcher(), make_patcher()
        o1.object_patches["diffusion_model.rope"] = torch.nn.Linear(1, 1)
        o2.object_patches["diffusion_model.rope"] = torch.nn.Identity()
        self.assertNotEqual(remote.model_signature(o1), remote.model_signature(o2))
        self.assertNotEqual(remote.model_signature(o1), remote.model_signature(o3))

        def w(executor, *args, scale=1, **kw):
            return executor(*args, **kw)

        def with_wrapper(fn):
            p = make_patcher()
            p.add_wrapper(pe.WrappersMP.DIFFUSION_MODEL, fn)
            return p

        p1 = remote.model_signature(with_wrapper(functools.partial(w, scale=1)))
        p2 = remote.model_signature(with_wrapper(functools.partial(w, scale=2)))
        self.assertNotEqual(p1, p2)
        self.assertEqual(p1, remote.model_signature(with_wrapper(functools.partial(w, scale=1))))
        self.assertNotEqual(p1, remote.model_signature(with_wrapper(w)))

    def test_unfingerprintable_model_never_matches(self):
        a = make_patcher()
        a.patches["diffusion_model.x"] = [("not a number", None, 1.0, None, None)]
        self.assertNotEqual(remote.model_signature(a), remote.model_signature(a))  # a fresh nonce every time

    def test_status_polls_and_heartbeats_keep_a_session_alive(self):
        clock = FakeClock()
        s = remote.WorkerSession("poll", "tok", clock)
        remote.reset_worker_state()
        remote.SESSIONS["poll"] = s
        clock.now += 50
        self.assertEqual(remote.core_status("poll", "tok")[0], 200)  # main polls while it waits for the worker
        clock.now += 50
        self.assertFalse(s.idle_expired(60))
        clock.now += 20
        self.assertTrue(s.idle_expired(60))
        self.assertEqual(remote.core_status("poll", "wrong")[0], 403)  # a wrong token does not count as a sign of life
        self.assertTrue(s.idle_expired(60))

    def test_main_polls_status_from_the_moment_the_prompt_is_queued(self):
        http = FakeHttp()
        s = make_session(http, poll_s=0.02)
        s.open()
        try:
            end = time.time() + 5
            while time.time() < end and sum(1 for _, p in http.paths() if p == "status") < 3:
                time.sleep(0.02)
        finally:
            s.close()
        self.assertGreaterEqual(sum(1 for _, p in http.paths() if p == "status"), 3)  # before any plan or ready

    def test_no_request_is_queued_after_teardown(self):
        remote.reset_worker_state()
        patcher = make_patcher()
        sig = remote.model_signature(patcher)
        th = threading.Thread(target=lambda: remote.worker_serve(patcher, "race", "tok", sig, poll=0.01, check_interrupt=lambda: None),
                              daemon=True)
        th.start()
        self.wait_for(lambda: "race" in remote.SESSIONS)
        s = remote.SESSIONS["race"]
        s.state = "ready"  # as if loaded; the sink is still waiting for a plan
        remote.core_done("race", "tok")
        th.join(10)
        self.assertFalse(th.is_alive())
        s.state = "ready"
        s.closed.set()
        remote.SESSIONS["race"] = s  # a late tile request on the closed session
        status, payload = remote.core_tile_submit("race", "tok", remote.encode({"a": torch.ones(1)}, {"setup": "x"}))
        self.assertIn(status, (409, 410, 400))  # refused at once, never queued
        self.assertTrue(s.requests.empty())
        # and with a valid body the closed check under the lock is what refuses it
        setup = types_ns(context=torch.ones(1, 5, 96))
        s.setups.put("k", setup)
        with unittest.mock.patch.object(remote, "validate_tile_request", lambda *a: None):
            status, payload = remote.core_tile_submit("race", "tok", remote.encode({"a": torch.ones(1)}, {"setup": "k"}))
        self.assertEqual(status, 409)
        self.assertTrue(s.requests.empty())
        remote.SESSIONS.pop("race", None)

    def test_signature_keeps_wrapper_order(self):
        def w1(executor, *a, **k):
            return executor(*a, **k)

        def w2(executor, *a, **k):
            return executor(*a, **k)

        a, b = make_patcher(), make_patcher()
        for fn in (w1, w2):
            a.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, fn.__name__, fn)
        for fn in (w2, w1):
            b.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, fn.__name__, fn)
        self.assertNotEqual(remote.model_signature(a), remote.model_signature(b))

    def test_stop_sequence(self):
        for ready, interrupted, expect in ((False, False, ["done", "queue"]), (True, False, ["done"]),
                                           (True, True, ["done", "interrupt"]), (False, True, ["done", "queue", "interrupt"])):
            http = FakeHttp()
            s = make_session(http)
            s.queued, s.ready = True, ready
            s.close(interrupted)
            s.close(interrupted)  # idempotent
            sent = [(m, p) for m, p, _ in http.log]
            self.assertEqual([p.rsplit("/", 1)[-1] for _, p in sent], expect)
            for method, path, data in http.log:
                if path.endswith("/queue"):
                    self.assertEqual(json.loads(data), {"delete": [s.prompt_id]})
                if path.endswith("/interrupt"):
                    self.assertEqual(json.loads(data), {"prompt_id": s.prompt_id})  # targeted, never global

    def test_bracket_closes_session_and_reports_interrupt(self):
        import comfy.model_management as mm
        closed = []

        class Sess:
            dead, ready = False, False

            def close(self, interrupted=False):
                closed.append(interrupted)

        run = make_run(Sess())
        bracket = remote.RemoteBracket(run)

        class Ex:
            class_obj = None

            def __init__(self, exc=None):
                self.exc = exc

            def __call__(self, *a, **k):
                if self.exc:
                    raise self.exc
                return "ok"

        self.assertEqual(bracket(Ex(), 1), "ok")
        with self.assertRaises(mm.InterruptProcessingException):
            bracket(Ex(mm.InterruptProcessingException()))
        with self.assertRaises(ValueError):
            bracket(Ex(ValueError("x")))
        self.assertEqual(closed, [False, True, False])

    def test_unreachable_worker_does_not_break_the_run(self):
        run = remote.RemoteRun("http://127.0.0.1:1", "sig", {}, ("1", 0))
        bracket = remote.RemoteBracket(run)

        class Ex:
            class_obj = None

            def __call__(self, *a, **k):
                return "sampled"

        self.assertEqual(bracket(Ex()), "sampled")
        self.assertIsNone(run.session)


# -- HTTP layer (real aiohttp routes, worker sink on a thread) -----------------------------------------------------

class HttpWorker:
    def __init__(self, patcher=None):
        self.patcher = patcher or make_patcher()
        self.requests = []
        self.sinks = []
        self.started = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        assert self.started.wait(20)
        return "http://127.0.0.1:%d" % self.port

    def _serve(self):
        from aiohttp import web
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        routes = web.RouteTableDef()
        remote.register_routes(routes)

        @routes.post("/prompt")
        async def prompt(request):
            body = await request.json()
            self.requests.append(("prompt", body))
            inputs = body["prompt"][remote.SINK_NODE_ID]["inputs"]
            result = {}
            th = threading.Thread(target=self._sink, args=(inputs, result), daemon=True)
            th.start()
            self.sinks.append((th, result))
            return web.json_response({"prompt_id": body["prompt_id"], "number": 1, "node_errors": {}})

        @routes.post("/queue")
        async def queue(request):
            self.requests.append(("queue", await request.json()))
            return web.Response()

        @routes.post("/interrupt")
        async def interrupt(request):
            self.requests.append(("interrupt", await request.json()))
            return web.Response()

        @routes.get("/history/{pid}")
        async def history(request):
            return web.json_response({})

        app = web.Application(client_max_size=100 * 1024 * 1024)
        app.add_routes(routes)
        self.runner = web.AppRunner(app)
        self.loop.run_until_complete(self.runner.setup())
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        self.loop.run_until_complete(site.start())
        self.port = site._server.sockets[0].getsockname()[1]
        self.started.set()
        self.loop.run_forever()

    def _sink(self, inputs, result):
        try:
            result["r"] = remote.worker_serve(self.patcher, inputs["session_id"], inputs["token"], inputs["main_signature"],
                                              poll=0.02, idle_timeout=30.0, check_interrupt=lambda: None)
        except BaseException as e:
            result["e"] = e

    def stop(self):
        asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)
        for th, _ in self.sinks:
            th.join(10)


def http(method, url, data=None, headers=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with opener.open(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class G_Http(unittest.TestCase):
    def setUp(self):
        remote.reset_worker_state()
        self.w = HttpWorker()
        self.url = self.w.start()

    def tearDown(self):
        self.w.stop()

    def wait_for(self, cond, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.02)
        self.fail("timed out")

    def test_body_is_not_read_before_auth_and_size_is_checked(self):
        from unittest import mock
        from aiohttp import streams
        base_url = self.url + remote.ROUTE_PREFIX
        remote.SESSIONS["sess2"] = remote.WorkerSession("sess2", "secret")
        reads = []
        orig = streams.StreamReader.iter_chunked

        def counting_iter(self_, n):
            reads.append(1)
            return orig(self_, n)

        with mock.patch.object(streams.StreamReader, "iter_chunked", counting_iter):
            for path in ("plan", "setup", "tile"):
                self.assertEqual(http("POST", "%s/sess2/%s" % (base_url, path), b"x" * 64, {remote.TOKEN_HEADER: "bad"})[0], 403)
                self.assertEqual(http("POST", "%s/nobody/%s" % (base_url, path), b"x" * 64)[0], 409)
            self.assertEqual(reads, [])  # nothing was read for unauthorized requests
            old = remote.MAX_BODY_BYTES
            remote.MAX_BODY_BYTES = 1000
            try:
                self.assertEqual(http("POST", base_url + "/sess2/tile", b"x" * 2000, {remote.TOKEN_HEADER: "secret"})[0], 413)
            finally:
                remote.MAX_BODY_BYTES = old
            self.assertEqual(reads, [])
            http("POST", base_url + "/sess2/plan", b"{}", {remote.TOKEN_HEADER: "secret"})
            self.assertEqual(reads, [1])  # authorized requests are read

    def test_chunked_body_is_capped_too(self):
        import http.client
        remote.SESSIONS["sess3"] = remote.WorkerSession("sess3", "secret")
        port = int(self.url.rsplit(":", 1)[1])
        old = remote.MAX_BODY_BYTES
        remote.MAX_BODY_BYTES = 1000
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
            conn.request("POST", remote.ROUTE_PREFIX + "/sess3/tile", body=iter([b"x" * 400] * 10),
                         headers={remote.TOKEN_HEADER: "secret", "Transfer-Encoding": "chunked"}, encode_chunked=True)
            resp = conn.getresponse()
            self.assertEqual(resp.status, 413)  # no Content-Length: the running cap stops it
            resp.read()
            conn.close()
        finally:
            remote.MAX_BODY_BYTES = old

    def test_token_and_unknown_session(self):
        base_url = self.url + remote.ROUTE_PREFIX
        self.assertEqual(http("GET", base_url + "/nope/status")[0], 200)
        self.assertEqual(json.loads(http("GET", base_url + "/nope/status")[1]), {"state": "pending"})
        self.assertEqual(http("POST", base_url + "/nope/heartbeat", b"", {remote.TOKEN_HEADER: "x"})[0], 409)
        self.assertEqual(http("POST", base_url + "/nope/tile", b"", {remote.TOKEN_HEADER: "x"})[0], 409)
        session = remote.WorkerSession("sess1", "secret")
        remote.SESSIONS["sess1"] = session
        for headers in ({}, {remote.TOKEN_HEADER: "wrong"}):
            for method, path in (("GET", "/sess1/status"), ("POST", "/sess1/heartbeat"), ("POST", "/sess1/plan"),
                                 ("POST", "/sess1/setup"), ("POST", "/sess1/tile"), ("POST", "/sess1/done")):
                self.assertEqual(http(method, base_url + path, b"" if method == "POST" else None, headers)[0], 403, path)
        self.assertNotIn("sess1", remote.TOMBSTONES)  # a bad token cannot close it
        self.assertFalse(session.closed.is_set())
        ok = {remote.TOKEN_HEADER: "secret"}
        self.assertEqual(http("POST", base_url + "/sess1/heartbeat", b"", ok)[0], 200)
        self.assertEqual(http("POST", base_url + "/sess1/tile", b"", ok)[0], 409)  # not ready yet
        self.assertEqual(http("POST", base_url + "/sess1/plan", b"{bad", ok)[0], 400)
        self.assertEqual(http("POST", base_url + "/sess1/plan", json.dumps({"tile_shape": [1, 2]}).encode(), ok)[0], 400)
        self.assertEqual(http("POST", base_url + "/sess1/done", b"", ok)[0], 200)
        self.assertTrue(session.closed.is_set())

    def test_end_to_end_two_servers_in_one_process(self):
        main_dm = tiny_model()
        x, ctx, ts, payload = make_inputs(h=22, w=34)
        patcher = make_patcher()
        signature = remote.model_signature(patcher)
        prompt = {"1": {"class_type": "Loader", "inputs": {"ckpt": "a"}}, "2": {"class_type": "Ours", "inputs": {"model": ["1", 0]}}}
        closure, source = remote.build_closure(prompt, "2", classes=FAKE_CLASSES)

        def factory():
            s = remote.RemoteSession(self.url, signature, closure, source, poll_s=0.05, heartbeat_s=0.3)
            s.open()
            return s

        run = remote.RemoteRun(self.url, signature, closure, source, session_factory=factory)
        fwd = remote.H3TiledForward2GPU(256, 256, 64, True, run)
        sample_sigmas = torch.tensor([1.0, 0.6, 0.0])
        ref = local_reference(main_dm, x, ctx, ts, payload)
        run.begin()
        try:
            def call():
                opts = wrapper_options(fwd)
                opts["sample_sigmas"] = sample_sigmas
                opts["sigmas"] = ts
                opts["cond_or_uncond"] = [0]
                opts["uuids"] = ["abc"]
                with torch.inference_mode():
                    return main_dm(x, ts, ctx, transformer_options=opts, minimax_payload=payload)

            self.assertEqual(self.w.requests[0][0], "prompt")
            sent = self.w.requests[0][1]
            self.assertEqual(sent["prompt"][remote.SINK_NODE_ID]["inputs"]["main_signature"], signature)
            out = call()  # worker not ready yet: all local, plan goes out
            self.assertTrue(torch.equal(out[0], ref[0]))
            self.wait_for(lambda: run.session.ready)
            out = call()  # calibration
            self.assertTrue(torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]))
            self.assertIsNotNone(run.ratio)
            self.assertFalse(run.session.dead, run.session.reason)
            sess_id = run.session.session_id
            out = call()
            self.assertTrue(torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]))
            self.assertFalse(run.session.dead, run.session.reason)
            time.sleep(0.8)  # heartbeats keep flowing
            self.assertFalse(run.session.dead)
        finally:
            run.finish(False)
        th, result = self.w.sinks[0]
        th.join(20)
        self.assertFalse(th.is_alive())
        self.assertIn("served", result.get("r", repr(result)))
        self.assertNotIn(sess_id, remote.SESSIONS)
        self.assertEqual([k for k, _ in self.w.requests if k != "prompt"], [])  # started and ready: no queue delete

    def test_unknown_session_gets_dropped_from_queue_on_close(self):
        # the worker's queue never starts the sink: close() removes the queued prompt
        w = self.w
        w.patcher = make_patcher()
        signature = remote.model_signature(w.patcher)
        s = remote.RemoteSession(self.url, signature, {"1": {"class_type": "Loader", "inputs": {}}}, ("1", 0))
        orig = w._sink
        w._sink = lambda inputs, result: None  # simulate a busy worker: the prompt is queued, not started
        s.open()
        s.close(False)
        kinds = [k for k, _ in w.requests]
        self.assertEqual(kinds, ["prompt", "queue"])
        self.assertEqual(w.requests[1][1], {"delete": [s.prompt_id]})
        self.assertIn(s.session_id, remote.TOMBSTONES)
        w._sink = orig

    def test_signature_mismatch_over_http_disables_remote_but_run_is_fine(self):
        main_dm = tiny_model()
        x, ctx, ts, payload = make_inputs()
        ref = local_reference(main_dm, x, ctx, ts, payload)
        closure = {"1": {"class_type": "Loader", "inputs": {}}}

        def factory():
            s = remote.RemoteSession(self.url, "wrong-signature", closure, ("1", 0), poll_s=0.05, heartbeat_s=0.3)
            s.open()
            return s

        run = remote.RemoteRun(self.url, "wrong-signature", closure, ("1", 0), session_factory=factory)
        fwd = remote.H3TiledForward2GPU(256, 256, 64, True, run)
        run.begin()
        try:
            with torch.inference_mode():
                out = main_dm(x, ts, ctx, transformer_options=wrapper_options(fwd), minimax_payload=payload)
            self.assertTrue(torch.equal(out[0], ref[0]))
            self.wait_for(lambda: run.session.dead)
            self.assertIn("refused", run.session.reason)
            with torch.inference_mode():
                out = main_dm(x, ts, ctx, transformer_options=wrapper_options(fwd), minimax_payload=payload)
            self.assertTrue(torch.equal(out[0], ref[0]))
        finally:
            run.finish(False)


# -- node ---------------------------------------------------------------------------------------------------------------

class H_Node(unittest.TestCase):
    def setUp(self):
        self._orig = remote._node_classes
        remote._node_classes = lambda: FAKE_CLASSES

    def tearDown(self):
        remote._node_classes = self._orig

    def prompt(self):
        return {"1": {"class_type": "Loader", "inputs": {"ckpt": "a"}}, "2": {"class_type": "Ours", "inputs": {"model": ["1", 0]}}}

    def test_schema(self):
        spec = remote.UltimateSDUpscaleH3TiledSampling2GPU.INPUT_TYPES()
        old = pack.UltimateSDUpscaleH3TiledSampling.INPUT_TYPES()
        self.assertEqual(list(spec["required"])[:len(old["required"])], list(old["required"]))
        self.assertEqual(spec["required"]["worker_url"][1]["default"], "http://127.0.0.1:8189")
        self.assertEqual(spec["hidden"], {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"})
        self.assertEqual(list(old), ["required"])  # the one-GPU node is untouched
        self.assertEqual(list(old["required"]), ["model", "tile_width", "tile_height", "tile_overlap", "true_positions"])
        self.assertIn("UltimateSDUpscaleH3TiledSampling2GPU", remote.NODE_CLASS_MAPPINGS)
        self.assertTrue(remote.UltimateSDUpscaleH3TileWorker.OUTPUT_NODE)

    def test_apply_adds_wrappers_and_signature_ignores_them(self):
        patcher = make_patcher()
        node = remote.UltimateSDUpscaleH3TiledSampling2GPU()
        (m,) = node.apply(patcher, 768, 768, 128, True, "http://127.0.0.1:8189", self.prompt(), "2")
        fwds = m.wrappers[pe.WrappersMP.DIFFUSION_MODEL][None]
        self.assertIsInstance(fwds[0], remote.H3TiledForward2GPU)
        self.assertIsNotNone(fwds[0].run)
        self.assertEqual(fwds[0].run.signature, remote.model_signature(patcher))
        self.assertEqual(remote.model_signature(m), remote.model_signature(patcher))
        self.assertIsInstance(m.wrappers[pe.WrappersMP.OUTER_SAMPLE]["usdu_h3_remote"][0], remote.RemoteBracket)
        self.assertNotIn(pe.WrappersMP.OUTER_SAMPLE, patcher.wrappers)  # the input model is not modified
        with self.assertRaises(RuntimeError):
            node.apply(m, 768, 768, 128, True, "http://127.0.0.1:8189", self.prompt(), "2")  # stacking

    def test_blank_or_bad_url_or_closure_means_local_only(self):
        node = remote.UltimateSDUpscaleH3TiledSampling2GPU()
        for url, prompt, uid in (("", self.prompt(), "2"), ("ftp://x", self.prompt(), "2"), ("http://u:p@x:1", self.prompt(), "2"),
                                 ("http://127.0.0.1:8189", self.prompt(), "7"), ("http://127.0.0.1:8189", None, None),
                                 ("http://127.0.0.1:8189", {"1": {"class_type": "Loader", "inputs": {}},
                                                           "3": {"class_type": "SaveModel", "inputs": {"model": ["1", 0]}},
                                                           "2": {"class_type": "Ours", "inputs": {"model": ["3", 0]}}}, "2")):
            (m,) = node.apply(make_patcher(), 768, 768, 128, True, url, prompt, uid)
            self.assertIsNone(m.wrappers[pe.WrappersMP.DIFFUSION_MODEL][None][0].run, url)
            self.assertNotIn(pe.WrappersMP.OUTER_SAMPLE, m.wrappers)

    def test_local_only_node_matches_one_gpu_node(self):
        dm = tiny_model()
        x, ctx, ts, payload = make_inputs()
        fwd = remote.H3TiledForward2GPU(256, 256, 64, True, None)
        with torch.inference_mode():
            a = dm(x, ts, ctx, transformer_options=wrapper_options(fwd), minimax_payload=payload)
        ref = local_reference(dm, x, ctx, ts, payload)
        self.assertTrue(torch.equal(a[0], ref[0]) and torch.equal(a[1], ref[1]))


if __name__ == "__main__":
    unittest.main(argv=["h3-remote-tests"], verbosity=2)
