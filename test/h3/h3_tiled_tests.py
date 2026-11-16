import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import h3_env  # noqa: E402,F401  (must come first: sets up the CPU-only ComfyUI import)
from h3_env import h3, load_pack, tiny_model, wrapper_options  # noqa: E402

import functools  # noqa: E402

import torch  # noqa: E402

import comfy.model_base  # noqa: E402
import comfy.nested_tensor  # noqa: E402
import comfy.patcher_extension as pe  # noqa: E402

pack = load_pack()
Wrapper = pack.H3TiledForward


def pointwise_executor(x, timestep, context, transformer_options=None, **kwargs):
    # position independent and pointwise: tiled blending must reproduce it exactly
    if isinstance(x, (list, tuple)):
        return [x[0] * 2 + 1, x[1]]
    return x * 2 + 1


def make_payload(h, w, t=3, audio_t=6, text_len=5, with_audio_kf=True, with_ref=True):
    g = torch.Generator().manual_seed(1)
    keyframes = [{"latent": torch.randn(1, 24, 2, h, w, generator=g), "resolved_frame_index": 0}]
    if with_audio_kf:
        keyframes.append({"audio_latent": torch.randn(1, 32, 2, 3, generator=g), "resolved_frame_index": 3})
    refs = []
    if with_ref:
        refs.append({"kind": "image", "latent_h": 6, "latent_w": 8, "latent": torch.randn(1, 24, 1, 6, 8, generator=g)})
    payload = {"keyframes": keyframes, "refs": refs, "seed": 0, "audio_scale": 1.0}
    payload.update(comfy.model_base.MiniMaxH3._cond_latents(keyframes, refs))
    payload["layout"] = h3.PackedLayout(text_len, t, h + h % 2, w + w % 2, audio_t, keyframes=keyframes, refs=refs)
    return payload


class TilePlan(unittest.TestCase):
    def test_axis_tiles(self):
        for length in (2, 8, 16, 18, 30, 64, 66):
            for tile in (4, 8, 16):
                for overlap in (0, 2, 4, 8, 16):
                    size, starts = pack.axis_tiles(length, tile, overlap)
                    self.assertEqual(starts[0], 0)
                    self.assertEqual(starts[-1] + size, length)
                    self.assertTrue(all(s % 2 == 0 for s in starts))
                    self.assertEqual(len(set(starts)), len(starts))
                    for a, b in zip(starts, starts[1:]):
                        self.assertLessEqual(b, a + size)  # no gaps
                    if length <= tile:
                        self.assertEqual((size, starts), (length, [0]))

    def test_weights_complement(self):
        size, starts = pack.axis_tiles(26, 16, 6)
        total = torch.zeros(26)
        for i, s in enumerate(starts):
            total[s:s + size] += pack.axis_weights(starts, size, i)
        self.assertTrue(torch.allclose(total, torch.ones(26)))  # two-tile overlaps already sum to one
        self.assertEqual(float(pack.axis_weights(starts, size, 0)[0]), 1.0)  # frame edge keeps weight 1
        self.assertEqual(float(pack.axis_weights(starts, size, len(starts) - 1)[-1]), 1.0)


class A_NoOp(unittest.TestCase):
    def test_tile_larger_than_frame_is_exact(self):
        calls = []

        def executor(x, *a, **k):
            calls.append(x)
            return pointwise_executor(x, *a, **k)

        x = [torch.randn(1, 24, 3, 20, 30), torch.randn(1, 32, 2, 6)]
        out = Wrapper(2048, 2048, 128)(executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {})
        ref = pointwise_executor(x, None, None)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], x)
        self.assertTrue(torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]))

    def test_real_model_noop(self):
        model = tiny_model()
        x = [torch.randn(1, 24, 3, 8, 12), torch.randn(1, 32, 2, 6)]
        ctx, ts = torch.randn(1, 5, 96), torch.tensor([500.0])
        with torch.inference_mode():
            ref = model(x, ts, ctx)
            out = model(x, ts, ctx, transformer_options=wrapper_options(Wrapper(2048, 2048, 128)))
        self.assertTrue(torch.equal(ref[0], out[0]) and torch.equal(ref[1], out[1]))


class B_Blending(unittest.TestCase):
    def test_pointwise_matches_full_frame(self):
        cases = [  # (latent h, w, tile_w px, tile_h px, overlap px)
            (20, 30, 256, 256, 64), (16, 48, 256, 512, 64), (22, 34, 224, 192, 32), (30, 18, 192, 256, 96),
            (21, 31, 224, 224, 64), (40, 40, 160, 160, 0), (64, 64, 256, 256, 224),
        ]
        for h, w, tw, th, ov in cases:
            with self.subTest(h=h, w=w, tw=tw, th=th, ov=ov):
                x = [torch.randn(1, 24, 3, h, w), torch.randn(1, 32, 2, 6)]
                seen = []

                def executor(xx, *a, **k):
                    seen.append(tuple(xx[0].shape[-2:]))
                    return pointwise_executor(xx, *a, **k)

                out = Wrapper(tw, th, ov)(executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {})
                ref = pointwise_executor(x, None, None)
                self.assertEqual(tuple(out[0].shape), tuple(x[0].shape))
                self.assertTrue(torch.allclose(out[0], ref[0], atol=1e-5, rtol=1e-5), float((out[0] - ref[0]).abs().max()))
                self.assertTrue(torch.allclose(out[1], ref[1], atol=1e-6))
                self.assertGreater(len(seen), 1)

    def test_image_model_tensor_input(self):
        x = torch.randn(1, 24, 1, 20, 30)
        out = Wrapper(256, 256, 64)(pointwise_executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {})
        self.assertTrue(torch.allclose(out, x * 2 + 1, atol=1e-5))

    def test_blend_is_a_weighted_average_of_tiles(self):
        # a tile-dependent executor: output stays within the range of the tile values, seams are monotone ramps
        x = [torch.zeros(1, 24, 1, 16, 48), torch.zeros(1, 32, 2, 4)]
        counter = {"n": 0}

        def executor(xx, *a, **k):
            counter["n"] += 1
            return [torch.full_like(xx[0], float(counter["n"])), xx[1]]

        out = Wrapper(256, 256, 128)(executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {})[0]
        row = out[0, 0, 0, 0]
        self.assertAlmostEqual(float(row[0]), 1.0, places=5)
        self.assertAlmostEqual(float(row[-1]), float(counter["n"]), places=5)
        self.assertTrue(bool((row[1:] >= row[:-1] - 1e-6).all()))


class C_TruePositions(unittest.TestCase):
    def check(self, h, w, t, tile_h, tile_w, y0, x0, payload, audio_t, text_len):
        full = payload["layout"]
        tile_payload = pack.crop_payload(payload, y0, y0 + tile_h, x0, x0 + tile_w)
        args = (full, text_len, t, tile_h, tile_w, audio_t, y0, x0, h, w, tile_payload["keyframes"], tile_payload["refs"])
        lay = pack.tile_layout(*args, True)
        local = pack.tile_layout(*args, False)
        self.assertEqual(lay.signature, (text_len, t, tile_h, tile_w, audio_t))
        self.assertEqual([k for _, _, k in lay.segments], [k for _, _, k in full.segments])
        n_checked = 0
        for (a, b, kind), (fa, fb, _), (la, lb, _) in zip(lay.segments, full.segments, local.segments):
            rows, frows, lrows = lay.position_ids[a:b], full.position_ids[fa:fb], local.position_ids[la:lb]
            if kind in ("video", "cond"):
                frames = (fb - fa) // ((h // 2) * (w // 2))
                expect = frows.reshape(frames, h // 2, w // 2, 3)[:, y0 // 2:(y0 + tile_h) // 2, x0 // 2:(x0 + tile_w) // 2]
                self.assertTrue(torch.equal(rows, expect.reshape(-1, 3)), kind)
                n_checked += 1
                if (tile_h, tile_w) != (h, w):
                    self.assertFalse(torch.equal(lrows[:, 1:], rows[:, 1:]))  # tile-local grid really differs
            elif kind in ("audio", "cond_audio"):
                self.assertTrue(torch.equal(rows, frows), kind)
            else:  # text, ref_img, ref_audio: untouched
                self.assertTrue(torch.equal(rows, lrows), kind)
                self.assertTrue(torch.equal(rows, frows), kind)
        self.assertGreaterEqual(n_checked, 2)  # keyframe cond and target video both covered

    def test_rows_match_full_frame(self):
        for h, w, tile_h, tile_w, y0, x0 in [(20, 30, 12, 16, 8, 14), (20, 30, 12, 16, 0, 0), (16, 48, 16, 16, 0, 32),
                                              (18, 18, 8, 8, 10, 10)]:
            with self.subTest(h=h, w=w, tile=(tile_h, tile_w), origin=(y0, x0)):
                self.check(h, w, 3, tile_h, tile_w, y0, x0, make_payload(h, w), 6, 5)

    def test_video_refs_unchanged(self):
        h, w = 20, 30
        g = torch.Generator().manual_seed(2)
        refs = [{"kind": "video_audio", "latent_h": 8, "latent_w": 12, "latent_t": 2, "ref_audio_t": 3,
                 "latent": torch.randn(1, 24, 2, 8, 12, generator=g), "audio_latent": torch.randn(1, 32, 2, 3, generator=g)}]
        full = h3.PackedLayout(5, 3, h, w, 6, keyframes=None, refs=refs)
        lay = pack.tile_layout(full, 5, 3, 12, 16, 6, 4, 6, h, w, [], refs, True)
        n_ref = 0
        for (a, b, kind), (fa, fb, _) in zip(lay.segments, full.segments):
            if kind.startswith("ref"):
                n_ref += 1
                self.assertTrue(torch.equal(lay.position_ids[a:b], full.position_ids[fa:fb]))
        self.assertEqual(n_ref, 2)


class D_Cropping(unittest.TestCase):
    def test_crop_payload_shapes(self):
        payload = make_payload(20, 30)
        tp = pack.crop_payload(payload, 4, 16, 6, 22)
        self.assertNotIn("layout", tp)
        self.assertEqual(tuple(tp["keyframes"][0]["latent"].shape), (1, 24, 2, 12, 16))
        self.assertIsNone(tp["keyframes"][1].get("latent"))
        self.assertIs(tp["keyframes"][1]["audio_latent"], payload["keyframes"][1]["audio_latent"])
        self.assertIs(tp["refs"], payload["refs"])
        cond = tp["cond_video_latents"]
        self.assertEqual(len(cond), 2)
        self.assertEqual(tuple(cond[0].shape), (1, 24, 2, 12, 16))
        self.assertIs(cond[1], payload["refs"][0]["latent"])  # reference whole, still after the keyframes
        self.assertTrue(torch.equal(cond[0], payload["keyframes"][0]["latent"][..., 4:16, 6:22]))
        self.assertIs(tp["cond_audio_latents"], payload["cond_audio_latents"])
        self.assertEqual(tuple(payload["keyframes"][0]["latent"].shape), (1, 24, 2, 20, 30))  # original untouched

    def test_wrapper_hands_cropped_inputs_to_each_tile(self):
        h, w, t, audio_t = 20, 30, 3, 6
        payload = make_payload(h, w, t, audio_t)
        mask = torch.rand(1, 1, t, h, w)
        audio_mask = torch.rand(1, 1, 2, audio_t)
        x = [torch.randn(1, 24, t, h, w), torch.randn(1, 32, 2, audio_t)]
        seen = []

        def executor(xx, ts, ctx, to, minimax_payload=None, denoise_mask=None, audio_denoise_mask=None, **kw):
            th, tw = xx[0].shape[-2:]
            seen.append((th, tw))
            self.assertEqual(tuple(denoise_mask.shape), (1, 1, t, th, tw))
            self.assertIs(audio_denoise_mask, audio_mask)
            self.assertIs(xx[1], x[1])
            self.assertEqual(tuple(minimax_payload["keyframes"][0]["latent"].shape), (1, 24, 2, th, tw))
            self.assertEqual(tuple(minimax_payload["cond_video_latents"][0].shape), (1, 24, 2, th, tw))
            self.assertIs(minimax_payload["refs"], payload["refs"])
            self.assertEqual(minimax_payload["layout"].signature, (5, t, th, tw, audio_t))  # forward would not rebuild it
            return [xx[0] * 1.0, xx[1] * 1.0]

        out = Wrapper(256, 192, 64)(executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {},
                                    minimax_payload=payload, denoise_mask=mask, audio_denoise_mask=audio_mask)
        self.assertGreater(len(seen), 2)
        self.assertEqual(len(set(seen)), 1)  # all tiles the same size
        self.assertTrue(torch.allclose(out[0], x[0], atol=1e-5))

    def test_layouts_cached_across_steps(self):
        wrapper = Wrapper(256, 192, 64)
        x = [torch.randn(1, 24, 3, 20, 30), torch.randn(1, 32, 2, 6)]
        for _ in range(3):
            wrapper(pointwise_executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {}, minimax_payload=make_payload(20, 30))
        first = dict(wrapper._layouts)
        self.assertGreater(len(first), 1)
        wrapper(pointwise_executor, x, torch.tensor([400.0]), torch.randn(1, 5, 96), {}, minimax_payload=make_payload(20, 30))
        self.assertEqual(set(first), set(wrapper._layouts))
        self.assertTrue(all(first[k] is wrapper._layouts[k] for k in first))


class E_RealModel(unittest.TestCase):
    def run_model(self, true_positions, tile=(256, 256, 64), t=3, h=16, w=24, mask=None):
        model = tiny_model()
        payload = make_payload(h, w, t, 6)
        x = [torch.randn(1, 24, t, h, w, generator=torch.Generator().manual_seed(3)),
             torch.randn(1, 32, 2, 6, generator=torch.Generator().manual_seed(4))]
        ctx, ts = torch.randn(1, 5, 96, generator=torch.Generator().manual_seed(5)), torch.tensor([600.0])
        kw = {"denoise_mask": mask} if mask is not None else {}
        with torch.inference_mode():
            out = model(x, ts, ctx, transformer_options=wrapper_options(Wrapper(tile[0], tile[1], tile[2], true_positions)),
                        minimax_payload=payload, **kw)
            ref = model(x, ts, ctx, minimax_payload=payload, **kw)
        return x, out, ref

    def test_two_tiles_with_keyframe(self):
        x, out, ref = self.run_model(True)
        self.assertEqual(tuple(out[0].shape), tuple(x[0].shape))
        self.assertEqual(tuple(out[1].shape), tuple(x[1].shape))
        for o in out:
            self.assertFalse(bool(torch.isnan(o).any()) or bool(torch.isinf(o).any()))
        self.assertGreater(float((out[0] - ref[0]).abs().max()), 0.0)  # it really tiled

    def test_four_tiles_with_mask(self):
        mask = torch.ones(1, 1, 3, 24, 24)
        mask[..., 4:12, :] = 0.5
        x, out, ref = self.run_model(True, tile=(256, 256, 96), h=24, w=24, mask=mask)
        self.assertEqual(tuple(out[0].shape), (1, 24, 3, 24, 24))
        self.assertFalse(bool(torch.isnan(out[0]).any()) or bool(torch.isnan(out[1]).any()))

    def test_true_positions_changes_result(self):
        _, a, _ = self.run_model(True)
        _, b, _ = self.run_model(False)
        self.assertFalse(torch.equal(a[0], b[0]))
        self.assertFalse(bool(torch.isnan(b[0]).any()))

    def test_odd_latent_size(self):
        # odd latent height: tiling happens on the even-padded grid and the output keeps the input size
        model = tiny_model()
        x = [torch.randn(1, 24, 3, 15, 24), torch.randn(1, 32, 2, 6)]
        with torch.inference_mode():
            out = model(x, torch.tensor([600.0]), torch.randn(1, 5, 96),
                        transformer_options=wrapper_options(Wrapper(256, 256, 64)))
        self.assertEqual(tuple(out[0].shape), tuple(x[0].shape))
        self.assertFalse(bool(torch.isnan(out[0]).any()))

    def test_image_model(self):
        model = tiny_model(image=True)
        x = torch.randn(1, 24, 1, 16, 24)
        ctx, ts = torch.randn(1, 5, 96), torch.tensor([600.0])
        with torch.inference_mode():
            out = model(x, ts, ctx, transformer_options=wrapper_options(Wrapper(256, 256, 64)), minimax_payload={})
            ref = model(x, ts, ctx, minimax_payload={})
        self.assertEqual(tuple(out.shape), tuple(x.shape))
        self.assertEqual(tuple(ref.shape), tuple(x.shape))
        self.assertFalse(bool(torch.isnan(out).any()))


class G_Guards(unittest.TestCase):
    class FunControlPatch:
        def diffusion_model_wrapper(self, executor, x, *a, **k):
            return executor(x, *a, **k)

    def test_funcontrol_and_spectrum_rejected(self):
        wp = self.FunControlPatch()
        wrappers = {pe.WrappersMP.DIFFUSION_MODEL: {None: [wp.diffusion_model_wrapper]}}
        with self.assertRaises(RuntimeError):
            Wrapper.check_compatible(wrappers)
        x = [torch.randn(1, 24, 3, 20, 30), torch.randn(1, 32, 2, 6)]
        with self.assertRaises(RuntimeError):
            Wrapper(256, 256, 64)(pointwise_executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96),
                                  {"wrappers": wrappers})

        def spectrum_wrapper(executor, *a, **k):
            return executor(*a, **k)
        spectrum_wrapper.__module__ = "comfyui_spectrum_h3.sampling"
        with self.assertRaises(RuntimeError):
            Wrapper.check_compatible({pe.WrappersMP.SAMPLER_SAMPLE: {None: [spectrum_wrapper]}})
        Wrapper.check_compatible({pe.WrappersMP.DIFFUSION_MODEL: {None: [Wrapper(256, 256, 64)]}})  # fine

    def test_node_adds_wrapper_to_clone(self):
        class FakeModel:
            def __init__(self):
                self.wrappers = pe.WrappersMP.init_wrappers()
                self.model_options = {"transformer_options": {}}

            def clone(self):
                m = FakeModel()
                m.wrappers = {k: {k1: list(v1) for k1, v1 in v.items()} for k, v in self.wrappers.items()}
                return m

            def add_wrapper(self, t, fn):
                self.wrappers.setdefault(t, {}).setdefault(None, []).append(fn)

        src = FakeModel()
        (out,) = pack.UltimateSDUpscaleH3TiledSampling().apply(src, 512, 640, 96, False)
        self.assertEqual(src.wrappers.get(pe.WrappersMP.DIFFUSION_MODEL, {}), {})
        fn = out.wrappers[pe.WrappersMP.DIFFUSION_MODEL][None][0]
        self.assertIsInstance(fn, Wrapper)
        self.assertEqual((fn.tile_width, fn.tile_height, fn.tile_overlap, fn.true_positions), (512, 640, 96, False))
        self.assertIn("UltimateSDUpscaleH3TiledSampling", pack.NODE_CLASS_MAPPINGS)


class F_HoldAudio(unittest.TestCase):
    def setUp(self):
        self.video = torch.randn(1, 24, 3, 8, 12)
        self.audio = torch.randn(1, 32, 2, 6)
        self.node = pack.UltimateSDUpscaleH3HoldAudio()

    def test_builds_mask(self):
        latent = {"samples": comfy.nested_tensor.NestedTensor([self.video, self.audio]), "batch_index": [0]}
        (out,) = self.node.apply(latent)
        v, a = out["noise_mask"].unbind()
        self.assertTrue(out["noise_mask"].is_nested)
        self.assertEqual(tuple(v.shape), (1, 1, 3, 8, 12))
        self.assertTrue(bool((v == 1).all()))
        self.assertEqual(tuple(a.shape), (1, 1, 2, 6))
        self.assertTrue(bool((a == 0).all()))
        self.assertIs(out["samples"], latent["samples"])
        self.assertEqual(out["batch_index"], [0])
        self.assertNotIn("noise_mask", latent)  # input latent not modified

    def test_combines_with_existing_masks(self):
        existing = torch.rand(1, 1, 3, 8, 12)
        samples = comfy.nested_tensor.NestedTensor([self.video, self.audio])
        (out,) = self.node.apply({"samples": samples, "noise_mask": existing})
        v, a = out["noise_mask"].unbind()
        self.assertTrue(torch.equal(v, existing) and bool((a == 0).all()))
        nested = comfy.nested_tensor.NestedTensor([existing, torch.ones(1, 1, 2, 6)])
        (out,) = self.node.apply({"samples": samples, "noise_mask": nested})
        v, a = out["noise_mask"].unbind()
        self.assertTrue(torch.equal(v, existing) and bool((a == 0).all()))

    def test_non_av_latent_passes_through(self):
        latent = {"samples": torch.randn(1, 24, 3, 8, 12)}
        (out,) = self.node.apply(latent)
        self.assertIs(out, latent)
        self.assertNotIn("noise_mask", out)

    def test_mask_reshapes_like_the_sampler(self):
        import comfy.sampler_helpers as sh
        (out,) = self.node.apply({"samples": comfy.nested_tensor.NestedTensor([self.video, self.audio])})
        for m, ref in zip(out["noise_mask"].unbind(), (self.video, self.audio)):
            r = sh.prepare_mask(m, tuple(ref.shape), "cpu")
            self.assertEqual(tuple(r.shape), tuple(ref.shape))


class H_ReviewFixes(unittest.TestCase):
    def test_odd_axis_split_with_keyframe_real_model(self):
        # width 29 (padded 30) is split into 2 tiles; the last tile's keyframe crop is odd and must be padded
        h, w, t = 16, 29, 3
        model = tiny_model()
        payload = make_payload(h, w, t, 6)
        x = [torch.randn(1, 24, t, h, w), torch.randn(1, 32, 2, 6)]
        wrapper = Wrapper(256, 256, 64)
        _, ys, _, xs = wrapper.grid(h, w)
        self.assertEqual((ys, xs[-1]), ([0], 14))
        self.assertGreater(len(xs), 1)
        with torch.inference_mode():
            out = model(x, torch.tensor([600.0]), torch.randn(1, 5, 96),
                        transformer_options=wrapper_options(wrapper), minimax_payload=payload)
        self.assertEqual(tuple(out[0].shape), tuple(x[0].shape))
        for o in out:
            self.assertFalse(bool(torch.isnan(o).any()))
        tp = pack.crop_payload(payload, 0, 16, 14, 30)
        self.assertEqual(tuple(tp["keyframes"][0]["latent"].shape[-2:]), (16, 16))
        self.assertEqual(tuple(tp["cond_video_latents"][0].shape[-2:]), (16, 16))
        # positions of the odd tile equal the cropped rows of the even-padded full layout
        C_TruePositions().check.__func__(self, 16, 30, t, 16, 16, 0, 14, payload, 6, 5)

    def test_audio_output_is_plain_mean_over_tiles(self):
        x = [torch.randn(1, 24, 1, 16, 48), torch.randn(1, 32, 2, 4)]
        means = []

        def executor(xx, *a, **k):
            m = float(xx[0].mean())
            means.append(m)
            return [xx[0] * 1.0, xx[1] + m]

        out = Wrapper(256, 256, 128)(executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {})
        self.assertGreater(len(means), 2)
        self.assertTrue(torch.allclose(out[1], x[1] + sum(means) / len(means), atol=1e-5))

    def test_image_model_without_payload_gets_true_positions(self):
        seen = []
        x = torch.randn(1, 24, 1, 16, 48)

        def executor(xx, ts, ctx, to, minimax_payload=None, **k):
            self.assertIsNotNone(minimax_payload)
            seen.append(minimax_payload["layout"].position_ids.clone())
            return xx * 1.0

        for tp in (True, False):
            Wrapper(256, 256, 128, tp)(executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {}, minimax_payload=None)
        n = len(seen) // 2
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(seen[:n], seen[n:])))
        # true positions: tile 1 (x0 = 8) holds the full-frame w positions of patch cols 4..11
        full = h3.PackedLayout(5, 1, 16, 48, 0).position_ids
        self.assertTrue(torch.equal(seen[1][-8 * 8:, 2].reshape(8, 8), full[-24 * 8:, 2].reshape(8, 24)[:, 4:12]))
        # and through the real tiny image model the outputs differ
        model = tiny_model(image=True)
        ctx, ts = torch.randn(1, 5, 96), torch.tensor([600.0])
        outs = []
        with torch.inference_mode():
            for tp in (True, False):
                outs.append(model(x, ts, ctx, transformer_options=wrapper_options(Wrapper(256, 256, 128, tp))))
        self.assertFalse(torch.equal(outs[0], outs[1]))

    def test_stacking_and_partial_guards(self):
        class FakeModel:
            def __init__(self):
                self.wrappers = pe.WrappersMP.init_wrappers()
                self.model_options = {"transformer_options": {}}

            def clone(self):
                m = FakeModel()
                m.wrappers = {k: {k1: list(v1) for k1, v1 in v.items()} for k, v in self.wrappers.items()}
                return m

            def add_wrapper(self, t, fn):
                self.wrappers.setdefault(t, {}).setdefault(None, []).append(fn)

        node = pack.UltimateSDUpscaleH3TiledSampling()
        (once,) = node.apply(FakeModel(), 512, 512, 64, True)
        with self.assertRaises(RuntimeError):
            node.apply(once, 512, 512, 64, True)
        two = {pe.WrappersMP.DIFFUSION_MODEL: {None: [Wrapper(256, 256, 64), functools.partial(Wrapper(256, 256, 64))]}}
        with self.assertRaises(RuntimeError):
            Wrapper.check_compatible(two)
        fc = G_Guards.FunControlPatch()
        hidden = {pe.WrappersMP.DIFFUSION_MODEL: {None: [functools.partial(fc.diffusion_model_wrapper)]}}
        with self.assertRaises(RuntimeError):
            Wrapper.check_compatible(hidden)

    def test_layout_cache_is_lru(self):
        wrapper = Wrapper(256, 192, 64)
        wrapper._layouts.size = 3
        x = [torch.randn(1, 24, 3, 40, 40), torch.randn(1, 32, 2, 6)]
        payload = make_payload(40, 40)
        wrapper(pointwise_executor, x, torch.tensor([500.0]), torch.randn(1, 5, 96), {}, minimax_payload=payload)
        self.assertEqual(len(wrapper._layouts), 3)  # bounded, evicts one at a time instead of clearing

    def test_hold_audio_keeps_device_of_inputs(self):
        video, audio = torch.randn(1, 24, 3, 8, 12), torch.randn(1, 32, 2, 6)
        (out,) = pack.UltimateSDUpscaleH3HoldAudio().apply({"samples": comfy.nested_tensor.NestedTensor([video, audio])})
        v, a = out["noise_mask"].unbind()
        self.assertEqual((v.device, a.device), (video.device, audio.device))


if __name__ == "__main__":
    unittest.main(argv=["h3-tests"], verbosity=2)
