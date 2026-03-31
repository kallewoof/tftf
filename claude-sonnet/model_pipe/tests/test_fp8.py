"""
Tests for FP8 dequantisation utilities and FP8DequantPipe.

All tests that create real FP8 tensors are guarded by ``skip_no_fp8``.
The dequant *math* and pipe *logic* are tested using float32 tensors with
the FP8 dtype check monkey-patched, so the arithmetic runs on any PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
import torch
from safetensors.torch import load_file, save_file

import model_pipe.utils.fp8 as fp8_mod
from model_pipe.io.null_writer import NullWriter
from model_pipe.io.reader import SafetensorsReader
from model_pipe.io.sharded_reader import ShardedSafetensorsReader
from model_pipe.io.sharded_writer import ShardedWriter
from model_pipe.io.writer import StreamingWriter
from model_pipe.pipeline import Pipeline
from model_pipe.pipes.base import Pipe, TensorMeta, TensorRecord
from model_pipe.pipes.dtype_cast import DTypeCastPipe
from model_pipe.pipes.fp8_dequant import FP8DequantPipe, _NOT_SCANNED
from model_pipe.pipes.lora_merge import LoRAMergePipe
from model_pipe.pipes.passthrough import PassthroughPipe
from model_pipe.utils.fp8 import (
    HAS_FP8,
    dequantize_fp8_weight,
    is_fp8_dtype,
    is_scale_key,
    scale_inv_key_for,
    scale_key_for,
    weight_key_for_scale,
)

# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

skip_no_fp8 = pytest.mark.skipif(not HAS_FP8, reason="torch.float8_e4m3fn not available")
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------

class fp8_patch:
    """Context manager that adds torch.float32 to _FP8_DTYPES for testing."""
    def __enter__(self):
        fp8_mod._FP8_DTYPES.add(torch.float32)
    def __exit__(self, *_):
        fp8_mod._FP8_DTYPES.discard(torch.float32)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _save(t: dict, p: Path) -> None:
    save_file(t, str(p))

def _load(p: Path) -> dict:
    return load_file(str(p))


def _make_fp8_model(
    tmp: Path, out_f: int = 256, in_f: int = 128, block_size: int = 128
) -> tuple[Path, dict[str, torch.Tensor]]:
    """
    Build a synthetic model file containing one quantised weight and one plain weight.
    When FP8 is unavailable the weight is stored as float32 and tests use the monkey-patch.
    Returns (path, ground_truth) where ground_truth maps weight keys to expected
    dequantised float32 tensors.
    """
    n_rb = (out_f + block_size - 1) // block_size
    n_cb = (in_f  + block_size - 1) // block_size

    original = torch.randn(out_f, in_f, dtype=torch.float32)

    # Build per-block scale_inv: scale_inv[br, bc] = max_abs_block / 448
    scale_inv = torch.zeros(n_rb, n_cb, dtype=torch.float32)
    for br in range(n_rb):
        for bc in range(n_cb):
            r0, r1 = br*block_size, min((br+1)*block_size, out_f)
            c0, c1 = bc*block_size, min((bc+1)*block_size, in_f)
            bmax = original[r0:r1, c0:c1].abs().max().item()
            scale_inv[br, bc] = bmax / 448.0 if bmax > 0 else 1.0

    tensors: dict[str, torch.Tensor] = {
        "model.norm.weight": torch.randn(in_f),
    }

    if HAS_FP8:
        # Quantise to FP8
        fp8_weight = torch.zeros(out_f, in_f, dtype=torch.float32)
        for br in range(n_rb):
            for bc in range(n_cb):
                r0, r1 = br*block_size, min((br+1)*block_size, out_f)
                c0, c1 = bc*block_size, min((bc+1)*block_size, in_f)
                s = scale_inv[br, bc]
                fp8_weight[r0:r1, c0:c1] = (original[r0:r1, c0:c1] / s).clamp(-448, 448)
        fp8_weight = fp8_weight.to(FP8_DTYPE)
        tensors["model.layers.0.mlp.gate_proj.weight"] = fp8_weight
        tensors["model.layers.0.mlp.gate_proj.weight_scale_inv"] = scale_inv

        # Ground truth: fp8_as_float * scale_inv
        gt = torch.zeros(out_f, in_f, dtype=torch.float32)
        fp8_f32 = fp8_weight.float()
        for br in range(n_rb):
            for bc in range(n_cb):
                r0, r1 = br*block_size, min((br+1)*block_size, out_f)
                c0, c1 = bc*block_size, min((bc+1)*block_size, in_f)
                gt[r0:r1, c0:c1] = fp8_f32[r0:r1, c0:c1] * scale_inv[br, bc]
    else:
        # Use float32 as stand-in for FP8 (monkey-patched in tests)
        tensors["model.layers.0.mlp.gate_proj.weight"] = original
        tensors["model.layers.0.mlp.gate_proj.weight_scale_inv"] = scale_inv

        # Ground truth: original * scale_inv block-wise
        gt = torch.zeros(out_f, in_f, dtype=torch.float32)
        for br in range(n_rb):
            for bc in range(n_cb):
                r0, r1 = br*block_size, min((br+1)*block_size, out_f)
                c0, c1 = bc*block_size, min((bc+1)*block_size, in_f)
                gt[r0:r1, c0:c1] = original[r0:r1, c0:c1] * scale_inv[br, bc]

    ground_truth: dict[str, torch.Tensor] = {
        "model.norm.weight": tensors["model.norm.weight"].clone(),
        "model.layers.0.mlp.gate_proj.weight": gt,
    }

    path = tmp / "model.safetensors"
    _save(tensors, path)
    return path, ground_truth


def _make_lora(tmp: Path, out_f=256, in_f=128, rank=4, alpha=8.0):
    t = {
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight":
            torch.randn(rank, in_f) * 0.01,
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight":
            torch.zeros(out_f, rank),
    }
    ap = tmp / "adapter_model.safetensors"
    _save(t, ap)
    (tmp / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": alpha, "target_modules": ["gate_proj"]})
    )
    return ap


# ===========================================================================
# Key helpers
# ===========================================================================

class TestFP8KeyHelpers:

    def test_is_scale_key_inv(self):
        assert is_scale_key("model.layers.0.q_proj.weight_scale_inv")

    def test_is_scale_key_fwd(self):
        assert is_scale_key("model.layers.0.q_proj.weight_scale")

    def test_is_scale_key_rejects_plain_weight(self):
        assert not is_scale_key("model.layers.0.q_proj.weight")

    def test_is_scale_key_no_false_positive_rope(self):
        # "rope_scale" should NOT be treated as a weight scale companion
        assert not is_scale_key("model.rope_scale")

    def test_is_scale_key_no_false_positive_layernorm(self):
        assert not is_scale_key("model.input_layernorm.weight_scale_inv_factor")

    def test_weight_key_for_scale_inv(self):
        assert weight_key_for_scale("model.layers.0.q_proj.weight_scale_inv") \
               == "model.layers.0.q_proj.weight"

    def test_weight_key_for_scale_fwd(self):
        assert weight_key_for_scale("model.layers.0.q_proj.weight_scale") \
               == "model.layers.0.q_proj.weight"

    def test_weight_key_for_scale_invalid(self):
        with pytest.raises(ValueError, match="Not a recognised"):
            weight_key_for_scale("model.layers.0.q_proj.weight")

    def test_scale_inv_key_for(self):
        assert scale_inv_key_for("model.layers.0.q_proj.weight") \
               == "model.layers.0.q_proj.weight_scale_inv"

    def test_scale_key_for(self):
        assert scale_key_for("model.layers.0.q_proj.weight") \
               == "model.layers.0.q_proj.weight_scale"

    @skip_no_fp8
    def test_is_fp8_dtype_e4m3fn(self):
        assert is_fp8_dtype(torch.float8_e4m3fn)

    def test_is_fp8_dtype_float32_is_false(self):
        assert not is_fp8_dtype(torch.float32)

    def test_has_fp8_reflects_torch_version(self):
        # HAS_FP8 should be True iff torch.float8_e4m3fn exists
        expected = hasattr(torch, "float8_e4m3fn")
        assert HAS_FP8 == expected


# ===========================================================================
# dequantize_fp8_weight — math correctness
# ===========================================================================

class TestDequantizeMath:

    def _w_and_scale(self, out_f, in_f, scale_val=2.0, B=128):
        n_rb = (out_f + B - 1) // B
        n_cb = (in_f  + B - 1) // B
        w = torch.randn(out_f, in_f)
        s = torch.full((n_rb, n_cb), scale_val)
        return w, s

    def test_uniform_scale_multiply(self):
        with fp8_patch():
            w, s = self._w_and_scale(256, 128, scale_val=3.0)
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32)
        assert torch.allclose(result, w * 3.0, atol=1e-5)

    def test_non_aligned_dimensions(self):
        """200×100 weight: partial last block must be handled correctly."""
        with fp8_patch():
            w, s = self._w_and_scale(200, 100, scale_val=2.0)
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32)
        assert torch.allclose(result, w * 2.0, atol=1e-5)
        assert result.shape == (200, 100)

    def test_single_block(self):
        with fp8_patch():
            w = torch.ones(64, 64)
            s = torch.full((1, 1), 5.0)
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32, block_size=128)
        assert torch.allclose(result, torch.full((64, 64), 5.0), atol=1e-5)

    def test_invert_scale_divides(self):
        with fp8_patch():
            w = torch.full((128, 128), 6.0)
            s = torch.full((1, 1), 2.0)
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32, invert_scale=True)
        assert torch.allclose(result, torch.full((128, 128), 3.0), atol=1e-5)

    def test_per_block_different_scales(self):
        """Two row blocks with different scales — each block must get its own scale."""
        with fp8_patch():
            w = torch.ones(256, 128)
            s = torch.tensor([[2.0], [5.0]])  # shape (2, 1)
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32)
        assert torch.allclose(result[:128], torch.full((128, 128), 2.0), atol=1e-5)
        assert torch.allclose(result[128:], torch.full((128, 128), 5.0), atol=1e-5)

    def test_output_dtype_bfloat16(self):
        with fp8_patch():
            w, s = self._w_and_scale(128, 128)
            result = dequantize_fp8_weight(w, s, target_dtype=torch.bfloat16)
        assert result.dtype == torch.bfloat16

    def test_output_is_contiguous(self):
        with fp8_patch():
            w, s = self._w_and_scale(200, 100)
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32)
        assert result.is_contiguous()

    def test_transposed_scale_tolerated(self):
        """Scale provided as (n_cb, n_rb) instead of (n_rb, n_cb)."""
        with fp8_patch():
            w = torch.ones(256, 128)   # n_rb=2, n_cb=1
            s_correct   = torch.full((2, 1), 3.0)
            s_transposed = s_correct.t()  # shape (1, 2)
            r1 = dequantize_fp8_weight(w, s_correct,    target_dtype=torch.float32)
            r2 = dequantize_fp8_weight(w, s_transposed, target_dtype=torch.float32)
        assert torch.allclose(r1, r2, atol=1e-5)

    def test_wrong_ndim_raises(self):
        with fp8_patch():
            with pytest.raises(ValueError, match="2-D"):
                dequantize_fp8_weight(torch.randn(4), torch.ones(1, 1),
                                      target_dtype=torch.float32)

    def test_non_fp8_dtype_raises(self):
        with pytest.raises(ValueError, match="FP8"):
            dequantize_fp8_weight(torch.randn(4, 4), torch.ones(1, 1),
                                  target_dtype=torch.float32)

    def test_wrong_scale_shape_raises(self):
        with fp8_patch():
            with pytest.raises(ValueError, match="inconsistent"):
                dequantize_fp8_weight(torch.randn(128, 128), torch.ones(5, 5),
                                      target_dtype=torch.float32)

    def test_vectorised_equals_loop(self):
        """Vectorised result must match a reference loop implementation."""
        B = 128
        out_f, in_f = 200, 100
        n_rb = (out_f + B - 1) // B
        n_cb = (in_f  + B - 1) // B
        w = torch.randn(out_f, in_f)
        s = torch.rand(n_rb, n_cb) + 0.5

        # Reference loop
        ref = torch.zeros(out_f, in_f)
        for br in range(n_rb):
            for bc in range(n_cb):
                r0, r1 = br*B, min((br+1)*B, out_f)
                c0, c1 = bc*B, min((bc+1)*B, in_f)
                ref[r0:r1, c0:c1] = w[r0:r1, c0:c1] * s[br, bc]

        with fp8_patch():
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32, block_size=B)

        assert torch.allclose(result, ref, atol=1e-5), \
            f"max diff: {(result - ref).abs().max().item()}"


# ===========================================================================
# FP8DequantPipe: process_meta
# ===========================================================================

class TestFP8DequantPipeMeta:

    def _metas(self, spec):
        return [TensorMeta(k, d, torch.Size(s)) for k, d, s in spec]

    def test_scale_inv_key_dropped(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("m.weight",           torch.float32, (128, 128)),
            ("m.weight_scale_inv", torch.float32, (1, 1)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        assert all("_scale" not in m.key for m in out)

    def test_fp8_weight_gets_target_dtype(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("m.weight",           torch.float32, (128, 128)),
            ("m.weight_scale_inv", torch.float32, (1, 1)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        w = [m for m in out if m.key == "m.weight"]
        assert len(w) == 1
        assert w[0].dtype == torch.bfloat16

    def test_fp8_weight_shape_unchanged(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("m.weight",           torch.float32, (200, 100)),
            ("m.weight_scale_inv", torch.float32, (2, 1)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        w = [m for m in out if m.key == "m.weight"][0]
        assert w.shape == torch.Size((200, 100))

    def test_plain_tensor_unchanged(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([("norm.weight", torch.float32, (64,))])
        out = list(pipe.process_meta(iter(metas)))
        assert out[0].dtype == torch.float32

    def test_output_count_correct(self):
        """1 FP8 weight + 1 scale_inv + 2 plain = 3 output metas."""
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("a.weight",           torch.float32, (128, 128)),
            ("a.weight_scale_inv", torch.float32, (1, 1)),
            ("b.weight",           torch.float32, (64,)),
            ("c.weight",           torch.float32, (64,)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        assert len(out) == 3

    def test_fp8_dtype_detected_without_scale_key(self):
        """A tensor whose dtype is FP8 should be flagged even with no scale in stream."""
        if not HAS_FP8:
            pytest.skip("FP8 dtype not available")
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([("m.weight", torch.float8_e4m3fn, (128, 128))])
        out = list(pipe.process_meta(iter(metas)))
        assert out[0].dtype == torch.bfloat16

    def test_state_populated_after_scan(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        assert pipe._fp8_weight_keys is _NOT_SCANNED
        metas = self._metas([
            ("m.weight",           torch.float32, (128, 128)),
            ("m.weight_scale_inv", torch.float32, (1, 1)),
        ])
        list(pipe.process_meta(iter(metas)))
        assert isinstance(pipe._fp8_weight_keys, set)
        assert "m.weight" in pipe._fp8_weight_keys


# ===========================================================================
# FP8DequantPipe: process
# ===========================================================================

class TestFP8DequantPipeProcess:

    def _make_pipe(self, key: str) -> FP8DequantPipe:
        """Return a pipe pre-configured to treat *key* as an FP8 weight."""
        pipe = FP8DequantPipe(torch.float32)
        pipe._fp8_weight_keys = {key}
        pipe._use_inv_scale = {key: True}
        return pipe

    def test_process_before_meta_raises(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        with pytest.raises(RuntimeError, match="process_meta"):
            list(pipe.process(iter([])))

    def test_weight_then_scale(self):
        key = "m.weight"
        with fp8_patch():
            pipe = self._make_pipe(key)
            w = torch.full((128, 128), 2.0)
            s = torch.full((1, 1), 3.0)
            out = list(pipe.process(iter([
                TensorRecord(key,                   w),
                TensorRecord(key + "_scale_inv",    s),
            ])))
        assert len(out) == 1
        assert torch.allclose(out[0].tensor, torch.full((128, 128), 6.0), atol=1e-4)

    def test_scale_then_weight(self):
        key = "m.weight"
        with fp8_patch():
            pipe = self._make_pipe(key)
            w = torch.full((128, 128), 2.0)
            s = torch.full((1, 1), 3.0)
            out = list(pipe.process(iter([
                TensorRecord(key + "_scale_inv",    s),
                TensorRecord(key,                   w),
            ])))
        assert len(out) == 1
        assert torch.allclose(out[0].tensor, torch.full((128, 128), 6.0), atol=1e-4)

    def test_plain_tensor_passes_through(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        pipe._fp8_weight_keys = set()
        pipe._use_inv_scale = {}
        norm = torch.randn(64)
        out = list(pipe.process(iter([TensorRecord("norm.weight", norm)])))
        assert len(out) == 1
        assert torch.allclose(out[0].tensor, norm)

    def test_multiple_pairs_interleaved(self):
        with fp8_patch():
            pipe = FP8DequantPipe(torch.float32)
            pipe._fp8_weight_keys = {"w0", "w1"}
            pipe._use_inv_scale = {"w0": True, "w1": True}
            out = list(pipe.process(iter([
                TensorRecord("w0",               torch.full((128, 128), 1.0)),
                TensorRecord("w1",               torch.full((128, 128), 2.0)),
                TensorRecord("w0.weight_scale_inv", torch.full((1, 1), 2.0)),
                TensorRecord("w1.weight_scale_inv", torch.full((1, 1), 3.0)),
            ])))
        assert len(out) == 2
        result = {r.key: r.tensor for r in out}
        assert torch.allclose(result["w0"], torch.full((128, 128), 2.0), atol=1e-4)
        assert torch.allclose(result["w1"], torch.full((128, 128), 6.0), atol=1e-4)

    def test_orphan_scale_warns(self, caplog):
        import logging
        with fp8_patch():
            pipe = FP8DequantPipe(torch.float32)
            pipe._fp8_weight_keys = {"orphan.weight"}
            pipe._use_inv_scale = {"orphan.weight": True}
            with caplog.at_level(logging.WARNING):
                list(pipe.process(iter([
                    TensorRecord("orphan.weight_scale_inv", torch.ones(1, 1))
                ])))
        assert any("scale" in m.lower() for m in caplog.messages)


# ===========================================================================
# Full pipeline integration
# ===========================================================================

class TestFP8PipelineIntegration:

    def test_no_fp8_model_passthrough(self, tmp_path):
        t = {"norm.weight": torch.randn(64), "embed.weight": torch.randn(128, 64)}
        _save(t, tmp_path / "model.safetensors")
        out = tmp_path / "out.safetensors"
        Pipeline(
            SafetensorsReader(tmp_path / "model.safetensors"),
            FP8DequantPipe(torch.bfloat16),
            StreamingWriter(out),
        ).run(show_progress=False)
        result = _load(out)
        for k, v in t.items():
            assert torch.allclose(result[k], v)

    def test_scale_keys_not_in_output(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path)
        out = tmp_path / "out.safetensors"
        with fp8_patch() if not HAS_FP8 else nullcontext():
            Pipeline(
                SafetensorsReader(model_path),
                FP8DequantPipe(torch.float32),
                StreamingWriter(out),
            ).run(show_progress=False)
        result = _load(out)
        assert not any(is_scale_key(k) for k in result)

    def test_dequant_dtype_correct(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path)
        out = tmp_path / "out_bf16.safetensors"
        with fp8_patch() if not HAS_FP8 else nullcontext():
            Pipeline(
                SafetensorsReader(model_path),
                FP8DequantPipe(torch.bfloat16),
                StreamingWriter(out),
            ).run(show_progress=False)
        result = _load(out)
        assert result["model.layers.0.mlp.gate_proj.weight"].dtype == torch.bfloat16

    def test_dequant_shape_preserved(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path, out_f=256, in_f=128)
        out = tmp_path / "out.safetensors"
        with fp8_patch() if not HAS_FP8 else nullcontext():
            Pipeline(
                SafetensorsReader(model_path),
                FP8DequantPipe(torch.float32),
                StreamingWriter(out),
            ).run(show_progress=False)
        result = _load(out)
        assert result["model.layers.0.mlp.gate_proj.weight"].shape == (256, 128)

    def test_dequant_values_match_ground_truth(self, tmp_path):
        model_path, gt = _make_fp8_model(tmp_path, out_f=256, in_f=128)
        out = tmp_path / "out.safetensors"
        with fp8_patch() if not HAS_FP8 else nullcontext():
            Pipeline(
                SafetensorsReader(model_path),
                FP8DequantPipe(torch.float32),
                StreamingWriter(out),
            ).run(show_progress=False)
        result = _load(out)
        key = "model.layers.0.mlp.gate_proj.weight"
        assert torch.allclose(result[key].float(), gt[key].float(), atol=1e-4), \
            f"max diff: {(result[key].float() - gt[key].float()).abs().max().item()}"

    def test_plain_tensors_unchanged(self, tmp_path):
        model_path, gt = _make_fp8_model(tmp_path)
        out = tmp_path / "out.safetensors"
        with fp8_patch() if not HAS_FP8 else nullcontext():
            Pipeline(
                SafetensorsReader(model_path),
                FP8DequantPipe(torch.bfloat16),
                StreamingWriter(out),
            ).run(show_progress=False)
        result = _load(out)
        assert torch.allclose(result["model.norm.weight"], gt["model.norm.weight"])

    def test_dry_run_passes(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path)
        w = NullWriter()
        with fp8_patch() if not HAS_FP8 else nullcontext():
            Pipeline(
                SafetensorsReader(model_path),
                FP8DequantPipe(torch.bfloat16),
                w,
            ).run(show_progress=False)
        assert w.report.ok, w.report.summary()

    def test_dequant_then_lora(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path, out_f=256, in_f=128)
        adapter_path = _make_lora(tmp_path, out_f=256, in_f=128)
        out = tmp_path / "merged.safetensors"
        with fp8_patch() if not HAS_FP8 else nullcontext():
            pipe = FP8DequantPipe(torch.float32) | LoRAMergePipe(adapter_path)
            Pipeline(
                SafetensorsReader(model_path), pipe, StreamingWriter(out)
            ).run(show_progress=False)
        result = _load(out)
        assert "model.layers.0.mlp.gate_proj.weight" in result
        assert "model.norm.weight" in result
        assert "model.layers.0.mlp.gate_proj.weight_scale_inv" not in result

    def test_dequant_sharded_output(self, tmp_path):
        model_path, gt = _make_fp8_model(tmp_path, out_f=256, in_f=128)
        out_dir = tmp_path / "shards"
        with fp8_patch() if not HAS_FP8 else nullcontext():
            Pipeline(
                SafetensorsReader(model_path),
                FP8DequantPipe(torch.float32),
                ShardedWriter(out_dir, max_shard_bytes=4096),
            ).run(show_progress=False)
        reader = ShardedSafetensorsReader.from_path(out_dir)
        result = {r.key: r.tensor for r in reader.iter_records()}
        assert not any(is_scale_key(k) for k in result)
        key = "model.layers.0.mlp.gate_proj.weight"
        assert torch.allclose(result[key].float(), gt[key].float(), atol=1e-4)


# ===========================================================================
# Repr
# ===========================================================================

class TestFP8Repr:

    def test_repr_unscanned(self):
        r = repr(FP8DequantPipe(torch.bfloat16))
        assert "FP8DequantPipe" in r
        assert "unscanned" in r

    def test_repr_after_scan(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path)
        pipe = FP8DequantPipe(torch.bfloat16)
        list(pipe.process_meta(SafetensorsReader(model_path).iter_meta()))
        r = repr(pipe)
        assert "fp8 key" in r

    def test_repr_in_compound(self):
        r = repr(FP8DequantPipe(torch.bfloat16) | DTypeCastPipe(torch.float16))
        assert "FP8DequantPipe" in r
        assert "DTypeCastPipe" in r


# ===========================================================================
# CLI
# ===========================================================================

class TestCLI:

    def test_dequant_fp8_registered(self):
        from model_pipe.cli import cli
        assert "dequant-fp8" in cli.commands

    def test_dequant_fp8_options(self):
        from model_pipe.cli import cli
        params = {p.name for p in cli.commands["dequant-fp8"].params}
        for expected in ("dtype", "lora_adapter", "dry_run", "sharded",
                         "max_shard_size", "block_size", "device"):
            assert expected in params, f"Missing option: {expected}"


# ---------------------------------------------------------------------------
# nullcontext shim for Python < 3.10 (contextlib.nullcontext exists in 3.7+
# but we import it here to be explicit)
# ---------------------------------------------------------------------------
from contextlib import nullcontext
