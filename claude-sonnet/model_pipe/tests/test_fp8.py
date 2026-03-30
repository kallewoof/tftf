"""
Tests for FP8 dequantisation.

Because torch.float8_e4m3fn may not exist on older PyTorch, every test
that constructs an actual FP8 tensor is guarded with a ``pytest.importorskip``
/ ``pytest.mark.skipif`` that checks for FP8 dtype availability.

The dequantisation *math* is tested using float32 inputs (pretending they
are FP8) so that the arithmetic can be verified on any PyTorch version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest
import torch
from safetensors.torch import load_file, save_file

from model_pipe.io.null_writer import NullWriter
from model_pipe.io.reader import SafetensorsReader
from model_pipe.io.sharded_reader import ShardedSafetensorsReader
from model_pipe.io.sharded_writer import ShardedWriter
from model_pipe.io.writer import StreamingWriter
from model_pipe.pipeline import Pipeline
from model_pipe.pipes.base import Pipe, TensorMeta, TensorRecord
from model_pipe.pipes.dtype_cast import DTypeCastPipe
from model_pipe.pipes.fp8_dequant import FP8DequantPipe
from model_pipe.pipes.lora_merge import LoRAMergePipe
from model_pipe.pipes.passthrough import PassthroughPipe
from model_pipe.utils.fp8 import (
    dequantize_fp8_weight,
    is_fp8_dtype,
    is_scale_key,
    scale_inv_key_for,
    scale_key_for,
    weight_key_for_scale,
)

# ---------------------------------------------------------------------------
# FP8 availability guard
# ---------------------------------------------------------------------------

HAS_FP8 = hasattr(torch, "float8_e4m3fn")
skip_no_fp8 = pytest.mark.skipif(not HAS_FP8, reason="torch.float8_e4m3fn not available")

FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(t: dict, p: Path) -> None:
    save_file(t, str(p))


def _load(p: Path) -> dict:
    return load_file(str(p))


def _make_fp8_model(
    tmp: Path,
    out_f: int = 256,
    in_f: int = 128,
    block_size: int = 128,
) -> tuple[Path, dict[str, torch.Tensor]]:
    """
    Build a synthetic FP8 model with one quantised weight and one plain weight.

    Returns (path, ground_truth) where ground_truth maps weight keys to their
    *expected dequantised float32* values (NOT the FP8 storage values).
    """
    n_row_blocks = (out_f + block_size - 1) // block_size
    n_col_blocks = (in_f  + block_size - 1) // block_size

    # Build a random BF16 weight, then simulate quantisation
    original = torch.randn(out_f, in_f, dtype=torch.float32)

    # Compute per-block max and scale_inv = 448.0 / block_max  (e4m3 max = 448)
    scale_inv = torch.zeros(n_row_blocks, n_col_blocks, dtype=torch.float32)
    for br in range(n_row_blocks):
        for bc in range(n_col_blocks):
            r0, r1 = br * block_size, min((br+1) * block_size, out_f)
            c0, c1 = bc * block_size, min((bc+1) * block_size, in_f)
            block_max = original[r0:r1, c0:c1].abs().max().item()
            if block_max == 0.0:
                scale_inv[br, bc] = 1.0
            else:
                scale_inv[br, bc] = block_max / 448.0  # typical e4m3 scale

    tensors: dict[str, torch.Tensor] = {
        "model.norm.weight": torch.randn(in_f),  # plain tensor, not FP8
    }
    ground_truth: dict[str, torch.Tensor] = {
        "model.norm.weight": tensors["model.norm.weight"].clone(),
    }

    if HAS_FP8:
        # Store real FP8 weights
        # Quantise: fp8 = clip(original / scale_inv, -448, 448)
        fp8_weight = torch.zeros(out_f, in_f, dtype=torch.float32)
        for br in range(n_row_blocks):
            for bc in range(n_col_blocks):
                r0, r1 = br*block_size, min((br+1)*block_size, out_f)
                c0, c1 = bc*block_size, min((bc+1)*block_size, in_f)
                s = scale_inv[br, bc]
                fp8_weight[r0:r1, c0:c1] = (original[r0:r1, c0:c1] / s).clamp(-448, 448)
        fp8_weight = fp8_weight.to(FP8_DTYPE)
        tensors["model.layers.0.mlp.gate_proj.weight"] = fp8_weight
        tensors["model.layers.0.mlp.gate_proj.weight_scale_inv"] = scale_inv

        # Ground truth = fp8_as_float * scale_inv (round-trip may lose precision)
        recon = torch.zeros(out_f, in_f, dtype=torch.float32)
        fp8_f32 = fp8_weight.float()
        for br in range(n_row_blocks):
            for bc in range(n_col_blocks):
                r0, r1 = br*block_size, min((br+1)*block_size, out_f)
                c0, c1 = bc*block_size, min((bc+1)*block_size, in_f)
                recon[r0:r1, c0:c1] = fp8_f32[r0:r1, c0:c1] * scale_inv[br, bc]
        ground_truth["model.layers.0.mlp.gate_proj.weight"] = recon
    else:
        # Simulate with float32 "pretending" to be FP8
        tensors["model.layers.0.mlp.gate_proj.weight"] = original
        tensors["model.layers.0.mlp.gate_proj.weight_scale_inv"] = scale_inv
        # Ground truth for simulation: orig * scale_inv block-wise
        recon = torch.zeros(out_f, in_f, dtype=torch.float32)
        for br in range(n_row_blocks):
            for bc in range(n_col_blocks):
                r0, r1 = br*block_size, min((br+1)*block_size, out_f)
                c0, c1 = bc*block_size, min((bc+1)*block_size, in_f)
                recon[r0:r1, c0:c1] = original[r0:r1, c0:c1] * scale_inv[br, bc]
        ground_truth["model.layers.0.mlp.gate_proj.weight"] = recon

    path = tmp / "model.safetensors"
    _save(tensors, path)
    return path, ground_truth


def _make_lora(tmp: Path, out_f: int = 256, in_f: int = 128,
               rank: int = 4, alpha: float = 8.0) -> tuple[Path, Path]:
    t = {
        "base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight":
            torch.randn(rank, in_f) * 0.01,
        "base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight":
            torch.zeros(out_f, rank),  # zero B → merge is identity
    }
    ap = tmp / "adapter_model.safetensors"
    _save(t, ap)
    cp = tmp / "adapter_config.json"
    cp.write_text(json.dumps({"r": rank, "lora_alpha": alpha, "target_modules": ["gate_proj"]}))
    return ap, cp


# ===========================================================================
# Utils: key helpers
# ===========================================================================

class TestFP8KeyHelpers:

    def test_is_scale_key_inv(self):
        assert is_scale_key("model.layers.0.weight_scale_inv")
        assert not is_scale_key("model.layers.0.weight")

    def test_is_scale_key_non_inv(self):
        assert is_scale_key("model.layers.0.weight_scale")

    def test_weight_key_for_scale_inv(self):
        assert weight_key_for_scale("a.b.weight_scale_inv") == "a.b.weight"

    def test_weight_key_for_scale_non_inv(self):
        assert weight_key_for_scale("a.b.weight_scale") == "a.b.weight"

    def test_weight_key_for_scale_invalid(self):
        with pytest.raises(ValueError):
            weight_key_for_scale("a.b.weight")

    def test_scale_inv_key_for(self):
        assert scale_inv_key_for("a.b.weight") == "a.b.weight_scale_inv"

    def test_scale_key_for(self):
        assert scale_key_for("a.b.weight") == "a.b.weight_scale"


# ===========================================================================
# Utils: dequantize_fp8_weight math
# ===========================================================================

class TestDequantizeFP8Weight:
    """Test dequantisation logic using float32 inputs (no FP8 dtype needed)."""

    def _make_weight_and_scale(self, out_f, in_f, block_size=128):
        """Return (weight_f32, scale_inv) where we fake FP8 with float32."""
        w = torch.randn(out_f, in_f)
        n_r = (out_f + block_size - 1) // block_size
        n_c = (in_f  + block_size - 1) // block_size
        s = torch.ones(n_r, n_c) * 2.0  # scale_inv = 2.0 everywhere
        return w, s

    def test_exact_block_aligned(self):
        """256×128 weight with 2-block rows — scale_inv=2.0 means output = w*2."""
        w, s = self._make_weight_and_scale(256, 128)
        # Monkey-patch is_fp8_dtype to accept float32 for this test
        import model_pipe.utils.fp8 as fp8_mod
        original_fp8 = fp8_mod._FP8_DTYPES.copy()
        fp8_mod._FP8_DTYPES.add(torch.float32)
        try:
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32, block_size=128)
        finally:
            fp8_mod._FP8_DTYPES.discard(torch.float32)
            fp8_mod._FP8_DTYPES.update(original_fp8)
        expected = w * 2.0
        assert torch.allclose(result, expected, atol=1e-5)

    def test_non_aligned_dimensions(self):
        """200×100 weight — partial last block must still dequantise correctly."""
        import model_pipe.utils.fp8 as fp8_mod
        w = torch.ones(200, 100)
        # scale_inv: (2, 1) — one scale for the whole thing since 200×100 < 256×128
        s = torch.full((2, 1), 3.0)  # 2 row-blocks (0:128, 128:200), 1 col-block

        fp8_mod._FP8_DTYPES.add(torch.float32)
        try:
            result = dequantize_fp8_weight(w, s, target_dtype=torch.float32, block_size=128)
        finally:
            fp8_mod._FP8_DTYPES.discard(torch.float32)

        assert torch.allclose(result, torch.full_like(result, 3.0), atol=1e-5)

    def test_invert_scale_flag(self):
        """invert_scale=True → divide by scale instead of multiply."""
        import model_pipe.utils.fp8 as fp8_mod
        w = torch.full((128, 128), 4.0)
        s = torch.full((1, 1), 2.0)

        fp8_mod._FP8_DTYPES.add(torch.float32)
        try:
            result_mul = dequantize_fp8_weight(w, s, target_dtype=torch.float32,
                                               invert_scale=False, block_size=128)
            result_div = dequantize_fp8_weight(w, s, target_dtype=torch.float32,
                                               invert_scale=True, block_size=128)
        finally:
            fp8_mod._FP8_DTYPES.discard(torch.float32)

        assert torch.allclose(result_mul, torch.full_like(result_mul, 8.0))   # 4*2
        assert torch.allclose(result_div, torch.full_like(result_div, 2.0))   # 4/2

    def test_wrong_ndim_raises(self):
        import model_pipe.utils.fp8 as fp8_mod
        fp8_mod._FP8_DTYPES.add(torch.float32)
        try:
            with pytest.raises(ValueError, match="2-D"):
                dequantize_fp8_weight(torch.randn(4), torch.ones(1, 1),
                                      target_dtype=torch.float32)
        finally:
            fp8_mod._FP8_DTYPES.discard(torch.float32)

    def test_wrong_dtype_raises(self):
        with pytest.raises(ValueError, match="FP8"):
            dequantize_fp8_weight(torch.randn(4, 4), torch.ones(1, 1),
                                  target_dtype=torch.float32)

    def test_wrong_scale_shape_raises(self):
        import model_pipe.utils.fp8 as fp8_mod
        fp8_mod._FP8_DTYPES.add(torch.float32)
        try:
            with pytest.raises(ValueError, match="inconsistent"):
                dequantize_fp8_weight(torch.randn(128, 128), torch.ones(5, 5),
                                      target_dtype=torch.float32)
        finally:
            fp8_mod._FP8_DTYPES.discard(torch.float32)

    def test_output_dtype_respected(self):
        import model_pipe.utils.fp8 as fp8_mod
        fp8_mod._FP8_DTYPES.add(torch.float32)
        try:
            result = dequantize_fp8_weight(
                torch.randn(128, 128), torch.ones(1, 1),
                target_dtype=torch.bfloat16,
            )
        finally:
            fp8_mod._FP8_DTYPES.discard(torch.float32)
        assert result.dtype == torch.bfloat16


# ===========================================================================
# FP8DequantPipe: process_meta
# ===========================================================================

class TestFP8DequantPipeMeta:

    def _metas(self, spec: list[tuple[str, torch.dtype, tuple]]):
        return [TensorMeta(k, d, torch.Size(s)) for k, d, s in spec]

    def test_drops_scale_inv_keys(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("w.weight",           torch.float32, (4, 4)),
            ("w.weight_scale_inv", torch.float32, (1, 1)),
        ])
        # Fake w.weight as FP8 by adding it to _fp8_weight_keys via scale presence
        out = list(pipe.process_meta(iter(metas)))
        keys = [m.key for m in out]
        assert "w.weight_scale_inv" not in keys

    def test_drops_weight_scale_keys(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("w.weight",       torch.float32, (4, 4)),
            ("w.weight_scale", torch.float32, (1, 1)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        assert all("_scale" not in m.key for m in out)

    def test_fp8_weight_dtype_updated(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        if not HAS_FP8:
            pytest.skip("FP8 dtype not available; testing via scale-key detection instead")
        metas = self._metas([
            ("layer.weight",           FP8_DTYPE, (256, 128)),
            ("layer.weight_scale_inv", torch.float32, (2, 1)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        weight_metas = [m for m in out if m.key == "layer.weight"]
        assert len(weight_metas) == 1
        assert weight_metas[0].dtype == torch.bfloat16

    def test_weight_shape_preserved(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("layer.weight",           torch.float32, (200, 100)),
            ("layer.weight_scale_inv", torch.float32, (2, 1)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        weight_metas = [m for m in out if m.key == "layer.weight"]
        assert weight_metas[0].shape == torch.Size((200, 100))

    def test_non_fp8_tensors_pass_through_unchanged(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("model.norm.weight", torch.float32, (128,)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        assert len(out) == 1
        assert out[0].dtype == torch.float32  # unchanged

    def test_output_key_count_correct(self):
        """One FP8 weight + one scale_inv + two plain = 3 output metas."""
        pipe = FP8DequantPipe(torch.bfloat16)
        metas = self._metas([
            ("a.weight",           torch.float32, (128, 128)),
            ("a.weight_scale_inv", torch.float32, (1, 1)),
            ("b.weight",           torch.float32, (64,)),
            ("c.weight",           torch.float32, (64,)),
        ])
        out = list(pipe.process_meta(iter(metas)))
        assert len(out) == 3  # a.weight, b.weight, c.weight (no scale_inv)


# ===========================================================================
# FP8DequantPipe: process (Phase 2)
# ===========================================================================

class TestFP8DequantPipeProcess:
    """
    Tests for streaming dequantisation.

    We fake the FP8 dtype situation by injecting float32 tensors directly
    into process() while pre-populating _fp8_weight_keys on the pipe.
    This lets us test the buffering and ordering logic without needing a
    real FP8-capable PyTorch build.
    """

    def _pipe_with_fp8_key(self, key: str) -> FP8DequantPipe:
        """Return a pipe pre-configured to treat *key* as an FP8 weight."""
        import model_pipe.utils.fp8 as fp8_mod
        pipe = FP8DequantPipe(torch.float32)
        pipe._fp8_weight_keys = {key}
        pipe._use_inv_scale = {key: True}
        # Monkey-patch is_fp8_dtype to accept float32 so process() can call dequant
        fp8_mod._FP8_DTYPES.add(torch.float32)
        return pipe

    def _teardown_fp8_patch(self):
        import model_pipe.utils.fp8 as fp8_mod
        fp8_mod._FP8_DTYPES.discard(torch.float32)

    def test_weight_before_scale(self):
        """Weight arrives first, scale second — should dequantise correctly."""
        key = "layer.weight"
        pipe = self._pipe_with_fp8_key(key)
        scale_key = scale_inv_key_for(key)

        w = torch.full((128, 128), 2.0)
        s = torch.full((1, 1), 3.0)

        records = [
            TensorRecord(key=key, tensor=w),
            TensorRecord(key=scale_key, tensor=s),
        ]
        try:
            out = list(pipe.process(iter(records)))
        finally:
            self._teardown_fp8_patch()

        assert len(out) == 1
        assert out[0].key == key
        assert torch.allclose(out[0].tensor, torch.full((128, 128), 6.0), atol=1e-4)

    def test_scale_before_weight(self):
        """Scale arrives first, weight second — should dequantise correctly."""
        key = "layer.weight"
        pipe = self._pipe_with_fp8_key(key)
        scale_key = scale_inv_key_for(key)

        w = torch.full((128, 128), 2.0)
        s = torch.full((1, 1), 3.0)

        records = [
            TensorRecord(key=scale_key, tensor=s),
            TensorRecord(key=key, tensor=w),
        ]
        try:
            out = list(pipe.process(iter(records)))
        finally:
            self._teardown_fp8_patch()

        assert len(out) == 1
        assert out[0].key == key
        assert torch.allclose(out[0].tensor, torch.full((128, 128), 6.0), atol=1e-4)

    def test_non_fp8_tensor_passes_through(self):
        """Non-FP8 keys should be yielded unchanged."""
        pipe = FP8DequantPipe(torch.bfloat16)
        pipe._fp8_weight_keys = set()

        norm = torch.randn(64)
        records = [TensorRecord(key="model.norm.weight", tensor=norm)]
        out = list(pipe.process(iter(records)))

        assert len(out) == 1
        assert out[0].key == "model.norm.weight"
        assert torch.allclose(out[0].tensor, norm)

    def test_multiple_fp8_pairs_interleaved(self):
        """Two FP8 weights with their scales interleaved."""
        import model_pipe.utils.fp8 as fp8_mod
        fp8_mod._FP8_DTYPES.add(torch.float32)
        try:
            pipe = FP8DequantPipe(torch.float32)
            pipe._fp8_weight_keys = {"w0", "w1"}
            pipe._use_inv_scale = {"w0": True, "w1": True}

            records = [
                TensorRecord("w0",               torch.full((128, 128), 1.0)),
                TensorRecord("w1",               torch.full((128, 128), 2.0)),
                TensorRecord("w0_scale_inv",     torch.full((1, 1), 2.0)),
                TensorRecord("w1_scale_inv",     torch.full((1, 1), 3.0)),
            ]
            out = list(pipe.process(iter(records)))
        finally:
            fp8_mod._FP8_DTYPES.discard(torch.float32)

        assert len(out) == 2
        result = {r.key: r.tensor for r in out}
        assert torch.allclose(result["w0"], torch.full((128, 128), 2.0), atol=1e-4)
        assert torch.allclose(result["w1"], torch.full((128, 128), 6.0), atol=1e-4)

    def test_scale_only_no_weight_emits_warning(self, caplog):
        """A scale that arrives without a matching weight should warn."""
        import logging
        import model_pipe.utils.fp8 as fp8_mod
        fp8_mod._FP8_DTYPES.add(torch.float32)
        try:
            pipe = FP8DequantPipe(torch.float32)
            pipe._fp8_weight_keys = {"orphan"}
            pipe._use_inv_scale = {"orphan": True}

            records = [TensorRecord("orphan_scale_inv", torch.ones(1, 1))]
            with caplog.at_level(logging.WARNING):
                list(pipe.process(iter(records)))
        finally:
            fp8_mod._FP8_DTYPES.discard(torch.float32)

        assert any("scale" in msg.lower() for msg in caplog.messages)


# ===========================================================================
# FP8DequantPipe: full pipeline integration
# ===========================================================================

class TestFP8DequantPipeIntegration:

    def test_passthrough_non_fp8_model(self, tmp_path):
        """A model with no FP8 weights should pass through unchanged."""
        tensors = {
            "model.norm.weight":   torch.randn(64),
            "model.embed_tokens.weight": torch.randn(128, 64),
        }
        _save(tensors, tmp_path / "model.safetensors")
        out = tmp_path / "out.safetensors"

        pipe = FP8DequantPipe(torch.bfloat16)
        Pipeline(
            SafetensorsReader(tmp_path / "model.safetensors"),
            pipe,
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert set(result.keys()) == set(tensors.keys())
        for k, v in tensors.items():
            assert torch.allclose(result[k], v)

    def test_scale_keys_absent_from_output(self, tmp_path):
        """weight_scale_inv tensors must not appear in the output file."""
        model_path, ground_truth = _make_fp8_model(tmp_path)
        out = tmp_path / "dequant.safetensors"

        pipe = FP8DequantPipe(torch.float32)
        Pipeline(SafetensorsReader(model_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        for key in result:
            assert not is_scale_key(key), f"Scale key leaked into output: {key}"

    def test_dequantised_weight_correct_dtype(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path)
        out = tmp_path / "dequant_bf16.safetensors"

        pipe = FP8DequantPipe(torch.bfloat16)
        Pipeline(SafetensorsReader(model_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        assert result["model.layers.0.mlp.gate_proj.weight"].dtype == torch.bfloat16

    def test_dequantised_weight_shape_preserved(self, tmp_path):
        out_f, in_f = 256, 128
        model_path, _ = _make_fp8_model(tmp_path, out_f=out_f, in_f=in_f)
        out = tmp_path / "dequant.safetensors"

        pipe = FP8DequantPipe(torch.float32)
        Pipeline(SafetensorsReader(model_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        assert result["model.layers.0.mlp.gate_proj.weight"].shape == (out_f, in_f)

    def test_dequantised_values_match_ground_truth(self, tmp_path):
        """Reconstructed values must match our reference dequantisation."""
        model_path, ground_truth = _make_fp8_model(tmp_path, out_f=256, in_f=128)
        out = tmp_path / "dequant.safetensors"

        pipe = FP8DequantPipe(torch.float32)
        Pipeline(SafetensorsReader(model_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        w_key = "model.layers.0.mlp.gate_proj.weight"
        assert torch.allclose(
            result[w_key].float(),
            ground_truth[w_key].float(),
            atol=1e-4,
        ), "Dequantised values don't match ground truth"

    def test_plain_tensors_unchanged_after_dequant(self, tmp_path):
        """model.norm.weight (non-FP8) must survive the dequant pipe unchanged."""
        model_path, ground_truth = _make_fp8_model(tmp_path)
        out = tmp_path / "dequant.safetensors"

        pipe = FP8DequantPipe(torch.bfloat16)
        Pipeline(SafetensorsReader(model_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        assert torch.allclose(result["model.norm.weight"], ground_truth["model.norm.weight"])

    def test_dry_run_passes(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path)
        w = NullWriter()

        pipe = FP8DequantPipe(torch.bfloat16)
        Pipeline(SafetensorsReader(model_path), pipe, w).run(show_progress=False)
        assert w.report.ok, w.report.summary()

    def test_dequant_then_lora_merge(self, tmp_path):
        """FP8DequantPipe | LoRAMergePipe — full composition."""
        model_path, _ = _make_fp8_model(tmp_path, out_f=256, in_f=128)
        adapter_path, _ = _make_lora(tmp_path, out_f=256, in_f=128)
        out = tmp_path / "merged.safetensors"

        pipe = FP8DequantPipe(torch.float32) | LoRAMergePipe(adapter_path)
        Pipeline(SafetensorsReader(model_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        # All base keys (minus scale tensors) must be present
        assert "model.layers.0.mlp.gate_proj.weight" in result
        assert "model.norm.weight" in result
        # Scale key must not be in output
        assert "model.layers.0.mlp.gate_proj.weight_scale_inv" not in result

    def test_dequant_with_sharded_writer(self, tmp_path):
        """FP8DequantPipe → ShardedWriter round-trip."""
        model_path, ground_truth = _make_fp8_model(tmp_path, out_f=256, in_f=128)
        out_dir = tmp_path / "sharded_out"

        pipe = FP8DequantPipe(torch.float32)
        Pipeline(
            SafetensorsReader(model_path),
            pipe,
            ShardedWriter(out_dir, max_shard_bytes=4096),
        ).run(show_progress=False)

        reader = ShardedSafetensorsReader.from_path(out_dir)
        result = {r.key: r.tensor for r in reader.iter_records()}

        assert not any(is_scale_key(k) for k in result)
        w_key = "model.layers.0.mlp.gate_proj.weight"
        assert torch.allclose(
            result[w_key].float(),
            ground_truth[w_key].float(),
            atol=1e-4,
        )

    def test_repr(self):
        pipe = FP8DequantPipe(torch.bfloat16)
        r = repr(pipe)
        assert "FP8DequantPipe" in r
        assert "bfloat16" in r

    def test_repr_after_meta_scan(self, tmp_path):
        model_path, _ = _make_fp8_model(tmp_path)
        pipe = FP8DequantPipe(torch.bfloat16)
        list(pipe.process_meta(SafetensorsReader(model_path).iter_meta()))
        r = repr(pipe)
        assert "fp8 keys" in r

    def test_compound_repr(self):
        pipe = FP8DequantPipe(torch.bfloat16) | DTypeCastPipe(torch.float16)
        r = repr(pipe)
        assert "FP8DequantPipe" in r
        assert "DTypeCastPipe" in r

    @skip_no_fp8
    def test_real_fp8_dtype_detected(self, tmp_path):
        """If a tensor actually has dtype float8_e4m3fn, it should be flagged."""
        from model_pipe.utils.fp8 import is_fp8_dtype
        assert is_fp8_dtype(torch.float8_e4m3fn)


# ===========================================================================
# CLI helper: dequant-fp8 command exists and is importable
# ===========================================================================

class TestDequantFP8CLI:

    def test_command_registered(self):
        from model_pipe.cli import cli
        assert "dequant-fp8" in cli.commands

    def test_command_has_dtype_option(self):
        from model_pipe.cli import cli
        cmd = cli.commands["dequant-fp8"]
        param_names = [p.name for p in cmd.params]
        assert "dtype" in param_names

    def test_command_has_merge_lora_option(self):
        from model_pipe.cli import cli
        cmd = cli.commands["dequant-fp8"]
        param_names = [p.name for p in cmd.params]
        assert "lora_adapter" in param_names

    def test_command_has_dry_run_option(self):
        from model_pipe.cli import cli
        cmd = cli.commands["dequant-fp8"]
        param_names = [p.name for p in cmd.params]
        assert "dry_run" in param_names

    def test_command_has_sharded_option(self):
        from model_pipe.cli import cli
        cmd = cli.commands["dequant-fp8"]
        param_names = [p.name for p in cmd.params]
        assert "sharded" in param_names
