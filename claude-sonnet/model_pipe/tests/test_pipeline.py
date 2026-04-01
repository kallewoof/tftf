"""
Tests for model-pipe.

These tests use small in-memory tensors so they run without downloading
any real model.  They verify:
- The two-pass pipeline produces a valid safetensors output.
- PassthroughPipe is an identity.
- DTypeCastPipe changes the dtype in both meta and data passes.
- LoRAMergePipe applies the correct merge formula.
- CompoundPipe (|) chains correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from model_pipe.io.reader import SafetensorsReader
from model_pipe.io.writer import StreamingWriter
from model_pipe.pipeline import Pipeline
from model_pipe.pipes.base import TensorMeta
from model_pipe.pipes.dtype_cast import DTypeCastPipe
from model_pipe.pipes.lora_merge import LoRAMergePipe
from model_pipe.pipes.passthrough import PassthroughPipe
from model_pipe.utils.lora import find_lora_keys, merge_lora


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(tensors: dict[str, torch.Tensor], path: Path) -> None:
    save_file(tensors, str(path))


def _load(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def _make_base(tmp: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 32),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 32),
        "model.embed_tokens.weight": torch.randn(128, 32),
        "model.norm.weight": torch.randn(32),
    }
    path = tmp / "model.safetensors"
    _save(tensors, path)
    return path, tensors


def _make_lora(tmp: Path, rank: int = 4, alpha: float = 8.0) -> tuple[Path, Path]:
    """Create a tiny synthetic LoRA adapter."""
    lora_tensors = {
        # q_proj LoRA
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, 32) * 0.01,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(64, rank),
        # v_proj LoRA
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(rank, 32) * 0.01,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": torch.zeros(64, rank),
    }
    adapter_path = tmp / "adapter_model.safetensors"
    _save(lora_tensors, adapter_path)

    config = {"r": rank, "lora_alpha": alpha, "target_modules": ["q_proj", "v_proj"]}
    config_path = tmp / "adapter_config.json"
    config_path.write_text(__import__("json").dumps(config))

    return adapter_path, config_path


# ---------------------------------------------------------------------------
# Unit tests: streaming writer
# ---------------------------------------------------------------------------


def test_writer_produces_valid_safetensors(tmp_path):
    src, tensors = _make_base(tmp_path)
    out = tmp_path / "out.safetensors"

    reader = SafetensorsReader(src)
    writer = StreamingWriter(out)
    pipe = PassthroughPipe()

    Pipeline(reader, pipe, writer).run(show_progress=False)

    result = _load(out)
    assert set(result.keys()) == set(tensors.keys())
    for k in tensors:
        assert torch.allclose(result[k], tensors[k]), f"Mismatch on {k}"


# ---------------------------------------------------------------------------
# Unit tests: PassthroughPipe
# ---------------------------------------------------------------------------


def test_passthrough_is_identity(tmp_path):
    src, tensors = _make_base(tmp_path)
    out = tmp_path / "pass.safetensors"

    Pipeline(SafetensorsReader(src), PassthroughPipe(), StreamingWriter(out)).run(show_progress=False)

    result = _load(out)
    for k, v in tensors.items():
        assert torch.allclose(result[k], v)


# ---------------------------------------------------------------------------
# Unit tests: DTypeCastPipe
# ---------------------------------------------------------------------------


def test_dtype_cast_changes_dtype(tmp_path):
    src, _ = _make_base(tmp_path)
    out = tmp_path / "cast.safetensors"

    pipe = PassthroughPipe() | DTypeCastPipe(torch.float16)
    Pipeline(SafetensorsReader(src), pipe, StreamingWriter(out)).run(show_progress=False)

    result = _load(out)
    for v in result.values():
        assert v.dtype == torch.float16


def test_dtype_cast_meta_reflects_new_dtype():
    pipe = DTypeCastPipe(torch.float16)
    metas = [TensorMeta(key="w", dtype=torch.float32, shape=torch.Size([4, 4]))]
    out = list(pipe.process_meta(iter(metas)))
    assert out[0].dtype == torch.float16


# ---------------------------------------------------------------------------
# Unit tests: LoRA key mapping
# ---------------------------------------------------------------------------


def test_find_lora_keys_standard_peft():
    adapter_keys = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
    }
    result = find_lora_keys(
        "model.layers.0.self_attn.q_proj.weight",
        adapter_keys,
    )
    assert result is not None
    a_key, b_key, is_emb = result
    assert "lora_A" in a_key
    assert "lora_B" in b_key
    assert not is_emb


def test_find_lora_keys_no_match():
    result = find_lora_keys("model.norm.weight", set())
    assert result is None


# ---------------------------------------------------------------------------
# Unit tests: merge math
# ---------------------------------------------------------------------------


def test_merge_lora_zero_b_is_identity():
    """If lora_B is all zeros, merged weight should equal original."""
    w = torch.randn(8, 4)
    a = torch.randn(2, 4)
    b = torch.zeros(8, 2)
    merged = merge_lora(w, a, b, scale=1.0)
    assert torch.allclose(merged, w)


def test_merge_lora_known_value():
    """Verify the formula: W + scale * B @ A."""
    w = torch.eye(4)
    a = torch.ones(2, 4)
    b = torch.ones(4, 2)
    scale = 0.5
    expected = w + scale * (b @ a)
    merged = merge_lora(w, a, b, scale=scale)
    assert torch.allclose(merged, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Integration test: LoRAMergePipe end-to-end
# ---------------------------------------------------------------------------


def test_lora_merge_pipe_end_to_end(tmp_path):
    base_path, base_tensors = _make_base(tmp_path)
    adapter_path, config_path = _make_lora(tmp_path, rank=4, alpha=8.0)
    out_path = tmp_path / "merged.safetensors"

    pipe = LoRAMergePipe(adapter_path=adapter_path, device="cpu")
    Pipeline(
        SafetensorsReader(base_path),
        pipe,
        StreamingWriter(out_path),
    ).run(show_progress=False)

    result = _load(out_path)

    # All base model keys must be present
    assert set(result.keys()) == set(base_tensors.keys())

    # norm.weight (no LoRA) should be unchanged
    assert torch.allclose(result["model.norm.weight"], base_tensors["model.norm.weight"])

    # embed_tokens.weight (no LoRA) should be unchanged
    assert torch.allclose(result["model.embed_tokens.weight"], base_tensors["model.embed_tokens.weight"])

    # q_proj (has LoRA with zero B) should equal original (B is zeros → delta=0)
    assert torch.allclose(
        result["model.layers.0.self_attn.q_proj.weight"],
        base_tensors["model.layers.0.self_attn.q_proj.weight"],
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Integration test: compound pipe
# ---------------------------------------------------------------------------


def test_compound_pipe_merge_then_cast(tmp_path):
    base_path, _ = _make_base(tmp_path)
    adapter_path, _ = _make_lora(tmp_path)
    out_path = tmp_path / "merged_fp16.safetensors"

    pipe = LoRAMergePipe(adapter_path=adapter_path) | DTypeCastPipe(torch.float16)
    Pipeline(SafetensorsReader(base_path), pipe, StreamingWriter(out_path)).run(show_progress=False)

    result = _load(out_path)
    for v in result.values():
        assert v.dtype == torch.float16


# ---------------------------------------------------------------------------
# Unit tests: Conv2d merge math
# ---------------------------------------------------------------------------


def test_merge_lora_conv2d_zero_b_is_identity():
    """Conv2d: if lora_B is all zeros the merged weight equals the original."""
    w = torch.randn(8, 3, 3, 3)   # (out, in, kH, kW)
    a = torch.randn(2, 3, 3, 3)   # (r, in, kH, kW)
    b = torch.zeros(8, 2, 1, 1)   # (out, r, 1, 1)
    merged = merge_lora(w, a, b, scale=1.0)
    assert merged.shape == w.shape
    assert torch.allclose(merged, w, atol=1e-5)


def test_merge_lora_conv2d_known_value():
    """
    Conv2d: delta = (B_2d @ A_2d).reshape(out, in, kH, kW).

    With a 1×1 kernel, this reduces to the linear case: delta = B @ A
    reshaped appropriately.
    """
    out_c, in_c, r = 4, 3, 2
    w = torch.zeros(out_c, in_c, 1, 1)
    a = torch.ones(r, in_c, 1, 1)       # all-ones
    b = torch.ones(out_c, r, 1, 1)       # all-ones
    scale = 0.5

    merged = merge_lora(w, a, b, scale=scale)

    # Expected: 0 + 0.5 * (B_2d @ A_2d) reshaped
    # B_2d = (4,2), A_2d = (2, 3*1*1) = (2,3)
    # B@A = (4,3) all-twos → delta = (4,3,1,1) of 2.0
    # merged = 0 + 0.5 * 2.0 = 1.0 everywhere
    assert merged.shape == (out_c, in_c, 1, 1)
    assert torch.allclose(merged, torch.ones(out_c, in_c, 1, 1), atol=1e-5)


def test_merge_lora_conv2d_shape_preserved():
    """Output of Conv2d merge must have the same shape as the base weight."""
    w = torch.randn(1152, 3, 14, 14)   # patch_embed.proj.weight shape (approx)
    r = 8
    a = torch.randn(r, 3, 14, 14)
    b = torch.zeros(1152, r, 1, 1)
    merged = merge_lora(w, a, b, scale=1.0)
    assert merged.shape == w.shape


def test_merge_lora_conv2d_matches_peft_formula():
    """
    Verify our formula matches PEFT's conv2d(permuted_A, B).permute(…) formula
    exactly, using a non-trivial example.
    """
    import torch.nn.functional as F

    out_c, in_c, kH, kW, r = 8, 4, 3, 3, 2
    w = torch.zeros(out_c, in_c, kH, kW)
    a = torch.randn(r, in_c, kH, kW)
    b = torch.randn(out_c, r, 1, 1)
    scale = 1.0

    # Our implementation
    ours = merge_lora(w, a, b, scale=scale)

    # PEFT reference formula: conv2d(a_permuted, b).permute(1,0,2,3) + w
    a_f = a.float()
    b_f = b.float()
    peft_delta = F.conv2d(
        a_f.permute(1, 0, 2, 3),  # (in_c, r, kH, kW)
        b_f,                        # (out_c, r, 1, 1)
    ).permute(1, 0, 2, 3)          # → (out_c, in_c, kH, kW)

    peft_merged = (w.float() + scale * peft_delta).to(w.dtype)

    assert torch.allclose(ours, peft_merged, atol=1e-5), \
        f"max diff: {(ours - peft_merged).abs().max().item()}"


def test_merge_lora_unsupported_ndim_raises():
    """Tensors with ndim < 2 must raise ValueError."""
    w = torch.randn(4)   # 1-D — not a valid LoRA target
    a = torch.randn(2, 4)
    b = torch.randn(4, 2)
    with pytest.raises(ValueError):
        merge_lora(w, a, b, scale=1.0)


def test_merge_lora_conv1d_zero_b_is_identity():
    """Conv1d (ndim=3): zero lora_B means no change."""
    w = torch.randn(8, 3, 5)        # (out, in, k)
    a = torch.randn(2, 3, 5)        # (r, in, k)
    b = torch.zeros(8, 2, 1)        # (out, r, 1)
    merged = merge_lora(w, a, b, scale=1.0)
    assert merged.shape == w.shape
    assert torch.allclose(merged, w, atol=1e-5)


def test_merge_lora_conv3d_exact_shape_from_crash():
    """
    The exact shape that was crashing: (1152, 3, 2, 16, 16) — Conv3d used
    in video/vision transformers like Qwen2-VL's temporal patch embedding.
    Zero lora_B → identity merge.
    """
    out_c, in_c, kD, kH, kW, r = 1152, 3, 2, 16, 16, 8
    w = torch.randn(out_c, in_c, kD, kH, kW)
    a = torch.randn(r, in_c, kD, kH, kW)       # (r, in, kD, kH, kW)
    b = torch.zeros(out_c, r, 1, 1, 1)          # (out, r, 1, 1, 1)
    merged = merge_lora(w, a, b, scale=1.0)
    assert merged.shape == w.shape
    assert torch.allclose(merged, w, atol=1e-5)


def test_merge_lora_conv3d_nonzero_delta():
    """Conv3d with a known, verifiable delta."""
    out_c, in_c, kD, kH, kW, r = 4, 2, 1, 1, 1, 2
    w = torch.zeros(out_c, in_c, kD, kH, kW)
    a = torch.ones(r, in_c, kD, kH, kW)         # all-ones, kernel_numel = 2
    b = torch.ones(out_c, r, 1, 1, 1)           # all-ones
    # b2d = (4,2), a2d = (2, 2*1*1*1) = (2,2)
    # b@a = (4,2) all-twos → delta = (4,2,1,1,1) of 2.0
    # merged = 0 + 1.0 * 2.0 = 2.0
    merged = merge_lora(w, a, b, scale=1.0)
    assert torch.allclose(merged, torch.full_like(merged, 2.0), atol=1e-5)


def test_merge_lora_conv_generalises_across_ndims():
    """
    The same flatten-matmul-reshape formula must give consistent results
    for Conv1d, Conv2d, and Conv3d when spatial dims are all 1.
    With all-spatial-1 kernels, every conv reduces to a linear matmul,
    so all three must yield the same numeric output.
    """
    out_c, in_c, r = 4, 3, 2
    w_vals = torch.randn(out_c, in_c)
    a_vals = torch.randn(r, in_c)
    b_vals = torch.randn(out_c, r)

    # Linear reference
    delta_ref = b_vals @ a_vals   # (4, 3)

    for extra_dims in [1, 2, 3]:   # Conv1d, Conv2d, Conv3d
        ones = (1,) * extra_dims
        w = w_vals.reshape(out_c, in_c, *ones)
        a = a_vals.reshape(r, in_c, *ones)
        b = b_vals.reshape(out_c, r, *ones)
        merged = merge_lora(w, a, b, scale=1.0)
        delta = merged - w
        assert torch.allclose(
            delta.reshape(out_c, in_c),
            delta_ref,
            atol=1e-5,
        ), f"Mismatch for ndim={2 + extra_dims}"


# ---------------------------------------------------------------------------
# Integration test: Conv2d LoRA pipeline end-to-end
# ---------------------------------------------------------------------------


def _make_conv2d_model(tmp: Path):
    """Tiny vision model with one Conv2d and one Linear weight."""
    tensors = {
        "model.visual.patch_embed.proj.weight": torch.randn(64, 3, 14, 14),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 32),
        "model.norm.weight": torch.randn(32),
    }
    path = tmp / "model.safetensors"
    _save(tensors, path)
    return path, tensors


def _make_conv2d_lora(tmp: Path, rank: int = 4, alpha: float = 8.0):
    """LoRA adapter targeting both the Conv2d and Linear layers."""
    import json
    lora_tensors = {
        # Conv2d LoRA (for patch_embed.proj) — zeros B → identity merge
        "base_model.model.model.visual.patch_embed.proj.lora_A.weight":
            torch.randn(rank, 3, 14, 14) * 0.01,
        "base_model.model.model.visual.patch_embed.proj.lora_B.weight":
            torch.zeros(64, rank, 1, 1),
        # Linear LoRA (for q_proj) — zeros B → identity merge
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            torch.randn(rank, 32) * 0.01,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
            torch.zeros(64, rank),
    }
    adapter_path = tmp / "adapter_model.safetensors"
    _save(lora_tensors, adapter_path)
    config = {
        "r": rank, "lora_alpha": alpha,
        "target_modules": ["patch_embed.proj", "q_proj"],
    }
    (tmp / "adapter_config.json").write_text(json.dumps(config))
    return adapter_path


def test_conv2d_lora_merge_end_to_end(tmp_path):
    """Merge a LoRA with both Conv2d and Linear targets — zero-B means identity."""
    base_path, base_tensors = _make_conv2d_model(tmp_path)
    adapter_path = _make_conv2d_lora(tmp_path)
    out_path = tmp_path / "merged.safetensors"

    pipe = LoRAMergePipe(adapter_path=adapter_path, device="cpu")
    Pipeline(
        SafetensorsReader(base_path),
        pipe,
        StreamingWriter(out_path),
    ).run(show_progress=False)

    result = _load(out_path)
    assert set(result.keys()) == set(base_tensors.keys())

    # Zero lora_B → merged == original for both weight types
    assert torch.allclose(
        result["model.visual.patch_embed.proj.weight"],
        base_tensors["model.visual.patch_embed.proj.weight"],
        atol=1e-5,
    ), "Conv2d weight changed despite zero lora_B"

    assert torch.allclose(
        result["model.layers.0.self_attn.q_proj.weight"],
        base_tensors["model.layers.0.self_attn.q_proj.weight"],
        atol=1e-5,
    ), "Linear weight changed despite zero lora_B"

    assert torch.allclose(
        result["model.norm.weight"],
        base_tensors["model.norm.weight"],
    ), "norm.weight (no LoRA) should be unchanged"


def test_conv2d_lora_merge_nonzero_delta(tmp_path):
    """When lora_B is non-zero the Conv2d weight must actually change."""
    import json
    # Simple 1×1 conv so we can verify the math easily
    base = {"vis.proj.weight": torch.zeros(4, 2, 1, 1)}
    _save(base, tmp_path / "model.safetensors")

    rank = 2
    lora = {
        "base_model.model.vis.proj.lora_A.weight": torch.ones(rank, 2, 1, 1),
        "base_model.model.vis.proj.lora_B.weight": torch.ones(4, rank, 1, 1),
    }
    _save(lora, tmp_path / "adapter_model.safetensors")
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": float(rank), "target_modules": ["proj"]})
    )

    out = tmp_path / "merged.safetensors"
    Pipeline(
        SafetensorsReader(tmp_path / "model.safetensors"),
        LoRAMergePipe(tmp_path / "adapter_model.safetensors"),
        StreamingWriter(out),
    ).run(show_progress=False)

    result = _load(out)
    merged = result["vis.proj.weight"]

    # alpha/r = 1.0; delta = B_2d @ A_2d reshaped
    # B_2d = ones(4,2), A_2d = ones(2, 2*1*1) = ones(2,2)
    # B@A = 2.0 everywhere → delta = (4,2,1,1) of 2.0
    # merged = 0 + 1.0 * 2.0 = 2.0
    assert torch.allclose(merged, torch.full((4, 2, 1, 1), 2.0), atol=1e-5)
