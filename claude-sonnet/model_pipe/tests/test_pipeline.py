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

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from model_pipe.io.reader import SafetensorsReader
from model_pipe.io.writer import StreamingWriter
from model_pipe.pipeline import Pipeline
from model_pipe.pipes.base import TensorRecord, TensorMeta
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
    import json

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
