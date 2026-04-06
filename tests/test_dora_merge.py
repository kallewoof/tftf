"""
Tests for DoRA / QDoRA adapter merging.

DoRA formula (per PEFT dora.py):
    W_merged   = W + scale * delta
    weight_norm = ||W_merged|| per output channel (row-wise L2)
    W_out      = (m / weight_norm)[:, None] * W_merged

QDoRA uses the identical merge formula; the quantisation was only active
during training.  So both variants are exercised by the same test helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tftf.io.reader import SafetensorsReader
from tftf.io.writer import StreamingWriter
from tftf.pipeline import Pipeline
from tftf.pipes.lora_merge import LoRAMergePipe
from tftf.utils.lora import find_magnitude_key, merge_dora


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RANK = 4
OUT = 64
IN = 32


def _save(tensors: dict[str, torch.Tensor], path: Path) -> None:
    save_file(tensors, str(path))


def _load(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


def _row_norms(w: torch.Tensor) -> torch.Tensor:
    """Per-row L2 norms — matches PEFT's DoRA weight_norm initialisation."""
    return w.norm(p=2, dim=1)


def _make_base(tmp: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    tensors = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(OUT, IN),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(OUT, IN),
        "model.norm.weight": torch.randn(IN),
    }
    path = tmp / "model.safetensors"
    _save(tensors, path)
    return path, tensors


def _make_dora_adapter(
    tmp: Path,
    base_tensors: dict[str, torch.Tensor],
    *,
    rank: int = RANK,
    alpha: float = float(RANK),
    zero_b: bool = True,
    target_modules: Optional[list[str]] = None,
    adapter_name: str = "default",
) -> tuple[Path, Path]:
    """
    Write a synthetic DoRA adapter (adapter_model.safetensors + adapter_config.json).

    Magnitude vectors are initialised to ||W + scale * B @ A||_r (using the
    provided *base_tensors*) so that when lora_B is zero the merge is
    numerically identity.
    """
    q_weight = base_tensors["model.layers.0.self_attn.q_proj.weight"]
    v_weight = base_tensors["model.layers.0.self_attn.v_proj.weight"]

    lora_a_q = torch.randn(rank, IN) * 0.01
    lora_b_q = torch.zeros(OUT, rank) if zero_b else torch.ones(OUT, rank) * 0.1
    lora_a_v = torch.randn(rank, IN) * 0.01
    lora_b_v = torch.zeros(OUT, rank) if zero_b else torch.ones(OUT, rank) * 0.1

    # PEFT initialises m = ||W + scale * B @ A||_r; with zero_b → m = ||W||_r
    scale = alpha / rank
    q_merged_init = q_weight + scale * lora_b_q @ lora_a_q
    v_merged_init = v_weight + scale * lora_b_v @ lora_a_v
    mag_q = _row_norms(q_merged_init)
    mag_v = _row_norms(v_merged_init)

    prefix = "base_model.model.model.layers.0.self_attn"
    lora_tensors = {
        f"{prefix}.q_proj.lora_A.{adapter_name}.weight": lora_a_q,
        f"{prefix}.q_proj.lora_B.{adapter_name}.weight": lora_b_q,
        f"{prefix}.q_proj.lora_magnitude_vector.{adapter_name}.weight": mag_q,
        f"{prefix}.v_proj.lora_A.{adapter_name}.weight": lora_a_v,
        f"{prefix}.v_proj.lora_B.{adapter_name}.weight": lora_b_v,
        f"{prefix}.v_proj.lora_magnitude_vector.{adapter_name}.weight": mag_v,
    }
    adapter_path = tmp / "adapter_model.safetensors"
    _save(lora_tensors, adapter_path)

    cfg: dict = {"r": rank, "lora_alpha": alpha}
    if target_modules is not None:
        cfg["target_modules"] = target_modules
    config_path = tmp / "adapter_config.json"
    config_path.write_text(json.dumps(cfg))

    return adapter_path, config_path


# ---------------------------------------------------------------------------
# Unit tests for merge_dora()
# ---------------------------------------------------------------------------


class TestMergeDoraUnit:

    def test_zero_b_identity(self):
        """With zero lora_B and magnitude = ||W||_r, the output equals W."""
        w = torch.randn(OUT, IN)
        a = torch.randn(RANK, IN) * 0.01
        b = torch.zeros(OUT, RANK)
        scale = 1.0

        # Initialise magnitude as PEFT does: row-norms of W + scale * B @ A
        m = _row_norms(w + scale * b @ a)

        result = merge_dora(w, a, b, m, scale, is_embedding=False)
        assert torch.allclose(result, w, atol=1e-5), "zero lora_B should be identity"

    def test_nonzero_b_changes_weight(self):
        """With non-zero lora_B, the merged weight must differ from the base."""
        w = torch.randn(OUT, IN)
        a = torch.randn(RANK, IN) * 0.01
        b = torch.ones(OUT, RANK) * 0.5
        scale = 1.0
        m = _row_norms(w + scale * b @ a)

        result = merge_dora(w, a, b, m, scale)
        assert not torch.allclose(result, w), "non-zero lora_B must change the weight"

    def test_output_shape_matches_input(self):
        w = torch.randn(OUT, IN)
        a = torch.randn(RANK, IN)
        b = torch.randn(OUT, RANK)
        m = torch.ones(OUT)

        result = merge_dora(w, a, b, m, scale=1.0)
        assert result.shape == w.shape

    def test_dtype_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            w = torch.randn(OUT, IN).to(dtype)
            a = torch.randn(RANK, IN).to(dtype)
            b = torch.randn(OUT, RANK).to(dtype)
            m = torch.ones(OUT).to(dtype)
            result = merge_dora(w, a, b, m, scale=1.0)
            assert result.dtype == dtype, f"dtype {dtype} not preserved"

    def test_magnitude_normalization_effect(self):
        """Doubling the magnitude vector should double the output norms."""
        w = torch.randn(OUT, IN)
        a = torch.zeros(RANK, IN)
        b = torch.zeros(OUT, RANK)
        m = torch.ones(OUT)
        scale = 0.0  # no LoRA delta, pure magnitude scaling

        r1 = merge_dora(w, a, b, m, scale)
        r2 = merge_dora(w, a, b, m * 2, scale)

        # With zero delta: W_merged = W, weight_norm = ||W||_r
        # r1 = (1 / ||W||_r)[:, None] * W
        # r2 = (2 / ||W||_r)[:, None] * W = 2 * r1
        assert torch.allclose(r2, r1 * 2, atol=1e-5)

    def test_ndim_lt_2_raises(self):
        with pytest.raises(ValueError, match="ndim=1"):
            merge_dora(torch.randn(4), torch.randn(2, 4), torch.randn(4, 2), torch.ones(4), 1.0)


# ---------------------------------------------------------------------------
# Unit tests for find_magnitude_key()
# ---------------------------------------------------------------------------


class TestFindMagnitudeKey:

    def _make_keys(self, stem: str, adapter: str = "default") -> set[str]:
        return {
            f"base_model.model.{stem}.lora_A.{adapter}.weight",
            f"base_model.model.{stem}.lora_B.{adapter}.weight",
            f"base_model.model.{stem}.lora_magnitude_vector.{adapter}.weight",
        }

    def test_finds_with_adapter_name(self):
        stem = "model.layers.0.self_attn.q_proj"
        keys = self._make_keys(stem, "default")
        found = find_magnitude_key(f"{stem}.weight", keys, "default")
        assert found is not None
        assert "lora_magnitude_vector" in found

    def test_returns_none_for_plain_lora(self):
        keys = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
        }
        found = find_magnitude_key(
            "model.layers.0.self_attn.q_proj.weight", keys, "default"
        )
        assert found is None

    def test_finds_without_adapter_name(self):
        stem = "model.layers.0.self_attn.q_proj"
        keys = {
            f"base_model.model.{stem}.lora_magnitude_vector.weight",
        }
        found = find_magnitude_key(f"{stem}.weight", keys, "default")
        assert found == f"base_model.model.{stem}.lora_magnitude_vector.weight"

    def test_finds_bare_key(self):
        stem = "model.layers.0.self_attn.q_proj"
        keys = {f"base_model.model.{stem}.lora_magnitude_vector"}
        found = find_magnitude_key(f"{stem}.weight", keys, "default")
        assert found == f"base_model.model.{stem}.lora_magnitude_vector"


# ---------------------------------------------------------------------------
# Integration: LoRAMergePipe with DoRA adapter
# ---------------------------------------------------------------------------


class TestDoRAMergePipeIntegration:

    def test_zero_b_identity(self, tmp_path):
        """Pipeline with zero lora_B and proper magnitude → output ≈ base."""
        base_path, base_tensors = _make_base(tmp_path)
        adapter_path, config_path = _make_dora_adapter(tmp_path, base_tensors, zero_b=True)
        out = tmp_path / "out.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=config_path),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert set(result.keys()) == set(base_tensors.keys())
        for k in ("model.layers.0.self_attn.q_proj.weight",
                  "model.layers.0.self_attn.v_proj.weight"):
            assert torch.allclose(result[k], base_tensors[k], atol=1e-4), \
                f"{k}: zero lora_B DoRA should be near-identity"

    def test_nonzero_b_changes_weight(self, tmp_path):
        """Non-zero lora_B must produce a different merged weight."""
        base_path, base_tensors = _make_base(tmp_path)
        adapter_path, config_path = _make_dora_adapter(tmp_path, base_tensors, zero_b=False)
        out = tmp_path / "out.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=config_path),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert not torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            base_tensors["model.layers.0.self_attn.q_proj.weight"],
        )

    def test_non_lora_keys_unchanged(self, tmp_path):
        """Keys with no LoRA pair (e.g. norm) must pass through unchanged."""
        base_path, base_tensors = _make_base(tmp_path)
        adapter_path, config_path = _make_dora_adapter(tmp_path, base_tensors, zero_b=False)
        out = tmp_path / "out.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=config_path),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert torch.allclose(result["model.norm.weight"], base_tensors["model.norm.weight"])

    def test_target_modules_respected(self, tmp_path):
        """target_modules limits which weights are DoRA-merged."""
        base_path, base_tensors = _make_base(tmp_path)
        adapter_path, config_path = _make_dora_adapter(
            tmp_path, base_tensors, zero_b=False, target_modules=["q_proj"]
        )
        out = tmp_path / "out.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=config_path),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        # q_proj in target_modules → changed
        assert not torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            base_tensors["model.layers.0.self_attn.q_proj.weight"],
        )
        # v_proj not in target_modules → unchanged
        assert torch.allclose(
            result["model.layers.0.self_attn.v_proj.weight"],
            base_tensors["model.layers.0.self_attn.v_proj.weight"],
        )

    def test_plain_lora_unaffected(self, tmp_path):
        """
        An adapter without magnitude vectors must still use the standard
        LoRA merge path (regression guard).
        """
        tensors = {"model.layers.0.self_attn.q_proj.weight": torch.randn(OUT, IN)}
        base_path = tmp_path / "model.safetensors"
        _save(tensors, base_path)

        lora_tensors = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
                torch.zeros(RANK, IN),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
                torch.zeros(OUT, RANK),
        }
        adapter_path = tmp_path / "adapter_model.safetensors"
        _save(lora_tensors, adapter_path)

        config_path = tmp_path / "adapter_config.json"
        config_path.write_text(json.dumps({"r": RANK, "lora_alpha": float(RANK)}))

        out = tmp_path / "out.safetensors"
        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=config_path),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            tensors["model.layers.0.self_attn.q_proj.weight"],
        )


# mypy / type-checker helper — Optional used in helper signature
from typing import Optional  # noqa: E402
