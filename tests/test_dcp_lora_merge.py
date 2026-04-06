"""Tests for DCPLoRAMergePipe — PyTorch DCP checkpoint LoRA merging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tftf.io.reader import SafetensorsReader
from tftf.io.writer import StreamingWriter
from tftf.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(tensors: dict[str, torch.Tensor], path: Path) -> None:
    save_file(tensors, str(path))


def _load(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


_BASE_TENSORS = {
    "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 32),
    "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 32),
    "model.layers.0.mlp.gate_proj.weight":    torch.randn(64, 32),
    "model.embed_tokens.weight":               torch.randn(128, 32),
    "model.norm.weight":                       torch.randn(32),
}


def _make_base_single(tmp: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    path = tmp / "model.safetensors"
    _save(_BASE_TENSORS, path)
    return path, _BASE_TENSORS


def _make_dcp_lora(
    tmp: Path,
    rank: int = 4,
    alpha: float = 8.0,
    zero_b: bool = True,
    target_modules: list[str] | None = None,
) -> Path:
    """
    Save a synthetic LoRA adapter as a PyTorch DCP checkpoint.
    Keys follow the axolotl/FSDP convention: nested under "model" wrapper,
    using base_model.model.* prefixes.
    """
    import torch.distributed.checkpoint as dist_cp

    lora_tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(rank, 32) * 0.01,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": (
            torch.zeros(64, rank) if zero_b else torch.ones(64, rank)
        ),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": torch.randn(rank, 32) * 0.01,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": (
            torch.zeros(64, rank) if zero_b else torch.ones(64, rank)
        ),
    }

    dcp_dir = tmp / "pytorch_model_fsdp_0"
    dcp_dir.mkdir()
    dist_cp.save(
        state_dict={"model": lora_tensors},
        storage_writer=dist_cp.FileSystemWriter(dcp_dir),
        no_dist=True,
    )

    if target_modules is not None:
        cfg = {"r": rank, "lora_alpha": alpha, "target_modules": target_modules}
        (tmp / "adapter_config.json").write_text(json.dumps(cfg))

    return dcp_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDCPLoRAMergePipe:

    def test_load_dcp_state_dict_round_trip(self, tmp_path):
        """load_dcp_state_dict reassembles the tensors saved by dist_cp.save."""
        from tftf.utils.dcp import load_dcp_state_dict

        dcp_dir = _make_dcp_lora(tmp_path, rank=4)
        weights = load_dcp_state_dict(dcp_dir)

        assert "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight" in weights
        assert weights["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"].shape == (4, 32)

    def test_missing_metadata_raises(self, tmp_path):
        from tftf.utils.dcp import load_dcp_state_dict

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match=".metadata"):
            load_dcp_state_dict(empty_dir)

    def test_zero_b_is_identity(self, tmp_path):
        """With zero lora_B the merge should not change the base weights."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_lora(tmp_path, zero_b=True)
        out = tmp_path / "merged.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            DCPLoRAMergePipe(checkpoint_dir=dcp_dir),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert set(result.keys()) == set(base_tensors.keys())
        for k in ("model.layers.0.self_attn.q_proj.weight",
                  "model.layers.0.self_attn.v_proj.weight"):
            assert torch.allclose(result[k], base_tensors[k], atol=1e-5), \
                f"{k} changed despite zero lora_B"

    def test_nonzero_b_changes_weight(self, tmp_path):
        """With non-zero lora_B the merged weights must differ from the base."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_lora(tmp_path, zero_b=False)
        out = tmp_path / "merged.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            DCPLoRAMergePipe(checkpoint_dir=dcp_dir),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert not torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            base_tensors["model.layers.0.self_attn.q_proj.weight"],
        ), "q_proj should have changed"

    def test_non_lora_keys_pass_through(self, tmp_path):
        """Keys with no matching LoRA pair in the DCP must be unchanged."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_lora(tmp_path, zero_b=False)
        out = tmp_path / "merged.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            DCPLoRAMergePipe(checkpoint_dir=dcp_dir),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert torch.allclose(result["model.norm.weight"], base_tensors["model.norm.weight"])
        assert torch.allclose(result["model.embed_tokens.weight"], base_tensors["model.embed_tokens.weight"])

    def test_target_modules_respected(self, tmp_path):
        """target_modules in adapter_config.json restricts which keys are merged."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_lora(tmp_path, zero_b=False, target_modules=["q_proj"])
        out = tmp_path / "merged.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            DCPLoRAMergePipe(checkpoint_dir=dcp_dir),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        # q_proj is in target_modules → must change
        assert not torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            base_tensors["model.layers.0.self_attn.q_proj.weight"],
        ), "q_proj should have been merged"
        # v_proj is NOT in target_modules → must be unchanged
        assert torch.allclose(
            result["model.layers.0.self_attn.v_proj.weight"],
            base_tensors["model.layers.0.self_attn.v_proj.weight"],
        ), "v_proj should NOT have been merged"

    def test_config_path_explicit(self, tmp_path):
        """config_path= overrides the auto-detected location."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        _make_base_single(tmp_path)
        dcp_dir = _make_dcp_lora(tmp_path, rank=4)

        cfg_dir = tmp_path / "explicit_config"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "adapter_config.json"
        cfg_file.write_text(json.dumps({"r": 4, "lora_alpha": 4.0, "target_modules": []}))

        pipe = DCPLoRAMergePipe(checkpoint_dir=dcp_dir, config_path=cfg_file)
        pipe.setup()
        assert pipe._config.r == 4
        assert pipe._config.lora_alpha == 4.0
        pipe.teardown()

    def test_repr(self, tmp_path):
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        dcp_dir = _make_dcp_lora(tmp_path)
        pipe = DCPLoRAMergePipe(checkpoint_dir=dcp_dir)
        r = repr(pipe)
        assert "DCPLoRAMergePipe" in r
        assert "pytorch_model_fsdp_0" in r
