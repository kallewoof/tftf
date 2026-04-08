"""Tests for DCPLoRAMergePipe — PyTorch DCP checkpoint LoRA and DoRA merging, and CLI checkpoint resolution."""

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


def _make_dcp_dora(
    tmp: Path,
    base_tensors: dict[str, torch.Tensor],
    rank: int = 4,
    alpha: float = 8.0,
    zero_b: bool = True,
    target_modules: list[str] | None = None,
    adapter_name: str = "default",
) -> Path:
    """
    Save a synthetic DoRA adapter as a PyTorch DCP checkpoint.

    Magnitude vectors are initialised from *base_tensors* (row-wise L2 norms
    of W + scale·B@A) so that when lora_B is zero the merge is identity.
    """
    import torch.distributed.checkpoint as dist_cp

    prefix = "base_model.model.model.layers.0.self_attn"
    out, in_ = 64, 32

    lora_a_q = torch.randn(rank, in_) * 0.01
    lora_b_q = torch.zeros(out, rank) if zero_b else torch.ones(out, rank) * 0.1
    lora_a_v = torch.randn(rank, in_) * 0.01
    lora_b_v = torch.zeros(out, rank) if zero_b else torch.ones(out, rank) * 0.1

    scale = alpha / rank
    mag_q = (base_tensors["model.layers.0.self_attn.q_proj.weight"] + scale * lora_b_q @ lora_a_q).norm(p=2, dim=1)
    mag_v = (base_tensors["model.layers.0.self_attn.v_proj.weight"] + scale * lora_b_v @ lora_a_v).norm(p=2, dim=1)

    dora_tensors = {
        f"{prefix}.q_proj.lora_A.{adapter_name}.weight": lora_a_q,
        f"{prefix}.q_proj.lora_B.{adapter_name}.weight": lora_b_q,
        f"{prefix}.q_proj.lora_magnitude_vector.{adapter_name}.weight": mag_q,
        f"{prefix}.v_proj.lora_A.{adapter_name}.weight": lora_a_v,
        f"{prefix}.v_proj.lora_B.{adapter_name}.weight": lora_b_v,
        f"{prefix}.v_proj.lora_magnitude_vector.{adapter_name}.weight": mag_v,
    }

    dcp_dir = tmp / "pytorch_model_fsdp_0"
    dcp_dir.mkdir(exist_ok=True)
    dist_cp.save(
        state_dict={"model": dora_tensors},
        storage_writer=dist_cp.FileSystemWriter(dcp_dir),
        no_dist=True,
    )

    cfg: dict = {"r": rank, "lora_alpha": alpha}
    if target_modules is not None:
        cfg["target_modules"] = target_modules
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

    def test_setup_raises_type_error_when_dcp_returns_dict_values(self, tmp_path, monkeypatch):
        """setup() raises TypeError with a helpful message when load_dcp_state_dict
        returns dict values instead of tensors (e.g. an optimizer DCP was loaded).

        This guards against the 'dict object has no attribute to' AttributeError
        that occurs when the resolver accidentally picks optimizer_0 over the
        model DCP directory.
        """
        import tftf.pipes.dcp_lora_merge as mod
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        dcp_dir = _make_dcp_lora(tmp_path)

        # Simulate optimizer DCP structure after two rounds of single-key unwrapping:
        # {"optimizer": {"state": {param: {exp_avg, step}}}} → {param: {exp_avg, step}}
        bad_weights = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": {
                "exp_avg": torch.zeros(4, 32),
                "step": torch.tensor(60),
            }
        }
        monkeypatch.setattr(mod, "load_dcp_state_dict", lambda _: bad_weights)

        pipe = DCPLoRAMergePipe(checkpoint_dir=dcp_dir)
        with pytest.raises(TypeError, match="optimizer state dict"):
            pipe.setup()


# ---------------------------------------------------------------------------
# DoRA via DCP
# ---------------------------------------------------------------------------


class TestDCPDoRAMergePipe:
    """DoRA magnitude vectors stored inside a DCP checkpoint are merged correctly."""

    def test_magnitude_keys_loaded_into_adapter_key_set(self, tmp_path):
        """After setup(), _adapter_key_set must include the magnitude vector keys."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        _, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_dora(tmp_path, base_tensors)

        pipe = DCPLoRAMergePipe(checkpoint_dir=dcp_dir)
        pipe.setup()
        mag_keys = [k for k in pipe._adapter_key_set if "lora_magnitude_vector" in k]
        assert len(mag_keys) == 2, f"Expected 2 magnitude keys, got: {mag_keys}"
        pipe.teardown()

    def test_zero_b_is_identity(self, tmp_path):
        """DoRA with zero lora_B and magnitude = ||W||_r must leave weights unchanged."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_dora(tmp_path, base_tensors, zero_b=True)
        out = tmp_path / "merged.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            DCPLoRAMergePipe(checkpoint_dir=dcp_dir),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        for k in ("model.layers.0.self_attn.q_proj.weight",
                  "model.layers.0.self_attn.v_proj.weight"):
            assert torch.allclose(result[k], base_tensors[k], atol=1e-4), \
                f"{k}: zero lora_B DoRA via DCP should be near-identity"

    def test_nonzero_b_changes_weight(self, tmp_path):
        """Non-zero lora_B must produce DoRA-merged weights that differ from base."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_dora(tmp_path, base_tensors, zero_b=False)
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
        ), "q_proj should have changed with non-zero lora_B"

    def test_non_dora_keys_unchanged(self, tmp_path):
        """Keys with no LoRA/DoRA entry in the DCP must pass through unchanged."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_dora(tmp_path, base_tensors, zero_b=False)
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
        """target_modules restricts DoRA merging to listed modules."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)
        dcp_dir = _make_dcp_dora(tmp_path, base_tensors, zero_b=False, target_modules=["q_proj"])
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
        ), "q_proj (in target_modules) should be merged"
        assert torch.allclose(
            result["model.layers.0.self_attn.v_proj.weight"],
            base_tensors["model.layers.0.self_attn.v_proj.weight"],
        ), "v_proj (not in target_modules) should be unchanged"

    def test_result_differs_from_plain_lora(self, tmp_path):
        """DoRA output must differ from the equivalent plain-LoRA merge."""
        from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe

        base_path, base_tensors = _make_base_single(tmp_path)

        # Build a DCP with non-zero lora_B and a magnitude ≠ ||W||_r
        # (scale magnitude by 2 so it can't accidentally equal the LoRA result)
        import torch.distributed.checkpoint as dist_cp

        prefix = "base_model.model.model.layers.0.self_attn"
        rank, alpha = 4, 8.0
        scale = alpha / rank
        lora_a = torch.randn(rank, 32) * 0.01
        lora_b = torch.ones(64, rank) * 0.1

        q_weight = base_tensors["model.layers.0.self_attn.q_proj.weight"]
        mag_q = (q_weight + scale * lora_b @ lora_a).norm(p=2, dim=1) * 2.0  # doubled

        tensors = {
            f"{prefix}.q_proj.lora_A.default.weight": lora_a,
            f"{prefix}.q_proj.lora_B.default.weight": lora_b,
            f"{prefix}.q_proj.lora_magnitude_vector.default.weight": mag_q,
        }
        dcp_dir = tmp_path / "pytorch_model_fsdp_0"
        dcp_dir.mkdir()
        dist_cp.save(
            state_dict={"model": tensors},
            storage_writer=dist_cp.FileSystemWriter(dcp_dir),
            no_dist=True,
        )
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"r": rank, "lora_alpha": alpha})
        )

        # Plain-LoRA checkpoint (same A/B, no magnitude)
        lora_only_tensors = {
            f"{prefix}.q_proj.lora_A.default.weight": lora_a,
            f"{prefix}.q_proj.lora_B.default.weight": lora_b,
        }
        dcp_dir_plain = tmp_path / "pytorch_model_fsdp_plain"
        dcp_dir_plain.mkdir()
        dist_cp.save(
            state_dict={"model": lora_only_tensors},
            storage_writer=dist_cp.FileSystemWriter(dcp_dir_plain),
            no_dist=True,
        )
        config_path = tmp_path / "adapter_config.json"

        out_dora = tmp_path / "out_dora.safetensors"
        out_lora = tmp_path / "out_lora.safetensors"

        Pipeline(SafetensorsReader(base_path),
                 DCPLoRAMergePipe(dcp_dir, config_path=config_path),
                 StreamingWriter(out_dora)).run(show_progress=False)

        Pipeline(SafetensorsReader(base_path),
                 DCPLoRAMergePipe(dcp_dir_plain, config_path=config_path),
                 StreamingWriter(out_lora)).run(show_progress=False)

        r_dora = _load(out_dora)
        r_lora = _load(out_lora)
        assert not torch.allclose(
            r_dora["model.layers.0.self_attn.q_proj.weight"],
            r_lora["model.layers.0.self_attn.q_proj.weight"],
        ), "DoRA and LoRA merges should produce different results"


# ---------------------------------------------------------------------------
# _resolve_dcp_checkpoint (CLI helper)
# ---------------------------------------------------------------------------


def _make_training_dir(tmp: Path, checkpoint_steps: list[int], dcp_name: str = "pytorch_model_fsdp_0") -> Path:
    """
    Create a synthetic training output directory:
        tmp/
          adapter_config.json
          config.json
          checkpoint-<N>/
            <dcp_name>/
              .metadata   ← marks it as a DCP dir
          ...
    Returns the training output directory path.
    """
    import torch.distributed.checkpoint as dist_cp

    (tmp / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8.0}))
    (tmp / "config.json").write_text("{}")
    for step in checkpoint_steps:
        ckpt = tmp / f"checkpoint-{step}"
        dcp = ckpt / dcp_name
        dcp.mkdir(parents=True)
        # Write a minimal DCP checkpoint so .metadata exists
        dist_cp.save(
            state_dict={"x": torch.zeros(1)},
            storage_writer=dist_cp.FileSystemWriter(dcp),
            no_dist=True,
        )
    return tmp


class TestResolveDcpCheckpoint:
    """Unit tests for tftf.cli._resolve_dcp_checkpoint."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from tftf.cli import _resolve_dcp_checkpoint
        self.resolve = _resolve_dcp_checkpoint

    def test_passthrough_when_already_dcp_dir(self, tmp_path):
        """A directory containing .metadata is returned unchanged."""
        import torch.distributed.checkpoint as dist_cp

        dcp = tmp_path / "pytorch_model_fsdp_0"
        dcp.mkdir()
        dist_cp.save(
            state_dict={"x": torch.zeros(1)},
            storage_writer=dist_cp.FileSystemWriter(dcp),
            no_dist=True,
        )
        assert self.resolve(dcp) == dcp

    def test_passthrough_when_no_checkpoints(self, tmp_path):
        """A directory without checkpoint-* subdirs is returned unchanged."""
        (tmp_path / "some_file.txt").write_text("hello")
        assert self.resolve(tmp_path) == tmp_path

    def test_selects_latest_checkpoint(self, tmp_path):
        """The checkpoint with the highest step number is selected."""
        training_dir = _make_training_dir(tmp_path, [30, 60, 111])
        result = self.resolve(training_dir)
        assert result.parent.name == "checkpoint-111"

    def test_selects_latest_with_non_sequential_steps(self, tmp_path):
        """Latest is determined numerically, not lexicographically."""
        # lexicographic order would pick checkpoint-90, numeric picks checkpoint-111
        training_dir = _make_training_dir(tmp_path, [30, 90, 111])
        result = self.resolve(training_dir)
        assert result.parent.name == "checkpoint-111"

    def test_single_checkpoint(self, tmp_path):
        """Works correctly when only one checkpoint exists."""
        training_dir = _make_training_dir(tmp_path, [42])
        result = self.resolve(training_dir)
        assert result.parent.name == "checkpoint-42"

    def test_returns_dcp_subdir(self, tmp_path):
        """Returned path points to the DCP subdir, not the checkpoint dir itself."""
        training_dir = _make_training_dir(tmp_path, [60], dcp_name="pytorch_model_fsdp_0")
        result = self.resolve(training_dir)
        assert (result / ".metadata").exists()
        assert result.name == "pytorch_model_fsdp_0"

    def test_no_dcp_subdir_raises(self, tmp_path):
        """Raises BadParameter when the latest checkpoint has no DCP subdir."""
        import click

        ckpt = tmp_path / "checkpoint-10"
        ckpt.mkdir()
        # no subdir with .metadata inside
        (tmp_path / "adapter_config.json").write_text("{}")

        with pytest.raises(click.BadParameter, match="No DCP directory"):
            self.resolve(tmp_path)

    def test_prefers_model_dcp_over_optimizer_dcp(self, tmp_path):
        """When optimizer_0 and pytorch_model_fsdp_0 both exist, picks the model dir.

        optimizer_0 sorts before pytorch_model_fsdp_0 alphabetically, so without
        the optimizer-filtering logic the wrong directory would be returned.
        """
        import torch.distributed.checkpoint as dist_cp

        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8.0}))
        ckpt = tmp_path / "checkpoint-100"

        # optimizer DCP (alphabetically first — the previously-buggy choice)
        opt_dcp = ckpt / "optimizer_0"
        opt_dcp.mkdir(parents=True)
        dist_cp.save({"x": torch.zeros(1)}, storage_writer=dist_cp.FileSystemWriter(opt_dcp), no_dist=True)

        # model DCP (alphabetically second — the correct choice)
        model_dcp = ckpt / "pytorch_model_fsdp_0"
        model_dcp.mkdir(parents=True)
        dist_cp.save({"x": torch.zeros(1)}, storage_writer=dist_cp.FileSystemWriter(model_dcp), no_dist=True)

        result = self.resolve(tmp_path)
        assert result.name == "pytorch_model_fsdp_0", (
            f"Expected model DCP directory, got {result.name!r}"
        )


# ---------------------------------------------------------------------------
# merge-lora auto-detection (_resolve_adapter + merge-lora command)
# ---------------------------------------------------------------------------


def _make_regular_lora_checkpoint(tmp: Path, steps: list[int], rank: int = 4, alpha: float = 8.0) -> Path:
    """
    Create a training output directory with regular LoRA safetensors checkpoints.
    """
    lora_tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.ones(rank, 32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.ones(64, rank),
    }
    cfg = {"r": rank, "lora_alpha": alpha, "target_modules": ["q_proj"]}
    (tmp / "adapter_config.json").write_text(json.dumps(cfg))
    for step in steps:
        ckpt = tmp / f"checkpoint-{step}"
        ckpt.mkdir()
        save_file(lora_tensors, str(ckpt / "adapter_model.safetensors"))
    return tmp


class TestResolveAdapter:
    """Unit tests for tftf.cli._resolve_adapter."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from tftf.cli import _resolve_adapter
        self.resolve = _resolve_adapter

    def test_passthrough_plain_lora_file(self, tmp_path):
        """A .safetensors file is returned as-is with kind='lora'."""
        f = tmp_path / "adapter_model.safetensors"
        save_file({"x": torch.zeros(1)}, str(f))
        path, kind, hint = self.resolve(f)
        assert path == f
        assert kind == "lora"
        assert hint is None

    def test_passthrough_dcp_dir(self, tmp_path):
        """A directory with .metadata is returned as-is with kind='dcp'."""
        import torch.distributed.checkpoint as dist_cp
        dcp = tmp_path / "pytorch_model_fsdp_0"
        dcp.mkdir()
        dist_cp.save({"x": torch.zeros(1)}, storage_writer=dist_cp.FileSystemWriter(dcp), no_dist=True)
        path, kind, hint = self.resolve(dcp)
        assert path == dcp
        assert kind == "dcp"
        assert hint is None

    def test_training_dir_with_dcp_resolves_to_dcp(self, tmp_path):
        """Training dir containing DCP checkpoints → kind='dcp', correct subdir."""
        training_dir = _make_training_dir(tmp_path, [30, 111])
        path, kind, hint = self.resolve(training_dir)
        assert kind == "dcp"
        assert (path / ".metadata").exists()
        assert path.parent.name == "checkpoint-111"

    def test_training_dir_with_dcp_returns_config_hint(self, tmp_path):
        """config_hint points to adapter_config.json at training dir level."""
        training_dir = _make_training_dir(tmp_path, [60])
        _, _, hint = self.resolve(training_dir)
        assert hint == training_dir / "adapter_config.json"

    def test_training_dir_no_config_hint_is_none(self, tmp_path):
        """If adapter_config.json is absent from training dir, hint is None."""
        training_dir = _make_training_dir(tmp_path, [60])
        (training_dir / "adapter_config.json").unlink()
        _, _, hint = self.resolve(training_dir)
        assert hint is None

    def test_training_dir_with_regular_lora(self, tmp_path):
        """Training dir with adapter_model.safetensors checkpoints → kind='lora'."""
        training_dir = _make_regular_lora_checkpoint(tmp_path, [30, 60, 90])
        path, kind, hint = self.resolve(training_dir)
        assert kind == "lora"
        assert (path / "adapter_model.safetensors").exists()
        assert path.name == "checkpoint-90"

    def test_training_dir_bad_content_raises(self, tmp_path):
        """Training dir whose latest checkpoint has neither DCP nor safetensors raises."""
        import click
        (tmp_path / "adapter_config.json").write_text("{}")
        (tmp_path / "checkpoint-1").mkdir()
        with pytest.raises(click.BadParameter, match="does not appear to be a training directory"):
            self.resolve(tmp_path)

    def test_prefers_model_dcp_over_optimizer_dcp(self, tmp_path):
        """When optimizer_0 and pytorch_model_fsdp_0 both exist, picks the model dir.

        optimizer_0 sorts before pytorch_model_fsdp_0 alphabetically, so without
        the optimizer-filtering logic the wrong directory would be returned.
        """
        import torch.distributed.checkpoint as dist_cp

        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8.0}))
        ckpt = tmp_path / "checkpoint-100"

        # optimizer DCP (alphabetically first — the previously-buggy choice)
        opt_dcp = ckpt / "optimizer_0"
        opt_dcp.mkdir(parents=True)
        dist_cp.save({"x": torch.zeros(1)}, storage_writer=dist_cp.FileSystemWriter(opt_dcp), no_dist=True)

        # model DCP (alphabetically second — the correct choice)
        model_dcp = ckpt / "pytorch_model_fsdp_0"
        model_dcp.mkdir(parents=True)
        dist_cp.save({"x": torch.zeros(1)}, storage_writer=dist_cp.FileSystemWriter(model_dcp), no_dist=True)

        path, kind, _ = self.resolve(tmp_path)
        assert kind == "dcp"
        assert path.name == "pytorch_model_fsdp_0", (
            f"Expected model DCP directory, got {path.name!r}"
        )


def _load_output(path: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from a sharded CLI output directory."""
    result = {}
    for sf in sorted(path.glob("*.safetensors")):
        result.update(load_file(str(sf)))
    return result


class TestMergeLoraCLIAutoDetect:
    """Integration tests: merge-lora command routes to the correct pipe automatically."""

    def _run_merge(self, base_path: Path, adapter_path: Path, out_path: Path) -> None:
        from click.testing import CliRunner

        from tftf.cli import cli
        result = CliRunner().invoke(cli, [
            "merge-lora",
            "-b", str(base_path),
            "-a", str(adapter_path),
            "-o", str(out_path),
        ])
        if result.exit_code != 0:
            raise AssertionError(result.output + (str(result.exception) if result.exception else ""))

    def test_training_dir_dcp_merges_correctly(self, tmp_path):
        """merge-lora accepts a DCP training dir and produces a merged model."""
        (tmp_path / "base").mkdir()
        base_path, base_tensors = _make_base_single(tmp_path / "base")
        training_dir = tmp_path / "training"
        training_dir.mkdir()
        _make_training_dir(training_dir, [60])

        out = tmp_path / "merged"
        self._run_merge(base_path, training_dir, out)
        result = _load_output(out)
        assert set(result.keys()) == set(base_tensors.keys())

    def test_training_dir_regular_lora_merges_correctly(self, tmp_path):
        """merge-lora accepts a regular-LoRA training dir and produces a merged model."""
        (tmp_path / "base").mkdir()
        base_path, base_tensors = _make_base_single(tmp_path / "base")
        training_dir = tmp_path / "training"
        training_dir.mkdir()
        _make_regular_lora_checkpoint(training_dir, [30, 90])

        out = tmp_path / "merged"
        self._run_merge(base_path, training_dir, out)
        result = _load_output(out)
        assert set(result.keys()) == set(base_tensors.keys())
        # lora_B is all-ones so q_proj must differ from base
        assert not torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            base_tensors["model.layers.0.self_attn.q_proj.weight"],
        )

    def test_extras_copied_from_base_dir(self, tmp_path):
        """Non-weight files from the base model directory appear in the output."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()
        base_path, base_tensors = _make_base_single(base_dir)
        # Plant some extra files that should be copied
        (base_dir / "config.json").write_text('{"model_type": "llama"}')
        (base_dir / "tokenizer.json").write_text("{}")
        # And files that must NOT be copied
        (base_dir / "model.gguf").write_bytes(b"fake gguf")

        training_dir = tmp_path / "training"
        training_dir.mkdir()
        _make_regular_lora_checkpoint(training_dir, [10])

        out = tmp_path / "merged"
        self._run_merge(base_path, training_dir, out)
        assert (out / "config.json").exists()
        assert (out / "tokenizer.json").exists()
        assert not (out / "model.gguf").exists()
        # model.safetensors.index.json is produced by the sharded writer — not a
        # stale copy from the base model (which had none in this test case)
        assert (out / "model.safetensors.index.json").exists()
