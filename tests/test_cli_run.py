"""Tests for the composable `tftf run` command and its PipeChainParser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tftf.pipes.base import CompoundPipe
from tftf.pipes.dtype_cast import DTypeCastPipe
from tftf.pipes.key_filter import KeyFilterPipe
from tftf.pipes.key_rename import KeyRenamePipe
from tftf.pipes.passthrough import PassthroughPipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TENSORS = {
    "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 32),
    "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 32),
    "model.norm.weight": torch.randn(32),
}


def _make_base(directory: Path) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / "model.safetensors"
    save_file(_BASE_TENSORS, str(path))
    return path


def _make_lora_adapter(directory: Path, rank: int = 4, alpha: float = 8.0) -> Path:
    """Create a minimal regular LoRA adapter directory."""
    directory.mkdir(exist_ok=True)
    lora_tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight":
            torch.ones(rank, 32),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight":
            torch.ones(64, rank),
    }
    save_file(lora_tensors, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps({"r": rank, "lora_alpha": alpha})
    )
    return directory


def _load_output(path: Path) -> dict[str, torch.Tensor]:
    """Load all tensors from a sharded output directory."""
    result = {}
    for sf in sorted(path.glob("*.safetensors")):
        result.update(load_file(str(sf)))
    return result


def _run(args: list[str]) -> tuple[int, str]:
    from click.testing import CliRunner

    from tftf.cli import cli
    result = CliRunner().invoke(cli, args)
    return result.exit_code, result.output + (
        f"\n{result.exception}" if result.exception else ""
    )


# ---------------------------------------------------------------------------
# Unit tests for _parse_pipe_chain
# ---------------------------------------------------------------------------


class TestPipeChainParser:

    @pytest.fixture(autouse=True)
    def _import(self):
        from tftf.cli import _parse_pipe_chain
        self.parse = _parse_pipe_chain

    def test_empty_returns_passthrough(self):
        assert isinstance(self.parse(()), PassthroughPipe)

    def test_single_dtype_cast(self):
        pipe = self.parse(("--dtype-cast", "--dtype", "bfloat16"))
        assert isinstance(pipe, DTypeCastPipe)

    def test_single_key_filter(self):
        pipe = self.parse(("--key-filter", "--include", "*q_proj*"))
        assert isinstance(pipe, KeyFilterPipe)

    def test_two_pipes_returns_compound(self):
        pipe = self.parse((
            "--key-filter", "--include", "*q_proj*",
            "--dtype-cast", "--dtype", "bfloat16",
        ))
        assert isinstance(pipe, CompoundPipe)

    def test_multiple_include_flags_accumulated(self):
        # Both --include flags should be collected without error
        pipe = self.parse((
            "--key-filter",
            "--include", "*q_proj*",
            "--include", "*v_proj*",
        ))
        assert isinstance(pipe, KeyFilterPipe)

    def test_key_rename_single_rule(self):
        pipe = self.parse(("--key-rename", "--rule", "^foo", "bar"))
        assert isinstance(pipe, KeyRenamePipe)

    def test_key_rename_multiple_rules(self):
        pipe = self.parse((
            "--key-rename",
            "--rule", "^foo", "bar",
            "--rule", "^baz", "qux",
        ))
        assert isinstance(pipe, KeyRenamePipe)

    def test_token_before_any_pipe_flag_raises(self):
        with pytest.raises(click.UsageError, match="Unexpected token"):
            self.parse(("--nonexistent-pipe",))

    def test_unknown_option_for_pipe_raises(self):
        with pytest.raises(click.UsageError, match="Unknown option"):
            self.parse(("--dtype-cast", "--typo", "bfloat16"))

    def test_missing_required_arg_raises(self):
        with pytest.raises(click.UsageError, match="requires"):
            self.parse(("--merge-lora",))  # --adapter is required

    def test_dtype_choices_validated(self):
        with pytest.raises(click.UsageError, match="not one of"):
            self.parse(("--dtype-cast", "--dtype", "not_a_dtype"))

    def test_missing_value_for_flag_raises(self):
        # --dtype with no following value (next token is another flag)
        with pytest.raises(click.UsageError, match="requires"):
            self.parse(("--dtype-cast", "--dtype"))

    def test_three_pipes_chained(self):
        pipe = self.parse((
            "--key-filter", "--include", "*q_proj*",
            "--dtype-cast", "--dtype", "bfloat16",
            "--key-rename", "--rule", "^model", "base",
        ))
        assert isinstance(pipe, CompoundPipe)

    # -- positional (anonymous) argument tests --------------------------------

    def test_dtype_cast_positional(self):
        pipe = self.parse(("--dtype-cast", "bfloat16"))
        assert isinstance(pipe, DTypeCastPipe)

    def test_dequant_fp8_positional_dtype(self):
        from tftf.pipes.fp8_dequant import FP8DequantPipe
        pipe = self.parse(("--dequant-fp8", "fp16"))
        assert isinstance(pipe, FP8DequantPipe)

    def test_dequant_fp8_positional_then_named(self):
        from tftf.pipes.fp8_dequant import FP8DequantPipe
        pipe = self.parse(("--dequant-fp8", "fp16", "--block-size", "64"))
        assert isinstance(pipe, FP8DequantPipe)

    def test_key_rename_positional_single_rule(self):
        pipe = self.parse(("--key-rename", "^foo", "bar"))
        assert isinstance(pipe, KeyRenamePipe)

    def test_key_rename_positional_multiple_rules(self):
        pipe = self.parse(("--key-rename", "^foo", "bar", "^baz", "qux"))
        assert isinstance(pipe, KeyRenamePipe)

    def test_key_filter_positional_include(self):
        pipe = self.parse(("--key-filter", "*q_proj*", "*v_proj*"))
        assert isinstance(pipe, KeyFilterPipe)

    def test_positional_invalid_choice_raises(self):
        with pytest.raises(click.UsageError, match="not one of"):
            self.parse(("--dtype-cast", "not_a_dtype"))

    def test_no_positional_for_pipe_raises(self):
        # --key-filter's positional is --include; passing a bare value after
        # exhausting positional slots still works (multiple), but a pipe with
        # no positional should reject bare values.  --dtype-cast has only one
        # positional slot (non-multiple), so a second bare value is an error.
        with pytest.raises(click.UsageError):
            self.parse(("--dtype-cast", "bfloat16", "extra"))


import click  # noqa: E402  (used in test bodies above via pytest.raises)


# ---------------------------------------------------------------------------
# Integration tests for `tftf run`
# ---------------------------------------------------------------------------


class TestRunCommand:

    def test_dtype_cast_changes_all_dtypes(self, tmp_path):
        base = _make_base(tmp_path / "base")
        out = tmp_path / "out"
        code, msg = _run(["run", "-i", str(base), "-o", str(out),
                          "--dtype-cast", "--dtype", "bfloat16"])
        assert code == 0, msg
        result = _load_output(out)
        assert all(t.dtype == torch.bfloat16 for t in result.values())

    def test_key_filter_keeps_only_matching(self, tmp_path):
        base = _make_base(tmp_path / "base")
        out = tmp_path / "out"
        code, msg = _run(["run", "-i", str(base), "-o", str(out),
                          "--key-filter", "--include", "*q_proj*"])
        assert code == 0, msg
        result = _load_output(out)
        assert result and all("q_proj" in k for k in result)

    def test_merge_lora_modifies_weights(self, tmp_path):
        base = _make_base(tmp_path / "base")
        adapter = _make_lora_adapter(tmp_path / "adapter")
        out = tmp_path / "out"
        code, msg = _run(["run", "-i", str(base), "-o", str(out),
                          "--merge-lora", "--adapter", str(adapter)])
        assert code == 0, msg
        result = _load_output(out)
        assert set(result.keys()) == set(_BASE_TENSORS.keys())
        # lora_A and lora_B are all-ones so q_proj must change
        assert not torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            _BASE_TENSORS["model.layers.0.self_attn.q_proj.weight"],
        )

    def test_merge_lora_then_dtype_cast(self, tmp_path):
        base = _make_base(tmp_path / "base")
        adapter = _make_lora_adapter(tmp_path / "adapter")
        out = tmp_path / "out"
        code, msg = _run(["run", "-i", str(base), "-o", str(out),
                          "--merge-lora", "--adapter", str(adapter),
                          "--dtype-cast", "--dtype", "bfloat16"])
        assert code == 0, msg
        result = _load_output(out)
        assert all(t.dtype == torch.bfloat16 for t in result.values())

    def test_no_pipes_is_passthrough(self, tmp_path):
        base = _make_base(tmp_path / "base")
        out = tmp_path / "out"
        code, msg = _run(["run", "-i", str(base), "-o", str(out)])
        assert code == 0, msg
        result = _load_output(out)
        assert set(result.keys()) == set(_BASE_TENSORS.keys())

    def test_dry_run_creates_no_output(self, tmp_path):
        base = _make_base(tmp_path / "base")
        out = tmp_path / "out"
        code, _ = _run(["run", "-i", str(base), "-o", str(out),
                        "--dtype-cast", "--dtype", "bfloat16", "--dry-run"])
        assert code == 0
        assert not out.exists()

    def test_positional_adapter_path(self, tmp_path):
        """--merge-lora /path is equivalent to --merge-lora --adapter /path."""
        base = _make_base(tmp_path / "base")
        adapter = _make_lora_adapter(tmp_path / "adapter")
        out = tmp_path / "out"
        code, msg = _run(["run", "-i", str(base), "-o", str(out),
                          "--merge-lora", str(adapter)])
        assert code == 0, msg
        result = _load_output(out)
        assert set(result.keys()) == set(_BASE_TENSORS.keys())

    def test_positional_dtype_cast(self, tmp_path):
        """--dtype-cast bf16 is equivalent to --dtype-cast --dtype bf16."""
        base = _make_base(tmp_path / "base")
        out = tmp_path / "out"
        code, msg = _run(["run", "-i", str(base), "-o", str(out),
                          "--dtype-cast", "bfloat16"])
        assert code == 0, msg
        result = _load_output(out)
        assert all(t.dtype == torch.bfloat16 for t in result.values())

    def test_unknown_pipe_flag_exits_nonzero(self, tmp_path):
        base = _make_base(tmp_path / "base")
        out = tmp_path / "out"
        code, _ = _run(["run", "-i", str(base), "-o", str(out),
                        "--no-such-pipe"])
        assert code != 0
