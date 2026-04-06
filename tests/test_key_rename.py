"""Tests for KeyRenamePipe — regex-based tensor key renaming."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tftf.io.reader import SafetensorsReader
from tftf.io.writer import StreamingWriter
from tftf.pipeline import Pipeline
from tftf.pipes.base import TensorMeta
from tftf.pipes.key_rename import KeyRenamePipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(tensors: dict[str, torch.Tensor], path: Path) -> None:
    save_file(tensors, str(path))


def _load(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKeyRenamePipe:

    def test_simple_prefix_strip(self):
        pipe = KeyRenamePipe([(r"^base_model\.model\.", "")])
        metas = [
            TensorMeta("base_model.model.layers.0.weight", torch.float32, torch.Size([4]))
        ]
        out = list(pipe.process_meta(iter(metas)))
        assert out[0].key == "layers.0.weight"

    def test_multiple_rules_applied_in_order(self):
        pipe = KeyRenamePipe([
            (r"^transformer\.h\.", "model.layers."),
            (r"\.attn\.", ".self_attn."),
        ])
        metas = [TensorMeta("transformer.h.0.attn.weight", torch.float32, torch.Size([4]))]
        out = list(pipe.process_meta(iter(metas)))
        assert out[0].key == "model.layers.0.self_attn.weight"

    def test_no_match_passes_through(self):
        pipe = KeyRenamePipe([(r"^xyz\.", "abc.")])
        metas = [TensorMeta("model.weight", torch.float32, torch.Size([4]))]
        out = list(pipe.process_meta(iter(metas)))
        assert out[0].key == "model.weight"

    def test_invalid_regex_raises(self):
        with pytest.raises(ValueError, match="Invalid regex"):
            KeyRenamePipe([(r"[invalid", "x")])

    def test_duplicate_key_raises(self):
        """Two keys mapping to the same output should raise."""
        pipe = KeyRenamePipe([(r"^(a|b)$", "c")])
        metas = [
            TensorMeta("a", torch.float32, torch.Size([4])),
            TensorMeta("b", torch.float32, torch.Size([4])),
        ]
        with pytest.raises(ValueError, match="same key"):
            list(pipe.process_meta(iter(metas)))

    def test_rename_e2e(self, tmp_path):
        tensors = {"old.weight": torch.randn(4, 4)}
        src = tmp_path / "model.safetensors"
        _save(tensors, src)
        out = tmp_path / "renamed.safetensors"

        pipe = KeyRenamePipe([(r"^old\.", "new.")])
        Pipeline(SafetensorsReader(src), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        assert "new.weight" in result
        assert torch.allclose(result["new.weight"], tensors["old.weight"])
