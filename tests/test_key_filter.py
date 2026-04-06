"""Tests for KeyFilterPipe — glob-based tensor key filtering."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from tftf.io.reader import SafetensorsReader
from tftf.io.writer import StreamingWriter
from tftf.pipeline import Pipeline
from tftf.pipes.base import TensorMeta
from tftf.pipes.key_filter import KeyFilterPipe


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKeyFilterPipe:

    def test_include_glob(self):
        pipe = KeyFilterPipe(include=["*q_proj*"])
        metas = [
            TensorMeta("model.q_proj.weight", torch.float32, torch.Size([4, 4])),
            TensorMeta("model.v_proj.weight", torch.float32, torch.Size([4, 4])),
            TensorMeta("model.norm.weight",   torch.float32, torch.Size([4])),
        ]
        out = list(pipe.process_meta(iter(metas)))
        assert [m.key for m in out] == ["model.q_proj.weight"]

    def test_exclude_glob(self):
        pipe = KeyFilterPipe(exclude=["*norm*"])
        metas = [
            TensorMeta("model.q_proj.weight", torch.float32, torch.Size([4, 4])),
            TensorMeta("model.norm.weight",   torch.float32, torch.Size([4])),
        ]
        out = list(pipe.process_meta(iter(metas)))
        assert [m.key for m in out] == ["model.q_proj.weight"]

    def test_include_and_exclude(self):
        pipe = KeyFilterPipe(include=["*proj*"], exclude=["*o_proj*"])
        keys = ["q_proj", "v_proj", "o_proj"]
        metas = [TensorMeta(k, torch.float32, torch.Size([4])) for k in keys]
        out = list(pipe.process_meta(iter(metas)))
        assert [m.key for m in out] == ["q_proj", "v_proj"]

    def test_no_patterns_passes_all(self):
        pipe = KeyFilterPipe()
        metas = [TensorMeta(f"key_{i}", torch.float32, torch.Size([4])) for i in range(5)]
        out = list(pipe.process_meta(iter(metas)))
        assert len(out) == 5

    def test_process_drops_tensor_correctly(self, tmp_path):
        src, _ = _make_base_single(tmp_path)
        out = tmp_path / "filtered.safetensors"

        pipe = KeyFilterPipe(include=["*q_proj*", "*norm*"])
        Pipeline(SafetensorsReader(src), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        assert all("q_proj" in k or "norm" in k for k in result)
        assert "v_proj" not in " ".join(result.keys())

    def test_process_and_process_meta_agree(self, tmp_path):
        """The set of keys output by process() must equal that of process_meta()."""
        src, _ = _make_base_single(tmp_path)
        pipe = KeyFilterPipe(include=["*proj*"])

        meta_keys = {m.key for m in pipe.process_meta(SafetensorsReader(src).iter_meta())}

        pipe2 = KeyFilterPipe(include=["*proj*"])
        data_keys = {r.key for r in pipe2.process(SafetensorsReader(src).iter_records())}

        assert meta_keys == data_keys
