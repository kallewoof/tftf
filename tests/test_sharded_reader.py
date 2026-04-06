"""Tests for ShardedSafetensorsReader and Pipeline.ReaderProtocol duck-typing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tftf.io.reader import SafetensorsReader
from tftf.io.sharded_reader import ShardedSafetensorsReader
from tftf.io.writer import StreamingWriter
from tftf.pipeline import Pipeline, ReaderProtocol
from tftf.pipes.passthrough import PassthroughPipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save(tensors: dict[str, torch.Tensor], path: Path) -> None:
    save_file(tensors, str(path))


def _load(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path))


BASE_TENSORS = {
    "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 32),
    "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 32),
    "model.layers.0.mlp.gate_proj.weight":    torch.randn(64, 32),
    "model.embed_tokens.weight":               torch.randn(128, 32),
    "model.norm.weight":                       torch.randn(32),
}


def _make_base_single(tmp: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    path = tmp / "model.safetensors"
    _save(BASE_TENSORS, path)
    return path, BASE_TENSORS


def _make_base_sharded(tmp: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    keys = list(BASE_TENSORS.keys())
    shard0 = {k: BASE_TENSORS[k] for k in keys[:3]}
    shard1 = {k: BASE_TENSORS[k] for k in keys[3:]}
    _save(shard0, tmp / "model-00001-of-00002.safetensors")
    _save(shard1, tmp / "model-00002-of-00002.safetensors")
    weight_map = dict.fromkeys(keys[:3], "model-00001-of-00002.safetensors")
    weight_map.update(dict.fromkeys(keys[3:], "model-00002-of-00002.safetensors"))
    index_path = tmp / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"metadata": {"total_size": 9999}, "weight_map": weight_map}))
    return index_path, BASE_TENSORS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShardedSafetensorsReader:

    def test_from_path_single_file(self, tmp_path):
        path, _ = _make_base_single(tmp_path)
        reader = ShardedSafetensorsReader.from_path(path)
        assert isinstance(reader, SafetensorsReader)

    def test_from_path_directory(self, tmp_path):
        _make_base_sharded(tmp_path)
        reader = ShardedSafetensorsReader.from_path(tmp_path)
        assert isinstance(reader, ShardedSafetensorsReader)

    def test_from_path_index_json(self, tmp_path):
        index_path, _ = _make_base_sharded(tmp_path)
        reader = ShardedSafetensorsReader.from_path(index_path)
        assert isinstance(reader, ShardedSafetensorsReader)

    def test_keys_order_and_count(self, tmp_path):
        _, tensors = _make_base_sharded(tmp_path)
        reader = ShardedSafetensorsReader.from_path(tmp_path)
        assert set(reader.keys()) == set(tensors.keys())
        assert reader.num_tensors() == len(tensors)

    def test_iter_meta_shapes_and_dtypes(self, tmp_path):
        index_path, tensors = _make_base_sharded(tmp_path)
        reader = ShardedSafetensorsReader(index_path)
        for meta in reader.iter_meta():
            expected = tensors[meta.key]
            assert meta.shape == expected.shape
            assert meta.dtype == expected.dtype

    def test_iter_records_values(self, tmp_path):
        index_path, tensors = _make_base_sharded(tmp_path)
        reader = ShardedSafetensorsReader(index_path)
        recovered = {r.key: r.tensor for r in reader.iter_records()}
        assert set(recovered.keys()) == set(tensors.keys())
        for k, v in tensors.items():
            assert torch.allclose(recovered[k], v)

    def test_satisfies_reader_protocol(self, tmp_path):
        index_path, _ = _make_base_sharded(tmp_path)
        reader = ShardedSafetensorsReader(index_path)
        assert isinstance(reader, ReaderProtocol)

    def test_pipeline_with_sharded_reader(self, tmp_path):
        index_path, tensors = _make_base_sharded(tmp_path)
        out = tmp_path / "out.safetensors"
        Pipeline(
            reader=ShardedSafetensorsReader(index_path),
            pipe=PassthroughPipe(),
            writer=StreamingWriter(out),
        ).run(show_progress=False)
        result = _load(out)
        assert set(result.keys()) == set(tensors.keys())
        for k, v in tensors.items():
            assert torch.allclose(result[k], v)

    def test_directory_without_index_falls_back_to_single(self, tmp_path):
        _save({"w": torch.randn(4, 4)}, tmp_path / "model.safetensors")
        reader = ShardedSafetensorsReader.from_path(tmp_path)
        assert isinstance(reader, SafetensorsReader)

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot determine"):
            ShardedSafetensorsReader.from_path(tmp_path)
