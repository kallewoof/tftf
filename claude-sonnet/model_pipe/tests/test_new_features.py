"""
Tests for:
- ShardedSafetensorsReader
- KeyFilterPipe
- KeyRenamePipe
- FSDPShardMergePipe
- FSDP shard utilities
- Pipeline.ReaderProtocol duck-typing
- merge-lora with sharded base model (integration)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from model_pipe.io.reader import SafetensorsReader
from model_pipe.io.sharded_reader import ShardedSafetensorsReader
from model_pipe.io.writer import StreamingWriter
from model_pipe.pipeline import Pipeline, ReaderProtocol
from model_pipe.pipes.base import TensorMeta, TensorRecord
from model_pipe.pipes.dtype_cast import DTypeCastPipe
from model_pipe.pipes.fsdp_lora_merge import FSDPShardMergePipe
from model_pipe.pipes.key_filter import KeyFilterPipe
from model_pipe.pipes.key_rename import KeyRenamePipe
from model_pipe.pipes.lora_merge import LoRAMergePipe
from model_pipe.pipes.passthrough import PassthroughPipe
from model_pipe.utils.fsdp import (
    check_for_flat_params,
    find_shard_files,
    reconstruct_from_shards,
)
from model_pipe.utils.lora import merge_lora


# ---------------------------------------------------------------------------
# Fixtures / helpers
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

RANK = 4
ALPHA = 8.0


def _make_base_single(tmp: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """Single-file base model."""
    path = tmp / "model.safetensors"
    _save(BASE_TENSORS, path)
    return path, BASE_TENSORS


def _make_base_sharded(tmp: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    """
    Two-shard base model with model.safetensors.index.json.
    Shard 0 gets the first 3 tensors; shard 1 gets the last 2.
    """
    keys = list(BASE_TENSORS.keys())
    shard0_keys = keys[:3]
    shard1_keys = keys[3:]

    shard0 = {k: BASE_TENSORS[k] for k in shard0_keys}
    shard1 = {k: BASE_TENSORS[k] for k in shard1_keys}

    shard0_path = tmp / "model-00001-of-00002.safetensors"
    shard1_path = tmp / "model-00002-of-00002.safetensors"
    _save(shard0, shard0_path)
    _save(shard1, shard1_path)

    weight_map = {}
    for k in shard0_keys:
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in shard1_keys:
        weight_map[k] = "model-00002-of-00002.safetensors"

    index = {"metadata": {"total_size": 9999}, "weight_map": weight_map}
    index_path = tmp / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index))

    return index_path, BASE_TENSORS


def _make_lora_single(
    tmp: Path, rank: int = RANK, alpha: float = ALPHA
) -> tuple[Path, Path]:
    """Standard single-file PEFT LoRA adapter."""
    lora_tensors = {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
            torch.randn(rank, 32) * 0.01,
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
            torch.zeros(64, rank),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight":
            torch.randn(rank, 32) * 0.01,
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight":
            torch.zeros(64, rank),
    }
    adapter_path = tmp / "adapter_model.safetensors"
    _save(lora_tensors, adapter_path)

    config = {"r": rank, "lora_alpha": alpha, "target_modules": ["q_proj", "v_proj"]}
    config_path = tmp / "adapter_config.json"
    config_path.write_text(json.dumps(config))
    return adapter_path, config_path


def _make_fsdp_shards(
    tmp: Path,
    world_size: int = 2,
    rank: int = RANK,
    alpha: float = ALPHA,
    out_features: int = 64,
    in_features: int = 32,
) -> tuple[Path, Path, dict]:
    """
    Simulate world_size FSDP per-rank adapter shard files.

    FSDP shards tensors along dim 0 for every parameter.  For LoRA:
      - lora_A shape (r, in):   chunk_a = r // world_size  rows per rank
      - lora_B shape (out, r):  chunk_b = out // world_size rows per rank

    Both are sharded along dim 0, so after cat(dim=0) we recover the full
    (r, in) and (out, r) tensors respectively.

    Returns shard_dir, config_path, and the expected reconstructed lora weights.
    """
    assert rank % world_size == 0, "rank must be divisible by world_size for this fixture"
    assert out_features % world_size == 0, "out_features must be divisible by world_size"
    chunk_a = rank        // world_size   # rows per rank for lora_A (r, in)
    chunk_b = out_features // world_size  # rows per rank for lora_B (out, r)

    # Full lora tensors (what we expect after reconstruction)
    full_lora_a_q = torch.randn(rank, in_features) * 0.01
    full_lora_b_q = torch.zeros(out_features, rank)
    full_lora_a_v = torch.randn(rank, in_features) * 0.01
    full_lora_b_v = torch.zeros(out_features, rank)

    shard_dir = tmp / "fsdp_shards"
    shard_dir.mkdir()

    for i in range(world_size):
        shard_tensors = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
                full_lora_a_q[i*chunk_a:(i+1)*chunk_a].clone(),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
                full_lora_b_q[i*chunk_b:(i+1)*chunk_b].clone(),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight":
                full_lora_a_v[i*chunk_a:(i+1)*chunk_a].clone(),
            "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight":
                full_lora_b_v[i*chunk_b:(i+1)*chunk_b].clone(),
        }
        shard_path = shard_dir / f"rank_{i:02d}.safetensors"
        _save(shard_tensors, shard_path)

    config = {"r": rank, "lora_alpha": alpha, "target_modules": ["q_proj", "v_proj"]}
    config_path = shard_dir / "adapter_config.json"
    config_path.write_text(json.dumps(config))

    expected = {
        "q_a": full_lora_a_q,
        "q_b": full_lora_b_q,
        "v_a": full_lora_a_v,
        "v_b": full_lora_b_v,
    }
    return shard_dir, config_path, expected


# ===========================================================================
# ShardedSafetensorsReader
# ===========================================================================


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
        """Directory containing only model.safetensors → SafetensorsReader."""
        t = {"w": torch.randn(4, 4)}
        _save(t, tmp_path / "model.safetensors")
        reader = ShardedSafetensorsReader.from_path(tmp_path)
        assert isinstance(reader, SafetensorsReader)

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Cannot determine"):
            ShardedSafetensorsReader.from_path(tmp_path)


# ===========================================================================
# KeyFilterPipe
# ===========================================================================


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
        src, tensors = _make_base_single(tmp_path)
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

        # Run process() and collect keys
        pipe2 = KeyFilterPipe(include=["*proj*"])
        data_keys = {r.key for r in pipe2.process(SafetensorsReader(src).iter_records())}

        assert meta_keys == data_keys


# ===========================================================================
# KeyRenamePipe
# ===========================================================================


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


# ===========================================================================
# FSDP utilities
# ===========================================================================


class TestFSDPUtils:

    def test_find_shard_files_from_list(self, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"rank_{i}.safetensors"
            p.touch()
            paths.append(p)
        result = find_shard_files(paths)
        assert result == paths

    def test_find_shard_files_from_directory(self, tmp_path):
        for i in range(4):
            (tmp_path / f"rank_{i:02d}.safetensors").touch()
        result = find_shard_files(tmp_path)
        assert len(result) == 4
        # Should be sorted
        assert result == sorted(result)

    def test_find_shard_files_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="No .safetensors"):
            find_shard_files(tmp_path)

    def test_reconstruct_from_shards(self):
        shards = [torch.ones(4, 8) * i for i in range(3)]
        full = reconstruct_from_shards(shards, shard_dim=0)
        assert full.shape == (12, 8)
        assert torch.all(full[:4] == 0)
        assert torch.all(full[4:8] == 1)
        assert torch.all(full[8:] == 2)

    def test_reconstruct_single_shard(self):
        t = torch.randn(8, 4)
        result = reconstruct_from_shards([t])
        assert torch.allclose(result, t)

    def test_check_flat_params_raises(self):
        with pytest.raises(NotImplementedError, match="Flat-param"):
            check_for_flat_params(["_flat_param_0", "normal.key"])

    def test_check_flat_params_clean(self):
        # Should not raise
        check_for_flat_params(["model.layers.0.weight", "model.norm.weight"])


# ===========================================================================
# FSDPShardMergePipe
# ===========================================================================


class TestFSDPShardMergePipe:

    def test_requires_shard_source(self):
        with pytest.raises(ValueError, match="shard_paths or shard_dir"):
            FSDPShardMergePipe()

    def test_rejects_both_sources(self, tmp_path):
        p = tmp_path / "x.safetensors"
        p.touch()
        with pytest.raises(ValueError, match="only one"):
            FSDPShardMergePipe(shard_paths=[p], shard_dir=tmp_path)

    def test_shard_dir_e2e_zero_b(self, tmp_path):
        """
        With lora_B all-zeros, merged weights should equal base weights.
        """
        base_path, base_tensors = _make_base_single(tmp_path)
        shard_dir, config_path, _ = _make_fsdp_shards(tmp_path, world_size=2, rank=RANK)
        out = tmp_path / "merged_fsdp.safetensors"

        pipe = FSDPShardMergePipe(shard_dir=shard_dir, device="cpu")
        Pipeline(SafetensorsReader(base_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        assert set(result.keys()) == set(base_tensors.keys())
        # lora_B is zeros → delta = 0 → merged == base
        assert torch.allclose(
            result["model.layers.0.self_attn.q_proj.weight"],
            base_tensors["model.layers.0.self_attn.q_proj.weight"],
            atol=1e-5,
        )
        # Untouched tensors
        assert torch.allclose(result["model.norm.weight"], base_tensors["model.norm.weight"])

    def test_shard_paths_e2e(self, tmp_path):
        """Same as above but using explicit shard_paths list."""
        base_path, base_tensors = _make_base_single(tmp_path)
        shard_dir, _, _ = _make_fsdp_shards(tmp_path, world_size=2, rank=RANK)
        shard_files = sorted(shard_dir.glob("rank_*.safetensors"))
        out = tmp_path / "merged_explicit.safetensors"

        pipe = FSDPShardMergePipe(shard_paths=shard_files, device="cpu")
        Pipeline(SafetensorsReader(base_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        assert set(result.keys()) == set(base_tensors.keys())

    def test_reconstruction_correctness(self, tmp_path):
        """
        Manually verify that shards are concatenated correctly.
        Use nonzero lora_B so we can check the actual merge value.

        FSDP shards both lora_A and lora_B along dim 0:
          lora_A (r, in)   → chunk_a = r // world_size  rows per rank
          lora_B (out, r)  → chunk_b = out // world_size rows per rank
        """
        world_size = 2
        r = 4
        out_f = 64
        in_f  = 32
        alpha = float(r)  # scale = 1.0

        full_a = torch.eye(r, in_f)[:r]           # (r, 32)
        full_b = torch.eye(out_f, r)[:out_f, :r]  # (64, r)

        chunk_a = r    // world_size  # 2
        chunk_b = out_f // world_size # 32

        shard_dir = tmp_path / "shards2"
        shard_dir.mkdir()

        for i in range(world_size):
            shard = {
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight":
                    full_a[i*chunk_a:(i+1)*chunk_a].clone(),
                "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight":
                    full_b[i*chunk_b:(i+1)*chunk_b].clone(),
            }
            _save(shard, shard_dir / f"rank_{i:02d}.safetensors")

        config = {"r": r, "lora_alpha": alpha, "target_modules": ["q_proj"]}
        (shard_dir / "adapter_config.json").write_text(json.dumps(config))

        base_w = torch.zeros(out_f, in_f)
        _save({"model.layers.0.self_attn.q_proj.weight": base_w}, tmp_path / "base.safetensors")

        out = tmp_path / "merged2.safetensors"
        pipe = FSDPShardMergePipe(shard_dir=shard_dir, scale=1.0)
        Pipeline(
            SafetensorsReader(tmp_path / "base.safetensors"),
            pipe,
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        merged = result["model.layers.0.self_attn.q_proj.weight"]

        # Expected: scale=alpha/r=1.0, delta = full_b @ full_a
        expected = (full_b.float() @ full_a.float()).to(base_w.dtype)
        assert torch.allclose(merged, expected, atol=1e-5)

    def test_fsdp_with_dtype_cast(self, tmp_path):
        base_path, _ = _make_base_single(tmp_path)
        shard_dir, _, _ = _make_fsdp_shards(tmp_path, world_size=2)
        out = tmp_path / "merged_fsdp_fp16.safetensors"

        pipe = FSDPShardMergePipe(shard_dir=shard_dir) | DTypeCastPipe(torch.float16)
        Pipeline(SafetensorsReader(base_path), pipe, StreamingWriter(out)).run(show_progress=False)

        result = _load(out)
        for v in result.values():
            assert v.dtype == torch.float16

    def test_fsdp_with_sharded_base_model(self, tmp_path):
        """End-to-end: sharded base model + FSDP adapter shards."""
        index_path, base_tensors = _make_base_sharded(tmp_path)
        shard_dir, _, _ = _make_fsdp_shards(tmp_path, world_size=2)
        out = tmp_path / "merged_both_sharded.safetensors"

        pipe = FSDPShardMergePipe(shard_dir=shard_dir)
        Pipeline(
            ShardedSafetensorsReader(index_path),
            pipe,
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert set(result.keys()) == set(base_tensors.keys())
