"""
Tests for:
- ShardedSafetensorsReader
- KeyFilterPipe
- KeyRenamePipe
- DCPLoRAMergePipe
- Pipeline.ReaderProtocol duck-typing
- merge-lora with sharded base model (integration)
"""

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
from tftf.pipes.base import TensorMeta
from tftf.pipes.key_filter import KeyFilterPipe
from tftf.pipes.key_rename import KeyRenamePipe
from tftf.pipes.passthrough import PassthroughPipe


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



# ---------------------------------------------------------------------------
# DCPLoRAMergePipe tests
# ---------------------------------------------------------------------------


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

        base_path, _ = _make_base_single(tmp_path)
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
