"""
Tests for MoE / grouped-expert LoRA merging (PEFT ``target_parameters``),
regex ``target_modules`` parsing, and the merge guardrails.

Regression context
------------------
An adapter trained on a Gemma-MoE model (``google_gemma-4-26B-A4B``) merged to
a byte-identical copy of the base model.  Two bugs plus a missing guardrail:

1. ``target_modules`` was a *regex string*; ``list(...)`` exploded it into
   single characters so the pre-filter rejected every tensor.
2. The MoE experts used PEFT ``target_parameters`` (stacked-expert LoRA on 3-D
   parameters), which was not implemented at all.
3. Nothing errored when zero tensors were merged — the no-op was silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tftf.io.reader import SafetensorsReader
from tftf.io.writer import StreamingWriter
from tftf.pipeline import Pipeline
from tftf.pipes.lora_merge import LoRAMergePipe
from tftf.utils.lora import (
    LoRAConfig,
    find_grouped_lora_pairs,
    merge_grouped_lora,
)


# Tiny MoE dimensions mirroring the real model's layout (E, out, in).
H = 8      # hidden_size
M = 3      # moe_intermediate_size
E = 3      # num_experts
R = 2      # lora rank
ALPHA = 4.0
SCALE = ALPHA / R  # 2.0

LAYER = "model.language_model.layers.0"
PREFIX = f"base_model.model.{LAYER}"


def _load(path: Path):
    from safetensors.torch import load_file

    return load_file(str(path))


# ---------------------------------------------------------------------------
# merge_grouped_lora — unit
# ---------------------------------------------------------------------------


class TestMergeGroupedLoraUnit:
    def test_matches_per_expert_reference(self):
        """Vectorised grouped merge == per-expert B_e @ A_e (PEFT convention)."""
        weight = torch.randn(E, 2 * M, H)          # (experts, out, in)
        a = torch.randn(E * R, H) * 0.1            # (E*r, in)
        b = torch.randn(2 * M, E * R) * 0.1        # (out, E*r)

        merged = merge_grouped_lora(weight, a, b, SCALE)

        # Independent reference: PEFT stacks experts row-wise in A (outer) and
        # interleaves them column-wise in B (inner, stride = num_experts).
        a3 = a.reshape(E, R, H)
        for e in range(E):
            a_e = a3[e]              # (r, in)
            b_e = b[:, e::E]         # (out, r)
            ref = weight[e] + SCALE * (b_e @ a_e)
            assert torch.allclose(merged[e], ref, atol=1e-5)

    def test_changes_weight(self):
        weight = torch.randn(E, 2 * M, H)
        a = torch.randn(E * R, H)
        b = torch.ones(2 * M, E * R)
        merged = merge_grouped_lora(weight, a, b, SCALE)
        assert not torch.allclose(merged, weight)

    def test_zero_b_is_identity(self):
        weight = torch.randn(E, 2 * M, H)
        a = torch.randn(E * R, H)
        b = torch.zeros(2 * M, E * R)
        merged = merge_grouped_lora(weight, a, b, SCALE)
        assert torch.allclose(merged, weight)

    def test_transposed_orientation(self):
        """A weight stored as (E, in, out) is also supported."""
        weight = torch.randn(E, H, 2 * M)          # (experts, in, out)
        a = torch.randn(E * R, H)                   # in = H
        b = torch.randn(2 * M, E * R)              # out = 2M
        merged = merge_grouped_lora(weight, a, b, SCALE)
        assert merged.shape == weight.shape
        assert not torch.allclose(merged, weight)

    def test_dtype_preserved(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            weight = torch.randn(E, 2 * M, H).to(dtype)
            a = torch.randn(E * R, H).to(dtype)
            b = torch.randn(2 * M, E * R).to(dtype)
            merged = merge_grouped_lora(weight, a, b, SCALE)
            assert merged.dtype == dtype

    def test_non_3d_raises(self):
        with pytest.raises(ValueError, match="3-D"):
            merge_grouped_lora(torch.randn(4, 4), torch.randn(2, 4), torch.randn(4, 2), 1.0)

    def test_rank_not_divisible_raises(self):
        weight = torch.randn(E, 2 * M, H)
        a = torch.randn(E * R + 1, H)   # not divisible by E
        b = torch.randn(2 * M, E * R + 1)
        with pytest.raises(ValueError, match="not divisible"):
            merge_grouped_lora(weight, a, b, SCALE)

    def test_shape_mismatch_raises(self):
        weight = torch.randn(E, 2 * M, H)
        a = torch.randn(E * R, H + 1)   # in doesn't match weight
        b = torch.randn(2 * M, E * R)
        with pytest.raises(ValueError, match="cannot reconcile"):
            merge_grouped_lora(weight, a, b, SCALE)


# ---------------------------------------------------------------------------
# LoRAConfig — regex target_modules parsing (the reported bug)
# ---------------------------------------------------------------------------


class TestLoRAConfigParsing:
    def test_regex_string_target_modules_not_exploded(self, tmp_path):
        """A string target_modules must stay a string, not a list of chars."""
        regex = r"model\.language_model\.layers\.[\d]+\.(mlp|self_attn)\.(q|k|v|o)_proj"
        cfg_path = tmp_path / "adapter_config.json"
        cfg_path.write_text(json.dumps({"r": 16, "lora_alpha": 32, "target_modules": regex}))

        cfg = LoRAConfig.from_file(cfg_path)
        assert isinstance(cfg.target_modules, str)
        assert cfg.target_modules == regex

    def test_regex_matching(self):
        cfg = LoRAConfig(target_modules=r".*\.self_attn\.q_proj")
        assert cfg.matches_module("model.layers.0.self_attn.q_proj")
        assert not cfg.matches_module("model.layers.0.self_attn.v_proj")

    def test_list_matching(self):
        cfg = LoRAConfig(target_modules=["q_proj", "v_proj"])
        assert cfg.matches_module("model.layers.0.self_attn.q_proj")
        assert not cfg.matches_module("model.layers.0.mlp.gate_proj")

    def test_empty_matches_everything(self):
        cfg = LoRAConfig(target_modules=[])
        assert cfg.matches_module("anything.at.all")

    def test_target_parameters_parsed(self, tmp_path):
        cfg_path = tmp_path / "adapter_config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
                }
            )
        )
        cfg = LoRAConfig.from_file(cfg_path)
        assert cfg.target_parameters == ["experts.gate_up_proj", "experts.down_proj"]
        assert cfg.matched_parameter(f"{LAYER}.experts.gate_up_proj") == "experts.gate_up_proj"
        assert cfg.matched_parameter(f"{LAYER}.self_attn.q_proj.weight") is None


# ---------------------------------------------------------------------------
# find_grouped_lora_pairs — nested base_layer wrappers
# ---------------------------------------------------------------------------


def test_find_grouped_lora_pairs():
    keys = {
        f"{PREFIX}.experts.lora_A.weight",
        f"{PREFIX}.experts.lora_B.weight",
        f"{PREFIX}.experts.base_layer.lora_A.weight",
        f"{PREFIX}.experts.base_layer.lora_B.weight",
        # unrelated
        f"{PREFIX}.self_attn.q_proj.lora_A.weight",
        f"{PREFIX}.self_attn.q_proj.lora_B.weight",
    }
    pairs = find_grouped_lora_pairs(f"{LAYER}.experts", keys)
    assert len(pairs) == 2
    a_keys = {a for a, _ in pairs}
    assert f"{PREFIX}.experts.lora_A.weight" in a_keys
    assert f"{PREFIX}.experts.base_layer.lora_A.weight" in a_keys


# ---------------------------------------------------------------------------
# Fixtures: a tiny MoE base model + PEFT adapter (mixed module + parameter LoRA)
# ---------------------------------------------------------------------------


def _make_moe_base(tmp: Path):
    tensors = {
        f"{LAYER}.self_attn.q_proj.weight": torch.randn(H, H),
        f"{LAYER}.experts.gate_up_proj": torch.randn(E, 2 * M, H),   # (E, out, in)
        f"{LAYER}.experts.down_proj": torch.randn(E, H, M),          # (E, out, in)
        "model.norm.weight": torch.randn(H),
    }
    path = tmp / "model.safetensors"
    save_file(tensors, str(path))
    return path, tensors


def _make_moe_adapter(tmp: Path, *, zero_b: bool = False):
    def b(out, inn):
        return torch.zeros(out, inn) if zero_b else torch.randn(out, inn) * 0.1

    tensors = {
        # module LoRA on q_proj
        f"{PREFIX}.self_attn.q_proj.lora_A.weight": torch.randn(R, H) * 0.1,
        f"{PREFIX}.self_attn.q_proj.lora_B.weight": b(H, R),
        # parameter LoRA on gate_up_proj (nested as base_layer): out=2M, in=H
        f"{PREFIX}.experts.base_layer.lora_A.weight": torch.randn(E * R, H) * 0.1,
        f"{PREFIX}.experts.base_layer.lora_B.weight": b(2 * M, E * R),
        # parameter LoRA on down_proj: out=H, in=M
        f"{PREFIX}.experts.lora_A.weight": torch.randn(E * R, M) * 0.1,
        f"{PREFIX}.experts.lora_B.weight": b(H, E * R),
    }
    adapter_path = tmp / "adapter_model.safetensors"
    save_file(tensors, str(adapter_path))

    cfg = {
        "r": R,
        "lora_alpha": ALPHA,
        # regex string, exactly the form that used to break
        "target_modules": r".*\.self_attn\.(q|k|v|o)_proj",
        "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
    }
    cfg_path = tmp / "adapter_config.json"
    cfg_path.write_text(json.dumps(cfg))
    return adapter_path, cfg_path, tensors


# ---------------------------------------------------------------------------
# Integration: full pipeline merges linear + grouped-expert LoRA
# ---------------------------------------------------------------------------


class TestMoEMergeIntegration:
    def test_all_targets_merged(self, tmp_path):
        base_path, base = _make_moe_base(tmp_path)
        adapter_path, cfg_path, _ = _make_moe_adapter(tmp_path, zero_b=False)
        out = tmp_path / "merged.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=cfg_path),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        assert set(result.keys()) == set(base.keys())

        # q_proj (module LoRA via regex target_modules) changed
        assert not torch.allclose(result[f"{LAYER}.self_attn.q_proj.weight"], base[f"{LAYER}.self_attn.q_proj.weight"])
        # both expert parameters (target_parameters) changed
        assert not torch.allclose(result[f"{LAYER}.experts.gate_up_proj"], base[f"{LAYER}.experts.gate_up_proj"])
        assert not torch.allclose(result[f"{LAYER}.experts.down_proj"], base[f"{LAYER}.experts.down_proj"])
        # untouched key unchanged
        assert torch.allclose(result["model.norm.weight"], base["model.norm.weight"])

    def test_grouped_merge_numerically_correct(self, tmp_path):
        base_path, base = _make_moe_base(tmp_path)
        adapter_path, cfg_path, adapter = _make_moe_adapter(tmp_path, zero_b=False)
        out = tmp_path / "merged.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=cfg_path),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)

        # Recompute gate_up_proj independently.
        a = adapter[f"{PREFIX}.experts.base_layer.lora_A.weight"]
        b = adapter[f"{PREFIX}.experts.base_layer.lora_B.weight"]
        ref = merge_grouped_lora(base[f"{LAYER}.experts.gate_up_proj"], a, b, SCALE)
        assert torch.allclose(result[f"{LAYER}.experts.gate_up_proj"], ref, atol=1e-5)

    def test_zero_b_identity_but_not_rejected(self, tmp_path):
        """Zero-B is a real (identity) merge — it must run, not error."""
        base_path, base = _make_moe_base(tmp_path)
        adapter_path, cfg_path, _ = _make_moe_adapter(tmp_path, zero_b=True)
        out = tmp_path / "merged.safetensors"

        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=cfg_path),
            StreamingWriter(out),
        ).run(show_progress=False)

        result = _load(out)
        for k in base:
            assert torch.allclose(result[k], base[k], atol=1e-5)


# ---------------------------------------------------------------------------
# Guardrails: a no-op merge must explode
# ---------------------------------------------------------------------------


class TestMergeGuardrails:
    def test_zero_matches_raises(self, tmp_path):
        """Adapter targeting a layer absent from the base model → hard error."""
        base_path, _ = _make_moe_base(tmp_path)

        # Adapter targets layer 5 (base only has layer 0) → nothing lines up.
        bad_prefix = "base_model.model.model.language_model.layers.5.self_attn.q_proj"
        save_file(
            {
                f"{bad_prefix}.lora_A.weight": torch.randn(R, H),
                f"{bad_prefix}.lora_B.weight": torch.randn(H, R),
            },
            str(tmp_path / "adapter_model.safetensors"),
        )
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": ALPHA}))
        out = tmp_path / "merged.safetensors"

        with pytest.raises(RuntimeError, match="ZERO base tensors"):
            Pipeline(
                SafetensorsReader(base_path),
                LoRAMergePipe(tmp_path / "adapter_model.safetensors"),
                StreamingWriter(out),
            ).run(show_progress=False)

    def test_regex_matching_nothing_raises(self, tmp_path):
        """A target_modules regex that matches no module → hard error."""
        base_path, _ = _make_moe_base(tmp_path)
        save_file(
            {
                f"{PREFIX}.self_attn.q_proj.lora_A.weight": torch.randn(R, H),
                f"{PREFIX}.self_attn.q_proj.lora_B.weight": torch.randn(H, R),
            },
            str(tmp_path / "adapter_model.safetensors"),
        )
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"r": R, "lora_alpha": ALPHA, "target_modules": r"this\.matches\.nothing"})
        )
        out = tmp_path / "merged.safetensors"

        with pytest.raises(RuntimeError):
            Pipeline(
                SafetensorsReader(base_path),
                LoRAMergePipe(tmp_path / "adapter_model.safetensors"),
                StreamingWriter(out),
            ).run(show_progress=False)

    def test_empty_adapter_raises(self, tmp_path):
        """An adapter with no lora_A/lora_B pairs → hard error."""
        base_path, _ = _make_moe_base(tmp_path)
        save_file({"some.random.tensor": torch.randn(4)}, str(tmp_path / "adapter_model.safetensors"))
        (tmp_path / "adapter_config.json").write_text(json.dumps({"r": R, "lora_alpha": ALPHA}))
        out = tmp_path / "merged.safetensors"

        with pytest.raises(RuntimeError, match="no recognizable LoRA"):
            Pipeline(
                SafetensorsReader(base_path),
                LoRAMergePipe(tmp_path / "adapter_model.safetensors"),
                StreamingWriter(out),
            ).run(show_progress=False)

    def test_partial_target_modules_ok(self, tmp_path):
        """Legit target_modules exclusion must NOT error (only warn)."""
        base_path = _make_moe_base(tmp_path)[0]
        # Adapter has q_proj + both experts; target_modules selects only q_proj,
        # and target_parameters still merges the experts.
        adapter_path, cfg_path, _ = _make_moe_adapter(tmp_path, zero_b=False)
        # Narrow target_modules to a pattern that matches q_proj only (it already does).
        out = tmp_path / "merged.safetensors"
        Pipeline(
            SafetensorsReader(base_path),
            LoRAMergePipe(adapter_path, config_path=cfg_path),
            StreamingWriter(out),
        ).run(show_progress=False)
        assert (out).exists()
