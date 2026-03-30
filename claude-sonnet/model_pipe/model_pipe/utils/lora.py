"""
LoRA key-mapping utilities and merge math.

Key naming conventions
----------------------
HuggingFace PEFT writes adapter weights with a `base_model.model.` prefix
and a `.lora_A.weight` / `.lora_B.weight` suffix (or `.lora_A.<name>.weight`
when an explicit adapter name is used).

For embedding layers the keys are `.lora_embedding_A` / `.lora_embedding_B`
(no `.weight` suffix, and the matrix multiplication order is reversed).

This module tries every common variant so it works with adapters produced by
different PEFT versions without requiring configuration.

Merge formula
-------------
Linear:    W_out = W  +  scale * (lora_B @ lora_A)
Embedding: W_out = W  +  scale * (lora_A @ lora_B)

where  scale = user_scale * (lora_alpha / r).

The computation is always done in float32 for numerical stability, then cast
back to the original weight dtype.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter config
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    """Parsed subset of adapter_config.json that is needed for merging."""

    r: int = 8
    lora_alpha: float = 8.0
    target_modules: list[str] = field(default_factory=list)
    adapter_name: str = "default"

    @property
    def default_scale(self) -> float:
        """The standard alpha/r scaling factor."""
        return self.lora_alpha / self.r

    @classmethod
    def from_file(cls, path: Path) -> "LoRAConfig":
        with open(path) as f:
            cfg = json.load(f)
        return cls(
            r=int(cfg.get("r", 8)),
            lora_alpha=float(cfg.get("lora_alpha", 8.0)),
            target_modules=list(cfg.get("target_modules", [])),
        )

    @classmethod
    def default(cls) -> "LoRAConfig":
        return cls()


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------

# All prefix / suffix combinations we try, in priority order.
_PREFIXES = ["base_model.model.", ""]

# (lora_A suffix, lora_B suffix) — plain linear variants
def _linear_variants(adapter_name: str) -> list[tuple[str, str]]:
    return [
        (f".lora_A.{adapter_name}.weight", f".lora_B.{adapter_name}.weight"),
        (".lora_A.weight", ".lora_B.weight"),
        (f".lora_A.{adapter_name}", f".lora_B.{adapter_name}"),
        (".lora_A", ".lora_B"),
    ]

# Embedding layers use a different suffix (no .weight) and swapped multiply
def _embedding_variants(adapter_name: str) -> list[tuple[str, str]]:
    return [
        (f".lora_embedding_A.{adapter_name}", f".lora_embedding_B.{adapter_name}"),
        (".lora_embedding_A", ".lora_embedding_B"),
    ]


def find_lora_keys(
    base_key: str,
    adapter_key_set: set[str],
    adapter_name: str = "default",
) -> Optional[tuple[str, str, bool]]:
    """
    Search *adapter_key_set* for the lora_A and lora_B tensors that correspond
    to *base_key*.

    Returns (a_key, b_key, is_embedding) or None if no match is found.
    """
    # Strip a trailing .weight to get the module path stem
    stem = base_key.removesuffix(".weight")

    for prefix in _PREFIXES:
        # Try linear variants first
        for a_suf, b_suf in _linear_variants(adapter_name):
            a_key = f"{prefix}{stem}{a_suf}"
            b_key = f"{prefix}{stem}{b_suf}"
            if a_key in adapter_key_set and b_key in adapter_key_set:
                return a_key, b_key, False

        # Try embedding variants
        for a_suf, b_suf in _embedding_variants(adapter_name):
            a_key = f"{prefix}{stem}{a_suf}"
            b_key = f"{prefix}{stem}{b_suf}"
            if a_key in adapter_key_set and b_key in adapter_key_set:
                return a_key, b_key, True

    return None


# ---------------------------------------------------------------------------
# Merge math
# ---------------------------------------------------------------------------


def merge_lora(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
    is_embedding: bool = False,
) -> torch.Tensor:
    """
    Fuse LoRA weights into *weight* and return the merged tensor.

    The merge is computed in float32 for numerical stability and the result
    is cast back to *weight*'s original dtype before returning.

    Args:
        weight:       Base weight tensor  (out_features, in_features).
        lora_a:       LoRA A matrix       (r, in_features)  for linear,
                                          (num_emb, r)      for embedding.
        lora_b:       LoRA B matrix       (out_features, r) for linear,
                                          (r, emb_dim)      for embedding.
        scale:        Combined scale = user_scale * (alpha / r).
        is_embedding: If True, use embedding multiply order (A @ B).
    """
    orig_dtype = weight.dtype
    w = weight.float()
    a = lora_a.float()
    b = lora_b.float()

    if is_embedding:
        # A: (num_embeddings, r),  B: (r, embedding_dim)
        delta = a @ b
    else:
        # A: (r, in_features),    B: (out_features, r)
        delta = b @ a

    return (w + scale * delta).to(orig_dtype)
