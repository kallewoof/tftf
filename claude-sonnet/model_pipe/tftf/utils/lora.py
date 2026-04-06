"""
LoRA key-mapping utilities and merge math.

Key naming conventions
----------------------
HuggingFace PEFT writes adapter weights with a ``base_model.model.`` prefix
and a ``.lora_A.weight`` / ``.lora_B.weight`` suffix.

For embedding layers the keys are ``.lora_embedding_A`` / ``.lora_embedding_B``
(no ``.weight`` suffix; multiply order is reversed).

This module tries every common variant so it works with adapters produced by
different PEFT versions without requiring configuration.

Merge formulas
--------------

Linear  (weight ndim == 2):
    delta = lora_B @ lora_A        # (out, r) @ (r, in) → (out, in)
    W_out = W + scale * delta

Embedding  (weight ndim == 2, but flagged is_embedding=True):
    delta = lora_A @ lora_B        # (num_emb, r) @ (r, emb_dim) → (num_emb, emb_dim)
    W_out = W + scale * delta

Convolutional  (weight ndim >= 3 — Conv1d / Conv2d / Conv3d):
    All PEFT convolutional LoRA variants follow the same pattern regardless
    of kernel dimensionality:

        lora_A  shape: (r,  in_channels, *kernel_size)  — full-kernel conv
        lora_B  shape: (out_channels, r, *ones)         — pointwise 1×…×1 conv

    The delta is computed via a 2-D matmul on flattened spatial dims:

        a2d   = lora_A.reshape(r, in_channels * prod(kernel_size))
        b2d   = lora_B.reshape(out_channels, r)
        delta = (b2d @ a2d).reshape(out_channels, in_channels, *kernel_size)

    This is equivalent to PEFT's conv-based formula for Conv2d and generalises
    naturally to Conv1d (ndim=3), Conv2d (ndim=4), and Conv3d (ndim=5).

All computation is done in float32 for numerical stability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

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


def _linear_variants(adapter_name: str) -> list[tuple[str, str]]:
    """(lora_A suffix, lora_B suffix) pairs for plain linear layers."""
    return [
        (f".lora_A.{adapter_name}.weight", f".lora_B.{adapter_name}.weight"),
        (".lora_A.weight", ".lora_B.weight"),
        (f".lora_A.{adapter_name}", f".lora_B.{adapter_name}"),
        (".lora_A", ".lora_B"),
    ]


def _embedding_variants(adapter_name: str) -> list[tuple[str, str]]:
    """(lora_A suffix, lora_B suffix) pairs for embedding layers."""
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

    Returns ``(a_key, b_key, is_embedding)`` or ``None`` if no match is found.

    The ``is_embedding`` flag tells ``merge_lora`` to use the reversed
    multiply order (A @ B instead of B @ A).  Conv2d layers use the same
    key suffixes as linear layers but are distinguished at merge time by
    weight ``ndim``.
    """
    # Strip a trailing .weight to get the module path stem
    stem = base_key.removesuffix(".weight")

    for prefix in _PREFIXES:
        # Try linear variants first (also used for Conv2d)
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

    Handles all layer types used by PEFT LoRA:

    - **Linear / Conv1D**  (ndim=2, is_embedding=False):
          delta = lora_B @ lora_A
    - **Embedding**        (ndim=2, is_embedding=True):
          delta = lora_A @ lora_B
    - **Convolutional**    (ndim >= 3 — Conv1d, Conv2d, Conv3d):
          Flatten spatial dims, 2-D matmul, reshape back.
          lora_A: (r, in_channels, *kernel_size)
          lora_B: (out_channels, r, *ones)
          delta = (b2d @ a2d).reshape(out_channels, in_channels, *kernel_size)

    All computation is performed in float32 for numerical stability.
    The result is cast back to ``weight``\'s original dtype before returning.

    Raises:
        ValueError: If ``weight.ndim`` < 2.
    """
    orig_dtype = weight.dtype

    w = weight.float()
    a = lora_a.float()
    b = lora_b.float()

    ndim = weight.ndim

    if ndim == 2:
        if is_embedding:
            # a: (num_embeddings, r),  b: (r, embedding_dim)
            delta = a @ b
        else:
            # a: (r, in_features),  b: (out_features, r)
            delta = b @ a

    elif ndim >= 3:
        # Convolutional weight: (out_channels, in_channels, *kernel_size)
        # PEFT lora_A: (r, in_channels, *kernel_size)
        # PEFT lora_B: (out_channels, r, *ones)
        #
        # Generalised matmul: flatten all dims after the first of A and B,
        # do a 2-D matmul, then reshape back to the weight shape.
        # Identical for Conv1d (ndim=3), Conv2d (ndim=4), Conv3d (ndim=5).
        out_channels = b.shape[0]
        r            = b.shape[1]
        kernel_shape = a.shape[2:]          # (k,) or (kH, kW) or (kD, kH, kW)
        in_channels  = a.shape[1]

        kernel_numel = 1
        for s in kernel_shape:
            kernel_numel *= s

        a2d = a.reshape(r, in_channels * kernel_numel)
        b2d = b.reshape(out_channels, r)

        delta = (b2d @ a2d).reshape(out_channels, in_channels, *kernel_shape)

    else:
        raise ValueError(
            f"merge_lora: weight has ndim={ndim}, shape={tuple(weight.shape)}.  "
            f"Expected ndim >= 2."
        )

    return (w + scale * delta).to(orig_dtype)
