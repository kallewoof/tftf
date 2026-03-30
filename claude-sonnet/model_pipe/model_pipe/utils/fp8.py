"""
fp8.py — utilities for fine-grained FP8 dequantisation.

Format
------
DeepSeek-V3 (and similar models) store linear-layer weights in the
``torch.float8_e4m3fn`` dtype alongside a companion ``weight_scale_inv``
tensor that encodes per-block dequantisation scales.

Naming convention (HuggingFace / DeepSeek layout)
---------------------------------------------------
  some.layer.weight           float8_e4m3fn  (out_features, in_features)
  some.layer.weight_scale_inv float32        (⌈out/128⌉, ⌈in/128⌉)

Block layout
------------
The weight matrix is divided into non-overlapping 128×128 blocks.
Each block has one float32 scale stored in ``weight_scale_inv``.

Dequantisation formula
-----------------------
  W_dequant[r:r+128, c:c+128] = W_fp8[r:r+128, c:c+128].float()
                                 * weight_scale_inv[r//128, c//128]

where ``r`` and ``c`` are the block row/col start indices.

When ``out_features`` or ``in_features`` is not a multiple of 128 the last
block is a partial block; the formula still applies — the scale covers
whatever portion of the 128-row/column window is actually present.

Padding note
------------
DeepSeek's quantiser zero-pads weights to the nearest 128 boundary before
computing scales, then stores the *unpadded* weight.  When dequantising we
must therefore handle non-multiple-of-128 dimensions correctly by operating
on the actual (possibly smaller) slice rather than a padded view.

Companion key detection
-----------------------
``scale_inv_key_for(weight_key)`` → ``weight_key + "_scale_inv"``
``weight_key_for_scale(scale_key)`` → strip ``"_scale_inv"`` suffix
``is_scale_inv_key(key)``          → key ends with ``"_scale_inv"``

Alternative naming
------------------
Some checkpoints use ``weight_scale`` instead of ``weight_scale_inv``.
We detect both.  When the tensor is named ``weight_scale`` (not ``_inv``)
the scale is applied as division rather than multiplication:

  W_dequant = W_fp8.float() / weight_scale[block_row, block_col]

The ``dequantize_fp8_weight`` function accepts an ``invert_scale`` flag
that callers should set based on which key name they observed.
"""

from __future__ import annotations

import torch

# -------------------------------------------------------------------------
# Key naming helpers
# -------------------------------------------------------------------------

_SCALE_INV_SUFFIX = "_scale_inv"
_SCALE_SUFFIX     = "_scale"
_FP8_DTYPES       = {torch.float8_e4m3fn, torch.float8_e4m3fnuz,
                     torch.float8_e5m2,   torch.float8_e5m2fnuz}

# Fallback for older PyTorch that may not have all fp8 variants
try:
    _FP8_DTYPES.add(torch.float8_e4m3fn)
except AttributeError:
    pass


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    """Return True if *dtype* is any supported FP8 variant."""
    return dtype in _FP8_DTYPES


def is_scale_key(key: str) -> bool:
    """Return True if *key* is a weight_scale_inv or weight_scale companion key."""
    return key.endswith(_SCALE_INV_SUFFIX) or key.endswith(_SCALE_SUFFIX)


def weight_key_for_scale(scale_key: str) -> str:
    """
    Derive the base weight key from a scale key.

    ``some.layer.weight_scale_inv`` → ``some.layer.weight``
    ``some.layer.weight_scale``     → ``some.layer.weight``
    """
    if scale_key.endswith(_SCALE_INV_SUFFIX):
        return scale_key[: -len(_SCALE_INV_SUFFIX)]
    if scale_key.endswith(_SCALE_SUFFIX):
        return scale_key[: -len(_SCALE_SUFFIX)]
    raise ValueError(f"Not a scale key: {scale_key!r}")


def scale_inv_key_for(weight_key: str) -> str:
    """Return the expected ``weight_scale_inv`` companion key for *weight_key*."""
    return weight_key + _SCALE_INV_SUFFIX


def scale_key_for(weight_key: str) -> str:
    """Return the expected ``weight_scale`` companion key for *weight_key*."""
    return weight_key + _SCALE_SUFFIX


# -------------------------------------------------------------------------
# Dequantisation
# -------------------------------------------------------------------------

_BLOCK = 128


def dequantize_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.bfloat16,
    invert_scale: bool = False,
    block_size: int = _BLOCK,
) -> torch.Tensor:
    """
    Dequantise a fine-grained FP8 weight tensor to *target_dtype*.

    Args:
        weight:       FP8 weight tensor, shape ``(out_features, in_features)``.
        scale:        Per-block scale, shape ``(⌈out/block⌉, ⌈in/block⌉)``.
                      dtype must be float32 (or will be cast to float32).
        target_dtype: Output dtype — typically ``torch.bfloat16`` or
                      ``torch.float16``.  The computation always uses
                      float32 internally regardless of this value.
        invert_scale: If ``True`` the scale is applied as *division*
                      (i.e. it is a ``weight_scale`` not ``weight_scale_inv``).
                      Default ``False`` → multiply (standard ``weight_scale_inv``).
        block_size:   Block dimension.  Default 128 (DeepSeek convention).

    Returns:
        Dequantised weight as a contiguous tensor in *target_dtype*.

    Raises:
        ValueError: If weight is not a 2-D FP8 tensor, or scale shape is
                    inconsistent with weight shape.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"FP8 dequantisation expects a 2-D weight tensor; "
            f"got shape {tuple(weight.shape)}"
        )
    if not is_fp8_dtype(weight.dtype):
        raise ValueError(
            f"Expected an FP8 dtype; got {weight.dtype}.  "
            f"Supported: {_FP8_DTYPES}"
        )

    out_f, in_f = weight.shape
    scale_f32 = scale.float()

    n_row_blocks = (out_f + block_size - 1) // block_size
    n_col_blocks = (in_f + block_size - 1) // block_size

    # Validate scale shape — tolerate transposed scale (some checkpoints)
    expected = (n_row_blocks, n_col_blocks)
    if scale_f32.shape == expected[::-1] and scale_f32.shape != expected:
        # Transposed scale — rare but documented in some conversions
        scale_f32 = scale_f32.t().contiguous()

    if scale_f32.shape != expected:
        raise ValueError(
            f"Scale shape {tuple(scale_f32.shape)} is inconsistent with "
            f"weight shape {tuple(weight.shape)} and block_size={block_size}. "
            f"Expected scale shape: {expected}"
        )

    # Allocate output in float32; cast to target_dtype at the end
    out = torch.empty(out_f, in_f, dtype=torch.float32, device=weight.device)

    # Process one 128×128 block at a time — avoids allocating the whole
    # weight in float32 simultaneously when iterating over blocks.
    w_f32 = weight.float()  # unavoidable: FP8→float32 for the whole tensor,
    # but this is at most ~(out*in) bytes of float32 which is still just the
    # single weight matrix — no worse than DTypeCastPipe for this tensor.

    for br in range(n_row_blocks):
        r0 = br * block_size
        r1 = min(r0 + block_size, out_f)
        for bc in range(n_col_blocks):
            c0 = bc * block_size
            c1 = min(c0 + block_size, in_f)
            s = scale_f32[br, bc]
            if invert_scale:
                out[r0:r1, c0:c1] = w_f32[r0:r1, c0:c1] / s
            else:
                out[r0:r1, c0:c1] = w_f32[r0:r1, c0:c1] * s

    del w_f32
    return out.to(target_dtype)
