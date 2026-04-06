"""
fp8.py — utilities for fine-grained FP8 dequantisation.

Format
------
DeepSeek-V3 / DeepSeek-R1 store linear weights as ``torch.float8_e4m3fn``
alongside a companion ``weight_scale_inv`` tensor that holds per-block
dequantisation scales:

    model.layers.N.self_attn.q_proj.weight            float8_e4m3fn  (out, in)
    model.layers.N.self_attn.q_proj.weight_scale_inv  float32        (⌈out/128⌉, ⌈in/128⌉)

Dequantisation formula
-----------------------
  W_dequant[r:r+B, c:c+B] = W_fp8[r:r+B, c:c+B].float()
                             * weight_scale_inv[r//B, c//B]

where B = block_size (128 by default).

Vectorised implementation
--------------------------
The naive Python loop over all (n_row_blocks × n_col_blocks) blocks is
O(n²) in the number of blocks and extremely slow for large matrices.  For
a 7168×7168 weight that is 56×56 = 3136 Python iterations per tensor.

The vectorised path pads the weight to the nearest block boundary, reshapes
to (n_rb, B, n_cb, B), transposes to (n_rb, n_cb, B, B), multiplies by the
(n_rb, n_cb, 1, 1) scale broadcast, then slices back to the original shape.
This is a single BLAS-level element-wise multiply — orders of magnitude
faster.

Scale naming
-------------
``weight_scale_inv``  →  multiply  (standard DeepSeek convention)
``weight_scale``      →  divide    (some other checkpoints)

The ``invert_scale`` argument to ``dequantize_fp8_weight`` selects which.

Key helpers
-----------
``scale_inv_key_for(w)``      → ``w + "_scale_inv"``
``scale_key_for(w)``          → ``w + "_scale"``
``weight_key_for_scale(s)``   → strip suffix
``is_scale_key(k)``           → k ends with ``".weight_scale_inv"`` or
                                 ``".weight_scale"`` (anchored to avoid
                                 false positives like ``rope_scale``)
``is_fp8_dtype(d)``           → True for any float8 variant
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# FP8 dtype set — built defensively to handle PyTorch < 2.1
# ---------------------------------------------------------------------------

_FP8_DTYPES: set[torch.dtype] = set()
for _name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz"):
    _attr = getattr(torch, _name, None)
    if _attr is not None:
        _FP8_DTYPES.add(_attr)

HAS_FP8 = len(_FP8_DTYPES) > 0


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    """Return True if *dtype* is any recognised FP8 variant."""
    return dtype in _FP8_DTYPES


# ---------------------------------------------------------------------------
# Key naming helpers
# ---------------------------------------------------------------------------

_SCALE_INV_SUFFIX = ".weight_scale_inv"
_SCALE_SUFFIX     = ".weight_scale"

# The raw suffixes without the leading dot, for stripping
_SCALE_INV_BARE = "weight_scale_inv"
_SCALE_BARE     = "weight_scale"


def is_scale_key(key: str) -> bool:
    """
    Return True if *key* is a weight_scale_inv or weight_scale companion.

    Anchored to ``.weight_scale_inv`` / ``.weight_scale`` to avoid false
    positives from unrelated keys like ``rope_scale`` or ``input_layernorm``.
    A bare ``weight_scale_inv`` (no dot prefix) is also accepted for
    top-level keys.
    """
    return (
        key.endswith(_SCALE_INV_SUFFIX)
        or key.endswith(_SCALE_SUFFIX)
        or key == _SCALE_INV_BARE
        or key == _SCALE_BARE
    )


def weight_key_for_scale(scale_key: str) -> str:
    """
    Derive the base weight key from a scale key.

    ``model.layer.weight_scale_inv`` → ``model.layer.weight``
    ``model.layer.weight_scale``     → ``model.layer.weight``
    """
    if scale_key.endswith(_SCALE_INV_SUFFIX):
        # e.g. "model.layers.0.q_proj.weight_scale_inv"
        #   → "model.layers.0.q_proj.weight"
        base = scale_key[: -len(_SCALE_INV_BARE)]  # strips "weight_scale_inv"
        return base + "weight"
    if scale_key.endswith(_SCALE_SUFFIX):
        base = scale_key[: -len(_SCALE_BARE)]
        return base + "weight"
    # Bare top-level scale key
    if scale_key == _SCALE_INV_BARE:
        return "weight"
    if scale_key == _SCALE_BARE:
        return "weight"
    raise ValueError(f"Not a recognised scale key: {scale_key!r}")


def scale_inv_key_for(weight_key: str) -> str:
    """
    Return the expected ``weight_scale_inv`` companion key.

    ``model.layer.weight`` → ``model.layer.weight_scale_inv``
    """
    # Replace trailing ".weight" with ".weight_scale_inv"
    if weight_key.endswith(".weight"):
        return weight_key + "_scale_inv"
    # No ".weight" suffix — just append
    return weight_key + _SCALE_INV_SUFFIX


def scale_key_for(weight_key: str) -> str:
    """Return the expected ``weight_scale`` companion key."""
    if weight_key.endswith(".weight"):
        return weight_key + "_scale"
    return weight_key + _SCALE_SUFFIX


# ---------------------------------------------------------------------------
# Dequantisation — vectorised
# ---------------------------------------------------------------------------

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

    Uses a vectorised broadcast-multiply rather than a Python block loop,
    making it orders of magnitude faster for large matrices.

    Args:
        weight:       FP8 weight tensor, shape ``(out_features, in_features)``.
        scale:        Per-block scale tensor, shape ``(⌈out/B⌉, ⌈in/B⌉)``,
                      dtype float32.
        target_dtype: Output dtype — ``torch.bfloat16``, ``torch.float16``,
                      or ``torch.float32``.
        invert_scale: ``False`` (default) → multiply by scale
                      (``weight_scale_inv`` convention).
                      ``True`` → divide by scale (``weight_scale`` convention).
        block_size:   Block dimension B.  Default 128.

    Returns:
        Dequantised weight, contiguous, in *target_dtype*.

    Raises:
        ValueError: If weight is not 2-D, not an FP8 dtype, or the scale
                    shape is inconsistent with the weight shape.
    """
    if weight.ndim != 2:
        raise ValueError(
            f"FP8 dequantisation expects a 2-D weight tensor; "
            f"got ndim={weight.ndim}, shape={tuple(weight.shape)}"
        )
    if not is_fp8_dtype(weight.dtype):
        raise ValueError(
            f"Expected an FP8 dtype; got {weight.dtype}.  "
            f"Supported FP8 dtypes: {_FP8_DTYPES or '(none — upgrade PyTorch to ≥2.1)'}"
        )

    out_f, in_f = weight.shape
    B = block_size

    n_rb = (out_f + B - 1) // B  # number of row blocks
    n_cb = (in_f  + B - 1) // B  # number of col blocks

    # ---- Validate / normalise scale shape ----------------------------------
    scale_f32 = scale.float()
    expected_shape = (n_rb, n_cb)

    if scale_f32.shape != expected_shape:
        # Tolerate transposed scales produced by some conversion scripts
        if scale_f32.shape == (n_cb, n_rb):
            scale_f32 = scale_f32.t().contiguous()
        else:
            raise ValueError(
                f"Scale shape {tuple(scale_f32.shape)} is inconsistent with "
                f"weight shape {tuple(weight.shape)} and block_size={B}. "
                f"Expected: {expected_shape}."
            )

    # ---- Vectorised dequantisation -----------------------------------------
    # Pad weight to an exact multiple of B in both dimensions
    pad_r = (B - out_f % B) % B
    pad_c = (B - in_f  % B) % B

    w_f32 = weight.float()  # FP8 → float32 (unavoidable for this tensor)

    if pad_r > 0 or pad_c > 0:
        w_f32 = torch.nn.functional.pad(w_f32, (0, pad_c, 0, pad_r))

    out_r = out_f + pad_r
    out_c = in_f  + pad_c

    # Reshape: (n_rb*B, n_cb*B) → (n_rb, B, n_cb, B) → (n_rb, n_cb, B, B)
    blocked = w_f32.view(n_rb, B, n_cb, B).permute(0, 2, 1, 3)

    # scale: (n_rb, n_cb) → (n_rb, n_cb, 1, 1) for broadcast
    s = scale_f32.unsqueeze(-1).unsqueeze(-1)

    if invert_scale:
        dequant_blocked = blocked / s
    else:
        dequant_blocked = blocked * s

    # Restore original layout: (n_rb, n_cb, B, B) → (n_rb, B, n_cb, B) → (out_r, out_c)
    result_padded = dequant_blocked.permute(0, 2, 1, 3).contiguous().view(out_r, out_c)

    # Trim padding
    result = result_padded[:out_f, :in_f]

    del w_f32, blocked, dequant_blocked, result_padded
    return result.to(target_dtype).contiguous()
