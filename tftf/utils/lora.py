"""
LoRA / DoRA key-mapping utilities and merge math.

Key naming conventions
----------------------
HuggingFace PEFT writes adapter weights with a ``base_model.model.`` prefix
and a ``.lora_A.weight`` / ``.lora_B.weight`` suffix.

For embedding layers the keys are ``.lora_embedding_A`` / ``.lora_embedding_B``
(no ``.weight`` suffix; multiply order is reversed).

DoRA adapters additionally store a magnitude vector under the key
``.lora_magnitude_vector.{adapter_name}.weight`` (or without the adapter name
in some PEFT versions).

This module tries every common variant so it works with adapters produced by
different PEFT versions without requiring configuration.

DoRA / QDoRA detection
----------------------
DoRA presence is detected by looking up a magnitude vector key for the base
weight.  If one is found, ``merge_dora()`` is used; otherwise the standard
``merge_lora()`` is applied.  QDoRA (DoRA trained on a quantized base model)
uses the identical merge formula — the quantisation was only active during
training.

Merge formulas
--------------

LoRA — Linear  (weight ndim == 2):
    delta = lora_B @ lora_A        # (out, r) @ (r, in) → (out, in)
    W_out = W + scale * delta

LoRA — Embedding  (weight ndim == 2, is_embedding=True):
    delta = lora_A @ lora_B        # (num_emb, r) @ (r, emb_dim)
    W_out = W + scale * delta

LoRA — Convolutional  (weight ndim >= 3):
    Flatten spatial dims, 2-D matmul, reshape back.
    W_out = W + scale * delta

DoRA — Linear / Embedding / Convolutional:
    Same delta computation as LoRA, then:
        W_merged   = W + scale * delta
        weight_norm = ||W_merged|| per output channel  (row-wise for 2-D,
                      all-but-first-dim for conv)
        W_out = (m / weight_norm).unsqueeze(-1...) * W_merged

    where m is the learned magnitude vector of shape (out_channels,).

All computation is done in float32 for numerical stability.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

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
    # PEFT allows ``target_modules`` to be either a list of module names
    # (matched by exact/suffix) OR a single string that is treated as a
    # regular expression matched against the full module name.  We preserve
    # whichever form the adapter used — see ``matches_module``.
    target_modules: Union[list[str], str] = field(default_factory=list)
    # PEFT ``target_parameters`` — LoRA applied directly to an nn.Parameter
    # rather than a module (used for MoE grouped-expert weights, e.g.
    # ``experts.gate_up_proj``).  These are stacked-expert LoRA tensors.
    target_parameters: list[str] = field(default_factory=list)
    adapter_name: str = "default"

    @property
    def default_scale(self) -> float:
        """The standard alpha/r scaling factor."""
        return self.lora_alpha / self.r

    def matches_module(self, module_name: str) -> bool:
        """
        PEFT ``target_modules`` matching semantics.

        - Empty ``target_modules`` → matches everything (no pre-filter; the
          authoritative check is whether an adapter tensor actually exists).
        - String ``target_modules`` → treated as a regex, matched with
          ``re.fullmatch`` against the full module name.
        - List ``target_modules`` → matches if the module name equals, or ends
          with ``.<entry>`` for, any listed entry.
        """
        tm = self.target_modules
        if not tm:
            return True
        if isinstance(tm, str):
            return re.fullmatch(tm, module_name) is not None
        return any(module_name == m or module_name.endswith(f".{m}") for m in tm)

    def matched_parameter(self, key: str) -> Optional[str]:
        """
        Return the ``target_parameters`` entry that *key* corresponds to, or
        ``None``.  A base key matches an entry if it equals it or ends with
        ``.<entry>`` (entries are relative parameter paths like
        ``experts.gate_up_proj``).
        """
        for tp in self.target_parameters:
            if key == tp or key.endswith(f".{tp}"):
                return tp
        return None

    @classmethod
    def from_file(cls, path: Path) -> LoRAConfig:
        with open(path) as f:
            cfg = json.load(f)
        # target_modules may legitimately be a str (regex) or a list — keep the
        # raw form.  (A previous version called ``list(...)`` on it, which
        # silently exploded a regex string into individual characters and made
        # the merge a no-op.)
        target_modules = cfg.get("target_modules") or []
        if not isinstance(target_modules, (str, list)):
            raise ValueError(
                f"adapter_config.json: 'target_modules' must be a string or list, "
                f"got {type(target_modules).__name__}"
            )
        target_parameters = cfg.get("target_parameters") or []
        if not isinstance(target_parameters, list):
            raise ValueError(
                f"adapter_config.json: 'target_parameters' must be a list, "
                f"got {type(target_parameters).__name__}"
            )
        return cls(
            r=int(cfg.get("r", 8)),
            lora_alpha=float(cfg.get("lora_alpha", 8.0)),
            target_modules=target_modules,
            target_parameters=list(target_parameters),
        )

    @classmethod
    def default(cls) -> LoRAConfig:
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


def _magnitude_variants(adapter_name: str) -> list[str]:
    """DoRA magnitude vector key suffixes, in priority order."""
    return [
        f".lora_magnitude_vector.{adapter_name}.weight",
        ".lora_magnitude_vector.weight",
        f".lora_magnitude_vector.{adapter_name}",
        ".lora_magnitude_vector",
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


def find_grouped_lora_pairs(
    module_path: str,
    adapter_key_set: set[str],
    adapter_name: str = "default",
) -> list[tuple[str, str]]:
    """
    Find every ``(a_key, b_key)`` LoRA pair attached to *module_path* via PEFT's
    ``target_parameters`` mechanism (used for MoE grouped-expert weights).

    Unlike ``find_lora_keys`` — which reconstructs the adapter key from a base
    *weight* key — ``target_parameters`` LoRA hangs off the *module* that owns
    the parameter, not a ``.weight`` sub-key.  When several parameters of one
    module are targeted, PEFT nests the wrappers, so the adapter keys look like::

        base_model.model.<module_path>.lora_A.weight              (param 1)
        base_model.model.<module_path>.base_layer.lora_A.weight   (param 2)

    Both reduce to *module_path* once the ``base_model.model.`` prefix and any
    trailing ``.base_layer`` wrappers are stripped.  Returns all such pairs;
    the caller disambiguates which pair fuses into which parameter by shape.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for prefix in _PREFIXES:
        for a_suf, b_suf in _linear_variants(adapter_name):
            for a_key in adapter_key_set:
                if not a_key.endswith(a_suf):
                    continue
                if prefix and not a_key.startswith(prefix):
                    continue

                core = a_key[len(prefix):-len(a_suf)] if prefix else a_key[:-len(a_suf)]
                reduced = core
                while reduced.endswith(".base_layer"):
                    reduced = reduced[: -len(".base_layer")]
                if reduced != module_path:
                    continue

                b_key = f"{prefix}{core}{b_suf}"
                pair = (a_key, b_key)
                if b_key in adapter_key_set and pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)

    return pairs


def find_magnitude_key(
    base_key: str,
    adapter_key_set: set[str],
    adapter_name: str = "default",
) -> Optional[str]:
    """
    Search *adapter_key_set* for the DoRA magnitude vector that corresponds
    to *base_key*.

    Returns the matched key string, or ``None`` if the adapter is plain LoRA
    (no magnitude vector).
    """
    stem = base_key.removesuffix(".weight")

    for prefix in _PREFIXES:
        for suf in _magnitude_variants(adapter_name):
            mag_key = f"{prefix}{stem}{suf}"
            if mag_key in adapter_key_set:
                return mag_key

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


def merge_dora(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    magnitude: torch.Tensor,
    scale: float,
    is_embedding: bool = False,
) -> torch.Tensor:
    """
    Fuse DoRA (or QDoRA) weights into *weight* and return the merged tensor.

    DoRA decomposes weight updates into direction (LoRA A/B matrices) and
    magnitude (a per-output-channel scalar vector *m*).  The merge formula is:

        W_merged   = W + scale * delta          (standard LoRA update)
        weight_norm = ||W_merged|| per output channel
        W_out      = (m / weight_norm) * W_merged

    where *delta* follows the same convention as ``merge_lora``:

    - **Linear / Conv1D** (ndim=2, is_embedding=False):   delta = lora_B @ lora_A
    - **Embedding**       (ndim=2, is_embedding=True):    delta = lora_A @ lora_B
    - **Convolutional**   (ndim>=3):   2-D matmul on flattened spatial dims

    The norm is taken row-wise (i.e. over all dims except the first) so that
    *weight_norm* has shape ``(out_channels,)``, matching the stored magnitude
    vector shape.

    QDoRA adapters (DoRA trained on a quantised base) use the identical formula;
    the quantisation affects only training, not static merging.

    All computation is performed in float32 for numerical stability.
    The result is cast back to ``weight``'s original dtype.

    Args:
        weight:     Base model weight tensor (ndim >= 2).
        lora_a:     LoRA A matrix.
        lora_b:     LoRA B matrix.
        magnitude:  Learned magnitude vector of shape ``(out_channels,)``
                    (or a squeezable variant stored by some PEFT versions).
        scale:      Effective scale factor (lora_alpha / r * user_scale).
        is_embedding: If True, use embedding multiply order (A @ B).

    Raises:
        ValueError: If ``weight.ndim`` < 2.
    """
    orig_dtype = weight.dtype

    w = weight.float()
    a = lora_a.float()
    b = lora_b.float()
    m = magnitude.float().squeeze()  # normalise storage quirks from conv

    ndim = weight.ndim

    if ndim == 2:
        if is_embedding:
            delta = a @ b
        else:
            delta = b @ a
    elif ndim >= 3:
        out_channels = b.shape[0]
        r = b.shape[1]
        kernel_shape = a.shape[2:]
        in_channels = a.shape[1]

        kernel_numel = 1
        for s in kernel_shape:
            kernel_numel *= s

        a2d = a.reshape(r, in_channels * kernel_numel)
        b2d = b.reshape(out_channels, r)
        delta = (b2d @ a2d).reshape(out_channels, in_channels, *kernel_shape)
    else:
        raise ValueError(
            f"merge_dora: weight has ndim={ndim}, shape={tuple(weight.shape)}.  "
            f"Expected ndim >= 2."
        )

    w_merged = w + scale * delta

    # Per-output-channel L2 norm: reduce over all dims except the first.
    reduce_dims = tuple(range(1, w_merged.dim()))
    weight_norm = w_merged.norm(p=2, dim=reduce_dims).clamp(min=1e-8)  # (out_channels,)

    mag_norm_scale = m / weight_norm  # (out_channels,)

    # Broadcast (out_channels,) to the full weight shape
    view_shape = (mag_norm_scale.shape[0],) + (1,) * (w_merged.dim() - 1)
    result = mag_norm_scale.view(view_shape) * w_merged

    return result.to(orig_dtype)


def merge_grouped_lora(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Fuse PEFT ``target_parameters`` (grouped-expert) LoRA into a 3-D stacked
    expert weight and return the merged tensor.

    A stacked-expert parameter has shape ``(num_experts, d1, d2)``.  PEFT stores
    a single LoRA per parameter with the per-expert ranks concatenated:

        lora_A : (num_experts * r, in_features)      # experts stacked row-wise
        lora_B : (out_features, num_experts * r)     # experts interleaved col-wise

    The reshape convention is taken **verbatim** from PEFT's
    ``ParamWrapper.get_delta_weight`` (peft/tuners/lora/layer.py) so that the
    fused result matches what PEFT would compute at inference:

        A → (num_experts, r, in_features)            # expert index is *outer*
        B → (out_features, r, num_experts)           # expert index is *inner*
        delta[e] = (B[:, :, e] @ A[e]) * scale       # standard LoRA, per expert

    The einsum output orientation is chosen to match *weight*'s layout:
    ``(E, out, in)`` (the common MoE ``is_transposed`` case) or ``(E, in, out)``.

    All computation is done in float32; the result is cast back to *weight*'s
    original dtype.

    Raises:
        ValueError: if *weight* is not 3-D, if the A/B ranks are inconsistent,
            or if the A/B shapes cannot be reconciled with *weight*'s shape.
    """
    if weight.ndim != 3:
        raise ValueError(
            f"merge_grouped_lora: expected a 3-D stacked-expert weight, got "
            f"ndim={weight.ndim}, shape={tuple(weight.shape)}."
        )

    orig_dtype = weight.dtype
    w = weight.float()
    a = lora_a.float()  # (E*r, in)
    b = lora_b.float()  # (out, E*r)

    num_experts = w.shape[0]
    er = a.shape[0]

    if b.shape[1] != er:
        raise ValueError(
            f"merge_grouped_lora: lora_A rows ({er}) and lora_B cols "
            f"({b.shape[1]}) disagree on num_experts*rank; "
            f"A={tuple(a.shape)}, B={tuple(b.shape)}."
        )
    if er % num_experts != 0:
        raise ValueError(
            f"merge_grouped_lora: stacked rank {er} is not divisible by "
            f"num_experts {num_experts} (A={tuple(a.shape)}, "
            f"weight={tuple(weight.shape)})."
        )

    r = er // num_experts
    in_features = a.shape[1]
    out_features = b.shape[0]

    # Replicate PEFT's exact reshapes (expert-outer for A, expert-inner for B).
    a3 = a.reshape(num_experts, r, in_features)   # "e r i"
    b3 = b.reshape(out_features, r, num_experts)  # "o r e"

    if tuple(w.shape) == (num_experts, out_features, in_features):
        delta = torch.einsum("o r e, e r i -> e o i", b3, a3)
    elif tuple(w.shape) == (num_experts, in_features, out_features):
        delta = torch.einsum("o r e, e r i -> e i o", b3, a3)
    else:
        raise ValueError(
            f"merge_grouped_lora: cannot reconcile weight shape "
            f"{tuple(weight.shape)} with LoRA (experts={num_experts}, "
            f"out={out_features}, in={in_features})."
        )

    return (w + scale * delta).to(orig_dtype)
