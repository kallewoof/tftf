"""
LoRAMergeBase — shared implementation of the LoRA merge Pipe interface.

Both LoRAMergePipe (single adapter file) and DCPLoRAMergePipe (DCP checkpoint)
perform the identical merge operation once the full lora_A / lora_B tensors
are assembled.  This base class holds the shared implementation; subclasses
only need to implement setup() to populate _lora_weights and _config.

Required contract for subclasses
---------------------------------
setup() must populate:
    self._config        : LoRAConfig
    self._lora_weights  : dict[str, torch.Tensor]   — full reconstructed weights
    self._adapter_key_set: set[str]                 — set(self._lora_weights)

teardown() clears them via super().teardown() (or a manual clear — see below).
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Iterator, Optional

import torch

from tftf.pipes.base import Pipe, TensorMeta, TensorRecord
from tftf.utils.lora import (
    LoRAConfig,
    find_grouped_lora_pairs,
    find_lora_keys,
    find_magnitude_key,
    merge_dora,
    merge_grouped_lora,
    merge_lora,
)


logger = logging.getLogger(__name__)


class LoRAMergeBase(Pipe):
    """
    Abstract base that provides process() and process_meta() for LoRA merging.

    Subclasses must implement setup() to fill the three attributes below.
    """

    # Populated by subclass setup()
    _config: Optional[LoRAConfig]
    _lora_weights: dict[str, torch.Tensor]
    _adapter_key_set: set[str]

    def __init__(
        self,
        adapter_name: str = "default",
        scale: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.adapter_name = adapter_name
        self.scale = scale
        self.device = device

        # Initialise to empty; subclass setup() fills them
        self._config = None
        self._lora_weights = {}
        self._adapter_key_set = set()

    # ------------------------------------------------------------------
    # Abstract — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self) -> None:
        """Load / reconstruct all LoRA tensors and populate the attributes."""
        ...

    # ------------------------------------------------------------------
    # Shared teardown
    # ------------------------------------------------------------------

    def teardown(self) -> None:
        self._lora_weights.clear()
        self._adapter_key_set.clear()
        self._config = None

    # ------------------------------------------------------------------
    # Shared Pipe interface
    # ------------------------------------------------------------------

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        """
        Pass base-model metas through unchanged.

        LoRA A/B keys are absorbed during the merge and never appear in
        the output, so the output tensor set is identical to the input.
        Shapes and dtypes are preserved (merge is dtype-stable).
        """
        yield from metas

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        """
        Merge LoRA weights into each matching base tensor as it streams past.

        Tensors that have no corresponding LoRA adapter are passed through
        unchanged.  All bookkeeping is done in float32 for numerical stability
        and the result is cast back to the original dtype.

        Two flavours of adapter are handled:

        - **Module LoRA** (``target_modules``): standard ``.lora_A`` / ``.lora_B``
          attached to a ``.weight`` (Linear / Embedding / Conv, and DoRA).
        - **Parameter LoRA** (``target_parameters``): stacked-expert LoRA
          applied directly to a 3-D MoE parameter (e.g. ``experts.gate_up_proj``).

        After the stream is exhausted, ``_validate_merge`` fails loudly if the
        merge was a no-op or if adapter weights went unused — catching silent
        "output identical to base model" bugs.
        """
        if self._config is None:
            raise RuntimeError(
                f"{type(self).__name__}: setup() must be called before process().  "
                "This is handled automatically by Pipeline.run()."
            )

        effective_scale = self.scale * self._config.default_scale
        has_target_modules = bool(self._config.target_modules)

        # ---- guardrail bookkeeping ----
        adapter_pairs = list(self._iter_adapter_pairs())
        consumed_a_keys: set[str] = set()
        n_merged = 0
        n_module_matched = 0  # base tensors that passed the target_modules filter

        for record in records:
            # ----------------------------------------------------------------
            # (1) target_parameters (grouped-expert) merge.  Independent of
            #     target_modules — PEFT resolves these separately, and the
            #     parameter owner (e.g. "…experts") never matches a linear
            #     target_modules pattern.
            # ----------------------------------------------------------------
            if self._config.matched_parameter(record.key) is not None and record.tensor.ndim == 3:
                module_path = record.key.rsplit(".", 1)[0]
                pairs = find_grouped_lora_pairs(
                    module_path, self._adapter_key_set, self.adapter_name
                )
                selected = self._select_grouped_pair(pairs, record.tensor.shape)
                if len(selected) > 1:
                    raise RuntimeError(
                        f"{type(self).__name__}: ambiguous target_parameters LoRA for "
                        f"{record.key!r}: {len(selected)} adapter pairs match its shape "
                        f"{tuple(record.tensor.shape)}.  Candidates: "
                        f"{[a for a, _ in selected]}"
                    )
                if len(selected) == 1:
                    a_key, b_key = selected[0]
                    lora_a = self._lora_weights[a_key].to(self.device)
                    lora_b = self._lora_weights[b_key].to(self.device)
                    merged = merge_grouped_lora(
                        weight=record.tensor.to(self.device),
                        lora_a=lora_a,
                        lora_b=lora_b,
                        scale=effective_scale,
                    )
                    logger.debug(
                        "Merged grouped-expert LoRA %s → %s  scale=%.6f",
                        type(self).__name__,
                        record.key,
                        effective_scale,
                    )
                    consumed_a_keys.add(a_key)
                    n_merged += 1
                    yield TensorRecord(
                        key=record.key, tensor=merged.cpu(), extra=record.extra
                    )
                    del merged, lora_a, lora_b
                    continue
                # No matching pair → fall through; left unconsumed and reported.

            # ----------------------------------------------------------------
            # (2) target_modules pre-filter for standard module LoRA.
            # ----------------------------------------------------------------
            if has_target_modules:
                stem = record.key.removesuffix(".weight")
                if not self._config.matches_module(stem):
                    yield record
                    continue
            n_module_matched += 1

            match = find_lora_keys(
                record.key,
                self._adapter_key_set,
                self.adapter_name,
            )

            if match is None:
                yield record
                continue

            a_key, b_key, is_embedding = match
            lora_a = self._lora_weights[a_key].to(self.device)
            lora_b = self._lora_weights[b_key].to(self.device)

            mag_key = find_magnitude_key(
                record.key,
                self._adapter_key_set,
                self.adapter_name,
            )

            if mag_key is not None:
                magnitude = self._lora_weights[mag_key].to(self.device)
                merged = merge_dora(
                    weight=record.tensor.to(self.device),
                    lora_a=lora_a,
                    lora_b=lora_b,
                    magnitude=magnitude,
                    scale=effective_scale,
                    is_embedding=is_embedding,
                )
                logger.debug(
                    "Merged DoRA %s → %s  [%s]  scale=%.6f",
                    type(self).__name__,
                    record.key,
                    "embedding" if is_embedding else "linear",
                    effective_scale,
                )
            else:
                merged = merge_lora(
                    weight=record.tensor.to(self.device),
                    lora_a=lora_a,
                    lora_b=lora_b,
                    scale=effective_scale,
                    is_embedding=is_embedding,
                )
                logger.debug(
                    "Merged LoRA %s → %s  [%s]  scale=%.6f",
                    type(self).__name__,
                    record.key,
                    "embedding" if is_embedding else "linear",
                    effective_scale,
                )

            consumed_a_keys.add(a_key)
            n_merged += 1
            yield TensorRecord(
                key=record.key,
                tensor=merged.cpu(),
                extra=record.extra,
            )
            del merged, lora_a, lora_b

        # ---- guardrails: run after the whole stream has been consumed ----
        self._validate_merge(
            adapter_pairs=adapter_pairs,
            consumed_a_keys=consumed_a_keys,
            n_merged=n_merged,
            n_module_matched=n_module_matched,
            has_target_modules=has_target_modules,
        )

    # ------------------------------------------------------------------
    # Guardrail helpers
    # ------------------------------------------------------------------

    def _iter_adapter_pairs(self):
        """
        Yield ``(a_key, b_key, base_stem, is_parameter)`` for every LoRA A/B pair
        present in the adapter.

        ``base_stem`` is the base-model path of the module (or parameter owner)
        the pair targets — used to decide whether an *unmerged* pair is a real
        failure or was legitimately excluded by ``target_modules``.
        """
        for a_key in self._adapter_key_set:
            if ".lora_A" in a_key:
                b_key = a_key.replace(".lora_A", ".lora_B")
            elif ".lora_embedding_A" in a_key:
                b_key = a_key.replace(".lora_embedding_A", ".lora_embedding_B")
            else:
                continue
            if b_key not in self._adapter_key_set:
                continue
            base_stem = self._adapter_key_to_base_stem(a_key)
            yield a_key, b_key, base_stem, self._is_parameter_stem(base_stem)

    @staticmethod
    def _adapter_key_to_base_stem(a_key: str) -> str:
        """Map an adapter ``lora_A`` key back to the base-model module path."""
        core = a_key
        for prefix in ("base_model.model.",):
            if core.startswith(prefix):
                core = core[len(prefix):]
                break
        for marker in (".lora_A", ".lora_embedding_A"):
            idx = core.find(marker)
            if idx != -1:
                core = core[:idx]
                break
        while core.endswith(".base_layer"):
            core = core[: -len(".base_layer")]
        return core

    def _is_parameter_stem(self, base_stem: str) -> bool:
        """True if *base_stem* owns a ``target_parameters`` entry."""
        assert self._config is not None
        for tp in self._config.target_parameters:
            if "." not in tp:
                continue
            tp_module = tp.rsplit(".", 1)[0]
            if base_stem == tp_module or base_stem.endswith(f".{tp_module}"):
                return True
        return False

    def _select_grouped_pair(
        self, pairs: list[tuple[str, str]], shape
    ) -> list[tuple[str, str]]:
        """
        From candidate ``target_parameters`` LoRA pairs, return those whose
        stacked A/B shapes reconcile with a 3-D expert weight *shape*.
        """
        num_experts = shape[0]
        matches: list[tuple[str, str]] = []
        for a_key, b_key in pairs:
            a = self._lora_weights[a_key]
            b = self._lora_weights[b_key]
            if a.ndim != 2 or b.ndim != 2:
                continue
            er = a.shape[0]
            if b.shape[1] != er or num_experts == 0 or er % num_experts != 0:
                continue
            in_f, out_f = a.shape[1], b.shape[0]
            if tuple(shape) in (
                (num_experts, out_f, in_f),
                (num_experts, in_f, out_f),
            ):
                matches.append((a_key, b_key))
        return matches

    def _validate_merge(
        self,
        *,
        adapter_pairs: list,
        consumed_a_keys: set,
        n_merged: int,
        n_module_matched: int,
        has_target_modules: bool,
    ) -> None:
        """
        Fail loudly on a merge that silently did nothing (or nearly nothing).

        Checks, in order:
          1. The adapter must contain at least one recognizable LoRA pair.
          2. At least one base tensor must have been merged (else the output is
             byte-identical to the base model — the classic no-op bug).
          3. If ``target_modules`` is set, it must have matched at least one
             base tensor (catches a mis-parsed / wrong pattern).
          4. Every adapter pair that *should* have merged (a parameter pair, or a
             module pair whose base module matches ``target_modules``) must have
             been consumed.
        """
        assert self._config is not None

        if not adapter_pairs:
            raise RuntimeError(
                f"{type(self).__name__}: the adapter contains no recognizable "
                f"LoRA weights (no '.lora_A'/'.lora_B' pairs among "
                f"{len(self._adapter_key_set)} tensors).  Is this a valid PEFT "
                f"adapter?"
            )

        if n_merged == 0:
            examples = [stem for _, _, stem, _ in adapter_pairs[:3]]
            raise RuntimeError(
                f"{type(self).__name__}: LoRA merge matched ZERO base tensors — "
                f"the output would be identical to the base model.  The adapter "
                f"defines {len(adapter_pairs)} LoRA modules (e.g. {examples}) but "
                f"none lined up with a base weight.  Likely causes: a "
                f"target_modules / target_parameters mismatch, a key-naming "
                f"difference, or the wrong base model."
            )

        if has_target_modules and n_module_matched == 0 and not self._config.target_parameters:
            raise RuntimeError(
                f"{type(self).__name__}: target_modules matched no base tensors "
                f"({self._config.target_modules!r}).  Check the pattern — a regex "
                f"string must fully match module names."
            )

        # Pairs that ought to have merged but did not.
        genuine_misses = []
        filtered_misses = []
        for a_key, _b_key, base_stem, is_param in adapter_pairs:
            if a_key in consumed_a_keys:
                continue
            should_merge = is_param or self._config.matches_module(base_stem)
            (genuine_misses if should_merge else filtered_misses).append(base_stem)

        if filtered_misses:
            logger.info(
                "%s: %d adapter module(s) skipped by target_modules (expected): %s",
                type(self).__name__,
                len(filtered_misses),
                sorted(set(filtered_misses))[:5],
            )

        if genuine_misses:
            raise RuntimeError(
                f"{type(self).__name__}: {len(genuine_misses)} adapter LoRA "
                f"module(s) were never merged despite targeting a present base "
                f"weight: {sorted(set(genuine_misses))[:8]}.  This indicates a "
                f"key-mapping failure (or an upstream pipe dropped those base "
                f"tensors before the merge)."
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._lora_weights)
        loaded = f"{n} tensors loaded" if n else "not loaded"
        return (
            f"{type(self).__name__}("
            f"adapter_name={self.adapter_name!r}, "
            f"scale={self.scale}, "
            f"device={self.device!r}, "
            f"{loaded})"
        )
