"""
LoRAMergeBase — shared implementation of the LoRA merge Pipe interface.

Both LoRAMergePipe (single adapter file) and FSDPShardMergePipe (per-rank
shard files) perform the identical merge operation once the full lora_A /
lora_B tensors are assembled.  Previously this logic was copy-pasted between
the two classes.  This base class holds the shared implementation; subclasses
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

from model_pipe.pipes.base import Pipe, TensorMeta, TensorRecord
from model_pipe.utils.lora import LoRAConfig, find_lora_keys, merge_lora

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
        """
        if self._config is None:
            raise RuntimeError(
                f"{type(self).__name__}: setup() must be called before process().  "
                "This is handled automatically by Pipeline.run()."
            )

        effective_scale = self.scale * self._config.default_scale

        for record in records:
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

            merged = merge_lora(
                weight=record.tensor.to(self.device),
                lora_a=lora_a,
                lora_b=lora_b,
                scale=effective_scale,
                is_embedding=is_embedding,
            )
            logger.debug(
                "Merged %s → %s  [%s]  scale=%.6f",
                type(self).__name__,
                record.key,
                "embedding" if is_embedding else "linear",
                effective_scale,
            )

            yield TensorRecord(
                key=record.key,
                tensor=merged.cpu(),
                extra=record.extra,
            )
            del merged, lora_a, lora_b

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
