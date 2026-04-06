"""
Core abstractions for the tftf pipeline.

Every stage in the pipeline is a Pipe.  A Pipe transforms a *lazy stream*
of TensorRecords, consuming one tensor at a time so the full model is never
in memory simultaneously.

The two-pass design
-------------------
Before any tensor data is written, the pipeline performs a cheap metadata
scan (Phase 1) so it can write the safetensors file header — which requires
knowing every output tensor's name, shape, and dtype upfront.

Phase 1  calls  pipe.process_meta(iter[TensorMeta])  → iter[TensorMeta]
Phase 2  calls  pipe.process(iter[TensorRecord])     → iter[TensorRecord]

TensorMeta carries only shape/dtype (no allocation).
TensorRecord carries the actual tensor.

Extending
---------
Subclass Pipe, implement process(), and optionally override process_meta()
if your pipe changes tensor keys, shapes, or dtypes.

Future: FSDPShardMergePipe
--------------------------
FSDP produces one .safetensors shard per rank.  A future FSDPShardMergePipe
will accept a list of shard paths, reconstruct the full parameter tensors
on-the-fly (via concatenation / all-gather), and yield them in base-model
key order.  The Pipe interface is identical — only the *source* changes.
To prepare for this: keep process() purely iterator→iterator, and put any
multi-source logic in a custom Reader (see io/reader.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import torch


# ---------------------------------------------------------------------------
# Data carriers
# ---------------------------------------------------------------------------


@dataclass
class TensorMeta:
    """Lightweight descriptor for one tensor — no tensor data allocated."""

    key: str
    dtype: torch.dtype
    shape: torch.Size
    # Arbitrary extras (e.g. shard provenance, quantisation params)
    extra: dict = field(default_factory=dict)


@dataclass
class TensorRecord:
    """One in-flight tensor plus its key and optional extras."""

    key: str
    tensor: torch.Tensor
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipe base
# ---------------------------------------------------------------------------


class Pipe(ABC):
    """
    Abstract base class for all pipeline stages.

    Subclasses must implement process().  Override process_meta() only when
    the pipe changes tensor names, shapes, or dtypes (e.g. a cast pipe).
    Override setup() / teardown() for one-time initialisation and cleanup.
    """

    # ------------------------------------------------------------------
    # Core interface — implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        """
        Transform a lazy stream of TensorRecords.

        Rules:
        - Be lazy: consume one record, yield zero-or-more records.
        - Do not buffer the whole stream.
        - It is legal to drop records (filter) or add new ones (dequantise).
        - Free tensors as soon as possible (del record.tensor after yield).
        """
        ...

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        """
        Transform a lazy stream of TensorMeta without loading tensor data.

        Called during Phase 1 (header scan).  The default is the identity.
        Override when your pipe changes keys, shapes, or dtypes.
        """
        yield from metas

    # ------------------------------------------------------------------
    # Lifecycle hooks — optionally override these
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Called once before process() begins.  Use for initialisation."""

    def teardown(self) -> None:
        """Called once after process() is exhausted.  Use for cleanup."""

    # ------------------------------------------------------------------
    # Composition operator
    # ------------------------------------------------------------------

    def __or__(self, other: "Pipe") -> "CompoundPipe":
        """Syntax sugar: pipe_a | pipe_b  →  CompoundPipe([pipe_a, pipe_b])."""
        return CompoundPipe([self, other])


# ---------------------------------------------------------------------------
# Compound pipe (pipe chain)
# ---------------------------------------------------------------------------


class CompoundPipe(Pipe):
    """A sequence of pipes executed left-to-right as a single Pipe."""

    def __init__(self, pipes: list[Pipe]) -> None:
        self.pipes = list(pipes)

    def setup(self) -> None:
        for p in self.pipes:
            p.setup()

    def teardown(self) -> None:
        # Tear down in reverse order (mirrors construction)
        for p in reversed(self.pipes):
            p.teardown()

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        stream: Iterator[TensorRecord] = records
        for pipe in self.pipes:
            stream = pipe.process(stream)
        return stream

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        stream: Iterator[TensorMeta] = metas
        for pipe in self.pipes:
            stream = pipe.process_meta(stream)
        return stream

    def __or__(self, other: "Pipe") -> "CompoundPipe":
        tail = other.pipes if isinstance(other, CompoundPipe) else [other]
        return CompoundPipe(self.pipes + tail)


    def __repr__(self) -> str:
        inner = " | ".join(repr(p) for p in self.pipes)
        return f"({inner})"
