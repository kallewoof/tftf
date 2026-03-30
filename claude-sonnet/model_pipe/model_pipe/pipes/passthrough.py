"""PassthroughPipe — identity transformation, useful for testing and benchmarking."""

from __future__ import annotations

from typing import Iterator

from model_pipe.pipes.base import Pipe, TensorMeta, TensorRecord


class PassthroughPipe(Pipe):
    """
    Passes every tensor through unchanged.

    Useful as:
    - A baseline to benchmark the streaming I/O layer.
    - A starting point when composing pipes (passthrough | some_other_pipe).
    - A safe default when no transformation is needed.
    """

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        yield from records

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        yield from metas

    def __repr__(self) -> str:
        return "PassthroughPipe()"
