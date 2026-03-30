"""DTypeCastPipe — cast tensors to a target dtype during the stream."""

from __future__ import annotations

from typing import Iterator

import torch

from model_pipe.pipes.base import Pipe, TensorMeta, TensorRecord


class DTypeCastPipe(Pipe):
    """
    Cast every tensor to *target_dtype* as it passes through the pipeline.

    This is useful for e.g. merging a LoRA in float32 and saving the result
    in float16:

        pipe = LoRAMergePipe(...) | DTypeCastPipe(torch.float16)

    process_meta() overrides the dtype so the writer allocates the correct
    number of bytes per tensor in the output header.
    """

    def __init__(self, target_dtype: torch.dtype) -> None:
        self.target_dtype = target_dtype

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        for meta in metas:
            yield TensorMeta(
                key=meta.key,
                dtype=self.target_dtype,
                shape=meta.shape,
                extra=meta.extra,
            )

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        for record in records:
            casted = record.tensor.to(self.target_dtype)
            yield TensorRecord(key=record.key, tensor=casted, extra=record.extra)
            del casted

    def __repr__(self) -> str:
        dtype_str = str(self.target_dtype).replace('torch.', '')
        return f"DTypeCastPipe({dtype_str!r})"
