"""
Pipeline — orchestrates the two-phase streaming workflow.

Phase 1  (metadata scan)
    Run pipe.process_meta() over the reader's TensorMeta stream.
    Collect the output list of TensorMeta and hand it to the writer's
    prepare() call.  This writes the safetensors header to disk.
    No tensor data is loaded.

Phase 2  (data write)
    Run pipe.setup(), then stream TensorRecords from the reader through
    pipe.process(), writing each output record to the writer immediately
    before loading the next tensor.  Finishes with pipe.teardown().

Only one tensor is in Python memory at any point during Phase 2.
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Protocol, runtime_checkable

from tqdm import tqdm

from tftf.io.writer import StreamingWriter
from tftf.pipes.base import Pipe, TensorMeta, TensorRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reader protocol — satisfied by SafetensorsReader and ShardedSafetensorsReader
# ---------------------------------------------------------------------------


@runtime_checkable
class ReaderProtocol(Protocol):
    """
    Duck-type interface for all model readers.

    Both SafetensorsReader (single file) and ShardedSafetensorsReader
    (multi-shard via index.json) satisfy this protocol.
    """

    def iter_meta(self) -> Iterator[TensorMeta]: ...
    def iter_records(self) -> Iterator[TensorRecord]: ...
    def metadata(self) -> dict[str, str]: ...
    def num_tensors(self) -> int: ...


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """
    Connects a Reader → Pipe chain → StreamingWriter.

    The reader can be any object satisfying ReaderProtocol:
    - ``SafetensorsReader``        — single .safetensors file
    - ``ShardedSafetensorsReader`` — multi-shard model via index.json

    Example::

        from tftf.io.sharded_reader import ShardedSafetensorsReader
        from tftf.io.writer import StreamingWriter
        from tftf.pipes.lora_merge import LoRAMergePipe
        from tftf.pipes.dtype_cast import DTypeCastPipe
        from tftf.pipeline import Pipeline
        import torch

        reader = ShardedSafetensorsReader.from_path("./llama-70b/")
        pipe   = LoRAMergePipe("adapter_model.safetensors") | DTypeCastPipe(torch.float16)
        writer = StreamingWriter("merged.safetensors")

        Pipeline(reader, pipe, writer).run()

    Args:
        reader:   Source of tensors (any ReaderProtocol implementor).
        pipe:     Pipe (or CompoundPipe) to apply.
        writer:   Destination file writer.
    """

    def __init__(
        self,
        reader: ReaderProtocol,
        pipe: Pipe,
        writer: StreamingWriter,
    ) -> None:
        self.reader = reader
        self.pipe = pipe
        self.writer = writer

    def run(
        self,
        show_progress: bool = True,
        progress_desc: Optional[str] = None,
    ) -> None:
        """
        Execute the two-pass pipeline.

        Args:
            show_progress: Show a tqdm progress bar during Phase 2.
            progress_desc: Override the progress bar label.
        """
        # ----------------------------------------------------------------
        # Phase 1: metadata scan — no tensor data loaded
        # ----------------------------------------------------------------
        reader_label = getattr(self.reader, "path", getattr(self.reader, "index_path", "reader"))
        logger.info("[1/2] Scanning tensor metadata from %s …", reader_label)

        meta_stream = self.reader.iter_meta()
        output_metas = list(self.pipe.process_meta(meta_stream))
        n = len(output_metas)

        logger.info("[1/2] Output will contain %d tensors.", n)

        file_meta = self.reader.metadata()
        self.writer.prepare(output_metas, file_metadata=file_meta or None)

        # ----------------------------------------------------------------
        # Phase 2: stream tensor data
        # ----------------------------------------------------------------
        logger.info("[2/2] Streaming tensors …")
        self.pipe.setup()

        try:
            record_stream = self.reader.iter_records()
            output_stream = self.pipe.process(record_stream)

            if show_progress:
                desc = progress_desc or self.writer.path.name
                output_stream = tqdm(
                    output_stream,
                    total=n,
                    unit="tensor",
                    desc=desc,
                    dynamic_ncols=True,
                )

            for record in output_stream:
                self.writer.write_record(record)
                # Drop reference to tensor so GC can reclaim it before
                # the next tensor is loaded from the mmap.
                del record

        finally:
            # Always call teardown, even if an exception was raised
            self.pipe.teardown()

        self.writer.finalize()
        logger.info("Done → %s", self.writer.path)
