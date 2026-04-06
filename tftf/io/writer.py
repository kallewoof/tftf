"""
StreamingWriter — write a safetensors file one tensor at a time.

The safetensors format requires the complete header (with tensor offsets)
to appear at the beginning of the file, *before* any data.  This creates
a chicken-and-egg problem for streaming: you cannot write the header
without knowing every output tensor's name, shape, dtype, and byte offset.

Two-phase solution
------------------

Phase 1 — prepare(metas)
    Accept a list of TensorMeta, compute all byte offsets, encode the JSON
    header, and write it to the output file.  No tensor data is loaded.

Phase 2 — write_record(record)
    Append raw tensor bytes sequentially.  The file position is always
    maintained; write_record() must be called in the same order as the
    metas passed to prepare().

finalize()
    Flush and close the file handle.

Memory profile
--------------
Only one tensor is in memory at a time.  The header is a small JSON object
(a few KiB even for multi-billion-parameter models).

safetensors binary layout
--------------------------
  [8 bytes]       uint64 LE   header_size
  [header_size]   UTF-8 JSON  tensor metadata + optional __metadata__
  [rest of file]  raw tensor data, concatenated in header order

data_offsets in the JSON are byte offsets *within* the data section
(i.e. relative to the byte immediately after the header), NOT the file.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import torch

from tftf.pipes.base import TensorMeta, TensorRecord


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype helpers
# ---------------------------------------------------------------------------

_TORCH_TO_ST: dict[torch.dtype, str] = {
    torch.float64:  "F64",
    torch.float32:  "F32",
    torch.float16:  "F16",
    torch.bfloat16: "BF16",
    torch.int64:    "I64",
    torch.int32:    "I32",
    torch.int16:    "I16",
    torch.int8:     "I8",
    torch.uint8:    "U8",
    torch.bool:     "BOOL",
}

_DTYPE_ITEMSIZE: dict[torch.dtype, int] = {
    torch.float64:  8,
    torch.float32:  4,
    torch.float16:  2,
    torch.bfloat16: 2,
    torch.int64:    8,
    torch.int32:    4,
    torch.int16:    2,
    torch.int8:     1,
    torch.uint8:    1,
    torch.bool:     1,
}


def _nbytes(dtype: torch.dtype, shape: torch.Size) -> int:
    numel = 1
    for d in shape:
        numel *= d
    return numel * _DTYPE_ITEMSIZE[dtype]


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class StreamingWriter:
    """
    Two-phase streaming writer for the safetensors format.

    The caller must invoke methods in order::

        writer = StreamingWriter("output.safetensors")
        writer.prepare(metas, file_metadata={...})   # Phase 1
        for record in output_stream:
            writer.write_record(record)              # Phase 2
        writer.finalize()
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._key_order: list[str] = []
        self._fh = None  # opened in prepare()

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def prepare(
        self,
        metas: list[TensorMeta],
        file_metadata: dict[str, str] | None = None,
    ) -> None:
        """
        Encode the safetensors header and write it to the output file.

        After this call the file exists on disk with the correct header.
        write_record() appends tensor data.

        Args:
            metas:         Ordered list of TensorMeta for every output tensor.
            file_metadata: Optional key→value strings for the
                           ``__metadata__`` section.
        """
        offset = 0
        header: dict = {}

        if file_metadata:
            header["__metadata__"] = {
                str(k): str(v) for k, v in file_metadata.items()
            }

        for meta in metas:
            try:
                dtype_str = _TORCH_TO_ST[meta.dtype]
            except KeyError:
                raise ValueError(
                    f"Tensor {meta.key!r} has unsupported dtype {meta.dtype}. "
                    f"Supported: {list(_TORCH_TO_ST)}"
                )

            nbytes = _nbytes(meta.dtype, meta.shape)
            header[meta.key] = {
                "dtype": dtype_str,
                "shape": list(meta.shape),
                "data_offsets": [offset, offset + nbytes],
            }
            offset += nbytes
            self._key_order.append(meta.key)

        # Encode header JSON
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

        # Pad to 8-byte boundary with ASCII spaces (matches reference impl)
        remainder = len(header_bytes) % 8
        if remainder:
            header_bytes += b" " * (8 - remainder)

        # Write: 8-byte length prefix + header
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)

        logger.debug(
            "Wrote header: %d tensors, %d bytes header, %.3f GiB data expected",
            len(metas),
            len(header_bytes) + 8,
            offset / 1024**3,
        )

        # Open in append-binary mode for the data phase
        self._fh = open(self.path, "ab")

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def write_record(self, record: TensorRecord) -> None:
        """
        Append one tensor's raw bytes to the file.

        The tensor is made contiguous in memory, then its bytes are written
        via a uint8 view (works for all dtypes including bfloat16, which
        NumPy does not natively support).
        """
        if self._fh is None:
            raise RuntimeError("Call prepare() before write_record().")

        tensor = record.tensor.contiguous()
        # View as uint8 to get raw bytes — universal across all dtypes
        raw: bytes = tensor.view(torch.uint8).numpy().tobytes()
        self._fh.write(raw)
        del raw  # free immediately

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Flush and close the output file."""
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
            logger.info("Wrote %s", self.path)
