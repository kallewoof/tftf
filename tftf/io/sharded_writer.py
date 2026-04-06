"""
ShardedWriter — two-phase streaming writer for sharded safetensors output.

Problem
-------
Large merged models (e.g. 70B parameters in bfloat16 = ~140 GiB) may not
fit in a single file on many filesystems (FAT32: 4 GiB limit; some network
filesystems have practical limits).  More importantly, HuggingFace
transformers and PEFT expect sharded models in the standard format:

    model-00001-of-00005.safetensors
    model-00002-of-00005.safetensors
    …
    model.safetensors.index.json

ShardedWriter produces exactly this layout, keeping the same two-phase
streaming design as StreamingWriter so the full model is never in RAM.

Design
------
Phase 1 — prepare(metas)
    Walk the output TensorMeta list in order.  Greedily assign tensors to
    shards: start a new shard whenever the current shard would exceed
    max_shard_bytes.  After all tensors are assigned:
    1. Write each shard file's safetensors header (no tensor data yet).
    2. Write model.safetensors.index.json.

    The shard filenames use a final-count placeholder that is resolved once
    all metas have been walked, so they always look like
    ``model-00001-of-00005.safetensors``.

Phase 2 — write_record(record)
    Each tensor is appended to the currently active shard file in sequence.
    When a shard is full (according to the plan from Phase 1), that file
    is closed and the next shard file is opened.

finalize()
    Flush and close the last shard file.

Memory profile
--------------
One tensor in RAM at a time.  Shard headers are small JSON (~a few KiB each).

Single-shard degeneration
--------------------------
If all tensors fit in one shard, ShardedWriter produces a single
``model-00001-of-00001.safetensors`` plus an ``index.json``.  Use
``StreamingWriter`` if you want a bare ``.safetensors`` with no index.
"""

from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

import torch

from tftf.io.writer import _TORCH_TO_ST, StreamingWriter, _nbytes
from tftf.pipes.base import TensorMeta, TensorRecord


logger = logging.getLogger(__name__)


class ShardedWriter(StreamingWriter):
    """
    Two-phase streaming writer that distributes output across multiple shards.

    Args:
        output_dir:         Directory to write shard files into (created if absent).
        max_shard_bytes:    Soft upper bound per shard in bytes.
                            A single tensor that exceeds this limit is placed
                            alone in its own shard (no tensor is split).
                            Default: 5 GiB.
        filename_stem:      Prefix for shard filenames, e.g. ``"model"`` →
                            ``model-00001-of-00003.safetensors``.
        index_filename:     Name of the index file written into *output_dir*.
    """

    DEFAULT_MAX_SHARD_BYTES = 20 * 1024**3  # 20 GiB

    def __init__(
        self,
        path: Path | str,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        filename_stem: str = "model",
        index_filename: str = "model.safetensors.index.json",
    ) -> None:
        super().__init__(path)
        self.max_shard_bytes = max_shard_bytes
        self.filename_stem = filename_stem
        self.index_filename = index_filename

        # Populated by prepare()
        self._shard_metas: list[list[TensorMeta]] = []   # [shard_idx][tensor_idx]
        self._shard_paths: list[Path] = []
        self._weight_map: dict[str, str] = {}            # key → shard filename

        # Phase 2 state
        self._current_shard_idx: int = 0
        self._current_tensor_idx: int = 0  # index within the current shard

    # ------------------------------------------------------------------
    # Phase 1 — assign tensors to shards and write headers
    # ------------------------------------------------------------------

    def prepare(
        self,
        metas: list[TensorMeta],
        file_metadata: dict[str, str] | None = None,
    ) -> None:
        """
        Assign tensors to shards, write all shard headers, write index.json.

        Args:
            metas:         Ordered list of output TensorMeta.
            file_metadata: Optional key→value pairs for each shard's
                           ``__metadata__`` section (same dict for all shards).
        """
        self.path.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------------------
        # Assign metas to shards (greedy bin-packing by byte size)
        # ----------------------------------------------------------------
        shard_groups: list[list[TensorMeta]] = []
        current_group: list[TensorMeta] = []
        current_bytes = 0

        for meta in metas:
            tensor_bytes = _nbytes(meta.dtype, meta.shape)
            # Start a new shard if this tensor would overflow — unless the
            # current shard is empty (single oversized tensor gets its own shard)
            if current_group and current_bytes + tensor_bytes > self.max_shard_bytes:
                shard_groups.append(current_group)
                current_group = []
                current_bytes = 0
            current_group.append(meta)
            current_bytes += tensor_bytes

        if current_group:
            shard_groups.append(current_group)

        n_shards = len(shard_groups)
        self._shard_metas = shard_groups

        logger.info(
            "ShardedWriter: %d tensors → %d shards  (max %.2f GiB/shard)",
            len(metas),
            n_shards,
            self.max_shard_bytes / 1024**3,
        )

        # ----------------------------------------------------------------
        # Compute shard filenames (need n_shards to know zero-padding width)
        # ----------------------------------------------------------------
        width = max(5, len(str(n_shards)))  # at least 5 digits like HF
        shard_paths = []
        for i in range(n_shards):
            name = (
                f"{self.filename_stem}-"
                f"{i+1:0{width}d}-of-{n_shards:0{width}d}.safetensors"
            )
            shard_paths.append(self.path / name)
        self._shard_paths = shard_paths

        # ----------------------------------------------------------------
        # Build weight_map and compute total data size
        # ----------------------------------------------------------------
        total_size = 0
        for shard_idx, (group, path) in enumerate(zip(shard_groups, shard_paths)):
            for meta in group:
                self._weight_map[meta.key] = path.name
                total_size += _nbytes(meta.dtype, meta.shape)

        # ----------------------------------------------------------------
        # Write each shard's safetensors header
        # ----------------------------------------------------------------
        for shard_idx, (group, path) in enumerate(zip(shard_groups, shard_paths)):
            self._write_shard_header(group, path, file_metadata)
            logger.debug(
                "Shard %d/%d: %s  (%d tensors)",
                shard_idx + 1, n_shards, path.name, len(group),
            )

        # ----------------------------------------------------------------
        # Write model.safetensors.index.json
        # ----------------------------------------------------------------
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": self._weight_map,
        }
        index_path = self.path / self.index_filename
        index_path.write_text(json.dumps(index, indent=2))
        logger.info("Wrote %s", index_path)

        # ----------------------------------------------------------------
        # Open the first shard for appending tensor data
        # ----------------------------------------------------------------
        if self._shard_paths:
            self._fh = open(self._shard_paths[0], "ab")
            self._current_shard_idx = 0
            self._current_tensor_idx = 0

    # ------------------------------------------------------------------
    # Phase 2 — append tensor bytes to the correct shard
    # ------------------------------------------------------------------

    def write_record(self, record: TensorRecord) -> None:
        """
        Append one tensor to the current shard file.

        Automatically advances to the next shard file when the current
        shard's tensor list is exhausted.
        """
        if self._fh is None:
            raise RuntimeError("Call prepare() before write_record().")

        # Advance to next shard if current shard is full
        current_group = self._shard_metas[self._current_shard_idx]
        if self._current_tensor_idx >= len(current_group):
            self._fh.flush()
            self._fh.close()
            self._current_shard_idx += 1
            self._current_tensor_idx = 0
            self._fh = open(self._shard_paths[self._current_shard_idx], "ab")

        tensor = record.tensor.contiguous()
        raw: bytes = tensor.view(torch.uint8).numpy().tobytes()
        self._fh.write(raw)
        del raw
        self._current_tensor_idx += 1

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Flush and close the final shard file."""
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None
        n = len(self._shard_paths)
        logger.info(
            "ShardedWriter: wrote %d shard(s) to %s", n, self.path
        )

    # ------------------------------------------------------------------
    # Internal: write one shard's safetensors header
    # ------------------------------------------------------------------

    @staticmethod
    def _write_shard_header(
        metas: list[TensorMeta],
        path: Path,
        file_metadata: dict[str, str] | None,
    ) -> None:
        """Encode and write the safetensors header for one shard."""
        offset = 0
        header: dict = {}

        if file_metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in file_metadata.items()}

        for meta in metas:
            dtype_str = _TORCH_TO_ST[meta.dtype]
            nb = _nbytes(meta.dtype, meta.shape)
            header[meta.key] = {
                "dtype": dtype_str,
                "shape": list(meta.shape),
                "data_offsets": [offset, offset + nb],
            }
            offset += nb

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        remainder = len(header_bytes) % 8
        if remainder:
            header_bytes += b" " * (8 - remainder)

        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes)))
            f.write(header_bytes)
