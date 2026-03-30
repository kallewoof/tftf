"""
SafetensorsReader — lazy, memory-mapped reader for .safetensors files.

The reader exposes two iteration modes:

iter_meta()     Phase 1 — yields TensorMeta by reading only the JSON header.
                No tensor data is allocated.  Uses safetensors' slice API
                (get_slice) which accesses the mmap header only.

iter_records()  Phase 2 — yields TensorRecord one tensor at a time.
                Each tensor is loaded from the mmap, yielded, then our
                reference is dropped so the GC can reclaim it before the
                next tensor is loaded.

The underlying safetensors library uses OS-level mmap, so tensor data is
paged in from disk only when it is explicitly read.  Even in iter_records(),
only the current tensor's pages are resident in RAM.

Sharded models
--------------
A future ShardedSafetensorsReader will accept a model.safetensors.index.json
and present a unified iter_meta() / iter_records() view across all shards,
loading each shard file in turn.  The Pipe interface is unaffected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from model_pipe.pipes.base import TensorMeta, TensorRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dtype resolution helpers
# ---------------------------------------------------------------------------

# safetensors dtype string → torch dtype (covers all documented ST dtypes)
_ST_TO_TORCH: dict[str, torch.dtype] = {
    "F64":  torch.float64,
    "F32":  torch.float32,
    "F16":  torch.float16,
    "BF16": torch.bfloat16,
    "I64":  torch.int64,
    "U32":  torch.int32,   # unsigned is stored identically, use signed torch dtype
    "I32":  torch.int32,
    "I16":  torch.int16,
    "I8":   torch.int8,
    "U8":   torch.uint8,
    "BOOL": torch.bool,
}

_TORCH_STR_TO_DTYPE: dict[str, torch.dtype] = {
    "torch.float64":  torch.float64,
    "torch.float32":  torch.float32,
    "torch.float16":  torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.int64":    torch.int64,
    "torch.int32":    torch.int32,
    "torch.int16":    torch.int16,
    "torch.int8":     torch.int8,
    "torch.uint8":    torch.uint8,
    "torch.bool":     torch.bool,
}


def _resolve_dtype(raw: object) -> torch.dtype:
    """
    Coerce whatever get_slice().get_dtype() returns into a torch.dtype.

    Different safetensors versions / frameworks return different types:
    - A torch.dtype object (most common with framework="pt")
    - A string like "F32", "BF16" (safetensors dtype notation)
    - A string like "torch.float32"  (older bindings)
    """
    if isinstance(raw, torch.dtype):
        return raw

    s = str(raw).strip()
    if s in _ST_TO_TORCH:
        return _ST_TO_TORCH[s]
    if s in _TORCH_STR_TO_DTYPE:
        return _TORCH_STR_TO_DTYPE[s]

    # Last resort: upper-case normalisation
    upper = s.upper()
    if upper in _ST_TO_TORCH:
        return _ST_TO_TORCH[upper]

    raise ValueError(
        f"Cannot resolve dtype {raw!r}.  "
        f"Known ST types: {list(_ST_TO_TORCH)}"
    )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class SafetensorsReader:
    """
    Lazy, memory-mapped reader for a single .safetensors file.

    Args:
        path:   Path to the .safetensors file.
        device: Torch device tensors are loaded onto in iter_records().
                Use ``"cpu"`` (default) to keep everything off the GPU.
    """

    def __init__(self, path: Path | str, device: str = "cpu") -> None:
        self.path = Path(path)
        self.device = device

    # ------------------------------------------------------------------
    # Header-only queries (no tensor data)
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        """Return tensor keys in file order."""
        with safe_open(str(self.path), framework="pt", device="cpu") as f:
            return list(f.keys())

    def metadata(self) -> dict[str, str]:
        """Return the file-level metadata dict (may be empty)."""
        with safe_open(str(self.path), framework="pt", device="cpu") as f:
            return f.metadata() or {}

    def num_tensors(self) -> int:
        return len(self.keys())

    # ------------------------------------------------------------------
    # Phase 1 — metadata iteration (no tensor allocation)
    # ------------------------------------------------------------------

    def iter_meta(self) -> Iterator[TensorMeta]:
        """
        Yield TensorMeta for every tensor without allocating tensor storage.

        Uses safetensors' slice API (get_slice) which reads only the JSON
        header section of the file — no tensor bytes are accessed.
        """
        with safe_open(str(self.path), framework="pt", device="cpu") as f:
            for key in f.keys():
                try:
                    sl = f.get_slice(key)
                    shape = torch.Size(sl.get_shape())
                    dtype = _resolve_dtype(sl.get_dtype())
                except Exception:
                    # Fallback: load the tensor just for shape/dtype metadata
                    logger.debug(
                        "get_slice() failed for %r — falling back to get_tensor()", key
                    )
                    t = f.get_tensor(key)
                    shape, dtype = t.shape, t.dtype
                    del t

                yield TensorMeta(key=key, dtype=dtype, shape=shape)

    # ------------------------------------------------------------------
    # Phase 2 — tensor iteration (one tensor in RAM at a time)
    # ------------------------------------------------------------------

    def iter_records(self) -> Iterator[TensorRecord]:
        """
        Yield TensorRecord for every tensor, one at a time.

        The file is kept open for the duration of iteration.  After each
        yield, the caller should process / write the tensor and let the
        reference go out of scope; this reader drops its own reference
        immediately, so only one tensor is in RAM at a time.
        """
        with safe_open(str(self.path), framework="pt", device=self.device) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                yield TensorRecord(key=key, tensor=tensor)
                del tensor  # drop our reference before loading the next one
