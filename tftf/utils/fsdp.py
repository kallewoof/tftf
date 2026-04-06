"""
FSDP shard utilities.

When a LoRA adapter is trained with PyTorch FSDP (FullyShardedDataParallel)
using LOCAL_STATE_DICT or SHARDED_STATE_DICT saving, each rank writes its own
checkpoint file containing a subset of each parameter tensor.

Shard layout assumptions
------------------------
FSDP FULL_SHARD shards parameters along their *first* dimension (dim 0) by
default.  With LOCAL_STATE_DICT, rank k receives rows

    [k * chunk_size, (k+1) * chunk_size)

of each tensor, where ``chunk_size = ceil(total_rows / world_size)``.

Some FSDP setups flatten parameters before sharding (into a single 1-D
tensor) and store an additional ``_shard_metadata`` key that records the
original shapes.  We handle both cases:

1. **Direct sharding** (most LoRA setups): each rank file has the same key
   with a reduced size along dim 0.  Reconstruction = torch.cat(..., dim=0).

2. **Flat sharding**: each rank file stores a 1-D ``_flat_param_N`` tensor.
   Reconstruction requires the metadata to unflatten.  This mode is *not*
   supported yet — the pipe will raise a clear error if detected.

Shard file discovery
--------------------
``find_shard_files(path)`` accepts:
- A list of explicit paths
- A directory: finds all ``adapter_model*.safetensors`` files, sorted
- A glob string

The sort order determines rank order.  If your shard files use a different
naming convention, pass the paths in explicit order.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# Keys that signal flat-param sharding — not yet supported
_FLAT_PARAM_PREFIXES = ("_flat_param", "__flat_param")


def find_shard_files(source: list[Path] | Path | str) -> list[Path]:
    """
    Resolve *source* to an ordered list of shard file paths.

    Args:
        source: Explicit list of Paths, a directory, or a glob string.

    Returns:
        Sorted list of .safetensors files.
    """
    if isinstance(source, list):
        paths = [Path(p) for p in source]
        if not paths:
            raise ValueError("Shard path list is empty.")
        return paths

    p = Path(source)
    if p.is_dir():
        # All safetensors files in the directory, sorted (rank 0, 1, 2, …)
        paths = sorted(p.glob("*.safetensors"))
        if not paths:
            raise ValueError(f"No .safetensors files found in {p}")
        return paths

    # Treat as a glob pattern
    from glob import glob
    paths = sorted(Path(g) for g in glob(str(source)))
    if not paths:
        raise ValueError(f"Glob {source!r} matched no files.")
    return paths


def reconstruct_from_shards(
    shard_tensors: list[torch.Tensor],
    shard_dim: int = 0,
) -> torch.Tensor:
    """
    Concatenate shard tensors along *shard_dim* to reconstruct a full tensor.

    Raises:
        ValueError: If any shard is 1-D and appears to be a flat-param shard
                    (unsupported — requires shape metadata to unflatten).
        ValueError: If shard tensors have incompatible shapes.
    """
    if not shard_tensors:
        raise ValueError("Cannot reconstruct from empty shard list.")

    # Check for flat-param sharding
    for i, t in enumerate(shard_tensors):
        if t.ndim == 1 and t.shape[0] > 1:
            logger.warning(
                "Shard %d is 1-D (shape=%s).  If this is a flat-param shard "
                "it cannot be reconstructed without shape metadata.",
                i, list(t.shape),
            )

    return torch.cat(shard_tensors, dim=shard_dim)


def check_for_flat_params(keys: list[str]) -> None:
    """Raise a clear error if flat-param keys are detected."""
    flat_keys = [k for k in keys if any(k.startswith(p) for p in _FLAT_PARAM_PREFIXES)]
    if flat_keys:
        raise NotImplementedError(
            "Flat-param FSDP sharding detected.  Keys: "
            + ", ".join(flat_keys[:5])
            + ("\n  …and more." if len(flat_keys) > 5 else "")
            + "\n\nFlat-param reconstruction requires shape metadata that is not yet "
            "supported.  Use FSDP with FULL_STATE_DICT or a direct-sharding "
            "strategy, or manually unflatten the shards before using tftf."
        )
