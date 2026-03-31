"""
ShardedSafetensorsReader — transparent multi-shard reader.

HuggingFace stores large models as N shard files:

    model-00001-of-00003.safetensors
    model-00002-of-00003.safetensors
    model-00003-of-00003.safetensors
    model.safetensors.index.json          ← maps key → shard filename

The index JSON looks like:

    {
      "metadata": {"total_size": 123456},
      "weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
        "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
        ...
      }
    }

ShardedSafetensorsReader presents a unified iter_meta() / iter_records()
view across all shards, loading each shard file in turn.  Only one shard
file is open at any time; within that file only one tensor is in RAM.

The key order follows the index.json weight_map order (which is the order
HuggingFace uses), grouping keys by shard so each file is opened at most once.

Auto-detection
--------------
Use ShardedSafetensorsReader.from_path(path) which accepts:
- A directory  → looks for model.safetensors.index.json inside
- An index.json file directly
- A single .safetensors file → falls back to SafetensorsReader

This makes it easy to write commands that accept either style.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

from model_pipe.io.reader import SafetensorsReader
from model_pipe.pipes.base import TensorMeta, TensorRecord

logger = logging.getLogger(__name__)

_INDEX_FILENAMES = [
    "model.safetensors.index.json",
    "model.fp16.safetensors.index.json",
    "model.bf16.safetensors.index.json",
]


def _find_index(directory: Path) -> Path | None:
    for name in _INDEX_FILENAMES:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


class ShardedSafetensorsReader(SafetensorsReader):
    """
    Unified reader over a sharded HuggingFace safetensors model.

    Args:
        index_path:  Path to the ``model.safetensors.index.json`` file.
        device:      Torch device for tensor loading (default: ``"cpu"``).
    """

    def __init__(self, index_path: Path | str, device: str = "cpu") -> None:
        self.index_path = Path(index_path)
        self.device = device
        self._base_dir = self.index_path.parent

        with open(self.index_path) as f:
            raw = json.load(f)

        self._file_metadata: dict[str, str] = raw.get("metadata", {})
        weight_map: dict[str, str] = raw["weight_map"]

        # Build an ordered mapping: shard_filename → [key, key, ...]
        # This groups keys by shard so we open each file only once.
        shard_to_keys: OrderedDict[str, list[str]] = OrderedDict()
        for key, shard_name in weight_map.items():
            shard_to_keys.setdefault(shard_name, []).append(key)

        self._shard_to_keys = shard_to_keys
        # Flat ordered list of all keys (in shard-group order)
        self._keys: list[str] = [k for keys in shard_to_keys.values() for k in keys]

        logger.debug(
            "ShardedSafetensorsReader: %d tensors across %d shards",
            len(self._keys),
            len(shard_to_keys),
        )

    # ------------------------------------------------------------------
    # Class method: smart constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_path(
        cls,
        path: Path | str,
        device: str = "cpu",
    ) -> "ShardedSafetensorsReader | SafetensorsReader":
        """
        Return the appropriate reader for *path*:

        - **directory**      → look for ``model.safetensors.index.json``
        - **index.json**     → ShardedSafetensorsReader
        - **single .safetensors** → SafetensorsReader (no sharding)

        Raises:
            FileNotFoundError: If no index or safetensors file is found.
            ValueError: If *path* is a directory without an index file.
        """
        p = Path(path)

        if p.is_dir():
            index = _find_index(p)
            if index is None:
                # Try a single model.safetensors in the directory
                single = p / "model.safetensors"
                if single.exists():
                    logger.debug("No index.json found; using single file %s", single)
                    return SafetensorsReader(single, device=device)
                raise ValueError(
                    f"Directory {p} contains no model.safetensors.index.json "
                    f"and no model.safetensors.  Cannot determine how to read it."
                )
            return cls(index, device=device)

        if p.suffix == ".json":
            return cls(p, device=device)

        # Assume single safetensors file
        return SafetensorsReader(p, device=device)

    # ------------------------------------------------------------------
    # Header-only queries
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        return list(self._keys)

    def metadata(self) -> dict[str, str]:
        return dict(self._file_metadata)

    def num_tensors(self) -> int:
        return len(self._keys)

    def shard_paths(self) -> list[Path]:
        return [self._base_dir / name for name in self._shard_to_keys]

    # ------------------------------------------------------------------
    # Phase 1 — metadata scan (no tensor data)
    # ------------------------------------------------------------------

    def iter_meta(self) -> Iterator[TensorMeta]:
        """
        Yield TensorMeta for every tensor across all shards.

        Opens each shard file once (header only), then closes it before
        moving to the next shard.
        """
        for shard_name, keys in self._shard_to_keys.items():
            shard_path = self._base_dir / shard_name
            reader = SafetensorsReader(shard_path, device="cpu")

            # Build a lookup from the shard's own meta iteration
            shard_metas: dict[str, TensorMeta] = {
                m.key: m for m in reader.iter_meta()
            }

            for key in keys:
                if key not in shard_metas:
                    raise KeyError(
                        f"Key {key!r} listed in index.json but not found "
                        f"in shard {shard_name!r}"
                    )
                yield shard_metas[key]

    # ------------------------------------------------------------------
    # Phase 2 — tensor streaming
    # ------------------------------------------------------------------

    def iter_records(self) -> Iterator[TensorRecord]:
        """
        Yield TensorRecord for every tensor across all shards.

        Each shard is opened once; tensors are fetched by key via
        safe_open.get_tensor() in index order — only one tensor is in
        RAM at a time.  The shard file is closed before the next opens.

        Why not iterate sequentially through the shard?
        -----------------------------------------------
        The index.json may list only a subset of shard keys and may
        order them differently from the file-internal order.  Sequential
        iteration would require buffering every shard tensor before
        yielding any in index order, defeating the streaming guarantee.
        Random-access via get_tensor() avoids the buffer entirely;
        safetensors mmap makes individual key access O(1) in IO cost.
        """
        from safetensors import safe_open

        for shard_name, keys in self._shard_to_keys.items():
            shard_path = self._base_dir / shard_name
            logger.debug("Opening shard: %s (%d tensors)", shard_name, len(keys))

            with safe_open(str(shard_path), framework="pt", device=self.device) as f:
                for key in keys:
                    tensor = f.get_tensor(key)
                    yield TensorRecord(key=key, tensor=tensor)
                    del tensor  # drop before fetching next
