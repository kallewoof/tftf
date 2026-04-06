"""
FSDPShardMergePipe — reconstruct a sharded FSDP LoRA adapter, then merge.

Background
----------
PyTorch FSDP (FullyShardedDataParallel) in LOCAL_STATE_DICT mode has each
rank write its own .safetensors checkpoint containing a slice of every
parameter tensor, sharded along dim 0 by default.

Example: world_size=4, LoRA rank=16
    rank 0: lora_A.weight  shape (4, in_features)   rows  0..3
    rank 1: lora_A.weight  shape (4, in_features)   rows  4..7
    rank 2: lora_A.weight  shape (4, in_features)   rows  8..11
    rank 3: lora_A.weight  shape (4, in_features)   rows 12..15

Reconstruction:  torch.cat([r0, r1, r2, r3], dim=0) → (16, in_features)

After reconstruction the full lora_A and lora_B are available and the merge
proceeds via the shared LoRAMergeBase.process(), identical to LoRAMergePipe.

Memory profile
--------------
LoRA tensors (all shards combined) typically total 30–100 MiB, so loading
them all simultaneously in setup() is acceptable.  Base model weights are
processed one tensor at a time.

Unsupported: flat-param sharding
---------------------------------
FSDP can also store all parameters concatenated into a single 1-D
``_flat_param_N`` tensor per shard.  Reconstructing these requires shape
metadata not present in plain safetensors files.  FSDPShardMergePipe will
raise a clear NotImplementedError if flat-param keys are detected.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from safetensors import safe_open

from tftf.pipes._lora_base import LoRAMergeBase
from tftf.utils.fsdp import (
    check_for_flat_params,
    find_shard_files,
    reconstruct_from_shards,
)
from tftf.utils.lora import LoRAConfig

logger = logging.getLogger(__name__)


class FSDPShardMergePipe(LoRAMergeBase):
    """
    Reconstruct a sharded FSDP LoRA adapter and fuse it into a base model.

    Provide exactly one of *shard_paths* or *shard_dir*.

    Args:
        shard_paths:  Ordered list of per-rank shard files (rank 0 first).
        shard_dir:    Directory of shard files, sorted alphabetically.
                      Ensure filenames sort in rank order, e.g.
                      ``rank_00.safetensors``, ``rank_01.safetensors``.
        config_path:  adapter_config.json.  Auto-detected from the first
                      shard's directory if omitted.
        adapter_name: PEFT adapter name (default: ``"default"``).
        scale:        Extra scale on top of alpha/r (1.0 = standard LoRA).
        shard_dim:    Dimension along which tensors are sharded (default: 0).
        device:       Torch device for reconstruction and merge.
    """

    def __init__(
        self,
        shard_paths: Optional[list[Path | str]] = None,
        shard_dir: Optional[Path | str] = None,
        config_path: Optional[Path | str] = None,
        adapter_name: str = "default",
        scale: float = 1.0,
        shard_dim: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(adapter_name=adapter_name, scale=scale, device=device)

        if shard_paths is None and shard_dir is None:
            raise ValueError("Provide either shard_paths or shard_dir.")
        if shard_paths is not None and shard_dir is not None:
            raise ValueError("Provide only one of shard_paths or shard_dir.")

        if shard_dir is not None:
            self._shard_files = find_shard_files(Path(shard_dir))
        else:
            self._shard_files = find_shard_files([Path(p) for p in shard_paths])  # type: ignore[arg-type]

        self.shard_dim = shard_dim

        if config_path is not None:
            self._config_path: Optional[Path] = Path(config_path)
        else:
            candidate = self._shard_files[0].parent / "adapter_config.json"
            self._config_path = candidate if candidate.exists() else None

    # ------------------------------------------------------------------
    # setup() — load + reconstruct all shard tensors
    # ------------------------------------------------------------------

    def setup(self) -> None:
        logger.info(
            "FSDPShardMergePipe: loading %d shard(s):", len(self._shard_files)
        )
        for p in self._shard_files:
            logger.info("  %s", p)

        # Load adapter config
        if self._config_path and self._config_path.exists():
            self._config = LoRAConfig.from_file(self._config_path)
            logger.info(
                "Adapter config: r=%d  lora_alpha=%.1f  effective_scale=%.6f",
                self._config.r,
                self._config.lora_alpha,
                self._config.default_scale * self.scale,
            )
        else:
            self._config = LoRAConfig.default()
            logger.warning(
                "adapter_config.json not found; using defaults (r=%d, alpha=%.1f).",
                self._config.r,
                self._config.lora_alpha,
            )

        # Accumulate per-key shard tensors: key → [rank0, rank1, …]
        shard_buckets: dict[str, list] = {}
        for rank, shard_path in enumerate(self._shard_files):
            logger.debug("Reading shard %d: %s", rank, shard_path)
            with safe_open(str(shard_path), framework="pt", device=self.device) as f:
                keys_in_shard = list(f.keys())
                check_for_flat_params(keys_in_shard)
                for key in keys_in_shard:
                    shard_buckets.setdefault(key, []).append(f.get_tensor(key))

        # Reconstruct full tensors
        world_size = len(self._shard_files)
        total_bytes = 0
        for key, shards in shard_buckets.items():
            if len(shards) != world_size:
                logger.warning(
                    "Key %r in only %d/%d shards — skipping.",
                    key, len(shards), world_size,
                )
                continue
            full = shards[0] if len(shards) == 1 else reconstruct_from_shards(
                shards, shard_dim=self.shard_dim
            )
            self._lora_weights[key] = full
            total_bytes += full.nbytes
            del shards

        self._adapter_key_set = set(self._lora_weights)
        logger.info(
            "Reconstructed %d LoRA tensors from %d shards  (%.1f MiB total)",
            len(self._lora_weights),
            world_size,
            total_bytes / 1024**2,
        )

    # ------------------------------------------------------------------
    # Repr override to show shard count
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._lora_weights)
        loaded = f"{n} tensors" if n else "not loaded"
        return (
            f"FSDPShardMergePipe("
            f"shards={len(self._shard_files)}, "
            f"shard_dim={self.shard_dim}, "
            f"scale={self.scale}, "
            f"device={self.device!r}, "
            f"{loaded})"
        )
