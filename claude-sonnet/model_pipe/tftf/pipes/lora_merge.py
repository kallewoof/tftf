"""
LoRAMergePipe — fuse a single PEFT LoRA adapter into a base model on the fly.

Memory profile
--------------
- LoRA adapter weights  → loaded entirely in RAM during setup().
  A typical LoRA (rank 16, 7B model) is ~30-80 MiB — always affordable.
- Base model weights    → one tensor at a time, freed immediately after merge.
- Merged tensor         → one at a time, freed immediately after write.

The full base model is never in memory.

Key mapping
-----------
PEFT LoRA adapters use a `base_model.model.` prefix and `.lora_A.weight` /
`.lora_B.weight` suffixes.  See utils/lora.py for the full mapping logic,
which handles adapter names, embedding layers, and several PEFT versions.

Future: FSDP sharded LoRA
--------------------------
For FSDP-sharded adapters (one .safetensors per rank), use FSDPShardMergePipe.
It inherits the same LoRAMergeBase and only differs in how setup() loads the
weights (by concatenating per-rank shards).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from safetensors import safe_open

from tftf.pipes._lora_base import LoRAMergeBase
from tftf.utils.lora import LoRAConfig

logger = logging.getLogger(__name__)


class LoRAMergePipe(LoRAMergeBase):
    """
    Fuse a LoRA adapter (.safetensors) into the base model weight stream.

    Args:
        adapter_path:  Path to ``adapter_model.safetensors``.
        config_path:   Path to ``adapter_config.json``.  Auto-detected from
                       the same directory as *adapter_path* if omitted.
        adapter_name:  PEFT adapter name (default: ``"default"``).
        scale:         Extra user scaling on top of ``alpha / r``.
                       ``1.0`` = no extra scaling (standard LoRA behaviour).
        device:        Torch device for the merge computation.
    """

    def __init__(
        self,
        adapter_path: Path | str,
        config_path: Optional[Path | str] = None,
        adapter_name: str = "default",
        scale: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(adapter_name=adapter_name, scale=scale, device=device)
        self.adapter_path = (
            Path(adapter_path)
            if str(adapter_path).endswith("adapter_model.safetensors")
            else adapter_path / Path("adapter_model.safetensors")
        )

        if config_path is not None:
            self._config_path: Optional[Path] = Path(config_path)
        else:
            candidate = self.adapter_path.parent / "adapter_config.json"
            self._config_path = candidate if candidate.exists() else None

    # ------------------------------------------------------------------
    # setup() — load the single adapter file
    # ------------------------------------------------------------------

    def setup(self) -> None:
        # 1. Load adapter config (for alpha / rank)
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
                "adapter_config.json not found; using defaults "
                "(r=%d, alpha=%.1f).  Pass config_path= to override.",
                self._config.r,
                self._config.lora_alpha,
            )

        # 2. Load all LoRA weights into RAM (they are small)
        logger.info("Loading LoRA weights from %s …", self.adapter_path)
        with safe_open(str(self.adapter_path), framework="pt", device=self.device) as f:
            for key in f.keys():
                self._lora_weights[key] = f.get_tensor(key)

        total_mib = sum(t.nbytes for t in self._lora_weights.values()) / 1024**2
        logger.info(
            "Loaded %d LoRA tensors  (%.1f MiB)",
            len(self._lora_weights),
            total_mib,
        )
        self._adapter_key_set = set(self._lora_weights)

    # ------------------------------------------------------------------
    # Repr override to show file path
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._lora_weights)
        loaded = f"{n} tensors" if n else "not loaded"
        return (
            f"LoRAMergePipe("
            f"adapter={self.adapter_path.name!r}, "
            f"scale={self.scale}, "
            f"device={self.device!r}, "
            f"{loaded})"
        )
