"""
DCPLoRAMergePipe — load a FSDP-sharded LoRA adapter from a PyTorch Distributed
Checkpoint (DCP) directory and fuse it into a base model weight stream.

Background
----------
When a LoRA adapter is trained with PyTorch FSDP and saved using
SHARDED_STATE_DICT (the default in modern axolotl), each rank writes a
``.distcp`` shard file into a directory such as ``pytorch_model_fsdp_0/``.
The DCP format stores each tensor's full shape in a ``.metadata`` file and
splits the data across chunk files.  Unlike per-rank ``.safetensors`` files,
DCP reassembles tensors automatically — no manual concatenation is needed.

Usage
-----
    from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe
    from tftf.pipeline import Pipeline

    pipe = DCPLoRAMergePipe(
        checkpoint_dir="path/to/pytorch_model_fsdp_0",
        config_path="path/to/adapter_config.json",  # optional
    )
    Pipeline(reader, pipe, writer).run()

Key naming
----------
axolotl saves DCP checkpoints with keys nested under a ``"model"`` wrapper
and using the standard PEFT ``base_model.model.`` prefix, e.g.:

    base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight

After the wrapper is stripped by :func:`~tftf.utils.dcp.load_dcp_state_dict`,
these keys are already compatible with :func:`~tftf.utils.lora.find_lora_keys`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from tftf.pipes._lora_base import LoRAMergeBase
from tftf.utils.dcp import load_dcp_state_dict
from tftf.utils.lora import LoRAConfig


logger = logging.getLogger(__name__)


class DCPLoRAMergePipe(LoRAMergeBase):
    """
    Load a FSDP-sharded LoRA adapter from a DCP checkpoint and merge it into
    the base model stream.

    Args:
        checkpoint_dir: Path to the DCP checkpoint directory (must contain
                        ``.metadata`` and ``.distcp`` files).
        config_path:    Path to ``adapter_config.json``.  Auto-detected from
                        the parent of *checkpoint_dir* if omitted.
        adapter_name:   PEFT adapter name (default: ``"default"``).
        scale:          Extra scale on top of alpha/r (1.0 = standard LoRA).
        device:         Torch device for the merge computation.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        config_path: Path | str | None = None,
        adapter_name: str = "default",
        scale: float = 1.0,
        device: str = "cpu",
    ) -> None:
        super().__init__(adapter_name=adapter_name, scale=scale, device=device)
        self.checkpoint_dir = Path(checkpoint_dir)

        if config_path is not None:
            self._config_path: Path | None = Path(config_path)
        else:
            # Common layout: adapter_config.json sits one level above the
            # pytorch_model_fsdp_0/ directory.
            candidate = self.checkpoint_dir.parent / "adapter_config.json"
            self._config_path = candidate if candidate.exists() else None

    # ------------------------------------------------------------------
    # setup() — load the DCP checkpoint
    # ------------------------------------------------------------------

    def setup(self) -> None:
        # 1. Adapter config (alpha / rank / target_modules)
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

        # 2. Load all LoRA tensors from DCP (chunks reassembled automatically)
        logger.info("Loading LoRA weights from DCP checkpoint: %s", self.checkpoint_dir)
        weights = load_dcp_state_dict(self.checkpoint_dir)

        for key, tensor in weights.items():
            self._lora_weights[key] = tensor.to(self.device)

        self._adapter_key_set = set(self._lora_weights)
        logger.info(
            "Loaded %d LoRA tensors from DCP checkpoint",
            len(self._lora_weights),
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self._lora_weights)
        loaded = f"{n} tensors" if n else "not loaded"
        return (
            f"DCPLoRAMergePipe("
            f"checkpoint_dir={self.checkpoint_dir.name!r}, "
            f"scale={self.scale}, "
            f"device={self.device!r}, "
            f"{loaded})"
        )
