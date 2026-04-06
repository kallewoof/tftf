"""
PyTorch Distributed Checkpoint (DCP) loading utilities.

DCP is the native checkpoint format produced by PyTorch FSDP when using
SHARDED_STATE_DICT or LOCAL_STATE_DICT saving strategies (e.g. via axolotl).
A checkpoint directory contains `__N_0.distcp` shard files plus a `.metadata`
file that records each tensor's full shape and the mapping of chunks to files.
The DCP reader reassembles tensor chunks automatically, so callers receive
complete (unsharded) tensors without any manual concatenation.

Typical checkpoint layout
--------------------------
    pytorch_model_fsdp_0/
        __0_0.distcp     ← rank-0 shard
        __1_0.distcp     ← rank-1 shard
        .metadata

Usage
-----
    from tftf.utils.dcp import load_dcp_state_dict
    weights = load_dcp_state_dict(Path("pytorch_model_fsdp_0"))
    # weights: dict[str, torch.Tensor] — full reconstructed tensors
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.format_utils as dist_cp_format_utils
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner

logger = logging.getLogger(__name__)


def load_dcp_state_dict(checkpoint_dir: Path | str) -> dict[str, torch.Tensor]:
    """
    Load a PyTorch DCP checkpoint directory into a flat dict of full tensors.

    The DCP reader reconstructs each tensor from its chunks automatically,
    so the returned dict contains complete (unsharded) tensors regardless of
    how many ranks were used during training.

    If the checkpoint was saved with a single-key wrapper (e.g. ``{"model": {...}}``,
    which is the common axolotl/FSDP pattern), the wrapper is stripped and the
    inner dict is returned directly.

    Args:
        checkpoint_dir: Path to the DCP checkpoint directory (must contain
                        a ``.metadata`` file and one or more ``.distcp`` files).

    Returns:
        ``dict[str, torch.Tensor]`` mapping parameter names to full tensors.

    Raises:
        FileNotFoundError: If *checkpoint_dir* does not exist or has no
                           ``.metadata`` file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not (checkpoint_dir / ".metadata").exists():
        raise FileNotFoundError(
            f"No .metadata file found in {checkpoint_dir}. "
            "Is this a valid PyTorch DCP checkpoint directory?"
        )

    state_dict: dict = {}
    dist_cp_format_utils._load_state_dict(
        state_dict,
        storage_reader=dist_cp.FileSystemReader(checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    # Unwrap single-key wrapper (axolotl saves as {"model": {param: tensor, …}})
    if len(state_dict) == 1:
        inner = next(iter(state_dict.values()))
        if isinstance(inner, dict):
            logger.debug(
                "Unwrapped single-key DCP wrapper %r → %d tensors",
                next(iter(state_dict.keys())),
                len(inner),
            )
            state_dict = inner

    n = len(state_dict)
    total_mib = sum(t.nbytes for t in state_dict.values() if isinstance(t, torch.Tensor)) / 1024**2
    logger.info("Loaded %d tensors from DCP checkpoint (%.1f MiB)", n, total_mib)
    return state_dict
