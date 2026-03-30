"""
FP8DequantPipe — streaming dequantisation of fine-grained FP8 weights.

Background
----------
Models such as DeepSeek-V3 / DeepSeek-R1 are distributed with weights stored
in ``torch.float8_e4m3fn`` alongside per-block dequantisation scales:

    model.layers.0.self_attn.q_proj.weight            float8_e4m3fn  (out, in)
    model.layers.0.self_attn.q_proj.weight_scale_inv  float32        (⌈out/128⌉, ⌈in/128⌉)

The scale tensor encodes one float32 multiplier per 128×128 block of the
weight matrix.  Dequantisation reconstructs the full-precision weight:

    W[r:r+128, c:c+128] = fp8_block.float() * scale_inv[r//128, c//128]

After dequantisation the ``weight_scale_inv`` tensors are no longer needed
and are dropped from the output stream, so the output is a standard
BF16/FP16 model that can be used with any framework or further pipe.

Streaming design
----------------
The pipe processes one tensor at a time.  The tricky part is that an FP8
weight and its companion scale tensor may arrive in *either order* within
the stream.  The pipe uses a small per-key pending dict to buffer whichever
half of the pair arrives first, then dequantises and yields as soon as both
are available.

Memory cost of buffering
------------------------
Scale tensors are tiny: for a 7168×7168 weight, the scale is
(56, 56) × 4 bytes = 12.5 KiB.  Even buffering all scale tensors for a
671B-parameter model would only be ~tens of MiB — acceptable.

FP8 weight tensors that are waiting for their scale are also buffered, but
only briefly: in practice the scale immediately precedes or follows the
weight in the shard file, so the buffer holds at most one weight at a time.

Phase 1 (process_meta)
----------------------
``process_meta`` scans the full list of incoming metas to build the set of
FP8 weight keys (those paired with a ``_scale_inv`` companion).  It then:
- Yields metas for FP8 weight keys with the target dtype and original shape.
- Silently drops metas for ``weight_scale_inv`` / ``weight_scale`` keys.
- Passes all other metas through unchanged.

Non-FP8 tensors (norms, embeddings, etc.) are passed through untouched even
if they happen to be stored in a non-FP8 dtype.

Usage
-----
    from model_pipe.pipes.fp8_dequant import FP8DequantPipe
    import torch

    pipe = FP8DequantPipe(target_dtype=torch.bfloat16)
    # Or as part of a chain:
    pipe = FP8DequantPipe(torch.bfloat16) | LoRAMergePipe("adapter_model.safetensors")
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional

import torch

from model_pipe.pipes.base import Pipe, TensorMeta, TensorRecord
from model_pipe.utils.fp8 import (
    _BLOCK,
    dequantize_fp8_weight,
    is_fp8_dtype,
    is_scale_key,
    scale_inv_key_for,
    scale_key_for,
    weight_key_for_scale,
)

logger = logging.getLogger(__name__)


class FP8DequantPipe(Pipe):
    """
    Dequantise fine-grained FP8 weights to *target_dtype* on the fly.

    Pairs each FP8 weight tensor with its ``weight_scale_inv`` (or
    ``weight_scale``) companion, applies block-wise dequantisation, and
    yields the result at *target_dtype*.  Scale tensors are consumed and
    do not appear in the output stream.

    Non-FP8 tensors are passed through unchanged.

    Args:
        target_dtype:  Output dtype for dequantised weights.
                       ``torch.bfloat16`` (default) or ``torch.float16``.
        block_size:    FP8 block dimension.  Must match the model's
                       ``weight_block_size`` config (default: 128).
        device:        Torch device for the dequantisation computation.
                       ``"cpu"`` keeps everything off the GPU.

    Typical pipeline::

        pipe = FP8DequantPipe(torch.bfloat16) | LoRAMergePipe("adapter_model.safetensors")

        Pipeline(
            reader=ShardedSafetensorsReader.from_path("./deepseek-v3/"),
            pipe=pipe,
            writer=ShardedWriter("./deepseek-v3-bf16/"),
        ).run()
    """

    def __init__(
        self,
        target_dtype: torch.dtype = torch.bfloat16,
        block_size: int = _BLOCK,
        device: str = "cpu",
    ) -> None:
        if target_dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError(
                f"target_dtype must be bfloat16, float16, or float32; "
                f"got {target_dtype}"
            )
        self.target_dtype = target_dtype
        self.block_size = block_size
        self.device = device

        # Populated during process_meta; used by process to know which
        # keys are FP8 weights vs plain tensors.
        self._fp8_weight_keys: set[str] = set()
        # key → bool: True means scale is applied as multiply (weight_scale_inv),
        #             False means divide (weight_scale).
        self._use_inv_scale: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Phase 1 — metadata scan
    # ------------------------------------------------------------------

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        """
        Scan all incoming metas to identify FP8 weight/scale pairs.

        Because identifying a weight as FP8 requires knowing whether a
        companion scale key exists *anywhere* in the stream, we must
        buffer all metas first.  This is fine: metadata lists are small
        (a few thousand entries at most).
        """
        all_metas: list[TensorMeta] = list(metas)
        all_keys: set[str] = {m.key for m in all_metas}

        # Identify FP8 weights by two signals:
        # 1. Their dtype is an FP8 type, OR
        # 2. A companion _scale_inv / _scale key exists in the stream
        #    (some checkpoints store weights in a non-FP8 dtype but still
        #    have scale tensors — treat them as FP8 in that case too).
        self._fp8_weight_keys.clear()
        self._use_inv_scale.clear()

        scale_keys: set[str] = {m.key for m in all_metas if is_scale_key(m.key)}

        for meta in all_metas:
            if is_scale_key(meta.key):
                # This is a scale tensor — register its base weight key
                base = weight_key_for_scale(meta.key)
                self._fp8_weight_keys.add(base)
                self._use_inv_scale[base] = meta.key.endswith("_scale_inv")
                continue  # will be dropped from output

            # Check if this key has a companion scale in the stream
            if (scale_inv_key_for(meta.key) in all_keys or
                    scale_key_for(meta.key) in all_keys):
                self._fp8_weight_keys.add(meta.key)
                self._use_inv_scale.setdefault(
                    meta.key,
                    scale_inv_key_for(meta.key) in all_keys
                )

            # Also detect by dtype alone (scale may be in a different shard)
            if is_fp8_dtype(meta.dtype):
                self._fp8_weight_keys.add(meta.key)
                self._use_inv_scale.setdefault(meta.key, True)

        # Yield output metas — FP8 weights at target_dtype, scale keys dropped
        for meta in all_metas:
            if is_scale_key(meta.key):
                logger.debug("FP8Dequant: dropping scale key from output: %s", meta.key)
                continue

            if meta.key in self._fp8_weight_keys:
                logger.debug(
                    "FP8Dequant: %s  %s → %s",
                    meta.key, meta.dtype, self.target_dtype
                )
                yield TensorMeta(
                    key=meta.key,
                    dtype=self.target_dtype,
                    shape=meta.shape,
                    extra=meta.extra,
                )
            else:
                yield meta

        n_fp8 = len(self._fp8_weight_keys)
        n_scale = len(scale_keys)
        if n_fp8:
            logger.info(
                "FP8DequantPipe: found %d FP8 weight(s) and %d scale tensor(s)",
                n_fp8, n_scale,
            )

    # ------------------------------------------------------------------
    # Phase 2 — streaming dequantisation
    # ------------------------------------------------------------------

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        """
        Dequantise FP8 weight tensors as they stream through.

        Buffers at most one tensor per FP8 weight/scale pair while waiting
        for the partner to arrive.  Scale tensors are consumed and not
        yielded.  All non-FP8 tensors are yielded unchanged.
        """
        # pending[weight_key] = {"weight": tensor | None, "scale": tensor | None}
        pending: dict[str, dict[str, Optional[torch.Tensor]]] = {}

        def _flush(weight_key: str) -> Optional[TensorRecord]:
            """Dequantise and return a TensorRecord if both halves are ready."""
            slot = pending.get(weight_key)
            if slot is None:
                return None
            w = slot.get("weight")
            s = slot.get("scale")
            if w is None or s is None:
                return None

            inv = self._use_inv_scale.get(weight_key, True)
            try:
                dequant = dequantize_fp8_weight(
                    weight=w.to(self.device),
                    scale=s.to(self.device),
                    target_dtype=self.target_dtype,
                    invert_scale=not inv,
                    block_size=self.block_size,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"FP8DequantPipe: failed to dequantise {weight_key!r}: {exc}"
                ) from exc

            del pending[weight_key]
            del w, s
            logger.debug(
                "FP8Dequant: dequantised %s → %s", weight_key, self.target_dtype
            )
            return TensorRecord(key=weight_key, tensor=dequant.cpu())

        for record in records:
            key = record.key

            # ---- Scale tensor ----
            if is_scale_key(key):
                weight_key = weight_key_for_scale(key)
                slot = pending.setdefault(weight_key, {"weight": None, "scale": None})
                slot["scale"] = record.tensor
                result = _flush(weight_key)
                if result is not None:
                    yield result
                del record
                continue

            # ---- FP8 weight tensor ----
            if key in self._fp8_weight_keys:
                slot = pending.setdefault(key, {"weight": None, "scale": None})
                slot["weight"] = record.tensor
                result = _flush(key)
                if result is not None:
                    yield result
                del record
                continue

            # ---- Plain (non-FP8) tensor — pass through ----
            yield record

        # Warn about any unpaired FP8 keys remaining in the buffer
        for weight_key, slot in pending.items():
            if slot["weight"] is not None and slot["scale"] is None:
                logger.warning(
                    "FP8DequantPipe: weight %r never received a scale tensor — "
                    "dequantisation skipped.  The output stream is MISSING this key.",
                    weight_key,
                )
            elif slot["scale"] is not None and slot["weight"] is None:
                logger.warning(
                    "FP8DequantPipe: scale for %r arrived but weight never did — "
                    "this scale tensor was silently discarded.",
                    weight_key,
                )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dtype_str = str(self.target_dtype).replace("torch.", "")
        n = len(self._fp8_weight_keys)
        loaded = f"{n} fp8 keys" if n else "unscanned"
        return (
            f"FP8DequantPipe("
            f"target_dtype={dtype_str!r}, "
            f"block_size={self.block_size}, "
            f"{loaded})"
        )
