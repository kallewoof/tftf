"""
FP8DequantPipe — streaming dequantisation of fine-grained FP8 weights.

Background
----------
Models such as DeepSeek-V3 / DeepSeek-R1 are distributed with weights in
``torch.float8_e4m3fn`` alongside per-block dequantisation scales:

    model.layers.N.self_attn.q_proj.weight            float8_e4m3fn  (out, in)
    model.layers.N.self_attn.q_proj.weight_scale_inv  float32        (⌈out/128⌉, ⌈in/128⌉)

The scale tensor encodes one float32 multiplier per 128×128 block.
Dequantisation reconstructs the full-precision weight:

    W[r:r+B, c:c+B] = fp8_block.float() * scale_inv[r//B, c//B]

After dequantisation the ``weight_scale_inv`` tensors are dropped from the
output stream, so the result is a standard BF16/FP16 model ready for
inference, further pipe stages, or LoRA merging.

Streaming design
----------------
The pipe processes one tensor at a time but must handle the fact that an
FP8 weight and its companion scale may arrive in *either order*.  A small
per-key pending dict buffers whichever half arrives first; dequantisation
fires as soon as both halves are available.

Memory cost of buffering
------------------------
Scale tensors are tiny (≈12 KiB for a 7168×7168 weight).  Even buffering
all scale tensors for a 671B model totals only tens of MiB.  FP8 weight
tensors are also buffered while waiting for their scale, but only briefly —
in practice scales are immediately adjacent in the shard file.

Two-pass requirement
--------------------
``process_meta()`` MUST be called before ``process()``.  The Pipeline
orchestrator always does this; if you call ``process()`` on a freshly
constructed pipe (e.g. in a test) you will get a RuntimeError with a
clear message.  Call ``pipe.process_meta(reader.iter_meta())`` first and
consume the iterator to populate the internal key maps.
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch

from tftf.pipes.base import Pipe, TensorMeta, TensorRecord
from tftf.utils.fp8 import (
    _BLOCK,
    dequantize_fp8_weight,
    is_fp8_dtype,
    is_scale_key,
    scale_inv_key_for,
    scale_key_for,
    weight_key_for_scale,
)


logger = logging.getLogger(__name__)

# Sentinel to distinguish "never scanned" from "scanned, found 0 FP8 keys"
_NOT_SCANNED = object()


class FP8DequantPipe(Pipe):
    """
    Dequantise fine-grained FP8 weights to *target_dtype* on the fly.

    Pairs each FP8 weight tensor with its companion ``weight_scale_inv``
    (or ``weight_scale``) tensor, applies block-wise (128×128) dequantisation
    using a vectorised broadcast-multiply, and yields the result at
    *target_dtype*.  Scale tensors are consumed and do not appear in the
    output stream.  Non-FP8 tensors pass through unchanged.

    Args:
        target_dtype:  Output dtype for dequantised weights.
                       ``torch.bfloat16`` (default), ``torch.float16``,
                       or ``torch.float32``.
        block_size:    FP8 block dimension.  Must match the model's
                       ``weight_block_size`` config (default: 128).
        device:        Torch device for the dequantisation computation.
                       ``"cpu"`` keeps everything off the GPU.

    Example — dequantise then fuse LoRA::

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

        # Set to a real set after process_meta() runs; sentinel otherwise
        self._fp8_weight_keys: set[str] | object = _NOT_SCANNED
        # weight_key → True (multiply, _scale_inv) | False (divide, _scale)
        self._use_inv_scale: dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Phase 1 — metadata scan
    # ------------------------------------------------------------------

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        """
        Scan all incoming metas to identify FP8 weight/scale pairs.

        Buffers the full meta list (metadata only — no tensor data) to
        detect companion scale keys that may appear anywhere in the stream.
        """
        all_metas: list[TensorMeta] = list(metas)
        all_keys: set[str] = {m.key for m in all_metas}

        fp8_keys: set[str] = set()
        use_inv: dict[str, bool] = {}
        scale_keys_found: set[str] = set()

        for meta in all_metas:
            if is_scale_key(meta.key):
                scale_keys_found.add(meta.key)
                base = weight_key_for_scale(meta.key)
                fp8_keys.add(base)
                use_inv[base] = meta.key.endswith("_scale_inv")
                continue

            # Detect by companion key presence
            has_inv  = scale_inv_key_for(meta.key) in all_keys
            has_fwd  = scale_key_for(meta.key)     in all_keys
            if has_inv or has_fwd:
                fp8_keys.add(meta.key)
                use_inv.setdefault(meta.key, has_inv)

            # Detect by dtype
            if is_fp8_dtype(meta.dtype):
                fp8_keys.add(meta.key)
                use_inv.setdefault(meta.key, True)

        self._fp8_weight_keys = fp8_keys
        self._use_inv_scale = use_inv

        n_fp8  = len(fp8_keys)
        n_scal = len(scale_keys_found)
        if n_fp8:
            logger.info(
                "FP8DequantPipe: %d FP8 weight(s) + %d scale tensor(s) → "
                "output dtype %s",
                n_fp8, n_scal, self.target_dtype,
            )
        else:
            logger.debug("FP8DequantPipe: no FP8 weights detected — pipe is a passthrough")

        # Yield output metas
        for meta in all_metas:
            if is_scale_key(meta.key):
                logger.debug("FP8Dequant: dropping scale key: %s", meta.key)
                continue
            if meta.key in fp8_keys:
                logger.debug(
                    "FP8Dequant: meta %s  %s → %s",
                    meta.key, meta.dtype, self.target_dtype,
                )
                yield TensorMeta(
                    key=meta.key,
                    dtype=self.target_dtype,
                    shape=meta.shape,
                    extra=meta.extra,
                )
            else:
                yield meta

    # ------------------------------------------------------------------
    # Phase 2 — streaming dequantisation
    # ------------------------------------------------------------------

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        """
        Stream tensors through, dequantising FP8 weights on the fly.

        Raises:
            RuntimeError: If ``process_meta()`` was never called (key maps
                          are not populated).
        """
        if self._fp8_weight_keys is _NOT_SCANNED:
            raise RuntimeError(
                "FP8DequantPipe.process() called before process_meta(). "
                "The Pipeline orchestrator always calls process_meta() first. "
                "If you are driving the pipe manually, call: "
                "  list(pipe.process_meta(reader.iter_meta()))  "
                "before calling process()."
            )

        fp8_keys = self._fp8_weight_keys  # now a real set[str]

        # Pending buffer: weight_key → {"weight": Tensor|None, "scale": Tensor|None}
        pending: dict[str, dict[str, torch.Tensor | None]] = {}

        def _try_flush(weight_key: str) -> TensorRecord | None:
            slot = pending.get(weight_key)
            if slot is None:
                return None
            w = slot["weight"]
            s = slot["scale"]
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
                    f"FP8DequantPipe: dequantisation failed for {weight_key!r}: {exc}"
                ) from exc

            del pending[weight_key], w, s
            logger.debug("FP8Dequant: dequantised %s → %s", weight_key, self.target_dtype)
            return TensorRecord(key=weight_key, tensor=dequant.cpu())

        for record in records:
            key = record.key

            if is_scale_key(key):
                # Companion scale tensor
                weight_key = weight_key_for_scale(key)
                pending.setdefault(weight_key, {"weight": None, "scale": None})
                pending[weight_key]["scale"] = record.tensor
                result = _try_flush(weight_key)
                if result is not None:
                    yield result
                del record
                continue

            if key in fp8_keys:
                # FP8 weight tensor
                pending.setdefault(key, {"weight": None, "scale": None})
                pending[key]["weight"] = record.tensor
                result = _try_flush(key)
                if result is not None:
                    yield result
                del record
                continue

            # Plain (non-FP8) tensor — pass through unchanged
            yield record

        # Report any dangling buffer entries (indicate broken checkpoint)
        for weight_key, slot in pending.items():
            if slot["weight"] is not None and slot["scale"] is None:
                logger.warning(
                    "FP8DequantPipe: weight %r has no companion scale — "
                    "dequantisation skipped.  This key is MISSING from the output.",
                    weight_key,
                )
            elif slot["scale"] is not None and slot["weight"] is None:
                logger.warning(
                    "FP8DequantPipe: scale arrived for %r but weight never did — "
                    "scale tensor silently discarded.",
                    weight_key,
                )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dtype_str = str(self.target_dtype).replace("torch.", "")
        if self._fp8_weight_keys is _NOT_SCANNED:
            state = "unscanned"
        else:
            n = len(self._fp8_weight_keys)  # type: ignore[arg-type]
            state = f"{n} fp8 key(s)"
        return (
            f"FP8DequantPipe("
            f"target_dtype={dtype_str!r}, "
            f"block_size={self.block_size}, "
            f"{state})"
        )
