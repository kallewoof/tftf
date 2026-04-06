"""
NullWriter — a dry-run writer that validates the pipeline without writing.

Use NullWriter to:
- Check that a pipe chain produces valid output metadata (shapes, dtypes, keys).
- Benchmark streaming throughput without disk I/O.
- Run the ``validate`` CLI command or any command's ``--dry-run`` flag.

NullWriter satisfies the same API as StreamingWriter and ShardedWriter so it
can be dropped in as a replacement without changing Pipeline code.

Validation checks
-----------------
- No duplicate output keys.
- All tensor dtypes are serialisable by safetensors.
- The tensor shape and dtype yielded in Phase 2 matches what was declared in
  Phase 1 (process_meta must agree with process).
- All tensors declared in Phase 1 are actually written in Phase 2.
- No tensors appear in Phase 2 that were not declared in Phase 1.
"""

from __future__ import annotations

import time
import types
from dataclasses import dataclass, field

from tftf.io.writer import _TORCH_TO_ST, StreamingWriter, _nbytes
from tftf.pipes.base import TensorMeta, TensorRecord


@dataclass
class ValidationReport:
    """Summary of a NullWriter dry run."""

    n_tensors: int = 0
    total_bytes: int = 0
    dtype_counts: dict[str, int] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    unsupported_dtypes: list[str] = field(default_factory=list)
    duplicate_keys: list[str] = field(default_factory=list)
    missing_keys: list[str] = field(default_factory=list)
    extra_keys: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 1024**3

    @property
    def ok(self) -> bool:
        return not (
            self.mismatches
            or self.unsupported_dtypes
            or self.duplicate_keys
            or self.missing_keys
            or self.extra_keys
        )

    def summary(self) -> str:
        lines = [
            f"Tensors : {self.n_tensors}",
            f"Size    : {self.total_gib:.3f} GiB  ({self.total_bytes:,} bytes)",
            f"Elapsed : {self.elapsed_seconds:.2f}s",
            "Dtypes  : " + (", ".join(
                f"{k}={v}" for k, v in sorted(self.dtype_counts.items())
            ) or "(none)"),
        ]
        if self.ok:
            lines.append("Result  : OK — all checks passed.")
        else:
            lines.append("Result  : FAILED")
            for prob_list, label in [
                (self.duplicate_keys,     "Duplicate keys"),
                (self.unsupported_dtypes, "Unsupported dtypes"),
                (self.mismatches,         "Shape/dtype mismatches"),
                (self.missing_keys,       "Missing tensors (declared but not written)"),
                (self.extra_keys,         "Extra tensors (written but not declared)"),
            ]:
                for item in prob_list:
                    lines.append(f"  [{label}]  {item}")
        return "\n".join(lines)


class NullWriter(StreamingWriter):
    """
    Drop-in replacement for StreamingWriter / ShardedWriter that discards data.

    Usage::

        writer = NullWriter()
        Pipeline(reader, pipe, writer).run()
        print(writer.report.summary())
        assert writer.report.ok
    """

    def __init__(self) -> None:
        self.report = ValidationReport()
        # Expose a .path attribute so Pipeline progress bar doesn't crash
        self.path = types.SimpleNamespace(name="<dry-run>")

        self._declared_index: dict[str, TensorMeta] = {}
        self._written_keys: list[str] = []
        self._t0: float = 0.0

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def prepare(
        self,
        metas: list[TensorMeta],
        file_metadata: dict[str, str] | None = None,
    ) -> None:
        self._t0 = time.monotonic()
        seen_keys: set[str] = set()

        for meta in metas:
            if meta.key in seen_keys:
                self.report.duplicate_keys.append(meta.key)
            seen_keys.add(meta.key)
            self._declared_index[meta.key] = meta

            if meta.dtype not in _TORCH_TO_ST:
                self.report.unsupported_dtypes.append(
                    f"{meta.key}: {meta.dtype} not serialisable by safetensors"
                )
            else:
                dtype_str = _TORCH_TO_ST[meta.dtype]
                self.report.dtype_counts[dtype_str] = (
                    self.report.dtype_counts.get(dtype_str, 0) + 1
                )
                self.report.total_bytes += _nbytes(meta.dtype, meta.shape)

        self.report.n_tensors = len(metas)

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def write_record(self, record: TensorRecord) -> None:
        self._written_keys.append(record.key)

        declared = self._declared_index.get(record.key)
        if declared is None:
            if record.key not in self.report.extra_keys:
                self.report.extra_keys.append(record.key)
            return

        if record.tensor.shape != declared.shape:
            self.report.mismatches.append(
                f"{record.key}: declared shape {tuple(declared.shape)} "
                f"but got {tuple(record.tensor.shape)}"
            )
        if record.tensor.dtype != declared.dtype:
            self.report.mismatches.append(
                f"{record.key}: declared dtype {declared.dtype} "
                f"but got {record.tensor.dtype}"
            )

    # ------------------------------------------------------------------
    # Finalise
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        self.report.elapsed_seconds = time.monotonic() - self._t0

        written_set = set(self._written_keys)
        declared_set = set(self._declared_index)
        self.report.missing_keys = sorted(declared_set - written_set)

        for key in self._written_keys:
            if key not in declared_set and key not in self.report.extra_keys:
                self.report.extra_keys.append(key)
