"""
KeyFilterPipe — include or exclude tensors by glob / substring pattern.

Filtering rules
---------------
- ``include`` patterns: only tensors whose key matches at least one pattern
  are passed through.  If *include* is empty, all keys are included.
- ``exclude`` patterns: tensors whose key matches any exclude pattern are
  dropped, even if they matched an include pattern.

Patterns are matched with ``fnmatch`` (shell-style globs):

    *          matches everything
    ?          matches any single character
    [seq]      matches any character in seq
    [!seq]     matches any character not in seq

Simple substring matching (no wildcards) also works because fnmatch treats
plain strings as literals that must match the whole key — use ``*substr*``
for contains-matching.

Examples
--------
Keep only attention weights::

    KeyFilterPipe(include=["*self_attn*"])

Drop all layernorm weights::

    KeyFilterPipe(exclude=["*layernorm*", "*layer_norm*"])

Keep q/k/v projections but not output projection::

    KeyFilterPipe(include=["*self_attn*proj*"], exclude=["*o_proj*"])

Use case in practice
--------------------
When running ``merge-dcp-lora`` on a very large sharded model, you may want
to merge only a subset of layers and pass the rest through unchanged.  Pipe
a KeyFilterPipe before the merge pipe to select the target tensors, then
merge the two output streams (not yet implemented — future SplitMergePipe).
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Iterator

from tftf.pipes.base import Pipe, TensorMeta, TensorRecord


logger = logging.getLogger(__name__)


class KeyFilterPipe(Pipe):
    """
    Drop or keep tensors based on glob patterns applied to their keys.

    Args:
        include:  Only pass tensors whose key matches one of these patterns.
                  Empty list = include everything (default).
        exclude:  Drop tensors whose key matches one of these patterns,
                  even if they matched an *include* pattern.
                  Empty list = exclude nothing (default).
    """

    def __init__(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self.include = list(include or [])
        self.exclude = list(exclude or [])

    def _keep(self, key: str) -> bool:
        if self.include:
            if not any(fnmatch.fnmatch(key, pat) for pat in self.include):
                return False
        if self.exclude:
            if any(fnmatch.fnmatch(key, pat) for pat in self.exclude):
                return False
        return True

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        for meta in metas:
            if self._keep(meta.key):
                yield meta
            else:
                logger.debug("KeyFilter: dropping %s", meta.key)

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        for record in records:
            if self._keep(record.key):
                yield record
            else:
                logger.debug("KeyFilter: dropping %s", record.key)
                del record.tensor

    def __repr__(self) -> str:
        parts = []
        if self.include:
            parts.append(f"include={self.include!r}")
        if self.exclude:
            parts.append(f"exclude={self.exclude!r}")
        return f"KeyFilterPipe({', '.join(parts)})"
