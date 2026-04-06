"""
KeyRenamePipe — rename tensor keys via regex substitution.

Useful when checkpoint naming conventions differ between frameworks or
training scripts.  For example, ``transformers`` saves weights under
``model.layers.N.*``, while some FSDP checkpoints use ``_fsdp_wrapped_module``
prefixes that need to be stripped before merging.

Usage
-----
Each rename rule is a ``(pattern, replacement)`` pair applied via
``re.sub(pattern, replacement, key)`` in order.  All rules are applied to
every key; the output of rule N is the input to rule N+1.

Examples
--------
Strip FSDP wrapper prefix::

    KeyRenamePipe([
        (r"^_fsdp_wrapped_module[.]", ""),
        (r"[.]_fsdp_wrapped_module[.]", "."),
    ])

Convert ``transformer.h.N`` → ``model.layers.N``::

    KeyRenamePipe([
        (r"^transformer[.]h[.]", "model.layers."),
    ])

Rename a single layer::

    KeyRenamePipe([
        (r"lm_head[.]weight", "model.embed_tokens.weight"),
    ])

If no rule matches a key it is passed through unchanged.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

from tftf.pipes.base import Pipe, TensorMeta, TensorRecord


logger = logging.getLogger(__name__)

# A rename rule: (compiled_pattern, replacement_string)
_Rule = tuple[re.Pattern, str]


class KeyRenamePipe(Pipe):
    """
    Apply a sequence of regex substitutions to every tensor key.

    Args:
        rules:  List of ``(pattern, replacement)`` pairs.
                *pattern* is a ``re``-compatible regex string.
                *replacement* supports backreferences (``\\1``, etc.).

    Raises:
        ValueError: If any pattern is not a valid regex.
    """

    def __init__(self, rules: list[tuple[str, str]]) -> None:
        compiled: list[_Rule] = []
        for pat, repl in rules:
            try:
                compiled.append((re.compile(pat), repl))
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex pattern {pat!r}: {exc}"
                ) from exc
        self._rules = compiled

    def _rename(self, key: str) -> str:
        result = key
        for pattern, repl in self._rules:
            result = pattern.sub(repl, result)
        if result != key:
            logger.debug("KeyRename: %r → %r", key, result)
        return result

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        seen: set[str] = set()
        for meta in metas:
            new_key = self._rename(meta.key)
            if new_key in seen:
                raise ValueError(
                    f"KeyRenamePipe: two tensors mapped to the same key {new_key!r}.  "
                    f"Check your rename rules."
                )
            seen.add(new_key)
            yield TensorMeta(key=new_key, dtype=meta.dtype, shape=meta.shape, extra=meta.extra)

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        for record in records:
            new_key = self._rename(record.key)
            yield TensorRecord(key=new_key, tensor=record.tensor, extra=record.extra)

    def __repr__(self) -> str:
        rules = [(p.pattern, r) for p, r in self._rules]
        return f"KeyRenamePipe(rules={rules!r})"
