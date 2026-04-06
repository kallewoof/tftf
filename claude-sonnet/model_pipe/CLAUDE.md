# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest -v

# Run a single test file
pytest tests/test_fp8.py -v

# Run a single test
pytest tests/test_pipeline.py::test_name -v

# CLI entry point
tftf --help
```

## Architecture

`tftf` is a **two-pass streaming pipeline** for non-destructive transformation of HuggingFace `.safetensors` models. The core constraint it solves: safetensors format requires the full header (all tensor metadata and byte offsets) at the beginning of the file, before any tensor data.

### Two-phase execution

**Phase 1 — Metadata scan** (no tensor data loaded):
- Reader maps the file header to get tensor names/shapes/dtypes
- Each pipe's `process_meta()` transforms the metadata declarations
- Writer uses final metadata to write the safetensors header with correct byte offsets

**Phase 2 — Data stream** (one tensor in RAM at a time):
- Reader mmaps tensors one-by-one
- Each pipe's `process()` transforms the tensor stream lazily
- Writer appends raw tensor bytes sequentially
- `del record.tensor` is called immediately after writing — GC reclaims before next tensor loads

### Components

**`pipeline.py`** — `Pipeline` class connects `Reader → Pipe chain → Writer` and drives both phases.

**`pipes/`** — Transformation stages. All inherit from `Pipe` (ABC in `pipes/base.py`):
- `PassthroughPipe` — identity/copy
- `DTypeCastPipe` — dtype conversion (e.g. fp32→bf16)
- `KeyFilterPipe` — glob-based include/exclude of tensor keys
- `KeyRenamePipe` — regex substitution for cross-framework key renaming
- `LoRAMergePipe` — fuse a single PEFT adapter into a base model
- `FSDPShardMergePipe` — fuse per-rank FSDP-sharded LoRA adapters
- `FP8DequantPipe` — dequantize fine-grained FP8 weights (DeepSeek-V3/R1 style)
- `CompoundPipe` — chain pipes using the `|` operator

**`io/`** — Readers and writers:
- `SafetensorsReader` — single `.safetensors` file via mmap
- `ShardedSafetensorsReader` — multi-shard models via `model.safetensors.index.json`
- `StreamingWriter` — single `.safetensors` output
- `ShardedWriter` — multi-shard output with index JSON
- `NullWriter` — dry-run validation returning a `ValidationReport`

**`utils/`** — Shared math helpers:
- `lora.py` — LoRA key mapping (PEFT naming conventions) and merge math for linear/embedding/conv layers
- `fp8.py` — FP8 dtype helpers and vectorized block-wise dequantization (128×128 blocks with broadcast-multiply)
- `fsdp.py` — FSDP shard discovery and reconstruction

### Pipe interface

```python
class Pipe(ABC):
    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]: ...
    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]: ...
    def setup(self) -> None: ...
    def teardown(self) -> None: ...
    def __or__(self, other) -> CompoundPipe: ...  # pipe_a | pipe_b | pipe_c
```

Key rule: pipes must be **lazy** — don't buffer streams, free tensors immediately after yielding. A pipe may drop records (filter) or emit additional ones (e.g. dequant reads scale tensors to augment weight tensors).

### Tests

Tests use synthetic in-memory tensors — no model downloads needed. All 91 tests run quickly. Test files map roughly to modules: `test_fp8.py`, `test_pipeline.py`, `test_writers_and_base.py`, `test_new_features.py`.
