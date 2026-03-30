# model-pipe

Streaming operations on HuggingFace `.safetensors` models.

Tensors are processed **one at a time**.  The full model is never loaded into RAM or VRAM simultaneously — each tensor is loaded, transformed, written, and freed before the next one is touched.

---

## Features

- **Truly streaming** — uses safetensors memory-mapping; only the current tensor is in RAM
- **Modular pipes** — composable `Pipe` interface with `|` operator
- **LoRA merge** — fuse PEFT adapters into a base model on-the-fly
- **Dtype casting** — cast weights to fp16/bf16 during any operation
- **CLI** — `model-pipe info / passthrough / merge-lora`
- **Extensible** — add new pipes by subclassing `Pipe`

---

## Install

```bash
pip install -e .          # from source
# or
pip install model-pipe
```

Requires Python ≥ 3.11, PyTorch ≥ 2.0.

---

## CLI usage

### Inspect a model

```bash
model-pipe info ./llama-7b/model.safetensors
model-pipe info ./model.safetensors --filter q_proj   # filter by key substring
```

### Copy without loading the full model

```bash
model-pipe passthrough \
    -i ./model.safetensors \
    -o ./copy.safetensors

# Copy and cast to bfloat16:
model-pipe passthrough \
    -i ./model-fp32.safetensors \
    -o ./model-bf16.safetensors \
    --dtype bfloat16
```

### Merge a LoRA adapter

```bash
model-pipe merge-lora \
    -b ./llama-7b/model.safetensors \
    -a ./my-lora/adapter_model.safetensors \
    -o ./merged.safetensors

# Merge and save as bfloat16:
model-pipe merge-lora \
    -b ./llama-7b/model.safetensors \
    -a ./my-lora/adapter_model.safetensors \
    -o ./merged-bf16.safetensors \
    --dtype bfloat16

# Use GPU for the merge computation:
model-pipe merge-lora \
    -b ./model.safetensors \
    -a ./adapter_model.safetensors \
    -o ./merged.safetensors \
    --device cuda

# Explicit adapter config path and extra scale:
model-pipe merge-lora \
    -b ./model.safetensors \
    -a ./adapter_model.safetensors \
    -o ./merged.safetensors \
    --adapter-config ./my-lora/adapter_config.json \
    --scale 0.8
```

---

## Python API

Pipes are composable with `|`:

```python
from model_pipe import Pipeline, SafetensorsReader, StreamingWriter
from model_pipe.pipes.lora_merge import LoRAMergePipe
from model_pipe.pipes.dtype_cast import DTypeCastPipe
import torch

# Merge LoRA then cast to bfloat16 — only one tensor in RAM at a time
pipe = LoRAMergePipe("adapter_model.safetensors") | DTypeCastPipe(torch.bfloat16)

Pipeline(
    reader=SafetensorsReader("model.safetensors"),
    pipe=pipe,
    writer=StreamingWriter("merged.safetensors"),
).run()
```

---

## Architecture

```
model-pipe
├── model_pipe/
│   ├── cli.py            Click CLI entry point
│   ├── pipeline.py       Two-pass orchestrator
│   ├── pipes/
│   │   ├── base.py       TensorRecord, TensorMeta, Pipe, CompoundPipe
│   │   ├── passthrough.py
│   │   ├── lora_merge.py
│   │   └── dtype_cast.py
│   ├── io/
│   │   ├── reader.py     Lazy mmap-backed safetensors reader
│   │   └── writer.py     Two-phase streaming safetensors writer
│   └── utils/
│       └── lora.py       Key mapping, merge math
└── tests/
    └── test_pipeline.py
```

### Two-pass pipeline

The safetensors format requires the complete header (tensor names, shapes, dtypes, byte offsets) at the **beginning** of the file, before any data.  This is handled in two passes:

```
Phase 1 — metadata scan
  reader.iter_meta()           (reads JSON header only, no tensor data)
      ↓
  pipe.process_meta()          (may change keys/shapes/dtypes)
      ↓
  writer.prepare(metas)        (writes safetensors header to disk)

Phase 2 — data stream
  reader.iter_records()        (mmap: one tensor paged in at a time)
      ↓
  pipe.process()               (transform: merge, cast, …)
      ↓
  writer.write_record()        (append raw bytes)
      ↓  del record            (free tensor, GC reclaims memory)
  (repeat for each tensor)
      ↓
  writer.finalize()
```

### Pipe interface

```python
class Pipe(ABC):
    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]: ...
    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]: ...
    def setup(self) -> None: ...     # called once before process()
    def teardown(self) -> None: ...  # called once after process()
    def __or__(self, other) -> CompoundPipe: ...
```

- **`process()`** — lazy generator: consume one record, yield one-or-more, free immediately.
- **`process_meta()`** — identity by default; override only if the pipe changes keys/shapes/dtypes.
- **`setup()` / `teardown()`** — one-time init/cleanup (e.g. loading adapter weights).

### Writing a new pipe

```python
from model_pipe.pipes.base import Pipe, TensorRecord, TensorMeta
from typing import Iterator

class QuantisePipe(Pipe):
    """Example: quantise every weight to int8."""

    def process_meta(self, metas: Iterator[TensorMeta]) -> Iterator[TensorMeta]:
        import torch
        for meta in metas:
            yield TensorMeta(key=meta.key, dtype=torch.int8, shape=meta.shape)

    def process(self, records: Iterator[TensorRecord]) -> Iterator[TensorRecord]:
        for record in records:
            q = record.tensor.to(torch.int8)
            yield TensorRecord(key=record.key, tensor=q)
            del q
```

Then expose it in `cli.py` as a new command or option.

---

## LoRA key mapping

`LoRAMergePipe` automatically detects PEFT naming conventions:

| Pattern | Example |
|---|---|
| `base_model.model.<key>.lora_{A,B}.weight` | Standard PEFT |
| `base_model.model.<key>.lora_{A,B}.<name>.weight` | Named adapter |
| `<key>.lora_{A,B}.weight` | No prefix |
| `base_model.model.<key>.lora_embedding_{A,B}` | Embedding layers |

If `adapter_config.json` is in the same directory as the adapter file it is read automatically for `r` and `lora_alpha`.

---

## Roadmap / extending for FSDP shards

The next planned pipe is **`FSDPShardMergePipe`** for adapters produced by FSDP training runs, where each rank saves its own `.safetensors` shard.

The design will be:

```
ShardedLoRAReader(shard_paths: list[Path])
    .iter_meta()     — union of all shard headers
    .iter_records()  — concatenate shard tensors in the right order

FSDPShardMergePipe(shard_paths, ...)
    setup()          — load all shard lora_A / lora_B, all-gather into
                       full parameter tensors (still small)
    process()        — identical to LoRAMergePipe.process()
```

The `Pipe.process()` / `Pipe.process_meta()` interface is **unchanged**.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest -v
```
