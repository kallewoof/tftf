"""tftf — streaming operations on HuggingFace .safetensors models."""

from tftf.pipeline import Pipeline, ReaderProtocol

# Readers
from tftf.io.reader import SafetensorsReader
from tftf.io.sharded_reader import ShardedSafetensorsReader

# Writers
from tftf.io.writer import StreamingWriter
from tftf.io.sharded_writer import ShardedWriter
from tftf.io.null_writer import NullWriter, ValidationReport

# Pipe base
from tftf.pipes.base import Pipe, CompoundPipe, TensorRecord, TensorMeta

# Concrete pipes
from tftf.pipes.passthrough import PassthroughPipe
from tftf.pipes.dtype_cast import DTypeCastPipe
from tftf.pipes.key_filter import KeyFilterPipe
from tftf.pipes.key_rename import KeyRenamePipe
from tftf.pipes._lora_base import LoRAMergeBase
from tftf.pipes.lora_merge import LoRAMergePipe
from tftf.pipes.fsdp_lora_merge import FSDPShardMergePipe
from tftf.pipes.fp8_dequant import FP8DequantPipe

__all__ = [
    # Pipeline
    "Pipeline",
    "ReaderProtocol",
    # Readers
    "SafetensorsReader",
    "ShardedSafetensorsReader",
    # Writers
    "StreamingWriter",
    "ShardedWriter",
    "NullWriter",
    "ValidationReport",
    # Pipe infrastructure
    "Pipe",
    "CompoundPipe",
    "TensorRecord",
    "TensorMeta",
    # Pipes
    "PassthroughPipe",
    "DTypeCastPipe",
    "KeyFilterPipe",
    "KeyRenamePipe",
    "LoRAMergeBase",
    "LoRAMergePipe",
    "FSDPShardMergePipe",
    "FP8DequantPipe",
]
