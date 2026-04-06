"""tftf — streaming operations on HuggingFace .safetensors models."""

from tftf.io.null_writer import NullWriter, ValidationReport

# Readers
from tftf.io.reader import SafetensorsReader
from tftf.io.sharded_reader import ShardedSafetensorsReader
from tftf.io.sharded_writer import ShardedWriter

# Writers
from tftf.io.writer import StreamingWriter
from tftf.pipeline import Pipeline, ReaderProtocol
from tftf.pipes._lora_base import LoRAMergeBase

# Pipe base
from tftf.pipes.base import CompoundPipe, Pipe, TensorMeta, TensorRecord
from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe
from tftf.pipes.dtype_cast import DTypeCastPipe
from tftf.pipes.fp8_dequant import FP8DequantPipe
from tftf.pipes.key_filter import KeyFilterPipe
from tftf.pipes.key_rename import KeyRenamePipe
from tftf.pipes.lora_merge import LoRAMergePipe

# Concrete pipes
from tftf.pipes.passthrough import PassthroughPipe


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
    "DCPLoRAMergePipe",
    "FP8DequantPipe",
]
