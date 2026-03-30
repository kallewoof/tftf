"""model_pipe — streaming operations on HuggingFace .safetensors models."""

from model_pipe.pipeline import Pipeline, ReaderProtocol

# Readers
from model_pipe.io.reader import SafetensorsReader
from model_pipe.io.sharded_reader import ShardedSafetensorsReader

# Writers
from model_pipe.io.writer import StreamingWriter
from model_pipe.io.sharded_writer import ShardedWriter
from model_pipe.io.null_writer import NullWriter, ValidationReport

# Pipe base
from model_pipe.pipes.base import Pipe, CompoundPipe, TensorRecord, TensorMeta

# Concrete pipes
from model_pipe.pipes.passthrough import PassthroughPipe
from model_pipe.pipes.dtype_cast import DTypeCastPipe
from model_pipe.pipes.key_filter import KeyFilterPipe
from model_pipe.pipes.key_rename import KeyRenamePipe
from model_pipe.pipes._lora_base import LoRAMergeBase
from model_pipe.pipes.lora_merge import LoRAMergePipe
from model_pipe.pipes.fsdp_lora_merge import FSDPShardMergePipe
from model_pipe.pipes.fp8_dequant import FP8DequantPipe

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
