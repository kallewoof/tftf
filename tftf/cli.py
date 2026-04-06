"""
tftf — streaming operations on HuggingFace .safetensors models.

Commands
--------
info              Print tensor names, shapes, dtypes, and file metadata.
passthrough       Copy a model one tensor at a time.
merge-lora        Fuse a single-file LoRA adapter into a base model.
merge-dcp-lora    Fuse a DCP-format FSDP-sharded LoRA adapter into a base model.
validate          Dry-run the pipeline and report validation results.

All commands accept single .safetensors files, sharded directories, or
model.safetensors.index.json files as model inputs.

All write commands support:
  --dry-run            Validate without writing anything to disk.
  --max-shard-size N   Maximum bytes per output shard (default 20 GiB).
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Union

import click
import torch

from tftf.io.null_writer import NullWriter, ValidationReport
from tftf.io.sharded_reader import ShardedSafetensorsReader
from tftf.io.sharded_writer import ShardedWriter
from tftf.io.writer import _DTYPE_ITEMSIZE
from tftf.pipeline import Pipeline
from tftf.pipes.base import Pipe
from tftf.pipes.dcp_lora_merge import DCPLoRAMergePipe
from tftf.pipes.dtype_cast import DTypeCastPipe
from tftf.pipes.key_filter import KeyFilterPipe
from tftf.pipes.key_rename import KeyRenamePipe
from tftf.pipes.lora_merge import LoRAMergePipe
from tftf.pipes.passthrough import PassthroughPipe


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DTYPE_CHOICES: dict[str, torch.dtype] = {
    "float32":  torch.float32,
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float64":  torch.float64,
}

_MODEL_PATH_TYPE = click.Path(exists=True, path_type=Path)

_DEFAULT_MAX_SHARD_BYTES = 20 * 1024 ** 3  # 20 GiB

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def _resolve_adapter(path: Path) -> tuple[Path, str, Path | None]:
    """
    Resolve an adapter path to a concrete location and detect its type.

    Returns ``(resolved_path, kind, config_hint)`` where *kind* is ``"dcp"``
    or ``"lora"`` and *config_hint* is an ``adapter_config.json`` path found
    during resolution (or ``None`` if the pipes should do their own
    auto-detection).

    Resolution rules
    ----------------
    1. Path contains ``.metadata`` → already a DCP checkpoint dir.
    2. Path contains ``checkpoint-<N>`` subdirectories → training output dir;
       pick the latest checkpoint, then look inside for a DCP subdir or
       ``adapter_model.safetensors``.  ``adapter_config.json`` at the training
       dir level is returned as *config_hint* so callers can pass it explicitly
       (the DCP pipe's own auto-detection would look one level too low).
    3. Otherwise → treat as a regular LoRA path and let the pipe handle it.
    """
    import re

    # Plain file (e.g. adapter_model.safetensors passed directly).
    if not path.is_dir():
        return path, "lora", None

    # Already a DCP shard directory.
    if (path / ".metadata").exists():
        return path, "dcp", None

    checkpoint_dirs = [
        d for d in path.iterdir()
        if d.is_dir() and re.fullmatch(r"checkpoint-\d+", d.name)
    ]
    if not checkpoint_dirs:
        return path, "lora", None

    latest = max(checkpoint_dirs, key=lambda d: int(d.name.split("-")[1]))
    click.echo(
        f"Training directory detected — using latest checkpoint: {latest.name}",
        err=True,
    )

    # adapter_config.json lives at the training dir level, not inside the checkpoint.
    config_hint: Path | None = path / "adapter_config.json"
    if not config_hint.exists():
        config_hint = None

    # DCP subdir wins if present.
    dcp_subdirs = [d for d in latest.iterdir() if d.is_dir() and (d / ".metadata").exists()]
    if dcp_subdirs:
        dcp_dir = dcp_subdirs[0]
        if len(dcp_subdirs) > 1:
            click.echo(f"Multiple DCP subdirectories found; using {dcp_dir.name}", err=True)
        return dcp_dir, "dcp", config_hint

    # Regular LoRA checkpoint.
    if (latest / "adapter_model.safetensors").exists():
        return latest, "lora", config_hint

    raise click.BadParameter(
        f"{path} does not appear to be a training directory with mergeable adapters.",
        param_hint="-a / --adapter",
    )


def _resolve_dcp_checkpoint(path: Path) -> Path:
    """
    If *path* looks like a training output directory, locate the latest
    checkpoint subdirectory and return the DCP directory within it.

    A training output directory is detected by the presence of
    ``adapter_config.json`` alongside ``checkpoint-<N>`` subdirectories.
    The latest checkpoint is selected by the highest step number.  Within
    that checkpoint the first subdirectory containing a ``.metadata`` file
    is used as the DCP root.

    If *path* does not look like a training directory it is returned unchanged.
    """
    import re

    # If this is already a DCP shard directory, use it as-is.
    if (path / ".metadata").exists():
        return path

    checkpoint_dirs = [
        d for d in path.iterdir()
        if d.is_dir() and re.fullmatch(r"checkpoint-\d+", d.name)
    ]
    if not checkpoint_dirs:
        return path

    latest = max(checkpoint_dirs, key=lambda d: int(d.name.split("-")[1]))
    click.echo(
        f"Training directory detected — using latest checkpoint: {latest.name}",
        err=True,
    )

    # Find the DCP subdirectory (contains a .metadata file)
    dcp_subdirs = [d for d in latest.iterdir() if d.is_dir() and (d / ".metadata").exists()]
    if not dcp_subdirs:
        raise click.BadParameter(
            f"No DCP directory (with .metadata) found inside {latest}.",
            param_hint="-c / --checkpoint-dir",
        )

    dcp_dir = dcp_subdirs[0]
    if len(dcp_subdirs) > 1:
        click.echo(
            f"Multiple DCP subdirectories found; using {dcp_dir.name}",
            err=True,
        )
    return dcp_dir


def _open_reader(path: Path, device: str = "cpu"):
    """Return SafetensorsReader or ShardedSafetensorsReader depending on path type."""
    return ShardedSafetensorsReader.from_path(path, device=device)


def _make_writer(
    output_path: Path,
    *,
    dry_run: bool = False,
    max_shard_size: int = _DEFAULT_MAX_SHARD_BYTES,
) -> Union[ShardedWriter, NullWriter]:
    """
    Return the appropriate writer for the given flags.

    *output_path* is always treated as a directory and created if absent.
    Output is always written as sharded safetensors; use *max_shard_size* to
    control the per-file cap (default 20 GiB).
    """
    if dry_run:
        return NullWriter()
    output_path.mkdir(parents=True, exist_ok=True)
    return ShardedWriter(output_path, max_shard_bytes=max_shard_size)


def _model_dir(path: Path) -> Path:
    """Return the directory that contains a model's non-weight files.

    If *path* is a directory (sharded model or model dir), return it directly.
    If *path* is a file (single ``.safetensors``), return its parent — that
    directory typically holds ``config.json``, tokenizer files, etc.
    """
    return path if path.is_dir() else path.parent


def _copy_model_extras(src: Path, dst: Path) -> None:
    """
    Copy non-weight files from *src* (a model directory) into *dst* (the
    output directory).

    Skipped: ``*.safetensors``, ``*.safetensors.index.json``, ``*.gguf``.
    These are either the weights being replaced or quantised derivatives that
    are no longer valid after merging.  Everything else — configs, tokenizer
    files, generation configs, etc. — is copied verbatim.

    Does nothing if *src* is not a directory (e.g. a single-file model input).
    """
    if not src.is_dir():
        return
    log = logging.getLogger(__name__)
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        n = f.name
        if n.endswith(".safetensors") or n.endswith(".safetensors.index.json") or n.endswith(".gguf"):
            continue
        shutil.copy2(f, dst / n)
        log.info("Copied %s", n)


def _finish_write(
    writer: Union[ShardedWriter, NullWriter],
    *,
    verbose: bool = False,
) -> None:
    """After Pipeline.run(), print the validation report for dry-run writers."""
    if isinstance(writer, NullWriter):
        report: ValidationReport = writer.report
        click.echo("\n" + report.summary())
        if not report.ok:
            raise SystemExit(1)


def _common_write_options(f):
    """Decorator that attaches --dry-run / --max-shard-size to a command."""
    f = click.option(
        "--dry-run", is_flag=True,
        help="Validate the full pipeline without writing any output.",
    )(f)
    f = click.option(
        "--max-shard-size", default=_DEFAULT_MAX_SHARD_BYTES,
        show_default="20 GiB", type=int,
        help="Maximum bytes per output shard.",
    )(f)
    return f


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option("0.3.0", prog_name="tftf")
def cli() -> None:
    """
    Streaming operations on HuggingFace .safetensors models.

    Tensors are processed one at a time — the full model is never loaded into
    RAM or VRAM simultaneously.  Pipes are composable and extensible.

    \b
    MODEL inputs accepted by all commands:
      ./model.safetensors              single file
      ./llama-70b/                     directory containing index.json
      ./llama-70b/model.safetensors.index.json   index file directly
    """


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("model", type=_MODEL_PATH_TYPE)
@click.option(
    "--filter", "key_filter", default=None, metavar="SUBSTR",
    help="Only show tensors whose key contains SUBSTR.",
)
@click.option(
    "--dtype-summary", is_flag=True,
    help="Print a dtype/count table instead of per-tensor rows.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def info(model: Path, key_filter: str | None, dtype_summary: bool, verbose: bool) -> None:
    """
    Print tensor metadata for MODEL.

    MODEL may be a single .safetensors file, a sharded model directory,
    or a model.safetensors.index.json file.
    """
    _setup_logging(verbose)

    reader = _open_reader(model)
    file_meta = reader.metadata()
    all_metas = list(reader.iter_meta())
    total = len(all_metas)

    metas = [m for m in all_metas if key_filter in m.key] if key_filter else all_metas

    # Header line
    if isinstance(reader, ShardedSafetensorsReader):
        n_shards = len(reader.shard_paths())
        click.echo(f"Model    : {model}  [sharded, {n_shards} files]")
    else:
        click.echo(f"Model    : {model}")
    click.echo(
        f"Tensors  : {total}"
        + (f"  (showing {len(metas)} after filter)" if key_filter else "")
    )

    if file_meta:
        click.echo("\nFile metadata:")
        for k, v in file_meta.items():
            click.echo(f"  {k}: {v}")

    if dtype_summary:
        from collections import Counter
        counts: Counter = Counter(str(m.dtype).replace("torch.", "") for m in metas)
        click.echo("\nDtype summary:")
        for dtype_str, count in counts.most_common():
            click.echo(f"  {dtype_str:<12}  {count:>6} tensors")
        return

    total_bytes = 0
    click.echo(f"\n{'Key':<64}  {'Dtype':<12}  Shape")
    click.echo("-" * 100)
    for m in metas:
        numel = 1
        for d in m.shape:
            numel *= d
        nb = numel * _DTYPE_ITEMSIZE.get(m.dtype, 4)
        total_bytes += nb
        shape_str = "×".join(str(s) for s in m.shape)
        dtype_str = str(m.dtype).replace("torch.", "")
        click.echo(f"  {m.key:<62}  {dtype_str:<12}  {shape_str}")

    click.echo(f"\nTotal tensor data: {total_bytes / 1024**3:.3f} GiB  ({total_bytes:,} bytes)")


# ---------------------------------------------------------------------------
# passthrough
# ---------------------------------------------------------------------------


@cli.command()
@click.option("-i", "--input", "input_path", required=True, type=_MODEL_PATH_TYPE,
              help="Input model (file, directory, or index.json).")
@click.option("-o", "--output", "output_path", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Output directory (created if absent).")
@click.option("--dtype", type=click.Choice(list(_DTYPE_CHOICES), case_sensitive=False),
              default=None, help="Cast all tensors to this dtype.")
@click.option("--include", "include_patterns", multiple=True, metavar="GLOB",
              help="Only copy tensors matching this glob (repeatable).")
@click.option("--exclude", "exclude_patterns", multiple=True, metavar="GLOB",
              help="Skip tensors matching this glob (repeatable).")
@_common_write_options
@click.option("--no-progress", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def passthrough(
    input_path: Path,
    output_path: Path,
    dtype: str | None,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
    dry_run: bool,
    max_shard_size: int,
    no_progress: bool,
    verbose: bool,
) -> None:
    """
    Copy a model one tensor at a time, never loading the full model.

    Supports single .safetensors files and sharded models.

    \b
    Examples:
        # Copy and cast to bfloat16
        tftf passthrough -i ./llama-70b/ -o ./out/ --dtype bfloat16

        # Validate without writing
        tftf passthrough -i ./model.safetensors -o ./out/ --dry-run

        # Copy only attention weights
        tftf passthrough -i ./model.safetensors -o ./attn/ \\
            --include '*self_attn*'
    """
    _setup_logging(verbose)

    pipe: Pipe = PassthroughPipe()
    if include_patterns or exclude_patterns:
        pipe = pipe | KeyFilterPipe(
            include=list(include_patterns),
            exclude=list(exclude_patterns),
        )
    if dtype:
        pipe = pipe | DTypeCastPipe(_DTYPE_CHOICES[dtype])

    writer = _make_writer(output_path, dry_run=dry_run, max_shard_size=max_shard_size)
    Pipeline(
        reader=_open_reader(input_path),
        pipe=pipe,
        writer=writer,
    ).run(show_progress=not no_progress)
    _finish_write(writer)
    if not dry_run:
        _copy_model_extras(_model_dir(input_path), output_path)


# ---------------------------------------------------------------------------
# merge-lora
# ---------------------------------------------------------------------------


@cli.command("merge-lora")
@click.option("-b", "--base", "base_path", required=True, type=_MODEL_PATH_TYPE,
              help="Base model (file, directory, or index.json).")
@click.option("-a", "--adapter", "adapter_path", required=True,
              type=click.Path(exists=True, dir_okay=True, path_type=Path),
              help="LoRA adapter_model.safetensors, adapter dir, or training output directory.")
@click.option("-o", "--output", "output_path", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Output directory (created if absent).")
@click.option("--adapter-config", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="adapter_config.json (auto-detected from adapter dir if omitted).")
@click.option("--adapter-name", default="default", show_default=True)
@click.option("--scale", default=1.0, show_default=True, type=float,
              help="Extra scale on top of alpha/r  (1.0 = standard LoRA).")
@click.option("--device", default="cpu", show_default=True,
              help="Torch device for merge computation.")
@click.option("--dtype", type=click.Choice(list(_DTYPE_CHOICES), case_sensitive=False),
              default=None, help="Cast merged weights to this dtype.")
@click.option("--rename", "rename_rules", multiple=True, nargs=2,
              metavar="PATTERN REPLACEMENT",
              help="Rename tensor keys via regex before merging (repeatable).")
@_common_write_options
@click.option("--no-progress", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def merge_lora(
    base_path: Path,
    adapter_path: Path,
    output_path: Path,
    adapter_config: Path | None,
    adapter_name: str,
    scale: float,
    device: str,
    dtype: str | None,
    rename_rules: tuple[tuple[str, str], ...],
    dry_run: bool,
    max_shard_size: int,
    no_progress: bool,
    verbose: bool,
) -> None:
    """
    Fuse a LoRA adapter into a base model on-the-fly.

    One base weight tensor is in RAM at a time.  The full LoRA adapter is kept
    in RAM (typically 30-100 MiB).  Supports sharded base models.

    -a accepts a single adapter_model.safetensors file, a directory containing
    one, a DCP checkpoint directory, or a training output directory.  Training
    directories are auto-detected: the latest checkpoint-<N> subfolder is used
    and the adapter format (regular LoRA or DCP/FSDP) is determined automatically.

    \b
    Examples:
        tftf merge-lora \\
            -b ./llama-7b/ -a ./my-lora/adapter_model.safetensors \\
            -o ./merged/ --dtype bfloat16

        # Pass a training output directory — format auto-detected
        tftf merge-lora \\
            -b ./llama-7b/ -a ./training-output/ \\
            -o ./merged/ --dtype bfloat16

        # Validate merge without writing
        tftf merge-lora \\
            -b ./llama-7b/ -a ./adapter_model.safetensors \\
            -o ./merged/ --dry-run

        # Key renaming
        tftf merge-lora \\
            -b ./llama-7b/ -a ./adapter_model.safetensors \\
            -o ./merged/ --rename '^transformer\\.h\\.' 'model.layers.'
    """
    _setup_logging(verbose)

    adapter_path, adapter_kind, config_hint = _resolve_adapter(adapter_path)
    effective_config = adapter_config or config_hint

    pipe: Pipe = PassthroughPipe()
    if rename_rules:
        pipe = pipe | KeyRenamePipe(list(rename_rules))
    if adapter_kind == "dcp":
        click.echo("Adapter format: DCP/FSDP — using DCPLoRAMergePipe", err=True)
        pipe = pipe | DCPLoRAMergePipe(
            checkpoint_dir=adapter_path,
            config_path=effective_config,
            adapter_name=adapter_name,
            scale=scale,
            device=device,
        )
    else:
        pipe = pipe | LoRAMergePipe(
            adapter_path=adapter_path,
            config_path=effective_config,
            adapter_name=adapter_name,
            scale=scale,
            device=device,
        )
    if dtype:
        pipe = pipe | DTypeCastPipe(_DTYPE_CHOICES[dtype])

    writer = _make_writer(output_path, dry_run=dry_run, max_shard_size=max_shard_size)
    Pipeline(
        reader=_open_reader(base_path, device=device),
        pipe=pipe,
        writer=writer,
    ).run(show_progress=not no_progress, progress_desc="merge-lora")
    _finish_write(writer)
    if not dry_run:
        _copy_model_extras(_model_dir(base_path), output_path)


# ---------------------------------------------------------------------------
# merge-dcp-lora
# ---------------------------------------------------------------------------


@cli.command("merge-dcp-lora")
@click.option("-b", "--base", "base_path", required=True, type=_MODEL_PATH_TYPE,
              help="Base model (file, directory, or index.json).")
@click.option("-c", "--checkpoint-dir", "checkpoint_dir", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="PyTorch DCP checkpoint directory (e.g. pytorch_model_fsdp_0/).")
@click.option("-o", "--output", "output_path", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Output directory (created if absent).")
@click.option("--adapter-config", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="adapter_config.json (auto-detected from parent of checkpoint-dir if omitted).")
@click.option("--adapter-name", default="default", show_default=True)
@click.option("--scale", default=1.0, show_default=True, type=float,
              help="Extra scale on top of alpha/r  (1.0 = standard LoRA).")
@click.option("--device", default="cpu", show_default=True,
              help="Torch device for merge computation.")
@click.option("--dtype", type=click.Choice(list(_DTYPE_CHOICES), case_sensitive=False),
              default=None, help="Cast merged weights to this dtype.")
@click.option("--rename", "rename_rules", multiple=True, nargs=2,
              metavar="PATTERN REPLACEMENT",
              help="Rename tensor keys via regex before merging (repeatable).")
@_common_write_options
@click.option("--no-progress", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def merge_dcp_lora(
    base_path: Path,
    checkpoint_dir: Path,
    output_path: Path,
    adapter_config: Path | None,
    adapter_name: str,
    scale: float,
    device: str,
    dtype: str | None,
    rename_rules: tuple[tuple[str, str], ...],
    dry_run: bool,
    max_shard_size: int,
    no_progress: bool,
    verbose: bool,
) -> None:
    """
    Fuse a DCP-format FSDP-sharded LoRA adapter into a base model.

    Reads a PyTorch Distributed Checkpoint (DCP) directory produced by FSDP
    training (e.g. axolotl's pytorch_model_fsdp_0/), reconstructs the full
    LoRA matrices automatically, and merges them into the base model stream.

    adapter_config.json is auto-detected from the parent directory of
    --checkpoint-dir (the typical axolotl layout).

    \b
    Examples:
        tftf merge-dcp-lora \\
            -b ./llama-7b/ \\
            -c ./checkpoint-60/pytorch_model_fsdp_0 \\
            -o ./merged/ --dtype bfloat16

        # Dry-run validation first
        tftf merge-dcp-lora \\
            -b ./llama-7b/ \\
            -c ./checkpoint-60/pytorch_model_fsdp_0 \\
            -o ./merged/ --dry-run
    """
    _setup_logging(verbose)

    checkpoint_dir = _resolve_dcp_checkpoint(checkpoint_dir)

    pipe: Pipe = PassthroughPipe()
    if rename_rules:
        pipe = pipe | KeyRenamePipe(list(rename_rules))
    pipe = pipe | DCPLoRAMergePipe(
        checkpoint_dir=checkpoint_dir,
        config_path=adapter_config,
        adapter_name=adapter_name,
        scale=scale,
        device=device,
    )
    if dtype:
        pipe = pipe | DTypeCastPipe(_DTYPE_CHOICES[dtype])

    writer = _make_writer(output_path, dry_run=dry_run, max_shard_size=max_shard_size)
    Pipeline(
        reader=_open_reader(base_path, device=device),
        pipe=pipe,
        writer=writer,
    ).run(show_progress=not no_progress, progress_desc="merge-dcp-lora")
    _finish_write(writer)
    if not dry_run:
        _copy_model_extras(_model_dir(base_path), output_path)


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("model", type=_MODEL_PATH_TYPE)
@click.option(
    "--pipe", "pipe_spec", default=None, metavar="SPEC",
    help=(
        "Optional pipe to apply before validating.  "
        "Supported: 'dtype:DTYPE', 'filter:GLOB', 'exclude:GLOB'."
    ),
)
@click.option("-v", "--verbose", is_flag=True)
def validate(model: Path, pipe_spec: str | None, verbose: bool) -> None:
    """
    Validate a model file or pipeline without writing anything to disk.

    Runs the full two-pass pipeline (metadata scan + tensor stream) and
    checks: tensor counts, sizes, dtype support, and that Phase 2 data
    matches Phase 1 declarations.  Exits with status 1 if any check fails.

    \b
    Examples:
        tftf validate ./model.safetensors
        tftf validate ./llama-70b/
        tftf validate ./model.safetensors --pipe dtype:bfloat16
        tftf validate ./model.safetensors --pipe filter:*q_proj*
    """
    _setup_logging(verbose)

    pipe: Pipe = PassthroughPipe()
    if pipe_spec:
        kind, _, arg = pipe_spec.partition(":")
        if kind == "dtype":
            if arg not in _DTYPE_CHOICES:
                raise click.BadParameter(
                    f"Unknown dtype {arg!r}.  Choices: {list(_DTYPE_CHOICES)}",
                    param_hint="--pipe",
                )
            pipe = pipe | DTypeCastPipe(_DTYPE_CHOICES[arg])
        elif kind == "filter":
            pipe = pipe | KeyFilterPipe(include=[arg])
        elif kind == "exclude":
            pipe = pipe | KeyFilterPipe(exclude=[arg])
        else:
            raise click.BadParameter(
                f"Unknown pipe spec {pipe_spec!r}.  "
                "Use 'dtype:DTYPE', 'filter:GLOB', or 'exclude:GLOB'.",
                param_hint="--pipe",
            )

    writer = NullWriter()
    Pipeline(
        reader=_open_reader(model),
        pipe=pipe,
        writer=writer,
    ).run(show_progress=True, progress_desc="validate")

    click.echo("\n" + writer.report.summary())
    if not writer.report.ok:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# dequant-fp8
# ---------------------------------------------------------------------------


@cli.command("dequant-fp8")
@click.option("-i", "--input", "input_path", required=True, type=_MODEL_PATH_TYPE,
              help="Fine-grained FP8 model (file, directory, or index.json).")
@click.option("-o", "--output", "output_path", required=True,
              type=click.Path(file_okay=False, path_type=Path),
              help="Output directory (created if absent).")
@click.option(
    "--dtype",
    type=click.Choice(["bfloat16", "float16", "float32"], case_sensitive=False),
    default="bfloat16",
    show_default=True,
    help="Target dtype for dequantised weights.",
)
@click.option(
    "--merge-lora", "lora_adapter", default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="ADAPTER",
    help="Optional: fuse this LoRA adapter_model.safetensors after dequantisation.",
)
@click.option(
    "--lora-config", default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="adapter_config.json for the LoRA adapter (auto-detected if omitted).",
)
@click.option("--lora-scale", default=1.0, show_default=True, type=float,
              help="Extra scale on top of LoRA alpha/r.")
@click.option("--lora-adapter-name", default="default", show_default=True,
              help="PEFT adapter name inside the adapter file.")
@click.option(
    "--block-size", default=128, show_default=True, type=int,
    help="FP8 block dimension (must match the model's weight_block_size).",
)
@click.option("--device", default="cpu", show_default=True,
              help="Torch device for dequantisation and merge computation.")
@_common_write_options
@click.option("--no-progress", is_flag=True)
@click.option("-v", "--verbose", is_flag=True)
def dequant_fp8(
    input_path: Path,
    output_path: Path,
    dtype: str,
    lora_adapter: Path | None,
    lora_config: Path | None,
    lora_scale: float,
    lora_adapter_name: str,
    block_size: int,
    device: str,
    dry_run: bool,
    max_shard_size: int,
    no_progress: bool,
    verbose: bool,
) -> None:
    """
    Dequantise a fine-grained FP8 model to BF16 / FP16 / FP32.

    Reads each FP8 weight and its companion weight_scale_inv tensor,
    applies block-wise (128×128) dequantisation, and streams the result
    to the output.  Scale tensors are consumed and dropped from output.
    Non-FP8 tensors (norms, embeddings, etc.) pass through unchanged.

    Optionally fuse a LoRA adapter after dequantisation via --merge-lora.

    \b
    Examples:
        # Dequantise DeepSeek-V3 to bfloat16
        tftf dequant-fp8 \\
            -i ./DeepSeek-V3/ \\
            -o ./DeepSeek-V3-bf16/

        # Dequantise then fuse a LoRA adapter, dry-run first
        tftf dequant-fp8 \\
            -i ./DeepSeek-V3/ \\
            -o ./merged/ \\
            --dtype bfloat16 \\
            --merge-lora ./my-lora/adapter_model.safetensors \\
            --dry-run

        # Dequantise to float16
        tftf dequant-fp8 \\
            -i ./model.safetensors \\
            -o ./model-fp16/ \\
            --dtype float16
    """
    from tftf.pipes.fp8_dequant import FP8DequantPipe

    _setup_logging(verbose)

    target_dtype = _DTYPE_CHOICES[dtype]

    pipe: Pipe = FP8DequantPipe(
        target_dtype=target_dtype,
        block_size=block_size,
        device=device,
    )

    if lora_adapter is not None:
        from tftf.pipes.lora_merge import LoRAMergePipe
        pipe = pipe | LoRAMergePipe(
            adapter_path=lora_adapter,
            config_path=lora_config,
            adapter_name=lora_adapter_name,
            scale=lora_scale,
            device=device,
        )

    writer = _make_writer(output_path, dry_run=dry_run, max_shard_size=max_shard_size)
    Pipeline(
        reader=_open_reader(input_path, device=device),
        pipe=pipe,
        writer=writer,
    ).run(show_progress=not no_progress, progress_desc="dequant-fp8")
    _finish_write(writer)
    if not dry_run:
        _copy_model_extras(_model_dir(input_path), output_path)
