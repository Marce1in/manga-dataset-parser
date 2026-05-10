"""Process-level cleanup orchestration for dataset cleaning."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.pipeline import build_default_manga_cleanup_pipeline, iter_supported_image_paths
from manga_artist_dataset.concurrency import bounded_worker_count, require_positive_worker_count


@dataclass(frozen=True)
class CleanupImageTask:
    """Input and output paths for one cleanup image.

    Example:
        `CleanupImageTask(Path("in.png"), Path("out.png"))`.
    """

    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class CleanupChunkTask:
    """A shard of image cleanup work for one detector process.

    Example:
        `CleanupChunkTask(CleanupConfig(), tasks)`.
    """

    cleanup_config: CleanupConfig
    image_tasks: list[CleanupImageTask]


type CleanupChunkRunner = Callable[[CleanupChunkTask], list[Path]]


def clean_directory_parallel(
    input_dir: Path,
    output_dir: Path,
    cleanup_config: CleanupConfig,
    worker_count: int,
    chunk_runner: CleanupChunkRunner | None = None,
) -> list[Path]:
    """Clean a directory with a bounded number of detector processes.

    Example:
        `clean_directory_parallel(Path("in"), Path("out"), CleanupConfig(), 2)`.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Cleanup input must be a directory; got {input_dir}.")
    require_positive_worker_count(worker_count, "detector_workers")
    image_tasks = cleanup_image_tasks(input_dir, output_dir, cleanup_config)
    return clean_image_tasks_parallel(cleanup_config, image_tasks, worker_count, chunk_runner)


def cleanup_image_tasks(
    input_dir: Path,
    output_dir: Path,
    cleanup_config: CleanupConfig,
) -> list[CleanupImageTask]:
    """Build stable cleanup tasks for all supported images under a directory.

    Example:
        `tasks = cleanup_image_tasks(Path("in"), Path("out"), CleanupConfig())`.
    """
    paths = iter_supported_image_paths(input_dir, cleanup_config.supported_extensions)
    return [CleanupImageTask(path, output_dir / path.relative_to(input_dir)) for path in paths]


def clean_image_tasks_parallel(
    cleanup_config: CleanupConfig,
    image_tasks: list[CleanupImageTask],
    worker_count: int,
    chunk_runner: CleanupChunkRunner | None = None,
) -> list[Path]:
    """Clean image tasks through serial or process-pool execution.

    Example:
        `clean_image_tasks_parallel(CleanupConfig(), tasks, 2)`.
    """
    require_positive_worker_count(worker_count, "detector_workers")
    if not image_tasks:
        return []
    active_runner = chunk_runner or clean_image_chunk
    if worker_count == 1 or len(image_tasks) <= 1:
        return active_runner(CleanupChunkTask(cleanup_config, image_tasks))
    return run_cleanup_chunks(cleanup_config, image_tasks, worker_count, active_runner)


def run_cleanup_chunks(
    cleanup_config: CleanupConfig,
    image_tasks: list[CleanupImageTask],
    worker_count: int,
    chunk_runner: CleanupChunkRunner,
) -> list[Path]:
    """Run cleanup shards and return paths in input order.

    Example:
        `paths = run_cleanup_chunks(CleanupConfig(), tasks, 2, clean_image_chunk)`.
    """
    chunks = cleanup_chunk_tasks(cleanup_config, image_tasks, worker_count)
    with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
        chunk_results = list(executor.map(chunk_runner, chunks))
    return [path for paths in chunk_results for path in paths]


def cleanup_chunk_tasks(
    cleanup_config: CleanupConfig,
    image_tasks: list[CleanupImageTask],
    worker_count: int,
) -> list[CleanupChunkTask]:
    """Split cleanup work into stable contiguous chunks.

    Example:
        `chunks = cleanup_chunk_tasks(CleanupConfig(), tasks, 2)`.
    """
    bounded_count = bounded_worker_count(worker_count, len(image_tasks), "detector_workers")
    chunk_size = max(1, (len(image_tasks) + bounded_count - 1) // bounded_count)
    chunks = [image_tasks[index : index + chunk_size] for index in range(0, len(image_tasks), chunk_size)]
    return [CleanupChunkTask(cleanup_config, chunk) for chunk in chunks]


def clean_image_chunk(chunk: CleanupChunkTask) -> list[Path]:
    """Clean one shard using one loaded detector pipeline.

    Example:
        `paths = clean_image_chunk(chunk)`.
    """
    pipeline = build_default_manga_cleanup_pipeline(chunk.cleanup_config)
    return [pipeline.clean_image(task.input_path, task.output_path) for task in chunk.image_tasks]
