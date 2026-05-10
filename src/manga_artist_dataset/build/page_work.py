"""Concurrent page processing for selected dataset pages."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from manga_artist_dataset.build.image_probe import accepted_page_outputs
from manga_artist_dataset.build.page_reader import read_page_bytes
from manga_artist_dataset.concurrency import bounded_worker_count, require_positive_worker_count
from manga_artist_dataset.io.http import HostDownloadLimiter
from manga_artist_dataset.models import BuildConfig, PageOutput, PageRef, TargetSpec


@dataclass(frozen=True)
class PageProcessingResult:
    """Accepted outputs and rejection reasons for one selected page.

    Example:
        `result.page.page_index`.
    """

    page: PageRef
    outputs: list[PageOutput]
    rejected: list[str]
    did_split: bool


class SelectedPageProcessor(Protocol):
    """Boundary for page byte loading, filtering, and splitting.

    Example:
        `processor.process(target, page, config)`.
    """

    def process(self, target: TargetSpec, page: PageRef, config: BuildConfig) -> PageProcessingResult:
        """Return deterministic processing results for one selected page.

        Example:
            `result = processor.process(target, page, config)`.
        """
        ...


@dataclass(frozen=True)
class DefaultSelectedPageProcessor:
    """Production processor backed by page readers and image filters.

    Example:
        `DefaultSelectedPageProcessor(limiter).process(target, page, config)`.
    """

    download_limiter: HostDownloadLimiter

    def process(self, target: TargetSpec, page: PageRef, config: BuildConfig) -> PageProcessingResult:
        """Read, filter, and optionally split one selected page.

        Example:
            `processor.process(target, page, config).outputs`.
        """
        if config.dry_run:
            return PageProcessingResult(page, [PageOutput(b"", None, None)], [], False)
        outputs, rejected, did_split = accepted_page_outputs(
            read_page_bytes(page, self.download_limiter),
            config,
            allow_split=target.label_id in config.split_double_spread_label_ids,
        )
        return PageProcessingResult(page, outputs, rejected, did_split)


def process_selected_pages(
    target: TargetSpec,
    pages: list[PageRef],
    config: BuildConfig,
    processor: SelectedPageProcessor,
) -> Iterator[PageProcessingResult]:
    """Yield page results in input order while processing through a bounded pool.

    Example:
        `list(process_selected_pages(target, pages, config, processor))`.
    """
    require_positive_worker_count(config.download_workers, "download_workers")
    if config.download_workers == 1 or len(pages) <= 1:
        yield from _process_pages_serially(target, pages, config, processor)
        return
    yield from _process_pages_with_threads(target, pages, config, processor)


def _process_pages_serially(
    target: TargetSpec,
    pages: list[PageRef],
    config: BuildConfig,
    processor: SelectedPageProcessor,
) -> Iterator[PageProcessingResult]:
    for page in pages:
        yield processor.process(target, page, config)


def _process_pages_with_threads(
    target: TargetSpec,
    pages: list[PageRef],
    config: BuildConfig,
    processor: SelectedPageProcessor,
) -> Iterator[PageProcessingResult]:
    worker_count = bounded_worker_count(config.download_workers, len(pages), "download_workers")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = _initial_page_futures(executor, target, pages, config, processor, worker_count)
        next_submit = len(pending)
        for index in range(len(pages)):
            yield pending.pop(index).result()
            if next_submit < len(pages):
                pending[next_submit] = _submit_page_future(executor, target, pages[next_submit], config, processor)
                next_submit += 1


def _initial_page_futures(
    executor: ThreadPoolExecutor,
    target: TargetSpec,
    pages: list[PageRef],
    config: BuildConfig,
    processor: SelectedPageProcessor,
    worker_count: int,
) -> dict[int, Future[PageProcessingResult]]:
    return {
        index: _submit_page_future(executor, target, pages[index], config, processor) for index in range(worker_count)
    }


def _submit_page_future(
    executor: ThreadPoolExecutor,
    target: TargetSpec,
    page: PageRef,
    config: BuildConfig,
    processor: SelectedPageProcessor,
) -> Future[PageProcessingResult]:
    return executor.submit(processor.process, target, page, config)
