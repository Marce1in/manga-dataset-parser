"""Command line interface for the manga dataset pipeline."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Callable
from pathlib import Path

from manga_artist_dataset.build.pipeline import build_dataset
from manga_artist_dataset.cleanup.config import CleanupConfig
from manga_artist_dataset.cleanup.dataset_cleaner import (
    DEFAULT_INPUT_ROOT as CLEAN_INPUT_ROOT,
)
from manga_artist_dataset.cleanup.dataset_cleaner import (
    DEFAULT_OUTPUT_ROOT as CLEAN_OUTPUT_ROOT,
)
from manga_artist_dataset.cleanup.dataset_cleaner import (
    DatasetCleanupConfig,
    clean_panels,
)
from manga_artist_dataset.cleanup.pipeline import build_default_manga_cleanup_pipeline
from manga_artist_dataset.errors import DatasetError
from manga_artist_dataset.models import BuildConfig
from manga_artist_dataset.polished.manifest_builder import main as polished_manifest_main
from manga_artist_dataset.reroll.bad_pages import main as reroll_main


def main(argv: list[str] | None = None) -> int:
    """Run the package CLI.

    Example:
        `main(["build", "--sources", "manifests/manga_sources.polished.json"])`.
    """
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = dispatch(args)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if isinstance(report, str):
        print(report)
        return 0
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manga artist dataset pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_build_parser(subparsers)
    add_manifest_parser(subparsers)
    add_reroll_parser(subparsers)
    add_panel_parser(subparsers)
    add_cleanup_parser(subparsers)
    return parser


def add_build_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("build", help="Build a page dataset from an explicit manifest.")
    parser.add_argument("--sources", default=Path("manifests/manga_sources.polished.json"), type=Path)
    parser.add_argument("--output", default=Path("artifacts/datasets/polished_pages"), type=Path)
    parser.add_argument("--pages-per-artist", default=60, type=positive_int)
    parser.add_argument("--trim-start", default=5, type=positive_int)
    parser.add_argument("--trim-end", default=5, type=positive_int)
    parser.add_argument("--min-chapters", default=3, type=positive_int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--allow-downloads", action="store_true")
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-all-sources", action="store_true")
    parser.add_argument("--filter-color-pages", action="store_true")
    parser.add_argument("--filter-double-spreads", action="store_true")
    parser.add_argument("--split-double-spreads", action="store_true")
    parser.add_argument("--split-double-spreads-for-labels", type=comma_separated_ints, default=set())
    parser.add_argument("--download-workers", default=12, type=worker_count)
    parser.add_argument("--download-host-delay-seconds", default=0.1, type=nonnegative_float)
    parser.add_argument("--no-strict-targets", action="store_true")


def add_manifest_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("build-polished-manifest", help="Generate the polished source manifest.")
    parser.add_argument("--output", type=Path, default=Path("manifests/manga_sources.polished.json"))
    parser.add_argument("--plan-output", type=Path, default=Path("artifacts/reports/chapter_plan.polished.json"))
    parser.add_argument("--min-usable-pages-per-anchor", type=int, default=30)


def add_reroll_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    subparsers.add_parser("reroll", help="Apply curated reroll replacements to the polished dataset.")


def add_panel_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("clean-panels", help="Clean and standardize the polished dataset.")
    parser.add_argument("--input-root", type=Path, default=CLEAN_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=CLEAN_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable-artwork-inpainting", action="store_true")
    parser.add_argument("--scratch-workers", default=8, type=worker_count)
    parser.add_argument("--detector-workers", default=2, type=worker_count)
    parser.add_argument("--standardize-workers", default=8, type=worker_count)
    parser.add_argument("--train-fraction", default=0.8, type=fraction)


def add_cleanup_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("cleanup", help="Clean one downloaded image or a directory of images.")
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--enable-artwork-inpainting", action="store_true")


def dispatch(args: argparse.Namespace) -> object:
    if args.command == "build":
        return build_dataset(build_config_from_args(args))
    if args.command == "build-polished-manifest":
        return command_exit_report(run_quietly(lambda: polished_manifest_main(polished_args(args))))
    if args.command == "reroll":
        return command_exit_report(run_quietly(reroll_main))
    if args.command == "clean-panels":
        return clean_panels(cleanup_config_from_args(args))
    if args.command == "cleanup":
        return run_cleanup_command(args)
    raise ValueError(f"Unknown command {args.command!r}.")


def build_config_from_args(args: argparse.Namespace) -> BuildConfig:
    return BuildConfig(
        sources_path=args.sources,
        output_dir=args.output,
        pages_per_artist=args.pages_per_artist,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        min_chapters=args.min_chapters,
        seed=args.seed,
        allow_downloads=args.allow_downloads,
        allow_short=args.allow_short,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        strict_targets=not args.no_strict_targets,
        use_all_sources=args.use_all_sources,
        filter_color_pages=args.filter_color_pages,
        filter_double_spreads=args.filter_double_spreads,
        split_double_spreads=args.split_double_spreads,
        split_double_spread_label_ids=args.split_double_spreads_for_labels,
        download_workers=args.download_workers,
        download_host_delay_seconds=args.download_host_delay_seconds,
    )


def cleanup_config_from_args(args: argparse.Namespace) -> DatasetCleanupConfig:
    return DatasetCleanupConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        overwrite=args.overwrite,
        cleanup_config=CleanupConfig(enable_artwork_inpainting=args.enable_artwork_inpainting),
        scratch_workers=args.scratch_workers,
        detector_workers=args.detector_workers,
        standardize_workers=args.standardize_workers,
        train_fraction=args.train_fraction,
    )


def run_cleanup_command(args: argparse.Namespace) -> str:
    input_path = args.input_path
    output_path = args.output_path
    config = CleanupConfig(enable_artwork_inpainting=args.enable_artwork_inpainting)
    pipeline = build_default_manga_cleanup_pipeline(config)
    if input_path.is_dir():
        saved = pipeline.clean_directory(input_path, output_path)
        return f"cleaned {len(saved)} image(s) into {output_path}"
    if input_path.is_file():
        saved_path = pipeline.clean_image(input_path, cleanup_file_output_path(input_path, output_path))
        return f"cleaned 1 image into {saved_path}"
    raise FileNotFoundError(f"Cleanup input must be a file or directory; got {input_path}.")


def cleanup_file_output_path(input_path: Path, output_path: Path) -> Path:
    if output_path.suffix:
        return output_path
    return output_path / input_path.name


def polished_args(args: argparse.Namespace) -> list[str]:
    return [
        "--output",
        str(args.output),
        "--plan-output",
        str(args.plan_output),
        "--min-usable-pages-per-anchor",
        str(args.min_usable_pages_per_anchor),
    ]


def command_exit_report(exit_code: int) -> dict[str, int]:
    if exit_code != 0:
        raise DatasetError(f"Command failed with exit code {exit_code}.")
    return {"exit_code": exit_code}


def run_quietly(command: Callable[[], int]) -> int:
    with contextlib.redirect_stdout(io.StringIO()):
        return command()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def worker_count(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("worker count must be >= 1")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def fraction(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed < 1:
        raise argparse.ArgumentTypeError("value must be > 0 and < 1")
    return parsed


def comma_separated_ints(value: str) -> set[int]:
    if not value.strip():
        return set()
    return {parse_label_id(part) for part in value.split(",") if part.strip()}


def parse_label_id(part: str) -> int:
    parsed = int(part.strip())
    if parsed < 0:
        raise argparse.ArgumentTypeError("labels must be >= 0")
    return parsed
