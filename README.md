# Manga Artist Dataset

Dataset pipeline for manga artist page-classification experiments.

The project builds a curated page dataset from manga source manifests, applies
manual rerolls when needed, and can clean speech-bubble text from the generated
pages. The package lives under `src/manga_artist_dataset` and is run with `uv`.

## Requirements

- Python 3.12 or newer
- `uv`
- Network access for first-time dependency sync and optional page downloads

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Confirm the tools are available:

```bash
python3 --version
uv --version
```

## Install

Clone the repository and enter the project directory:

```bash
git clone git@github.com:Marce1in/manga-dataset-parser.git
cd manga-dataset-parser
```

Sync the project environment:

```bash
uv sync
```

Verify the CLI is installed in the project environment:

```bash
uv run manga-dataset --help
```

All commands in this README use `uv run` so they run against the locked project
dependencies from `uv.lock`.

## Quick Start

Build the polished source manifest:

```bash
uv run manga-dataset build-polished-manifest
```

Build the page dataset from that manifest:

```bash
uv run manga-dataset build \
  --sources manifests/manga_sources.polished.json \
  --output artifacts/datasets/polished_pages \
  --pages-per-artist 60 \
  --trim-start 5 \
  --trim-end 5 \
  --min-chapters 3 \
  --allow-downloads \
  --use-all-sources \
  --filter-color-pages \
  --filter-double-spreads \
  --download-workers 12 \
  --download-host-delay-seconds 0.1 \
  --overwrite
```

Apply curated replacements for rejected pages:

```bash
uv run manga-dataset reroll
```

Clean and standardize the dataset into 512x768 PNG files:

```bash
uv run manga-dataset clean-panels \
  --scratch-workers 8 \
  --detector-workers 2 \
  --standardize-workers 8 \
  --train-fraction 0.8 \
  --overwrite
```

The final cleaned dataset is written to
`artifacts/datasets/panel_cleaned_pages`.

## Usage Tutorial

### 1. Generate a Source Manifest

The polished manifest describes which manga pages should be considered for each
artist class.

```bash
uv run manga-dataset build-polished-manifest \
  --output manifests/manga_sources.polished.json \
  --plan-output artifacts/reports/chapter_plan.polished.json \
  --min-usable-pages-per-anchor 30
```

Outputs:

- `manifests/manga_sources.polished.json`
- `artifacts/reports/chapter_plan.polished.json`

### 2. Build the Source Page Dataset

Use the manifest to download or reuse source pages and write a class-folder
dataset.

```bash
uv run manga-dataset build \
  --sources manifests/manga_sources.polished.json \
  --output artifacts/datasets/polished_pages \
  --pages-per-artist 60 \
  --allow-downloads
```

Useful options:

- `--dry-run`: report the planned dataset without writing files.
- `--overwrite`: replace an existing output directory.
- `--allow-short`: allow classes with fewer pages than requested.
- `--use-all-sources`: read every source in the manifest instead of sampling
  the minimum needed.
- `--filter-color-pages`: skip color pages.
- `--filter-double-spreads`: skip double-page spreads.
- `--split-double-spreads`: split double-page spreads instead of skipping them.
- `--download-workers`: set the number of parallel download workers.
- `--download-host-delay-seconds`: pause between requests to the same host.
- `--no-strict-targets`: disable strict expected-class validation.

The dataset output contains one folder per class plus:

- `metadata.jsonl`
- `dataset_report.json`

### 3. Reroll Rejected Pages

Run the reroll command after building `artifacts/datasets/polished_pages` when
curated rejected pages need nearby replacements.

```bash
uv run manga-dataset reroll
```

This updates the polished page cache and writes reroll reporting under
`artifacts/reports`.

### 4. Clean Panels

Run the panel cleaner after the polished source dataset exists.

```bash
uv run manga-dataset clean-panels \
  --input-root artifacts/datasets/polished_pages \
  --output-root artifacts/datasets/panel_cleaned_pages \
  --scratch-workers 8 \
  --detector-workers 2 \
  --standardize-workers 8 \
  --overwrite
```

The cleaner removes normal speech-bubble text, writes final PNG files, and
standardizes every output image to 512x768.

Worker options control different stages:

- `--scratch-workers`: converts source images into temporary PNG inputs.
- `--detector-workers`: runs RT-DETR cleanup shards.
- `--standardize-workers`: writes the final standardized PNG dataset.

Each detector worker loads its own model, so keep `--detector-workers` low until
memory use has been measured.

### 5. Clean One Image or Directory

For ad hoc cleanup outside the dataset builder, pass an input file or directory
and an output file or directory.

```bash
uv run manga-dataset cleanup ./downloaded-images ./cleaned-images
```

For a single image:

```bash
uv run manga-dataset cleanup ./page.jpg ./cleaned/page.png
```

Add `--enable-artwork-inpainting` only when you want the optional artwork text
inpainting path. It is disabled by default.

## Artifact Layout

Generated files are intentionally kept out of source control.

- `manifests/`: source manifests
- `artifacts/datasets/polished_pages`: source page cache
- `artifacts/datasets/panel_cleaned_pages`: final cleaned 512x768 PNG dataset
- `artifacts/reports/`: chapter plans, reroll reports, and other reports

See `docs/artifacts.md` and `docs/source-provenance.md` for more detail.

## Development Checks

Run the full local check suite before committing changes:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest
```

Format and fix lint issues with Ruff:

```bash
uv run ruff format .
uv run ruff check . --fix
```

## More Documentation

- `docs/architecture.md`: package boundaries
- `docs/artifacts.md`: generated file layout
- `docs/cleanup.md`: cleanup pipeline behavior
- `docs/commands.md`: command reference
- `docs/source-provenance.md`: source and reroll notes
