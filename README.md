# Manga Artist Dataset

Dataset pipeline for manga artist page-classification experiments.

The code is packaged under `src/manga_artist_dataset` and is run through `uv`.
Generated datasets live under `artifacts/datasets`; source manifests live under
`manifests`.

## Commands

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
  --overwrite
```

```bash
uv run manga-dataset clean-panels --overwrite
```

```bash
uv run python -m manga_artist_dataset cleanup ./downloaded-images ./cleaned-images
```

## Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest
```

See `docs/` for architecture, artifact layout, and command details.
