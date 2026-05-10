# Commands

Canonical entrypoint:

```bash
uv run manga-dataset <command>
```

Build a dataset:

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

Generate the polished source manifest:

```bash
uv run manga-dataset build-polished-manifest
```

Apply curated rerolls:

```bash
uv run manga-dataset reroll
```

Clean arbitrary downloaded images:

```bash
uv run python -m manga_artist_dataset cleanup ./downloaded-images ./cleaned-images
```

Run the local cleanup pipeline and write the final 512x768 PNG dataset:

```bash
uv run manga-dataset clean-panels \
  --scratch-workers 8 \
  --detector-workers 2 \
  --standardize-workers 8 \
  --train-fraction 0.8 \
  --overwrite
```

Quality checks:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy .
uv run pytest
```
