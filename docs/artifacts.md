# Artifacts

`manifests/` stores source manifests.

`artifacts/datasets/` stores generated image datasets:

- `polished_pages`: source page cache used to avoid repeated downloads
- `panel_cleaned_pages`: final local cleanup output standardized to 512x768 PNG

`panel_cleaned_pages` stores images under split and class folders:

- `train/<class_slug>/*.png`
- `test/<class_slug>/*.png`

Each generated dataset root also contains:

- `metadata.jsonl`
- `dataset_report.json`

`clean-panels` creates temporary PNG inputs for the local cleanup pipeline and deletes them
after the final standardized dataset is written.

`artifacts/reports/` stores non-image reports such as chapter plans and reroll
logs.

Large generated folders and local environments are ignored
by git. Rebuild them through the documented `uv run manga-dataset` commands.
