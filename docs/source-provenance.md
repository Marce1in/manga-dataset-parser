# Source Provenance

The current source cache is `artifacts/datasets/polished_pages/`.

Generated source-control inputs:

- `manifests/manga_sources.polished.json`
- `artifacts/reports/chapter_plan.polished.json`
- `artifacts/reports/reroll_report.json`

The normal title anchors are 20%, 90%, and 50% through the chapter list. For
`[1 2 3 4 5 6 7 8 9 10]`, that gives `2`, `9`, and `5`.

JoJo uses midpoint chapters from part 2, part 4, and part 7 instead. JoJo
sampling is balanced by part through `sample_group` values in the manifest.

Dragon Ball and Dragon Ball Z are counted as one 519-chapter run.

No Manga Plus sources were used.

Primary source families:

- MangaDex public API and at-home image server
- Monster Cubari manifest: `Dinis-CM/MonsterCubari`
- One Piece Cubari manifest: `celsiusnarhwal/punk-records`
- Jujutsu Kaisen Cubari manifest: `mcradcliffe2490/hidden-inventory`
- Berserk GitHub ZIPs: `s1ddly/Berserk-DL`
- Naruto Mangapill pages with direct CDN image URLs

Final source notes:

- Jujutsu Kaisen uses the `hidden-inventory` Cubari manifest for all anchors.
- Naruto uses Mangapill chapter pages with a required `Referer` header.
- JoJo part 4 uses MangaDex `pt-br` chapters 86, 87, and 88.
- Earlier EverythingMoe monitor checks exposed uptime data, not usable page
  arrays for this dataset.

Manual rerolls are handled by:

```bash
uv run manga-dataset reroll
```

The reroll script searches nearby valid pages and preserves 60 pages per class.

Final polished-cache verification:

- `metadata.jsonl`: 600 records
- every class folder: 60 files
- split `left/right` filenames: 0
- Naruto source refs: 60 current CDN refs, 0 old PDF refs
- Jujutsu Kaisen source refs: 60 current Cubari CDN refs

Final cleaned dataset verification:

- `panel_cleaned_pages`: 600 PNG files
- every image size: 512x768
- every class folder: 60 files
