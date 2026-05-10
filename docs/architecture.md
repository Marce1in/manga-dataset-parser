# Architecture

The project is frameworkless and uses explicit composition.

`manga_artist_dataset.build` owns the source-manifest dataset builder:
manifest parsing, source expansion, page filtering, selection, and writing.

`manga_artist_dataset.cleanup` owns post-processing:
RT-DETR speech-bubble detection, OpenCV mask extraction, Pillow/OpenCV
dark-stroke clearing, temporary PNG preparation, and final 512x768
standardization.

`manga_artist_dataset.polished` owns the curated 60-page source manifest
generation workflow.

`manga_artist_dataset.reroll` owns manual replacement of rejected polished
pages.

`manga_artist_dataset.io` contains boundary helpers for JSONL, files, and HTTP.

Domain objects live in `models.py`; project exceptions live in `errors.py`.
Third-party libraries are kept at boundary modules where possible.
