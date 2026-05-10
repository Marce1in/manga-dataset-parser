# Cleanup Pipeline

The cleanup stage removes normal manga speech-bubble text after images have
already been downloaded into `artifacts/datasets/polished_pages`.

It does not translate, OCR, render replacement text, or clean every artwork/SFX
sound effect yet. Artwork text inpainting is optional and disabled by default.

## Pipeline

The production pipeline lives in `manga_artist_dataset.cleanup`:

- `detector.py`: RT-DETR-v2 boundary for `ogkalu/comic-text-and-bubble-detector`
- `bubble_masks.py`: OpenCV bright-region masks for detected speech bubbles
- `text_cleaner.py`: dark-stroke clearing inside the bubble mask
- `pipeline.py`: image and directory orchestration
- `dataset_cleaner.py`: final dataset wrapper, metadata, and standardization

The detector emits project-owned `DetectedRegion` values. Hugging Face model
outputs do not leave the detector boundary.

## Commands

Clean arbitrary downloaded images:

```bash
uv run python -m manga_artist_dataset cleanup ./downloaded-images ./cleaned-images
```

Rebuild the existing final dataset artifact:

```bash
uv run manga-dataset clean-panels --overwrite
```

`clean-panels` converts source images to scratch PNGs, runs the local cleanup
pipeline, standardizes final images to 512x768 PNG, and writes metadata.

## Strength

The default profile follows the FrankYomik reference behavior: detector
confidence is 0.35, dark text is detected below luminance 160, the text mask is
dilated once with a 3px kernel, and the final clear rectangle gets a 1px margin.
Bubble mask extraction uses a brightness threshold of 200 with 10px bbox
padding.

## Dependencies

Required runtime dependencies are Pillow, NumPy, OpenCV, Torch, and
Transformers. Torch/Transformers are only used by the detector boundary.

LaMa artwork inpainting is optional. `simple-lama-inpainting` is not a required
dependency because its transitive pins can conflict with image libraries.

## pcleaner

This project no longer shells out to `pcleaner-cli`. The old stage was replaced
because the new implementation needs project-owned detection, mask extraction,
and bubble text clearing that can be tested and extended independently.
