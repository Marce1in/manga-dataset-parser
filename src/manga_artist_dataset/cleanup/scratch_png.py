"""Temporary PNG preparation for the local cleanup stage."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

from manga_artist_dataset.io.files import sha256_file, workspace_path
from manga_artist_dataset.json_types import JsonObject


def prepare_scratch_png_records(records: list[JsonObject], input_root: Path, output_root: Path) -> list[JsonObject]:
    """Write scratch PNG files and return transient metadata for them.

    Example:
        `prepare_scratch_png_records(records, Path("polished"), Path("scratch/png"))`.
    """
    return [scratch_png_record(record, input_root, output_root) for record in records]


def scratch_png_record(record: JsonObject, input_root: Path, output_root: Path) -> JsonObject:
    """Write one source image as scratch PNG metadata.

    Example:
        `scratch_png_record(record, Path("polished"), Path("scratch/png"))`.
    """
    source_path = resolve_record_path(record, "output_path")
    ensure_under_root(source_path, input_root)
    destination = scratch_png_path(source_path, input_root, output_root)
    width, height = convert_one_image(source_path, destination)
    return transient_png_record(record, destination, width, height)


def resolve_record_path(record: JsonObject, key: str) -> Path:
    """Resolve a path-valued metadata key.

    Example:
        `resolve_record_path(record, "output_path")`.
    """
    value = record.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Record key {key} must be a path string; got {value!r}.")
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def ensure_under_root(path: Path, root: Path) -> None:
    """Reject paths outside the expected dataset root.

    Example:
        `ensure_under_root(Path("root/a").resolve(), Path("root").resolve())`.
    """
    if path.is_relative_to(root):
        return
    raise ValueError(f"Source image {path} must be under input root {root}.")


def scratch_png_path(source_path: Path, input_root: Path, output_root: Path) -> Path:
    """Map a source image path to its scratch PNG path.

    Example:
        `scratch_png_path(source, input_root, output_root)`.
    """
    relative = source_path.relative_to(input_root)
    return output_root / relative.parent / f"{source_path.stem}.png"


def convert_one_image(source_path: Path, destination: Path) -> tuple[int, int]:
    """Convert one image to RGB PNG.

    Example:
        `convert_one_image(Path("page.jpg"), Path("scratch/page.png"))`.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rgb = image_to_rgb(image)
        rgb.save(destination, format="PNG", compress_level=6)
        return rgb.size


def image_to_rgb(image: Image.Image) -> Image.Image:
    """Normalize a Pillow image to RGB over a white background.

    Example:
        `rgb = image_to_rgb(image)`.
    """
    transposed = ImageOps.exif_transpose(image)
    if transposed.mode in {"RGBA", "LA"} or "transparency" in transposed.info:
        return flatten_alpha(transposed)
    return transposed.convert("RGB")


def flatten_alpha(image: Image.Image) -> Image.Image:
    """Composite transparent pixels over white before RGB conversion.

    Example:
        `flatten_alpha(image)`.
    """
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def transient_png_record(record: JsonObject, destination: Path, width: int, height: int) -> JsonObject:
    """Return scratch metadata used only during the current cleanup run.

    Example:
        `transient_png_record(record, Path("scratch/page.png"), 512, 768)`.
    """
    transient = dict(record)
    transient["stage"] = "scratch_png_pages"
    transient["output_path"] = workspace_path(destination)
    transient["image_format"] = "PNG"
    transient["color_mode"] = "RGB"
    transient["sha256"] = sha256_file(destination)
    transient["width"] = width
    transient["height"] = height
    transient["bytes"] = destination.stat().st_size
    return transient
