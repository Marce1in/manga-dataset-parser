"""Configuration for manga speech bubble cleanup."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CleanupConfig:
    detector_model_id: str = "ogkalu/comic-text-and-bubble-detector"
    detector_confidence: float = 0.35
    artwork_text_min_confidence: float = 0.60
    deduplicate_overlap_ratio: float = 0.50
    dark_text_threshold: int = 160
    text_clear_margin: int = 1
    text_dilation_kernel_size: int = 3
    text_dilation_iterations: int = 1
    bubble_mask_threshold: int = 200
    bubble_mask_padding: int = 10
    enable_artwork_inpainting: bool = False
    inpaint_padding: int = 20
    supported_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
