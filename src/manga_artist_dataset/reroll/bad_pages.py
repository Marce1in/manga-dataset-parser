#!/usr/bin/env python3
"""Replace manually rejected output images with nearby valid candidates."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from manga_artist_dataset.build.chapter_sources import chapters_from_source
from manga_artist_dataset.build.image_probe import (
    accepted_page_outputs,
    image_dimensions,
    page_rejection_reason,
)
from manga_artist_dataset.build.manifest import load_manifest
from manga_artist_dataset.build.page_reader import read_page_bytes
from manga_artist_dataset.build.writer import output_file_name
from manga_artist_dataset.models import BuildConfig, Chapter, PageRef, TargetSpec

BAD_STEMS = [
    "03_tatsuki_fujimoto__20_anchor_chapter_48__p0012__0040",
    "03_tatsuki_fujimoto__20_anchor_chapter_48__p0009__0051",
    "03_tatsuki_fujimoto__20_anchor_chapter_48__p0010__0003",
    "05_kentaro_miura__20_anchor_chapter_73__p0018__0016",
    "05_kentaro_miura__20_anchor_chapter_74__p0009__0032",
    "06_masashi_kishimoto__20_anchor_chapter_140_volume_16__p0038__right__0002",
    "06_masashi_kishimoto__20_anchor_chapter_140_volume_16__p0037__right__0002",
    "06_masashi_kishimoto__20_anchor_chapter_140_volume_16__p0074__left__0013",
    "06_masashi_kishimoto__20_anchor_chapter_140_volume_16__p0074__right__0014",
    "06_masashi_kishimoto__50_anchor_chapter_350_volume_39__p0006__right__0016",
    "06_masashi_kishimoto__50_anchor_chapter_350_volume_39__p0051__right__0052",
    "06_masashi_kishimoto__50_anchor_chapter_350_volume_39__p0088__right__0040",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0006__0059",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0006__0053",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0007__0059",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0007__0053",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0008__0037",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0008__0053",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0010__0053",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0010__0037",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0009__left__0037",
    "06_masashi_kishimoto__50_anchor_chapter_350__p0009__right__0053",
    "06_masashi_kishimoto__90_anchor_chapter_630_volume_66__p0040__left__0029",
    "06_masashi_kishimoto__90_anchor_chapter_630_volume_66__p0040__right__0030",
    "06_masashi_kishimoto__90_anchor_chapter_630_volume_66__p0059__right__0036",
    "07_eiichiro_oda__20_anchor_chapter_237_high_in_the_sky__p0007__0013",
    "07_eiichiro_oda__20_anchor_chapter_237_high_in_the_sky__p0006__0013",
    "09_hirohiko_araki__part_4_midpoint_chapter_86__p0006__0054",
    "10_gege_akutami__50_anchor_chapter_135_the_shibuya_incident_part_52__p0008__0032",
    "10_gege_akutami__50_anchor_chapter_135_the_shibuya_incident_part_52__p0023__0060",
    "10_gege_akutami__50_anchor_chapter_135_the_shibuya_incident_part_52__p0025__0051",
    "10_gege_akutami__50_anchor_chapter_135_the_shibuya_incident_part_52__p0027__0056",
    "10_gege_akutami__50_anchor_chapter_135_the_shibuya_incident_part_52__p0029__0007",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0006__0008",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0007__0039",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0008__0045",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0009__0002",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0010__0014",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0011__0057",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0012__0026",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0013__0033",
    "10_gege_akutami__50_anchor_chapter_136_the_shibuya_incident_part_53__p0014__0020",
]


@dataclass(frozen=True)
class CandidateOutput:
    page: PageRef
    data: bytes
    split_part: str | None
    suffix_override: str | None
    distance: tuple[Any, ...]


def record_key(record: dict[str, Any]) -> tuple[int, str, str | None]:
    return (
        int(record["label_id"]),
        f"{record['chapter']}::p{int(record['original_page_index']):04d}",
        record.get("split_part"),
    )


def candidate_key(label_id: int, page: PageRef, split_part: str | None) -> tuple[int, str, str | None]:
    return (label_id, f"{page.chapter_label}::p{page.page_index:04d}", split_part)


def output_index_from_path(path: str) -> int:
    match = re.search(r"__(\d{4})\.[^.]+$", path)
    if not match:
        raise RuntimeError(f"Could not parse output index from {path}")
    return int(match.group(1))


def anchor_prefix(label: str) -> str:
    return label.split(":", 1)[0] if ":" in label else label


def chapter_number(label: str) -> float | None:
    match = re.search(r"Chapter\s+(\d+(?:\.\d+)?)", label, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def load_target_chapters(
    target: TargetSpec,
    manifest_dir: Path,
    temp_dir: Path,
) -> list[Chapter]:
    return [
        chapters_from_source(target, source, manifest_dir, temp_dir, allow_downloads=True) for source in target.sources
    ]


def http_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "manga-reroll/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def extract_cubari_pages(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item["url"] if isinstance(item, dict) else item) for item in value]
    if isinstance(value, str) and value.startswith("/proxy/api/imgur/chapter/"):
        album = value.rstrip("/").split("/")[-1]
        payload = http_json(f"https://cubari.moe/read/api/imgur/chapter/{album}/")
        return [str(item["src"]) for item in payload if isinstance(item, dict) and item.get("src")]
    return []


def extra_jjk_chapters(target: TargetSpec, manifest_dir: Path, temp_dir: Path) -> list[Chapter]:
    jjk = http_json("https://raw.githubusercontent.com/JJKvault/JJK-chapters/master/JJK.json")
    chapters: list[Chapter] = []
    # Extra nearby Shibuya/Culling Game chapters to replace the chapter 135/136
    # watermark pages without jumping to a different manga/source family.
    for key in ["134", "139", "140", "141", "142", "143", "144", "145", "146", "147", "148"]:
        item = jjk["chapters"].get(key)
        if not item:
            continue
        pages: list[str] = []
        for value in (item.get("groups") or {}).values():
            pages = extract_cubari_pages(value)
            if pages:
                break
        if not pages:
            continue
        source = {
            "chapter": f"50% anchor: Chapter {key} - {item.get('title') or ''}".strip(),
            "pages": pages,
            "permission_note": "Extra public Cubari/JJKvault pages used only for manual reroll replacements.",
        }
        chapters.append(chapters_from_source(target, source, manifest_dir, temp_dir, allow_downloads=True))
    return chapters


def candidate_chapter_order(chapters: list[Chapter], old_chapter: str) -> list[Chapter]:
    old_prefix = anchor_prefix(old_chapter)
    old_number = chapter_number(old_chapter)

    def score(chapter: Chapter) -> tuple[int, float, tuple[Any, ...]]:
        same = chapter.label == old_chapter
        same_anchor = anchor_prefix(chapter.label) == old_prefix
        number = chapter_number(chapter.label)
        number_distance = abs(number - old_number) if number is not None and old_number is not None else 9999.0
        if same:
            group = 0
        elif same_anchor:
            group = 1
        else:
            group = 2
        return (group, number_distance, chapter.sort_key)

    return sorted(chapters, key=score)


def candidate_page_order(chapter: Chapter, old_record: dict[str, Any]) -> list[PageRef]:
    old_index = int(old_record["original_page_index"])
    pages = chapter.usable_pages(5, 5)
    return sorted(pages, key=lambda page: (abs(page.page_index - old_index), page.page_index))


def candidate_outputs_for_page(
    page: PageRef,
    config: BuildConfig,
    allow_split: bool,
    distance: tuple[Any, ...],
) -> list[CandidateOutput]:
    data = read_page_bytes(page)
    outputs, rejected, _ = accepted_page_outputs(data, config, allow_split=allow_split)
    if rejected and not outputs:
        return []
    return [
        CandidateOutput(
            page=page,
            data=output.content,
            split_part=output.split_part,
            suffix_override=output.suffix_override,
            distance=distance,
        )
        for output in outputs
    ]


def choose_replacement(
    target: TargetSpec,
    chapters: list[Chapter],
    old_record: dict[str, Any],
    used_keys: set[tuple[int, str, str | None]],
    banned_keys: set[tuple[int, str, str | None]],
    banned_output_stems: set[str],
    config: BuildConfig,
) -> CandidateOutput:
    old_chapter = str(old_record["chapter"])
    old_page_index = int(old_record["original_page_index"])
    old_split = old_record.get("split_part")
    output_index = output_index_from_path(old_record["output_path"])
    allow_split = target.label_id in config.split_double_spread_label_ids

    for chapter_rank, chapter in enumerate(candidate_chapter_order(chapters, old_chapter)):
        for page in candidate_page_order(chapter, old_record):
            page_rank = abs(page.page_index - old_page_index)
            outputs = candidate_outputs_for_page(
                page,
                config,
                allow_split=allow_split,
                distance=(chapter_rank, page_rank, page.page_index),
            )
            if old_split is not None:
                outputs.sort(key=lambda item: (item.split_part != old_split, item.distance))
            for candidate in outputs:
                key = candidate_key(target.label_id, candidate.page, candidate.split_part)
                if key in used_keys or key in banned_keys:
                    continue
                candidate_stem = Path(
                    output_file_name(
                        target,
                        candidate.page,
                        output_index,
                        suffix_override=candidate.suffix_override,
                        split_part=candidate.split_part,
                    )
                ).stem
                if candidate_stem in banned_output_stems:
                    continue
                return candidate

    raise RuntimeError(f"No replacement found for {old_record['output_path']}")


def main() -> int:
    output_dir = Path("artifacts/datasets/polished_pages")
    metadata_path = output_dir / "metadata.jsonl"
    report_path = output_dir / "dataset_report.json"
    manifest_path = Path("manifests/manga_sources.polished.json")

    records = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
    records_by_stem = {Path(record["output_path"]).stem: record for record in records}
    process_stems = [stem for stem in BAD_STEMS if stem in records_by_stem]
    if not process_stems:
        print(json.dumps({"rerolled": 0, "report": "artifacts/reports/reroll_report.json"}, indent=2))
        return 0

    bad_records = [records_by_stem[stem] for stem in process_stems]
    banned_keys = {record_key(record) for record in bad_records}
    used_keys = {record_key(record) for record in records if record_key(record) not in banned_keys}

    targets = {target.label_id: target for target in load_manifest(manifest_path)}
    config = BuildConfig(
        sources_path=manifest_path,
        output_dir=output_dir,
        pages_per_artist=60,
        trim_start=5,
        trim_end=5,
        min_chapters=3,
        allow_downloads=True,
        use_all_sources=True,
        filter_color_pages=True,
        filter_double_spreads=True,
        split_double_spread_label_ids=set(),
    )

    rerolls = []
    with tempfile.TemporaryDirectory(prefix="manga_reroll_") as temp_name:
        temp_dir = Path(temp_name)
        chapters_by_label: dict[int, list[Chapter]] = {}
        for label_id in sorted({int(record["label_id"]) for record in bad_records}):
            target_chapters = load_target_chapters(targets[label_id], manifest_path.parent, temp_dir)
            if label_id == 10:
                target_chapters.extend(extra_jjk_chapters(targets[label_id], manifest_path.parent, temp_dir))
            chapters_by_label[label_id] = target_chapters

        replacement_plans = []
        for stem in process_stems:
            old_record = records_by_stem[stem]
            label_id = int(old_record["label_id"])
            target = targets[label_id]
            replacement = choose_replacement(
                target,
                chapters_by_label[label_id],
                old_record,
                used_keys,
                banned_keys,
                set(BAD_STEMS),
                config,
            )
            output_index = output_index_from_path(old_record["output_path"])
            old_path = Path(old_record["output_path"])
            new_path = old_path.parent / output_file_name(
                target,
                replacement.page,
                output_index,
                suffix_override=replacement.suffix_override,
                split_part=replacement.split_part,
            )
            if old_path.exists():
                pass
            width, height = image_dimensions(replacement.data)
            reason = page_rejection_reason(replacement.data, config)
            if reason is not None:
                raise RuntimeError(f"Replacement failed filter: {new_path} -> {reason}")

            new_key = candidate_key(label_id, replacement.page, replacement.split_part)
            used_keys.add(new_key)
            replacement_plans.append(
                {
                    "stem": stem,
                    "old_record": old_record,
                    "old_path": old_path,
                    "new_path": new_path,
                    "replacement": replacement,
                    "width": width,
                    "height": height,
                }
            )

        for plan in replacement_plans:
            old_record = plan["old_record"]
            old_path = plan["old_path"]
            new_path = plan["new_path"]
            replacement = plan["replacement"]
            if old_path.exists():
                old_path.unlink()
            new_path.write_bytes(replacement.data)
            old_record.update(
                {
                    "chapter": replacement.page.chapter_label,
                    "original_page_index": replacement.page.page_index,
                    "split_part": replacement.split_part,
                    "output_path": str(new_path),
                    "source_ref": replacement.page.source_ref,
                    "sha256": hashlib.sha256(replacement.data).hexdigest(),
                    "width": plan["width"],
                    "height": plan["height"],
                    "bytes": len(replacement.data),
                }
            )
            rerolls.append(
                {
                    "old_stem": plan["stem"],
                    "new_file": str(new_path),
                    "new_chapter": replacement.page.chapter_label,
                    "new_page_index": replacement.page.page_index,
                    "new_split_part": replacement.split_part,
                }
            )

    metadata_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    manual_rerolls = list(report.get("manual_rerolls") or []) + rerolls
    report["manual_rerolls"] = manual_rerolls
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reroll_report_path = Path("artifacts/reports/reroll_report.json")
    reroll_report_path.parent.mkdir(parents=True, exist_ok=True)
    reroll_report_path.write_text(
        json.dumps(manual_rerolls, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"rerolled": len(rerolls), "report": "artifacts/reports/reroll_report.json"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
