from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from maple_price_tool.config import load_config


TASK_DIRS = {
    "option_label": "option_labels",
    "option_value": "option_values",
    "price": "prices",
    "rejected": "rejected",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find or quarantine bad dataset samples.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--task", choices=tuple(TASK_DIRS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    dataset_dir = args.dataset_dir or config.vision.training_dataset_dir
    metadata = dataset_dir / TASK_DIRS[args.task] / "samples.jsonl"
    rows = read_rows(metadata)
    flagged = find_flagged_rows(metadata, rows, args.task)
    for index, row, reasons in flagged:
        print(f"{index:04d} reasons={','.join(reasons)} image={row.get('image_path', '')}")
    print(f"flagged={len(flagged)} total={len(rows)} mode={'apply' if args.apply else 'dry-run'}")
    if args.apply:
        quarantine(metadata, rows, flagged, dataset_dir, args.task)
    return 0


def read_rows(metadata: Path) -> list[dict]:
    if not metadata.exists():
        return []
    return [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_flagged_rows(metadata: Path, rows: list[dict], task: str) -> list[tuple[int, dict, list[str]]]:
    hash_labels: dict[str, set[str]] = {}
    hash_counts: dict[str, int] = {}
    for row in rows:
        content_hash = str(row.get("content_hash", ""))
        if not content_hash:
            continue
        hash_labels.setdefault(content_hash, set()).add(str(row.get("label", "")))
        hash_counts[content_hash] = hash_counts.get(content_hash, 0) + 1
    flagged = []
    for index, row in enumerate(rows):
        reasons = row_reasons(metadata, row, task, hash_labels, hash_counts)
        if reasons:
            flagged.append((index, row, reasons))
    return flagged


def row_reasons(
    metadata: Path,
    row: dict,
    task: str,
    hash_labels: dict[str, set[str]],
    hash_counts: dict[str, int],
) -> list[str]:
    reasons: list[str] = []
    image_path = Path(str(row.get("image_path", "")))
    if not image_path.is_absolute():
        image_path = metadata.parent / image_path
    if row.get("review_status") == "rejected" or row.get("label_quality") == "rejected":
        reasons.append("rejected")
    content_hash = str(row.get("content_hash", ""))
    if content_hash and hash_counts.get(content_hash, 0) > 1:
        reasons.append("duplicate_hash")
    if content_hash and len(hash_labels.get(content_hash, set())) > 1:
        reasons.append("conflicting_label")
    source = str(row.get("source_image_path", "")).strip()
    if source and not Path(source).exists():
        reasons.append("source_image_missing")
    if task == "price" and not re.fullmatch(r"[0-9,]+", str(row.get("label", ""))):
        reasons.append("invalid_price_format")
    if not image_path.exists():
        reasons.append("missing_image")
        return reasons
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        reasons.append("empty_image")
        return reasons
    if image.shape[1] > 384:
        reasons.append("crop_too_large")
    if image.shape[0] < 6 or image.shape[1] < 8:
        reasons.append("crop_too_small")
    if int(np.max(image)) == int(np.min(image)):
        reasons.append("empty_image")
    return reasons


def quarantine(
    metadata: Path,
    rows: list[dict],
    flagged: list[tuple[int, dict, list[str]]],
    dataset_dir: Path,
    task: str,
) -> None:
    flagged_indices = {index for index, _row, _reasons in flagged}
    quarantine_dir = dataset_dir / "quarantine" / task
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    manifest = quarantine_dir / "manifest.jsonl"
    kept_rows = []
    for index, row in enumerate(rows):
        if index not in flagged_indices:
            kept_rows.append(row)
            continue
        reasons = next(reasons for flagged_index, _row, reasons in flagged if flagged_index == index)
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = metadata.parent / image_path
        moved_to = ""
        if image_path.exists():
            target = quarantine_dir / image_path.name
            shutil.move(str(image_path), str(target))
            moved_to = str(target)
        manifest.open("a", encoding="utf-8").write(
            json.dumps({"row": row, "reasons": reasons, "moved_to": moved_to}, ensure_ascii=False) + "\n"
        )
    backup = metadata.with_suffix(metadata.suffix + ".bak")
    if metadata.exists():
        shutil.copyfile(metadata, backup)
    metadata.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in kept_rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
