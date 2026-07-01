from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from maple_price_tool.config import load_config
from recognition.ctc_decoder import OPTION_VALUE_CHARSET, PRICE_CHARSET
from recognition.dataset import SampleRecord, duplicate_hashes, split_saved_training_image
from recognition.option_classifier import default_option_class_names
from recognition.training_samples import FIELD_TO_OPTION_KEY, canonical_option_key, is_non_option_line, normalize_value_text


TASKS = ("item_metadata", "option_label", "option_value", "price", "rejected")
TASK_DIRS = {
    "item_metadata": "item_metadata",
    "option_label": "option_labels",
    "option_value": "option_values",
    "price": "prices",
    "rejected": "rejected",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect recognition training datasets.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--show-invalid", action="store_true")
    parser.add_argument("--export-report")
    parser.add_argument("--export-html")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    root = args.dataset_dir or config.vision.training_dataset_dir
    tasks = [args.task] if args.task else list(TASKS)
    report = {task: inspect_task(root, task) for task in tasks}
    for task, info in report.items():
        readiness = info["readiness"]
        print(
            f"[{task}] samples={info['total']} sessions={info['sessions']} "
            f"missing_images={info['missing_images']} duplicates={info['duplicate_hashes']} "
            f"issues={info['issue_count']}"
        )
        print(f"  quality={info['label_quality_counts']}")
        print(f"  labels={info['label_counts']}")
        print(f"  readiness={readiness}")
        if info["issues"] and args.show_invalid:
            for issue in info["issues"][:50]:
                print(f"  issue={issue}")
        if info["total"] == 0:
            print("  no samples yet; save reviewed records to collect data.")
    if args.export_report:
        target = Path(args.export_report)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in {".html", ".htm"}:
            target.write_text(render_html_report(report), encoding="utf-8")
        else:
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.export_html:
        target = Path(args.export_html)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_html_report(report), encoding="utf-8")
    return 0


def inspect_task(root: Path, task: str) -> dict:
    metadata = root / TASK_DIRS[task] / "samples.jsonl"
    if not metadata.exists():
        return empty_report(task)
    rows = read_rows(metadata)
    records = [record_from_row(metadata, row) for row in rows]
    issues = collect_quality_issues(records, rows, task)
    label_counts = Counter(record.label for record in records)
    quality_counts = Counter(record.label_quality for record in records)
    sessions = {record.session_id for record in records if record.session_id}
    conflicting = conflicting_labels(rows)
    duplicate_count = len(duplicate_hashes(records))
    reason_counts = Counter(str(row.get("rejection_reason") or row.get("selection_reason") or "") for row in rows if task == "rejected")
    metadata_key_counts = Counter(str(row.get("metadata_key") or "") for row in rows if task == "item_metadata")
    report = {
        "task": task,
        "total": len(records),
        "label_quality_counts": dict(quality_counts),
        "review_status_counts": dict(Counter(record.review_status for record in records)),
        "sessions": len(sessions),
        "label_counts": dict(label_counts),
        "metadata_key_counts": dict(metadata_key_counts),
        "length_distribution": dict(Counter(len(record.label) for record in records)),
        "missing_images": sum(1 for record in records if not record.image_path.exists()),
        "duplicate_hashes": duplicate_count,
        "conflicting_labels": conflicting,
        "rejection_reason_counts": dict(reason_counts),
        "issues": issues + [f"conflicting_label content_hash={key} labels={labels}" for key, labels in conflicting.items()],
        "issue_count": len(issues) + len(conflicting),
        "readiness": readiness_levels(len(records), len(sessions), len(issues) + len(conflicting), label_counts, task),
        "too_few_classes": [label for label, count in label_counts.items() if count < 2],
    }
    return report


def collect_quality_issues(records: list[SampleRecord], rows: list[dict], task: str) -> list[str]:
    issues: list[str] = []
    class_names = set(default_option_class_names())
    charset = OPTION_VALUE_CHARSET if task == "option_value" else PRICE_CHARSET if task == "price" else ""
    for record, row in zip(records, rows):
        prefix = str(record.image_path)
        if not record.session_id:
            issues.append(f"{prefix}: missing_session_id")
        if not str(row.get("capture_pair_id", "")).strip():
            issues.append(f"{prefix}: missing_capture_pair_id")
        if record.field_type != task:
            issues.append(f"{prefix}: field_type_mismatch {record.field_type}!={task}")
        if task == "item_metadata":
            metadata_key = str(row.get("metadata_key") or "")
            if metadata_key not in {"req_level", "equipment_category"}:
                issues.append(f"{prefix}: unsupported_metadata_key {metadata_key}")
            if metadata_key == "req_level" and not re.fullmatch(r"\d{1,3}", record.label):
                issues.append(f"{prefix}: invalid_req_level {record.label}")
            if metadata_key == "equipment_category" and not record.label.strip():
                issues.append(f"{prefix}: empty_equipment_category")
        if task == "option_label" and record.label not in class_names:
            issues.append(f"{prefix}: unknown_option_label {record.label}")
        if charset and set(record.label) - set(charset):
            issues.append(f"{prefix}: invalid_chars {record.label}")
        if task == "price" and not re.fullmatch(r"[0-9,]+", record.label):
            issues.append(f"{prefix}: invalid_price_format {record.label}")
        if task == "price" and record.label.replace(",", "").lstrip("0") == "":
            issues.append(f"{prefix}: zero_or_empty_price {record.label}")
        if task == "option_value" and not any(char.isdigit() for char in record.label):
            issues.append(f"{prefix}: option_value_without_digit {record.label}")
        issues.extend(semantic_issues(record, row, prefix, task))
        source_text = str(row.get("source_image_path", "")).strip()
        source = Path(source_text) if source_text else None
        if source is not None and not source.exists():
            issues.append(f"{prefix}: source_image_missing {source}")
        issues.extend(image_issues(record, row, prefix, task))
    return issues


def semantic_issues(record: SampleRecord, row: dict, prefix: str, task: str) -> list[str]:
    issues: list[str] = []
    semantic_status = str(row.get("semantic_validation_status", ""))
    semantic_reason = str(row.get("semantic_validation_reason", ""))
    if task in {"option_label", "option_value"} and record.review_status == "rejected":
        reason = str(row.get("rejection_reason") or semantic_reason or "rejected_sample")
        issues.append(f"{prefix}: rejected_sample {reason}")
    if semantic_status == "failed":
        issues.append(f"{prefix}: semantic_validation_failed {semantic_reason}")
    line_text = str(row.get("line_text") or row.get("parsed_line_text") or row.get("original_line_text") or "")
    if task == "option_label":
        issues.extend(option_label_semantic_issues(record, row, prefix, line_text))
    elif task == "option_value":
        issues.extend(option_value_semantic_issues(record, row, prefix, line_text))
    return issues


def option_label_semantic_issues(record: SampleRecord, row: dict, prefix: str, line_text: str) -> list[str]:
    issues: list[str] = []
    if is_non_option_line(line_text):
        issues.append(f"{prefix}: non_option_line")
    if line_text and not any(char.isdigit() for char in line_text):
        issues.append(f"{prefix}: non_option_line")
    if bool(row.get("contains_colon_like_text")):
        issues.append(f"{prefix}: label_contains_colon")
    if bool(row.get("contains_value_like_text")):
        issues.append(f"{prefix}: label_contains_value")
    parsed_key = canonical_option_key(str(row.get("parsed_option_key") or row.get("option_key") or ""))
    if parsed_key and parsed_key != record.label:
        issues.append(f"{prefix}: semantic_label_mismatch parsed={parsed_key} label={record.label}")
    field_name = str(row.get("field_name") or "")
    field_key = canonical_option_key(FIELD_TO_OPTION_KEY.get(field_name.removesuffix("_label"), field_name.removesuffix("_label")))
    if field_key and not field_name.startswith("potential_") and field_key != record.label and not row.get("line_order_corrected"):
        issues.append(f"{prefix}: trace_field_mismatch field={field_key} label={record.label}")
    if record.label == "magic_attack" and bool(row.get("contains_value_like_text")):
        issues.append(f"{prefix}: magic_attack_label_contains_value")
    return issues


def option_value_semantic_issues(record: SampleRecord, row: dict, prefix: str, line_text: str) -> list[str]:
    issues: list[str] = []
    if is_non_option_line(line_text):
        issues.append(f"{prefix}: non_option_line")
    if bool(row.get("contains_colon_like_text")):
        issues.append(f"{prefix}: option_value_contains_colon")
    if bool(row.get("contains_label_text")):
        issues.append(f"{prefix}: option_value_contains_label_text")
    if bool(row.get("value_sign_without_digit")):
        reason = "option_value_only_sign" if record.label.strip() in {"+", "-"} else "option_value_has_no_digit"
        issues.append(f"{prefix}: {reason} {record.label}")
    parsed_value = normalize_value_text(str(row.get("parsed_value_text") or row.get("value_text") or ""))
    if parsed_value and parsed_value != normalize_value_text(record.label):
        issues.append(f"{prefix}: semantic_label_mismatch parsed_value={parsed_value} label={record.label}")
    crop_width = int(row.get("crop_width") or crop_rect_width(row.get("crop_rect")) or 0)
    label_width = len(record.label.replace(",", ""))
    if crop_width > max(96, label_width * 18 + 30):
        issues.append(f"{prefix}: option_value_crop_not_tight width={crop_width}")
    return issues


def crop_rect_width(rect: object) -> int:
    if not isinstance(rect, dict):
        return 0
    return max(0, int(rect.get("right", 0)) - int(rect.get("left", 0)))


def image_issues(record: SampleRecord, row: dict, prefix: str, task: str) -> list[str]:
    issues: list[str] = []
    if not record.image_path.exists():
        return [f"{prefix}: missing_image"]
    image = cv2.imdecode(np.fromfile(str(record.image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        return [f"{prefix}: unreadable_or_empty_image"]
    if image.shape[0] < 6 or image.shape[1] < 8:
        issues.append(f"{prefix}: crop_too_small {image.shape[:2]}")
    max_width = 384 if task == "price" else 900
    if image.shape[0] > 160 or image.shape[1] > max_width:
        issues.append(f"{prefix}: crop_too_large {image.shape[:2]}")
    residual, gray, mask = split_saved_training_image(image)
    if int(np.max(gray)) == int(np.min(gray)):
        issues.append(f"{prefix}: empty_gray_channel")
    foreground_ratio = float(np.count_nonzero(mask)) / float(mask.size) if mask.size else 0.0
    if foreground_ratio <= 0.001:
        issues.append(f"{prefix}: foreground_ratio_too_low {foreground_ratio:.5f}")
    if foreground_ratio >= 0.85:
        issues.append(f"{prefix}: foreground_ratio_too_high {foreground_ratio:.5f}")
    if np.array_equal(residual, gray):
        issues.append(f"{prefix}: residual_duplicates_gray")
    rect = row.get("crop_rect") or {}
    source_text = str(row.get("source_image_path", "")).strip()
    source = Path(source_text) if source_text else None
    if isinstance(rect, dict) and source is not None and source.exists():
        source_image = cv2.imdecode(np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if source_image is not None:
            left, top, right, bottom = (int(rect.get(key, 0)) for key in ("left", "top", "right", "bottom"))
            if left < 0 or top < 0 or right > source_image.shape[1] or bottom > source_image.shape[0] or right <= left or bottom <= top:
                issues.append(f"{prefix}: crop_rect_out_of_bounds {rect}")
    return issues


def conflicting_labels(rows: list[dict]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        content_hash = str(row.get("content_hash", ""))
        if content_hash:
            grouped[content_hash].add(str(row.get("label", "")))
    return {key: sorted(labels) for key, labels in grouped.items() if len(labels) > 1}


def readiness_levels(total: int, sessions: int, issue_count: int, labels: Counter, task: str) -> dict[str, bool]:
    clean = issue_count == 0
    diverse = len(labels) >= 2 if task == "option_label" else True
    return {
        "pipeline_smoke_ready": total >= 1 and clean,
        "pilot_training_ready": total >= 20 and sessions >= 2 and clean and diverse,
        "recommended_training_ready": total >= 200 and sessions >= 10 and clean and diverse,
    }


def read_rows(metadata: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(metadata.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            rows.append(
                {
                    "image_path": f"<invalid-json-line-{line_number}>",
                    "label": "",
                    "session_id": "",
                    "field_type": "",
                    "label_quality": "",
                    "parse_error": str(exc),
                }
            )
    return rows


def record_from_row(metadata: Path, row: dict) -> SampleRecord:
    image_path = Path(str(row.get("image_path", "")))
    if not image_path.is_absolute():
        image_path = metadata.parent / image_path
    return SampleRecord(
        image_path=image_path,
        label=str(row.get("label", "")),
        session_id=str(row.get("session_id", "")),
        field_type=str(row.get("field_type", "")),
        was_corrected=bool(row.get("was_corrected", False)),
        label_quality=str(row.get("label_quality", "human_confirmed")),
        content_hash=str(row.get("content_hash", "")),
        review_status=str(row.get("review_status", "unreviewed")),
    )


def render_html_report(report: dict) -> str:
    sections = []
    for task, info in report.items():
        issues = "".join(f"<li>{html.escape(issue)}</li>" for issue in info.get("issues", [])[:200])
        sections.append(
            f"<section><h2>{html.escape(task)}</h2>"
            f"<pre>{html.escape(json.dumps({k: v for k, v in info.items() if k != 'issues'}, ensure_ascii=False, indent=2))}</pre>"
            f"<ul>{issues}</ul></section>"
        )
    return "<!doctype html><meta charset='utf-8'><title>Dataset inspection</title><body>" + "\n".join(sections) + "</body>"


def empty_report(task: str) -> dict:
    return {
        "task": task,
        "total": 0,
        "label_quality_counts": {},
        "review_status_counts": {},
        "sessions": 0,
        "label_counts": {},
        "metadata_key_counts": {},
        "length_distribution": {},
        "missing_images": 0,
        "duplicate_hashes": 0,
        "conflicting_labels": {},
        "rejection_reason_counts": {},
        "issues": [],
        "issue_count": 0,
        "readiness": readiness_levels(0, 0, 0, Counter(), task),
        "too_few_classes": [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
