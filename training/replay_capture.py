from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from maple_price_tool.config import load_config
from maple_price_tool.domain import CaptureResult, Rect
from maple_price_tool.identity import capture_pair_id_from_path, session_id_from_pair_id
from maple_price_tool.storage import Storage, final_record_from_analysis
from maple_price_tool.vision import OpenCvTemplateRecognizer
from recognition.dataset import RecognitionJsonlDataset
from recognition.option_classifier import default_option_class_names
from recognition.training_samples import SampleSaveSummary, TrainingSampleWriter, semantic_validate_trace


TASK_DIRS = {
    "item_metadata": "item_metadata",
    "option_label": "option_labels",
    "option_value": "option_values",
    "price": "prices",
    "rejected": "rejected",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay one before/after capture into a temporary DB and dataset.")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--confirmed-values", type=Path, required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--debug-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--no-save-samples", action="store_true", help="Analyze and report crop quality without writing training samples.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_replay(
        before=args.before,
        after=args.after,
        confirmed_values_path=args.confirmed_values,
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        db_path=args.db_path,
        debug_dir=args.debug_dir,
        save_samples=not args.no_save_samples,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0 if not report["analysis"].get("error") else 1


def run_replay(
    *,
    before: Path,
    after: Path,
    confirmed_values_path: Path,
    config_path: str | Path = "config.yaml",
    dataset_dir: Path | None = None,
    db_path: Path | None = None,
    debug_dir: Path | None = None,
    save_samples: bool = True,
) -> dict[str, Any]:
    temp_root = Path(tempfile.mkdtemp(prefix="maple_replay_"))
    dataset_dir = dataset_dir or temp_root / "datasets"
    db_path = db_path or temp_root / "auction_records.sqlite3"
    debug_dir = debug_dir or temp_root / "debug"

    config = load_config(config_path)
    capture_fixture_input = is_capture_fixture_path(before) or is_capture_fixture_path(after)
    effective_save_samples = bool(save_samples and not capture_fixture_input)
    config.database_path = db_path
    config.vision = replace(
        config.vision,
        save_debug_images=True,
        save_training_samples=effective_save_samples,
        training_dataset_dir=dataset_dir,
    )

    values = json.loads(confirmed_values_path.read_text(encoding="utf-8"))
    capture_pair_id = capture_pair_id_from_path(before)
    session_id = session_id_from_pair_id(capture_pair_id)
    capture = CaptureResult(
        image_path=after,
        capture_rect=Rect(0, 0, 0, 0),
        mouse_x=0,
        mouse_y=0,
        captured_at=datetime.now(),
        before_image_path=before,
        capture_pair_id=capture_pair_id,
        session_id=session_id,
    )

    report: dict[str, Any] = {
        "temp_root": str(temp_root),
        "dataset_dir": str(dataset_dir),
        "db_path": str(db_path),
        "debug_dir": str(debug_dir),
        "analysis": {},
        "db": {},
        "samples": {},
        "reload": {},
        "crop_quality": {},
    }
    try:
        analysis = OpenCvTemplateRecognizer(config.vision, debug_dir=debug_dir).analyze(capture)
        report["analysis"] = {
            "item_key": analysis.item_key,
            "trace_count": len(analysis.traces),
            "artifact_keys": sorted(analysis.analysis_artifacts),
            "debug_images": [str(path) for path in analysis.debug_images],
        }
        report["crop_quality"] = crop_quality_report(analysis)
    except Exception as exc:
        report["analysis"] = {"error": str(exc)}
        return report

    record_id = Storage(db_path).save(final_record_from_analysis(analysis, values))
    report["db"] = {"record_id": record_id, "exists": db_path.exists()}

    if effective_save_samples:
        summary = TrainingSampleWriter(config.vision).save_confirmed_samples(analysis, values)
        report["samples"] = sample_summary_to_dict(summary)
    else:
        reason = "capture_fixture_input" if capture_fixture_input else "disabled"
        report["samples"] = {**sample_summary_to_dict(SampleSaveSummary()), "skipped_reason": reason}

    for task in ("item_metadata", "option_label", "option_value", "price"):
        metadata = dataset_dir / TASK_DIRS[task] / "samples.jsonl"
        if not metadata.exists():
            report["reload"][task] = {"loaded": 0, "exists": False}
            continue
        try:
            class_names = default_option_class_names() if task == "option_label" else None
            dataset = RecognitionJsonlDataset(
                metadata,
                task=task,
                class_names=class_names,
                review_statuses={"approved", "unreviewed"},
            )
            report["reload"][task] = {"loaded": len(dataset), "exists": True}
        except Exception as exc:
            report["reload"][task] = {"error": str(exc), "exists": True}
    return report


def is_capture_fixture_path(path: Path) -> bool:
    return any(part.lower() == "captures" for part in path.resolve().parts)


def crop_quality_report(analysis) -> dict[str, Any]:
    rows = [trace_quality_row(trace) for trace in analysis.traces]
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("validation_status") or row.get("field_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {
        "trace_count": len(rows),
        "validation_counts": counts,
        "rows": rows,
    }


def trace_quality_row(trace) -> dict[str, Any]:
    metadata = trace.crop_metadata or {}
    field_type = trace.field_type or ""
    label = validation_label_for_trace(trace)
    validation_status = ""
    validation_reason = ""
    if field_type in {"item_metadata", "option_label", "option_value", "price"} and label:
        validation = semantic_validate_trace(trace, field_type, label)
        validation_status = "passed" if validation.ok else "failed"
        validation_reason = validation.reason
    elif field_type == "rejected" or metadata.get("rejection_reason"):
        validation_status = "rejected"
        validation_reason = str(metadata.get("rejection_reason") or trace.selection_reason or "rejected")
    elif field_type in {"ignored", "ui_label", "ui_value"} or metadata.get("ui_only"):
        validation_status = "ignored"
        validation_reason = str(metadata.get("ignored_reason") or trace.selection_reason or "ignored")
    return {
        "field_name": trace.field_name,
        "field_type": field_type,
        "line_index": trace.line_index,
        "line_type": metadata.get("line_type", ""),
        "label": label,
        "selected_prediction": trace.selected_prediction,
        "coordinate_system": metadata.get("coordinate_system", "full_image"),
        "validation_status": validation_status or "unvalidated",
        "validation_reason": validation_reason,
        "crop_rect": rect_to_dict(trace.crop_rect),
        "raw_line_rect": metadata.get("raw_line_rect") or metadata.get("price_search_rect") or metadata.get("search_rect"),
        "label_crop_rect": metadata.get("label_crop_rect") or metadata.get("trimmed_label_rect"),
        "value_crop_rect": metadata.get("value_crop_rect") or metadata.get("price_tight_rect") or metadata.get("tight_rect") or metadata.get("raw_value_rect"),
        "line_text": metadata.get("line_text") or metadata.get("raw_line_text") or metadata.get("parsed_line_text") or "",
        "rejection_reason": metadata.get("rejection_reason", ""),
    }


def validation_label_for_trace(trace) -> str:
    metadata = trace.crop_metadata or {}
    field_type = trace.field_type or ""
    if field_type == "item_metadata":
        return str(metadata.get("parsed_value_text") or trace.selected_prediction or "")
    if field_type == "option_label":
        return str(metadata.get("parsed_option_key") or trace.selected_prediction or "")
    if field_type == "option_value":
        return str(metadata.get("parsed_value_text") or trace.selected_prediction or trace.raw_prediction or "")
    if field_type == "price":
        return str(trace.selected_prediction or trace.raw_prediction or "")
    return ""


def sample_summary_to_dict(summary: SampleSaveSummary) -> dict[str, Any]:
    return {
        "item_metadata_count": summary.item_metadata_count,
        "option_label_count": summary.option_label_count,
        "option_value_count": summary.option_value_count,
        "price_count": summary.price_count,
        "rejected_count": summary.rejected_count,
        "skipped_count": summary.skipped_count,
        "skipped_reasons": list(summary.skipped_reasons),
        "errors": list(summary.errors),
        "saved_count": summary.saved_count,
    }


def rect_to_dict(rect) -> dict[str, int] | None:
    if rect is None:
        return None
    return {"left": rect.left, "top": rect.top, "right": rect.right, "bottom": rect.bottom}


if __name__ == "__main__":
    raise SystemExit(main())
