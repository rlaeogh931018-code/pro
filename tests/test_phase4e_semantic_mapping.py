from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from maple_price_tool.config import VisionConfig
from maple_price_tool.domain import AnalysisResult, FieldResult, RecognitionTrace, Rect
from recognition.dataset import RecognitionJsonlDataset
from recognition.option_classifier import default_option_class_names
from recognition.training_samples import TrainingSampleWriter
from training.inspect_dataset import inspect_task


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def make_analysis(tmp_path: Path, traces: list[RecognitionTrace]) -> AnalysisResult:
    image_path = tmp_path / "after_20260701_120000_000001.png"
    image = np.zeros((90, 180, 3), dtype=np.uint8)
    image[10:70, 20:160] = (120, 180, 240)
    write_png(image_path, image)
    residual_path = tmp_path / "debug" / "residual_full.png"
    mask_path = tmp_path / "debug" / "foreground_text_mask_full.png"
    residual = np.zeros((90, 180), dtype=np.uint8)
    residual[10:70, 20:160] = 77
    mask = np.zeros((90, 180), dtype=np.uint8)
    mask[10:70, 20:160] = 255
    write_png(residual_path, residual)
    write_png(mask_path, mask)
    return AnalysisResult(
        item_key="98 / hat",
        req_level=FieldResult(98, 0.9),
        equipment_type=FieldResult("hat", 0.9),
        price_meso=FieldResult(11111111, 0.9),
        str_value=FieldResult(0, 0.0),
        dex_value=FieldResult(0, 0.0),
        int_value=FieldResult(5, 0.9),
        luk_value=FieldResult(0, 0.0),
        attack=FieldResult(0, 0.0),
        magic_attack=FieldResult(127, 0.9),
        upgrade_count=FieldResult(0, 0.9),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("", 0.0),
        potential=FieldResult("", 0.0),
        image_path=image_path,
        captured_at=datetime.now(),
        capture_pair_id="20260701_120000_000001",
        session_id="20260701",
        analysis_artifacts={"residual_full": residual_path, "foreground_text_mask_full": mask_path},
        traces=traces,
    )


def save_one(tmp_path: Path, trace: RecognitionTrace, values: dict | None = None):
    analysis = make_analysis(tmp_path, [trace])
    final_values = analysis.editable_values()
    final_values.update(values or {})
    return TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        final_values,
    )


def first_row(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def test_attack_speed_crop_saved_as_int_is_rejected(tmp_path):
    trace = RecognitionTrace(
        "int_value_label",
        field_type="option_label",
        line_index=1,
        selected_prediction="int",
        crop_rect=Rect(20, 10, 70, 30),
        confidence=0.9,
        crop_metadata={"line_text": "공격속도 : 빠름", "parsed_option_key": "int"},
    )

    summary = save_one(tmp_path, trace, {"equipment_options": "INT +5"})

    assert summary.rejected_count == 1
    row = first_row(tmp_path / "datasets" / "rejected" / "samples.jsonl")
    assert row["rejection_reason"] == "non_option_line_saved_as_option_label"
    assert not (tmp_path / "datasets" / "option_labels" / "samples.jsonl").exists()


def test_req_and_equipment_category_are_not_saved_as_option_label(tmp_path):
    traces = [
        RecognitionTrace(
            "int_value_label",
            field_type="option_label",
            line_index=1,
            selected_prediction="int",
            crop_rect=Rect(20, 10, 70, 30),
            confidence=0.9,
            crop_metadata={"line_text": "REQ LEV : 98", "parsed_option_key": "int"},
        ),
        RecognitionTrace(
            "int_value_label",
            field_type="option_label",
            line_index=2,
            selected_prediction="int",
            crop_rect=Rect(20, 35, 80, 55),
            confidence=0.9,
            crop_metadata={"line_text": "장비분류 : 모자", "parsed_option_key": "int"},
        ),
    ]
    analysis = make_analysis(tmp_path, traces)
    values = analysis.editable_values()
    values["equipment_options"] = "INT +5"

    summary = TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        values,
    )

    assert summary.rejected_count == 2
    assert not (tmp_path / "datasets" / "option_labels" / "samples.jsonl").exists()


def test_full_magic_attack_line_is_not_saved_as_option_label_but_tight_label_is(tmp_path):
    full_line = RecognitionTrace(
        "magic_attack_label",
        field_type="option_label",
        line_index=3,
        selected_prediction="magic_attack",
        crop_rect=Rect(20, 10, 110, 30),
        confidence=0.9,
        crop_metadata={"line_text": "magic_attack +127", "parsed_option_key": "magic_attack", "contains_value_like_text": True},
    )
    rejected = save_one(tmp_path / "bad", full_line, {"equipment_options": "magic_attack +127"})
    assert rejected.rejected_count == 1

    tight = RecognitionTrace(
        "magic_attack_label",
        field_type="option_label",
        line_index=3,
        selected_prediction="magic_attack",
        crop_rect=Rect(20, 10, 70, 30),
        confidence=0.9,
        crop_metadata={"line_text": "magic_attack +127", "parsed_option_key": "magic_attack", "contains_value_like_text": False},
    )
    saved = save_one(tmp_path / "good", tight, {"equipment_options": "magic_attack +127"})
    assert saved.option_label_count == 1
    row = first_row(tmp_path / "good" / "datasets" / "option_labels" / "samples.jsonl")
    assert row["label"] == "magic_attack"
    assert row["label_quality"] == "pending_review"
    assert row["review_status"] == "unreviewed"


def test_full_option_value_line_is_rejected_but_tight_value_is_saved(tmp_path):
    full_line = RecognitionTrace(
        "int_value",
        field_type="option_value",
        line_index=4,
        selected_prediction="+9%",
        crop_rect=Rect(20, 10, 150, 30),
        confidence=0.9,
        crop_metadata={
            "line_text": "INT : +9%",
            "parsed_option_key": "int",
            "parsed_value_text": "+9%",
            "contains_label_text": True,
        },
    )
    rejected = save_one(tmp_path / "bad", full_line, {"equipment_options": "INT +9%"})
    assert rejected.rejected_count == 1

    tight_value = RecognitionTrace(
        "int_value",
        field_type="option_value",
        line_index=4,
        selected_prediction="+9%",
        crop_rect=Rect(80, 10, 120, 30),
        confidence=0.9,
        crop_metadata={
            "line_text": "INT : +9%",
            "parsed_option_key": "int",
            "parsed_value_text": "+9%",
            "contains_label_text": False,
        },
    )
    saved = save_one(tmp_path / "good", tight_value, {"equipment_options": "INT +9%"})
    assert saved.option_value_count == 1
    row = first_row(tmp_path / "good" / "datasets" / "option_values" / "samples.jsonl")
    assert row["label"] == "+9%"
    assert row["semantic_validation_status"] == "passed"


def test_default_dataset_loader_requires_approved_even_for_human_confirmed(tmp_path):
    root = tmp_path / "datasets" / "option_values"
    write_png(root / "images" / "sample.png", np.ones((16, 32, 3), dtype=np.uint8) * 120)
    metadata = root / "samples.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "image_path": "images/sample.png",
                "label": "+5",
                "session_id": "s1",
                "field_type": "option_value",
                "label_quality": "human_confirmed",
                "review_status": "unreviewed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert len(RecognitionJsonlDataset(metadata, task="option_value")) == 0


def test_approved_pending_review_can_load_and_rejected_is_excluded(tmp_path):
    root = tmp_path / "datasets" / "option_labels"
    write_png(root / "images" / "approved.png", np.ones((16, 32, 3), dtype=np.uint8) * 120)
    write_png(root / "images" / "rejected.png", np.ones((16, 32, 3), dtype=np.uint8) * 120)
    metadata = root / "samples.jsonl"
    rows = [
        {
            "image_path": "images/approved.png",
            "label": "int",
            "session_id": "s1",
            "field_type": "option_label",
            "label_quality": "pending_review",
            "review_status": "approved",
        },
        {
            "image_path": "images/rejected.png",
            "label": "int",
            "session_id": "s1",
            "field_type": "option_label",
            "label_quality": "rejected",
            "review_status": "rejected",
        },
    ]
    metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    dataset = RecognitionJsonlDataset(metadata, task="option_label", class_names=default_option_class_names())

    assert len(dataset) == 1


def test_inspect_dataset_catches_semantic_mapping_issues(tmp_path):
    root = tmp_path / "datasets"
    image_dir = root / "option_values" / "images"
    image = np.ones((18, 140, 3), dtype=np.uint8) * 120
    write_png(image_dir / "full_line.png", image)
    metadata = root / "option_values" / "samples.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "image_path": "images/full_line.png",
                "capture_pair_id": "pair",
                "session_id": "session",
                "field_name": "int_value",
                "field_type": "option_value",
                "label": "+5",
                "label_quality": "human_confirmed",
                "review_status": "unreviewed",
                "line_text": "INT : +9%",
                "parsed_option_key": "int",
                "parsed_value_text": "+9%",
                "contains_label_text": True,
                "crop_rect": {"left": 0, "top": 0, "right": 140, "bottom": 18},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = inspect_task(root, "option_value")

    assert report["issue_count"] > 0
    assert any("option_value_contains_label_text" in issue for issue in report["issues"])
    assert any("semantic_label_mismatch" in issue for issue in report["issues"])
