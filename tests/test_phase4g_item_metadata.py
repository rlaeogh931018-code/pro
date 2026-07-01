from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from maple_price_tool.config import VisionConfig
from maple_price_tool.domain import AnalysisResult, FieldResult, RecognitionTrace, Rect
from maple_price_tool.storage import Storage, final_record_from_analysis
from recognition.training_samples import TrainingSampleWriter
from training.inspect_dataset import inspect_task
from training.preview_dataset import render_preview
from training.review_dataset import main as review_main


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def make_analysis(tmp_path: Path, traces: list[RecognitionTrace]) -> AnalysisResult:
    image_path = tmp_path / "after_20260701_120000_000001.png"
    image = np.zeros((90, 180, 3), dtype=np.uint8)
    image[10:70, 20:150] = (120, 180, 240)
    write_png(image_path, image)
    residual_path = tmp_path / "debug" / "residual_full.png"
    mask_path = tmp_path / "debug" / "foreground_text_mask_full.png"
    residual = np.zeros((90, 180), dtype=np.uint8)
    residual[10:70, 20:150] = 88
    mask = np.zeros((90, 180), dtype=np.uint8)
    mask[10:70, 20:150] = 255
    write_png(residual_path, residual)
    write_png(mask_path, mask)
    return AnalysisResult(
        item_key="98 / 완드",
        req_level=FieldResult(98, 0.9, "REQ LEV : 98"),
        equipment_type=FieldResult("완드", 0.9, "장비분류 : 완드"),
        price_meso=FieldResult(11111111, 0.9),
        str_value=FieldResult(0, 0.0),
        dex_value=FieldResult(0, 0.0),
        int_value=FieldResult(4, 0.9),
        luk_value=FieldResult(0, 0.0),
        attack=FieldResult(72, 0.9),
        magic_attack=FieldResult(122, 0.9),
        upgrade_count=FieldResult(0, 0.9),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("INT +4\n공격력 +72\n마력 +122\n업그레이드 가능 횟수 0", 0.9),
        potential=FieldResult("마력 +9%", 0.9),
        image_path=image_path,
        captured_at=datetime.now(),
        capture_pair_id="20260701_120000_000001",
        session_id="20260701",
        analysis_artifacts={"residual_full": residual_path, "foreground_text_mask_full": mask_path},
        traces=traces,
    )


def req_level_trace() -> RecognitionTrace:
    return RecognitionTrace(
        "req_level",
        field_type="item_metadata",
        selected_prediction="98",
        crop_rect=Rect(72, 10, 110, 28),
        confidence=0.9,
        crop_metadata={
            "metadata_key": "req_level",
            "line_type": "metadata_req_level",
            "raw_line_text": "REQ LEV : 98",
            "line_text": "REQ LEV : 98",
            "raw_line_rect": {"left": 20, "top": 10, "right": 110, "bottom": 28},
            "label_crop_rect": {"left": 20, "top": 10, "right": 70, "bottom": 28},
            "value_crop_rect": {"left": 72, "top": 10, "right": 110, "bottom": 28},
        },
    )


def equipment_category_trace() -> RecognitionTrace:
    return RecognitionTrace(
        "equipment_category",
        field_type="item_metadata",
        selected_prediction="완드",
        crop_rect=Rect(82, 30, 130, 48),
        confidence=0.9,
        crop_metadata={
            "metadata_key": "equipment_category",
            "line_type": "metadata_equipment_category",
            "raw_line_text": "장비분류 : 완드",
            "line_text": "장비분류 : 완드",
            "raw_line_rect": {"left": 20, "top": 30, "right": 130, "bottom": 48},
            "label_crop_rect": {"left": 20, "top": 30, "right": 80, "bottom": 48},
            "value_crop_rect": {"left": 82, "top": 30, "right": 130, "bottom": 48},
        },
    )


def option_traces() -> list[RecognitionTrace]:
    return [
        RecognitionTrace(
            "int_value_label",
            field_type="option_label",
            selected_prediction="int",
            line_index=5,
            crop_rect=Rect(20, 50, 45, 68),
            crop_metadata={"line_text": "INT +4", "parsed_option_key": "int", "line_type": "base_option"},
            confidence=0.9,
        ),
        RecognitionTrace(
            "int_value",
            field_type="option_value",
            selected_prediction="+4",
            line_index=5,
            crop_rect=Rect(70, 50, 95, 68),
            crop_metadata={"line_text": "INT +4", "parsed_option_key": "int", "parsed_value_text": "+4", "line_type": "base_option"},
            confidence=0.9,
        ),
        RecognitionTrace(
            "potential_1_label",
            field_type="option_label",
            selected_prediction="magic_attack",
            line_index=8,
            crop_rect=Rect(20, 50, 55, 68),
            crop_metadata={"line_text": "마력 +9%", "parsed_option_key": "magic_attack", "line_type": "potential_option"},
            confidence=0.9,
        ),
        RecognitionTrace(
            "potential_1",
            field_type="option_value",
            selected_prediction="+9%",
            line_index=8,
            crop_rect=Rect(70, 50, 105, 68),
            crop_metadata={"line_text": "마력 +9%", "parsed_option_key": "magic_attack", "parsed_value_text": "+9%", "line_type": "potential_option"},
            confidence=0.9,
        ),
    ]


def confirmed_values() -> dict[str, object]:
    return {
        "req_level": 98,
        "equipment_type": "완드",
        "price_meso": 11111111,
        "str_value": 0,
        "dex_value": 0,
        "int_value": 4,
        "luk_value": 0,
        "attack": 72,
        "magic_attack": 122,
        "upgrade_count": 0,
        "black_crystal": "",
        "equipment_options": "INT +4",
        "potential": "마력 +9%",
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_req_level_and_equipment_category_are_saved_as_item_metadata(tmp_path):
    analysis = make_analysis(tmp_path, [req_level_trace(), equipment_category_trace()])

    summary = TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        confirmed_values(),
    )

    assert summary.item_metadata_count == 2
    rows = read_jsonl(tmp_path / "datasets" / "item_metadata" / "samples.jsonl")
    assert {row["metadata_key"] for row in rows} == {"req_level", "equipment_category"}
    assert {row["label"] for row in rows} == {"98", "완드"}
    assert all(row["review_status"] == "unreviewed" for row in rows)
    for row in rows:
        assert row["crop_rect"] == row["value_crop_rect"]
        assert row["crop_rect"] != row["raw_line_rect"]


def test_metadata_lines_do_not_enter_option_datasets_and_ignored_lines_are_not_saved(tmp_path):
    ignored = RecognitionTrace(
        "req_int",
        field_type="item_metadata",
        selected_prediction="295",
        crop_rect=Rect(20, 10, 110, 28),
        crop_metadata={"metadata_key": "req_int", "line_type": "ignored", "line_text": "REQ INT : 295"},
    )
    speed = RecognitionTrace(
        "attack_speed",
        field_type="item_metadata",
        selected_prediction="보통",
        crop_rect=Rect(20, 10, 110, 28),
        crop_metadata={"metadata_key": "attack_speed", "line_type": "ignored", "line_text": "공격속도 : 보통"},
    )
    analysis = make_analysis(tmp_path, [req_level_trace(), equipment_category_trace(), ignored, speed])

    summary = TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        confirmed_values(),
    )

    assert summary.item_metadata_count == 2
    assert not (tmp_path / "datasets" / "option_labels" / "samples.jsonl").exists()
    assert not (tmp_path / "datasets" / "option_values" / "samples.jsonl").exists()


def test_base_and_potential_options_still_save_to_option_datasets(tmp_path):
    analysis = make_analysis(tmp_path, [req_level_trace(), equipment_category_trace(), *option_traces()])

    summary = TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        confirmed_values(),
    )

    assert summary.item_metadata_count == 2
    assert summary.option_label_count == 2
    assert summary.option_value_count == 2
    option_rows = read_jsonl(tmp_path / "datasets" / "option_labels" / "samples.jsonl")
    assert {row["line_type"] for row in option_rows} == {"base_option", "potential_option"}
    assert all(row["field_type"] == "option_label" for row in option_rows)


def test_preview_inspect_and_review_support_item_metadata(tmp_path):
    analysis = make_analysis(tmp_path, [req_level_trace(), equipment_category_trace()])
    TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        confirmed_values(),
    )
    metadata = tmp_path / "datasets" / "item_metadata" / "samples.jsonl"

    html = render_preview(metadata, "item_metadata")
    report = inspect_task(tmp_path / "datasets", "item_metadata")

    assert "metadata_key" in html
    assert report["metadata_key_counts"] == {"req_level": 1, "equipment_category": 1}
    assert review_main(["--dataset-dir", str(tmp_path / "datasets"), "--task", "item_metadata", "--index", "0", "--set-status", "approved"]) == 0
    rows = read_jsonl(metadata)
    assert rows[0]["review_status"] == "approved"


def test_storage_adds_equipment_category_without_breaking_existing_db(tmp_path):
    db_path = tmp_path / "records.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE item_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_key TEXT NOT NULL,
                req_level INTEGER NOT NULL,
                equipment_type TEXT NOT NULL,
                price_meso INTEGER NOT NULL,
                int_value INTEGER NOT NULL,
                magic_attack INTEGER NOT NULL,
                upgrade_count INTEGER NOT NULL,
                potential TEXT NOT NULL,
                raw_values_json TEXT NOT NULL,
                confidences_json TEXT NOT NULL,
                image_path TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                saved_at TEXT NOT NULL
            )
            """
        )
    analysis = make_analysis(tmp_path, [])
    storage = Storage(db_path)
    record = final_record_from_analysis(analysis, confirmed_values())

    record_id = storage.save(record)

    assert record_id == 1
    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(item_records)").fetchall()}
        row = conn.execute("SELECT equipment_category FROM item_records WHERE id = ?", (record_id,)).fetchone()
    assert "equipment_category" in columns
    assert row[0] == "완드"
