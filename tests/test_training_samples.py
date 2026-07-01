from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from maple_price_tool.config import VisionConfig
from maple_price_tool.domain import AnalysisResult, FieldResult, RecognitionTrace, Rect
from recognition.dataset import RecognitionJsonlDataset, split_saved_training_image
from recognition.preprocessing import DEFAULT_TARGET_HEIGHT, OPTION_VALUE_MAX_WIDTH, prepare_line_sample
from recognition.training_samples import (
    TrainingSampleWriter,
    normalize_training_label,
    parse_equipment_final_line,
    parse_option_line,
    semantic_validate_trace,
)


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def make_analysis(tmp_path: Path) -> AnalysisResult:
    image_path = tmp_path / "after_20260701_233158_413526.png"
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[10:40, 20:90] = (120, 180, 240)
    write_png(image_path, image)
    residual_path = tmp_path / "residual_full.png"
    mask_path = tmp_path / "foreground_text_mask_full.png"
    residual = np.zeros((80, 120), dtype=np.uint8)
    residual[10:40, 20:90] = 77
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[10:40, 20:90] = 255
    write_png(residual_path, residual)
    write_png(mask_path, mask)
    return AnalysisResult(
        item_key="120 / 완드",
        req_level=FieldResult(120, 0.9),
        equipment_type=FieldResult("완드", 0.9),
        price_meso=FieldResult(1234567, 0.9),
        str_value=FieldResult(None, 0.0),
        dex_value=FieldResult(0, 0.9),
        int_value=FieldResult(3, 0.8),
        luk_value=FieldResult(0, 0.9),
        attack=FieldResult(0, 0.9),
        magic_attack=FieldResult(130, 0.61),
        upgrade_count=FieldResult(0, 0.9),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("留덈젰 +130\n?낃렇?덉씠??媛???잛닔: 0", 0.9),
        potential=FieldResult("INT +9%", 0.7),
        image_path=image_path,
        captured_at=datetime.now(),
        capture_pair_id="20260701_233158_413526",
        session_id="20260701",
        analysis_artifacts={
            "residual_full": residual_path,
            "foreground_text_mask_full": mask_path,
        },
        traces=[
            RecognitionTrace(
                "magic_attack_label",
                field_type="option_label",
                line_index=3,
                selected_prediction="magic_attack",
                crop_rect=Rect(20, 10, 55, 30),
                confidence=0.8,
                crop_metadata={
                    "line_type": "base_option",
                    "coordinate_system": "full_image",
                    "line_text": "留덈젰 +130",
                    "parsed_option_key": "magic_attack",
                    "parsed_value_text": "+130",
                },
            ),
            RecognitionTrace(
                "magic_attack",
                field_type="option_value",
                line_index=3,
                selected_prediction="+130",
                crop_rect=Rect(55, 10, 90, 30),
                confidence=0.61,
                crop_metadata={
                    "line_type": "base_option",
                    "coordinate_system": "full_image",
                    "line_text": "留덈젰 +130",
                    "parsed_option_key": "magic_attack",
                    "parsed_value_text": "+130",
                },
            ),
            RecognitionTrace(
                "upgrade_count",
                field_type="option_value",
                line_index=4,
                selected_prediction="0",
                crop_rect=Rect(55, 30, 90, 45),
                confidence=0.9,
                crop_metadata={
                    "line_type": "base_option",
                    "coordinate_system": "full_image",
                    "line_text": "?낃렇?덉씠??媛???잛닔: 0",
                    "parsed_option_key": "upgrade_count",
                    "parsed_value_text": "0",
                },
            ),
            RecognitionTrace(
                "price_meso",
                field_type="price",
                selected_prediction="1234567",
                crop_rect=Rect(20, 45, 100, 65),
                confidence=0.9,
                crop_metadata={
                    "line_type": "price",
                    "coordinate_system": "full_image",
                    "price_tight_rect": {"left": 20, "top": 45, "right": 100, "bottom": 65},
                    "value_crop_rect": {"left": 20, "top": 45, "right": 100, "bottom": 65},
                },
            ),
        ],
    )


def test_normalize_training_label_preserves_signs_and_zero():
    assert normalize_training_label("magic_attack", 130, "option_value") == "+130"
    assert normalize_training_label("upgrade_count", 0, "option_value") == "0"
    assert normalize_training_label("potential_1", "+9%", "option_value") == "+9%"
    assert normalize_training_label("status_duration", "-2", "option_value") == "-2"
    assert normalize_training_label("price_meso", "1,299,999,999", "price") == "1,299,999,999"
    assert normalize_training_label("int_value", None, "option_value") is None


def test_parse_option_line():
    assert parse_option_line("INT +9%") == {"option_key": "int", "value_text": "+9%", "full_text": "INT +9%"}
    assert parse_option_line("잠재능력 인식 필요") is None


def test_parse_korean_equipment_option_lines():
    assert parse_option_line("마력 +130") == {"option_key": "magic_attack", "value_text": "+130", "full_text": "마력 +130"}
    assert parse_option_line("업그레이드 가능 횟수: 0") == {
        "option_key": "upgrade_count",
        "value_text": "0",
        "full_text": "업그레이드 가능 횟수: 0",
    }


def test_equipment_option_text_overrides_hidden_scalar_defaults(tmp_path):
    analysis = make_analysis(tmp_path)
    values = {
        "req_level": 120,
        "equipment_type": "Wand",
        "price_meso": 1234567,
        "str_value": 0,
        "dex_value": 0,
        "int_value": 0,
        "luk_value": 0,
        "attack": 0,
        "magic_attack": 0,
        "upgrade_count": 0,
        "black_crystal": "",
        "equipment_options": "마력 +130\n업그레이드 가능 횟수: 0",
        "potential": "INT +9%",
    }

    assert parse_equipment_final_line(analysis.traces[1], values) == {
        "option_key": "magic_attack",
        "value_text": "+130",
        "full_text": "마력 +130",
    }

    summary = TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        values,
    )

    assert summary.option_value_count == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "datasets" / "option_values" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(row["field_name"] == "magic_attack" and row["label"] == "+130" for row in rows)


def test_training_sample_writer_saves_png_jsonl_and_deduplicates(tmp_path):
    analysis = make_analysis(tmp_path)
    config = VisionConfig(training_dataset_dir=tmp_path / "datasets")
    values = analysis.editable_values()
    values["magic_attack"] = 130

    first = TrainingSampleWriter(config).save_confirmed_samples(analysis, values)
    second = TrainingSampleWriter(config).save_confirmed_samples(analysis, values)

    assert first.option_label_count == 1
    assert first.option_value_count == 2
    assert first.price_count == 1
    assert second.saved_count == 0
    metadata = tmp_path / "datasets" / "option_values" / "samples.jsonl"
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["image_path"].startswith("images/")
    assert rows[0]["capture_pair_id"] == "20260701_233158_413526"
    assert rows[0]["channel_order"] == ["normalized_residual", "after_grayscale", "foreground_mask"]
    assert any(row["label"] == "+130" for row in rows)
    assert any(row["label"] == "0" for row in rows)


def test_price_training_sample_uses_tight_crop_and_user_confirmed_comma_label(tmp_path):
    analysis = make_analysis(tmp_path)
    price_trace = next(trace for trace in analysis.traces if trace.field_name == "price_meso")
    price_trace.crop_rect = Rect(0, 40, 120, 75)
    price_trace.selected_prediction = "23588919"
    price_trace.crop_metadata.update(
        {
            "crop_source": "price_tight_crop",
            "crop_width": 80,
            "crop_height": 20,
            "price_search_rect": {"left": 0, "top": 40, "right": 120, "bottom": 75},
            "price_tight_rect": {"left": 20, "top": 45, "right": 100, "bottom": 65},
            "foreground_ratio": 0.1,
            "component_count": 8,
        }
    )
    values = analysis.editable_values()
    values["price_meso"] = 23588919
    values["price_meso_text"] = "23,588,919"

    summary = TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        values,
    )

    assert summary.price_count == 1
    row = json.loads((tmp_path / "datasets" / "prices" / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["label"] == "23,588,919"
    assert row["crop_source"] == "price_tight_crop"
    assert row["crop_rect"] == {"left": 20, "top": 45, "right": 100, "bottom": 65}
    saved = cv2.imdecode(
        np.fromfile(str(tmp_path / "datasets" / "prices" / row["image_path"]), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    assert saved.shape[:2] == (20, 80)


def test_price_training_sample_rejects_missing_tight_crop_and_bad_label(tmp_path):
    analysis = make_analysis(tmp_path)
    price_trace = next(trace for trace in analysis.traces if trace.field_name == "price_meso")
    price_trace.crop_metadata = {"line_type": "price", "coordinate_system": "full_image"}
    values = analysis.editable_values()
    values["price_meso_text"] = "1,234,567"

    summary = TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        values,
    )

    assert summary.price_count == 0
    assert summary.rejected_count == 1
    row = next(json.loads(line) for line in (tmp_path / "datasets" / "rejected" / "samples.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(line)["field_name"] == "price_meso")
    assert row["rejection_reason"] == "price_tight_crop_missing"

    bad_label_trace = RecognitionTrace(
        "price_meso",
        field_type="price",
        selected_prediction="1234",
        crop_rect=Rect(20, 45, 100, 65),
        crop_metadata={
            "line_type": "price",
            "coordinate_system": "full_image",
            "crop_source": "price_tight_crop",
            "price_tight_rect": {"left": 20, "top": 45, "right": 100, "bottom": 65},
            "crop_width": 80,
            "crop_height": 20,
        },
    )
    assert semantic_validate_trace(bad_label_trace, "price", "12원").reason == "invalid_price_charset"


def test_training_crop_uses_real_residual_channel_not_gray(tmp_path):
    analysis = make_analysis(tmp_path)
    config = VisionConfig(training_dataset_dir=tmp_path / "datasets")

    summary = TrainingSampleWriter(config).save_confirmed_samples(analysis, analysis.editable_values())

    assert summary.errors == ()
    assert summary.option_value_count == 2
    row = next(
        json.loads(line)
        for line in (tmp_path / "datasets" / "option_values" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["field_name"] == "magic_attack"
    )
    image = cv2.imdecode(
        np.fromfile(str(tmp_path / "datasets" / "option_values" / row["image_path"]), dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    residual, gray, mask = split_saved_training_image(image)
    assert np.all(residual == 77)
    assert not np.array_equal(residual, gray)
    assert np.all(mask == 255)


def test_saved_crop_matches_dataset_preprocessing(tmp_path):
    analysis = make_analysis(tmp_path)
    config = VisionConfig(training_dataset_dir=tmp_path / "datasets")
    TrainingSampleWriter(config).save_confirmed_samples(analysis, analysis.editable_values())
    metadata = tmp_path / "datasets" / "option_values" / "samples.jsonl"
    dataset = RecognitionJsonlDataset(metadata, task="option_value", review_statuses={"unreviewed", "approved"})

    image = cv2.imdecode(np.fromfile(str(dataset.records[0].image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    residual, gray, mask = split_saved_training_image(image)
    expected = prepare_line_sample(
        residual,
        gray,
        mask,
        target_height=DEFAULT_TARGET_HEIGHT,
        max_width=OPTION_VALUE_MAX_WIDTH,
    ).tensor

    assert np.allclose(dataset[0]["image"].numpy(), expected.numpy())


def test_save_training_samples_false_skips(tmp_path):
    analysis = make_analysis(tmp_path)
    config = VisionConfig(training_dataset_dir=tmp_path / "datasets", save_training_samples=False)

    summary = TrainingSampleWriter(config).save_confirmed_samples(analysis, analysis.editable_values())

    assert summary.saved_count == 0
    assert not (tmp_path / "datasets").exists()


def test_potential_line_count_mismatch_goes_to_rejected(tmp_path):
    analysis = make_analysis(tmp_path)
    analysis.traces.append(
        RecognitionTrace("potential_1", field_type="option_value", line_index=5, selected_prediction="+9%", crop_rect=Rect(20, 10, 55, 30))
    )
    values = analysis.editable_values()
    values["potential"] = "INT +9%\nLUK +6%"
    config = VisionConfig(training_dataset_dir=tmp_path / "datasets")

    summary = TrainingSampleWriter(config).save_confirmed_samples(analysis, values)

    assert summary.rejected_count == 1
    assert (tmp_path / "datasets" / "rejected" / "samples.jsonl").exists()
    row = json.loads((tmp_path / "datasets" / "rejected" / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["label_quality"] == "rejected"
    assert row["review_status"] == "rejected"
    assert row["rejection_reason"] in {"manual_mapping_required", "trace_field_mismatch"}
