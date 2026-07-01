from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from maple_price_tool.domain import AnalysisResult, FieldResult, RecognitionTrace, Rect
from training.inspect_dataset import inspect_task
from training.preview_dataset import render_preview
from training.replay_capture import run_replay


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def confirmed_values() -> dict:
    return {
        "req_level": 120,
        "equipment_type": "Wand",
        "price_meso": 1234567,
        "str_value": 0,
        "dex_value": 0,
        "int_value": 3,
        "luk_value": 0,
        "attack": 0,
        "magic_attack": 130,
        "upgrade_count": 0,
        "black_crystal": "",
        "equipment_options": "magic_attack +130",
        "potential": "INT +9%",
    }


def make_analysis(tmp_path: Path, image_path: Path | None = None) -> AnalysisResult:
    image_path = image_path or (tmp_path / "after_20260701_120000_000001.png")
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[10:40, 20:90] = (120, 180, 240)
    write_png(image_path, image)
    residual_path = tmp_path / "debug" / "residual_full.png"
    mask_path = tmp_path / "debug" / "foreground_text_mask_full.png"
    residual = np.zeros((80, 120), dtype=np.uint8)
    residual[10:65, 20:100] = 88
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[10:65, 20:100] = 255
    write_png(residual_path, residual)
    write_png(mask_path, mask)
    return AnalysisResult(
        item_key="120 / Wand",
        req_level=FieldResult(120, 0.9),
        equipment_type=FieldResult("Wand", 0.9),
        price_meso=FieldResult(1234567, 0.9),
        str_value=FieldResult(0, 0.9),
        dex_value=FieldResult(0, 0.9),
        int_value=FieldResult(3, 0.9),
        luk_value=FieldResult(0, 0.9),
        attack=FieldResult(0, 0.9),
        magic_attack=FieldResult(130, 0.9),
        upgrade_count=FieldResult(0, 0.9),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("magic_attack +130", 0.9),
        potential=FieldResult("INT +9%", 0.7),
        image_path=image_path,
        captured_at=datetime.now(),
        before_image_path=tmp_path / "before_20260701_120000_000001.png",
        capture_pair_id="20260701_120000_000001",
        session_id="20260701",
        analysis_artifacts={"residual_full": residual_path, "foreground_text_mask_full": mask_path},
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
                    "line_text": "magic_attack +130",
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
                    "line_text": "magic_attack +130",
                    "parsed_option_key": "magic_attack",
                    "parsed_value_text": "+130",
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


def test_replay_capture_uses_temp_paths_and_reloads_dataset(tmp_path, monkeypatch):
    before = tmp_path / "before_20260701_120000_000001.png"
    after = tmp_path / "after_20260701_120000_000001.png"
    write_png(before, np.zeros((80, 120, 3), dtype=np.uint8))
    values_path = tmp_path / "confirmed.json"
    values_path.write_text(json.dumps(confirmed_values()), encoding="utf-8")

    class FakeRecognizer:
        def __init__(self, config, debug_dir=None):
            self.config = config
            self.debug_dir = debug_dir

        def analyze(self, capture):
            return make_analysis(tmp_path, after)

    monkeypatch.setattr("training.replay_capture.OpenCvTemplateRecognizer", FakeRecognizer)

    report = run_replay(before=before, after=after, confirmed_values_path=values_path, config_path="config.yaml")

    assert Path(report["db_path"]).exists()
    assert Path(report["dataset_dir"]).exists()
    assert "ITEMDB" not in report["db_path"]
    assert report["samples"]["option_label_count"] == 1
    assert report["samples"]["option_value_count"] == 1
    assert report["samples"]["price_count"] == 1
    assert report["reload"]["option_value"]["loaded"] == 1


def test_preview_dataset_renders_channels_and_metadata(tmp_path):
    analysis = make_analysis(tmp_path)
    from recognition.training_samples import TrainingSampleWriter
    from maple_price_tool.config import VisionConfig

    TrainingSampleWriter(VisionConfig(training_dataset_dir=tmp_path / "datasets")).save_confirmed_samples(
        analysis,
        confirmed_values(),
    )
    metadata = tmp_path / "datasets" / "option_values" / "samples.jsonl"

    html_text = render_preview(metadata, "option_value")

    assert "Residual" in html_text
    assert "Gray" in html_text
    assert "Mask" in html_text
    assert "magic_attack" in html_text
    assert "data:image/png;base64" in html_text


def test_inspect_dataset_detects_empty_conflicting_and_readiness(tmp_path):
    root = tmp_path / "datasets"
    image_dir = root / "option_values" / "images"
    image_dir.mkdir(parents=True)
    write_png(image_dir / "empty.png", np.zeros((16, 32, 3), dtype=np.uint8))
    rows = [
        {
            "image_path": "images/empty.png",
            "capture_pair_id": "",
            "session_id": "",
            "field_name": "magic_attack",
            "field_type": "option_value",
            "label": "+130",
            "label_quality": "human_confirmed",
            "source_image_path": str(tmp_path / "missing.png"),
            "crop_rect": {"left": 0, "top": 0, "right": 999, "bottom": 999},
            "content_hash": "same",
        },
        {
            "image_path": "images/empty.png",
            "capture_pair_id": "pair",
            "session_id": "session",
            "field_name": "magic_attack",
            "field_type": "option_value",
            "label": "+131",
            "label_quality": "human_confirmed",
            "content_hash": "same",
        },
    ]
    metadata = root / "option_values" / "samples.jsonl"
    metadata.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = inspect_task(root, "option_value")

    assert report["duplicate_hashes"] == 1
    assert report["conflicting_labels"] == {"same": ["+130", "+131"]}
    assert any("empty_gray_channel" in issue for issue in report["issues"])
    assert any("missing_session_id" in issue for issue in report["issues"])
    assert report["readiness"]["pipeline_smoke_ready"] is False


def test_rejected_samples_are_excluded_from_normal_dataset(tmp_path):
    root = tmp_path / "datasets"
    image_dir = root / "option_values" / "images"
    image_dir.mkdir(parents=True)
    write_png(image_dir / "sample.png", np.ones((16, 32, 3), dtype=np.uint8) * 255)
    metadata = root / "option_values" / "samples.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "image_path": "images/sample.png",
                "capture_pair_id": "pair",
                "session_id": "session",
                "field_name": "magic_attack",
                "field_type": "option_value",
                "label": "+130",
                "label_quality": "rejected",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    from recognition.dataset import RecognitionJsonlDataset

    dataset = RecognitionJsonlDataset(metadata, task="option_value")

    assert len(dataset) == 0
