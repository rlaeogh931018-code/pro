from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from maple_price_tool.config import VisionConfig
from maple_price_tool.domain import Rect
from maple_price_tool.vision import OpenCvTemplateRecognizer, price_color_mask
from recognition.dataset import RecognitionJsonlDataset
from training.clean_dataset import find_flagged_rows
from training.review_dataset import main as review_main


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def synthetic_price_image(width: int = 500, second_row: bool = False) -> np.ndarray:
    image = np.zeros((120, width, 3), dtype=np.uint8)
    image[:] = (25, 25, 25)
    cv2.putText(image, "1,299,999,999", (40, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 235, 235), 2, cv2.LINE_AA)
    if second_row:
        cv2.putText(image, "450,000,000", (42, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 235, 235), 2, cv2.LINE_AA)
    return image


def test_price_color_mask_uses_color_and_brightness():
    image = synthetic_price_image()
    mask = price_color_mask(image)

    assert mask.sum() > 0
    assert np.count_nonzero(mask) < mask.size * 0.20


def test_tight_price_bounding_box_detects_one_row():
    recognizer = OpenCvTemplateRecognizer(VisionConfig(save_debug_images=False))
    image = synthetic_price_image()
    result = recognizer.detect_tight_price_crop(
        image,
        Rect(0, 20, 420, 80),
        value=1299999999,
        confidence=0.9,
        raw_digits="1299999999",
        selected_row_index=0,
        selected_row_y=55,
        detection_method="test",
    )

    assert result.tight_rect is not None
    assert result.rejection_reason == ""
    assert result.tight_rect.width < 260
    assert result.component_count >= 8
    assert 0.005 < result.foreground_ratio < 0.75


def test_multiple_price_rows_are_rejected():
    recognizer = OpenCvTemplateRecognizer(VisionConfig(save_debug_images=False))
    image = synthetic_price_image(second_row=True)
    result = recognizer.detect_tight_price_crop(
        image,
        Rect(0, 20, 420, 112),
        value=1299999999,
        confidence=0.9,
        raw_digits="1299999999",
        selected_row_index=0,
        selected_row_y=55,
        detection_method="test",
    )

    assert result.rejection_reason == "multiple_price_rows_detected"
    assert result.needs_review is True


def test_missing_price_text_is_rejected():
    recognizer = OpenCvTemplateRecognizer(VisionConfig(save_debug_images=False))
    image = np.zeros((80, 300, 3), dtype=np.uint8)
    result = recognizer.detect_tight_price_crop(
        image,
        Rect(0, 0, 250, 60),
        value=None,
        confidence=0.0,
        raw_digits="",
        selected_row_index=None,
        selected_row_y=None,
        detection_method="test",
    )

    assert result.rejection_reason == "price_text_not_found"


def test_review_status_flow_and_approved_dataset_filter(tmp_path):
    root = tmp_path / "datasets" / "prices"
    write_png(root / "images" / "sample.png", np.ones((20, 90, 3), dtype=np.uint8) * 120)
    metadata = root / "samples.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "image_path": "images/sample.png",
                "label": "1234567",
                "session_id": "s1",
                "field_type": "price",
                "label_quality": "human_confirmed",
                "review_status": "unreviewed",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert len(RecognitionJsonlDataset(metadata, task="price")) == 0
    assert review_main(["--dataset-dir", str(tmp_path / "datasets"), "--task", "price", "--index", "0", "--set-status", "approved"]) == 0
    assert len(RecognitionJsonlDataset(metadata, task="price")) == 1
    assert metadata.with_suffix(".jsonl.bak").exists()


def test_label_edit_and_clean_dataset_dry_run(tmp_path):
    root = tmp_path / "datasets" / "prices"
    write_png(root / "images" / "sample.png", np.zeros((20, 420, 3), dtype=np.uint8))
    metadata = root / "samples.jsonl"
    row = {
        "image_path": "images/sample.png",
        "label": "ABC",
        "session_id": "s1",
        "field_type": "price",
        "label_quality": "human_confirmed",
        "review_status": "unreviewed",
        "content_hash": "hash1",
    }
    metadata.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert review_main(["--dataset-dir", str(tmp_path / "datasets"), "--task", "price", "--index", "0", "--label", "1299999999"]) == 0
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["label"] == "1299999999"
    assert rows[0]["review_status"] == "relabel_required"

    flagged = find_flagged_rows(metadata, rows, "price")
    assert flagged
    assert "crop_too_large" in flagged[0][2]
