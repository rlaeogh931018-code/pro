import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

from maple_price_tool.config import VisionConfig
from maple_price_tool.domain import CaptureResult, Rect
from maple_price_tool.vision import (
    OpenCvTemplateRecognizer,
    TooltipLineAnalysis,
    build_diff_foreground_mask,
    normalize_potential_value,
    normalize_potential_value_with_trace,
    prepare_ocr_roi,
)


def test_prepare_ocr_roi_scales_grayscale_image():
    roi = np.zeros((10, 20), dtype=np.uint8)

    prepared = prepare_ocr_roi(roi)

    assert prepared.shape == (30, 60)


def test_diff_foreground_mask_smoke():
    before = np.full((80, 80, 3), 100, dtype=np.uint8)
    after = np.full((80, 80, 3), 100, dtype=np.uint8)
    cv2.rectangle(after, (25, 30), (55, 36), (240, 240, 240), thickness=-1)

    mask, debug_images, stats = build_diff_foreground_mask(before, after, Rect(10, 10, 70, 70))

    assert mask.shape == before.shape[:2]
    assert "residual" in debug_images
    assert "final_mask" in debug_images
    assert stats["threshold"] >= 5.0


def test_build_diff_line_mask_returns_none_for_size_mismatch(tmp_path):
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    cv2.imwrite(str(before_path), np.zeros((20, 20, 3), dtype=np.uint8))
    cv2.imwrite(str(after_path), np.zeros((30, 20, 3), dtype=np.uint8))
    recognizer = OpenCvTemplateRecognizer(VisionConfig(template_dir=Path("templates")))
    capture = CaptureResult(after_path, Rect(0, 0, 0, 0), 0, 0, datetime.now(), before_path)

    result = recognizer.build_diff_line_mask(capture, np.zeros((30, 20, 3), dtype=np.uint8), Rect(0, 0, 20, 20))

    assert result is None


def test_easyocr_fallback_disabled_does_not_initialize_reader():
    recognizer = OpenCvTemplateRecognizer(VisionConfig(template_dir=Path("templates"), enable_easyocr_fallback=False))

    candidates, raw, confidence = recognizer.read_int_candidates_from_roi(np.zeros((8, 12), dtype=np.uint8))

    assert recognizer.ocr_reader is None
    assert candidates == []
    assert raw == "easyocr fallback disabled"
    assert confidence == 0.0


def test_verified_rows_not_used_in_general_layout(monkeypatch, tmp_path):
    recognizer = OpenCvTemplateRecognizer(VisionConfig(template_dir=Path("templates"), enable_easyocr_fallback=False))
    image_path = tmp_path / "after.png"
    image_path.write_bytes(b"fake")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    tooltip = Rect(10, 10, 90, 90)
    capture = CaptureResult(image_path, Rect(0, 0, 0, 0), 0, 0, datetime.now())

    monkeypatch.setattr("maple_price_tool.vision.find_yellow_tooltip_rect", lambda _image: tooltip)
    monkeypatch.setattr(recognizer, "build_diff_line_mask", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(recognizer, "read_req_level_from_tooltip", lambda *_args, **_kwargs: (120, 0.9, "req"))
    monkeypatch.setattr(recognizer, "read_equipment_type_from_tooltip", lambda *_args, **_kwargs: ("완드", 0.9, "type"))
    monkeypatch.setattr(recognizer, "read_maple_price", lambda *_args, **_kwargs: (950_000_000, 0.9, "price"))
    monkeypatch.setattr(
        recognizer,
        "read_tooltip_line_analysis",
        lambda *_args, **_kwargs: TooltipLineAnalysis(
            "INT +3",
            "마력 +6%",
            {"int_value": 3, "upgrade_count": 0},
            {"int_value": 0.8, "upgrade_count": 0.8},
            "raw lines",
        ),
    )
    monkeypatch.setattr(recognizer, "write_layout_debug_image", lambda *_args, **_kwargs: [])

    result = recognizer.analyze_maple_layout(capture, image)

    assert result is not None
    assert result.int_value.value == 3
    assert result.upgrade_count.value == 0
    assert not hasattr(recognizer, "analyze_verified_row")


def test_legacy_potential_correction_is_traceable():
    corrected, reason = normalize_potential_value_with_trace("ignore_defense", 5, True)

    assert corrected == 15
    assert "legacy correction" in reason
    assert normalize_potential_value("ignore_defense", 5, True) == 15


def test_verified_row_values_not_in_runtime_vision_code():
    source = Path("maple_price_tool/vision.py").read_text(encoding="utf-8")

    assert "VERIFIED_ROW_VALUES" not in source
    assert "def analyze_verified_row" not in source
