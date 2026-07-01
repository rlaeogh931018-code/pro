from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from maple_price_tool.domain import AnalysisResult, FieldResult, RecognitionTrace, Rect
from maple_price_tool.ui import format_label_value_preview, label_value_crop_rows, parse_optional_int, parse_required_int


def test_optional_numeric_fields_default_to_zero_when_blank():
    assert parse_optional_int("") == 0
    assert parse_optional_int("  ") == 0
    assert parse_optional_int("None") == 0


def test_numeric_parsing_accepts_commas():
    assert parse_required_int("1,111,111") == 1111111
    assert parse_optional_int("1,234") == 1234


def test_required_numeric_field_rejects_blank():
    with pytest.raises(ValueError):
        parse_required_int("")


def test_label_value_preview_shows_corrected_trace_mapping(tmp_path):
    analysis = AnalysisResult(
        item_key="98 / hat",
        req_level=FieldResult(98, 0.9),
        equipment_type=FieldResult("hat", 0.9),
        price_meso=FieldResult(11111111, 0.9),
        str_value=FieldResult(0, 0.0),
        dex_value=FieldResult(0, 0.0),
        int_value=FieldResult(5, 0.9),
        luk_value=FieldResult(0, 0.0),
        attack=FieldResult(71, 0.8),
        magic_attack=FieldResult(127, 0.9),
        upgrade_count=FieldResult(0, 0.9),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("INT +5\n공격력 +71", 0.8),
        potential=FieldResult("", 0.0),
        image_path=tmp_path / "after.png",
        captured_at=datetime.now(),
        traces=[
            RecognitionTrace(
                "int_value_label",
                field_type="option_label",
                line_index=1,
                selected_prediction="int",
                crop_rect=Rect(0, 0, 10, 10),
                crop_metadata={"line_text": "INT +5", "parsed_option_key": "int", "parsed_value_text": "+5"},
            ),
            RecognitionTrace(
                "int_value",
                field_type="option_value",
                line_index=1,
                selected_prediction="+5",
                crop_rect=Rect(10, 0, 20, 10),
                crop_metadata={"line_text": "INT +5", "parsed_option_key": "int", "parsed_value_text": "+5"},
            ),
            RecognitionTrace(
                "int_value_label",
                field_type="option_label",
                line_index=2,
                selected_prediction="int",
                crop_rect=Rect(0, 10, 10, 20),
                crop_metadata={"line_text": "INT -71", "parsed_option_key": "int", "parsed_value_text": "-71"},
            ),
            RecognitionTrace(
                "int_value",
                field_type="option_value",
                line_index=2,
                selected_prediction="-71",
                crop_rect=Rect(10, 10, 20, 20),
                crop_metadata={"line_text": "INT -71", "parsed_option_key": "int", "parsed_value_text": "-71"},
            ),
        ],
    )

    text = format_label_value_preview(analysis)

    assert "line 1: label=int / value=+5" in text
    assert "line 2: label=attack / value=+71" in text
    assert "corrected from label:int, value:-71" in text
    rows = label_value_crop_rows(analysis)
    assert len(rows) == 2
    assert rows[0]["label_trace"] is not None
    assert rows[0]["value_trace"] is not None
    assert rows[1]["label"] == "attack"
    assert rows[1]["value"] == "+71"
