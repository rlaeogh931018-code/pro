from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from maple_price_tool.domain import AnalysisResult, FieldResult, RecognitionTrace, Rect
from maple_price_tool.ui import (
    build_crop_preview_summary,
    crop_fallback_text,
    crop_preview_group_label,
    format_crop_preview_summary,
    format_label_value_preview,
    format_sample_save_summary,
    label_value_crop_rows,
    parse_optional_int,
    parse_required_int,
)
from recognition.training_samples import SampleSaveSummary


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


def test_crop_rows_include_req_level_and_do_not_mark_template_only_rejected(tmp_path):
    analysis = AnalysisResult(
        item_key="98 / hat",
        req_level=FieldResult(98, 0.9),
        equipment_type=FieldResult("hat", 0.9),
        price_meso=FieldResult(11111111, 0.9),
        str_value=FieldResult(0, 0.0),
        dex_value=FieldResult(0, 0.0),
        int_value=FieldResult(0, 0.0),
        luk_value=FieldResult(0, 0.0),
        attack=FieldResult(0, 0.0),
        magic_attack=FieldResult(0, 0.0),
        upgrade_count=FieldResult(0, 0.0),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("", 0.0),
        potential=FieldResult("INT +9%", 0.8),
        image_path=tmp_path / "after.png",
        captured_at=datetime.now(),
        traces=[
            RecognitionTrace(
                "req_level_label",
                field_type="ui_label",
                selected_prediction="req_level",
                raw_prediction="REQ LEV",
                crop_rect=Rect(0, 0, 30, 10),
                crop_metadata={"ui_only": True, "line_text": "REQ LEV : 98", "parsed_option_key": "req_level"},
            ),
            RecognitionTrace(
                "req_level",
                field_type="ui_value",
                selected_prediction="98",
                crop_rect=Rect(35, 0, 55, 10),
                crop_metadata={"ui_only": True, "line_text": "REQ LEV : 98", "parsed_value_text": "98"},
            ),
            RecognitionTrace(
                "potential_1",
                field_type="option_value",
                line_index=7,
                selected_prediction="+9%",
                selection_reason="template_only",
                crop_rect=Rect(10, 10, 40, 20),
                crop_metadata={"line_text": "INT +9%", "parsed_option_key": "int", "parsed_value_text": "+9%"},
            ),
        ],
    )

    rows = label_value_crop_rows(analysis)
    req_row = next(row for row in rows if row["sort_key"] == "req_level")
    potential_row = next(row for row in rows if row["line_index"] == 7)

    assert req_row["label_trace"] is not None
    assert req_row["value_trace"] is not None
    assert req_row["value"] == "98"
    assert potential_row["status"] == "ok"


def test_phase4h_ui_crop_rows_show_metadata_options_ignored_and_rejected(tmp_path):
    analysis = AnalysisResult(
        item_key="98 / wand",
        req_level=FieldResult(98, 0.9),
        equipment_type=FieldResult("완드", 0.9),
        price_meso=FieldResult(11111111, 0.9),
        str_value=FieldResult(0, 0.0),
        dex_value=FieldResult(0, 0.0),
        int_value=FieldResult(4, 0.9),
        luk_value=FieldResult(0, 0.0),
        attack=FieldResult(72, 0.9),
        magic_attack=FieldResult(122, 0.9),
        upgrade_count=FieldResult(0, 0.9),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("마력 +122", 0.9),
        potential=FieldResult("마력 +9%", 0.9),
        image_path=tmp_path / "after.png",
        captured_at=datetime.now(),
        traces=[
            RecognitionTrace(
                "req_level",
                field_type="item_metadata",
                selected_prediction="98",
                crop_rect=Rect(60, 0, 80, 10),
                crop_metadata={
                    "metadata_key": "req_level",
                    "line_type": "metadata_req_level",
                    "line_text": "REQ LEV : 98",
                    "parsed_value_text": "98",
                    "raw_line_rect": {"left": 0, "top": 0, "right": 80, "bottom": 10},
                    "label_crop_rect": {"left": 0, "top": 0, "right": 50, "bottom": 10},
                    "value_crop_rect": {"left": 60, "top": 0, "right": 80, "bottom": 10},
                },
            ),
            RecognitionTrace(
                "equipment_category",
                field_type="item_metadata",
                selected_prediction="완드",
                crop_rect=Rect(70, 12, 100, 24),
                crop_metadata={
                    "metadata_key": "equipment_category",
                    "line_type": "metadata_equipment_category",
                    "line_text": "장비분류 : 완드",
                    "parsed_value_text": "완드",
                },
            ),
            RecognitionTrace(
                "magic_attack_label",
                field_type="option_label",
                line_index=13,
                selected_prediction="magic_attack",
                crop_rect=Rect(0, 30, 30, 42),
                crop_metadata={
                    "line_type": "base_option",
                    "line_text": "마력 : +122",
                    "parsed_option_key": "magic_attack",
                    "parsed_value_text": "+122",
                    "raw_line_rect": {"left": 0, "top": 30, "right": 90, "bottom": 42},
                    "trimmed_label_rect": {"left": 0, "top": 30, "right": 30, "bottom": 42},
                },
            ),
            RecognitionTrace(
                "magic_attack",
                field_type="option_value",
                line_index=13,
                selected_prediction="+122",
                crop_rect=Rect(50, 30, 90, 42),
                crop_metadata={
                    "line_type": "base_option",
                    "line_text": "마력 : +122",
                    "parsed_option_key": "magic_attack",
                    "parsed_value_text": "+122",
                    "value_crop_rect": {"left": 50, "top": 30, "right": 90, "bottom": 42},
                },
            ),
            RecognitionTrace(
                "attack_speed",
                field_type="ignored",
                line_index=6,
                selected_prediction="보통",
                selection_reason="ignored_attack_speed",
                crop_rect=Rect(0, 45, 80, 55),
                crop_metadata={"line_type": "ignored", "line_text": "공격속도 : 보통", "ignored_reason": "ignored_attack_speed"},
            ),
            RecognitionTrace(
                "bad_value",
                field_type="rejected",
                line_index=14,
                selected_prediction="+122",
                crop_rect=Rect(0, 60, 100, 72),
                crop_metadata={
                    "line_type": "base_option",
                    "line_text": "마력 : +122",
                    "parsed_value_text": "+122",
                    "rejection_reason": "option_value_contains_label_text",
                    "semantic_validation_status": "failed",
                    "semantic_validation_reason": "option_value_contains_label_text",
                },
            ),
        ],
    )

    rows = label_value_crop_rows(analysis)
    summary = build_crop_preview_summary(analysis)

    req_row = next(row for row in rows if row["field_name"] == "req_level")
    category_row = next(row for row in rows if row["field_name"] == "equipment_category")
    magic_row = next(row for row in rows if row["line_index"] == 13)
    ignored_row = next(row for row in rows if row["status"] == "ignored")
    rejected_row = next(row for row in rows if row["status"] == "rejected")

    assert req_row["line_type"] == "metadata_req_level"
    assert req_row["label"] == "req_level"
    assert req_row["value"] == "98"
    assert category_row["line_type"] == "metadata_equipment_category"
    assert category_row["label"] == "equipment_category"
    assert magic_row["label"] == "magic_attack"
    assert magic_row["value"] == "+122"
    assert ignored_row["reason"] == "ignored_attack_speed"
    assert rejected_row["reason"] == "option_value_contains_label_text"
    assert summary["item_metadata"] == 2
    assert summary["option_label"] == 1
    assert summary["option_value"] == 1
    assert summary["ignored"] == 1
    assert summary["rejected"] == 1
    assert "item_metadata=2" in format_crop_preview_summary(summary)


def test_phase4h_ui_missing_crop_and_save_summary_text():
    summary = SampleSaveSummary(item_metadata_count=2, option_label_count=7, option_value_count=7, price_count=1, rejected_count=1)

    assert crop_fallback_text("raw") == "raw line crop: not saved"
    assert crop_fallback_text("value") == "value crop: split failed"
    assert "item_metadata saved=2" in format_sample_save_summary(summary)
    assert "option_label saved=7" in format_sample_save_summary(summary)
    assert "option_value saved=7" in format_sample_save_summary(summary)
    assert "price saved=1" in format_sample_save_summary(summary)


def test_crop_preview_rows_are_grouped_in_review_order(tmp_path):
    analysis = AnalysisResult(
        item_key="98 / wand",
        req_level=FieldResult(98, 0.9),
        equipment_type=FieldResult("wand", 0.9),
        price_meso=FieldResult(11111111, 0.9),
        str_value=FieldResult(0, 0.0),
        dex_value=FieldResult(0, 0.0),
        int_value=FieldResult(4, 0.9),
        luk_value=FieldResult(0, 0.0),
        attack=FieldResult(72, 0.9),
        magic_attack=FieldResult(122, 0.9),
        upgrade_count=FieldResult(0, 0.9),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("INT +4", 0.9),
        potential=FieldResult("INT +9%", 0.9),
        image_path=tmp_path / "after.png",
        captured_at=datetime.now(),
        traces=[
            RecognitionTrace(
                "potential_1",
                field_type="option_value",
                line_index=20,
                selected_prediction="+9%",
                crop_rect=Rect(50, 90, 90, 102),
                crop_metadata={"line_type": "potential_option", "line_text": "INT : +9%", "parsed_value_text": "+9%"},
            ),
            RecognitionTrace(
                "int_value",
                field_type="option_value",
                line_index=13,
                selected_prediction="+4",
                crop_rect=Rect(50, 50, 80, 62),
                crop_metadata={"line_type": "base_option", "line_text": "INT : +4", "parsed_value_text": "+4"},
            ),
            RecognitionTrace(
                "price",
                field_type="price",
                selected_prediction="11111111",
                crop_rect=Rect(100, 10, 150, 24),
                crop_metadata={"line_type": "price", "line_text": "11111111", "parsed_value_text": "11111111"},
            ),
            RecognitionTrace(
                "equipment_category",
                field_type="item_metadata",
                selected_prediction="wand",
                crop_rect=Rect(70, 12, 100, 24),
                crop_metadata={"metadata_key": "equipment_category", "line_type": "metadata_equipment_category"},
            ),
            RecognitionTrace(
                "req_level",
                field_type="item_metadata",
                selected_prediction="98",
                crop_rect=Rect(60, 0, 80, 10),
                crop_metadata={"metadata_key": "req_level", "line_type": "metadata_req_level", "parsed_value_text": "98"},
            ),
        ],
    )

    rows = label_value_crop_rows(analysis)
    ordered_groups = []
    for row in rows:
        group = crop_preview_group_label(row)
        if not ordered_groups or ordered_groups[-1] != group:
            ordered_groups.append(group)

    assert ordered_groups == ["REQ LEV", "장비분류", "가격", "장비옵션", "잠재능력"]
