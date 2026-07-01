from __future__ import annotations

import cv2
import numpy as np
import pytest

from maple_price_tool.domain import Rect
from maple_price_tool.vision import TooltipLine, make_line_training_traces, option_key_from_line_text, should_create_option_training_traces


def make_line(text: str, origin_x: int = 100, origin_y: int = 100, draw_x: int = 8) -> TooltipLine:
    image = np.zeros((34, 260, 3), dtype=np.uint8)
    cv2.putText(image, text, (draw_x, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)
    return TooltipLine(
        rect=Rect(origin_x, origin_y, origin_x + image.shape[1], origin_y + image.shape[0]),
        image=image,
        match_image=None,
    )


def test_int_label_crop_uses_line_bbox_not_template_rect():
    line = make_line("INT : +5")
    traces = make_line_training_traces(
        line,
        ("int", 0.9, Rect(18, 9, 30, 20)),
        "INT : +5",
        0.9,
        0,
    )

    label_trace, value_trace = traces

    assert label_trace.field_type == "option_label"
    assert label_trace.crop_rect is not None
    assert label_trace.crop_rect.width > 25
    assert value_trace.field_type == "option_value"
    assert label_trace.crop_metadata["touches_right_edge"] is False
    assert label_trace.crop_metadata["contains_value_like_text"] is False


def test_leading_bullet_is_trimmed_and_recorded():
    line = make_line("MAGIC +130", draw_x=24)
    cv2.circle(line.image, (8, 17), 3, (245, 245, 245), -1)
    traces = make_line_training_traces(
        line,
        ("magic_attack", 0.9, Rect(28, 9, 55, 20)),
        "MAGIC +130",
        0.9,
        0,
    )

    label_trace = traces[0]

    assert label_trace.crop_metadata["contains_leading_bullet"] is True
    assert label_trace.crop_rect is not None
    assert label_trace.crop_rect.left > line.rect.left + 8
    assert label_trace.crop_metadata["rejection_reason"] == ""


def test_long_upgrade_label_stays_before_value():
    line = make_line("UPGRADE COUNT 0")
    traces = make_line_training_traces(
        line,
        ("upgrade_count", 0.9, Rect(8, 9, 60, 20)),
        "UPGRADE COUNT 0",
        0.9,
        0,
    )

    label_trace, value_trace = traces

    assert label_trace.crop_rect is not None
    assert label_trace.crop_rect.width > 120
    assert value_trace.field_type == "option_value"
    assert label_trace.crop_metadata["contains_value_like_text"] is False


def test_edge_touching_label_is_rejected():
    line = make_line("INT +5", draw_x=-8)
    traces = make_line_training_traces(
        line,
        ("int", 0.9, Rect(0, 9, 20, 20)),
        "INT +5",
        0.9,
        0,
    )

    label_trace = traces[0]

    assert label_trace.field_type == "rejected"
    assert label_trace.needs_review is True
    assert label_trace.crop_metadata["rejection_reason"] == "label_crop_clipped"


def test_option_value_crop_is_tight_to_value_text():
    line = make_line("INT : +5")
    local_rect = Rect(18, 9, 30, 20)
    traces = make_line_training_traces(line, ("int", 0.9, local_rect), "INT : +5", 0.9, 0)

    value_trace = traces[1]

    assert value_trace.field_type == "option_value"
    assert value_trace.crop_rect is not None
    assert value_trace.crop_rect.left > line.rect.left + local_rect.right
    assert value_trace.crop_rect.width < line.rect.width // 2
    assert value_trace.crop_metadata["parsed_value_text"] == "+5"
    assert value_trace.crop_metadata["contains_label_text"] is False


@pytest.mark.parametrize(
    ("text", "key", "value"),
    [
        ("MAGIC : +127", "magic_attack", "+127"),
        ("ATTACK : +5", "attack", "+5"),
        ("INT : +9%", "int", "+9%"),
        ("UPGRADE COUNT : 0", "upgrade_count", "0"),
    ],
)
def test_colon_split_keeps_label_and_value_separate(text, key, value):
    line = make_line(text)
    traces = make_line_training_traces(line, (key, 0.9, Rect(8, 9, 50, 20)), text, 0.9, 0)

    label_trace, value_trace = traces

    assert label_trace.field_type == "option_label"
    assert label_trace.crop_metadata["split_reason"].startswith("colon_value_split")
    assert label_trace.crop_metadata["contains_colon_like_text"] is False
    assert label_trace.crop_metadata["contains_value_like_text"] is False
    assert value_trace.field_type == "option_value"
    assert value_trace.crop_metadata["parsed_value_text"] == value
    assert value_trace.crop_metadata["contains_label_text"] is False
    assert value_trace.crop_metadata["contains_colon_like_text"] is False
    assert value_trace.crop_rect is not None
    assert label_trace.crop_rect is not None
    assert value_trace.crop_rect.left > label_trace.crop_rect.right


@pytest.mark.parametrize(
    ("text", "key", "value"),
    [
        ("MAGIC : +122", "magic_attack", "+122"),
        ("ATTACK : +72", "attack", "+72"),
        ("INT : +4", "int", "+4"),
        ("UPGRADE COUNT : 0", "upgrade_count", "0"),
        ("MAGIC : +9%", "magic_attack", "+9%"),
    ],
)
def test_phase4h_label_value_crop_is_not_whole_line(text, key, value):
    line = make_line(text)
    label_trace, value_trace = make_line_training_traces(line, (key, 0.9, Rect(8, 9, 50, 20)), text, 0.9, 0)

    assert label_trace.field_type == "option_label"
    assert value_trace.field_type == "option_value"
    assert label_trace.crop_metadata["rejection_reason"] == ""
    assert value_trace.crop_metadata["rejection_reason"] == ""
    assert label_trace.crop_metadata["contains_colon_like_text"] is False
    assert label_trace.crop_metadata["contains_value_like_text"] is False
    assert value_trace.crop_metadata["contains_label_text"] is False
    assert value_trace.crop_metadata["contains_colon_like_text"] is False
    assert value_trace.crop_metadata["parsed_value_text"] == value
    assert label_trace.crop_rect is not None
    assert value_trace.crop_rect is not None
    assert label_trace.crop_rect.right < value_trace.crop_rect.left
    assert value_trace.crop_rect.width < line.rect.width // 3


@pytest.mark.parametrize(
    ("key", "text", "value"),
    [
        ("int", "REQ STR : 0", 0),
        ("int", "REQ INT : 285", 285),
        ("int", "ITEM LEV : 3", 3),
        ("attack_speed", "공격속도", None),
        ("int", "초보자 전사 마법사 궁수 도적 해적", None),
    ],
)
def test_requirement_and_job_lines_do_not_create_option_training_traces(key, text, value):
    assert should_create_option_training_traces(key, text, value) is False


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("INT +9%", "int"),
        ("마력 +9%", "magic_attack"),
        ("공격력 +12", "attack"),
    ],
)
def test_potential_text_maps_to_actual_label_key(text, key):
    assert option_key_from_line_text(text) == key
