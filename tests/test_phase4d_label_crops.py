from __future__ import annotations

import cv2
import numpy as np

from maple_price_tool.domain import Rect
from maple_price_tool.vision import TooltipLine, make_line_training_traces


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


def test_option_value_crop_is_unchanged_from_template_split():
    line = make_line("INT : +5")
    local_rect = Rect(18, 9, 30, 20)
    traces = make_line_training_traces(line, ("int", 0.9, local_rect), "INT : +5", 0.9, 0)

    value_trace = traces[1]

    assert value_trace.field_type == "option_value"
    assert value_trace.crop_rect == Rect(line.rect.left + local_rect.right, line.rect.top, line.rect.right, line.rect.bottom)
