from maple_price_tool.capture import build_capture_rect, is_point_in_client
from maple_price_tool.config import CaptureConfig
from datetime import datetime

from maple_price_tool.domain import AnalysisResult, FieldResult, GameWindow, Rect, RecognitionCandidate, RecognitionTrace


def test_rect_contains_and_dimensions():
    rect = Rect(10, 20, 30, 50)

    assert rect.width == 20
    assert rect.height == 30
    assert rect.contains(10, 20)
    assert not rect.contains(30, 50)


def test_rect_clamp_within_preserves_size_when_possible():
    rect = Rect(-5, -10, 15, 10)
    outer = Rect(0, 0, 100, 100)

    clamped = rect.clamp_within(outer)

    assert clamped.left == 0
    assert clamped.top == 0
    assert clamped.width == rect.width
    assert clamped.height == rect.height


def test_analysis_result_can_hold_recognition_traces(tmp_path):
    image_path = tmp_path / "capture.png"
    candidate = RecognitionCandidate(value="+9%", label="int", score=0.9, source="template")
    trace = RecognitionTrace(
        field_name="potential_1",
        line_index=0,
        template_candidates=[candidate],
        raw_prediction="+8%",
        selected_prediction="+9%",
        selection_reason="rule candidate rerank",
        confidence=0.8,
        needs_review=True,
    )

    result = AnalysisResult(
        item_key="120 / 완드",
        req_level=FieldResult(120, 0.9),
        equipment_type=FieldResult("완드", 0.9),
        price_meso=FieldResult(1_000, 0.9),
        str_value=FieldResult(None, 0.0, needs_review=True),
        dex_value=FieldResult(0, 0.9),
        int_value=FieldResult(3, 0.9),
        luk_value=FieldResult(0, 0.9),
        attack=FieldResult(None, 0.0, needs_review=True),
        magic_attack=FieldResult(150, 0.9),
        upgrade_count=FieldResult(0, 0.9),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("", 0.0),
        potential=FieldResult("INT +9%", 0.8),
        image_path=image_path,
        captured_at=datetime.now(),
        traces=[trace],
    )

    assert result.traces[0].template_candidates[0].value == "+9%"
    assert result.str_value.value is None
    assert result.upgrade_count.value == 0


def test_build_capture_rect_clamps_to_client_area():
    window = GameWindow(hwnd=1, title="Maple", client_rect=Rect(100, 100, 500, 400))
    config = CaptureConfig(left=500, right=900, up=100, down=250)

    rect = build_capture_rect(window, 120, 130, config)

    assert rect.left >= window.client_rect.left
    assert rect.top >= window.client_rect.top
    assert rect.right <= window.client_rect.right
    assert rect.bottom <= window.client_rect.bottom


def test_is_point_in_client():
    window = GameWindow(hwnd=1, title="Maple", client_rect=Rect(100, 100, 500, 400))

    assert is_point_in_client(window, 250, 250)
    assert not is_point_in_client(window, 80, 250)
