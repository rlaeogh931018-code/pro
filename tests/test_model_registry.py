from pathlib import Path

from maple_price_tool.config import VisionConfig
from recognition.fusion import build_initial_fusion_trace
from recognition.model_registry import ModelRegistry
from maple_price_tool.domain import RecognitionCandidate


def test_model_registry_checkpoint_missing_fallback(tmp_path):
    config = VisionConfig(template_dir=Path("templates"), option_classifier_checkpoint=tmp_path / "missing.pt")
    registry = ModelRegistry(config)

    assert registry.get_option_classifier() is None
    status = registry.status("option_classifier")
    assert not status.available
    assert status.reason == "checkpoint_missing"


def test_model_registry_ml_disabled_does_not_load(tmp_path):
    config = VisionConfig(
        template_dir=Path("templates"),
        ml_enabled=False,
        option_classifier_checkpoint=tmp_path / "missing.pt",
    )
    registry = ModelRegistry(config)

    assert registry.get_option_classifier() is None
    assert registry.status("option_classifier").reason == "ml_disabled"
    assert registry.load_counts == {}


def test_initial_fusion_conflict_marks_needs_review():
    trace = build_initial_fusion_trace(
        "option_1",
        [RecognitionCandidate(value="magic_attack", score=0.9, source="template")],
        [RecognitionCandidate(value="attack", score=0.9, source="mobilenet")],
        confidence=0.9,
    )

    assert trace.selection_reason == "template_ml_conflict"
    assert trace.needs_review
    assert trace.selected_prediction == "magic_attack"


def test_initial_fusion_agreement_records_reason():
    trace = build_initial_fusion_trace(
        "option_1",
        [RecognitionCandidate(value="magic_attack", score=0.9, source="template")],
        [RecognitionCandidate(value="magic_attack", score=0.8, source="mobilenet")],
        confidence=0.9,
    )

    assert trace.selection_reason == "template_ml_agree"
    assert not trace.needs_review
