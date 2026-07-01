from __future__ import annotations

from maple_price_tool.domain import RecognitionCandidate, RecognitionTrace


def build_initial_fusion_trace(
    field_name: str,
    template_candidates: list[RecognitionCandidate],
    model_candidates: list[RecognitionCandidate],
    confidence: float,
    line_index: int | None = None,
    model_status_reason: str = "",
) -> RecognitionTrace:
    template_top = template_candidates[0].value if template_candidates else None
    model_top = model_candidates[0].value if model_candidates else None
    if not model_candidates:
        reason = model_status_reason or "model_unavailable"
        needs_review = False
        selected = template_top
    elif template_top == model_top:
        reason = "template_ml_agree"
        needs_review = False
        selected = template_top
    else:
        reason = "template_ml_conflict"
        needs_review = True
        selected = template_top
    return RecognitionTrace(
        field_name=field_name,
        line_index=line_index,
        template_candidates=template_candidates,
        model_candidates=model_candidates,
        raw_prediction=model_top,
        selected_prediction=selected,
        selection_reason=reason,
        confidence=confidence,
        needs_review=needs_review,
    )
