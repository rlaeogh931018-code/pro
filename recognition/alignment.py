from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from maple_price_tool.domain import Rect


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentResult:
    image: np.ndarray
    applied: bool
    dx: float
    dy: float
    response: float
    reason: str = ""


def align_before_to_after(
    before: np.ndarray,
    after: np.ndarray,
    excluded_rect: Rect | None = None,
    max_shift: float = 3.0,
    min_response: float = 0.20,
) -> AlignmentResult:
    """Align before to after using translation only.

    The tooltip region is excluded by replacing it with the same neutral value in
    both correlation inputs. No rotation, scale, or perspective transform is
    attempted.
    """
    if before.shape != after.shape:
        return AlignmentResult(before, False, 0.0, 0.0, 0.0, "shape_mismatch")
    if before.size == 0:
        return AlignmentResult(before, False, 0.0, 0.0, 0.0, "empty_image")

    before_gray = _prepare_correlation_image(before)
    after_gray = _prepare_correlation_image(after)
    if excluded_rect is not None:
        _neutralize_rect(before_gray, after_gray, excluded_rect)

    try:
        (dx, dy), response = cv2.phaseCorrelate(before_gray, after_gray)
    except Exception:
        logger.exception("before/after alignment failed")
        return AlignmentResult(before, False, 0.0, 0.0, 0.0, "phase_correlate_failed")

    dx = float(dx)
    dy = float(dy)
    response = float(response)
    if response < min_response:
        logger.debug("alignment skipped: low response dx=%.3f dy=%.3f response=%.3f", dx, dy, response)
        return AlignmentResult(before, False, dx, dy, response, "low_response")
    if abs(dx) > max_shift or abs(dy) > max_shift:
        logger.debug("alignment skipped: shift too large dx=%.3f dy=%.3f response=%.3f", dx, dy, response)
        return AlignmentResult(before, False, dx, dy, response, "shift_too_large")

    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        before,
        matrix,
        (before.shape[1], before.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("alignment applied dx=%.3f dy=%.3f response=%.3f", dx, dy, response)
    return AlignmentResult(aligned, True, dx, dy, response, "applied")


def _prepare_correlation_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 50, 150)
    source = edges if np.count_nonzero(edges) > 32 else gray
    return source.astype(np.float32)


def _neutralize_rect(before_gray: np.ndarray, after_gray: np.ndarray, rect: Rect) -> None:
    bounds = Rect(0, 0, before_gray.shape[1], before_gray.shape[0])
    clipped = rect.clamp_within(bounds)
    if clipped.width <= 0 or clipped.height <= 0:
        return
    neutral = float(np.median(before_gray))
    before_gray[clipped.top : clipped.bottom, clipped.left : clipped.right] = neutral
    after_gray[clipped.top : clipped.bottom, clipped.left : clipped.right] = neutral
