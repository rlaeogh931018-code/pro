import cv2
import numpy as np

from maple_price_tool.domain import Rect
from recognition.alignment import align_before_to_after


def make_alignment_image() -> np.ndarray:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.rectangle(image, (12, 14), (24, 28), (255, 255, 255), thickness=-1)
    cv2.circle(image, (55, 45), 7, (180, 180, 180), thickness=-1)
    cv2.line(image, (8, 70), (72, 62), (120, 120, 120), thickness=2)
    return image


def shift_image(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REPLICATE)


def test_alignment_identity_not_shifted():
    image = make_alignment_image()

    result = align_before_to_after(image, image, max_shift=3.0, min_response=0.20)

    assert result.applied
    assert abs(result.dx) < 0.1
    assert abs(result.dy) < 0.1


def test_alignment_applies_small_translation():
    before = make_alignment_image()
    after = shift_image(before, 2, -1)

    result = align_before_to_after(before, after, excluded_rect=Rect(30, 30, 45, 45), max_shift=3.0, min_response=0.20)

    assert result.applied
    assert abs(result.dx - 2) < 0.35
    assert abs(result.dy + 1) < 0.35


def test_alignment_skips_large_translation():
    before = make_alignment_image()
    after = shift_image(before, 8, 0)

    result = align_before_to_after(before, after, max_shift=3.0, min_response=0.20)

    assert not result.applied
    assert result.reason == "shift_too_large"


def test_alignment_skips_low_response():
    before = np.zeros((80, 80, 3), dtype=np.uint8)
    after = np.zeros((80, 80, 3), dtype=np.uint8)

    result = align_before_to_after(before, after, max_shift=3.0, min_response=0.99)

    assert not result.applied
