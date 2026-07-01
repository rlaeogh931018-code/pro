import numpy as np
import pytest

torch = pytest.importorskip("torch")

from recognition.preprocessing import OPTION_LABEL_MAX_WIDTH, prepare_line_sample


def test_prepare_line_tensor_shape_range_and_aspect_ratio():
    residual = np.full((16, 40), 255, dtype=np.uint8)
    gray = np.full((16, 40), 128, dtype=np.uint8)
    mask = np.zeros((16, 40), dtype=np.uint8)

    prepared = prepare_line_sample(residual, gray, mask, target_height=32, max_width=OPTION_LABEL_MAX_WIDTH)

    assert prepared.tensor.shape == (3, 32, OPTION_LABEL_MAX_WIDTH)
    assert prepared.tensor.min().item() >= 0.0
    assert prepared.tensor.max().item() <= 1.0
    assert prepared.resized_width == 80
    assert torch.all(prepared.tensor[:, :, 80:] == 0)


def test_prepare_line_tensor_caps_long_width_without_stretching():
    residual = np.full((16, 400), 255, dtype=np.uint8)
    gray = np.full((16, 400), 128, dtype=np.uint8)
    mask = np.zeros((16, 400), dtype=np.uint8)

    prepared = prepare_line_sample(residual, gray, mask, target_height=32, max_width=128)

    assert prepared.tensor.shape == (3, 32, 128)
    assert prepared.resized_width == 128


def test_prepare_line_tensor_requires_matching_shapes():
    with pytest.raises(ValueError):
        prepare_line_sample(np.zeros((10, 10)), np.zeros((10, 11)), np.zeros((10, 10)))
