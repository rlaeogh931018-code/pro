from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


try:  # pragma: no cover - availability depends on optional ML install.
    import torch
except Exception:  # pragma: no cover
    torch = None


OPTION_LABEL_MAX_WIDTH = 256
OPTION_VALUE_MAX_WIDTH = 192
PRICE_MAX_WIDTH = 384
DEFAULT_TARGET_HEIGHT = 32


class MissingTorchError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedLineTensor:
    tensor: "torch.Tensor"
    original_width: int
    original_height: int
    resized_width: int
    target_height: int
    max_width: int


def prepare_line_sample(
    residual_crop: np.ndarray,
    gray_crop: np.ndarray,
    mask_crop: np.ndarray,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    max_width: int = OPTION_LABEL_MAX_WIDTH,
) -> PreparedLineTensor:
    if torch is None:
        raise MissingTorchError("torch is required for ML line preprocessing. Install requirements-ml.txt.")
    stacked = stack_line_channels(residual_crop, gray_crop, mask_crop)
    original_height, original_width = stacked.shape[:2]
    if original_height <= 0 or original_width <= 0:
        raise ValueError("line crop must be non-empty")
    scale = target_height / float(original_height)
    resized_width = max(1, min(max_width, int(round(original_width * scale))))
    resized = cv2.resize(stacked, (resized_width, target_height), interpolation=cv2.INTER_AREA)
    padded = np.zeros((target_height, max_width, 3), dtype=np.float32)
    padded[:, :resized_width, :] = resized
    tensor = torch.from_numpy(padded.transpose(2, 0, 1)).float()
    return PreparedLineTensor(
        tensor=tensor,
        original_width=original_width,
        original_height=original_height,
        resized_width=resized_width,
        target_height=target_height,
        max_width=max_width,
    )


def prepare_line_tensor(
    residual_crop: np.ndarray,
    gray_crop: np.ndarray,
    mask_crop: np.ndarray,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    max_width: int = OPTION_LABEL_MAX_WIDTH,
) -> "torch.Tensor":
    return prepare_line_sample(residual_crop, gray_crop, mask_crop, target_height, max_width).tensor


def stack_line_channels(residual_crop: np.ndarray, gray_crop: np.ndarray, mask_crop: np.ndarray) -> np.ndarray:
    residual = _to_single_channel(residual_crop)
    gray = _to_single_channel(gray_crop)
    mask = _to_single_channel(mask_crop)
    if residual.shape != gray.shape or residual.shape != mask.shape:
        raise ValueError(
            "residual, gray, and mask crops must have identical shape: "
            f"{residual.shape}, {gray.shape}, {mask.shape}"
        )
    return np.stack([_normalize(residual), _normalize(gray), _normalize(mask)], axis=2).astype(np.float32)


def _to_single_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    raise ValueError(f"expected 2D or 3-channel image, got shape {image.shape}")


def _normalize(image: np.ndarray) -> np.ndarray:
    image_float = image.astype(np.float32)
    maximum = float(image_float.max()) if image_float.size else 0.0
    if maximum > 1.0:
        image_float /= 255.0
    return np.clip(image_float, 0.0, 1.0)
