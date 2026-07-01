import json

import cv2
import numpy as np
import pytest

from recognition.dataset import RecognitionJsonlDataset
from recognition.option_classifier import default_option_class_names


def write_sample(root, field_type="option_value", label="+130"):
    image_dir = root / "images"
    image_dir.mkdir()
    image_path = image_dir / "sample.png"
    ok, encoded = cv2.imencode(".png", np.zeros((16, 32, 3), dtype=np.uint8))
    assert ok
    encoded.tofile(str(image_path))
    metadata = root / "samples.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "image_path": "images/sample.png",
                "label": label,
                "session_id": "session_1",
                "field_type": field_type,
                "was_corrected": False,
                "review_status": "approved",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def test_dataset_loads_valid_option_value_metadata(tmp_path):
    metadata = write_sample(tmp_path)

    dataset = RecognitionJsonlDataset(metadata, task="option_value")

    assert len(dataset) == 1
    assert dataset.records[0].session_id == "session_1"


def test_dataset_rejects_invalid_label_characters(tmp_path):
    metadata = write_sample(tmp_path, label="ABC")

    with pytest.raises(ValueError):
        RecognitionJsonlDataset(metadata, task="option_value")


def test_dataset_rejects_missing_image(tmp_path):
    metadata = tmp_path / "samples.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "image_path": "images/missing.png",
                "label": "+8",
                "session_id": "session_1",
                "field_type": "option_value",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        RecognitionJsonlDataset(metadata, task="option_value")


def test_dataset_validates_option_label_class(tmp_path):
    metadata = write_sample(tmp_path, field_type="option_label", label="magic_attack")

    dataset = RecognitionJsonlDataset(metadata, task="option_label", class_names=default_option_class_names())

    assert dataset.records[0].label == "magic_attack"
