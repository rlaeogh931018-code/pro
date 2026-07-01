from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .ctc_decoder import OPTION_VALUE_CHARSET, PRICE_CHARSET
from .preprocessing import DEFAULT_TARGET_HEIGHT, OPTION_LABEL_MAX_WIDTH, OPTION_VALUE_MAX_WIDTH, PRICE_MAX_WIDTH, prepare_line_sample


@dataclass(frozen=True)
class SampleRecord:
    image_path: Path
    label: str
    session_id: str
    field_type: str
    was_corrected: bool = False
    label_quality: str = "human_confirmed"
    content_hash: str = ""
    review_status: str = "unreviewed"


class RecognitionJsonlDataset:
    def __init__(
        self,
        metadata_path: Path,
        task: str,
        class_names: list[str] | None = None,
        charset: str | None = None,
        label_qualities: set[str] | None = None,
        review_statuses: set[str] | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.root = metadata_path.parent
        self.task = task
        self.class_names = class_names
        self.charset = charset or _charset_for_task(task)
        self.max_width = _max_width_for_task(task)
        self.label_qualities = label_qualities or {"human_confirmed", "human_confirmed_corrected"}
        self.review_statuses = review_statuses or {"approved"}
        self.records = [
            record
            for record in load_records(metadata_path)
            if record.label_quality in self.label_qualities and record.review_status in self.review_statuses
        ]
        for record in self.records:
            validate_record(record, task, class_names, self.charset)
        duplicates = duplicate_hashes(self.records)
        if duplicates:
            raise ValueError(f"duplicate content_hash values: {sorted(duplicates)}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = cv2.imdecode(np.fromfile(str(record.image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(str(record.image_path))
        residual, gray, mask = split_saved_training_image(image)
        prepared = prepare_line_sample(residual, gray, mask, DEFAULT_TARGET_HEIGHT, self.max_width)
        if self.task == "option_label":
            assert self.class_names is not None
            label = self.class_names.index(record.label)
        else:
            label = record.label
        return {
            "image": prepared.tensor,
            "label": label,
            "text": record.label,
            "session_id": record.session_id,
            "was_corrected": record.was_corrected,
        }


def load_records(metadata_path: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        data = json.loads(line)
        image_path = Path(str(data["image_path"]))
        if not image_path.is_absolute():
            image_path = metadata_path.parent / image_path
        record = SampleRecord(
            image_path=image_path,
            label=str(data["label"]),
            session_id=str(data["session_id"]),
            field_type=str(data["field_type"]),
            was_corrected=bool(data.get("was_corrected", False)),
            label_quality=str(data.get("label_quality", "human_confirmed")),
            content_hash=str(data.get("content_hash", "")),
            review_status=str(data.get("review_status", "unreviewed")),
        )
        if not record.image_path.exists():
            raise FileNotFoundError(f"line {line_number}: missing image {record.image_path}")
        records.append(record)
    return records


def read_jsonl_objects(metadata_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line_number, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{metadata_path}:{line_number}: invalid JSONL: {exc}") from exc
    return rows


def duplicate_hashes(records: list[SampleRecord]) -> set[str]:
    counts = Counter(record.content_hash for record in records if record.content_hash)
    return {content_hash for content_hash, count in counts.items() if count > 1}


def validate_record(record: SampleRecord, task: str, class_names: list[str] | None, charset: str) -> None:
    if record.field_type != task:
        raise ValueError(f"field_type {record.field_type!r} does not match task {task!r}")
    if task == "option_label":
        if class_names is None:
            raise ValueError("class_names are required for option_label datasets")
        if record.label not in class_names:
            raise ValueError(f"unknown option label {record.label!r}")
        return
    invalid = sorted(set(record.label) - set(charset))
    if invalid:
        raise ValueError(f"label {record.label!r} contains invalid characters: {invalid}")


def split_saved_training_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if image.ndim == 2:
        return image, image, image
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, 0], image[:, :, 1], image[:, :, 2]
    raise ValueError(f"unsupported training image shape {image.shape}")


def group_session_ids(records: Iterable[SampleRecord]) -> dict[str, list[SampleRecord]]:
    groups: dict[str, list[SampleRecord]] = {}
    for record in records:
        groups.setdefault(record.session_id, []).append(record)
    return groups


def _charset_for_task(task: str) -> str:
    if task == "option_value":
        return OPTION_VALUE_CHARSET
    if task == "price":
        return PRICE_CHARSET
    return ""


def _max_width_for_task(task: str) -> int:
    if task == "option_label":
        return OPTION_LABEL_MAX_WIDTH
    if task == "price":
        return PRICE_MAX_WIDTH
    return OPTION_VALUE_MAX_WIDTH
