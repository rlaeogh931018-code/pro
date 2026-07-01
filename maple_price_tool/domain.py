from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class AppState(str, Enum):
    IDLE = "IDLE"
    CAPTURING = "CAPTURING"
    ANALYZING = "ANALYZING"
    REVIEWING = "REVIEWING"
    EDITING = "EDITING"
    SAVING = "SAVING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def clamp_within(self, outer: "Rect") -> "Rect":
        width = self.width
        height = self.height
        left = min(max(self.left, outer.left), max(outer.left, outer.right - width))
        top = min(max(self.top, outer.top), max(outer.top, outer.bottom - height))
        right = min(left + width, outer.right)
        bottom = min(top + height, outer.bottom)
        return Rect(left, top, right, bottom)


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    title: str
    client_rect: Rect


@dataclass(frozen=True)
class CaptureResult:
    image_path: Path
    capture_rect: Rect
    mouse_x: int
    mouse_y: int
    captured_at: datetime
    before_image_path: Path | None = None
    capture_pair_id: str = ""
    session_id: str = ""


@dataclass
class FieldResult:
    value: Any
    confidence: float
    raw_value: Any = None
    method: str = ""
    needs_review: bool = False
    candidates: list["RecognitionCandidate"] = field(default_factory=list)
    raw_prediction: Any = None
    corrected_prediction: Any = None
    correction_reason: str = ""


@dataclass
class RecognitionCandidate:
    value: Any = None
    label: str = ""
    score: float = 0.0
    source: str = ""


@dataclass
class RecognitionTrace:
    field_name: str
    field_type: str = ""
    line_index: int | None = None
    template_candidates: list[RecognitionCandidate] = field(default_factory=list)
    model_candidates: list[RecognitionCandidate] = field(default_factory=list)
    raw_prediction: Any = None
    selected_prediction: Any = None
    selection_reason: str = ""
    confidence: float = 0.0
    needs_review: bool = False
    label_crop_path: Path | None = None
    value_crop_path: Path | None = None
    crop_rect: Rect | None = None
    user_corrected_value: Any = None
    crop_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    item_key: str
    req_level: FieldResult
    equipment_type: FieldResult
    price_meso: FieldResult
    str_value: FieldResult
    dex_value: FieldResult
    int_value: FieldResult
    luk_value: FieldResult
    attack: FieldResult
    magic_attack: FieldResult
    upgrade_count: FieldResult
    black_crystal: FieldResult
    equipment_options: FieldResult
    potential: FieldResult
    image_path: Path
    captured_at: datetime
    debug_images: list[Path] = field(default_factory=list)
    traces: list[RecognitionTrace] = field(default_factory=list)
    analysis_artifacts: dict[str, Path] = field(default_factory=dict)
    before_image_path: Path | None = None
    capture_pair_id: str = ""
    session_id: str = ""

    def editable_values(self) -> dict[str, Any]:
        return {
            "item_key": self.item_key,
            "req_level": self.req_level.value,
            "equipment_type": self.equipment_type.value,
            "price_meso": self.price_meso.value,
            "str_value": self.str_value.value,
            "dex_value": self.dex_value.value,
            "int_value": self.int_value.value,
            "luk_value": self.luk_value.value,
            "attack": self.attack.value,
            "magic_attack": self.magic_attack.value,
            "upgrade_count": self.upgrade_count.value,
            "black_crystal": self.black_crystal.value,
            "equipment_options": self.equipment_options.value,
            "potential": self.potential.value,
        }


@dataclass
class FinalItemRecord:
    item_key: str
    req_level: int
    equipment_type: str
    price_meso: int
    str_value: int
    dex_value: int
    int_value: int
    luk_value: int
    attack: int
    magic_attack: int
    upgrade_count: int
    black_crystal: str
    equipment_options: str
    potential: str
    raw_values: dict[str, Any]
    confidences: dict[str, float]
    image_path: Path
    captured_at: datetime
    saved_at: datetime
    capture_pair_id: str = ""
    session_id: str = ""
