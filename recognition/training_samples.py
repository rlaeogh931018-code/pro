from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from maple_price_tool.config import VisionConfig
from maple_price_tool.domain import AnalysisResult, RecognitionCandidate, RecognitionTrace, Rect
from .preprocessing import stack_line_channels


logger = logging.getLogger(__name__)

CHANNEL_ORDER = ["normalized_residual", "after_grayscale", "foreground_mask"]
TRAINING_LABEL_QUALITIES = {"human_confirmed", "human_confirmed_corrected"}
SCALAR_VALUE_FIELDS = {
    "req_level",
    "str_value",
    "dex_value",
    "int_value",
    "luk_value",
    "attack",
    "magic_attack",
    "upgrade_count",
}


@dataclass(frozen=True)
class SampleSaveSummary:
    option_label_count: int = 0
    option_value_count: int = 0
    price_count: int = 0
    rejected_count: int = 0
    skipped_count: int = 0
    skipped_reasons: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def saved_count(self) -> int:
        return self.option_label_count + self.option_value_count + self.price_count + self.rejected_count


def normalize_training_label(field_name: str, value: Any, field_type: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    if field_type == "price":
        return text.replace(" ", "")
    if field_name == "upgrade_count":
        return text
    if field_type == "option_value" and field_name in SCALAR_VALUE_FIELDS:
        if text.startswith(("+", "-")):
            return text
        return f"+{text}"
    return text


def split_user_lines(text: Any) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def parse_option_line(text: str) -> dict[str, str] | None:
    parts = text.strip().split()
    if len(parts) < 2:
        return None
    value_text = parts[-1]
    if not any(char.isdigit() for char in value_text):
        return None
    option_key = parts[0].lower()
    aliases = {"올스탯": "all_stat", "마력": "magic_attack", "공격력": "attack"}
    return {
        "option_key": aliases.get(option_key, option_key),
        "value_text": value_text,
        "full_text": text.strip(),
    }


def parse_potential_final_line(trace: RecognitionTrace, final_values: dict[str, Any]) -> dict[str, str] | None:
    if not trace.field_name.startswith("potential_"):
        return None
    match = re.match(r"^potential_(\d+)", trace.field_name)
    if match:
        line_index = int(match.group(1)) - 1
    else:
        line_index = trace.line_index if trace.line_index is not None else -1
    lines = split_user_lines(final_values.get("potential", ""))
    if line_index < 0 or line_index >= len(lines):
        return None
    return parse_option_line(lines[line_index])


class TrainingSampleWriter:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self.root = config.training_dataset_dir

    def save_confirmed_samples(self, analysis: AnalysisResult, final_values: dict[str, Any]) -> SampleSaveSummary:
        if not self.config.save_training_samples:
            return SampleSaveSummary()
        errors: list[str] = []
        skipped_reasons: list[str] = []
        counts = {"option_label": 0, "option_value": 0, "price": 0, "rejected": 0, "skipped": 0}
        for trace in self._confirmed_traces(analysis, final_values):
            try:
                saved_type = self._save_trace(analysis, trace, final_values)
                counts[saved_type] += 1
            except SkipSample as exc:
                counts["skipped"] += 1
                skipped_reasons.append(str(exc.reason))
            except Exception as exc:
                logger.exception("failed to save training sample field=%s", trace.field_name)
                errors.append(f"{trace.field_name}: {exc}")
        return SampleSaveSummary(
            option_label_count=counts["option_label"],
            option_value_count=counts["option_value"],
            price_count=counts["price"],
            rejected_count=counts["rejected"],
            skipped_count=counts["skipped"],
            skipped_reasons=tuple(skipped_reasons),
            errors=tuple(errors),
        )

    def _confirmed_traces(self, analysis: AnalysisResult, final_values: dict[str, Any]) -> list[RecognitionTrace]:
        traces = list(analysis.traces)
        potential_traces = [
            trace
            for trace in traces
            if trace.field_name.startswith("potential_") and trace.field_type == "option_value"
        ]
        final_potential = split_user_lines(final_values.get("potential", ""))
        if potential_traces and len(potential_traces) != len(final_potential):
            traces.append(
                RecognitionTrace(
                    field_name="potential",
                    field_type="rejected",
                    raw_prediction="\n".join(trace.raw_prediction or str(trace.selected_prediction or "") for trace in potential_traces),
                    selected_prediction="\n".join(final_potential),
                    selection_reason="manual_mapping_required",
                    needs_review=True,
                    crop_rect=potential_traces[0].crop_rect if potential_traces else None,
                    line_index=potential_traces[0].line_index if potential_traces else None,
                )
            )
        return traces

    def _save_trace(self, analysis: AnalysisResult, trace: RecognitionTrace, final_values: dict[str, Any]) -> str:
        field_type = trace.field_type or infer_field_type(trace.field_name)
        if field_type == "rejected":
            if not self.config.save_rejected_samples:
                raise SkipSample("rejected_disabled")
            label = str(final_values.get(trace.field_name, trace.selected_prediction or "")).strip() or "manual_mapping_required"
            label_quality = "rejected"
            target = "rejected"
        else:
            label = self._confirmed_label(trace, final_values, field_type)
            if label is None:
                raise SkipSample("missing_label")
            was_corrected = str(trace.selected_prediction or "") not in {"", label}
            label_quality = "human_confirmed_corrected" if was_corrected else "human_confirmed"
            target = plural_dir_for_field_type(field_type)
        crop = self._build_training_crop(analysis, trace, field_type)
        content_hash = hashlib.sha256(crop.tobytes()).hexdigest()[:12]
        image_dir = self.root / target / "images"
        metadata_path = self.root / target / "samples.jsonl"
        image_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        filename = sample_filename(analysis.capture_pair_id, trace.line_index, trace.field_name, content_hash)
        image_path = image_dir / filename
        relative_image_path = Path("images") / filename
        metadata = self._metadata(
            analysis,
            trace,
            field_type,
            label,
            label_quality,
            relative_image_path,
            content_hash,
        )
        if metadata_has_hash(metadata_path, content_hash):
            raise SkipSample("duplicate_content_hash")
        write_png(image_path, crop)
        try:
            append_jsonl(metadata_path, metadata)
        except Exception:
            try:
                image_path.unlink(missing_ok=True)
            finally:
                raise
        return "rejected" if target == "rejected" else field_type

    def _confirmed_label(self, trace: RecognitionTrace, final_values: dict[str, Any], field_type: str) -> str | None:
        if field_type == "option_label":
            if trace.field_name.startswith("potential_"):
                parsed = parse_potential_final_line(trace, final_values)
                return parsed.get("option_key") if parsed else None
            return str(trace.selected_prediction or "").strip() or None
        if trace.field_name.startswith("potential_"):
            parsed = parse_potential_final_line(trace, final_values)
            return parsed.get("value_text") if parsed else None
        return normalize_training_label(trace.field_name, final_values.get(trace.field_name), field_type)

    def _metadata(
        self,
        analysis: AnalysisResult,
        trace: RecognitionTrace,
        field_type: str,
        label: str,
        label_quality: str,
        relative_image_path: Path,
        content_hash: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.config.training_sample_schema_version,
            "image_path": str(relative_image_path).replace("\\", "/"),
            "capture_pair_id": analysis.capture_pair_id,
            "session_id": analysis.session_id,
            "field_name": trace.field_name,
            "field_type": field_type,
            "label": label,
            "label_quality": label_quality,
            "raw_prediction": trace.raw_prediction,
            "selected_prediction": trace.selected_prediction,
            "template_candidates": [candidate_to_dict(candidate) for candidate in trace.template_candidates],
            "model_candidates": [candidate_to_dict(candidate) for candidate in trace.model_candidates],
            "selection_reason": "user_confirmed",
            "review_status": "rejected" if label_quality == "rejected" else "unreviewed",
            "rejection_reason": trace.crop_metadata.get("rejection_reason") or (trace.selection_reason if label_quality == "rejected" else ""),
            "original_text": trace.raw_prediction or "",
            "related_traces": [candidate_to_dict(candidate) for candidate in trace.template_candidates + trace.model_candidates]
            if label_quality == "rejected"
            else [],
            "confidence": trace.confidence,
            "needs_review": label_quality == "rejected",
            "was_corrected": label_quality == "human_confirmed_corrected",
            "source_image_path": str(analysis.image_path),
            "before_image_path": str(analysis.before_image_path) if analysis.before_image_path else "",
            "crop_rect": rect_to_dict(trace.crop_rect),
            "channel_order": CHANNEL_ORDER,
            "content_hash": content_hash,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **trace.crop_metadata,
        }

    def _build_training_crop(self, analysis: AnalysisResult, trace: RecognitionTrace, field_type: str) -> np.ndarray:
        image = cv2.imdecode(np.fromfile(str(analysis.image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(str(analysis.image_path))
        rect = trace.crop_rect
        if rect is None:
            rect = Rect(0, 0, image.shape[1], image.shape[0])
        rect = rect.clamp_within(Rect(0, 0, image.shape[1], image.shape[0]))
        if rect.width <= 0 or rect.height <= 0:
            raise ValueError("empty crop rect")
        crop = image[rect.top : rect.bottom, rect.left : rect.right]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        residual_full = read_artifact_image(analysis, ("residual_full", "analysis_binary_full"))
        mask_full = read_artifact_image(analysis, ("foreground_text_mask_full", "final_mask_full"))
        residual = residual_full[rect.top : rect.bottom, rect.left : rect.right]
        mask = mask_full[rect.top : rect.bottom, rect.left : rect.right]
        if field_type == "price" or trace.field_name == "price_meso":
            from maple_price_tool.vision import price_color_mask

            price_mask = price_color_mask(crop)
            if np.count_nonzero(price_mask):
                mask = price_mask
        if residual.shape != gray.shape or mask.shape != gray.shape:
            raise ValueError(
                "artifact crop shape mismatch: "
                f"residual={residual.shape} gray={gray.shape} mask={mask.shape}"
            )
        channels = stack_line_channels(residual, gray, mask)
        return np.clip(channels * 255.0, 0, 255).astype(np.uint8)


class SkipSample(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def infer_field_type(field_name: str) -> str:
    if field_name == "price_meso":
        return "price"
    if field_name.endswith("_label"):
        return "option_label"
    return "option_value"


def plural_dir_for_field_type(field_type: str) -> str:
    return {"option_label": "option_labels", "option_value": "option_values", "price": "prices"}[field_type]


def sample_filename(capture_pair_id: str, line_index: int | None, field_name: str, content_hash: str) -> str:
    safe_pair = capture_pair_id or "unknown_pair"
    line = f"line{line_index:02d}" if line_index is not None else "lineNA"
    safe_field = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in field_name)
    return f"{safe_pair}_{line}_{safe_field}_{content_hash[:6]}.png"


def read_artifact_image(analysis: AnalysisResult, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        path = analysis.analysis_artifacts.get(name)
        if path is None:
            continue
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image
    raise SkipSample(f"missing_analysis_artifact:{'|'.join(names)}")


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"failed to encode png: {path}")
    encoded.tofile(str(path))


def append_jsonl(path: Path, metadata: dict[str, Any]) -> None:
    line = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def metadata_has_hash(path: Path, content_hash: str) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            if json.loads(line).get("content_hash") == content_hash:
                return True
        except json.JSONDecodeError:
            continue
    return False


def rect_to_dict(rect: Rect | None) -> dict[str, int] | None:
    if rect is None:
        return None
    return asdict(rect)


def candidate_to_dict(candidate: RecognitionCandidate) -> dict[str, Any]:
    return {
        "value": candidate.value,
        "label": candidate.label,
        "score": candidate.score,
        "source": candidate.source,
    }
