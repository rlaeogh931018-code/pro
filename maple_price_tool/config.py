from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CaptureConfig:
    left: int = 500
    right: int = 900
    up: int = 100
    down: int = 250
    output_dir: Path = Path("captures")
    debug_dir: Path = Path("debug")


@dataclass
class VisionConfig:
    template_dir: Path = Path("templates")
    option_threshold: float = 0.82
    digit_threshold: float = 0.86
    easyocr_languages: list[str] = field(default_factory=lambda: ["ko", "en"])
    easyocr_gpu: bool = False
    save_debug_images: bool = True
    alignment_enabled: bool = True
    alignment_max_shift: float = 3.0
    alignment_min_response: float = 0.20
    ml_enabled: bool = True
    device: str = "auto"
    model_dir: Path = Path("models")
    option_classifier_checkpoint: Path = Path("models") / "option_classifier.pt"
    option_classifier_pretrained: bool = False
    option_model_accept_threshold: float = 0.85
    option_model_margin_threshold: float = 0.12
    option_value_crnn_checkpoint: Path = Path("models") / "option_value_crnn.pt"
    price_crnn_checkpoint: Path = Path("models") / "price_crnn.pt"
    ctc_beam_width: int = 10
    ctc_top_k: int = 3
    value_model_accept_threshold: float = 0.80
    value_model_margin_threshold: float = 0.08
    template_fallback_enabled: bool = True
    save_training_samples: bool = True
    training_dataset_dir: Path = Path("datasets")
    save_auto_agreement_samples: bool = False
    save_rejected_samples: bool = True
    training_sample_schema_version: int = 1
    enable_easyocr_fallback: bool = False


@dataclass
class AppConfig:
    window_title_keyword: str = "MapleStory Worlds"
    database_path: Path = Path("ITEMDB") / "auction_records.sqlite3"
    log_path: Path = Path("logs") / "app.log"
    duplicate_window_seconds: int = 180
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)


def _path(value: Any, default: Path) -> Path:
    if value is None:
        return default
    return Path(str(value))


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    capture_data = data.get("capture", {}) or {}
    vision_data = data.get("vision", {}) or {}

    capture = CaptureConfig(
        left=int(capture_data.get("left", CaptureConfig.left)),
        right=int(capture_data.get("right", CaptureConfig.right)),
        up=int(capture_data.get("up", CaptureConfig.up)),
        down=int(capture_data.get("down", CaptureConfig.down)),
        output_dir=_path(capture_data.get("output_dir"), CaptureConfig.output_dir),
        debug_dir=_path(capture_data.get("debug_dir"), CaptureConfig.debug_dir),
    )
    vision = VisionConfig(
        template_dir=_path(vision_data.get("template_dir"), VisionConfig.template_dir),
        option_threshold=float(vision_data.get("option_threshold", VisionConfig.option_threshold)),
        digit_threshold=float(vision_data.get("digit_threshold", VisionConfig.digit_threshold)),
        easyocr_languages=list(vision_data.get("easyocr_languages", ["ko", "en"])),
        easyocr_gpu=bool(vision_data.get("easyocr_gpu", False)),
        save_debug_images=bool(vision_data.get("save_debug_images", VisionConfig.save_debug_images)),
        alignment_enabled=bool(vision_data.get("alignment_enabled", VisionConfig.alignment_enabled)),
        alignment_max_shift=float(vision_data.get("alignment_max_shift", VisionConfig.alignment_max_shift)),
        alignment_min_response=float(vision_data.get("alignment_min_response", VisionConfig.alignment_min_response)),
        ml_enabled=bool(vision_data.get("ml_enabled", VisionConfig.ml_enabled)),
        device=str(vision_data.get("device", VisionConfig.device)),
        model_dir=_path(vision_data.get("model_dir"), VisionConfig.model_dir),
        option_classifier_checkpoint=_path(
            vision_data.get("option_classifier_checkpoint"),
            VisionConfig.option_classifier_checkpoint,
        ),
        option_classifier_pretrained=bool(
            vision_data.get("option_classifier_pretrained", VisionConfig.option_classifier_pretrained)
        ),
        option_model_accept_threshold=float(
            vision_data.get("option_model_accept_threshold", VisionConfig.option_model_accept_threshold)
        ),
        option_model_margin_threshold=float(
            vision_data.get("option_model_margin_threshold", VisionConfig.option_model_margin_threshold)
        ),
        option_value_crnn_checkpoint=_path(
            vision_data.get("option_value_crnn_checkpoint"),
            VisionConfig.option_value_crnn_checkpoint,
        ),
        price_crnn_checkpoint=_path(vision_data.get("price_crnn_checkpoint"), VisionConfig.price_crnn_checkpoint),
        ctc_beam_width=int(vision_data.get("ctc_beam_width", VisionConfig.ctc_beam_width)),
        ctc_top_k=int(vision_data.get("ctc_top_k", VisionConfig.ctc_top_k)),
        value_model_accept_threshold=float(
            vision_data.get("value_model_accept_threshold", VisionConfig.value_model_accept_threshold)
        ),
        value_model_margin_threshold=float(
            vision_data.get("value_model_margin_threshold", VisionConfig.value_model_margin_threshold)
        ),
        template_fallback_enabled=bool(
            vision_data.get("template_fallback_enabled", VisionConfig.template_fallback_enabled)
        ),
        save_training_samples=bool(vision_data.get("save_training_samples", VisionConfig.save_training_samples)),
        training_dataset_dir=_path(vision_data.get("training_dataset_dir"), VisionConfig.training_dataset_dir),
        save_auto_agreement_samples=bool(
            vision_data.get("save_auto_agreement_samples", VisionConfig.save_auto_agreement_samples)
        ),
        save_rejected_samples=bool(vision_data.get("save_rejected_samples", VisionConfig.save_rejected_samples)),
        training_sample_schema_version=int(
            vision_data.get("training_sample_schema_version", VisionConfig.training_sample_schema_version)
        ),
        enable_easyocr_fallback=bool(
            vision_data.get("enable_easyocr_fallback", VisionConfig.enable_easyocr_fallback)
        ),
    )

    return AppConfig(
        window_title_keyword=str(data.get("window_title_keyword", AppConfig.window_title_keyword)),
        database_path=_path(data.get("database_path"), AppConfig.database_path),
        log_path=_path(data.get("log_path"), AppConfig.log_path),
        duplicate_window_seconds=int(data.get("duplicate_window_seconds", AppConfig.duplicate_window_seconds)),
        capture=capture,
        vision=vision,
    )


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "window_title_keyword": config.window_title_keyword,
        "database_path": str(config.database_path),
        "log_path": str(config.log_path),
        "duplicate_window_seconds": config.duplicate_window_seconds,
        "capture": {
            "left": config.capture.left,
            "right": config.capture.right,
            "up": config.capture.up,
            "down": config.capture.down,
            "output_dir": str(config.capture.output_dir),
            "debug_dir": str(config.capture.debug_dir),
        },
        "vision": {
            "template_dir": str(config.vision.template_dir),
            "option_threshold": config.vision.option_threshold,
            "digit_threshold": config.vision.digit_threshold,
            "easyocr_languages": config.vision.easyocr_languages,
            "easyocr_gpu": config.vision.easyocr_gpu,
            "save_debug_images": config.vision.save_debug_images,
            "alignment_enabled": config.vision.alignment_enabled,
            "alignment_max_shift": config.vision.alignment_max_shift,
            "alignment_min_response": config.vision.alignment_min_response,
            "ml_enabled": config.vision.ml_enabled,
            "device": config.vision.device,
            "model_dir": str(config.vision.model_dir),
            "option_classifier_checkpoint": str(config.vision.option_classifier_checkpoint),
            "option_classifier_pretrained": config.vision.option_classifier_pretrained,
            "option_model_accept_threshold": config.vision.option_model_accept_threshold,
            "option_model_margin_threshold": config.vision.option_model_margin_threshold,
            "option_value_crnn_checkpoint": str(config.vision.option_value_crnn_checkpoint),
            "price_crnn_checkpoint": str(config.vision.price_crnn_checkpoint),
            "ctc_beam_width": config.vision.ctc_beam_width,
            "ctc_top_k": config.vision.ctc_top_k,
            "value_model_accept_threshold": config.vision.value_model_accept_threshold,
            "value_model_margin_threshold": config.vision.value_model_margin_threshold,
            "template_fallback_enabled": config.vision.template_fallback_enabled,
            "save_training_samples": config.vision.save_training_samples,
            "training_dataset_dir": str(config.vision.training_dataset_dir),
            "save_auto_agreement_samples": config.vision.save_auto_agreement_samples,
            "save_rejected_samples": config.vision.save_rejected_samples,
            "training_sample_schema_version": config.vision.training_sample_schema_version,
            "enable_easyocr_fallback": config.vision.enable_easyocr_fallback,
        },
    }


def save_config(config: AppConfig, path: str | Path = "config.yaml") -> None:
    config_path = Path(path)
    config_path.write_text(
        yaml.safe_dump(config_to_dict(config), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
