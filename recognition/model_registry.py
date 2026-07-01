from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maple_price_tool.config import VisionConfig


logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional ML dependency.
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class ModelStatus:
    name: str
    available: bool
    reason: str
    device: str = "cpu"
    checkpoint_path: Path | None = None


@dataclass
class LoadedModel:
    model: Any
    metadata: dict[str, Any]
    status: ModelStatus


class ModelRegistry:
    def __init__(self, config: VisionConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._models: dict[str, LoadedModel | None] = {}
        self._statuses: dict[str, ModelStatus] = {}
        self.load_counts: dict[str, int] = {}

    def device(self) -> str:
        requested = self.config.device.lower()
        if requested == "auto":
            if torch is not None and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if requested == "cuda" and (torch is None or not torch.cuda.is_available()):
            return "cpu"
        return requested

    def status(self, name: str) -> ModelStatus:
        if name in self._statuses:
            return self._statuses[name]
        if not self.config.ml_enabled:
            return ModelStatus(name, False, "ml_disabled", self.device())
        path = self._checkpoint_for(name)
        if path is None:
            return ModelStatus(name, False, "unknown_model", self.device())
        if not path.exists():
            return ModelStatus(name, False, "checkpoint_missing", self.device(), path)
        return ModelStatus(name, False, "not_loaded", self.device(), path)

    def get_option_classifier(self) -> LoadedModel | None:
        return self._get_or_load("option_classifier", self._load_option_classifier)

    def get_option_value_crnn(self) -> LoadedModel | None:
        return self._get_or_load("option_value_crnn", lambda path, device: self._load_crnn(path, device, "option_value"))

    def get_price_crnn(self) -> LoadedModel | None:
        return self._get_or_load("price_crnn", lambda path, device: self._load_crnn(path, device, "price"))

    def _get_or_load(self, name: str, loader) -> LoadedModel | None:
        with self._lock:
            if name in self._models:
                return self._models[name]
            if not self.config.ml_enabled:
                self._statuses[name] = ModelStatus(name, False, "ml_disabled", self.device(), self._checkpoint_for(name))
                self._models[name] = None
                return None
            path = self._checkpoint_for(name)
            if path is None or not path.exists():
                self._statuses[name] = ModelStatus(name, False, "checkpoint_missing", self.device(), path)
                self._models[name] = None
                return None
            if torch is None:
                self._statuses[name] = ModelStatus(name, False, "torch_unavailable", "cpu", path)
                self._models[name] = None
                return None
            try:
                loaded = loader(path, self.device())
            except Exception as exc:
                logger.exception("failed to load model %s from %s", name, path)
                self._statuses[name] = ModelStatus(name, False, f"load_failed: {exc}", self.device(), path)
                self._models[name] = None
                return None
            self.load_counts[name] = self.load_counts.get(name, 0) + 1
            self._models[name] = loaded
            self._statuses[name] = loaded.status
            return loaded

    def _checkpoint_for(self, name: str) -> Path | None:
        if name == "option_classifier":
            return self.config.option_classifier_checkpoint
        if name == "option_value_crnn":
            return self.config.option_value_crnn_checkpoint
        if name == "price_crnn":
            return self.config.price_crnn_checkpoint
        return None

    def _load_option_classifier(self, path: Path, device: str) -> LoadedModel:
        from .option_classifier import load_option_classifier_checkpoint

        model, class_names, checkpoint = load_option_classifier_checkpoint(
            path,
            device=device,
            pretrained=self.config.option_classifier_pretrained,
        )
        preprocessing = checkpoint.get("preprocessing_config", {})
        if int(preprocessing.get("target_height", 32)) != 32:
            raise ValueError("preprocessing target_height mismatch")
        metadata = dict(checkpoint)
        metadata["class_names"] = class_names
        return LoadedModel(
            model=model,
            metadata=metadata,
            status=ModelStatus("option_classifier", True, "loaded", device, path),
        )

    def _load_crnn(self, path: Path, device: str, task: str) -> LoadedModel:
        from .crnn import CRNN, option_value_crnn_config, price_crnn_config

        expected_config = option_value_crnn_config() if task == "option_value" else price_crnn_config()
        checkpoint = torch.load(path, map_location=device)
        if checkpoint.get("model_type") != "crnn":
            raise ValueError("model_type mismatch")
        if checkpoint.get("task") != task:
            raise ValueError("task mismatch")
        if checkpoint.get("charset") != expected_config.charset:
            raise ValueError("charset mismatch")
        model_config = checkpoint.get("model_config", {})
        config = expected_config
        if model_config:
            config = type(expected_config)(
                charset=expected_config.charset,
                input_channels=int(model_config.get("input_channels", expected_config.input_channels)),
                hidden_size=int(model_config.get("hidden_size", expected_config.hidden_size)),
                lstm_layers=int(model_config.get("lstm_layers", expected_config.lstm_layers)),
                dropout=float(model_config.get("dropout", expected_config.dropout)),
            )
        model = CRNN(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return LoadedModel(model=model, metadata=dict(checkpoint), status=ModelStatus(f"{task}_crnn", True, "loaded", device, path))
