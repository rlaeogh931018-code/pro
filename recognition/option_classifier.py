from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from maple_price_tool.domain import RecognitionCandidate


try:  # pragma: no cover - optional ML dependency.
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
except Exception:  # pragma: no cover
    torch = None
    nn = None
    MobileNet_V3_Small_Weights = None
    mobilenet_v3_small = None


OPTION_CLASS_NAMES = (
    "str",
    "dex",
    "int",
    "luk",
    "hp",
    "mp",
    "str_value",
    "dex_value",
    "int_value",
    "luk_value",
    "hp_value",
    "mp_value",
    "attack",
    "magic_attack",
    "physical_defense",
    "magic_defense",
    "black_crystal",
    "speed",
    "jump",
    "slip_prevention",
    "upgrade_count",
    "boss_damage",
    "ignore_defense",
    "all_stat",
    "maxhp",
    "maxmp",
    "total_damage",
    "invincible_after_hit",
    "status_duration",
    "usable_hyper_body",
    "usable_haste",
    "usable_sharp_eyes",
    "sealed_ability",
    "sealed_need_item",
    "unknown",
)


@dataclass(frozen=True)
class OptionClassifierOutput:
    candidates: list[RecognitionCandidate]
    top1_score: float
    margin: float
    unknown_score: float
    checkpoint_available: bool
    device: str


def default_option_class_names() -> list[str]:
    return list(OPTION_CLASS_NAMES)


def build_option_classifier(num_classes: int, pretrained: bool = False):
    if torch is None or nn is None or mobilenet_v3_small is None:
        raise RuntimeError("torch and torchvision are required for the option classifier.")
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def predict_option_topk(
    model: Any,
    tensor: "torch.Tensor",
    class_names: list[str],
    top_k: int = 3,
    source: str = "mobilenet",
) -> OptionClassifierOutput:
    if torch is None:
        raise RuntimeError("torch is required for option classifier inference.")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    count = min(top_k, len(class_names))
    scores, indexes = torch.topk(probabilities, k=count)
    candidates = [
        RecognitionCandidate(value=class_names[int(index)], score=float(score), source=source)
        for score, index in zip(scores.detach().cpu(), indexes.detach().cpu())
    ]
    top1_score = candidates[0].score if candidates else 0.0
    second = candidates[1].score if len(candidates) > 1 else 0.0
    unknown_score = float(probabilities[class_names.index("unknown")]) if "unknown" in class_names else 0.0
    return OptionClassifierOutput(
        candidates=candidates,
        top1_score=top1_score,
        margin=top1_score - second,
        unknown_score=unknown_score,
        checkpoint_available=True,
        device=str(next(model.parameters()).device),
    )


def validate_option_checkpoint_metadata(checkpoint: dict[str, Any], expected_classes: list[str]) -> tuple[bool, str]:
    class_names = checkpoint.get("class_names")
    if class_names != expected_classes:
        return False, "class_names mismatch"
    if checkpoint.get("model_type") != "mobilenet_v3_small":
        return False, "model_type mismatch"
    if checkpoint.get("task") != "option_classifier":
        return False, "task mismatch"
    return True, ""


def load_option_classifier_checkpoint(path: Path, device: str, pretrained: bool = False):
    if torch is None:
        raise RuntimeError("torch is required to load option classifier checkpoints.")
    class_names = default_option_class_names()
    checkpoint = torch.load(path, map_location=device)
    valid, reason = validate_option_checkpoint_metadata(checkpoint, class_names)
    if not valid:
        raise ValueError(reason)
    model = build_option_classifier(len(class_names), pretrained=pretrained)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, checkpoint
