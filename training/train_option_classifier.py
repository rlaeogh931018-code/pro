from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

from maple_price_tool.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Maple option-label MobileNetV3 classifier.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--metadata", default="datasets/option_labels/samples.jsonl")
    parser.add_argument("--output", default="models/option_classifier.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--class-weight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset
    except Exception as exc:
        raise SystemExit(f"ML dependencies are not installed. Install requirements-ml.txt. ({exc})")

    from recognition.dataset import RecognitionJsonlDataset, group_session_ids
    from recognition.option_classifier import build_option_classifier, default_option_class_names

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = load_config(args.config)
    class_names = default_option_class_names()
    metadata = Path(args.metadata)
    if not metadata.exists():
        print(f"No dataset metadata found: {metadata}")
        return 0
    dataset = RecognitionJsonlDataset(metadata, task="option_label", class_names=class_names)
    if len(dataset) == 0:
        print("Dataset is empty; nothing to train.")
        return 0
    train_indexes, val_indexes = split_by_session(dataset.records, seed=args.seed)
    if not train_indexes or not val_indexes:
        print("Need at least two session_id groups for train/validation split.")
        return 0

    device = select_device(config.vision.device, torch)
    model = build_option_classifier(len(class_names), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=build_class_weights(dataset, len(class_names), torch).to(device) if args.class_weight else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(Subset(dataset, train_indexes), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indexes), batch_size=args.batch_size)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device, torch)
        metrics = evaluate_classifier(model, val_loader, device, torch)
        print(f"epoch={epoch} val_accuracy={metrics['accuracy']:.4f} top3={metrics['top3_accuracy']:.4f}")
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            save_checkpoint(args.output, model, optimizer, class_names, epoch, metrics)
    return 0


def split_by_session(records, seed: int) -> tuple[list[int], list[int]]:
    groups = group_session_ids(records)
    sessions = sorted(groups)
    random.Random(seed).shuffle(sessions)
    split = max(1, int(len(sessions) * 0.8))
    train_sessions = set(sessions[:split])
    train = [index for index, record in enumerate(records) if record.session_id in train_sessions]
    val = [index for index, record in enumerate(records) if record.session_id not in train_sessions]
    return train, val


def build_class_weights(dataset, class_count: int, torch):
    counts = torch.ones(class_count)
    for record in dataset.records:
        counts[dataset.class_names.index(record.label)] += 1
    return counts.sum() / counts


def train_one_epoch(model, loader, criterion, optimizer, device: str, torch) -> None:
    model.train()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()


def evaluate_classifier(model, loader, device: str, torch) -> dict[str, float]:
    model.eval()
    total = correct = top3 = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            total += int(labels.numel())
            correct += int((predictions == labels).sum().item())
            top_indexes = torch.topk(logits, k=min(3, logits.shape[1]), dim=1).indices
            top3 += int((top_indexes == labels.unsqueeze(1)).any(dim=1).sum().item())
    return {"accuracy": correct / max(total, 1), "macro_f1": 0.0, "top3_accuracy": top3 / max(total, 1)}


def save_checkpoint(output: str, model, optimizer, class_names: list[str], epoch: int, metrics: dict[str, float]) -> None:
    import torch

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": "mobilenet_v3_small",
            "task": "option_classifier",
            "class_names": class_names,
            "preprocessing_config": {"target_height": 32, "max_width": 256, "channels": 3},
            "model_config": {"pretrained": False},
            "epoch": epoch,
            "validation_metrics": metrics,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        output_path,
    )


def select_device(requested: str, torch) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


if __name__ == "__main__":
    raise SystemExit(main())
