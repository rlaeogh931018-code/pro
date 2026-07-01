from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

from maple_price_tool.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Maple CRNN value or price recognizers.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task", choices=["option_value", "price"], required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
        from torch.utils.data import DataLoader, Subset
    except Exception as exc:
        raise SystemExit(f"ML dependencies are not installed. Install requirements-ml.txt. ({exc})")

    from recognition.crnn import CRNN, option_value_crnn_config, price_crnn_config
    from recognition.ctc_decoder import CTCCodec, prefix_beam_search
    from recognition.dataset import RecognitionJsonlDataset
    from training.train_option_classifier import select_device, split_by_session

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    app_config = load_config(args.config)
    model_config = option_value_crnn_config() if args.task == "option_value" else price_crnn_config()
    codec = CTCCodec(model_config.charset)
    metadata = Path(args.metadata or default_metadata(args.task))
    output = Path(args.output or default_output(args.task))
    if not metadata.exists():
        print(f"No dataset metadata found: {metadata}")
        return 0
    dataset = RecognitionJsonlDataset(metadata, task=args.task, charset=model_config.charset)
    if len(dataset) == 0:
        print("Dataset is empty; nothing to train.")
        return 0
    train_indexes, val_indexes = split_by_session(dataset.records, seed=args.seed)
    if not train_indexes or not val_indexes:
        print("Need at least two session_id groups for train/validation split.")
        return 0
    device = select_device(app_config.vision.device, torch)
    model = CRNN(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_loader = DataLoader(Subset(dataset, train_indexes), batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_crnn(b, codec, torch))
    val_loader = DataLoader(Subset(dataset, val_indexes), batch_size=args.batch_size, collate_fn=lambda b: collate_crnn(b, codec, torch))
    best_exact = -1.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, device, torch)
        metrics = evaluate_crnn(model, val_loader, codec, device, torch, app_config.vision.ctc_beam_width)
        print(f"epoch={epoch} exact={metrics['exact_accuracy']:.4f} cer={metrics['cer']:.4f}")
        if metrics["exact_accuracy"] > best_exact:
            best_exact = metrics["exact_accuracy"]
            save_checkpoint(output, model, optimizer, model_config, args.task, epoch, metrics, torch)
    return 0


def collate_crnn(batch, codec, torch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    encoded = [torch.tensor(codec.encode(text), dtype=torch.long) for text in texts]
    targets = torch.cat(encoded) if encoded else torch.empty(0, dtype=torch.long)
    target_lengths = torch.tensor([len(item) for item in encoded], dtype=torch.long)
    input_lengths = torch.full((len(batch),), images.shape[-1] // 4, dtype=torch.long)
    return {"image": images, "targets": targets, "target_lengths": target_lengths, "input_lengths": input_lengths, "texts": texts}


def train_one_epoch(model, loader, optimizer, device: str, torch) -> None:
    model.train()
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        loss = model.ctc_loss(log_probs, batch["targets"].to(device), batch["input_lengths"].to(device), batch["target_lengths"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate_crnn(model, loader, codec, device: str, torch, beam_width: int) -> dict[str, float]:
    model.eval()
    total = exact = distance = chars = topk_exact = 0
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probs = torch.softmax(logits, dim=2).detach().cpu().numpy()
            for index, expected in enumerate(batch["texts"]):
                candidates = prefix_beam_search(probs[:, index, :], codec, beam_width=beam_width, top_k=3)
                predicted = candidates[0].text if candidates else ""
                total += 1
                exact += int(predicted == expected)
                topk_exact += int(any(candidate.text == expected for candidate in candidates))
                distance += levenshtein(predicted, expected)
                chars += len(expected)
    return {"exact_accuracy": exact / max(total, 1), "cer": distance / max(chars, 1), "top_k_exact_accuracy": topk_exact / max(total, 1)}


def save_checkpoint(output: Path, model, optimizer, model_config, task: str, epoch: int, metrics: dict[str, float], torch) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": "crnn",
            "task": task,
            "charset": model_config.charset,
            "preprocessing_config": {"target_height": 32, "channels": 3},
            "model_config": model_config.__dict__,
            "epoch": epoch,
            "validation_metrics": metrics,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        output,
    )


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def default_metadata(task: str) -> str:
    return "datasets/option_values/samples.jsonl" if task == "option_value" else "datasets/prices/samples.jsonl"


def default_output(task: str) -> str:
    return "models/option_value_crnn.pt" if task == "option_value" else "models/price_crnn.pt"


if __name__ == "__main__":
    raise SystemExit(main())
