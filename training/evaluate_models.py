from __future__ import annotations

import argparse
from pathlib import Path

from maple_price_tool.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Maple recognition checkpoints.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--task", choices=["option_classifier", "option_value", "price"], default="option_classifier")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--checkpoint", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    metadata = Path(args.metadata or default_metadata(args.task))
    checkpoint = Path(args.checkpoint or default_checkpoint(args.task, config))
    if not metadata.exists():
        print(f"No evaluation metadata found: {metadata}")
        return 0
    if not checkpoint.exists():
        print(f"No checkpoint found: {checkpoint}")
        return 0
    print("Evaluation requires installed ML dependencies and validation data.")
    print(f"task={args.task} metadata={metadata} checkpoint={checkpoint}")
    return 0


def default_metadata(task: str) -> str:
    if task == "option_classifier":
        return "datasets/option_labels/samples.jsonl"
    if task == "option_value":
        return "datasets/option_values/samples.jsonl"
    return "datasets/prices/samples.jsonl"


def default_checkpoint(task: str, config) -> str:
    if task == "option_classifier":
        return str(config.vision.option_classifier_checkpoint)
    if task == "option_value":
        return str(config.vision.option_value_crnn_checkpoint)
    return str(config.vision.price_crnn_checkpoint)


if __name__ == "__main__":
    raise SystemExit(main())
