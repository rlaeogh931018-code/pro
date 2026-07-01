from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from maple_price_tool.config import load_config


TASK_DIRS = {
    "option_label": "option_labels",
    "option_value": "option_values",
    "price": "prices",
    "rejected": "rejected",
}
STATUSES = {"unreviewed", "approved", "rejected", "relabel_required"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review dataset samples without training a model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--task", choices=tuple(TASK_DIRS), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--image-path")
    parser.add_argument("--set-status", choices=sorted(STATUSES))
    parser.add_argument("--label")
    parser.add_argument("--reason", default="")
    parser.add_argument("--limit", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    dataset_dir = args.dataset_dir or config.vision.training_dataset_dir
    metadata = dataset_dir / TASK_DIRS[args.task] / "samples.jsonl"
    rows = read_rows(metadata)
    if args.index is None and args.image_path is None:
        print_listing(rows, limit=args.limit)
        return 0
    selected = select_row(rows, args.index, args.image_path)
    if selected is None:
        print("sample not found")
        return 1
    index, row = selected
    if args.set_status:
        row["review_status"] = args.set_status
        if args.set_status == "rejected" and args.reason:
            row["rejection_reason"] = args.reason
    if args.label is not None:
        row["label"] = args.label
        row["review_status"] = args.set_status or "relabel_required"
    if args.reason:
        row["rejection_reason"] = args.reason
    rows[index] = row
    atomic_write_rows(metadata, rows)
    print(f"updated index={index} image={row.get('image_path')} status={row.get('review_status', 'unreviewed')} label={row.get('label')}")
    return 0


def read_rows(metadata: Path) -> list[dict]:
    if not metadata.exists():
        return []
    rows = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def print_listing(rows: list[dict], limit: int) -> None:
    for index, row in enumerate(rows[: max(0, limit)]):
        print(
            f"{index:04d} status={row.get('review_status', 'unreviewed')} "
            f"label={row.get('label', '')} image={row.get('image_path', '')} "
            f"reason={row.get('rejection_reason', '')}"
        )
    print(f"listed={min(len(rows), max(0, limit))} total={len(rows)}")


def select_row(rows: list[dict], index: int | None, image_path: str | None) -> tuple[int, dict] | None:
    if index is not None:
        if 0 <= index < len(rows):
            return index, dict(rows[index])
        return None
    if image_path is None:
        return None
    normalized = image_path.replace("\\", "/")
    for row_index, row in enumerate(rows):
        if str(row.get("image_path", "")).replace("\\", "/") == normalized:
            return row_index, dict(row)
    return None


def atomic_write_rows(metadata: Path, rows: list[dict]) -> None:
    metadata.parent.mkdir(parents=True, exist_ok=True)
    if metadata.exists():
        shutil.copyfile(metadata, metadata.with_suffix(metadata.suffix + ".bak"))
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(metadata.parent)) as handle:
        temp_path = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temp_path.replace(metadata)


if __name__ == "__main__":
    raise SystemExit(main())
