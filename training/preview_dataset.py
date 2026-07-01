from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from maple_price_tool.config import load_config
from recognition.dataset import split_saved_training_image


TASK_DIRS = {
    "option_label": "option_labels",
    "option_value": "option_values",
    "price": "prices",
    "rejected": "rejected",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a recognition dataset contact sheet.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--task", choices=tuple(TASK_DIRS), required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    dataset_dir = args.dataset_dir or config.vision.training_dataset_dir
    metadata = dataset_dir / TASK_DIRS[args.task] / "samples.jsonl"
    html_text = render_preview(metadata, args.task, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_text, encoding="utf-8")
    print(f"preview={args.output} task={args.task}")
    return 0


def render_preview(metadata: Path, task: str, limit: int = 100) -> str:
    rows = read_rows(metadata)[: max(0, limit)]
    body = []
    if not rows:
        body.append("<p>No samples found.</p>")
    for index, row in enumerate(rows, start=1):
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.is_absolute():
            image_path = metadata.parent / image_path
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED) if image_path.exists() else None
        cells = []
        if image is None:
            cells.append("<td colspan='4' class='missing'>missing image</td>")
        else:
            residual, gray, mask = split_saved_training_image(image)
            cells.append(f"<td>{img_tag(image)}</td>")
            cells.append(f"<td>{img_tag(residual)}</td>")
            cells.append(f"<td>{img_tag(gray)}</td>")
            cells.append(f"<td>{img_tag(mask)}</td>")
        metadata_html = metadata_block(row)
        body.append(
            "<tr>"
            f"<th>{index}</th>"
            f"<td class='label'>{html.escape(str(row.get('label', '')))}</td>"
            + "".join(cells)
            + f"<td class='meta'>{metadata_html}</td>"
            "</tr>"
        )
    return HTML_TEMPLATE.format(task=html.escape(task), rows="\n".join(body))


def read_rows(metadata: Path) -> list[dict]:
    if not metadata.exists():
        return []
    rows = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def img_tag(image: np.ndarray) -> str:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return "<span class='missing'>encode failed</span>"
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}' alt='sample'>"


def metadata_block(row: dict) -> str:
    keys = [
        "field_name",
        "field_type",
        "label_quality",
        "review_status",
        "capture_pair_id",
        "session_id",
        "selected_prediction",
        "raw_prediction",
        "was_corrected",
        "confidence",
        "crop_rect",
        "source_image_path",
        "rejection_reason",
    ]
    lines = []
    for key in keys:
        value = row.get(key, "")
        if value in ("", None, [], {}):
            continue
        lines.append(f"<div><b>{html.escape(key)}</b>: {html.escape(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))}</div>")
    return "".join(lines)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset preview - {task}</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2933; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #d9e2ec; padding: 8px; vertical-align: top; }}
th {{ text-align: right; color: #52606d; }}
img {{ image-rendering: pixelated; max-width: 220px; max-height: 72px; background: #111; }}
.label {{ font-weight: 700; white-space: nowrap; }}
.meta {{ font-size: 12px; line-height: 1.45; }}
.missing {{ color: #b00020; }}
</style>
</head>
<body>
<h1>{task}</h1>
<table>
<thead><tr><th>#</th><th>Label</th><th>Composite</th><th>Residual</th><th>Gray</th><th>Mask</th><th>Metadata</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
