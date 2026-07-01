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
    "item_metadata": "item_metadata",
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
            if task == "option_label":
                raw_line_crop = source_rect_crop(row, "raw_line_rect")
                raw_crop = source_rect_crop(row, "raw_label_rect")
                trimmed_crop = source_rect_crop(row, "trimmed_label_rect")
                cells.append(f"<td>{img_or_not_saved(raw_line_crop)}</td>")
                cells.append(f"<td>{img_tag(raw_crop) if raw_crop is not None else ''}</td>")
                cells.append(f"<td>{img_tag(trimmed_crop) if trimmed_crop is not None else ''}</td>")
                cells.append(f"<td>{img_tag(model_input_preview(image, 256))}</td>")
                cells.append(f"<td>{img_tag(image)}</td>")
                cells.append(f"<td>{img_tag(mask)}</td>")
            elif task == "option_value":
                raw_line_crop = source_rect_crop(row, "raw_line_rect")
                raw_value_crop = first_source_rect_crop(row, ("raw_value_rect", "value_rect"))
                cells.append(f"<td>{img_or_not_saved(raw_line_crop)}</td>")
                cells.append(f"<td>{img_or_not_saved(raw_value_crop)}</td>")
                cells.append(f"<td>{img_tag(model_input_preview(image, 256))}</td>")
                cells.append(f"<td>{img_tag(image)}</td>")
                cells.append(f"<td>{img_tag(residual)}</td>")
                cells.append(f"<td>{img_tag(gray)}</td>")
                cells.append(f"<td>{img_tag(mask)}</td>")
            elif task == "item_metadata":
                raw_line_crop = source_rect_crop(row, "raw_line_rect")
                label_crop = source_rect_crop(row, "label_crop_rect")
                value_crop = source_rect_crop(row, "value_crop_rect")
                cells.append(f"<td>{img_or_not_saved(raw_line_crop)}</td>")
                cells.append(f"<td>{img_or_not_saved(label_crop)}</td>")
                cells.append(f"<td>{img_or_not_saved(value_crop)}</td>")
                cells.append(f"<td>{img_tag(model_input_preview(image, 256))}</td>")
                cells.append(f"<td>{img_tag(image)}</td>")
            else:
                cells.append(f"<td>{img_tag(image)}</td>")
                cells.append(f"<td>{img_tag(residual)}</td>")
                cells.append(f"<td>{img_tag(gray)}</td>")
                cells.append(f"<td>{img_tag(mask)}</td>")
        metadata_html = metadata_block(row)
        row_class = "bad" if is_bad_row(row) else ""
        body.append(
            f"<tr class='{row_class}'>"
            f"<th>{index}</th>"
            f"<td class='label'>{html.escape(str(row.get('label', '')))}</td>"
            + "".join(cells)
            + f"<td class='meta'>{metadata_html}</td>"
            "</tr>"
        )
    headers = headers_for_task(task)
    return HTML_TEMPLATE.format(task=html.escape(task), headers=headers, rows="\n".join(body))


def headers_for_task(task: str) -> str:
    if task == "option_label":
        return "<th>Raw Line</th><th>Raw Label</th><th>Label Crop</th><th>Model</th><th>Composite</th><th>Mask</th>"
    if task == "option_value":
        return "<th>Raw Line</th><th>Value Crop</th><th>Model</th><th>Composite</th><th>Residual</th><th>Gray</th><th>Mask</th>"
    if task == "item_metadata":
        return "<th>Raw Line</th><th>Metadata Label</th><th>Metadata Value</th><th>Model</th><th>Composite</th>"
    return "<th>Composite</th><th>Residual</th><th>Gray</th><th>Mask</th>"


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


def img_or_not_saved(image: np.ndarray | None) -> str:
    if image is None:
        return "<span class='missing'>not saved</span>"
    return img_tag(image)


def source_rect_crop(row: dict, rect_key: str) -> np.ndarray | None:
    source_text = str(row.get("source_image_path", "")).strip()
    rect = row.get(rect_key)
    if not source_text or not isinstance(rect, dict):
        return None
    source_path = Path(source_text)
    if not source_path.exists():
        return None
    image = cv2.imdecode(np.fromfile(str(source_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    left = max(0, int(rect.get("left", 0)))
    top = max(0, int(rect.get("top", 0)))
    right = min(image.shape[1], int(rect.get("right", 0)))
    bottom = min(image.shape[0], int(rect.get("bottom", 0)))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def first_source_rect_crop(row: dict, rect_keys: tuple[str, ...]) -> np.ndarray | None:
    for rect_key in rect_keys:
        image = source_rect_crop(row, rect_key)
        if image is not None:
            return image
    return None


def model_input_preview(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.size == 0:
        return image
    target_height = 32
    scale = target_height / float(image.shape[0])
    resized_width = max(1, min(max_width, int(round(image.shape[1] * scale))))
    resized = cv2.resize(image, (resized_width, target_height), interpolation=cv2.INTER_AREA)
    padded = np.zeros((target_height, max_width, image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype)
    if image.ndim == 2:
        padded = padded[:, :, 0]
        padded[:, :resized_width] = resized
    else:
        padded[:, :resized_width, :] = resized
    return padded


def metadata_block(row: dict) -> str:
    keys = [
        "field_name",
        "field_type",
        "metadata_key",
        "original_field_type",
        "label_quality",
        "review_status",
        "semantic_validation_status",
        "semantic_validation_reason",
        "rejection_reason",
        "capture_pair_id",
        "session_id",
        "line_text",
        "raw_line_text",
        "line_type",
        "parsed_line_text",
        "parsed_option_key",
        "parsed_value_text",
        "selected_prediction",
        "raw_prediction",
        "was_corrected",
        "confidence",
        "crop_rect",
        "coordinate_system",
        "raw_line_rect",
        "price_search_rect",
        "price_tight_rect",
        "crop_source",
        "price_search_roi_path",
        "price_tight_crop_path",
        "price_color_mask_path",
        "price_component_mask_path",
        "label_crop_rect",
        "value_crop_rect",
        "raw_label_rect",
        "trimmed_label_rect",
        "label_rect",
        "value_rect",
        "raw_value_rect",
        "crop_quality_score",
        "touches_left_edge",
        "touches_right_edge",
        "touches_top_edge",
        "touches_bottom_edge",
        "contains_leading_bullet",
        "contains_value_like_text",
        "contains_colon_like_text",
        "contains_label_text",
        "value_sign_without_digit",
        "value_crop_full_line_like",
        "source_image_path",
    ]
    lines = []
    for key in keys:
        value = row.get(key, "")
        if value in ("", None, [], {}):
            continue
        lines.append(f"<div><b>{html.escape(key)}</b>: {html.escape(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))}</div>")
    return "".join(lines)


def is_bad_row(row: dict) -> bool:
    return bool(
        row.get("rejection_reason")
        or row.get("review_status") == "rejected"
        or row.get("label_quality") == "rejected"
        or row.get("semantic_validation_status") == "failed"
    )


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
.bad {{ background: #fff3f3; }}
</style>
</head>
<body>
<h1>{task}</h1>
<table>
<thead><tr><th>#</th><th>Label</th>{headers}<th>Metadata</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
