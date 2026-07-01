from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


CAPTURE_STEM_RE = re.compile(r"^(?:before|after|capture)_(\d{8})_(\d{6})_(\d+)$")


def capture_pair_id_from_path(path: Path) -> str:
    match = CAPTURE_STEM_RE.match(path.stem)
    if match:
        return f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    return stable_id_from_path(path)


def session_id_from_pair_id(capture_pair_id: str) -> str:
    match = re.match(r"^(\d{8})", capture_pair_id)
    if match:
        return match.group(1)
    return datetime.now().strftime("%Y%m%d")


def stable_id_from_path(path: Path) -> str:
    try:
        stat = path.stat()
        stamp = datetime.fromtimestamp(stat.st_mtime).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{abs(hash(str(path.resolve()))) & 0xFFFFFF:06x}"
    except OSError:
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def sidecar_payload(before_image_path: Path, capture_pair_id: str, session_id: str) -> str:
    return json.dumps(
        {
            "before_image_path": str(before_image_path),
            "capture_pair_id": capture_pair_id,
            "session_id": session_id,
        },
        ensure_ascii=False,
    )


def parse_sidecar(text: str) -> dict[str, str]:
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        data = json.loads(stripped)
        return {str(key): str(value) for key, value in data.items() if value is not None}
    return {"before_image_path": stripped}
