from pathlib import Path

from maple_price_tool.identity import capture_pair_id_from_path, parse_sidecar, session_id_from_pair_id, sidecar_payload
from maple_price_tool.ui import read_before_sidecar, read_capture_pair_id_sidecar, read_session_id_sidecar, write_before_sidecar


def test_capture_pair_and_session_id_from_capture_name():
    pair_id = capture_pair_id_from_path(Path("before_20260701_233158_413526.png"))

    assert pair_id == "20260701_233158_413526"
    assert session_id_from_pair_id(pair_id) == "20260701"


def test_sidecar_json_and_legacy_text_compatibility(tmp_path):
    before = tmp_path / "before_20260701_233158_413526.png"
    after = tmp_path / "after_20260701_233200_000001.png"
    before.write_bytes(b"x")
    after.write_bytes(b"x")

    write_before_sidecar(after, before)

    assert read_before_sidecar(after) == before
    assert read_capture_pair_id_sidecar(after) == "20260701_233158_413526"
    assert read_session_id_sidecar(after) == "20260701"
    assert parse_sidecar(str(before))["before_image_path"] == str(before)
    assert "capture_pair_id" in parse_sidecar(sidecar_payload(before, "p", "s"))
