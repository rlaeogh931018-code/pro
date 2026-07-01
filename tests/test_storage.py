from datetime import datetime
from pathlib import Path

from maple_price_tool.domain import AnalysisResult, FieldResult
from maple_price_tool.storage import Storage, final_record_from_analysis


def make_analysis(tmp_path: Path) -> AnalysisResult:
    image_path = tmp_path / "capture.png"
    image_path.write_bytes(b"fake")
    return AnalysisResult(
        item_key="118 / Wand",
        req_level=FieldResult(118, 0.9, "REQ LEV : 118"),
        equipment_type=FieldResult("Wand", 0.8, "장비분류 : 완드"),
        price_meso=FieldResult(12_000_000, 0.85, "12,000,000"),
        str_value=FieldResult(0, 0.0, ""),
        dex_value=FieldResult(0, 0.0, ""),
        int_value=FieldResult(3, 0.92, "INT : +3"),
        luk_value=FieldResult(0, 0.0, ""),
        attack=FieldResult(0, 0.0, ""),
        magic_attack=FieldResult(141, 0.93, "마력 : +141"),
        upgrade_count=FieldResult(0, 0.94, "업그레이드 가능 횟수 : 0"),
        black_crystal=FieldResult(0, 0.0, ""),
        equipment_options=FieldResult("INT +3", 0.8, "INT : +3"),
        potential=FieldResult("Boss damage 30%", 0.7, "raw potential"),
        image_path=image_path,
        captured_at=datetime.now(),
    )


def test_storage_saves_record_and_detects_duplicate(tmp_path: Path):
    storage = Storage(tmp_path / "records.sqlite3")
    analysis = make_analysis(tmp_path)
    record = final_record_from_analysis(analysis, analysis.editable_values())

    record_id = storage.save(record)

    assert record_id == 1
    assert storage.has_recent_duplicate(record, within_seconds=300)


def test_final_record_from_analysis_uses_edited_values(tmp_path: Path):
    analysis = make_analysis(tmp_path)
    values = analysis.editable_values()
    values["magic_attack"] = 150

    record = final_record_from_analysis(analysis, values)

    assert record.item_key == "118 / Wand"
    assert record.magic_attack == 150
    assert record.confidences["magic_attack"] == 0.93
    assert record.raw_values["magic_attack"] == "마력 : +141"
