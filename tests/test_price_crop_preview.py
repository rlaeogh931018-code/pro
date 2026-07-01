from __future__ import annotations

from datetime import datetime

from maple_price_tool.domain import AnalysisResult, FieldResult, RecognitionTrace, Rect
from maple_price_tool.ui import crop_row_title, label_value_crop_rows


def test_price_crop_preview_row_uses_tight_crop_artifacts(tmp_path):
    analysis = AnalysisResult(
        item_key="98 / wand",
        req_level=FieldResult(98, 0.9),
        equipment_type=FieldResult("wand", 0.9),
        price_meso=FieldResult(23588919, 0.9),
        str_value=FieldResult(0, 0.0),
        dex_value=FieldResult(0, 0.0),
        int_value=FieldResult(0, 0.0),
        luk_value=FieldResult(0, 0.0),
        attack=FieldResult(0, 0.0),
        magic_attack=FieldResult(0, 0.0),
        upgrade_count=FieldResult(0, 0.0),
        black_crystal=FieldResult("", 0.0),
        equipment_options=FieldResult("", 0.0),
        potential=FieldResult("", 0.0),
        image_path=tmp_path / "after.png",
        captured_at=datetime.now(),
        traces=[
            RecognitionTrace(
                "price_meso",
                field_type="price",
                selected_prediction=23588919,
                crop_rect=Rect(20, 10, 100, 30),
                crop_metadata={
                    "line_type": "price",
                    "crop_source": "price_tight_crop",
                    "price_search_rect": {"left": 0, "top": 0, "right": 160, "bottom": 40},
                    "price_tight_rect": {"left": 20, "top": 10, "right": 100, "bottom": 30},
                    "price_search_roi_path": str(tmp_path / "price_search_roi.png"),
                    "price_tight_crop_path": str(tmp_path / "price_tight_crop.png"),
                    "price_color_mask_path": str(tmp_path / "price_color_mask.png"),
                    "price_component_mask_path": str(tmp_path / "price_component_mask.png"),
                },
            )
        ],
    )

    row = label_value_crop_rows(analysis)[0]

    assert row["sort_key"] == "price"
    assert row["value_crop_rect"] == Rect(20, 10, 100, 30)
    assert row["price_tight_crop_path"].endswith("price_tight_crop.png")
    assert "value split failed" not in crop_row_title(row)
