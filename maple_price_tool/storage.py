from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

from .domain import AnalysisResult, FinalItemRecord


SCHEMA = """
CREATE TABLE IF NOT EXISTS item_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_key TEXT NOT NULL,
    req_level INTEGER NOT NULL,
    equipment_type TEXT NOT NULL,
    equipment_category TEXT NOT NULL DEFAULT '',
    price_meso INTEGER NOT NULL,
    str_value INTEGER NOT NULL DEFAULT 0,
    dex_value INTEGER NOT NULL DEFAULT 0,
    int_value INTEGER NOT NULL,
    luk_value INTEGER NOT NULL DEFAULT 0,
    attack INTEGER NOT NULL DEFAULT 0,
    magic_attack INTEGER NOT NULL,
    upgrade_count INTEGER NOT NULL,
    black_crystal TEXT NOT NULL DEFAULT '',
    equipment_options TEXT NOT NULL DEFAULT '',
    potential TEXT NOT NULL,
    raw_values_json TEXT NOT NULL,
    confidences_json TEXT NOT NULL,
    image_path TEXT NOT NULL,
    capture_pair_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    captured_at TEXT NOT NULL,
    saved_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_item_records_key_saved
ON item_records (item_key, saved_at);
"""


class Storage:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._ensure_columns(conn)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(item_records)").fetchall()
        }
        columns = {
            "str_value": "INTEGER NOT NULL DEFAULT 0",
            "dex_value": "INTEGER NOT NULL DEFAULT 0",
            "equipment_category": "TEXT NOT NULL DEFAULT ''",
            "luk_value": "INTEGER NOT NULL DEFAULT 0",
            "attack": "INTEGER NOT NULL DEFAULT 0",
            "black_crystal": "TEXT NOT NULL DEFAULT ''",
            "equipment_options": "TEXT NOT NULL DEFAULT ''",
            "capture_pair_id": "TEXT NOT NULL DEFAULT ''",
            "session_id": "TEXT NOT NULL DEFAULT ''",
        }
        for name, definition in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE item_records ADD COLUMN {name} {definition}")
        if "equipment_category" in existing or "equipment_category" in columns:
            conn.execute("UPDATE item_records SET equipment_category = equipment_type WHERE equipment_category = ''")

    def save(self, record: FinalItemRecord) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO item_records (
                    item_key, req_level, equipment_type, equipment_category, price_meso, str_value,
                    dex_value, int_value, luk_value, attack, magic_attack,
                    upgrade_count, black_crystal, equipment_options, potential, raw_values_json,
                    confidences_json, image_path, capture_pair_id, session_id, captured_at, saved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.item_key,
                    record.req_level,
                    record.equipment_type,
                    record.equipment_category or record.equipment_type,
                    record.price_meso,
                    record.str_value,
                    record.dex_value,
                    record.int_value,
                    record.luk_value,
                    record.attack,
                    record.magic_attack,
                    record.upgrade_count,
                    record.black_crystal,
                    record.equipment_options,
                    record.potential,
                    json.dumps(record.raw_values, ensure_ascii=False),
                    json.dumps(record.confidences, ensure_ascii=False),
                    str(record.image_path),
                    record.capture_pair_id,
                    record.session_id,
                    record.captured_at.isoformat(timespec="seconds"),
                    record.saved_at.isoformat(timespec="seconds"),
                ),
            )
            return int(cursor.lastrowid)

    def has_recent_duplicate(self, record: FinalItemRecord, within_seconds: int) -> bool:
        cutoff = datetime.now() - timedelta(seconds=within_seconds)
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM item_records
                WHERE item_key = ?
                  AND price_meso = ?
                  AND equipment_category = ?
                  AND str_value = ?
                  AND dex_value = ?
                  AND int_value = ?
                  AND luk_value = ?
                  AND attack = ?
                  AND magic_attack = ?
                  AND upgrade_count = ?
                  AND black_crystal = ?
                  AND equipment_options = ?
                  AND potential = ?
                  AND saved_at >= ?
                LIMIT 1
                """,
                (
                    record.item_key,
                    record.price_meso,
                    record.equipment_category or record.equipment_type,
                    record.str_value,
                    record.dex_value,
                    record.int_value,
                    record.luk_value,
                    record.attack,
                    record.magic_attack,
                    record.upgrade_count,
                    record.black_crystal,
                    record.equipment_options,
                    record.potential,
                    cutoff.isoformat(timespec="seconds"),
                ),
            ).fetchone()
        return row is not None

    def list_records(self) -> Iterable[sqlite3.Row]:
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            yield from conn.execute("SELECT * FROM item_records ORDER BY saved_at DESC")


def final_record_from_analysis(analysis: AnalysisResult, values: dict[str, object]) -> FinalItemRecord:
    # TODO(phase-4): block saving in the UI when required fields are None.
    # The SQLite schema is still NOT NULL, so confirmed values are converted
    # here only at the final user-approved save boundary.
    confidences = {
        "req_level": analysis.req_level.confidence,
        "equipment_type": analysis.equipment_type.confidence,
        "price_meso": analysis.price_meso.confidence,
        "str_value": analysis.str_value.confidence,
        "dex_value": analysis.dex_value.confidence,
        "int_value": analysis.int_value.confidence,
        "luk_value": analysis.luk_value.confidence,
        "attack": analysis.attack.confidence,
        "magic_attack": analysis.magic_attack.confidence,
        "upgrade_count": analysis.upgrade_count.confidence,
        "black_crystal": analysis.black_crystal.confidence,
        "equipment_options": analysis.equipment_options.confidence,
        "potential": analysis.potential.confidence,
    }
    raw_values = {
        "req_level": analysis.req_level.raw_value,
        "equipment_type": analysis.equipment_type.raw_value,
        "price_meso": analysis.price_meso.raw_value,
        "str_value": analysis.str_value.raw_value,
        "dex_value": analysis.dex_value.raw_value,
        "int_value": analysis.int_value.raw_value,
        "luk_value": analysis.luk_value.raw_value,
        "attack": analysis.attack.raw_value,
        "magic_attack": analysis.magic_attack.raw_value,
        "upgrade_count": analysis.upgrade_count.raw_value,
        "black_crystal": analysis.black_crystal.raw_value,
        "equipment_options": analysis.equipment_options.raw_value,
        "potential": analysis.potential.raw_value,
    }
    req_level = int(values["req_level"])
    equipment_type = str(values["equipment_type"]).strip()
    return FinalItemRecord(
        item_key=f"{req_level} / {equipment_type}",
        req_level=req_level,
        equipment_type=equipment_type,
        price_meso=int(values["price_meso"]),
        str_value=int(values["str_value"]),
        dex_value=int(values["dex_value"]),
        int_value=int(values["int_value"]),
        luk_value=int(values["luk_value"]),
        attack=int(values["attack"]),
        magic_attack=int(values["magic_attack"]),
        upgrade_count=int(values["upgrade_count"]),
        black_crystal=str(values["black_crystal"]).strip(),
        equipment_options=str(values["equipment_options"]).strip(),
        potential=str(values["potential"]).strip(),
        raw_values=raw_values,
        confidences=confidences,
        image_path=analysis.image_path,
        captured_at=analysis.captured_at,
        saved_at=datetime.now(),
        capture_pair_id=analysis.capture_pair_id,
        session_id=analysis.session_id,
        equipment_category=equipment_type,
    )
