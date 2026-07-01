from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .config import VisionConfig
from .domain import AnalysisResult, CaptureResult, FieldResult, RecognitionCandidate, RecognitionTrace, Rect
from recognition.model_registry import ModelRegistry
from recognition.alignment import align_before_to_after
from recognition.preprocessing import PRICE_MAX_WIDTH

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None


logger = logging.getLogger(__name__)


class VisionError(RuntimeError):
    pass


LABEL_TEMPLATE_NAMES = {
    "req_level": "REQ LEV.png",
    "equipment_type": "장비분류.png",
    "str": "str.png",
    "dex": "dex.png",
    "int_value": "int.png",
    "luk": "luk.png",
    "attack": "attack.png",
    "magic_attack": "마력.png",
    "upgrade_count": "업그레이드가능횟수.png",
    "black_crystal": "black_crystal.png",
    "potential_ignore_defense": "잠재_공격시 몬스터의 방어율.png",
    "potential_magic": "잠재_마력.png",
    "potential_boss_damage": "잠재_보스공격시.png",
}

REQUIRED_LABEL_KEYS = {
    "req_level",
    "equipment_type",
    "magic_attack",
    "upgrade_count",
}

EQUIPMENT_TYPE_TEMPLATE_NAMES = {
    "wand": "wand.png",
    "staff": "장비분류_스태프.png",
}

EQUIPMENT_TYPE_DISPLAY = {
    "wand": "완드",
    "staff": "스태프",
    "crossbow": "석궁",
    "claw": "아대",
    "spear": "창",
    "shoes": "신발",
    "shoes_newres": "신발",
    "two_handed_sword": "두손검",
    "bow": "활",
    "shield": "방패",
    "dagger": "단검",
    "polearm": "폴암",
    "one_handed_sword": "한손검",
    "gun": "건",
    "knuckle": "너클",
    "hat": "모자",
    "overall": "한벌옷",
    "glove": "장갑",
    "top": "상의",
}

VALUE_PATTERN_NAMES = {
    "req_level": [(120, "req_level_120.png")],
    "int_value": [
        (5, "int_5.png"),
        (6, "int_6.png"),
        (8, "int_8.png"),
        (9, "int_9.png"),
        (10, "int_10.png"),
        (11, "int_11.png"),
        (12, "int_12.png"),
        (13, "int_13.png"),
    ],
    "attack": [(86, "attack_86.png"), (88, "attack_88.png"), (89, "attack_89.png"), (92, "attack_92.png")],
    "magic_attack": [
        (144, "magic_attack_144.png"),
        (146, "magic_attack_146.png"),
        (147, "magic_attack_147.png"),
        (148, "magic_attack_148.png"),
        (150, "magic_attack_150.png"),
        (153, "magic_attack_153.png"),
        (155, "magic_attack_155.png"),
        (156, "magic_attack_156.png"),
    ],
    "upgrade_count": [(0, "upgrade_count_0.png")],
    "black_crystal": [
        ("공격력, 마력 +1", "black_crystal_attack_magic_1.png"),
        ("공격력, 마력 +2", "black_crystal_attack_magic_2.png"),
    ],
    "potential_9": [
        ("마력 +9%", "potential_magic_9.png"),
        ("마력 +9%", "potential_magic_newres_9_b.png"),
        ("마력 +9%", "potential_magic_newres_9_c.png"),
    ],
    "potential_6": [("마력 +6%", "potential_magic_6.png")],
    "potential_magic_6_newres": [("마력 +6%", "potential_magic_newres_6_percent.png")],
    "potential_magic_14": [("마력 +14", "potential_magic_newres_14.png")],
    "potential_magic_16": [("마력 +16", "potential_magic_newres_16.png")],
    "potential_boss_30": [
        ("보스 공격 시 데미지 +30%", "potential_boss_damage_30.png"),
        ("보스 공격 시 데미지 +30%", "potential_boss_damage_30_b.png"),
        ("보스 공격 시 데미지 +30%", "potential_boss_damage_newres_30.png"),
    ],
    "potential_boss_20": [("보스 공격 시 데미지 +20%", "potential_boss_damage_20.png")],
    "potential_ignore_30": [
        ("공격 시 몬스터의 방어율 30% 무시", "potential_ignore_defense_30.png"),
        ("공격 시 몬스터의 방어율 30% 무시", "potential_ignore_defense_30_b.png"),
    ],
    "potential_ignore_15": [("공격 시 몬스터의 방어율 15% 무시", "potential_ignore_defense_15.png")],
    "potential_dex_16": [("DEX +16", "potential_dex_16.png")],
    "potential_dex_14": [("DEX +14", "potential_dex_14.png")],
    "potential_luk_14": [("LUK +14", "potential_luk_newres_14.png")],
    "potential_dex_6_percent": [("DEX +6%", "potential_dex_lowres_6_percent.png")],
    "potential_dex_4_percent": [("DEX +4%", "potential_dex_lowres_4_percent.png")],
    "potential_dex_6": [("DEX +6", "potential_dex_lowres_6.png")],
    "potential_maxhp_180": [("MaxHP +180", "potential_maxhp_180.png")],
    "potential_maxmp_6_percent": [("MaxMP +6%", "potential_maxmp_newres_6_percent.png")],
    "potential_str_6": [("STR +6%", "potential_str_6.png")],
    "potential_str_14": [("STR +14", "potential_str_newres_14.png")],
    "potential_int_9": [
        ("INT +9%", "potential_int_newres_9_percent.png"),
        ("INT +9%", "potential_int_newres_9_b.png"),
        ("INT +9%", "potential_int_newres_9_c.png"),
    ],
    "potential_int_6": [
        ("INT +6%", "potential_int_newres_6_percent.png"),
        ("INT +6%", "potential_int_newres_6_b.png"),
    ],
    "potential_int_12": [("INT +12", "potential_int_newres_12.png")],
    "potential_all_stat_3": [
        ("올스탯 +3%", "potential_all_stat_newres_3_percent.png"),
        ("올스탯 +3%", "potential_all_stat_newres_3_b.png"),
    ],
    "potential_all_stat_6": [("올스탯 +6%", "potential_all_stat_newres_6_percent.png")],
    "potential_status_duration_minus2": [("모든 상태이상의 지속시간 -2초", "potential_status_duration_newres_minus2.png")],
    "potential_usable_haste": [("<쓸만한 헤이스트> 스킬 사용 가능", "potential_usable_haste_newres.png")],
    "potential_speed_8": [("이동속도 +8", "potential_speed_newres_8.png")],
    "option_str_11": [(("str", "STR +11", 11), "option_str_lowres_11.png")],
    "option_mp_10": [(("mp", "MP +10", 10), "option_mp_lowres_10.png")],
    "option_int_20": [(("int", "INT +20", 20), "option_int_newres_20.png")],
    "option_physical_defense_36": [(("physical_defense", "물리 방어력 +36", 36), "option_physical_defense_newres_36.png")],
    "option_black_crystal_2": [(("black_crystal", "흑수정 강화 공격력, 마력 +2", 2), "option_black_crystal_newres_2.png")],
}

PRICE_PATTERN_NAMES = {
    950_000_000: "price_950000000.png",
    499_999_999: "price_499999999.png",
    1_299_999_999: "price_1299999999.png",
    480_000_000: "price_480000000.png",
    458_888_888: "price_458888888.png",
    455_555_555: "price_455555555.png",
    450_000_000: "price_450000000.png",
    430_000_000: "price_430000000.png",
    422_222_222: "price_422222222.png",
    385_000_000: "price_385000000.png",
    320_000_000: "price_320000000.png",
    299_999_999: "price_299999999.png",
    277_777_777: "price_277777777.png",
    249_999_999: "price_249999999.png",
    244_444_444: "price_244444444.png",
}

OPTION_LINE_LABELS = {
    "str": "STR",
    "dex": "DEX",
    "int": "INT",
    "luk": "LUK",
    "hp": "HP",
    "mp": "MP",
    "str_value": "STR",
    "dex_value": "DEX",
    "int_value": "INT",
    "luk_value": "LUK",
    "hp_value": "HP",
    "mp_value": "MP",
    "attack": "공격력",
    "magic_attack": "마력",
    "physical_defense": "물리 방어력",
    "magic_defense": "마법 방어력",
    "black_crystal": "흑수정 강화",
    "speed": "이동속도",
    "jump": "점프력",
    "slip_prevention": "미끄럼 방지 추가",
    "upgrade_count": "업그레이드 가능 횟수",
}

OPTION_VALUE_FIELDS = {
    "str": "str_value",
    "dex": "dex_value",
    "int": "int_value",
    "luk": "luk_value",
    "str_value": "str_value",
    "dex_value": "dex_value",
    "int_value": "int_value",
    "luk_value": "luk_value",
    "attack": "attack",
    "magic_attack": "magic_attack",
    "black_crystal": "black_crystal",
    "upgrade_count": "upgrade_count",
}

POTENTIAL_LINE_LABELS = {
    "boss_damage": "보스 공격 시 데미지",
    "ignore_defense": "공격 시 몬스터의 방어율 무시",
    "str": "STR",
    "int": "INT",
    "dex": "DEX",
    "luk": "LUK",
    "all_stat": "올스탯",
    "attack": "공격력",
    "magic_attack": "마력",
    "maxhp": "MaxHP",
    "maxmp": "MaxMP",
    "total_damage": "총 데미지",
    "invincible_after_hit": "피격 후 무적시간",
    "speed": "이동속도",
    "jump": "점프력",
    "status_duration": "모든 상태이상의 지속시간",
    "usable_hyper_body": "<쓸만한 하이퍼 바디> 스킬 사용 가능",
    "usable_haste": "<쓸만한 헤이스트> 스킬 사용 가능",
    "usable_sharp_eyes": "<쓸만한 샤프아이즈> 스킬 사용 가능",
    "sealed_ability": "잠재능력이 봉인되어 있습니다.",
    "sealed_need_item": "(필요 아이템: 돋보기)",
}

OPTION_SCALAR_KEYS = {
    "str",
    "dex",
    "int",
    "luk",
    "str_value",
    "dex_value",
    "int_value",
    "luk_value",
    "attack",
    "magic_attack",
    "upgrade_count",
}

NON_EXTRACTABLE_REQUIREMENT_PATTERNS = (
    "req str",
    "req dex",
    "req int",
    "req luk",
    "req pop",
    "item lev",
    "item level",
    "초보자",
    "전사",
    "마법사",
    "궁수",
    "도적",
    "해적",
    "beginner",
    "warrior",
    "magician",
    "mage",
    "bowman",
    "archer",
    "thief",
    "pirate",
)

AUCTION_ROW_CENTERS = [136, 208, 280, 352, 424, 496, 568, 640]


@dataclass(frozen=True)
class Match:
    key: str
    rect: Rect
    confidence: float


@dataclass(frozen=True)
class TooltipLine:
    rect: Rect
    image: np.ndarray
    match_image: np.ndarray | None = None

    def recognition_image(self) -> np.ndarray:
        return self.match_image if self.match_image is not None else self.image


@dataclass
class TooltipLineAnalysis:
    equipment_options: str
    potential: str
    values: dict[str, int]
    confidences: dict[str, float]
    raw: str
    traces: list[RecognitionTrace] | None = None


def rect_to_dict(rect: Rect | None) -> dict[str, int] | None:
    if rect is None:
        return None
    return {"left": rect.left, "top": rect.top, "right": rect.right, "bottom": rect.bottom}


@dataclass
class PriceDetectionResult:
    value: int | None
    confidence: float
    raw_digits: str
    search_rect: Rect
    tight_rect: Rect | None = None
    selected_row_index: int | None = None
    selected_row_y: int | None = None
    detection_method: str = "unknown"
    crop_quality_score: float = 0.0
    foreground_ratio: float = 0.0
    component_count: int = 0
    needs_review: bool = False
    rejection_reason: str = ""
    multiple_rows_detected: bool = False
    color_mask: np.ndarray | None = None
    component_mask: np.ndarray | None = None
    tight_crop: np.ndarray | None = None
    search_roi: np.ndarray | None = None

    def as_tuple(self) -> tuple[int | None, float, str]:
        return self.value, self.confidence, self.raw_digits

    def metadata(self) -> dict[str, object]:
        rect = self.tight_rect or self.search_rect
        return {
            "crop_width": rect.width,
            "crop_height": rect.height,
            "foreground_ratio": self.foreground_ratio,
            "component_count": self.component_count,
            "selected_row_index": self.selected_row_index,
            "selected_row_y": self.selected_row_y,
            "detection_method": self.detection_method,
            "crop_quality_score": self.crop_quality_score,
            "needs_review": self.needs_review,
            "rejection_reason": self.rejection_reason,
            "multiple_rows_detected": self.multiple_rows_detected,
            "search_rect": rect_to_dict(self.search_rect),
            "tight_rect": rect_to_dict(self.tight_rect),
        }


class TemplateManager:
    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir
        self.option_dir = template_dir / "option_labels"
        self.digit_dir = template_dir / "digits"
        self.equipment_dir = template_dir / "equipment_types"
        self.line_label_dir = template_dir / "line_labels"
        self.value_dir = template_dir / "value_patterns"
        self.price_dir = template_dir / "price_patterns"

    def list_templates(self) -> list[Path]:
        if not self.template_dir.exists():
            return []
        return sorted(self.template_dir.rglob("*.png"))

    def load_required(self, directory: Path, names: dict[str, str]) -> dict[str, np.ndarray]:
        loaded: dict[str, np.ndarray] = {}
        missing = []
        for key, filename in names.items():
            path = directory / filename
            image = read_image(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                missing.append(str(path))
                continue
            loaded[key] = image
        if missing:
            raise VisionError("Missing template files:\n" + "\n".join(missing))
        return loaded

    def load_optional(self, directory: Path, names: dict[str, str]) -> dict[str, np.ndarray]:
        loaded: dict[str, np.ndarray] = {}
        for key, filename in names.items():
            path = directory / filename
            image = read_image(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                loaded[key] = image
        return loaded

    def load_all_pngs(self, directory: Path) -> dict[str, np.ndarray]:
        loaded: dict[str, np.ndarray] = {}
        if not directory.exists():
            return loaded
        for path in sorted(directory.glob("*.png")):
            image = read_image(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                loaded[path.stem] = image
        return loaded

    def load_variant_pngs(self, directory: Path) -> dict[str, list[np.ndarray]]:
        loaded: dict[str, list[np.ndarray]] = {}
        if not directory.exists():
            return loaded
        for path in sorted(directory.glob("*.png")):
            image = read_image(path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            key = path.stem.split("__", 1)[0]
            loaded.setdefault(key, []).append(image)
        return loaded


class MockRecognizer:
    name = "mock"

    def __init__(self, config: VisionConfig, debug_dir: Path | None = None) -> None:
        self.config = config
        self.debug_dir = debug_dir or Path("debug")

    def analyze(self, capture: CaptureResult) -> AnalysisResult:
        debug_images: list[Path] = []
        if self.config.save_debug_images:
            debug_images = self._write_mock_debug_images(capture.image_path)

        return AnalysisResult(
            item_key="118 / Wand",
            req_level=FieldResult(118, 0.91, 118),
            equipment_type=FieldResult("Wand", 0.88, "Wand"),
            price_meso=FieldResult(12_000_000, 0.87, "12,000,000"),
            str_value=FieldResult(0, 0.0, ""),
            dex_value=FieldResult(0, 0.0, ""),
            int_value=FieldResult(3, 0.92, "+3"),
            luk_value=FieldResult(0, 0.0, ""),
            attack=FieldResult(0, 0.0, ""),
            magic_attack=FieldResult(141, 0.93, "+141"),
            upgrade_count=FieldResult(0, 0.94, "0"),
            black_crystal=FieldResult(0, 0.0, ""),
            equipment_options=FieldResult("", 0.0, ""),
            potential=FieldResult(
                "Ignore monster defense 30%\nBoss damage 30%\nIgnore monster defense 15%",
                0.72,
                "mock potential",
            ),
            image_path=capture.image_path,
            captured_at=capture.captured_at,
            debug_images=debug_images,
            analysis_artifacts={},
            before_image_path=capture.before_image_path,
            capture_pair_id=capture.capture_pair_id,
            session_id=capture.session_id,
        )

    def _write_mock_debug_images(self, image_path: Path) -> list[Path]:
        debug_dir = self.debug_dir
        debug_dir.mkdir(parents=True, exist_ok=True)
        output = debug_dir / f"mock_roi_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        try:
            shutil.copyfile(image_path, output)
        except OSError:
            return []
        return [output]


class OpenCvTemplateRecognizer:
    name = "opencv-template"
    _easyocr_reader = None

    def __init__(self, config: VisionConfig, debug_dir: Path | None = None) -> None:
        self.config = config
        self.debug_dir = debug_dir or Path("debug")
        self.templates = TemplateManager(config.template_dir)
        self.label_templates = self.templates.load_required(
            self.templates.option_dir,
            {key: LABEL_TEMPLATE_NAMES[key] for key in REQUIRED_LABEL_KEYS},
        )
        self.optional_label_templates = self.templates.load_optional(
            self.templates.option_dir,
            {key: value for key, value in LABEL_TEMPLATE_NAMES.items() if key not in REQUIRED_LABEL_KEYS},
        )
        self.equipment_templates = self.templates.load_all_pngs(self.templates.equipment_dir)
        self.equipment_templates.update(self.templates.load_optional(self.templates.option_dir, EQUIPMENT_TYPE_TEMPLATE_NAMES))
        self.option_line_templates = self.templates.load_variant_pngs(self.templates.line_label_dir / "options")
        self.potential_line_templates = self.templates.load_variant_pngs(self.templates.line_label_dir / "potentials")
        self.value_patterns = self.load_value_patterns()
        self.price_patterns = self.load_price_patterns()
        self.digit_templates = self.load_maple_digit_templates()
        self.ocr_reader = None
        self.model_registry = ModelRegistry(config)
        self.latest_analysis_artifacts: dict[str, Path] = {}

    def load_value_patterns(self) -> dict[str, list[tuple[object, np.ndarray]]]:
        loaded: dict[str, list[tuple[object, np.ndarray]]] = {}
        for key, patterns in VALUE_PATTERN_NAMES.items():
            for value, filename in patterns:
                image = read_image(self.templates.value_dir / filename, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    loaded.setdefault(key, []).append((value, image))
        return loaded

    def load_price_patterns(self) -> list[tuple[int, np.ndarray]]:
        loaded: list[tuple[int, np.ndarray]] = []
        price_dir = self.config.template_dir / "price_patterns"
        pattern_paths = set(price_dir.glob("price_*.png"))
        for filename in PRICE_PATTERN_NAMES.values():
            pattern_paths.add(price_dir / filename)
        for path in sorted(pattern_paths):
            match = re.match(r"price_(\d+)", path.stem)
            if match is None:
                continue
            value = int(match.group(1))
            image = read_image(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                loaded.append((value, image))
        return loaded

    def load_maple_digit_templates(self) -> dict[str, list[np.ndarray]]:
        loaded: dict[str, list[np.ndarray]] = {}
        digit_dir = self.config.template_dir / "maple_digits"
        for path in digit_dir.glob("*.png"):
            label = path.name[:1]
            if not label.isdigit():
                continue
            image = read_image(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                loaded.setdefault(label, []).append(normalize_digit_mask(image))
        return loaded

    @classmethod
    def get_easyocr_reader(cls, config: VisionConfig):
        if not config.enable_easyocr_fallback:
            return None
        if easyocr is None:
            logger.warning("EasyOCR fallback requested but easyocr is not installed.")
            return None
        if cls._easyocr_reader is None:
            cls._easyocr_reader = easyocr.Reader(config.easyocr_languages, gpu=config.easyocr_gpu, verbose=False)
        return cls._easyocr_reader

    def analyze(self, capture: CaptureResult) -> AnalysisResult:
        self.latest_analysis_artifacts = {}
        image = read_image(capture.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise VisionError(f"Could not read capture image: {capture.image_path}")

        layout_result = self.analyze_maple_layout(capture, image)
        if layout_result is not None:
            return self.add_ml_status_trace(layout_result)
        if find_yellow_tooltip_rect(image) is not None:
            raise VisionError("Could not verify tooltip values against image patterns.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label_matches = {
            key: self.find_best_match(gray, key, template)
            for key, template in self.label_templates.items()
        }
        optional_label_matches = {
            key: self.find_best_match(gray, key, template)
            for key, template in self.optional_label_templates.items()
        }
        missing = [key for key, match in label_matches.items() if match is None]
        if missing:
            raise VisionError("Could not find required label templates: " + ", ".join(missing))

        assert label_matches["req_level"] is not None
        assert label_matches["equipment_type"] is not None
        assert label_matches["magic_attack"] is not None
        assert label_matches["upgrade_count"] is not None

        str_match = label_matches.get("str") or optional_label_matches.get("str")
        dex_match = label_matches.get("dex") or optional_label_matches.get("dex")
        int_match = label_matches.get("int_value") or optional_label_matches.get("int_value")
        luk_match = label_matches.get("luk") or optional_label_matches.get("luk")
        attack_match = label_matches.get("attack") or optional_label_matches.get("attack")
        black_crystal_match = label_matches.get("black_crystal") or optional_label_matches.get("black_crystal")
        req_level = self.read_int_right_of(gray, label_matches["req_level"])
        equipment_type = self.read_equipment_type_right_of(gray, label_matches["equipment_type"])
        str_value = self.read_int_right_of(gray, str_match)
        dex_value = self.read_int_right_of(gray, dex_match)
        int_value = self.read_int_right_of(gray, int_match)
        luk_value = self.read_int_right_of(gray, luk_match)
        attack = self.read_int_right_of(gray, attack_match)
        magic_attack = self.read_int_right_of(gray, label_matches["magic_attack"])
        upgrade_count = self.read_int_right_of(gray, label_matches["upgrade_count"])
        black_crystal = self.read_int_right_of(gray, black_crystal_match)

        potential = self.read_potential_lines(gray, label_matches["upgrade_count"], lines=3)
        price = self.read_price_near_target_row(gray)
        debug_images = self.write_debug_image(
            image,
            capture.image_path,
            list(label_matches.values()) + list(optional_label_matches.values()),
        )

        req_level_value = req_level[0] or 0
        equipment_value = equipment_type[0] or "Unknown"
        result = AnalysisResult(
            item_key=f"{req_level_value} / {equipment_value}",
            req_level=FieldResult(req_level_value, req_level[1], req_level[2]),
            equipment_type=FieldResult(equipment_value, equipment_type[1], equipment_type[2]),
            price_meso=FieldResult(price[0] or 0, price[1], price[2]),
            str_value=FieldResult(str_value[0] or 0, str_value[1], str_value[2]),
            dex_value=FieldResult(dex_value[0] or 0, dex_value[1], dex_value[2]),
            int_value=FieldResult(int_value[0] or 0, int_value[1], int_value[2]),
            luk_value=FieldResult(luk_value[0] or 0, luk_value[1], luk_value[2]),
            attack=FieldResult(attack[0] or 0, attack[1], attack[2]),
            magic_attack=FieldResult(magic_attack[0] or 0, magic_attack[1], magic_attack[2]),
            upgrade_count=FieldResult(upgrade_count[0] or 0, upgrade_count[1], upgrade_count[2]),
            black_crystal=FieldResult(black_crystal[0] or 0, black_crystal[1], black_crystal[2]),
            potential=FieldResult(potential[0], potential[1], potential[2]),
            image_path=capture.image_path,
            captured_at=capture.captured_at,
            debug_images=debug_images,
            analysis_artifacts=dict(self.latest_analysis_artifacts),
            before_image_path=capture.before_image_path,
            capture_pair_id=capture.capture_pair_id,
            session_id=capture.session_id,
        )
        return self.add_ml_status_trace(result)

    def add_ml_status_trace(self, result: AnalysisResult) -> AnalysisResult:
        if any(trace.field_name == "option_classifier" for trace in result.traces):
            return result
        if not self.config.ml_enabled:
            reason = "ml_disabled"
            needs_review = False
        else:
            loaded = self.model_registry.get_option_classifier()
            status = self.model_registry.status("option_classifier")
            reason = "model_available" if loaded is not None and status.available else "model_unavailable"
            if status.reason.startswith("load_failed"):
                reason = "model_load_failed"
            elif status.reason not in {"loaded", "not_loaded"}:
                reason = status.reason
            needs_review = False
        result.traces.append(
            RecognitionTrace(
                field_name="option_classifier",
                selection_reason=reason,
                confidence=0.0,
                needs_review=needs_review,
            )
        )
        return result

    def analyze_maple_layout(self, capture: CaptureResult, image: np.ndarray) -> AnalysisResult | None:
        tooltip = find_yellow_tooltip_rect(image)
        if tooltip is None and capture.before_image_path is not None:
            before = read_image(capture.before_image_path, cv2.IMREAD_COLOR)
            if before is not None and before.shape == image.shape:
                tooltip = find_diff_tooltip_rect(before, image)
        search_rect = tooltip or Rect(0, 0, image.shape[1], image.shape[0])
        line_mask = self.build_diff_line_mask(capture, image, tooltip) if tooltip is not None else None

        req_level = self.read_req_level_from_tooltip(image, search_rect)
        diff_req_level = self.read_req_level_from_diff_lines(image, search_rect, line_mask) if line_mask is not None else (None, 0.0, "")
        if diff_req_level[0] is not None and diff_req_level[1] >= 0.35:
            original_value = req_level[0]
            if (
                original_value is None
                or (original_value < 30 <= diff_req_level[0])
                or diff_req_level[1] > req_level[1] + 0.10
            ):
                req_level = diff_req_level
        if req_level[0] is None and diff_req_level[1] >= 0.35:
            req_level = diff_req_level
        if req_level[0] is None:
            req_level = self.read_req_level_from_tooltip(image, search_rect)
        equipment_type = self.read_equipment_type_from_tooltip(image, search_rect)
        if equipment_type[0] in {None, "Unknown"}:
            equipment_type = self.read_equipment_type_from_diff_lines(image, search_rect, line_mask) if line_mask is not None else (None, 0.0, "")
        price_detection = self.detect_maple_price(image)
        price = price_detection.as_tuple()
        line_analysis = self.read_tooltip_line_analysis(image, search_rect, line_mask=line_mask)
        if price[0] is None:
            return self.build_price_only_result(
                capture,
                image,
                search_rect,
                price,
                req_level,
                equipment_type,
                line_analysis,
                price_detection=price_detection,
            )

        if tooltip is None:
            return self.build_price_only_result(
                capture,
                image,
                search_rect,
                price,
                req_level,
                equipment_type,
                line_analysis,
                price_detection=price_detection,
            )

        if line_analysis.equipment_options or line_analysis.potential:
            return self.build_price_only_result(
                capture,
                image,
                tooltip,
                price,
                req_level,
                equipment_type,
                line_analysis,
                price_detection=price_detection,
            )

        int_value = self.match_value_pattern(image, "int_value", Rect(tooltip.left + 80, tooltip.top + 430, tooltip.left + 170, tooltip.top + 490))
        attack = self.match_value_pattern(image, "attack", Rect(tooltip.left + 90, tooltip.top + 465, tooltip.left + 180, tooltip.top + 530))
        magic_attack = self.match_value_pattern(image, "magic_attack", Rect(tooltip.left + 85, tooltip.top + 500, tooltip.left + 190, tooltip.top + 565))
        upgrade_count = self.match_value_pattern(image, "upgrade_count", Rect(tooltip.left + 260, tooltip.top + 540, tooltip.left + 330, tooltip.top + 600))
        black_crystal = self.match_value_pattern(image, "black_crystal", Rect(tooltip.left + 200, tooltip.top + 575, tooltip.left + 390, tooltip.top + 640))
        potential_1 = self.match_best_value_pattern(
            image,
            all_potential_pattern_keys(),
            Rect(tooltip.left, tooltip.top + 630, tooltip.left + 390, tooltip.top + 705),
        )
        potential_2 = self.match_best_value_pattern(
            image,
            all_potential_pattern_keys(),
            Rect(tooltip.left, tooltip.top + 665, tooltip.left + 390, tooltip.top + 740),
        )
        potential_3 = self.match_best_value_pattern(
            image,
            all_potential_pattern_keys(),
            Rect(tooltip.left, tooltip.top + 715, tooltip.left + 390, tooltip.top + 765),
        )

        if black_crystal[1] < 0.75:
            black_crystal = ("", black_crystal[1], black_crystal[2])

        if min(
            int_value[1],
            attack[1],
            magic_attack[1],
            upgrade_count[1],
            potential_1[1],
            potential_2[1],
            potential_3[1],
        ) < 0.65:
            return self.build_price_only_result(
                capture,
                image,
                tooltip,
                price,
                req_level,
                equipment_type,
                price_detection=price_detection,
            )

        potential = f"{potential_1[0]}\n{potential_2[0]}\n{potential_3[0]}"
        debug_images = self.write_layout_debug_image(
            image,
            capture.image_path,
            tooltip,
        )
        req_level_value = req_level[0] or 0
        equipment_value = equipment_type[0] or "Unknown"
        return AnalysisResult(
            item_key=f"{req_level_value} / {equipment_value}",
            req_level=FieldResult(req_level_value, req_level[1], req_level[2]),
            equipment_type=FieldResult(equipment_value, equipment_type[1], equipment_type[2]),
            price_meso=FieldResult(price[0], price[1], price[2]),
            str_value=FieldResult(0, 0.0, ""),
            dex_value=FieldResult(0, 0.0, ""),
            int_value=FieldResult(int_value[0], int_value[1], int_value[2]),
            luk_value=FieldResult(0, 0.0, ""),
            attack=FieldResult(attack[0], attack[1], attack[2]),
            magic_attack=FieldResult(magic_attack[0], magic_attack[1], magic_attack[2]),
            upgrade_count=FieldResult(upgrade_count[0], upgrade_count[1], upgrade_count[2]),
            black_crystal=FieldResult(black_crystal[0], black_crystal[1], black_crystal[2]),
            equipment_options=FieldResult("", 0.0, ""),
            potential=FieldResult(potential, min(potential_1[1], potential_2[1], potential_3[1]), potential),
            image_path=capture.image_path,
            captured_at=capture.captured_at,
            debug_images=debug_images,
            traces=[
                *self.req_level_display_traces(image, tooltip, req_level),
                *self.item_metadata_traces(image, tooltip, req_level, equipment_type),
                self.price_trace_from_detection(price_detection, price[0], price[1]),
            ],
            analysis_artifacts=dict(self.latest_analysis_artifacts),
            before_image_path=capture.before_image_path,
            capture_pair_id=capture.capture_pair_id,
            session_id=capture.session_id,
        )

    def build_price_only_result(
        self,
        capture: CaptureResult,
        image: np.ndarray,
        tooltip: Rect,
        price: tuple[int | None, float, str],
        req_level: tuple[int | None, float, str] | None = None,
        equipment_type: tuple[str | None, float, str] | None = None,
        line_analysis: TooltipLineAnalysis | None = None,
        price_detection: PriceDetectionResult | None = None,
    ) -> AnalysisResult:
        debug_images = self.write_layout_debug_image(image, capture.image_path, tooltip)
        price_value = price[0] if price[0] is not None else None
        req_level = req_level or (None, 0.0, "")
        equipment_type = equipment_type or (None, 0.0, "")
        line_analysis = line_analysis or TooltipLineAnalysis("", "", {}, {}, "", [])
        req_level_value = req_level[0] or 0
        equipment_value = equipment_type[0] or "Unknown"
        values = line_analysis.values
        confidences = line_analysis.confidences
        traces = list(line_analysis.traces or [])
        traces.extend(self.req_level_display_traces(image, tooltip, req_level))
        traces.extend(self.item_metadata_traces(image, tooltip, req_level, equipment_type))
        if price_detection is not None:
            traces.append(self.price_trace_from_detection(price_detection, price_value, price[1]))
        elif price_value is not None:
            traces.append(
                RecognitionTrace(
                    field_name="price_meso",
                    field_type="rejected",
                    selected_prediction=price_value,
                    selection_reason="selected_row_unknown",
                    confidence=price[1],
                    needs_review=True,
                    crop_rect=tooltip,
                    crop_metadata={"rejection_reason": "selected_row_unknown"},
                    template_candidates=[RecognitionCandidate(value=price_value, score=price[1], source="template")],
                )
            )
        return AnalysisResult(
            item_key=f"{req_level_value} / {equipment_value}" if req_level_value or equipment_value != "Unknown" else "Unknown",
            req_level=FieldResult(req_level_value, req_level[1], req_level[2]),
            equipment_type=FieldResult(equipment_value, equipment_type[1], equipment_type[2]),
            price_meso=FieldResult(price_value, price[1], price[2]),
            str_value=FieldResult(values.get("str_value"), confidences.get("str_value", 0.0), line_analysis.raw),
            dex_value=FieldResult(values.get("dex_value"), confidences.get("dex_value", 0.0), line_analysis.raw),
            int_value=FieldResult(values.get("int_value"), confidences.get("int_value", 0.0), line_analysis.raw),
            luk_value=FieldResult(values.get("luk_value"), confidences.get("luk_value", 0.0), line_analysis.raw),
            attack=FieldResult(values.get("attack"), confidences.get("attack", 0.0), line_analysis.raw),
            magic_attack=FieldResult(values.get("magic_attack"), confidences.get("magic_attack", 0.0), line_analysis.raw),
            upgrade_count=FieldResult(values.get("upgrade_count"), confidences.get("upgrade_count", 0.0), line_analysis.raw),
            black_crystal=FieldResult("", 0.0, ""),
            equipment_options=FieldResult(line_analysis.equipment_options, min(confidences.values(), default=0.0), line_analysis.raw),
            potential=FieldResult(line_analysis.potential, min(confidences.values(), default=0.0), line_analysis.raw),
            image_path=capture.image_path,
            captured_at=capture.captured_at,
            debug_images=debug_images,
            traces=traces,
            analysis_artifacts=dict(self.latest_analysis_artifacts),
            before_image_path=capture.before_image_path,
            capture_pair_id=capture.capture_pair_id,
            session_id=capture.session_id,
        )

    def build_diff_line_mask(self, capture: CaptureResult, image: np.ndarray, tooltip: Rect) -> np.ndarray | None:
        if capture.before_image_path is None:
            return None
        before = read_image(capture.before_image_path, cv2.IMREAD_COLOR)
        if before is None:
            logger.warning("before image not readable: %s", capture.before_image_path)
            return None
        if before.shape != image.shape:
            logger.warning("before/after image sizes differ: before=%s after=%s", before.shape, image.shape)
            return None
        alignment_result = None
        if self.config.alignment_enabled:
            alignment_result = align_before_to_after(
                before,
                image,
                excluded_rect=tooltip,
                max_shift=self.config.alignment_max_shift,
                min_response=self.config.alignment_min_response,
            )
            before = alignment_result.image
            logger.info(
                "before/after alignment applied=%s dx=%.3f dy=%.3f response=%.3f reason=%s",
                alignment_result.applied,
                alignment_result.dx,
                alignment_result.dy,
                alignment_result.response,
                alignment_result.reason,
            )
        try:
            mask, debug_images, stats = build_diff_foreground_mask(before, image, tooltip)
        except Exception:
            logger.exception("diff foreground mask failed")
            return None
        if alignment_result is not None:
            debug_images["aligned_before_roi"] = crop_rect(before, tooltip)
            stats.update(
                {
                    "alignment_applied": float(alignment_result.applied),
                    "alignment_dx": alignment_result.dx,
                    "alignment_dy": alignment_result.dy,
                    "alignment_response": alignment_result.response,
                }
            )
        if self.config.save_debug_images or self.config.save_training_samples:
            self.latest_analysis_artifacts.update(self.write_diff_debug_images(capture.image_path, debug_images, stats))
        logger.info(
            "diff foreground mask ready before=%s after=%s beta=%.4f alpha=%.4f threshold=%.2f",
            capture.before_image_path,
            capture.image_path,
            stats["beta"],
            stats["alpha"],
            stats["threshold"],
        )
        return mask

    def read_tooltip_line_analysis(self, image: np.ndarray, search_rect: Rect, line_mask: np.ndarray | None = None) -> TooltipLineAnalysis:
        lines = self.extract_tooltip_lines(image, search_rect, line_mask=line_mask)
        if not lines:
            return TooltipLineAnalysis("", "", {}, {}, "", [])

        option_lines: list[str] = []
        potential_lines: list[str] = []
        values: dict[str, int] = {}
        confidences: dict[str, float] = {}
        raw_lines: list[str] = []
        traces: list[RecognitionTrace] = []
        in_potential = False
        seen_sealed_ability = False
        unmatched_potential_lines: list[TooltipLine] = []
        potential_candidate_lines: list[TooltipLine] = []
        used_potential_tops: set[int] = set()
        seen_option_keys: set[str] = set()
        equipment_info_seen = False
        seen_job_requirement_line = False

        for line_index, line in enumerate(lines):
            recognition_line = line.recognition_image()
            if not equipment_info_seen:
                if line.rect.width > 250 and count_red_text_pixels(line.image) > 120:
                    seen_job_requirement_line = True
                    continue
                if not seen_job_requirement_line:
                    continue
                if seen_job_requirement_line:
                    equipment_info_seen = True
                    continue
                continue
            if not in_potential and line.rect.width > 250 and count_red_text_pixels(line.image) > 120:
                continue
            option_match = self.match_option_line_label(recognition_line)
            potential_match = self.match_potential_line_label(recognition_line)
            if not in_potential and potential_match and potential_match[0] in OPTION_VALUE_FIELDS:
                if option_match is None or potential_match[1] >= option_match[1] + 0.03:
                    option_match = potential_match
            if potential_match and potential_match[0].startswith("sealed_"):
                sealed_key, sealed_score, _sealed_rect = potential_match
                should_accept_sealed = (
                    (sealed_key == "sealed_ability" and sealed_score >= 0.88)
                    or (sealed_key == "sealed_need_item" and seen_sealed_ability and sealed_score >= 0.84)
                )
                if not should_accept_sealed:
                    potential_match = None
                else:
                    if sealed_key == "sealed_ability":
                        seen_sealed_ability = True
                    text, _value, confidence = self.format_potential_line(line.image, potential_match)
                    if text and text not in potential_lines:
                        potential_lines.append(text)
                        confidences[f"potential_{len(potential_lines)}"] = confidence
                        raw_lines.append(text)
                        used_potential_tops.add(line.rect.top)
                    in_potential = True
                    continue
            if option_match:
                key, score, rect = option_match
                weapon_magic_defense = False
                refined_key = self.refine_equipment_option_key(line, key, seen_option_keys)
                if refined_key != key:
                    option_match = (refined_key, score, rect)
                    key = refined_key
                if (
                    key == "physical_defense"
                    and "attack" in seen_option_keys
                    and "magic_attack" in seen_option_keys
                    and "magic_defense" not in seen_option_keys
                ):
                    option_match = ("magic_defense", score, rect)
                    key = "magic_defense"
                    weapon_magic_defense = True
                if key == "physical_defense" and "physical_defense" in seen_option_keys and "magic_defense" not in seen_option_keys:
                    option_match = ("magic_defense", score, rect)
                elif key == "magic_defense" and "physical_defense" not in seen_option_keys and not weapon_magic_defense:
                    option_match = ("physical_defense", score, rect)
            if option_match and option_match[0] == "upgrade_count":
                text, value, confidence = self.format_option_line(line.image, option_match)
                traces.extend(make_line_training_traces(line, option_match, text, confidence, line_index))
                option_lines.append(text)
                if value is not None:
                    values["upgrade_count"] = value
                    confidences["upgrade_count"] = confidence
                raw_lines.append(text)
                in_potential = True
                seen_option_keys.add("upgrade_count")
                continue

            if in_potential and option_match and option_match[0] == "black_crystal" and line_has_orange_text(line.image):
                text, value, confidence = self.format_option_line(line.image, option_match)
                traces.extend(make_line_training_traces(line, option_match, text, confidence, line_index))
                if value is not None:
                    if text:
                        option_lines.append(text)
                        raw_lines.append(text)
                    values["black_crystal"] = value
                    confidences["black_crystal"] = confidence
                seen_option_keys.add("black_crystal")
                continue

            if in_potential:
                if line_has_orange_text(line.image):
                    known_option = self.match_known_option_line(recognition_line)
                    if known_option is not None and known_option[0] == "black_crystal":
                        key, text, value, confidence = known_option
                        traces.append(
                            RecognitionTrace(
                                field_name=key,
                                field_type="option_value",
                                line_index=line_index,
                                selected_prediction=value,
                                selection_reason="template_only",
                                confidence=confidence,
                                crop_rect=line.rect,
                                template_candidates=[RecognitionCandidate(value=value, score=confidence, source="template")],
                            )
                        )
                        option_lines.append(text)
                        raw_lines.append(text)
                        values["black_crystal"] = value
                        confidences["black_crystal"] = confidence
                        seen_option_keys.add(key)
                        continue
                if not line_has_orange_text(line.image) and line.rect.width >= 60:
                    potential_candidate_lines.append(line)
                known_potential = self.match_known_potential_line(recognition_line)
                if known_potential is not None:
                    key, text, confidence = known_potential
                    traces.extend(
                        make_line_training_traces(
                            line,
                            (key, confidence, Rect(0, 0, 1, line.image.shape[0])),
                            text,
                            confidence,
                            line_index,
                            potential_index=len(potential_lines) + 1,
                        )
                    )
                    potential_lines.append(text)
                    confidences[f"potential_{len(potential_lines)}"] = confidence
                    raw_lines.append(text)
                    used_potential_tops.add(line.rect.top)
                    continue
                if potential_match:
                    text, _value, confidence = self.format_potential_line(line.image, potential_match)
                    traces.extend(make_line_training_traces(line, potential_match, text, confidence, line_index, potential_index=len(potential_lines) + 1))
                    if text and _value not in {None, 0}:
                        potential_lines.append(text)
                        confidences[f"potential_{len(potential_lines)}"] = confidence
                        raw_lines.append(text)
                        used_potential_tops.add(line.rect.top)
                        continue
                if potential_match:
                    text, _value, confidence = self.format_potential_line(line.image, potential_match)
                    traces.extend(make_line_training_traces(line, potential_match, text, confidence, line_index, potential_index=len(potential_lines) + 1))
                    if text:
                        potential_lines.append(text)
                        confidences[f"potential_{len(potential_lines)}"] = confidence
                        raw_lines.append(text)
                        used_potential_tops.add(line.rect.top)
                        continue
                if line.rect.width >= 160:
                    unmatched_potential_lines.append(line)
                continue

            if option_match:
                text, value, confidence = self.format_option_line(line.image, option_match)
                key = option_match[0]
                if value is None or (key in OPTION_SCALAR_KEYS and value == 0):
                    known_option = self.match_known_option_line(recognition_line)
                    if known_option is not None:
                        key, text, value, confidence = known_option
                        option_match = (key, confidence, option_match[2])
                if is_non_extractable_requirement_text(text):
                    continue
                if key in OPTION_SCALAR_KEYS and key != "upgrade_count" and value == 0:
                    continue
                if value is None and key != "slip_prevention":
                    if key in {"physical_defense", "magic_defense"}:
                        seen_option_keys.add(key)
                    continue
                if should_create_option_training_traces(key, text, value):
                    traces.extend(make_line_training_traces(line, option_match, text, confidence, line_index))
                if text:
                    option_lines.append(text)
                    raw_lines.append(text)
                seen_option_keys.add(key)
                if value is not None and key in OPTION_SCALAR_KEYS:
                    field_key = OPTION_VALUE_FIELDS.get(key, key)
                    if field_key in values and key not in {"black_crystal", "upgrade_count"}:
                        continue
                    values[field_key] = value
                    confidences[field_key] = confidence

        if not potential_lines and unmatched_potential_lines:
            potential_lines.append("잠재능력이 봉인되어 있습니다.")
            confidences["potential_1"] = 0.50
            raw_lines.append("잠재능력이 봉인되어 있습니다.")
            if len(unmatched_potential_lines) >= 2:
                potential_lines.append("(필요 아이템: 돋보기)")
                confidences["potential_2"] = 0.50
                raw_lines.append("(필요 아이템: 돋보기)")

        if len(potential_lines) < 3 and not any("봉인" in line for line in potential_lines):
            for text, confidence in self.match_potential_patterns_in_rect(image, search_rect, potential_lines):
                if len(potential_lines) >= 3:
                    break
                potential_lines.append(text)
                confidences[f"potential_{len(potential_lines)}"] = confidence
                raw_lines.append(text)

        if len(potential_lines) < 3 and not any("봉인" in line for line in potential_lines):
            for candidate_line in potential_candidate_lines:
                if candidate_line.rect.top in used_potential_tops:
                    continue
                candidate_recognition = candidate_line.recognition_image()
                loose_match = self.match_potential_line_label(candidate_recognition, loose=True)
                if loose_match is None:
                    if len(potential_lines) > 0 and line_has_maple_text(candidate_line.image):
                        text = "잠재능력 인식 필요"
                        if text not in potential_lines:
                            potential_lines.append(text)
                            confidences[f"potential_{len(potential_lines)}"] = 0.0
                            raw_lines.append(text)
                            used_potential_tops.add(candidate_line.rect.top)
                    if len(potential_lines) >= 3:
                        break
                    continue
                text, _value, confidence = self.format_potential_line(candidate_line.image, loose_match)
                if not text or text in potential_lines:
                    continue
                potential_lines.append(text)
                confidences[f"potential_{len(potential_lines)}"] = confidence
                raw_lines.append(text)
                used_potential_tops.add(candidate_line.rect.top)
                if len(potential_lines) >= 3:
                    break

        if potential_lines and not any("봉인" in line for line in potential_lines):
            while len(potential_lines) < 3:
                potential_lines.append("잠재능력 인식 필요")
                confidences[f"potential_{len(potential_lines)}"] = 0.0
                raw_lines.append("잠재능력 인식 필요")

        return TooltipLineAnalysis(
            "\n".join(option_lines),
            "\n".join(potential_lines[:3]),
            values,
            confidences,
            "\n".join(raw_lines),
            traces,
        )

    def line_looks_like_equipment_info(self, line: TooltipLine) -> bool:
        template = self.label_templates.get("equipment_type")
        recognition_line = line.recognition_image()
        if template is not None:
            matches = self.match_line_label_scores(recognition_line, {"equipment_type": template})
            if matches.get("equipment_type", (0.0, Rect(0, 0, 0, 0)))[0] >= 0.38:
                return True
        return False

    def match_known_potential_line(self, line: np.ndarray) -> tuple[str, str, float] | None:
        gray = normalize_template_matching_image(line)
        best_value: object | None = None
        best_score = 0.0
        for key in all_potential_pattern_keys():
            for value, pattern in self.value_patterns.get(key, []):
                pattern_binary = normalize_template_matching_image(pattern)
                if gray.shape[0] < pattern_binary.shape[0] or gray.shape[1] < pattern_binary.shape[1]:
                    continue
                result = match_binary_template(gray, pattern_binary)
                _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_value = value
                    best_score = float(max_val)
        if best_value is None or best_score < 0.72:
            return None
        text = str(best_value)
        key = option_key_from_line_text(text)
        if key is None:
            return None
        return key, text, best_score

    def match_potential_patterns_in_rect(
        self,
        image: np.ndarray,
        rect: Rect,
        existing_lines: list[str],
    ) -> list[tuple[str, float]]:
        gray = normalize_template_matching_image(image)
        top = rect.top + int(rect.height * 0.52)
        roi = crop_rect(gray, Rect(rect.left, top, rect.right, rect.bottom))
        if roi.size == 0:
            return []
        existing_counts: dict[str, int] = {}
        for line in existing_lines:
            existing_counts[line] = existing_counts.get(line, 0) + 1
        seen_counts: dict[str, int] = {}
        candidates: list[tuple[int, str, float]] = []
        for key in all_potential_pattern_keys():
            for value, pattern in self.value_patterns.get(key, []):
                pattern_binary = normalize_template_matching_image(pattern)
                if roi.shape[0] < pattern_binary.shape[0] or roi.shape[1] < pattern_binary.shape[1]:
                    continue
                result = match_binary_template(roi, pattern_binary)
                _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val < 0.86:
                    continue
                text = str(value)
                seen_counts[text] = seen_counts.get(text, 0) + 1
                if seen_counts[text] <= existing_counts.get(text, 0):
                    continue
                candidates.append((top + int(max_loc[1]), text, float(max_val)))
        candidates.sort(key=lambda item: item[0])
        recovered: list[tuple[str, float]] = []
        used_tops: list[int] = []
        for candidate_top, text, confidence in candidates:
            if any(abs(candidate_top - used_top) < 10 for used_top in used_tops):
                continue
            recovered.append((text, confidence))
            used_tops.append(candidate_top)
        return recovered

    def match_known_option_line(self, line: np.ndarray) -> tuple[str, str, int, float] | None:
        gray = normalize_template_matching_image(line)
        best_value: object | None = None
        best_score = 0.0
        for key in (
            "option_str_11",
            "option_mp_10",
            "option_int_20",
            "option_physical_defense_36",
            "option_black_crystal_2",
        ):
            for value, pattern in self.value_patterns.get(key, []):
                pattern_binary = normalize_template_matching_image(pattern)
                if gray.shape[0] < pattern_binary.shape[0] or gray.shape[1] < pattern_binary.shape[1]:
                    continue
                result = match_binary_template(gray, pattern_binary)
                _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_value = value
                    best_score = float(max_val)
        if best_value is None or best_score < 0.88:
            return None
        key, text, value = best_value
        return str(key), str(text), int(value), best_score

    def extract_tooltip_lines(self, image: np.ndarray, search_rect: Rect, line_mask: np.ndarray | None = None) -> list[TooltipLine]:
        if line_mask is not None:
            return self.extract_mask_lines(image, search_rect, line_mask)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equipment_template = self.label_templates.get("equipment_type")
        if equipment_template is None:
            return []
        equipment_match = self.find_best_match_in_rect(gray, "equipment_type", equipment_template, search_rect, 0.60)
        if equipment_match is None:
            return []
        left = max(0, equipment_match.rect.left - 25)
        right_limit = search_rect.right if search_rect.width < image.shape[1] else image.shape[1]
        bottom_limit = search_rect.bottom if search_rect.height < image.shape[0] else image.shape[0]
        right = min(right_limit, equipment_match.rect.left + 430)
        top = max(0, equipment_match.rect.top - 10)
        bottom = min(bottom_limit, equipment_match.rect.bottom + 430)
        roi = crop_rect(image, Rect(left, top, right, bottom))
        mask_source = line_mask if line_mask is not None else image
        mask = crop_rect(mask_source, Rect(left, top, right, bottom))
        if line_mask is None:
            mask = maple_foreground_text_mask(roi)
        row_counts = (mask > 0).sum(axis=1)
        row_limit = max(40, int(mask.shape[1] * 0.72))
        min_row_pixels = 4 if mask.shape[0] < 500 else 8
        row_gap_tolerance = 3 if mask.shape[0] < 500 else 2
        active_rows = np.where((row_counts > min_row_pixels) & (row_counts < row_limit))[0]
        if len(active_rows) == 0:
            return []

        spans: list[tuple[int, int]] = []
        start = int(active_rows[0])
        previous = int(active_rows[0])
        for row_value in active_rows[1:]:
            row = int(row_value)
            if row - previous > row_gap_tolerance:
                spans.append((start, previous))
                start = row
            previous = row
        spans.append((start, previous))

        lines: list[TooltipLine] = []
        for span_top, span_bottom in spans:
            if span_bottom - span_top < 8:
                continue
            crop_top = max(0, span_top - 3)
            crop_bottom = min(roi.shape[0], span_bottom + 4)
            line = roi[crop_top:crop_bottom, :]
            line_text_mask = mask[crop_top:crop_bottom, :]
            if line_mask is None:
                line_text_mask = maple_foreground_text_mask(line)
            cols = np.where((line_text_mask > 0).sum(axis=0) > 0)[0]
            if len(cols) == 0:
                continue
            crop_left = max(0, int(cols[0]) - 3)
            crop_right = min(line.shape[1], int(cols[-1]) + 4)
            line = line[:, crop_left:crop_right]
            if line.shape[1] < 35:
                continue
            match_image = normalize_template_matching_image(line)
            lines.append(
                TooltipLine(
                    rect=Rect(left + crop_left, top + crop_top, left + crop_right, top + crop_bottom),
                    image=line,
                    match_image=match_image,
                )
            )
        return lines

    def write_diff_debug_images(self, image_path: Path, debug_images: dict[str, np.ndarray], stats: dict[str, float]) -> dict[str, Path]:
        debug_dir = self.debug_dir / "diff"
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        outputs: dict[str, Path] = {}
        for name, image in debug_images.items():
            output = debug_dir / f"{stem}_{name}.png"
            ok, encoded = cv2.imencode(".png", image)
            if ok:
                encoded.tofile(str(output))
                outputs[name] = output
        info = debug_dir / f"{stem}_stats.txt"
        info.write_text(
            "\n".join(f"{key}: {value}" for key, value in sorted(stats.items())),
            encoding="utf-8",
        )
        return outputs

    def match_line_label(self, line: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float, Rect] | None:
        if not templates:
            return None
        gray = normalize_template_matching_image(line)
        best_key = ""
        best_score = 0.0
        best_rect = Rect(0, 0, 0, 0)
        for key, template in templates.items():
            normalized_template = normalize_template_matching_image(template)
            for adjusted_template in scaled_label_templates(normalized_template, gray.shape[0]):
                if gray.shape[0] < adjusted_template.shape[0] or gray.shape[1] < adjusted_template.shape[1]:
                    continue
                result = match_binary_template(gray, adjusted_template)
                _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    h, w = adjusted_template.shape[:2]
                    x, y = max_loc
                    best_key = key
                    best_score = float(max_val)
                    best_rect = Rect(x, y, x + w, y + h)
        if not best_key or best_score < 0.58:
            return None
        return best_key, best_score, best_rect

    def match_option_line_label(self, line: np.ndarray) -> tuple[str, float, Rect] | None:
        matches = self.match_line_label_scores(line, self.option_line_templates)
        if not matches:
            return None
        ranked = sorted(matches.items(), key=lambda item: item[1][0], reverse=True)
        second_score = ranked[1][1][0] if len(ranked) > 1 else 0.0
        priority_thresholds = {
            "upgrade_count": 0.76,
            "slip_prevention": 0.66,
            "attack_speed": 0.64,
            "speed": 0.64,
        }
        for key, threshold in priority_thresholds.items():
            if key in matches and matches[key][0] >= threshold and matches[key][0] - second_score >= 0.035:
                score, rect = matches[key]
                return key, score, rect
        defense_matches = {
            key: value
            for key, value in matches.items()
            if key in {"physical_defense", "magic_defense"} and value[0] >= 0.55
        }
        if defense_matches:
            key, (score, rect) = max(defense_matches.items(), key=lambda item: item[1][0])
            return key, score, rect
        key, (score, rect) = max(matches.items(), key=lambda item: item[1][0])
        if score < 0.64:
            return None
        if score < 0.82 and score - second_score < 0.035:
            return None
        return key, score, rect

    def match_potential_line_label(self, line: np.ndarray, loose: bool = False) -> tuple[str, float, Rect] | None:
        matches = self.match_line_label_scores(line, self.potential_line_templates)
        if not matches:
            return None
        short_labels = {"str", "dex", "luk", "attack", "magic_attack", "jump"}
        relaxed_short_labels = {"int", "all_stat", "maxhp", "maxmp"}
        skill_labels = {
            "usable_hyper_body",
            "usable_haste",
            "usable_sharp_eyes",
        }
        long_labels = {
            "boss_damage",
            "ignore_defense",
            "total_damage",
            "invincible_after_hit",
            "speed",
            "status_duration",
            "sealed_ability",
            "sealed_need_item",
        }
        thresholds = {
            **{key: (0.66 if loose else 0.90) for key in short_labels},
            **{key: (0.62 if loose else 0.76) for key in relaxed_short_labels},
            **{key: (0.55 if loose else 0.70) for key in long_labels},
            **{key: 0.88 for key in skill_labels},
        }
        candidates = [
            (key, score, rect)
            for key, (score, rect) in matches.items()
            if score >= thresholds.get(key, 0.76)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])

    def refine_equipment_option_key(self, line: TooltipLine, key: str, seen_option_keys: set[str]) -> str:
        if key not in {"attack", "magic_attack"}:
            return key
        if line_has_orange_text(line.image) and line.rect.width >= 200:
            return "black_crystal"
        if line.rect.width < 205:
            return key
        if "attack" in seen_option_keys and "magic_attack" in seen_option_keys:
            return "black_crystal"
        if "physical_defense" not in seen_option_keys:
            return "physical_defense"
        if "magic_defense" not in seen_option_keys:
            return "magic_defense"
        return key

    def match_line_label_scores(self, line: np.ndarray, templates: dict[str, np.ndarray]) -> dict[str, tuple[float, Rect]]:
        scores: dict[str, tuple[float, Rect]] = {}
        if not templates:
            return scores
        gray = normalize_template_matching_image(line)
        for key, template_or_templates in templates.items():
            best_score = 0.0
            best_rect = Rect(0, 0, 0, 0)
            template_list = template_or_templates if isinstance(template_or_templates, list) else [template_or_templates]
            for template in template_list:
                normalized_template = normalize_template_matching_image(template)
                for adjusted_template in scaled_label_templates(normalized_template, gray.shape[0]):
                    if gray.shape[0] < adjusted_template.shape[0] or gray.shape[1] < adjusted_template.shape[1]:
                        continue
                    result = match_binary_template(gray, adjusted_template)
                    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
                    if float(max_val) > best_score:
                        h, w = adjusted_template.shape[:2]
                        x, y = max_loc
                        best_score = float(max_val)
                        best_rect = Rect(x, y, x + w, y + h)
            if best_score > 0.0:
                scores[key] = (best_score, best_rect)
        return scores

    def format_option_line(self, line: np.ndarray, match: tuple[str, float, Rect]) -> tuple[str, int | None, float]:
        key, confidence, rect = match
        label = OPTION_LINE_LABELS.get(key, key)
        if key in {"slip_prevention", "attack_speed"}:
            return label, None, confidence
        value, digit_confidence = read_signed_number_from_line(line, rect, self.digit_templates)
        confidence = min(confidence, digit_confidence) if digit_confidence else confidence
        if value is None:
            return label, None, confidence
        if key == "upgrade_count":
            return f"{label}: {value}", value, confidence
        if key == "black_crystal":
            return f"{label}: 공격력, 마력 {format_signed_value(value)}", value, confidence
        return f"{label} {format_signed_value(value)}", value, confidence

    def format_potential_line(self, line: np.ndarray, match: tuple[str, float, Rect]) -> tuple[str, int | None, float]:
        key, confidence, rect = match
        label = POTENTIAL_LINE_LABELS.get(key, key)
        if key.startswith("usable_") or key.startswith("sealed_"):
            return label, None, confidence
        value_region = crop_rect(line, Rect(min(line.shape[1], rect.right), 0, line.shape[1], line.shape[0]))
        has_percent = line_has_percent_symbol(value_region)
        value, digit_confidence = read_signed_potential_number_from_line(line, rect, self.digit_templates)
        confidence = min(confidence, digit_confidence) if digit_confidence else confidence
        if value is None:
            return label, None, confidence
        raw_value = value
        value, correction_reason = normalize_potential_value_with_trace(key, value, has_percent)
        if correction_reason:
            logger.debug(
                "potential value normalized key=%s raw=%s corrected=%s reason=%s",
                key,
                raw_value,
                value,
                correction_reason,
            )
        if key in {"invincible_after_hit", "status_duration"}:
            if abs(value) >= 10:
                value = abs(value) % 10
            value = -abs(value) if key == "status_duration" else value
            return f"{label} {format_signed_value(value)}초", value, confidence
        if key in {"maxhp", "maxmp"}:
            has_percent = False
        suffix = "%" if has_percent or key in {"boss_damage", "ignore_defense", "total_damage"} else ""
        if key == "ignore_defense":
            return f"공격 시 몬스터의 방어율 {abs(value)}% 무시", value, confidence
        return f"{label} {format_signed_value(value)}{suffix}", value, confidence

    def read_req_level_from_diff_lines(
        self,
        image: np.ndarray,
        tooltip: Rect,
        line_mask: np.ndarray | None,
    ) -> tuple[int | None, float, str]:
        if line_mask is None:
            return None, 0.0, ""
        template = self.label_templates.get("req_level")
        if template is None:
            return None, 0.0, ""
        lines = self.extract_mask_lines(image, tooltip, line_mask)
        for line in lines:
            recognition_line = line.recognition_image()
            if line.rect.top > tooltip.top + 260:
                break
            if count_red_text_pixels(line.image) < 35:
                continue
            red_value = read_req_level_from_red_requirement_block(line.image, self.digit_templates)
            if red_value[0] is not None:
                return red_value
            matches = self.match_line_label_scores(recognition_line, {"req_level": template})
            if "req_level" not in matches or matches["req_level"][0] < 0.45:
                continue
            score, rect = matches["req_level"]
            for width in (42, 55, 72, 95):
                value_roi = line.image[
                    :,
                    min(line.image.shape[1], rect.right) : min(line.image.shape[1], rect.right + width),
                ]
                digits, digit_confidence = recognize_req_level_digits(value_roi, self.digit_templates)
                if not digits:
                    continue
                try:
                    value = int(digits[-3:])
                except ValueError:
                    continue
                if 10 <= value <= 200:
                    return value, min(score, digit_confidence), digits
        return None, 0.0, ""

    def read_equipment_type_from_diff_lines(
        self,
        image: np.ndarray,
        tooltip: Rect,
        line_mask: np.ndarray | None,
    ) -> tuple[str | None, float, str]:
        if line_mask is None:
            return None, 0.0, ""
        template = self.label_templates.get("equipment_type")
        if template is None:
            return None, 0.0, ""
        lines = self.extract_mask_lines(image, tooltip, line_mask)
        for line in lines:
            recognition_line = line.recognition_image()
            if line.rect.top < tooltip.top + 180:
                continue
            if line.rect.top > tooltip.top + 460:
                break
            matches = self.match_line_label_scores(recognition_line, {"equipment_type": template})
            gray = ensure_grayscale(recognition_line)
            if "equipment_type" in matches and matches["equipment_type"][0] >= 0.45:
                score, rect = matches["equipment_type"]
                full_line_value = self.match_equipment_type_in_line(gray)
                value = self.read_equipment_type_right_of(
                    gray,
                    Match("equipment_type", rect, score),
                    min_confidence=0.50,
                )
                if (
                    full_line_value[0] not in {None, "Unknown"}
                    and (value[0] in {None, "Unknown"} or full_line_value[1] >= value[1] + 0.04)
                ):
                    return full_line_value
                if value[0] not in {None, "Unknown"}:
                    return value
                if full_line_value[0] not in {None, "Unknown"}:
                    return full_line_value
            if line.rect.width <= 260 and tooltip.top + 185 <= line.rect.top <= tooltip.top + 260:
                value = self.match_equipment_type_in_line(gray)
                if value[0] not in {None, "Unknown"}:
                    return value
        return None, 0.0, ""

    def match_equipment_type_in_line(self, gray: np.ndarray) -> tuple[str | None, float, str]:
        roi = normalize_template_matching_image(gray)
        best_key = None
        best_score = 0.0
        for key, template in self.equipment_templates.items():
            normalized_template = normalize_template_matching_image(template)
            for adjusted_template in scaled_label_templates(normalized_template, roi.shape[0]):
                if roi.shape[0] < adjusted_template.shape[0] or roi.shape[1] < adjusted_template.shape[1]:
                    continue
                result = match_binary_template(roi, adjusted_template)
                _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_key = key
                    best_score = float(max_val)
        if best_key is None or best_score < 0.50:
            return None, best_score, ""
        return EQUIPMENT_TYPE_DISPLAY.get(best_key, best_key), best_score, best_key

    def extract_mask_lines(self, image: np.ndarray, rect: Rect, line_mask: np.ndarray) -> list[TooltipLine]:
        roi = crop_rect(image, rect)
        residual_roi = crop_rect(line_mask, rect)
        segmentation_mask = normalize_template_matching_image(residual_roi)
        row_counts = (segmentation_mask > 0).sum(axis=1)
        row_limit = max(40, int(segmentation_mask.shape[1] * 0.72))
        min_row_pixels = 4 if segmentation_mask.shape[0] < 500 else 8
        row_gap_tolerance = 3 if segmentation_mask.shape[0] < 500 else 2
        active_rows = np.where((row_counts > min_row_pixels) & (row_counts < row_limit))[0]
        if len(active_rows) == 0:
            return []

        spans: list[tuple[int, int]] = []
        start = int(active_rows[0])
        previous = int(active_rows[0])
        for row_value in active_rows[1:]:
            row = int(row_value)
            if row - previous > row_gap_tolerance:
                spans.append((start, previous))
                start = row
            previous = row
        spans.append((start, previous))

        lines: list[TooltipLine] = []
        for span_top, span_bottom in spans:
            if span_bottom - span_top < 8:
                continue
            crop_top = max(0, span_top - 3)
            crop_bottom = min(roi.shape[0], span_bottom + 4)
            line = roi[crop_top:crop_bottom, :]
            residual_line = residual_roi[crop_top:crop_bottom, :]
            line_text_mask = segmentation_mask[crop_top:crop_bottom, :]
            cols = np.where((line_text_mask > 0).sum(axis=0) > 0)[0]
            if len(cols) == 0:
                continue
            crop_left = max(0, int(cols[0]) - 3)
            crop_right = min(line.shape[1], int(cols[-1]) + 4)
            if crop_right - crop_left < 35:
                continue
            match_image = normalize_template_matching_image(residual_line[:, crop_left:crop_right])
            lines.append(
                TooltipLine(
                    rect=Rect(rect.left + crop_left, rect.top + crop_top, rect.left + crop_right, rect.top + crop_bottom),
                    image=line[:, crop_left:crop_right],
                    match_image=match_image,
                )
            )
        return lines

    def read_req_level_from_tooltip(self, image: np.ndarray, tooltip: Rect) -> tuple[int | None, float, str]:
        template = self.label_templates.get("req_level")
        if template is None:
            return None, 0.0, ""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.find_req_level_candidates(image, gray, template, tooltip)
        if not candidates:
            return None, 0.0, ""
        value, confidence, digits = candidates[0]
        return value, confidence, digits

    def find_req_level_candidates(
        self,
        image: np.ndarray,
        gray: np.ndarray,
        template: np.ndarray,
        tooltip: Rect,
    ) -> list[tuple[int, float, str]]:
        roi = crop_rect(gray, tooltip)
        candidates: list[tuple[int, float, str, int, float]] = []
        seen_positions: set[tuple[int, int]] = set()
        for adjusted_template in scaled_label_templates(template, max(12, roi.shape[0])):
            if roi.shape[0] < adjusted_template.shape[0] or roi.shape[1] < adjusted_template.shape[1]:
                continue
            result = cv2.matchTemplate(roi, adjusted_template, cv2.TM_CCOEFF_NORMED)
            h, w = adjusted_template.shape[:2]
            ys, xs = np.where(result >= 0.42)
            for y_value, x_value in zip(ys, xs):
                x = int(x_value)
                y = int(y_value)
                position_key = (x // 3, y // 3)
                if position_key in seen_positions:
                    continue
                seen_positions.add(position_key)
                label_rect = Rect(tooltip.left + x, tooltip.top + y, tooltip.left + x + w, tooltip.top + y + h)
                red_pixels = count_red_text_pixels(crop_rect(image, label_rect))
                if red_pixels < 60:
                    continue
                value_roi = crop_rect(
                    image,
                    Rect(
                        label_rect.right,
                        label_rect.top - 4,
                        label_rect.right + 95,
                        label_rect.bottom + 4,
                    ),
                )
                digits, digit_confidence = recognize_req_level_digits(value_roi, self.digit_templates)
                if not digits:
                    continue
                try:
                    value = int(digits[-3:])
                except ValueError:
                    continue
                if not 10 <= value <= 200:
                    continue
                confidence = min(float(result[y, x]), digit_confidence)
                candidates.append((value, confidence, digits, label_rect.top, float(result[y, x])))
        candidates.sort(key=lambda item: (item[3], -item[4]))
        return [(value, confidence, digits) for value, confidence, digits, _top, _score in candidates]

    def read_equipment_type_from_tooltip(self, image: np.ndarray, tooltip: Rect) -> tuple[str | None, float, str]:
        template = self.label_templates.get("equipment_type")
        if template is None:
            return None, 0.0, ""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        match = self.find_best_match_in_rect(gray, "equipment_type", template, tooltip, min_confidence=0.70)
        return self.read_equipment_type_right_of(gray, match, min_confidence=0.62)

    def match_value_pattern(self, image: np.ndarray, key: str, rect: Rect) -> tuple[object | None, float, str]:
        return self.match_best_value_pattern(image, [key], rect)

    def match_best_value_pattern(self, image: np.ndarray, keys: list[str], rect: Rect) -> tuple[object | None, float, str]:
        patterns = [
            pattern
            for key in keys
            for pattern in self.value_patterns.get(key, [])
        ]
        if not patterns:
            return None, 0.0, ""
        gray = normalize_template_matching_image(image)
        roi = crop_rect(gray, rect)
        best_value: object | None = None
        best_score = 0.0
        for value, pattern in patterns:
            pattern_binary = normalize_template_matching_image(pattern)
            if roi.shape[0] < pattern_binary.shape[0] or roi.shape[1] < pattern_binary.shape[1]:
                continue
            result = match_binary_template(roi, pattern_binary)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_value = value
                best_score = float(max_val)
        return best_value, best_score, "" if best_value is None else str(best_value)

    def read_maple_price(self, image: np.ndarray) -> tuple[int | None, float, str]:
        return self.detect_maple_price(image).as_tuple()

    def detect_maple_price(self, image: np.ndarray) -> PriceDetectionResult:
        tooltip = find_yellow_tooltip_rect(image)
        tooltip_hand_y = find_hover_hand_y_near_tooltip(image, tooltip) if tooltip is not None else None
        hand_y = tooltip_hand_y or find_hover_hand_y(image)
        if hand_y is not None:
            row_index = nearest_auction_row_index(hand_y)
            target_y = AUCTION_ROW_CENTERS[row_index]
        else:
            row_index = None
            target_y = tooltip.top + 95 if tooltip else image.shape[0] // 3
        search_rect = self.price_search_rect(image, tooltip, target_y)

        best_value: int | None = None
        best_confidence = 0.0
        best_digits = ""
        detection_method = "auction_row_tight"

        if tooltip is not None and tooltip_hand_y is not None:
            tooltip_row_price = self.read_price_right_of_tooltip_at_y(image, tooltip, tooltip_hand_y)
            if tooltip_row_price[0] is not None:
                best_value, best_confidence, best_digits = tooltip_row_price
                detection_method = "tooltip_hand_y_tight"
        if tooltip is not None:
            tooltip_price = self.read_price_right_of_tooltip(image, tooltip)
            if best_value is None and tooltip_price[0] is not None:
                best_value, best_confidence, best_digits = tooltip_price
                detection_method = "tooltip_right_tight"

        left = search_rect.left
        right = search_rect.right

        primary = self.read_price_at_row(image, target_y, left, right)
        if best_value is None and primary[0] is not None:
            best_value, best_confidence, best_digits = primary

        candidates: list[tuple[int, float, str, int]] = []
        best_failed_confidence = primary[1]
        best_failed_digits = primary[2]
        for row_y in AUCTION_ROW_CENTERS:
            value, confidence, digits = self.read_price_at_row(image, row_y, left, right)
            if value is not None:
                candidates.append((value, confidence, digits, row_y))
            elif confidence > best_failed_confidence:
                best_failed_confidence = confidence
                best_failed_digits = digits

        if candidates:
            candidates.sort(key=lambda item: (abs(item[3] - target_y), -item[1]))
            value, confidence, digits, _row_y = candidates[0]
            if best_value is None:
                best_value, best_confidence, best_digits = value, confidence, digits
        if best_value is None:
            best_confidence = best_failed_confidence
            best_digits = best_failed_digits

        result = self.detect_tight_price_crop(
            image,
            search_rect,
            value=best_value,
            confidence=best_confidence,
            raw_digits=best_digits,
            selected_row_index=row_index,
            selected_row_y=target_y,
            detection_method=detection_method,
        )
        if self.config.save_debug_images or self.config.save_training_samples:
            self.latest_analysis_artifacts.update(self.write_price_debug_images(image, result))
        return result

    def price_search_rect(self, image: np.ndarray, tooltip: Rect | None, target_y: int) -> Rect:
        left = max(0, image.shape[1] - 520)
        right = max(left, image.shape[1] - 155)
        if right <= left:
            right = image.shape[1]
        if tooltip is not None and tooltip.right + 120 < image.shape[1]:
            left = max(left, tooltip.right + 80)
            right = min(image.shape[1], max(right, tooltip.right + 700))
        if right <= left:
            left = 0
            right = image.shape[1]
        top = max(0, target_y - 34)
        bottom = min(image.shape[0], target_y + 34)
        return Rect(left, top, right, bottom)

    def detect_tight_price_crop(
        self,
        image: np.ndarray,
        search_rect: Rect,
        value: int | None,
        confidence: float,
        raw_digits: str,
        selected_row_index: int | None,
        selected_row_y: int | None,
        detection_method: str,
    ) -> PriceDetectionResult:
        search_roi = crop_rect(image, search_rect)
        color_mask = price_color_mask(search_roi)
        plausible_rows = price_mask_row_candidates(color_mask)
        if not plausible_rows:
            return PriceDetectionResult(
                value=value,
                confidence=confidence,
                raw_digits=raw_digits,
                search_rect=search_rect,
                selected_row_index=selected_row_index,
                selected_row_y=selected_row_y,
                detection_method=detection_method,
                needs_review=True,
                rejection_reason="price_text_not_found",
                color_mask=color_mask,
                component_mask=np.zeros(color_mask.shape, dtype="uint8"),
                search_roi=search_roi,
            )
        row_center = (selected_row_y - search_rect.top) if selected_row_y is not None else search_roi.shape[0] // 2
        plausible_rows.sort(key=lambda row: (abs(row["center_y"] - row_center), -row["right"], -row["width"]))
        selected = plausible_rows[0]
        multiple_rows = sum(1 for row in plausible_rows if abs(row["center_y"] - selected["center_y"]) > 14) > 0
        left = int(selected["left"])
        top = int(selected["top"])
        right = int(selected["right"])
        bottom = int(selected["bottom"])
        tight_rect = Rect(search_rect.left + left, search_rect.top + top, search_rect.left + right, search_rect.top + bottom)
        tight_crop = crop_rect(image, tight_rect)
        component_mask = np.zeros(color_mask.shape, dtype="uint8")
        component_mask[top:bottom, left:right] = color_mask[top:bottom, left:right]
        crop_mask = color_mask[top:bottom, left:right]
        foreground_ratio = float(np.count_nonzero(crop_mask)) / float(crop_mask.size) if crop_mask.size else 0.0
        width = tight_rect.width
        height = tight_rect.height
        digit_count = sum(1 for char in raw_digits if char.isdigit())
        component_count = max(int(selected["component_count"]), digit_count)
        rejection_reason = ""
        if multiple_rows:
            rejection_reason = "multiple_price_rows_detected"
        elif width > PRICE_MAX_WIDTH:
            rejection_reason = "crop_too_large"
        elif width < 18 or height < 10:
            rejection_reason = "crop_too_small"
        elif component_count < 3 or foreground_ratio <= 0.002 or foreground_ratio >= 0.75:
            rejection_reason = "invalid_component_layout"
        quality = price_crop_quality_score(width, height, foreground_ratio, component_count, multiple_rows)
        return PriceDetectionResult(
            value=value,
            confidence=confidence,
            raw_digits=raw_digits,
            search_rect=search_rect,
            tight_rect=tight_rect,
            selected_row_index=selected_row_index,
            selected_row_y=selected_row_y,
            detection_method=detection_method,
            crop_quality_score=quality,
            foreground_ratio=foreground_ratio,
            component_count=component_count,
            needs_review=bool(rejection_reason),
            rejection_reason=rejection_reason,
            multiple_rows_detected=multiple_rows,
            color_mask=color_mask,
            component_mask=component_mask,
            tight_crop=tight_crop,
            search_roi=search_roi,
        )

    def price_trace_from_detection(
        self,
        detection: PriceDetectionResult,
        price_value: int | None,
        confidence: float,
    ) -> RecognitionTrace:
        tight_ok = detection.tight_rect is not None and not detection.needs_review and price_value is not None
        field_type = "price" if tight_ok else "rejected"
        reason = "tight_price_crop" if tight_ok else (detection.rejection_reason or "selected_row_unknown")
        return RecognitionTrace(
            field_name="price_meso",
            field_type=field_type,
            selected_prediction=price_value,
            raw_prediction=detection.raw_digits,
            selection_reason=reason,
            confidence=confidence,
            needs_review=not tight_ok,
            crop_rect=detection.tight_rect if tight_ok else detection.search_rect,
            crop_metadata=detection.metadata(),
            template_candidates=[RecognitionCandidate(value=price_value, score=confidence, source="template")],
        )

    def req_level_display_traces(
        self,
        image: np.ndarray,
        tooltip: Rect,
        req_level: tuple[int | None, float, str],
    ) -> list[RecognitionTrace]:
        value, confidence, raw_digits = req_level
        if value is None:
            return []
        template = self.label_templates.get("req_level")
        if template is None:
            return []
        best = self.find_top_req_level_label_rect(image, tooltip, template)
        if best is None:
            return []
        _top, score, label_rect = best
        value_rect = Rect(label_rect.right, label_rect.top - 4, label_rect.right + 95, label_rect.bottom + 4).clamp_within(
            Rect(0, 0, image.shape[1], image.shape[0])
        )
        metadata = {
            "ui_only": True,
            "line_text": f"REQ LEV : {value}",
            "parsed_option_key": "req_level",
            "parsed_value_text": str(value),
            "raw_prediction": raw_digits,
        }
        return [
            RecognitionTrace(
                field_name="req_level_label",
                field_type="ui_label",
                selected_prediction="req_level",
                raw_prediction="REQ LEV",
                selection_reason="req_level_template",
                confidence=min(confidence, score),
                crop_rect=label_rect,
                crop_metadata=metadata,
                template_candidates=[RecognitionCandidate(value="req_level", score=score, source="template")],
            ),
            RecognitionTrace(
                field_name="req_level",
                field_type="ui_value",
                selected_prediction=str(value),
                raw_prediction=raw_digits,
                selection_reason="req_level_digits",
                confidence=confidence,
                crop_rect=value_rect,
                crop_metadata=metadata,
                template_candidates=[RecognitionCandidate(value=value, score=confidence, source="template")],
            ),
        ]

    def item_metadata_traces(
        self,
        image: np.ndarray,
        tooltip: Rect,
        req_level: tuple[int | None, float, str],
        equipment_type: tuple[str | None, float, str],
    ) -> list[RecognitionTrace]:
        traces: list[RecognitionTrace] = []
        traces.extend(self.req_level_metadata_traces(image, tooltip, req_level))
        traces.extend(self.equipment_category_metadata_traces(image, tooltip, equipment_type))
        return traces

    def req_level_metadata_traces(
        self,
        image: np.ndarray,
        tooltip: Rect,
        req_level: tuple[int | None, float, str],
    ) -> list[RecognitionTrace]:
        value, confidence, raw_digits = req_level
        if value is None:
            return []
        template = self.label_templates.get("req_level")
        if template is None:
            return []
        best = self.find_top_req_level_label_rect(image, tooltip, template)
        if best is None:
            return []
        _top, score, label_rect = best
        image_bounds = Rect(0, 0, image.shape[1], image.shape[0])
        value_rect = Rect(label_rect.right, label_rect.top - 4, label_rect.right + 95, label_rect.bottom + 4).clamp_within(image_bounds)
        line_rect = Rect(label_rect.left, min(label_rect.top, value_rect.top), value_rect.right, max(label_rect.bottom, value_rect.bottom)).clamp_within(image_bounds)
        metadata = {
            "metadata_key": "req_level",
            "line_type": "metadata_req_level",
            "raw_line_text": f"REQ LEV : {value}",
            "line_text": f"REQ LEV : {value}",
            "label_crop_rect": rect_to_dict(label_rect),
            "value_crop_rect": rect_to_dict(value_rect),
            "parsed_value_text": str(value),
            "raw_prediction": raw_digits,
        }
        return [
            RecognitionTrace(
                field_name="req_level",
                field_type="item_metadata",
                selected_prediction=str(value),
                raw_prediction=raw_digits,
                selection_reason="metadata_req_level",
                confidence=min(confidence, score),
                crop_rect=line_rect,
                crop_metadata=metadata,
                template_candidates=[RecognitionCandidate(value=value, score=min(confidence, score), source="template")],
            )
        ]

    def equipment_category_metadata_traces(
        self,
        image: np.ndarray,
        tooltip: Rect,
        equipment_type: tuple[str | None, float, str],
    ) -> list[RecognitionTrace]:
        value, confidence, raw_value = equipment_type
        if value in {None, "", "Unknown"}:
            return []
        template = self.label_templates.get("equipment_type")
        if template is None:
            return []
        best = self.find_top_scaled_label_rect(image, tooltip, template, min_confidence=0.38, top_fraction=0.70)
        if best is None:
            return []
        image_bounds = Rect(0, 0, image.shape[1], image.shape[0])
        _top, score, label_rect = best
        label_rect = label_rect.clamp_within(image_bounds)
        value_rect = Rect(label_rect.right, label_rect.top - 6, label_rect.right + 180, label_rect.bottom + 8).clamp_within(image_bounds)
        line_rect = Rect(label_rect.left, min(label_rect.top, value_rect.top), value_rect.right, max(label_rect.bottom, value_rect.bottom)).clamp_within(image_bounds)
        metadata = {
            "metadata_key": "equipment_category",
            "line_type": "metadata_equipment_category",
            "raw_line_text": f"장비분류 : {value}",
            "line_text": f"장비분류 : {value}",
            "label_crop_rect": rect_to_dict(label_rect),
            "value_crop_rect": rect_to_dict(value_rect),
            "parsed_value_text": str(value),
            "raw_prediction": raw_value,
        }
        return [
            RecognitionTrace(
                field_name="equipment_category",
                field_type="item_metadata",
                selected_prediction=str(value),
                raw_prediction=raw_value,
                selection_reason="metadata_equipment_category",
                confidence=min(confidence, score),
                crop_rect=line_rect,
                crop_metadata=metadata,
                template_candidates=[RecognitionCandidate(value=value, score=min(confidence, score), source="template")],
            )
        ]

    def find_top_req_level_label_rect(
        self,
        image: np.ndarray,
        tooltip: Rect,
        template: np.ndarray,
    ) -> tuple[int, float, Rect] | None:
        return self.find_top_scaled_label_rect(image, tooltip, template, min_confidence=0.42, top_fraction=0.40)

    def find_top_scaled_label_rect(
        self,
        image: np.ndarray,
        tooltip: Rect,
        template: np.ndarray,
        min_confidence: float,
        top_fraction: float,
    ) -> tuple[int, float, Rect] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        roi = crop_rect(gray, tooltip)
        best: tuple[int, float, Rect] | None = None
        upper_limit = tooltip.top + max(80, int(tooltip.height * top_fraction))
        for adjusted_template in scaled_label_templates(template, max(12, roi.shape[0])):
            if roi.shape[0] < adjusted_template.shape[0] or roi.shape[1] < adjusted_template.shape[1]:
                continue
            result = cv2.matchTemplate(roi, adjusted_template, cv2.TM_CCOEFF_NORMED)
            h, w = adjusted_template.shape[:2]
            ys, xs = np.where(result >= min_confidence)
            for y, x in zip(ys, xs):
                rect = Rect(tooltip.left + int(x), tooltip.top + int(y), tooltip.left + int(x) + w, tooltip.top + int(y) + h)
                if rect.top > upper_limit:
                    continue
                score = float(result[int(y), int(x)])
                if best is None or rect.top < best[0] or (rect.top == best[0] and score > best[1]):
                    best = (rect.top, score, rect)
        return best

    def write_price_debug_images(self, image: np.ndarray, result: PriceDetectionResult) -> dict[str, Path]:
        debug_dir = self.debug_dir / "price"
        debug_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        outputs: dict[str, Path] = {}
        images = {
            "price_search_roi": result.search_roi,
            "price_color_mask": result.color_mask,
            "price_component_mask": result.component_mask,
            "price_tight_crop": result.tight_crop,
        }
        annotated = image.copy()
        cv2.rectangle(
            annotated,
            (result.search_rect.left, result.search_rect.top),
            (result.search_rect.right, result.search_rect.bottom),
            (255, 180, 0),
            2,
        )
        if result.tight_rect is not None:
            cv2.rectangle(
                annotated,
                (result.tight_rect.left, result.tight_rect.top),
                (result.tight_rect.right, result.tight_rect.bottom),
                (0, 255, 0) if not result.needs_review else (0, 0, 255),
                2,
            )
        images["price_annotated"] = annotated
        for name, debug_image in images.items():
            if debug_image is None:
                continue
            output = debug_dir / f"{stamp}_{name}.png"
            ok, encoded = cv2.imencode(".png", debug_image)
            if ok:
                encoded.tofile(str(output))
                outputs[name] = output
        return outputs

    def read_price_right_of_tooltip(self, image: np.ndarray, tooltip: Rect) -> tuple[int | None, float, str]:
        best: tuple[int | None, float, str] = (None, 0.0, "")
        if tooltip.right + 220 >= image.shape[1]:
            return best
        left = min(image.shape[1], tooltip.right + 220)
        right = min(image.shape[1], tooltip.right + 600)
        for offset_y in (25, 35, 45):
            target_y = tooltip.top + offset_y
            top = max(0, target_y - 25)
            bottom = min(image.shape[0], target_y + 25)
            roi = image[top:bottom, left:right]
            digits, confidence = recognize_maple_price_digits(roi, self.digit_templates)
            if is_valid_selected_price_digits(digits, confidence) and confidence > best[1]:
                best = (int(digits), confidence, digits)
        return best

    def read_price_right_of_tooltip_at_y(self, image: np.ndarray, tooltip: Rect, target_y: int) -> tuple[int | None, float, str]:
        if tooltip.right + 80 >= image.shape[1]:
            return None, 0.0, ""
        best: tuple[int | None, float, str] = (None, 0.0, "")
        left = min(image.shape[1], tooltip.right + 120)
        right = min(image.shape[1], tooltip.right + 700)
        for half_height in (20, 25, 30):
            top = max(0, target_y - half_height)
            bottom = min(image.shape[0], target_y + half_height)
            roi = image[top:bottom, left:right]
            digits, confidence = recognize_maple_price_digits(roi, self.digit_templates)
            if is_valid_selected_price_digits(digits, confidence) and confidence > best[1]:
                best = (int(digits), confidence, digits)
        return best

    def read_price_at_row(self, image: np.ndarray, target_y: int, left: int, right: int) -> tuple[int | None, float, str]:
        top = max(0, target_y - 25)
        bottom = min(image.shape[0], target_y + 25)
        roi = image[top:bottom, left:right]
        if roi.size == 0:
            return None, 0.0, ""

        digits, digit_confidence = recognize_maple_price_digits(roi, self.digit_templates)
        if is_valid_price_digits(digits, digit_confidence):
            return int(digits), digit_confidence, digits

        pattern_value, pattern_score = self.match_price_pattern(roi)
        if pattern_value is not None and pattern_score >= 0.82:
            return pattern_value, pattern_score, str(pattern_value)
        return None, max(pattern_score, digit_confidence), digits

    def match_price_pattern(self, roi: np.ndarray) -> tuple[int | None, float]:
        if not self.price_patterns or roi.size == 0:
            return None, 0.0
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        best_value = None
        best_score = 0.0
        for value, pattern in self.price_patterns:
            if gray.shape[0] < pattern.shape[0] or gray.shape[1] < pattern.shape[1]:
                continue
            result = cv2.matchTemplate(gray, pattern, cv2.TM_CCOEFF_NORMED)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_value = value
                best_score = float(max_val)
        return best_value, best_score

    def find_best_match(self, gray: np.ndarray, key: str, template: np.ndarray) -> Match | None:
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val < self.config.option_threshold:
            logger.info("label not found key=%s confidence=%.3f", key, max_val)
            return None
        h, w = template.shape[:2]
        x, y = max_loc
        return Match(key=key, rect=Rect(x, y, x + w, y + h), confidence=float(max_val))

    def find_best_match_in_rect(
        self,
        gray: np.ndarray,
        key: str,
        template: np.ndarray,
        rect: Rect,
        min_confidence: float,
    ) -> Match | None:
        roi = crop_rect(gray, rect)
        if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            return None
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val < min_confidence:
            logger.info("label not found in tooltip key=%s confidence=%.3f", key, max_val)
            return None
        h, w = template.shape[:2]
        x, y = max_loc
        left = rect.left + x
        top = rect.top + y
        return Match(key=key, rect=Rect(left, top, left + w, top + h), confidence=float(max_val))

    def find_best_red_match_in_rect(
        self,
        image: np.ndarray,
        gray: np.ndarray,
        key: str,
        template: np.ndarray,
        rect: Rect,
        min_confidence: float,
    ) -> Match | None:
        roi = crop_rect(gray, rect)
        if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            return None
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        h, w = template.shape[:2]
        ys, xs = np.where(result >= min_confidence)
        best: tuple[float, int, int, int] | None = None
        for y_value, x_value in zip(ys, xs):
            x = int(x_value)
            y = int(y_value)
            candidate = Rect(rect.left + x, rect.top + y, rect.left + x + w, rect.top + y + h)
            red_pixels = count_red_text_pixels(crop_rect(image, candidate))
            if red_pixels < 20:
                continue
            score = float(result[y, x])
            if best is None or (score, red_pixels) > (best[0], best[1]):
                best = (score, red_pixels, x, y)
        if best is None:
            logger.info("red label not found key=%s", key)
            return None
        score, _red_pixels, x, y = best
        left = rect.left + x
        top = rect.top + y
        return Match(key=key, rect=Rect(left, top, left + w, top + h), confidence=score)

    def read_int_right_of(self, gray: np.ndarray, match: Match | None) -> tuple[int | None, float, str]:
        if match is None:
            return None, 0.0, ""
        roi = crop_right_of(gray, match.rect, width=220, y_pad=6)
        return self.read_int_from_roi(roi)

    def read_equipment_type_right_of(
        self,
        gray: np.ndarray,
        match: Match | None,
        min_confidence: float | None = None,
    ) -> tuple[str | None, float, str]:
        if match is None:
            return None, 0.0, ""
        if not self.equipment_templates:
            return "Unknown", 0.0, ""
        roi = normalize_template_matching_image(crop_right_of(gray, match.rect, width=180, y_pad=8))
        best_key = None
        best_score = 0.0
        for key, template in self.equipment_templates.items():
            template_binary = normalize_template_matching_image(template)
            if roi.shape[0] < template_binary.shape[0] or roi.shape[1] < template_binary.shape[1]:
                continue
            result = match_binary_template(roi, template_binary)
            _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_key = key
                best_score = float(max_val)
        threshold = self.config.option_threshold if min_confidence is None else min_confidence
        if best_key is None or best_score < threshold:
            return "Unknown", best_score, ""
        return EQUIPMENT_TYPE_DISPLAY.get(best_key, best_key), best_score, best_key

    def read_int_from_roi(self, roi: np.ndarray) -> tuple[int | None, float, str]:
        candidates, raw, confidence = self.read_int_candidates_from_roi(roi)
        if not candidates:
            return None, 0.0, raw
        return candidates[0], confidence, raw

    def read_largest_int_from_roi(self, roi: np.ndarray) -> tuple[int | None, float, str]:
        candidates, raw, confidence = self.read_int_candidates_from_roi(roi)
        if not candidates:
            return None, 0.0, raw
        return max(candidates, key=abs), confidence, raw

    def read_int_candidates_from_roi(self, roi: np.ndarray) -> tuple[list[int], str, float]:
        reader = self.ocr_reader or self.get_easyocr_reader(self.config)
        if reader is None:
            return [], "easyocr fallback disabled", 0.0
        self.ocr_reader = reader
        prepared = prepare_ocr_roi(roi)
        results = reader.readtext(prepared, detail=1, paragraph=False, allowlist="0123456789,+-")
        raw_parts = []
        confidences = []
        for _bbox, text, confidence in results:
            raw_parts.append(str(text))
            confidences.append(float(confidence))
        raw = " ".join(raw_parts)
        normalized = raw.replace(",", "")
        candidates = []
        for match in re.finditer(r"[+-]?\d+", normalized):
            try:
                candidates.append(int(match.group(0)))
            except ValueError:
                continue
        confidence = max(confidences, default=0.0)
        return candidates, raw, confidence

    def read_potential_lines(self, gray: np.ndarray, upgrade_match: Match | None, lines: int = 3) -> tuple[str, float, str]:
        if upgrade_match is None:
            return "", 0.0, ""
        line_height = max(18, upgrade_match.rect.height + 4)
        start_y = min(gray.shape[0], upgrade_match.rect.bottom + line_height)
        line_values = []
        confidences = []
        raw_values = []
        for index in range(lines):
            top = start_y + index * line_height
            bottom = min(gray.shape[0], top + line_height)
            if top >= bottom:
                line_values.append("")
                continue
            roi = gray[top:bottom, :]
            value, confidence, raw = self.read_int_from_roi(roi)
            line_values.append("" if value is None else str(value))
            raw_values.append(raw)
            if confidence:
                confidences.append(confidence)
        return "\n".join(line_values), min(confidences, default=0.0), "\n".join(raw_values)

    def read_price_near_target_row(self, gray: np.ndarray) -> tuple[int | None, float, str]:
        # MVP fallback: read the largest integer in the right half. A later pass should use a price-column ROI.
        right_half = gray[:, gray.shape[1] // 2 :]
        return self.read_largest_int_from_roi(right_half)

    def write_debug_image(
        self,
        image: np.ndarray,
        image_path: Path,
        matches: Iterable[Match | None],
    ) -> list[Path]:
        if not self.config.save_debug_images:
            return []
        debug = image.copy()
        for match in matches:
            if match is None:
                continue
            cv2.rectangle(
                debug,
                (match.rect.left, match.rect.top),
                (match.rect.right, match.rect.bottom),
                (0, 255, 255),
                2,
            )
            cv2.putText(
                debug,
                f"{match.key}:{match.confidence:.2f}",
                (match.rect.left, max(0, match.rect.top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        output = self.debug_dir / f"debug_{image_path.stem}.png"
        cv2.imwrite(str(output), debug)
        return [output]

    def write_layout_debug_image(self, image: np.ndarray, image_path: Path, tooltip: Rect) -> list[Path]:
        if not self.config.save_debug_images:
            return []
        debug = image.copy()
        cv2.rectangle(debug, (tooltip.left, tooltip.top), (tooltip.right, tooltip.bottom), (0, 255, 255), 2)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        output = self.debug_dir / f"layout_debug_{image_path.stem}.png"
        cv2.imwrite(str(output), debug)
        return [output]


def crop_right_of(gray: np.ndarray, rect: Rect, width: int, y_pad: int) -> np.ndarray:
    top = max(0, rect.top - y_pad)
    bottom = min(gray.shape[0], rect.bottom + y_pad)
    left = min(gray.shape[1], rect.right)
    right = min(gray.shape[1], rect.right + width)
    return gray[top:bottom, left:right]


def crop_rect(image: np.ndarray, rect: Rect) -> np.ndarray:
    top = max(0, rect.top)
    bottom = min(image.shape[0], rect.bottom)
    left = max(0, rect.left)
    right = min(image.shape[1], rect.right)
    return image[top:bottom, left:right]


def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    if image.size == 0:
        return np.zeros((0, 0), dtype="uint8")
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_template_matching_image(image: np.ndarray) -> np.ndarray:
    gray = ensure_grayscale(image)
    if gray.size == 0:
        return np.zeros((0, 0), dtype="uint8")
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype("uint8")

    nonzero = gray[gray > 0]
    if nonzero.size == 0:
        return np.zeros_like(gray, dtype="uint8")

    low = float(np.percentile(nonzero, 5))
    high = float(np.percentile(nonzero, 95))
    if high <= low + 1.0:
        normalized = gray.copy()
    else:
        normalized = ((gray.astype("float32") - low) * (255.0 / (high - low))).clip(0, 255).astype("uint8")

    blur = cv2.GaussianBlur(normalized, (0, 0), sigmaX=0.7, sigmaY=0.7)
    otsu_value, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(18.0, min(float(otsu_value), 135.0))
    binary = (normalized >= threshold).astype("uint8") * 255
    return remove_matching_noise(binary)


def remove_matching_noise(binary_mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for index in range(1, count):
        _x, _y, width, height, area = stats[index]
        is_tiny_noise = area < 2
        is_horizontal_rule = width > 95 and height <= 4
        is_large_icon = area > 700 and width > 35 and height > 35
        if not (is_tiny_noise or is_horizontal_rule or is_large_icon):
            cleaned[labels == index] = 255
    return cleaned


def match_binary_template(roi: np.ndarray, template: np.ndarray) -> np.ndarray:
    if roi.size == 0 or template.size == 0:
        return np.zeros((1, 1), dtype="float32")
    roi_binary = normalize_template_matching_image(roi)
    template_binary = normalize_template_matching_image(template)
    if roi_binary.shape[0] < template_binary.shape[0] or roi_binary.shape[1] < template_binary.shape[1]:
        return np.zeros((1, 1), dtype="float32")
    template_mask = (template_binary > 0).astype("uint8") * 255
    if int(template_mask.sum()) == 0:
        return np.zeros((1, 1), dtype="float32")
    try:
        result = cv2.matchTemplate(roi_binary, template_binary, cv2.TM_CCORR_NORMED, mask=template_mask)
    except cv2.error:
        result = cv2.matchTemplate(roi_binary, template_binary, cv2.TM_CCORR_NORMED)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def scaled_label_templates(template: np.ndarray, line_height: int) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    source_height, source_width = template.shape[:2]
    for scale in (1.0, 0.9, 0.8, 0.7, 0.62, 1.1):
        height = max(1, int(round(source_height * scale)))
        width = max(1, int(round(source_width * scale)))
        if height < 8 or width < 8:
            continue
        if height > line_height + 3:
            continue
        resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
        candidates.append(resized)
    if source_height > line_height and source_height - line_height <= 3:
        candidates.append(template[:line_height, :])
    return candidates


def find_yellow_tooltip_rect(image: np.ndarray) -> Rect | None:
    b, g, r = cv2.split(image)
    mask = ((r > 180) & (g > 145) & (g < 235) & (b < 90)).astype("uint8") * 255
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[int, Rect]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 200 or h < 250:
            continue
        candidates.append((w * h, Rect(x, y, x + w, y + h)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def find_diff_tooltip_rect(before: np.ndarray, after: np.ndarray) -> Rect | None:
    if before.shape != after.shape:
        return None
    diff = cv2.absdiff(after, before)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _threshold, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    candidates: list[tuple[int, Rect]] = []
    height, width = after.shape[:2]
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        if component_width < 180 or component_height < 200:
            continue
        if area < 20_000:
            continue
        if component_width > width * 0.55 or component_height > height * 0.80:
            continue
        candidates.append((int(area), Rect(int(x), int(y), int(x + component_width), int(y + component_height))))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def find_hover_hand_y(image: np.ndarray) -> int | None:
    crop = image[:, :130]
    b, g, r = cv2.split(crop)
    mask = ((r > 225) & (g > 225) & (b > 225)).astype("uint8") * 255
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not (35 <= x <= 70 and 100 <= y <= 700 and 10 <= w <= 35 and 18 <= h <= 45):
            continue
        if area < 150:
            continue
        candidates.append((area, y + h // 2))
    if not candidates:
        return None
    return int(max(candidates, key=lambda item: item[0])[1])


def find_hover_hand_y_near_tooltip(image: np.ndarray, tooltip: Rect | None) -> int | None:
    if tooltip is None:
        return None
    b, g, r = cv2.split(image)
    mask = ((r > 225) & (g > 225) & (b > 225)).astype("uint8") * 255
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[float, int]] = []
    min_x = max(0, tooltip.left - 45)
    max_x = min(image.shape[1], tooltip.left + 45)
    min_y = max(0, tooltip.top - 60)
    max_y = min(image.shape[0], tooltip.bottom + 20)
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if x < min_x or x > max_x or y < min_y or y > max_y:
            continue
        if not (8 <= width <= 35 and 12 <= height <= 42 and 80 <= area <= 520):
            continue
        center_x, center_y = centroids[index]
        distance = abs(float(center_x) - tooltip.left)
        score = float(area) - distance * 6.0
        candidates.append((score, int(round(float(center_y)))))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def nearest_auction_row_index(y: int) -> int:
    return min(range(len(AUCTION_ROW_CENTERS)), key=lambda index: abs(AUCTION_ROW_CENTERS[index] - y))


def all_potential_pattern_keys() -> list[str]:
    return [
        "potential_9",
        "potential_6",
        "potential_magic_6_newres",
        "potential_magic_14",
        "potential_magic_16",
        "potential_boss_30",
        "potential_boss_20",
        "potential_ignore_30",
        "potential_ignore_15",
        "potential_dex_16",
        "potential_dex_14",
        "potential_luk_14",
        "potential_dex_6_percent",
        "potential_dex_4_percent",
        "potential_dex_6",
        "potential_maxhp_180",
        "potential_maxmp_6_percent",
        "potential_str_6",
        "potential_str_14",
        "potential_int_9",
        "potential_int_6",
        "potential_int_12",
        "potential_all_stat_3",
        "potential_all_stat_6",
        "potential_status_duration_minus2",
        "potential_usable_haste",
        "potential_speed_8",
    ]


def make_line_training_traces(
    line: TooltipLine,
    match: tuple[str, float, Rect],
    text: str,
    confidence: float,
    line_index: int,
    potential_index: int | None = None,
) -> list[RecognitionTrace]:
    key, score, local_rect = match
    field_name = f"potential_{potential_index}" if potential_index is not None else OPTION_VALUE_FIELDS.get(key, key)
    line_type = "potential_option" if potential_index is not None else "base_option"
    label_crop = build_option_label_crop_rect(line, local_rect, text)
    label_rect = label_crop["trimmed_rect"]
    label_metadata = {
        "line_text": text,
        "parsed_line_text": text,
        "line_type": line_type,
        "parsed_option_key": key,
        "selected_prediction": key,
        "raw_label_rect": rect_to_dict(label_crop["raw_rect"]),
        "trimmed_label_rect": rect_to_dict(label_crop["trimmed_rect"]),
        "label_crop_quality": label_crop["quality"],
        **label_crop["quality"],
    }
    selection_reason = "line_text_bbox"
    if label_crop["quality"]["contains_leading_bullet"]:
        selection_reason += "_trimmed_bullet"
    if label_crop["quality"]["rejection_reason"]:
        selection_reason += "_needs_review"
    label_field_type = "rejected" if label_crop["quality"]["rejection_reason"] else "option_label"
    label_needs_review = bool(label_crop["quality"]["rejection_reason"])
    label_candidates = [RecognitionCandidate(value=key, score=score, source="template")]
    label_trace = RecognitionTrace(
        field_name=f"{field_name}_label",
        field_type=label_field_type,
        line_index=line_index,
        selected_prediction=key,
        selection_reason=selection_reason,
        confidence=score,
        needs_review=label_needs_review,
        crop_rect=label_rect,
        crop_metadata=label_metadata,
        template_candidates=label_candidates,
    )
    label_value = key
    value_text = extract_value_text(text)
    value_crop = build_option_value_crop_rect(line, label_crop["trimmed_rect"], value_text, text)
    value_rect = value_crop["trimmed_rect"]
    value_metadata = {
        "line_text": text,
        "parsed_line_text": text,
        "line_type": line_type,
        "parsed_option_key": key,
        "parsed_value_text": value_text,
        "label_rect": rect_to_dict(label_crop["trimmed_rect"]),
        "value_rect": rect_to_dict(value_crop["trimmed_rect"]),
        "raw_value_rect": rect_to_dict(value_crop["raw_rect"]),
        **value_crop["quality"],
    }
    return [
        label_trace,
        RecognitionTrace(
            field_name=field_name,
            field_type="option_value",
            line_index=line_index,
            raw_prediction=value_text,
            selected_prediction=value_text,
            selection_reason="template_only",
            confidence=confidence,
            crop_rect=value_rect,
            crop_metadata=value_metadata,
            template_candidates=[RecognitionCandidate(value=value_text, score=confidence, source="template")],
        ),
    ]


def should_create_option_training_traces(key: str, text: str, value: int | None) -> bool:
    if key in {"attack_speed"}:
        return False
    if is_non_extractable_requirement_text(text):
        return False
    if value is None:
        return False
    if key in OPTION_SCALAR_KEYS and key != "upgrade_count" and value == 0:
        return False
    return True


def option_key_from_line_text(text: str) -> str | None:
    value_text = extract_value_text(text)
    label_text = str(text or "")
    if value_text:
        label_text = label_text.rsplit(value_text, 1)[0]
    compact = label_text.strip().lower().replace(":", "").replace(" ", "")
    aliases = {
        "str": "str",
        "dex": "dex",
        "int": "int",
        "luk": "luk",
        "공격력": "attack",
        "마력": "magic_attack",
        "공격": "attack",
        "magicattack": "magic_attack",
        "allstat": "all_stat",
        "올스탯": "all_stat",
        "maxhp": "maxhp",
        "maxmp": "maxmp",
    }
    if compact in aliases:
        return aliases[compact]
    if "마력" in label_text:
        return "magic_attack"
    if "공격" in label_text:
        return "attack"
    if "방어" in label_text and "무시" in label_text:
        return "ignore_defense"
    if "보스" in label_text:
        return "boss_damage"
    return None


def is_non_extractable_requirement_text(text: str) -> bool:
    compact = normalize_requirement_text(text)
    return any(pattern.replace(" ", "") in compact for pattern in NON_EXTRACTABLE_REQUIREMENT_PATTERNS)


def normalize_requirement_text(text: str) -> str:
    return str(text or "").strip().lower().replace(":", "").replace(" ", "")


def extract_value_text(text: str) -> str:
    for token in reversed(str(text).replace(":", " ").split()):
        if any(char.isdigit() for char in token):
            return token
    return str(text)


def build_option_label_crop_rect(line: TooltipLine, local_rect: Rect, text: str) -> dict[str, object]:
    mask = maple_text_mask(line.image)
    text_bbox = mask_bbox(mask) or Rect(0, 0, line.image.shape[1], line.image.shape[0])
    spans = column_word_spans(mask)
    value_text = extract_value_text(text)
    split = find_label_value_split(mask, spans, text_bbox, text, value_text)
    value_start = split["value_start"]
    label_right = split["label_right"]
    label_padding_left = 4
    label_padding_y = 3
    raw_left = max(0, text_bbox.left - label_padding_left)
    raw_top = max(0, text_bbox.top - label_padding_y)
    raw_right = min(line.image.shape[1], max(raw_left + 1, label_right))
    raw_bottom = min(line.image.shape[0], text_bbox.bottom + label_padding_y)
    raw_rect = Rect(line.rect.left + raw_left, line.rect.top + raw_top, line.rect.left + raw_right, line.rect.top + raw_bottom)

    trim_left = raw_left
    trim_right = raw_right
    contains_bullet = False
    if spans and spans[0][0] < raw_right:
        first_left, first_right = spans[0]
        first_mask = mask[:, first_left:first_right]
        first_width = first_right - first_left
        first_area = int(np.count_nonzero(first_mask))
        if first_width <= 7 and first_area <= 36 and len(spans) >= 2:
            trim_left = max(trim_left, spans[1][0] - 1)
            contains_bullet = True
    label_spans = [(left, right) for left, right in spans if left >= trim_left and right <= raw_right]
    if label_spans:
        last_left, last_right = label_spans[-1]
        last_mask = mask[:, last_left:last_right]
        if last_right - last_left <= 5 and int(np.count_nonzero(last_mask)) <= 28 and len(label_spans) >= 2:
            trim_right = min(trim_right, max(trim_left + 1, last_left - 1))
    trimmed_rect = Rect(
        line.rect.left + trim_left,
        line.rect.top + raw_top,
        line.rect.left + max(trim_left + 1, trim_right),
        line.rect.top + raw_bottom,
    )
    quality = option_label_crop_quality(
        mask,
        Rect(trim_left, raw_top, max(trim_left + 1, trim_right), raw_bottom),
        value_start=value_start,
        label_right=label_right,
        colon_x=split["colon_x"],
        contains_bullet=contains_bullet,
    )
    quality.update(
        {
            "split_reason": split["split_reason"],
            "split_colon_x": split["colon_x"],
            "split_value_start": split["value_start"],
            "split_label_right": split["label_right"],
            "text_bbox": rect_to_dict(
                Rect(
                    line.rect.left + text_bbox.left,
                    line.rect.top + text_bbox.top,
                    line.rect.left + text_bbox.right,
                    line.rect.top + text_bbox.bottom,
                )
            ),
        }
    )
    quality["template_rect"] = rect_to_dict(
        Rect(
            line.rect.left + local_rect.left,
            line.rect.top + local_rect.top,
            line.rect.left + local_rect.right,
            line.rect.top + local_rect.bottom,
        )
    )
    return {"raw_rect": raw_rect, "trimmed_rect": trimmed_rect, "quality": quality}


def build_option_value_crop_rect(line: TooltipLine, label_rect: Rect, value_text: str, line_text: str) -> dict[str, object]:
    mask = maple_text_mask(line.image)
    text_bbox = mask_bbox(mask) or Rect(0, 0, line.image.shape[1], line.image.shape[0])
    spans = column_word_spans(mask)
    local_label_right = max(0, label_rect.right - line.rect.left)
    split = find_label_value_split(mask, spans, text_bbox, line_text, value_text)
    value_start = split["value_start"]
    if value_start <= local_label_right and spans:
        value_start = first_span_after(spans, local_label_right) or max(local_label_right + 2, spans[-1][0])
    value_spans = [(left, right) for left, right in spans if right > value_start - 2]
    if value_spans:
        raw_left = max(0, min(left for left, _right in value_spans) - 2)
        raw_right = min(line.image.shape[1], max(right for _left, right in value_spans) + 3)
    else:
        raw_left = min(line.image.shape[1] - 1, max(local_label_right + 2, value_start - 2))
        raw_right = min(line.image.shape[1], max(raw_left + 1, text_bbox.right + 2))
    raw_top = max(0, text_bbox.top - 3)
    raw_bottom = min(line.image.shape[0], text_bbox.bottom + 3)
    raw_rect = Rect(line.rect.left + raw_left, line.rect.top + raw_top, line.rect.left + raw_right, line.rect.top + raw_bottom)
    trimmed_rect = raw_rect
    quality = option_value_crop_quality(
        mask,
        Rect(raw_left, raw_top, raw_right, raw_bottom),
        local_label_right=local_label_right,
        line_text=line_text,
        value_text=value_text,
        colon_x=split["colon_x"],
    )
    quality.update(
        {
            "split_reason": split["split_reason"],
            "split_colon_x": split["colon_x"],
            "split_value_start": split["value_start"],
            "split_label_right": split["label_right"],
        }
    )
    return {"raw_rect": raw_rect, "trimmed_rect": trimmed_rect, "quality": quality}


def mask_bbox(mask: np.ndarray) -> Rect | None:
    if mask.size == 0:
        return None
    rows = np.where((mask > 0).sum(axis=1) > 0)[0]
    cols = np.where((mask > 0).sum(axis=0) > 0)[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    return Rect(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def column_word_spans(mask: np.ndarray, gap_threshold: int = 5) -> list[tuple[int, int]]:
    cols = np.where((mask > 0).sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(cols[0])
    previous = int(cols[0])
    for col_value in cols[1:]:
        col = int(col_value)
        if col - previous > gap_threshold:
            spans.append((start, previous + 1))
            start = col
        previous = col
    spans.append((start, previous + 1))
    return spans


def find_value_start(spans: list[tuple[int, int]], text_bbox: Rect, value_text: str) -> int:
    if len(spans) >= 2 and any(char.isdigit() for char in value_text):
        return spans[-1][0]
    return text_bbox.right


def find_label_value_split(
    mask: np.ndarray,
    spans: list[tuple[int, int]],
    text_bbox: Rect,
    line_text: str,
    value_text: str,
) -> dict[str, object]:
    value_start = find_value_start(spans, text_bbox, value_text)
    colon_x = find_colon_candidate_x(mask, spans, value_start, line_text)
    if colon_x is not None:
        after_colon = first_span_after(spans, colon_x)
        if after_colon is not None:
            value_start = after_colon
    elif spans and any(char in str(value_text) for char in "+-") and len(spans) >= 2:
        value_start = spans[-1][0]
    split_boundary = min(value_start, colon_x if colon_x is not None else value_start)
    label_right = max(text_bbox.left + 1, split_boundary - 3)
    split_reason = "colon_value_split" if colon_x is not None else "value_start_split"
    if label_right <= text_bbox.left + 4:
        label_right = max(text_bbox.left + 1, value_start - 3)
        split_reason += "_fallback"
    return {
        "colon_x": colon_x,
        "value_start": value_start,
        "label_right": label_right,
        "split_reason": split_reason,
    }


def find_colon_candidate_x(
    mask: np.ndarray,
    spans: list[tuple[int, int]],
    value_start: int,
    line_text: str,
) -> int | None:
    candidates: list[tuple[int, int]] = []
    for left, right in spans:
        if right >= value_start:
            continue
        width = right - left
        if width > 5:
            continue
        span_mask = mask[:, left:right]
        rows = np.where((span_mask > 0).sum(axis=1) > 0)[0]
        height = int(rows[-1] - rows[0] + 1) if len(rows) else 0
        area = int(np.count_nonzero(span_mask))
        next_left = first_span_after(spans, right)
        previous_right = previous_span_right(spans, left)
        next_gap = next_left - right if next_left is not None else 999
        previous_gap = left - previous_right if previous_right is not None else 999
        if area <= 80 and height <= 18 and next_gap <= 14 and previous_gap >= 3:
            candidates.append((left, right))
    if candidates:
        return candidates[-1][0]
    if ":" in str(line_text) and len(spans) >= 3:
        return spans[-2][0]
    return None


def first_span_after(spans: list[tuple[int, int]], x: int) -> int | None:
    for left, right in spans:
        if left > x:
            return left
    return None


def previous_span_right(spans: list[tuple[int, int]], x: int) -> int | None:
    previous: int | None = None
    for left, right in spans:
        if right >= x:
            return previous
        previous = right
    return previous


def option_label_crop_quality(
    mask: np.ndarray,
    rect: Rect,
    value_start: int,
    label_right: int,
    colon_x: int | None,
    contains_bullet: bool,
) -> dict[str, object]:
    crop_mask = mask[rect.top : rect.bottom, rect.left : rect.right]
    foreground = crop_mask > 0
    touches_left = bool(foreground[:, :1].any()) if foreground.size else False
    touches_right = bool(foreground[:, -1:].any()) if foreground.size else False
    touches_top = bool(foreground[:1, :].any()) if foreground.size else False
    touches_bottom = bool(foreground[-1:, :].any()) if foreground.size else False
    foreground_ratio = float(np.count_nonzero(foreground)) / float(foreground.size) if foreground.size else 0.0
    contains_colon_like_text = colon_x is not None and rect.right > colon_x
    contains_value_like_text = rect.right > label_right or rect.right > value_start or contains_colon_like_text
    crop_width = rect.width
    crop_height = rect.height
    rejection_reason = ""
    if crop_width < 12 or crop_height < 8:
        rejection_reason = "label_crop_too_tight"
    elif touches_left or touches_right or touches_top or touches_bottom:
        rejection_reason = "label_crop_clipped"
    elif contains_value_like_text:
        rejection_reason = "option_label_contains_value"
    elif foreground_ratio <= 0.005:
        rejection_reason = "label_crop_too_tight"
    score = 1.0
    if rejection_reason:
        score = 0.25
    elif contains_bullet:
        score = 0.80
    return {
        "crop_width": crop_width,
        "crop_height": crop_height,
        "foreground_ratio": foreground_ratio,
        "touches_left_edge": touches_left,
        "touches_right_edge": touches_right,
        "touches_top_edge": touches_top,
        "touches_bottom_edge": touches_bottom,
        "was_truncated_by_max_width": False,
        "contains_value_like_text": contains_value_like_text,
        "contains_colon_like_text": contains_colon_like_text,
        "contains_leading_bullet": contains_bullet,
        "crop_quality_score": score,
        "rejection_reason": rejection_reason,
    }


def option_value_crop_quality(
    mask: np.ndarray,
    rect: Rect,
    local_label_right: int,
    line_text: str,
    value_text: str,
    colon_x: int | None,
) -> dict[str, object]:
    crop_mask = mask[rect.top : rect.bottom, rect.left : rect.right]
    foreground = crop_mask > 0
    foreground_ratio = float(np.count_nonzero(foreground)) / float(foreground.size) if foreground.size else 0.0
    touches_left = bool(foreground[:, :1].any()) if foreground.size else False
    touches_right = bool(foreground[:, -1:].any()) if foreground.size else False
    contains_label_text = rect.left <= local_label_right
    contains_colon_like_text = colon_x is not None and rect.left <= colon_x < rect.right
    has_value_digit = any(char.isdigit() for char in str(value_text))
    sign_without_digit = str(value_text).strip() in {"+", "-"} or not has_value_digit
    full_line_like = bool(line_text and value_text and len(str(line_text).strip()) > len(str(value_text).strip()) + 4 and rect.width > 90)
    rejection_reason = ""
    if rect.width < 6 or rect.height < 8:
        rejection_reason = "value_crop_too_tight"
    elif foreground_ratio <= 0.001:
        rejection_reason = "value_crop_too_tight"
    elif sign_without_digit:
        rejection_reason = "semantic_label_mismatch"
    elif contains_label_text or contains_colon_like_text or full_line_like:
        rejection_reason = "option_value_contains_label_text"
    return {
        "crop_width": rect.width,
        "crop_height": rect.height,
        "foreground_ratio": foreground_ratio,
        "touches_left_edge": touches_left,
        "touches_right_edge": touches_right,
        "contains_label_text": contains_label_text,
        "contains_colon_like_text": contains_colon_like_text,
        "value_sign_without_digit": sign_without_digit,
        "value_crop_full_line_like": full_line_like,
        "crop_quality_score": 0.25 if rejection_reason else 1.0,
        "rejection_reason": rejection_reason,
    }


def maple_text_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((0, 0), dtype="uint8")
    if len(crop.shape) == 2:
        gray = crop.astype("uint8", copy=False)
        if int(gray.max(initial=0)) <= 1:
            return (gray > 0).astype("uint8") * 255
        return (gray >= 24).astype("uint8") * 255
    b, g, r = cv2.split(crop)
    white = (r > 175) & (g > 175) & (b > 175)
    yellow = (r > 180) & (g > 70) & (b < 150)
    red = (r > 180) & (g < 110) & (b < 130)
    return (white | yellow | red).astype("uint8") * 255


def maple_foreground_text_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((0, 0), dtype="uint8")
    b, g, r = cv2.split(crop)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    local_background = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
    contrast = gray.astype("int16") - local_background.astype("int16")

    # Text seen through the translucent tooltip is dimmer and softer than the foreground text.
    # Keep very bright glyphs, or moderately bright glyphs only when they still have a crisp edge.
    bright_foreground = (gray >= 205) | ((gray >= 188) & (contrast >= 18))
    white = (r > 185) & (g > 185) & (b > 185) & bright_foreground
    yellow = (r > 205) & (g > 95) & (b < 155) & ((gray >= 150) | (contrast >= 18))
    red = (r > 205) & (g < 115) & (b < 145) & ((gray >= 80) | (contrast >= 16))
    return (white | yellow | red).astype("uint8") * 255


def build_diff_foreground_mask(
    before: np.ndarray,
    after: np.ndarray,
    tooltip: Rect,
    border_padding: int = 6,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    x1 = max(0, tooltip.left + border_padding)
    y1 = max(0, tooltip.top + border_padding)
    x2 = min(after.shape[1], tooltip.right - border_padding)
    y2 = min(after.shape[0], tooltip.bottom - border_padding)
    if x2 <= x1 or y2 <= y1:
        raise VisionError("Invalid tooltip ROI for diff foreground mask.")

    before_roi = before[y1:y2, x1:x2]
    after_roi = after[y1:y2, x1:x2]
    beta, offsets, predicted_background, residual = fit_translucent_background(before_roi, after_roi)
    threshold = calculate_auto_threshold(residual)
    binary_mask = (residual > threshold).astype("uint8") * 255
    binary_clean = remove_small_noise(binary_mask)
    ocr_mask = make_ocr_mask(binary_clean)
    text_mask = maple_text_mask(after_roi)
    final_mask = cv2.bitwise_and(text_mask, ocr_mask)
    foreground_text_mask = maple_foreground_text_mask(after_roi)
    final_mask = cv2.bitwise_or(final_mask, foreground_text_mask)
    final_mask = remove_small_noise(final_mask)

    residual_mask = np.clip(residual * 3.0, 0, 255).astype("uint8")
    analysis_binary = normalize_template_matching_image(residual_mask)
    whole_mask = np.zeros(after.shape[:2], dtype="uint8")
    whole_mask[y1:y2, x1:x2] = residual_mask
    whole_analysis_binary = np.zeros(after.shape[:2], dtype="uint8")
    whole_analysis_binary[y1:y2, x1:x2] = analysis_binary
    whole_foreground_text_mask = np.zeros(after.shape[:2], dtype="uint8")
    whole_foreground_text_mask[y1:y2, x1:x2] = foreground_text_mask
    whole_final_mask = np.zeros(after.shape[:2], dtype="uint8")
    whole_final_mask[y1:y2, x1:x2] = final_mask

    foreground_color = np.zeros_like(after_roi)
    foreground_color[binary_clean > 0] = after_roi[binary_clean > 0]
    foreground_color_final = np.zeros_like(after_roi)
    foreground_color_final[final_mask > 0] = after_roi[final_mask > 0]

    estimated_alpha = 1.0 - beta
    estimated_overlay = offsets / estimated_alpha if estimated_alpha > 1e-6 else offsets
    debug_images = {
        "before_roi": before_roi,
        "after_roi": after_roi,
        "predicted_background": np.clip(predicted_background, 0, 255).astype("uint8"),
        "residual": residual_mask,
        "residual_full": whole_mask,
        "analysis_binary": analysis_binary,
        "analysis_binary_full": whole_analysis_binary,
        "binary": binary_clean,
        "ocr_mask": ocr_mask,
        "text_mask": text_mask,
        "foreground_text_mask": foreground_text_mask,
        "foreground_text_mask_full": whole_foreground_text_mask,
        "final_mask": final_mask,
        "final_mask_full": whole_final_mask,
        "foreground_color": foreground_color,
        "foreground_color_final": foreground_color_final,
    }
    stats = {
        "tooltip_left": float(tooltip.left),
        "tooltip_top": float(tooltip.top),
        "tooltip_right": float(tooltip.right),
        "tooltip_bottom": float(tooltip.bottom),
        "beta": float(beta),
        "alpha": float(estimated_alpha),
        "overlay_b": float(estimated_overlay[0]),
        "overlay_g": float(estimated_overlay[1]),
        "overlay_r": float(estimated_overlay[2]),
        "threshold": float(threshold),
    }
    return whole_mask, debug_images, stats


def fit_translucent_background(
    before_roi: np.ndarray,
    after_roi: np.ndarray,
    keep_ratio: float = 0.70,
    iterations: int = 5,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    before_float = before_roi.astype("float32")
    after_float = after_roi.astype("float32")
    height, width = before_float.shape[:2]
    keep_mask = np.ones((height, width), dtype=bool)
    beta = 1.0
    offsets = np.zeros(3, dtype="float64")
    before_flat = before_float.reshape(-1, 3)
    after_flat = after_float.reshape(-1, 3)

    for _iteration in range(iterations):
        valid_indices = np.flatnonzero(keep_mask.ravel())
        if len(valid_indices) < 100:
            raise VisionError("Not enough pixels to fit translucent background.")
        xtx = np.zeros((4, 4), dtype="float64")
        xty = np.zeros(4, dtype="float64")
        for channel in range(3):
            before_channel = before_flat[valid_indices, channel].astype("float64")
            after_channel = after_flat[valid_indices, channel].astype("float64")
            xtx[0, 0] += np.dot(before_channel, before_channel)
            xtx[0, 1 + channel] += before_channel.sum()
            xtx[1 + channel, 0] += before_channel.sum()
            xtx[1 + channel, 1 + channel] += len(valid_indices)
            xty[0] += np.dot(before_channel, after_channel)
            xty[1 + channel] += after_channel.sum()
        theta = np.linalg.solve(xtx + np.eye(4) * 1e-8, xty)
        beta = float(theta[0])
        offsets = theta[1:]
        predicted_background = before_float * beta + offsets.reshape(1, 1, 3)
        residual = np.sqrt(np.mean((after_float - predicted_background) ** 2, axis=2))
        cutoff = np.quantile(residual, keep_ratio)
        keep_mask = residual <= cutoff

    predicted_background = before_float * beta + offsets.reshape(1, 1, 3)
    residual = np.sqrt(np.mean((after_float - predicted_background) ** 2, axis=2))
    return beta, offsets, predicted_background, residual


def calculate_auto_threshold(residual: np.ndarray) -> float:
    log_residual = np.log1p(residual)
    minimum = float(log_residual.min())
    maximum = float(log_residual.max())
    normalized = ((log_residual - minimum) / max(maximum - minimum, 1e-6) * 255).astype("uint8")
    otsu_value, _thresholded = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = np.expm1(minimum + (otsu_value / 255.0) * (maximum - minimum))
    return float(np.clip(threshold, 5.0, 12.0))


def remove_small_noise(binary_mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for index in range(1, count):
        _x, _y, _w, _h, area = stats[index]
        if area >= 2:
            cleaned[labels == index] = 255
    return cleaned


def make_ocr_mask(binary_mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)
    for index in range(1, count):
        _x, _y, width, height, area = stats[index]
        is_large_icon = area > 700
        is_horizontal_line = width > 100 and height <= 4
        if not is_large_icon and not is_horizontal_line:
            cleaned[labels == index] = 255
    return cleaned


def count_red_text_pixels(crop: np.ndarray) -> int:
    if crop.size == 0:
        return 0
    b, g, r = cv2.split(crop)
    return int(((r > 150) & (g < 130) & (b < 140)).sum())


def line_has_orange_text(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return False
    b, g, r = cv2.split(crop)
    orange_pixels = ((r > 165) & (g > 85) & (g < 190) & (b < 110)).sum()
    white_pixels = ((r > 175) & (g > 175) & (b > 175)).sum()
    return int(orange_pixels) >= 35 and int(orange_pixels) > int(white_pixels) // 3


def line_has_maple_text(crop: np.ndarray) -> bool:
    if crop.size == 0:
        return False
    mask = maple_text_mask(crop)
    return int((mask > 0).sum()) >= 45


def split_maple_characters(mask: np.ndarray) -> list[np.ndarray]:
    cols = np.where(mask.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return []
    groups: list[tuple[int, int]] = []
    start = int(cols[0])
    previous = int(cols[0])
    for column_value in cols[1:]:
        column = int(column_value)
        if column - previous > 2:
            groups.append((start, previous))
            start = column
        previous = column
    groups.append((start, previous))

    characters: list[np.ndarray] = []
    for left, right in groups:
        char = mask[:, max(0, left - 1) : min(mask.shape[1], right + 2)]
        rows = np.where(char.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        char = char[max(0, int(rows[0]) - 1) : min(char.shape[0], int(rows[-1]) + 2), :]
        characters.append(char)
    return characters


def normalize_digit_mask(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _threshold, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return cv2.resize(binary, (16, 28), interpolation=cv2.INTER_NEAREST)


def classify_maple_digit(char: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    normalized = normalize_digit_mask(char)
    scores: list[tuple[str, float]] = []
    for digit, digit_templates in templates.items():
        best_for_digit = 0.0
        for template in digit_templates:
            intersection = np.logical_and(normalized > 0, template > 0).sum()
            union = np.logical_or(normalized > 0, template > 0).sum()
            score = float(intersection / union) if union else 0.0
            if score > best_for_digit:
                best_for_digit = score
        scores.append((digit, best_for_digit))
    if not scores:
        return "", 0.0
    scores.sort(key=lambda item: item[1], reverse=True)
    best_digit, best_score = scores[0]
    score_map = dict(scores)
    if best_digit == "8":
        score_for_three = score_map.get("3", 0.0)
        if score_for_three >= best_score - 0.10 and looks_like_three(normalized):
            return "3", score_for_three
    return best_digit, best_score


def classify_maple_potential_digit(char: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    digit, score = classify_maple_digit(char, templates)
    normalized = normalize_digit_mask(char)
    if digit == "8":
        score_for_six = 0.0
        for template in templates.get("6", []):
            intersection = np.logical_and(normalized > 0, template > 0).sum()
            union = np.logical_or(normalized > 0, template > 0).sum()
            candidate_score = float(intersection / union) if union else 0.0
            score_for_six = max(score_for_six, candidate_score)
        area = int((char > 0).sum())
        if area < 85 and score_for_six >= score - 0.14:
            return "6", score_for_six
    if digit != "8":
        return digit, score
    score_for_three = 0.0
    for template in templates.get("3", []):
        intersection = np.logical_and(normalized > 0, template > 0).sum()
        union = np.logical_or(normalized > 0, template > 0).sum()
        candidate_score = float(intersection / union) if union else 0.0
        score_for_three = max(score_for_three, candidate_score)
    if score_for_three >= score - 0.10 and looks_like_three(char):
        return "3", score_for_three
    return digit, score


def looks_like_three(char: np.ndarray) -> bool:
    height, width = char.shape[:2]
    if height <= 0 or width <= 0:
        return False
    band_width = max(1, width // 3)
    left_density = float((char[:, :band_width] > 0).sum() / max(1, height * band_width))
    right_density = float((char[:, -band_width:] > 0).sum() / max(1, height * band_width))
    return right_density - left_density >= 0.18


def recognize_maple_digits(crop: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    mask = maple_text_mask(crop)
    digits: list[str] = []
    scores: list[float] = []
    for char in split_maple_characters(mask):
        height, width = char.shape[:2]
        area = int(char.sum() // 255)
        if height < 8 or area < 8 or width < 5 or width > 14:
            continue
        best_digit, best_score = classify_maple_digit(char, templates)
        if len(digits) >= 7 and width >= 10 and best_score < 0.45:
            break
        if best_digit and best_score >= 0.25:
            digits.append(best_digit)
            scores.append(best_score)
    return "".join(digits), min(scores, default=0.0)


def normalize_price_digits(digits: str) -> str:
    if len(digits) >= 9 and digits.endswith("91"):
        digits = digits[:-2]
    if len(digits) == 10 and digits.endswith("9") and len(set(digits[:-1])) == 1:
        digits = digits[:-1]
    if len(digits) == 10 and not digits.startswith("1"):
        digits = digits[:-1]
    return digits


def classify_price_digit(char: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    digit, score = classify_maple_digit(char, templates)
    if digit == "0" and looks_like_small_price_nine(char):
        return "9", max(score, 0.50)
    return digit, score


def looks_like_small_price_nine(char: np.ndarray) -> bool:
    if len(char.shape) == 3:
        gray = cv2.cvtColor(char, cv2.COLOR_BGR2GRAY)
    else:
        gray = char
    mask = gray > 0
    height, width = mask.shape[:2]
    if not (7 <= height <= 12 and 6 <= width <= 10):
        return False
    top = mask[: max(1, height // 2), :]
    bottom = mask[height // 2 :, :]
    top_density = float(top.sum() / max(1, top.size))
    bottom_density = float(bottom.sum() / max(1, bottom.size))
    return top_density >= 0.17 and bottom_density <= 0.18


def recognize_maple_digits_strict(crop: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    mask = maple_text_mask(crop)
    digits: list[str] = []
    scores: list[float] = []
    for left, _right, char in split_maple_character_groups(mask):
        height, width = char.shape[:2]
        area = int(char.sum() // 255)
        if height >= 12 and 3 <= width <= 4 and 10 <= area <= 25 and (digits or left >= 7):
            digits.append("1")
            scores.append(0.92)
            continue
        if height < 14 or area < 10 or width < 5 or width > 14:
            continue
        best_digit, best_score = classify_maple_digit(char, templates)
        if best_digit and best_score >= 0.38:
            digits.append(best_digit)
            scores.append(best_score)
    if len(digits) > 3:
        digits = digits[-3:]
        scores = scores[-3:]
    return "".join(digits), min(scores, default=0.0)


def read_signed_number_from_line(
    line: np.ndarray,
    label_rect: Rect,
    templates: dict[str, list[np.ndarray]],
) -> tuple[int | None, float]:
    right = crop_rect(
        line,
        Rect(
            min(line.shape[1], label_rect.right),
            0,
            min(line.shape[1], label_rect.right + 180),
            line.shape[0],
        ),
    )
    digits, confidence = recognize_maple_digits_strict(right, templates)
    if not digits:
        return None, 0.0
    try:
        value = int(digits)
    except ValueError:
        return None, confidence
    if line_has_minus_before_digits(right):
        value = -value
    return value, confidence


def read_signed_potential_number_from_line(
    line: np.ndarray,
    label_rect: Rect,
    templates: dict[str, list[np.ndarray]],
) -> tuple[int | None, float]:
    right = crop_rect(
        line,
        Rect(
            min(line.shape[1], label_rect.right),
            0,
            min(line.shape[1], label_rect.right + 220),
            line.shape[0],
        ),
    )
    digits, confidence = recognize_maple_potential_digits(right, templates)
    if not digits:
        return None, 0.0
    try:
        value = int(digits)
    except ValueError:
        return None, confidence
    if line_has_minus_before_digits(right):
        value = -value
    return value, confidence


def recognize_maple_potential_digits(crop: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    mask = maple_text_mask(crop)
    digits: list[str] = []
    scores: list[float] = []
    started = False
    for _left, _right, char in split_maple_character_groups(mask):
        height, width = char.shape[:2]
        area = int(char.sum() // 255)
        if height < 8 or area < 8:
            continue
        if height < 13:
            continue
        if not started and width <= 7 and height >= 12:
            best_digit, best_score = classify_maple_potential_digit(char, templates)
            if best_digit == "1" and best_score >= 0.60:
                digits.append("1")
                scores.append(best_score)
                started = True
                continue
            continue
        if width < 5 or width > 14:
            if started and len(digits) >= 1:
                break
            continue
        best_digit, best_score = classify_maple_potential_digit(char, templates)
        if best_digit and best_score >= 0.30:
            digits.append(best_digit)
            scores.append(best_score)
            started = True
            if len(digits) >= 3:
                break
        elif started:
            break
    return "".join(digits), min(scores, default=0.0)


def line_has_minus_before_digits(crop: np.ndarray) -> bool:
    mask = maple_text_mask(crop)
    groups = split_maple_character_groups(mask)
    for _left, _right, char in groups[:3]:
        height, width = char.shape[:2]
        area = int(char.sum() // 255)
        if height <= 4 and width >= 5 and area >= 5:
            return True
    return False


def line_has_percent_symbol(line: np.ndarray) -> bool:
    mask = maple_text_mask(line)
    for _left, _right, char in split_maple_character_groups(mask):
        height, width = char.shape[:2]
        area = int(char.sum() // 255)
        if height >= 12 and 8 <= width <= 24 and area >= 15:
            if width >= 11:
                return True
    return False


def format_signed_value(value: int) -> str:
    return f"+{value}" if value >= 0 else str(value)


def normalize_potential_value(key: str, value: int, has_percent: bool) -> int:
    return normalize_potential_value_with_trace(key, value, has_percent)[0]


def normalize_potential_value_with_trace(key: str, value: int, has_percent: bool) -> tuple[int, str]:
    sign = -1 if value < 0 else 1
    absolute = abs(value)
    if key in {"maxhp", "maxmp"}:
        if 100 < absolute < 200:
            corrected = sign * ((absolute - 100) * 10)
            return corrected, f"{key}: inferred hundreds ghost from {value}"
        text = str(absolute)
        if len(text) > 2 and text.startswith(("1", "2")):
            stripped = int(text[1:])
            if stripped:
                corrected = sign * stripped
                return corrected, f"{key}: stripped leading ghost digit from {value}"
        return value, ""
    if key in {"boss_damage", "ignore_defense"}:
        if 100 < absolute < 200:
            corrected = sign * (absolute - 100)
            return corrected, f"{key}: removed leading ghost 1 from {value}"
        if absolute == 80:
            corrected = sign * 30
            return corrected, f"{key}: legacy correction 80 -> 30 from {value}"
        if absolute == 5 and key == "ignore_defense":
            corrected = sign * 15
            return corrected, f"{key}: legacy correction 5 -> 15 from {value}"
    if key == "total_damage" and absolute == 5:
        corrected = sign * 6
        return corrected, f"{key}: legacy correction 5 -> 6 from {value}"
    if key in {"str", "int", "dex", "luk", "all_stat", "attack", "magic_attack"} and has_percent:
        if key == "all_stat" and absolute == 8:
            corrected = sign * 6
            return corrected, f"{key}: legacy correction 8 -> 6 from {value}"
        if key in {"str", "int", "dex", "luk", "attack", "magic_attack"} and absolute in {5, 8}:
            corrected = sign * 6
            return corrected, f"{key}: legacy correction {absolute} -> 6 from {value}"
        if key in {"attack", "magic_attack"} and absolute == 8:
            corrected = sign * 6
            return corrected, f"{key}: legacy correction 8 -> 6 from {value}"
        text = str(absolute)
        for prefix in ("1", "2"):
            if len(text) > 1 and text.startswith(prefix):
                stripped = int(text[1:])
                if key in {"str", "int", "dex", "luk", "attack", "magic_attack"} and stripped == 5:
                    stripped = 6
                if key in {"attack", "magic_attack"} and stripped == 8:
                    stripped = 6
                if stripped in {3, 4, 5, 6, 8, 9, 12}:
                    if key == "all_stat" and stripped == 8:
                        stripped = 6
                    corrected = sign * stripped
                    return corrected, f"{key}: stripped leading ghost digit from {value}"
        if absolute > 99:
            for candidate in (12, 9, 8, 6, 5, 4, 3):
                if str(candidate) in text:
                    corrected = sign * candidate
                    return corrected, f"{key}: selected embedded candidate {candidate} from {value}"
    return value, ""


def is_valid_price_digits(digits: str, confidence: float) -> bool:
    if not digits:
        return False
    length = len(digits)
    if length >= 6:
        return confidence >= 0.35
    if length == 5:
        return confidence >= 0.45
    if length == 4:
        return confidence >= 0.55
    return False


def is_valid_selected_price_digits(digits: str, confidence: float) -> bool:
    if not digits:
        return False
    length = len(digits)
    if length >= 8:
        return confidence >= 0.28
    return is_valid_price_digits(digits, confidence)


def recognize_req_level_digits(crop: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    mask = maple_text_mask(crop)
    digits: list[str] = []
    scores: list[float] = []
    for _left, _right, char in split_maple_character_groups(mask):
        height, width = char.shape[:2]
        area = int(char.sum() // 255)
        if height < 8 or area < 8:
            continue
        if width <= 8 and height >= 12 and area <= 45:
            digits.append("1")
            scores.append(0.92)
            continue
        if width < 5 or width > 14:
            continue
        best_digit, best_score = classify_maple_digit(char, templates)
        if best_digit and best_score >= 0.25:
            digits.append(best_digit)
            scores.append(best_score)
    return normalize_req_level_digits("".join(digits)), min(scores, default=0.0)


def normalize_req_level_digits(digits: str) -> str:
    if digits == "119":
        return "110"
    if digits == "58":
        return "98"
    if len(digits) == 3 and digits.startswith("11") and digits[-1] == "9":
        return "110"
    return digits


def read_req_level_from_red_requirement_block(
    line: np.ndarray,
    templates: dict[str, list[np.ndarray]],
) -> tuple[int | None, float, str]:
    if line.size == 0 or line.shape[0] < 30:
        return None, 0.0, ""
    b, g, r = cv2.split(line)
    red_mask = ((r > 145) & (g < 95) & (b < 110)).astype("uint8") * 255
    row_counts = (red_mask > 0).sum(axis=1)
    active_rows = np.where(row_counts > 5)[0]
    if len(active_rows) == 0:
        return None, 0.0, ""
    spans: list[tuple[int, int]] = []
    start = int(active_rows[0])
    previous = int(active_rows[0])
    for row_value in active_rows[1:]:
        row = int(row_value)
        if row - previous > 2:
            spans.append((start, previous))
            start = row
        previous = row
    spans.append((start, previous))

    top, bottom = spans[0]
    first_red_row = red_mask[max(0, top - 1) : min(red_mask.shape[0], bottom + 2), :]
    digit_start_x = int(first_red_row.shape[1] * 0.76)
    digits: list[str] = []
    scores: list[float] = []
    for left, _right, char in split_maple_character_groups(first_red_row):
        if left < digit_start_x:
            continue
        height, width = char.shape[:2]
        area = int(char.sum() // 255)
        if height < 8 or area < 8:
            continue
        if width <= 6 and area <= 24:
            digits.append("1")
            scores.append(0.92)
            continue
        if width < 4 or width > 14:
            continue
        digit, score = classify_maple_digit(char, templates)
        if digit and score >= 0.18:
            digits.append(digit)
            scores.append(score)
    if not digits:
        return None, 0.0, ""
    raw = "".join(digits[-3:])
    try:
        value = int(raw)
    except ValueError:
        return None, min(scores, default=0.0), raw
    if 10 <= value <= 200:
        return value, min(scores, default=0.0), raw
    return None, min(scores, default=0.0), raw


def recognize_maple_price_digits(crop: np.ndarray, templates: dict[str, list[np.ndarray]]) -> tuple[str, float]:
    crop = crop_price_text_band(crop)
    mask = maple_text_mask(crop)
    groups = split_maple_character_groups(mask)
    digits: list[str] = []
    scores: list[float] = []
    previous_digit_right: int | None = None
    separator_since_digit = False
    digits_since_separator = 0

    for left, right, char in groups:
        height, width = char.shape[:2]
        area = int(char.sum() // 255)
        if height < 8 or area < 8:
            if height < 8 and area >= 1:
                separator_since_digit = True
                digits_since_separator = 0
            continue
        if len(digits) >= 10:
            break
        if 2 <= width <= 4 and height >= 10 and 8 <= area <= 30:
            if previous_digit_right is not None and left - previous_digit_right > 18 and not separator_since_digit:
                break
            best_digit, best_score = classify_price_digit(char, templates)
            if best_digit == "1" and best_score >= 0.45:
                digits.append("1")
                scores.append(best_score)
                previous_digit_right = right
                separator_since_digit = False
                digits_since_separator += 1
                continue
        if width < 5 or width > 14:
            if previous_digit_right is not None and left - previous_digit_right > 18 and not separator_since_digit:
                break
            continue
        if previous_digit_right is not None and left - previous_digit_right > 18 and not separator_since_digit:
            break
        best_digit, best_score = classify_price_digit(char, templates)
        if len(digits) >= 7 and width >= 10 and best_score < 0.45:
            break
        if best_digit and best_score >= 0.25:
            digits.append(best_digit)
            scores.append(best_score)
            previous_digit_right = right
            separator_since_digit = False
            digits_since_separator += 1
    return normalize_price_digits("".join(digits)), min(scores, default=0.0)


def crop_price_text_band(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0 or crop.shape[0] <= 28:
        return crop
    mask = maple_text_mask(crop)
    row_counts = (mask.sum(axis=1) // 255).astype(np.int32)
    if int(row_counts.max(initial=0)) <= 0:
        return crop
    smoothed = np.convolve(row_counts, np.ones(5, dtype=np.int32), mode="same")
    center = int(np.argmax(smoothed))
    half_height = 11
    top = max(0, center - half_height)
    bottom = min(crop.shape[0], center + half_height + 1)
    if bottom - top < 12:
        return crop
    return crop[top:bottom, :]


def price_color_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros(crop.shape[:2], dtype="uint8")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0, sigmaY=2.0)
    contrast = gray.astype("int16") - background.astype("int16")
    h, s, v = cv2.split(hsv)
    l_channel, a_channel, b_channel = cv2.split(lab)
    bright = (v >= 135) & (l_channel >= 130)
    yellow = (h >= 15) & (h <= 45) & (s >= 35) & bright
    green_cyan = (h >= 45) & (h <= 105) & (s >= 25) & bright
    bright_white = (s <= 90) & (v >= 185) & (l_channel >= 175)
    lab_green_or_yellow = ((a_channel <= 150) | (b_channel >= 135)) & bright
    crisp_text = (v >= 95) & (l_channel >= 85) & (contrast >= 14)
    mask = (yellow | green_cyan | bright_white | lab_green_or_yellow | crisp_text).astype("uint8") * 255
    mask = cv2.medianBlur(mask, 3)
    kernel = np.ones((2, 2), dtype="uint8")
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def price_text_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    if mask.size == 0:
        return []
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype("uint8"), 8)
    components: list[tuple[int, int, int, int, int]] = []
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        w = int(stats[index, cv2.CC_STAT_WIDTH])
        h = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if h < 7 or h > 28 or w < 1 or w > 22 or area < 4:
            continue
        if area / float(max(1, w * h)) < 0.10:
            continue
        components.append((x, y, w, h, area))
    return sorted(components, key=lambda item: (item[1], item[0]))


def price_mask_row_candidates(mask: np.ndarray) -> list[dict[str, float]]:
    if mask.size == 0:
        return []
    row_counts = (mask > 0).sum(axis=1)
    active = np.where(row_counts >= 2)[0]
    if len(active) == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(active[0])
    previous = int(active[0])
    for row_value in active[1:]:
        row = int(row_value)
        if row - previous > 3:
            spans.append((start, previous))
            start = row
        previous = row
    spans.append((start, previous))

    candidates: list[dict[str, float]] = []
    for top, bottom in spans:
        top = max(0, top - 3)
        bottom = min(mask.shape[0], bottom + 4)
        if bottom - top < 8:
            continue
        row_mask = mask[top:bottom, :]
        cols = np.where((row_mask > 0).sum(axis=0) > 0)[0]
        if len(cols) == 0:
            continue
        col_spans: list[tuple[int, int]] = []
        col_start = int(cols[0])
        previous_col = int(cols[0])
        for col_value in cols[1:]:
            col = int(col_value)
            if col - previous_col > 16:
                col_spans.append((col_start, previous_col))
                col_start = col
            previous_col = col
        col_spans.append((col_start, previous_col))
        for span_left, span_right in col_spans:
            left = max(0, span_left - 4)
            right = min(mask.shape[1], span_right + 5)
            if right - left < 24:
                continue
            groups = split_maple_character_groups(row_mask[:, left:right])
            component_count = max(len(groups), len(price_text_components(row_mask[:, left:right])))
            candidates.append(
                {
                    "left": float(left),
                    "top": float(top),
                    "right": float(right),
                    "bottom": float(bottom),
                    "width": float(right - left),
                    "center_y": float((top + bottom) / 2.0),
                    "component_count": float(component_count),
                }
            )
    return candidates


def group_price_components_by_row(components: list[tuple[int, int, int, int, int]]) -> list[list[tuple[int, int, int, int, int]]]:
    rows: list[list[tuple[int, int, int, int, int]]] = []
    for component in components:
        center_y = component[1] + component[3] / 2.0
        placed = False
        for row in rows:
            if abs(component_row_center_y(row) - center_y) <= 10:
                row.append(component)
                placed = True
                break
        if not placed:
            rows.append([component])
    for row in rows:
        row.sort(key=lambda item: item[0])
    return rows


def component_row_center_y(row: list[tuple[int, int, int, int, int]]) -> float:
    if not row:
        return 0.0
    weighted = sum((y + h / 2.0) * area for _x, y, _w, h, area in row)
    total = sum(area for _x, _y, _w, _h, area in row)
    return weighted / float(max(1, total))


def component_row_width(row: list[tuple[int, int, int, int, int]]) -> int:
    if not row:
        return 0
    left = min(x for x, _y, _w, _h, _area in row)
    right = max(x + w for x, _y, w, _h, _area in row)
    return right - left


def component_bbox(
    row: list[tuple[int, int, int, int, int]],
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left = max(0, min(x for x, _y, _w, _h, _area in row) - padding)
    top = max(0, min(y for _x, y, _w, _h, _area in row) - padding)
    right = min(width, max(x + w for x, _y, w, _h, _area in row) + padding)
    bottom = min(height, max(y + h for _x, y, _w, h, _area in row) + padding)
    return left, top, right, bottom


def price_crop_quality_score(
    width: int,
    height: int,
    foreground_ratio: float,
    component_count: int,
    multiple_rows: bool,
) -> float:
    width_score = min(1.0, max(0.0, width / 120.0))
    height_score = 1.0 if 12 <= height <= 34 else 0.4
    ratio_score = 1.0 if 0.01 <= foreground_ratio <= 0.45 else 0.35
    component_score = min(1.0, component_count / 8.0)
    penalty = 0.35 if multiple_rows else 0.0
    return max(0.0, min(1.0, (width_score + height_score + ratio_score + component_score) / 4.0 - penalty))


def split_maple_character_groups(mask: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    cols = np.where(mask.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return []
    spans: list[tuple[int, int]] = []
    start = int(cols[0])
    previous = int(cols[0])
    for column_value in cols[1:]:
        column = int(column_value)
        if column - previous > 2:
            spans.append((start, previous))
            start = column
        previous = column
    spans.append((start, previous))

    groups: list[tuple[int, int, np.ndarray]] = []
    for left, right in spans:
        char = mask[:, max(0, left - 1) : min(mask.shape[1], right + 2)]
        rows = np.where(char.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        char = char[max(0, int(rows[0]) - 1) : min(char.shape[0], int(rows[-1]) + 2), :]
        groups.append((left, right, char))
    return groups


def read_image(path: Path, flags: int) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def prepare_ocr_roi(roi: np.ndarray) -> np.ndarray:
    if roi.size == 0:
        return roi
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    scaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    return cv2.equalizeHist(scaled)
