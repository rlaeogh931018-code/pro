from maple_price_tool.config import load_config, save_config


def test_load_config_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
window_title_keyword: "Game"
database_path: "db/test.sqlite3"
capture:
  left: 10
  right: 20
  up: 30
  down: 40
vision:
  option_threshold: 0.5
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.window_title_keyword == "Game"
    assert str(config.database_path) == "db\\test.sqlite3" or str(config.database_path) == "db/test.sqlite3"
    assert config.capture.left == 10
    assert config.capture.down == 40
    assert config.vision.option_threshold == 0.5
    assert config.vision.alignment_enabled is True
    assert config.vision.alignment_max_shift == 3.0
    assert config.vision.enable_easyocr_fallback is False


def test_load_config_defaults_when_file_missing(tmp_path):
    config = load_config(tmp_path / "missing.yaml")

    assert config.vision.alignment_enabled is True
    assert config.vision.alignment_min_response == 0.20
    assert config.vision.device == "auto"
    assert config.vision.template_fallback_enabled is True
    assert config.vision.enable_easyocr_fallback is False


def test_save_config_roundtrip(tmp_path):
    config = load_config()
    config.capture.left = 123
    config.capture.right = 456
    config_path = tmp_path / "config.yaml"

    save_config(config, config_path)
    loaded = load_config(config_path)

    assert loaded.capture.left == 123
    assert loaded.capture.right == 456
