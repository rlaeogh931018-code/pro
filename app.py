from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from maple_price_tool.config import load_config
from maple_price_tool.storage import Storage
from maple_price_tool.ui import run_app


def setup_logging(log_path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"),
        logging.StreamHandler(),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def main() -> int:
    config = load_config()
    setup_logging(config.log_path)
    storage = Storage(config.database_path)
    return run_app(config, storage)


if __name__ == "__main__":
    raise SystemExit(main())
