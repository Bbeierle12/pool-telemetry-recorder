from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6 import QtWidgets

from .config import load_config, save_config
from .db import get_db_path, init_db
from .exporter import ExportManager
from .gemini.processor import EventProcessor
from .sessions import SessionManager
from .ui.main_window import MainWindow


def _configure_logging(data_dir: str) -> None:
    log_dir = Path(data_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> int:
    config = load_config()
    save_config(config)
    _configure_logging(config.storage.data_directory)
    init_db(config).close()
    db_path = get_db_path(config)
    event_processor = EventProcessor(str(db_path))
    session_manager = SessionManager(db_path)
    export_manager = ExportManager(db_path)

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(config, event_processor, session_manager, export_manager)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
