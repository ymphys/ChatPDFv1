from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOG_NAME = "chatpdf"
DEFAULT_LOG_FILENAME = "chatpdf.log"


def configure_logging(
    *,
    app_name: str = DEFAULT_LOG_NAME,
    base_dir: Path | None = None,
) -> logging.Logger:
    """
    Configure application logging with both console and file handlers.
    Subsequent calls return the already-configured logger.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent

    log_path = base_dir / DEFAULT_LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(app_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


__all__ = ["configure_logging"]
