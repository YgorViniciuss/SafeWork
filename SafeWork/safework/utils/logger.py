"""Configuração de logging: console e arquivo rotativo em logs/safework.log."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from safework.core.config import PROJECT_ROOT

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configura o logger raiz do pacote. Pode ser chamada mais de uma vez."""
    logger = logging.getLogger("safework")
    if logger.handlers:
        return

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler()
    file_handler = RotatingFileHandler(
        log_dir / "safework.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    for handler in (console, file_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
