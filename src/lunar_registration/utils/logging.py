"""
Structured logging for the SelenoFuse pipeline.

Provides configurable logging with colored console output
and optional file logging for reproducible experiment tracking.
"""

import logging
import sys
from pathlib import Path


# ANSI color codes
_COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[1;31m", # Bold Red
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to log levels."""

    def format(self, record):
        levelname = record.levelname
        if levelname in _COLORS:
            record.levelname = (
                f"{_COLORS[levelname]}{levelname}{_COLORS['RESET']}"
            )
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    fmt: str | None = None,
    name: str = "selenofuse",
) -> logging.Logger:
    """
    Set up structured logging for the pipeline.

    Parameters
    ----------
    level : str
        Logging level: DEBUG, INFO, WARNING, ERROR.
    log_file : str, Path, or None
        If provided, also log to this file.
    fmt : str or None
        Log format string. Uses a sensible default if None.
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(ColoredFormatter(fmt))
    logger.addHandler(console)

    # File handler (plain)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "selenofuse") -> logging.Logger:
    """
    Get an existing logger or create a basic one.

    Parameters
    ----------
    name : str
        Logger name (typically module path).

    Returns
    -------
    logging.Logger
        Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        setup_logging(name=name)

    return logger
