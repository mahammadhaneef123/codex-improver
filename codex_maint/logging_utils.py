"""Logging helpers shared by the CLI and library code."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure(level: str = "INFO") -> None:
    """Configure root logging exactly once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(name)
