"""Timing utilities used to record how long pipeline stages take."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar

from ..logging_utils import get_logger

F = TypeVar("F", bound=Callable[..., Any])
logger = get_logger(__name__)


class Stopwatch:
    """Context manager that measures elapsed wall-clock seconds."""

    def __init__(self) -> None:
        self.started_at = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> "Stopwatch":
        self.started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self.started_at


def timed(func: F) -> F:
    """Decorator that logs the runtime of a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with Stopwatch() as watch:
            result = func(*args, **kwargs)
        logger.info("%s finished in %.3fs", func.__name__, watch.elapsed)
        return result

    return wrapper  # type: ignore[return-value]
