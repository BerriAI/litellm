import logging
from collections.abc import Callable, Iterator
from typing import Final

import pytest

from litellm._logging import verbose_proxy_logger


@pytest.fixture
def warning_messages() -> Iterator[Callable[[str], list[str]]]:
    records: Final[list[logging.LogRecord]] = []  # mutable-ok: the handler appends each record
    handler: Final = logging.Handler(level=logging.WARNING)
    handler.emit = records.append
    previous_level: Final = verbose_proxy_logger.level
    verbose_proxy_logger.addHandler(handler)
    verbose_proxy_logger.setLevel(logging.WARNING)

    def containing(needle: str) -> list[str]:
        return [message for message in (record.getMessage() for record in records) if needle in message]

    yield containing
    verbose_proxy_logger.removeHandler(handler)
    verbose_proxy_logger.setLevel(previous_level)
