import importlib
import logging
from collections.abc import Iterator
from typing import Final

import pytest

import litellm_proxy_extras._logging as extras_logging


@pytest.fixture
def fresh_extras_logger() -> Iterator[logging.Logger]:
    logger = logging.getLogger("litellm_proxy_extras")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    logger.handlers[:] = []
    try:
        yield logger
    finally:
        logger.handlers[:] = saved_handlers
        logger.setLevel(saved_level)


@pytest.mark.parametrize("litellm_log", ["ERROR", " ERROR ", "\terror\n"])
def test_litellm_log_error_silences_extras_info_lines(
    monkeypatch: pytest.MonkeyPatch, fresh_extras_logger: logging.Logger, litellm_log: str
) -> None:
    monkeypatch.setenv("LITELLM_LOG", litellm_log)
    reloaded: Final = importlib.reload(extras_logging).logger
    assert reloaded is fresh_extras_logger
    assert reloaded.level == logging.ERROR
    assert reloaded.isEnabledFor(logging.INFO) is False
    assert reloaded.isEnabledFor(logging.ERROR) is True


@pytest.mark.parametrize(
    ("litellm_log", "expected_level"),
    [
        (None, logging.INFO),
        ("info", logging.INFO),
        ("DEBUG", logging.DEBUG),
        ("DEBUG ", logging.DEBUG),
        ("\tInFo\n", logging.INFO),
    ],
)
def test_unset_or_verbose_litellm_log_keeps_extras_info_lines(
    monkeypatch: pytest.MonkeyPatch,
    fresh_extras_logger: logging.Logger,
    litellm_log: str | None,
    expected_level: int,
) -> None:
    if litellm_log is None:
        monkeypatch.delenv("LITELLM_LOG", raising=False)
    else:
        monkeypatch.setenv("LITELLM_LOG", litellm_log)
    reloaded: Final = importlib.reload(extras_logging).logger
    assert reloaded is fresh_extras_logger
    assert reloaded.level == expected_level
    assert reloaded.isEnabledFor(logging.INFO) is True


@pytest.mark.parametrize("litellm_log", ["", " ", " DE BUG ", "INVALID", "10"])
def test_invalid_litellm_log_is_rejected_by_extras(
    monkeypatch: pytest.MonkeyPatch, fresh_extras_logger: logging.Logger, litellm_log: str
) -> None:
    monkeypatch.setenv("LITELLM_LOG", litellm_log)
    with pytest.raises(ValueError, match="Unknown level"):
        importlib.reload(extras_logging)
