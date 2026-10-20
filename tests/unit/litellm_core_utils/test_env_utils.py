import logging
from collections.abc import Callable, Iterator

import pytest

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.env_utils import (
    get_env_float,
    get_env_float_or_none,
    get_env_int,
    get_env_int_or_none,
)


@pytest.fixture
def litellm_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    verbose_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=verbose_logger.name):
            yield caplog
    finally:
        verbose_logger.removeHandler(caplog.handler)


def _logged_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in dict.fromkeys(caplog.records)]


@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_blank_numeric_env_value_means_unset_without_warning(
    monkeypatch: pytest.MonkeyPatch, litellm_log: pytest.LogCaptureFixture, raw: str
) -> None:
    monkeypatch.setenv("ENV_UTILS_TEST_KNOB", raw)

    assert get_env_float("ENV_UTILS_TEST_KNOB", 1.5) == 1.5
    assert get_env_int("ENV_UTILS_TEST_KNOB", 7) == 7
    assert get_env_float_or_none("ENV_UTILS_TEST_KNOB") is None
    assert get_env_int_or_none("ENV_UTILS_TEST_KNOB") is None
    assert _logged_messages(litellm_log) == []


def test_set_numeric_env_value_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENV_UTILS_TEST_FLOAT", " 2.5 ")
    monkeypatch.setenv("ENV_UTILS_TEST_INT", " 12 ")

    assert get_env_float("ENV_UTILS_TEST_FLOAT", 1.5) == 2.5
    assert get_env_int("ENV_UTILS_TEST_INT", 7) == 12
    assert get_env_float_or_none("ENV_UTILS_TEST_FLOAT") == 2.5
    assert get_env_int_or_none("ENV_UTILS_TEST_INT") == 12


@pytest.mark.parametrize(
    ("read", "raw", "expected"),
    [
        (lambda name: get_env_int(name, 7), "fast-int", 7),
        (lambda name: get_env_float(name, 1.5), "fast-float", 1.5),
        (get_env_int_or_none, "fast-int-or-none", None),
        (get_env_float_or_none, "fast-float-or-none", None),
    ],
)
def test_unparsable_numeric_env_value_falls_back_and_warns_once(
    monkeypatch: pytest.MonkeyPatch,
    litellm_log: pytest.LogCaptureFixture,
    read: Callable[[str], float | None],
    raw: str,
    expected: float | None,
) -> None:
    monkeypatch.setenv("ENV_UTILS_TEST_KNOB", raw)

    assert read("ENV_UTILS_TEST_KNOB") == expected
    assert read("ENV_UTILS_TEST_KNOB") == expected

    warnings = [message for message in _logged_messages(litellm_log) if raw in message]
    assert len(warnings) == 1
    assert "ENV_UTILS_TEST_KNOB" in warnings[0]


@pytest.mark.parametrize("env_var", ["MAX_STRING_LENGTH_STDOUT_LOG", "MAX_BASE64_LENGTH_STDOUT_LOG"])
def test_unparsable_stdout_log_limit_does_not_break_logging(
    monkeypatch: pytest.MonkeyPatch, litellm_log: pytest.LogCaptureFixture, env_var: str
) -> None:
    """
    Regression: the LiteLLM loggers' own filter reads these limits through get_env_int on every
    record, so warning about a garbage value re-entered the filter and every log call raised
    RecursionError.
    """
    raw = f"not-a-number-{env_var}"
    monkeypatch.setenv(env_var, raw)

    verbose_logger.warning("first line")
    verbose_logger.warning("second line")

    messages = _logged_messages(litellm_log)
    assert "first line" in messages
    assert "second line" in messages
    assert len([message for message in messages if raw in message]) == 1
