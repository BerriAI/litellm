import logging

import pytest

from litellm.litellm_core_utils.env_utils import get_env_float, get_env_int


@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_blank_numeric_env_value_means_unset(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("WF_TEST_FLOAT", raw)
    monkeypatch.setenv("WF_TEST_INT", raw)

    assert get_env_float("WF_TEST_FLOAT", 1.5) == 1.5
    assert get_env_int("WF_TEST_INT", 7) == 7


def test_set_numeric_env_value_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WF_TEST_FLOAT", " 2.5 ")
    monkeypatch.setenv("WF_TEST_INT", " 12 ")

    assert get_env_float("WF_TEST_FLOAT", 1.5) == 2.5
    assert get_env_int("WF_TEST_INT", 7) == 12


def test_unparsable_numeric_env_value_falls_back_and_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("WF_TEST_FLOAT", "fast")
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            assert get_env_float("WF_TEST_FLOAT", 1.5) == 1.5
    finally:
        litellm_logger.removeHandler(caplog.handler)

    assert "WF_TEST_FLOAT" in caplog.text
    assert "fast" in caplog.text
