"""Tests for Vantage end-user export configuration."""

import pytest

from litellm.integrations.vantage.vantage_logger import VantageLogger


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes"])
def test_vantage_end_user_export_can_be_enabled_by_environment(monkeypatch: pytest.MonkeyPatch, value: str):
    monkeypatch.setenv("VANTAGE_INCLUDE_END_USER", value)

    logger = VantageLogger(api_key="test-key", integration_token="test-token")

    assert logger.include_end_user is True
    assert logger._ensure_engine()._database.include_end_user is True


def test_vantage_end_user_export_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VANTAGE_INCLUDE_END_USER", raising=False)

    logger = VantageLogger(api_key="test-key", integration_token="test-token")

    assert logger.include_end_user is False
    assert logger._ensure_engine()._database.include_end_user is False
