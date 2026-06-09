"""Tests for spend-log flush retry configuration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.constants import (
    DEFAULT_SPEND_LOG_FLUSH_MAX_RETRIES,
    SPEND_LOG_FLUSH_MAX_RETRIES,
)
from litellm.proxy.utils import get_spend_log_flush_max_retries, update_spend_logs_job


def test_get_spend_log_flush_max_retries_defaults_to_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(proxy_server_mod, "general_settings", {})
    assert get_spend_log_flush_max_retries() == SPEND_LOG_FLUSH_MAX_RETRIES


def test_get_spend_log_flush_max_retries_general_settings_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.constants as constants_mod
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(constants_mod, "SPEND_LOG_FLUSH_MAX_RETRIES", 5)
    monkeypatch.setattr(
        proxy_server_mod,
        "general_settings",
        {"spend_log_flush_max_retries": 7},
    )
    assert get_spend_log_flush_max_retries() == 7


def test_get_spend_log_flush_max_retries_clamps_negative_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(
        proxy_server_mod,
        "general_settings",
        {"spend_log_flush_max_retries": -2},
    )
    assert get_spend_log_flush_max_retries() == 0


def test_get_spend_log_flush_max_retries_env_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.constants as constants_mod
    import litellm.proxy.proxy_server as proxy_server_mod

    monkeypatch.setattr(constants_mod, "SPEND_LOG_FLUSH_MAX_RETRIES", 5)
    monkeypatch.setattr(proxy_server_mod, "general_settings", {})
    assert get_spend_log_flush_max_retries() == 5


@pytest.mark.asyncio
async def test_update_spend_logs_job_uses_configured_max_retries(
    mock_prisma_client: object,
    make_spend_log_row: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm.proxy.guardrails.usage_tracking as guard_mod
    import litellm.proxy.db.spend_log_tool_index as tool_mod
    import litellm.proxy.utils as utils_mod

    monkeypatch.setattr(utils_mod, "get_spend_log_flush_max_retries", lambda: 2)

    captured: dict[str, int] = {}

    async def _capture_update_spend_logs(**kwargs: object) -> None:
        captured["n_retry_times"] = kwargs["n_retry_times"]  # type: ignore[index]

    monkeypatch.setattr(
        utils_mod.ProxyUpdateSpend,
        "update_spend_logs",
        staticmethod(_capture_update_spend_logs),
    )
    monkeypatch.setattr(
        guard_mod, "process_spend_logs_guardrail_usage", AsyncMock(), raising=False
    )
    monkeypatch.setattr(
        tool_mod, "process_spend_logs_tool_usage", AsyncMock(), raising=False
    )

    mock_prisma_client.spend_log_transactions = [  # type: ignore[attr-defined]
        make_spend_log_row(request_id="r1")  # type: ignore[operator]
    ]
    proxy_logging = MagicMock()

    await update_spend_logs_job(
        prisma_client=mock_prisma_client,  # type: ignore[arg-type]
        db_writer_client=None,
        proxy_logging_obj=proxy_logging,
    )
    assert captured["n_retry_times"] == 2
