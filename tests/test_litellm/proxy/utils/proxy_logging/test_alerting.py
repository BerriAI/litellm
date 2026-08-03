"""Pin alerting helpers on ``ProxyLogging``.

Covers ``failed_tracking_alert``, ``budget_alerts``, ``alerting_handler``,
``failure_handler``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy._types import AlertType, CallInfo


# ---------------------------------------------------------------------------
# failed_tracking_alert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_tracking_alert_no_op_when_alerting_none(proxy_logging):
    proxy_logging.alerting = None
    proxy_logging.slack_alerting_instance = MagicMock(failed_tracking_alert=AsyncMock())
    await proxy_logging.failed_tracking_alert(error_message="x", failing_model="m")
    proxy_logging.slack_alerting_instance.failed_tracking_alert.assert_not_called()


@pytest.mark.asyncio
async def test_failed_tracking_alert_forwards_to_slack(proxy_logging):
    proxy_logging.alerting = ["slack"]
    captured: Dict[str, Any] = {}

    async def fake_alert(**kwargs):
        captured.update(kwargs)

    proxy_logging.slack_alerting_instance = MagicMock(failed_tracking_alert=fake_alert)
    await proxy_logging.failed_tracking_alert(error_message="db down", failing_model="gpt-4")
    snapshot = {
        "error_message": captured["error_message"],
        "failing_model": captured["failing_model"],
        "captured_keys": sorted(captured.keys()),
    }
    assert snapshot == {
        "error_message": "db down",
        "failing_model": "gpt-4",
        "captured_keys": ["error_message", "failing_model"],
    }


@pytest.mark.asyncio
async def test_failed_tracking_alert_slack_error_raises(proxy_logging):
    proxy_logging.alerting = ["slack"]
    proxy_logging.slack_alerting_instance = MagicMock(
        failed_tracking_alert=AsyncMock(side_effect=RuntimeError("slack down"))
    )
    with pytest.raises(RuntimeError):
        await proxy_logging.failed_tracking_alert(error_message="x", failing_model="m")


# ---------------------------------------------------------------------------
# budget_alerts
# ---------------------------------------------------------------------------


def _user_info(alert_emails=None):
    return CallInfo(
        spend=0.0,
        max_budget=1.0,
        token="tok",
        user_id="u1",
        team_id="t1",
        team_alias=None,
        user_email=None,
        key_alias=None,
        projected_exceeded_date=None,
        projected_spend=None,
        event_group="user",
        event="threshold_crossed",
        alert_emails=alert_emails,
    )


@pytest.mark.asyncio
async def test_budget_alerts_no_op_when_alerting_off_and_no_emails(proxy_logging):
    proxy_logging.alerting = None
    proxy_logging.slack_alerting_instance = MagicMock(budget_alerts=AsyncMock())
    proxy_logging.email_logging_instance = MagicMock(budget_alerts=AsyncMock())
    await proxy_logging.budget_alerts(type="user_budget", user_info=_user_info())
    proxy_logging.slack_alerting_instance.budget_alerts.assert_not_called()
    proxy_logging.email_logging_instance.budget_alerts.assert_not_called()


@pytest.mark.asyncio
async def test_budget_alerts_slack_when_slack_alerting(proxy_logging):
    proxy_logging.alerting = ["slack"]
    captured: Dict[str, Any] = {}

    async def fake_alert(**kwargs):
        captured.update(kwargs)

    proxy_logging.slack_alerting_instance = MagicMock(budget_alerts=fake_alert)
    proxy_logging.email_logging_instance = None
    user_info = _user_info()
    await proxy_logging.budget_alerts(type="user_budget", user_info=user_info)
    snapshot = {
        "type": captured["type"],
        "user_info_is_callinfo": isinstance(captured["user_info"], CallInfo),
        "user_id": captured["user_info"].user_id,
    }
    assert snapshot == {"type": "user_budget", "user_info_is_callinfo": True, "user_id": "u1"}


@pytest.mark.asyncio
async def test_budget_alerts_soft_budget_with_alert_emails_bypasses_global(proxy_logging):
    proxy_logging.alerting = None
    proxy_logging.slack_alerting_instance = MagicMock(budget_alerts=AsyncMock())
    proxy_logging.email_logging_instance = MagicMock(budget_alerts=AsyncMock())
    info = _user_info(alert_emails=["a@b.c"])
    await proxy_logging.budget_alerts(type="soft_budget", user_info=info)
    proxy_logging.email_logging_instance.budget_alerts.assert_called_once()
    proxy_logging.slack_alerting_instance.budget_alerts.assert_not_called()


@pytest.mark.asyncio
async def test_budget_alerts_slack_failure_raises(proxy_logging):
    proxy_logging.alerting = ["slack"]
    proxy_logging.slack_alerting_instance = MagicMock(
        budget_alerts=AsyncMock(side_effect=ConnectionError("slack"))
    )
    proxy_logging.email_logging_instance = None
    with pytest.raises(ConnectionError):
        await proxy_logging.budget_alerts(type="user_budget", user_info=_user_info())


# ---------------------------------------------------------------------------
# alerting_handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_alerting_handler_no_op_when_alerting_is_none(proxy_logging):
    proxy_logging.alerting = None
    proxy_logging.slack_alerting_instance = MagicMock(send_alert=AsyncMock())
    await proxy_logging.alerting_handler(message="x", level="High", alert_type=AlertType.db_exceptions)
    proxy_logging.slack_alerting_instance.send_alert.assert_not_called()


@pytest.mark.asyncio
async def test_alerting_handler_sends_to_slack(proxy_logging):
    proxy_logging.alerting = ["slack"]
    captured: Dict[str, Any] = {}

    async def fake_send(**kwargs):
        captured.update(kwargs)

    proxy_logging.slack_alerting_instance = MagicMock(send_alert=fake_send)
    await proxy_logging.alerting_handler(
        message="hi", level="High", alert_type=AlertType.db_exceptions, request_data={"metadata": {}}
    )
    snapshot = {
        "message": captured["message"],
        "level": captured["level"],
        "alert_type": captured["alert_type"],
        "user_info": captured["user_info"],
    }
    assert snapshot == {
        "message": "hi",
        "level": "High",
        "alert_type": AlertType.db_exceptions,
        "user_info": None,
    }


@pytest.mark.asyncio
async def test_alerting_handler_sentry_without_sdk_error_raises(proxy_logging, monkeypatch):
    proxy_logging.alerting = ["sentry"]
    monkeypatch.setattr(litellm.utils, "sentry_sdk_instance", None)
    with pytest.raises(Exception, match="SENTRY_DSN"):
        await proxy_logging.alerting_handler(message="x", level="Low", alert_type=AlertType.db_exceptions)


# ---------------------------------------------------------------------------
# failure_handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_handler_skips_when_db_exceptions_not_in_alert_types(proxy_logging):
    proxy_logging.alert_types = ["llm_too_slow"]  # type: ignore[list-item]
    proxy_logging.alerting_handler = AsyncMock()
    proxy_logging.service_logging_obj = MagicMock(async_service_failure_hook=AsyncMock())
    await proxy_logging.failure_handler(original_exception=Exception("x"), duration=1.0, call_type="db_read")
    proxy_logging.alerting_handler.assert_not_called()
    proxy_logging.service_logging_obj.async_service_failure_hook.assert_not_called()


@pytest.mark.asyncio
async def test_failure_handler_logs_db_error_and_calls_service_logging(proxy_logging, monkeypatch):
    proxy_logging.alert_types = [AlertType.db_exceptions]
    proxy_logging.alerting_handler = AsyncMock()
    proxy_logging.service_logging_obj = MagicMock(async_service_failure_hook=AsyncMock())
    monkeypatch.setattr(litellm.utils, "capture_exception", None)
    await proxy_logging.failure_handler(
        original_exception=HTTPException(status_code=500, detail="boom"),
        duration=1.5,
        call_type="db_write",
    )
    call_kwargs = proxy_logging.service_logging_obj.async_service_failure_hook.call_args.kwargs
    snapshot = {
        "service": call_kwargs["service"].value if hasattr(call_kwargs["service"], "value") else call_kwargs["service"],
        "duration": call_kwargs["duration"],
        "call_type": call_kwargs["call_type"],
    }
    assert snapshot == {
        "service": "postgres",
        "duration": 1.5,
        "call_type": "db_write",
    }


@pytest.mark.asyncio
async def test_failure_handler_with_capture_exception_invoked(proxy_logging, monkeypatch):
    proxy_logging.alert_types = [AlertType.db_exceptions]
    proxy_logging.alerting_handler = AsyncMock()
    proxy_logging.service_logging_obj = MagicMock(async_service_failure_hook=AsyncMock())
    captured: Dict[str, Any] = {}

    def fake_capture(error):
        captured["error"] = error

    monkeypatch.setattr(litellm.utils, "capture_exception", fake_capture)
    err = RuntimeError("real")
    await proxy_logging.failure_handler(original_exception=err, duration=1.0, call_type="db_read")
    snapshot = {
        "captured_is_input": captured["error"] is err,
        "service_failure_called": proxy_logging.service_logging_obj.async_service_failure_hook.called,
        "alerting_handler_scheduled": proxy_logging.alerting_handler.called,
    }
    assert snapshot == {
        "captured_is_input": True,
        "service_failure_called": True,
        "alerting_handler_scheduled": True,
    }


@pytest.mark.asyncio
async def test_failure_handler_propagates_service_logging_error_raises(proxy_logging, monkeypatch):
    proxy_logging.alert_types = [AlertType.db_exceptions]
    proxy_logging.alerting_handler = AsyncMock()
    proxy_logging.service_logging_obj = MagicMock(
        async_service_failure_hook=AsyncMock(side_effect=RuntimeError("svc"))
    )
    monkeypatch.setattr(litellm.utils, "capture_exception", None)
    with pytest.raises(RuntimeError):
        await proxy_logging.failure_handler(
            original_exception=Exception("x"), duration=0.0, call_type="db_read"
        )
