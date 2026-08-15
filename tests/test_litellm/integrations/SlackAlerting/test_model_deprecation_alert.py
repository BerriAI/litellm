"""Tests for the Slack alerting model deprecation hook."""

import asyncio
import os
import sys
from itertools import chain, repeat
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.constants import SLACK_MODEL_DEPRECATION_LOCK_ID
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import AlertType
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    DEPRECATION_IDLE_POLL_SECONDS,
)


def _make_router(deployments):
    router = MagicMock()
    router.get_model_list.return_value = deployments
    return router


@pytest.mark.asyncio
async def test_should_skip_when_alert_type_disabled():
    alerting = SlackAlerting(
        alerting=["slack"],
        alert_types=[AlertType.llm_exceptions],
    )
    sent = await alerting.send_model_deprecation_alert(llm_router=MagicMock())
    assert sent is False


@pytest.mark.asyncio
async def test_should_skip_when_no_alerting_configured():
    alerting = SlackAlerting(
        alerting=None,
        alert_types=[AlertType.model_deprecation_warnings],
    )
    sent = await alerting.send_model_deprecation_alert(llm_router=MagicMock())
    assert sent is False


@pytest.mark.asyncio
async def test_should_skip_when_no_deprecations_found(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", {})
    alerting = SlackAlerting(
        alerting=["slack"],
        alert_types=[AlertType.model_deprecation_warnings],
    )
    router = _make_router(
        [
            {
                "model_name": "fresh",
                "litellm_params": {"model": "openai/gpt-4o"},
                "model_info": {"id": "x"},
            }
        ]
    )
    sent = await alerting.send_model_deprecation_alert(llm_router=router)
    assert sent is False


@pytest.mark.asyncio
async def test_should_dispatch_high_severity_when_deprecated(monkeypatch):
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "dead-model": {
                "deprecation_date": "2020-01-01",
                "litellm_provider": "openai",
            }
        },
    )
    alerting = SlackAlerting(
        alerting=["slack"],
        alert_types=[AlertType.model_deprecation_warnings],
    )
    router = _make_router(
        [
            {
                "model_name": "dead-alias",
                "litellm_params": {"model": "dead-model"},
                "model_info": {"id": "1"},
            }
        ]
    )

    with patch.object(
        alerting, "send_alert", new_callable=AsyncMock
    ) as mock_send_alert:
        sent = await alerting.send_model_deprecation_alert(llm_router=router)

    assert sent is True
    mock_send_alert.assert_awaited_once()
    call_kwargs = mock_send_alert.await_args.kwargs
    assert call_kwargs["alert_type"] == AlertType.model_deprecation_warnings
    assert call_kwargs["level"] == "High"
    assert call_kwargs["alerting_metadata"]["deprecated_count"] == 1
    assert call_kwargs["alerting_metadata"]["imminent_count"] == 0
    assert "dead-alias" in call_kwargs["message"]


@pytest.mark.asyncio
async def test_should_alert_once_the_alert_type_and_router_arrive_after_startup(
    monkeypatch,
):
    """The loop starts before config reload, so a disabled pass must not cost a day of alerts"""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"dead-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"}},
    )
    alerting = SlackAlerting(alerting=["slack"], alert_types=[AlertType.llm_exceptions])
    router = _make_router(
        [
            {
                "model_name": "dead-alias",
                "litellm_params": {"model": "dead-model"},
                "model_info": {"id": "1"},
            }
        ]
    )

    slept: list[float] = []

    async def stop_after_second_pass(seconds):
        slept.append(seconds)
        if alerting.alert_types == [AlertType.llm_exceptions]:
            alerting.update_values(
                alert_types=[AlertType.model_deprecation_warnings]
            )  # simulates a config reload enabling the alert
            return
        raise asyncio.CancelledError

    with (
        patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert,
        patch(
            "litellm.integrations.SlackAlerting.slack_alerting.asyncio.sleep",
            side_effect=stop_after_second_pass,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await alerting.run_scheduled_deprecation_check(get_llm_router=lambda: router)

    assert slept == [
        DEPRECATION_IDLE_POLL_SECONDS,
        DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    ]
    mock_send_alert.assert_awaited_once()
    assert "dead-alias" in mock_send_alert.await_args.kwargs["message"]


@pytest.mark.asyncio
async def test_should_wait_for_the_router_instead_of_sleeping_a_full_day(monkeypatch):
    """Config load can start the loop before the router exists, which must not cost a day of alerts"""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"dead-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"}},
    )
    alerting = SlackAlerting(
        alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings]
    )
    router = _make_router(
        [
            {
                "model_name": "dead-alias",
                "litellm_params": {"model": "dead-model"},
                "model_info": {"id": "1"},
            }
        ]
    )
    router_absent_passes = 100
    routers = chain(repeat(None, router_absent_passes), repeat(router))
    slept: list[float] = []

    async def record_sleep(seconds):
        slept.append(seconds)
        if len(slept) > router_absent_passes:
            raise asyncio.CancelledError

    with (
        patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert,
        patch(
            "litellm.integrations.SlackAlerting.slack_alerting.asyncio.sleep",
            side_effect=record_sleep,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await alerting.run_scheduled_deprecation_check(
            get_llm_router=lambda: next(routers)
        )

    assert slept == [DEPRECATION_IDLE_POLL_SECONDS] * router_absent_passes + [
        DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS
    ]
    mock_send_alert.assert_awaited_once()
    assert "dead-alias" in mock_send_alert.await_args.kwargs["message"]


@pytest.mark.parametrize(
    "lock_acquired, expect_alert",
    [(True, True), (None, True), (False, False)],
    ids=["lock won", "no redis lock", "another pod holds the lock"],
)
@pytest.mark.asyncio
async def test_should_alert_only_from_the_pod_holding_the_daily_lock(
    monkeypatch, lock_acquired, expect_alert
):
    """Every pod runs the loop, so a fleet must not send one identical alert per replica"""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"dead-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"}},
    )
    alerting = SlackAlerting(
        alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings]
    )
    router = _make_router(
        [
            {
                "model_name": "dead-alias",
                "litellm_params": {"model": "dead-model"},
                "model_info": {"id": "1"},
            }
        ]
    )
    pod_lock_manager = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=lock_acquired)

    with (
        patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert,
        patch(
            "litellm.integrations.SlackAlerting.slack_alerting.asyncio.sleep",
            side_effect=asyncio.CancelledError,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await alerting.run_scheduled_deprecation_check(
            get_llm_router=lambda: router, pod_lock_manager=pod_lock_manager
        )

    assert mock_send_alert.await_count == int(expect_alert)
    assert pod_lock_manager.acquire_lock.await_args.kwargs == {
        "cronjob_id": SLACK_MODEL_DEPRECATION_LOCK_ID,
        "ttl": DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
        "allow_reentrant": False,
    }
