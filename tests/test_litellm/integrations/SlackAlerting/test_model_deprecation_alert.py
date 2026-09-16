"""Tests for the Slack alerting model deprecation hook."""

import asyncio
from itertools import chain, repeat
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


import litellm
from litellm.constants import SLACK_MODEL_DEPRECATION_LOCK_ID
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import AlertType
from litellm.types.integrations.slack_alerting import SlackAlertingCacheKeys
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    DEPRECATION_IDLE_POLL_SECONDS,
)

DEAD_MODEL_COST = {
    "dead-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"}
}
DEAD_ALIAS_DEPLOYMENT = {
    "model_name": "dead-alias",
    "litellm_params": {"model": "dead-model"},
    "model_info": {"id": "1"},
}


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
    assert isinstance(
        await alerting.internal_usage_cache.async_get_cache(
            key=SlackAlertingCacheKeys.deprecation_alert_sent_key.value
        ),
        float,
    )


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

    async def stop_after_third_pass(seconds):
        slept.append(seconds)
        if alerting.alert_types == [AlertType.llm_exceptions]:
            alerting.update_values(
                alert_types=[AlertType.model_deprecation_warnings]
            )  # simulates a config reload enabling the alert
        if len(slept) == 3:
            raise asyncio.CancelledError

    with (
        patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert,
        patch(
            "litellm.integrations.SlackAlerting.slack_alerting.asyncio.sleep",
            side_effect=stop_after_third_pass,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await alerting.run_scheduled_deprecation_check(get_llm_router=lambda: router)

    assert slept == [DEPRECATION_IDLE_POLL_SECONDS] * 3
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

    assert slept == [DEPRECATION_IDLE_POLL_SECONDS] * (router_absent_passes + 1)
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


@pytest.mark.asyncio
async def test_should_retry_on_the_next_poll_when_the_lock_claim_fails(monkeypatch):
    """A redis blip at claim time returns False like a held lock, and must not cost every pod a day of alerts"""
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting = SlackAlerting(
        alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings]
    )
    router = _make_router([DEAD_ALIAS_DEPLOYMENT])
    pod_lock_manager = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(side_effect=[False, True])
    slept: list[float] = []

    async def stop_after_second_pass(seconds):
        slept.append(seconds)
        if len(slept) == 2:
            raise asyncio.CancelledError

    with (
        patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert,
        patch(
            "litellm.integrations.SlackAlerting.slack_alerting.asyncio.sleep",
            side_effect=stop_after_second_pass,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await alerting.run_scheduled_deprecation_check(
            get_llm_router=lambda: router, pod_lock_manager=pod_lock_manager
        )

    assert slept == [DEPRECATION_IDLE_POLL_SECONDS] * 2
    assert pod_lock_manager.acquire_lock.await_count == 2
    mock_send_alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_should_not_claim_the_lock_when_there_is_nothing_to_report(monkeypatch):
    """An empty pass must not hold the daily lock, or a sunset added later waits out the whole window"""
    monkeypatch.setattr(litellm, "model_cost", {})
    alerting = SlackAlerting(
        alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings]
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
    pod_lock_manager = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=True)

    with patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert:
        sent = await alerting.send_model_deprecation_alert(
            llm_router=router, pod_lock_manager=pod_lock_manager
        )

    assert sent is False
    pod_lock_manager.acquire_lock.assert_not_awaited()
    mock_send_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_not_alert_or_claim_the_lock_within_a_day_of_a_sent_alert(monkeypatch):
    """The shared sent stamp keeps sibling pods and restarts from re-alerting or re-asking redis for a day"""
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting = SlackAlerting(
        alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings]
    )
    await alerting.internal_usage_cache.async_set_cache(
        key=SlackAlertingCacheKeys.deprecation_alert_sent_key.value,
        value=1.0,
        ttl=DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    )
    router = _make_router([DEAD_ALIAS_DEPLOYMENT])
    pod_lock_manager = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=True)

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

    pod_lock_manager.acquire_lock.assert_not_awaited()
    mock_send_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_back_off_a_full_day_after_a_pass_raises(monkeypatch):
    """A misconfigured webhook raises on every send, which must log once a day rather than every poll"""
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting = SlackAlerting(
        alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings]
    )
    router = _make_router([DEAD_ALIAS_DEPLOYMENT])
    slept: list[float] = []

    async def stop_after_second_pass(seconds):
        slept.append(seconds)
        if len(slept) == 2:
            raise asyncio.CancelledError

    with (
        patch.object(
            alerting,
            "send_alert",
            new_callable=AsyncMock,
            side_effect=ValueError("Missing SLACK_WEBHOOK_URL from environment"),
        ) as mock_send_alert,
        patch(
            "litellm.integrations.SlackAlerting.slack_alerting.asyncio.sleep",
            side_effect=stop_after_second_pass,
        ),
        pytest.raises(asyncio.CancelledError),
    ):
        await alerting.run_scheduled_deprecation_check(get_llm_router=lambda: router)

    assert slept == [DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS] * 2
    assert mock_send_alert.await_count == 2
