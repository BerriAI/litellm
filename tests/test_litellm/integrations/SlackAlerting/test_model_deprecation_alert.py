"""Tests for the Slack alerting model deprecation hook."""

import asyncio
from itertools import chain, repeat
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.constants import SLACK_MODEL_DEPRECATION_LOCK_ID
from litellm.integrations.SlackAlerting.ms_teams import MS_TEAMS_ALERTING_DESTINATION
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.proxy._types import AlertType
from litellm.proxy.common_utils.model_deprecation_notifications import DeprecationEmailContext
from litellm.types.integrations.slack_alerting import SlackAlertingCacheKeys
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    DEPRECATION_IDLE_POLL_SECONDS,
)

DEAD_MODEL_COST = {"dead-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"}}
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

    with patch.object(alerting, "send_alert", new_callable=AsyncMock) as mock_send_alert:
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
    alerting = SlackAlerting(alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings])
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
        await alerting.run_scheduled_deprecation_check(get_llm_router=lambda: next(routers))

    assert slept == [DEPRECATION_IDLE_POLL_SECONDS] * (router_absent_passes + 1)
    mock_send_alert.assert_awaited_once()
    assert "dead-alias" in mock_send_alert.await_args.kwargs["message"]


@pytest.mark.parametrize(
    "lock_acquired, expect_alert",
    [(True, True), (None, True), (False, False)],
    ids=["lock won", "no redis lock", "another pod holds the lock"],
)
@pytest.mark.asyncio
async def test_should_alert_only_from_the_pod_holding_the_daily_lock(monkeypatch, lock_acquired, expect_alert):
    """Every pod runs the loop, so a fleet must not send one identical alert per replica"""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {"dead-model": {"deprecation_date": "2020-01-01", "litellm_provider": "openai"}},
    )
    alerting = SlackAlerting(alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings])
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
        await alerting.run_scheduled_deprecation_check(get_llm_router=lambda: router, pod_lock_manager=pod_lock_manager)

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
    alerting = SlackAlerting(alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings])
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
        await alerting.run_scheduled_deprecation_check(get_llm_router=lambda: router, pod_lock_manager=pod_lock_manager)

    assert slept == [DEPRECATION_IDLE_POLL_SECONDS] * 2
    assert pod_lock_manager.acquire_lock.await_count == 2
    mock_send_alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_should_not_claim_the_lock_when_there_is_nothing_to_report(monkeypatch):
    """An empty pass must not hold the daily lock, or a sunset added later waits out the whole window"""
    monkeypatch.setattr(litellm, "model_cost", {})
    alerting = SlackAlerting(alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings])
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
        sent = await alerting.send_model_deprecation_alert(llm_router=router, pod_lock_manager=pod_lock_manager)

    assert sent is False
    pod_lock_manager.acquire_lock.assert_not_awaited()
    mock_send_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_not_alert_or_claim_the_lock_within_a_day_of_a_sent_alert(monkeypatch):
    """The shared sent stamp keeps sibling pods and restarts from re-alerting or re-asking redis for a day"""
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting = SlackAlerting(alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings])
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
        await alerting.run_scheduled_deprecation_check(get_llm_router=lambda: router, pod_lock_manager=pod_lock_manager)

    pod_lock_manager.acquire_lock.assert_not_awaited()
    mock_send_alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_skip_a_raising_pass_for_a_day_and_keep_polling(monkeypatch):
    """A misconfigured webhook raises on every send, which must log once a day rather than every poll"""
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting = SlackAlerting(alerting=["slack"], alert_types=[AlertType.model_deprecation_warnings])
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

    assert slept == [DEPRECATION_IDLE_POLL_SECONDS] * 2
    assert mock_send_alert.await_count == 1


def _email_only_alerting() -> SlackAlerting:
    return SlackAlerting(alerting=["email"], alert_types=[AlertType.model_deprecation_warnings])


class _EmailRun:
    def __init__(self, result: int = 1, fail: bool = False):
        self.result = result
        self.fail = fail
        self.contexts = []

    async def __call__(self, ctx: DeprecationEmailContext) -> int:
        self.contexts.append(ctx)
        if self.fail:
            raise ValueError("smtp exploded")
        return self.result


@pytest.mark.parametrize(
    ("alerting", "alert_types", "thresholds", "expected"),
    [
        (None, [AlertType.model_deprecation_warnings], (30,), False),
        (["slack"], [AlertType.model_deprecation_warnings], (30,), False),
        (["email"], [AlertType.llm_exceptions], (30,), False),
        (["email"], [AlertType.model_deprecation_warnings], (), False),
        (["email"], [AlertType.model_deprecation_warnings], (30,), True),
        (["slack", "email"], [AlertType.model_deprecation_warnings], (30, 7, 0), True),
    ],
)
def test_deprecation_emails_enabled_truth_table(alerting, alert_types, thresholds, expected):
    alerting_obj: Final = SlackAlerting(
        alerting=alerting, alert_types=alert_types, alerting_args={"model_deprecation_email_thresholds": thresholds}
    )
    assert alerting_obj._deprecation_emails_enabled() is expected


@pytest.mark.asyncio
async def test_should_not_run_the_slack_pass_without_a_slack_or_teams_channel(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    pod_lock_manager: Final = MagicMock()
    pod_lock_manager.acquire_lock = AsyncMock(return_value=True)

    sent: Final = await _email_only_alerting().send_model_deprecation_alert(
        llm_router=_make_router([DEAD_ALIAS_DEPLOYMENT]), pod_lock_manager=pod_lock_manager
    )

    assert sent is False
    pod_lock_manager.acquire_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_should_run_the_email_pass_with_email_only_alerting(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting: Final = _email_only_alerting()
    prisma: Final = SimpleNamespace()
    run: Final = _EmailRun(result=2)

    await alerting._run_deprecation_passes(
        get_llm_router=lambda: _make_router([DEAD_ALIAS_DEPLOYMENT]),
        pod_lock_manager=None,
        get_prisma_client=lambda: prisma,
        get_email_logger=lambda: None,
        send_emails=run,
    )

    assert len(run.contexts) == 1
    assert run.contexts[0].prisma_client is prisma
    assert run.contexts[0].alerting_args.model_deprecation_email_thresholds == (30, 7, 0)
    assert run.contexts[0].cache is alerting.internal_usage_cache
    assert alerting.deprecation_email_backoff_until == 0.0


@pytest.mark.asyncio
async def test_should_still_email_when_the_slack_pass_raises_and_back_off_only_slack(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting: Final = SlackAlerting(alerting=["slack", "email"], alert_types=[AlertType.model_deprecation_warnings])
    run: Final = _EmailRun()

    with patch.object(alerting, "send_alert", new_callable=AsyncMock, side_effect=ValueError("no webhook")):
        await alerting._run_deprecation_passes(
            get_llm_router=lambda: _make_router([DEAD_ALIAS_DEPLOYMENT]),
            pod_lock_manager=None,
            get_prisma_client=SimpleNamespace,
            get_email_logger=lambda: None,
            send_emails=run,
        )

    assert len(run.contexts) == 1
    assert alerting.deprecation_alert_backoff_until > 0.0
    assert alerting.deprecation_email_backoff_until == 0.0


@pytest.mark.asyncio
async def test_should_back_off_only_the_email_pass_when_it_raises(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting: Final = _email_only_alerting()
    run: Final = _EmailRun(fail=True)

    await alerting._run_deprecation_passes(
        get_llm_router=lambda: _make_router([DEAD_ALIAS_DEPLOYMENT]),
        pod_lock_manager=None,
        get_prisma_client=SimpleNamespace,
        get_email_logger=lambda: None,
        send_emails=run,
    )

    assert alerting.deprecation_email_backoff_until > 0.0
    assert alerting.deprecation_alert_backoff_until == 0.0


@pytest.mark.asyncio
async def test_should_skip_the_email_pass_within_a_day_of_the_last_one(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting: Final = _email_only_alerting()
    await alerting.internal_usage_cache.async_set_cache(
        key=SlackAlertingCacheKeys.deprecation_email_pass_key.value, value=1.0
    )
    run: Final = _EmailRun()

    sent: Final = await alerting._run_deprecation_email_pass(
        _make_router([DEAD_ALIAS_DEPLOYMENT]), None, SimpleNamespace, lambda: None, run
    )

    assert sent == 0
    assert run.contexts == []


@pytest.mark.asyncio
async def test_should_skip_the_email_pass_without_a_database(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    run: Final = _EmailRun()

    sent: Final = await _email_only_alerting()._run_deprecation_email_pass(
        _make_router([DEAD_ALIAS_DEPLOYMENT]), None, lambda: None, lambda: None, run
    )

    assert sent == 0
    assert run.contexts == []


@pytest.mark.asyncio
async def test_should_skip_the_email_pass_without_a_router():
    run: Final = _EmailRun()
    prisma_lookups: Final = []

    def get_prisma_client():
        prisma_lookups.append(True)
        return SimpleNamespace()

    sent: Final = await _email_only_alerting()._run_deprecation_email_pass(
        None, None, get_prisma_client, lambda: None, run
    )

    assert sent == 0
    assert prisma_lookups == []
    assert run.contexts == []


@pytest.mark.asyncio
async def test_should_keep_polling_every_thirty_seconds_after_an_email_failure(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting: Final = _email_only_alerting()
    slept: list[float] = []

    async def stop_after_second_pass(seconds):
        slept.append(seconds)
        if len(slept) == 2:
            raise asyncio.CancelledError

    run: Final = _EmailRun(fail=True)
    with pytest.raises(asyncio.CancelledError):
        await alerting.run_scheduled_deprecation_check(
            get_llm_router=lambda: _make_router([DEAD_ALIAS_DEPLOYMENT]),
            get_prisma_client=SimpleNamespace,
            get_email_logger=lambda: None,
            send_emails=run,
            sleep=stop_after_second_pass,
        )

    assert slept == [DEPRECATION_IDLE_POLL_SECONDS] * 2
    assert len(run.contexts) == 1


@pytest.mark.parametrize(
    ("alerting", "expected"),
    [
        (None, False),
        ([], False),
        (["email"], False),
        (["webhook"], False),
        (["slack"], True),
        ([MS_TEAMS_ALERTING_DESTINATION], True),
        (["slack", "email"], True),
    ],
)
def test_deprecation_alerts_enabled_requires_a_slack_or_teams_channel(alerting, expected):
    alerting_obj: Final = SlackAlerting(alerting=alerting, alert_types=[AlertType.model_deprecation_warnings])
    assert alerting_obj._deprecation_alerts_enabled() is expected


@pytest.mark.asyncio
async def test_should_keep_emailing_while_a_broken_slack_pass_is_backed_off(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting: Final = SlackAlerting(alerting=["slack", "email"], alert_types=[AlertType.model_deprecation_warnings])
    run: Final = _EmailRun()

    with patch.object(
        alerting, "send_alert", new_callable=AsyncMock, side_effect=ValueError("no webhook")
    ) as send_alert:
        for _ in range(2):
            await alerting._run_deprecation_passes(
                get_llm_router=lambda: _make_router([DEAD_ALIAS_DEPLOYMENT]),
                pod_lock_manager=None,
                get_prisma_client=SimpleNamespace,
                get_email_logger=lambda: None,
                send_emails=run,
                now=lambda: 1_000.0,
            )

    assert send_alert.await_count == 1
    assert len(run.contexts) == 2


@pytest.mark.asyncio
async def test_should_retry_a_backed_off_email_pass_after_a_day(monkeypatch):
    monkeypatch.setattr(litellm, "model_cost", DEAD_MODEL_COST)
    alerting: Final = _email_only_alerting()
    run: Final = _EmailRun(fail=True)

    for moment in (
        1_000.0,
        1_000.0 + DEPRECATION_IDLE_POLL_SECONDS,
        1_000.0 + DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    ):
        await alerting._run_deprecation_passes(
            get_llm_router=lambda: _make_router([DEAD_ALIAS_DEPLOYMENT]),
            pod_lock_manager=None,
            get_prisma_client=SimpleNamespace,
            get_email_logger=lambda: None,
            send_emails=run,
            now=lambda moment=moment: moment,
        )

    assert len(run.contexts) == 2
