import itertools
import logging
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from litellm.constants import PTU_ROLLUP_JOB_ID, PTU_ROLLUP_LOCK_TTL_SECONDS
from litellm.proxy._types import ScheduledJobStaggerSettings
from litellm.proxy.common_utils.scheduled_job_stagger import (
    apply_scheduled_job_stagger,
    attach_job_timing_logger,
    offset_seconds,
    parse_stagger_settings,
    resolve_stagger_identity,
    stagger_trigger,
)

OPERATOR_CRON_JOB_ID = "spend_log_cleanup_job"
SHARED_INTERVAL_JOB_IDS = ("periodic_reload_job", "get_credentials_job", "add_deployment_job")


async def _noop() -> None: ...


def _scheduler() -> AsyncIOScheduler:
    return AsyncIOScheduler(
        jobstores={"default": MemoryJobStore()},
        executors={"default": AsyncIOExecutor()},
        timezone=None,
    )


def _with_jobs(scheduler: AsyncIOScheduler) -> AsyncIOScheduler:
    for job_id in SHARED_INTERVAL_JOB_IDS:
        scheduler.add_job(_noop, "interval", seconds=30, id=job_id, replace_existing=True)
    scheduler.add_job(
        _noop, "cron", hour=0, minute=15, timezone=timezone.utc, id=PTU_ROLLUP_JOB_ID, replace_existing=True
    )
    # an operator-supplied crontab, which must survive untouched
    scheduler.add_job(_noop, CronTrigger.from_crontab("0 3 * * *"), id=OPERATOR_CRON_JOB_ID, replace_existing=True)
    return scheduler


def _next_run_times(scheduler: AsyncIOScheduler) -> dict[str, datetime]:
    scheduler.start(paused=True)
    try:
        return {job.id: job.next_run_time for job in scheduler.get_jobs()}
    finally:
        scheduler.shutdown(wait=False)


def _settings(**overrides) -> ScheduledJobStaggerSettings:
    return ScheduledJobStaggerSettings(**overrides)


def _stagger(scheduler: AsyncIOScheduler, identity: str = "pod-a:1", **overrides):
    return apply_scheduled_job_stagger(scheduler=scheduler, settings=_settings(**overrides), identity=identity)


def _fire_times(trigger, start: datetime, steps: int) -> tuple[datetime, ...]:
    """The fire times APScheduler would produce, each computed from the one before it"""
    return tuple(
        itertools.accumulate(
            range(steps - 1),
            lambda previous, _: trigger.get_next_fire_time(previous, previous),
            initial=trigger.get_next_fire_time(None, start),
        )
    )


async def test_jobs_sharing_an_interval_no_longer_share_a_firing_instant():
    """The defect: APScheduler anchors every interval job at ``now + interval``"""
    unstaggered = _next_run_times(_with_jobs(_scheduler()))
    base_times = [unstaggered[job_id] for job_id in SHARED_INTERVAL_JOB_IDS]
    assert max(base_times) - min(base_times) < timedelta(seconds=1)

    scheduler = _with_jobs(_scheduler())
    _stagger(scheduler)
    staggered = _next_run_times(scheduler)

    shifted_times = [staggered[job_id] for job_id in SHARED_INTERVAL_JOB_IDS]
    assert len(set(shifted_times)) == len(SHARED_INTERVAL_JOB_IDS)
    assert max(shifted_times) - min(shifted_times) >= timedelta(seconds=1)


def test_replicas_do_not_start_the_same_job_at_the_same_instant():
    offsets = {
        identity: offset_seconds(job_id="update_spend_job", identity=identity, window_seconds=300)
        for identity in ("pod-a:1", "pod-b:1", "pod-c:1", "pod-a:2")
    }
    assert len(set(offsets.values())) == len(offsets)


def test_offset_is_reproducible_for_a_given_job_and_identity():
    first = offset_seconds(job_id="update_spend_job", identity="pod-a:7", window_seconds=300)
    second = offset_seconds(job_id="update_spend_job", identity="pod-a:7", window_seconds=300)
    assert first == second


def test_offset_never_exceeds_one_period_of_an_interval_job():
    """A job may be phase shifted, never delayed past the wait it already had"""
    scheduler = _scheduler()
    scheduler.add_job(_noop, "interval", seconds=5, id="tight_job", replace_existing=True)
    applied = _stagger(scheduler, window_seconds=300)

    assert 0 <= applied["tight_job"] < 5


async def test_operator_supplied_cron_keeps_its_exact_schedule():
    unstaggered = _next_run_times(_with_jobs(_scheduler()))

    scheduler = _with_jobs(_scheduler())
    applied = _stagger(scheduler)
    staggered = _next_run_times(scheduler)

    assert applied[OPERATOR_CRON_JOB_ID] == 0
    assert staggered[OPERATOR_CRON_JOB_ID] == unstaggered[OPERATOR_CRON_JOB_ID]


def test_default_cron_is_staggered_and_keeps_its_offset_on_every_later_fire():
    """
    A cron trigger recomputes each fire from the wall clock, so an offset applied only to
    the first run would snap straight back onto the shared instant
    """
    scheduler = _with_jobs(_scheduler())
    applied = _stagger(scheduler)
    assert applied[PTU_ROLLUP_JOB_ID] > 0

    trigger = next(job.trigger for job in scheduler.get_jobs() if job.id == PTU_ROLLUP_JOB_ID)
    fires = _fire_times(trigger, datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), 3)

    expected = timedelta(minutes=15) + timedelta(seconds=applied[PTU_ROLLUP_JOB_ID])
    assert [fire - fire.replace(hour=0, minute=0, second=0, microsecond=0) for fire in fires] == [expected] * 3


async def test_explicit_offset_overrides_the_derived_one_and_zero_pins_a_job():
    scheduler = _with_jobs(_scheduler())
    applied = _stagger(scheduler, offsets={"periodic_reload_job": 0, PTU_ROLLUP_JOB_ID: 7})
    unstaggered = _next_run_times(_with_jobs(_scheduler()))
    staggered = _next_run_times(scheduler)

    assert applied["periodic_reload_job"] == 0
    assert applied[PTU_ROLLUP_JOB_ID] == 7
    assert staggered[PTU_ROLLUP_JOB_ID] - unstaggered[PTU_ROLLUP_JOB_ID] == timedelta(seconds=7)


async def test_disabling_the_stagger_leaves_every_schedule_untouched():
    unstaggered = _next_run_times(_with_jobs(_scheduler()))

    scheduler = _with_jobs(_scheduler())
    applied = _stagger(scheduler, enabled=False)
    staggered = _next_run_times(scheduler)

    assert set(applied.values()) == {0}
    assert {job_id: run for job_id, run in staggered.items() if job_id != OPERATOR_CRON_JOB_ID}.keys() == {
        job_id for job_id in unstaggered if job_id != OPERATOR_CRON_JOB_ID
    }
    assert staggered[PTU_ROLLUP_JOB_ID] == unstaggered[PTU_ROLLUP_JOB_ID]


async def test_a_job_that_anchored_its_own_first_fire_is_left_alone():
    anchor = datetime.now(timezone.utc) + timedelta(seconds=90)
    scheduler = _scheduler()
    scheduler.add_job(
        _noop, "interval", days=7, next_run_time=anchor, id="weekly_spend_report_job", replace_existing=True
    )
    applied = _stagger(scheduler)

    assert applied["weekly_spend_report_job"] == 0
    assert _next_run_times(scheduler)["weekly_spend_report_job"] == anchor


async def test_applying_after_the_scheduler_started_is_refused_loudly(caplog):
    """
    Every job carries a next_run_time once the scheduler is running, so the sweep would skip
    all of them and report success while changing nothing
    """
    scheduler = _with_jobs(_scheduler())
    scheduler.start(paused=True)
    try:
        before = {job.id: job.next_run_time for job in scheduler.get_jobs()}
        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            applied = _stagger(scheduler)
        after = {job.id: job.next_run_time for job in scheduler.get_jobs()}
    finally:
        scheduler.shutdown(wait=False)

    assert set(applied.values()) == {0}
    assert after == before
    assert "already running" in caplog.text


async def test_a_leader_elected_cron_is_never_spread_past_its_dedupe_window():
    """
    These crons hold a lock that marks the window's work done. Two replicas further apart
    than that both find the key free and both run, so the monthly report goes out twice.
    """
    scheduler = _with_jobs(_scheduler())
    applied = _stagger(scheduler, window_seconds=100_000)

    assert 0 < applied[PTU_ROLLUP_JOB_ID] < PTU_ROLLUP_LOCK_TTL_SECONDS


async def test_an_explicit_offset_past_the_dedupe_window_is_clamped_and_warned(caplog):
    scheduler = _with_jobs(_scheduler())
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        applied = _stagger(scheduler, offsets={PTU_ROLLUP_JOB_ID: 100_000})

    assert applied[PTU_ROLLUP_JOB_ID] == PTU_ROLLUP_LOCK_TTL_SECONDS - 1
    assert PTU_ROLLUP_JOB_ID in caplog.text


async def test_an_explicit_offset_on_an_ordinary_job_is_honored_as_given():
    scheduler = _with_jobs(_scheduler())
    applied = _stagger(scheduler, offsets={"periodic_reload_job": 100_000})

    assert applied["periodic_reload_job"] == 100_000


def test_a_job_registered_after_startup_still_gets_its_offset():
    """
    The runtime reschedule path adds to a started scheduler, where the sweep cannot see the
    job, so the trigger has to carry the offset before it is handed over
    """
    base = IntervalTrigger(seconds=3600, timezone=timezone.utc)
    shifted = stagger_trigger(
        job_id="spend_log_cleanup_job",
        trigger=base,
        period_seconds=3600,
        settings=_settings(),
        identity="pod-a:1",
    )
    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)

    offset = _fire_times(shifted, start, 1)[0] - _fire_times(base, start, 1)[0]
    assert timedelta(0) < offset < timedelta(seconds=3600)
    assert _fire_times(shifted, start, 2)[1] - _fire_times(shifted, start, 1)[0] == timedelta(seconds=3600)


@pytest.mark.parametrize(
    "raw, expected_window",
    [
        (None, 300),
        ({"window_seconds": 45}, 45),
        ({"bogus_key": 1}, 300),
        ({"window_seconds": -1}, 300),
        ("not-a-mapping", 300),
    ],
)
def test_settings_parse_and_fall_back_to_defaults_when_invalid(raw, expected_window):
    general_settings = {} if raw is None else {"scheduled_job_stagger": raw}
    assert parse_stagger_settings(general_settings).window_seconds == expected_window


def test_a_config_shaped_block_parses_whole():
    """The block arrives as plain YAML-decoded dicts, so every key has to survive that shape"""
    settings = parse_stagger_settings(
        {
            "scheduled_job_stagger": {
                "enabled": False,
                "window_seconds": 600,
                "identity": "replica-3",
                "offsets": {"update_spend_job": 0, PTU_ROLLUP_JOB_ID: 900},
            }
        }
    )

    assert (settings.enabled, settings.window_seconds, settings.identity) == (False, 600, "replica-3")
    assert dict(settings.offsets) == {"update_spend_job": 0, PTU_ROLLUP_JOB_ID: 900}


def test_identity_prefers_pod_name_and_separates_workers_on_one_host(monkeypatch):
    monkeypatch.setenv("POD_NAME", "litellm-abc")
    monkeypatch.setenv("HOSTNAME", "litellm-abc")
    identity = resolve_stagger_identity(None)

    assert identity.startswith("litellm-abc:")
    assert identity == f"litellm-abc:{os.getpid()}"

    monkeypatch.delenv("POD_NAME")
    assert resolve_stagger_identity(None).startswith("litellm-abc:")
    assert resolve_stagger_identity("explicit").startswith("explicit:")


def test_job_timing_is_logged_with_scheduled_and_actual_start(caplog):
    scheduler = _scheduler()
    attach_job_timing_logger(scheduler)
    scheduled = datetime.now(timezone.utc) - timedelta(seconds=2)
    listener = next(iter(scheduler._listeners))[0]

    with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
        listener(SimpleNamespace(job_id="update_spend_job", scheduled_run_times=[scheduled]))

    message = caplog.text
    assert "update_spend_job" in message
    assert f"scheduled_run_time={scheduled.isoformat()}" in message
    assert "actual_start_time=" in message
    assert "delay=2." in message
