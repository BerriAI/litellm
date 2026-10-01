from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.engine.models import Check, Engine, EngineSettings, Evidence, FindingDraft, Scope, Worker
from litellm.proxy.engine.state import can_access, claim_job, current_job, merge_finding, queue_job, renew_budget

NOW: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)


def engine() -> Engine:
    return Engine(
        id="engine",
        scope=Scope(team_id="alpha"),
        settings=EngineSettings(
            name="Research", model="analysis", checks=(Check(id="retries", instruction="Find unrecovered retries"),)
        ),
        created_at=NOW,
        next_run_at=NOW,
        budget_month="2026-01",
    )


def worker(team: str = "alpha", identity: str = "worker") -> Worker:
    return Worker(id=identity, name=identity, scope=Scope(team_id=team), last_seen=NOW)


def finding(execution: str) -> FindingDraft:
    return FindingDraft(
        title="Repeated failed searches",
        description="The agent repeats the same failed search",
        check_id="retries",
        evidence=(Evidence(execution_id=execution, span_id="span", quote="timeout"),),
    )


@pytest.mark.parametrize(
    ("viewer", "target", "allowed"),
    (
        (Scope(team_id="alpha"), Scope(team_id="beta"), False),
        (Scope(team_id="alpha"), Scope(all_teams=True), False),
        (Scope(all_teams=True), Scope(team_id="alpha"), True),
        (Scope(api_key_hash="one"), Scope(api_key_hash="two"), False),
        (Scope(team_id="alpha", api_key_hash="one"), Scope(team_id="alpha"), True),
    ),
)
def test_scope_never_crosses_another_team_or_key(viewer: Scope, target: Scope, allowed: bool) -> None:
    assert can_access(viewer, target) is allowed


def test_queue_is_idempotent_and_settings_are_frozen() -> None:
    original: Final = engine()
    queued: Final = queue_job(original, NOW, "job")
    edited: Final = queued.model_copy(
        update={"settings": original.settings.model_copy(update={"model": "replacement"})}
    )

    assert queue_job(edited, NOW, "duplicate") is edited
    assert edited.jobs[0].settings.model == "analysis"
    assert (edited.jobs[0].start, edited.jobs[0].end) == (
        NOW - timedelta(hours=24),
        NOW - timedelta(minutes=2),
    )


def test_one_off_overrides_do_not_change_saved_monitoring_settings() -> None:
    original: Final = engine()
    override: Final = original.settings.model_copy(
        update={"sample_percent": 10, "sample_size": None, "concurrency": 3, "lookback_hours": 72}
    )
    queued: Final = queue_job(original, NOW, "one-off", settings=override)
    assert queued.settings == original.settings
    assert queued.jobs[0].settings == override
    assert queued.jobs[0].start == NOW - timedelta(hours=72)
    later: Final = queue_job(original, NOW + timedelta(days=1), "scheduled")
    assert later.jobs[0].settings == original.settings
    assert later.jobs[0].start == NOW


def test_behavior_description_is_sufficient_without_separate_checks() -> None:
    settings: Final = EngineSettings(name="Behavior", model="analysis", context="Answer using cited sources")
    assert tuple(c.id for c in settings.analysis_checks) == ("expected_behavior",)
    assert settings.sample_size is None
    assert settings.sample_percent == 100


@pytest.mark.parametrize(
    "field,value", (("sample_percent", 0), ("sample_percent", 101), ("sample_size", 0), ("concurrency", 0))
)
def test_invalid_selection_and_parallelism_are_rejected(field: str, value: int) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EngineSettings.model_validate({**engine().settings.model_dump(), field: value})


def test_lease_prevents_double_claim_and_expires_with_bounded_retries() -> None:
    queued: Final = queue_job(engine(), NOW, "job")
    first: Final = claim_job(queued, worker(), NOW)
    assert claim_job(first, worker(identity="second"), NOW) is first
    assert claim_job(first, worker(team="beta"), NOW + timedelta(minutes=6)) is first
    second: Final = claim_job(first, worker(identity="second"), NOW + timedelta(minutes=6))
    assert second.jobs[0].worker_id == "second"
    third: Final = claim_job(second, worker(), NOW + timedelta(minutes=12))
    exhausted: Final = claim_job(third, worker(), NOW + timedelta(minutes=18))
    assert current_job(exhausted) is None
    assert exhausted.jobs[0].status == "failed"
    assert exhausted.next_run_at > NOW + timedelta(minutes=18)


def test_replaying_evidence_does_not_reopen_but_new_occurrence_does() -> None:
    from litellm.proxy.engine.state import snapshot_finding

    original: Final = engine()
    resolved: Final = merge_finding(original, finding("run1"), 1, NOW).model_copy(update={"status": "resolved"})
    reviewed: Final = original.model_copy(update={"findings": (resolved,)})
    assert merge_finding(reviewed, finding("run1"), 1, NOW).status == "resolved"
    comparison: Final = finding("run1").model_copy(
        update={
            "evidence": (
                *finding("run1").evidence,
                Evidence(execution_id="recovered", span_id="step", quote="Recovered", role="counterexample"),
            )
        }
    )
    compared: Final = merge_finding(reviewed, comparison, 1, NOW + timedelta(days=1))
    assert compared.status == "resolved"
    assert compared.occurrences == ("run1",)
    assert compared.last_seen == resolved.last_seen
    assert compared.evidence[-1].role == "counterexample"
    assert snapshot_finding(reviewed, comparison, 1, NOW).occurrences == ("run1",)
    recurring: Final = merge_finding(reviewed, finding("run2"), 1, NOW + timedelta(days=1))
    assert recurring.status == "open"
    assert recurring.occurrences == ("run1", "run2")
    dismissed: Final = reviewed.model_copy(update={"findings": (resolved.model_copy(update={"status": "dismissed"}),)})
    assert merge_finding(dismissed, finding("run2"), 1, NOW).status == "dismissed"


def test_monthly_budget_renews_without_erasing_job_costs() -> None:
    spent: Final = queue_job(engine(), NOW, "job").model_copy(update={"spent": 12})
    renewed: Final = renew_budget(spent, datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert renewed.spent == 0
    assert renewed.jobs == spent.jobs
    assert renew_budget(spent, NOW) is spent


@pytest.mark.parametrize("hours", (24, 168, 720))
def test_every_scan_uses_the_configured_lookback_window(hours: int) -> None:
    original: Final = engine()
    configured: Final = original.model_copy(
        update={"settings": original.settings.model_copy(update={"lookback_hours": hours})}
    )
    first: Final = queue_job(configured, NOW, "first")
    assert first.jobs[0].start == NOW - timedelta(hours=hours)
    resumed: Final = configured.model_copy(update={"last_scan_at": NOW - timedelta(hours=1)})
    assert queue_job(resumed, NOW, "next").jobs[0].start == NOW - timedelta(hours=hours)


def test_finding_keeps_uncertainty_separate_from_the_main_summary() -> None:
    draft: Final = finding("run1").model_copy(update={"limitation": "The final response was not recorded."})
    saved: Final = merge_finding(engine(), draft, 1, NOW)
    assert saved.limitation == draft.limitation
    assert saved.description == draft.description


@pytest.mark.parametrize("interval", (1, 2, 37, 90, 10080))
def test_custom_schedule_does_not_overlap_an_active_scan(interval: int) -> None:
    original: Final = engine()
    settings: Final = EngineSettings.model_validate({**original.settings.model_dump(), "interval_minutes": interval})
    configured: Final = original.model_copy(update={"settings": settings})
    running: Final = claim_job(queue_job(configured, NOW, "first"), worker(), NOW)
    assert queue_job(running, NOW + timedelta(minutes=interval), "second") is running


@pytest.mark.parametrize("interval", (0, -1, 10081, 1.5))
def test_invalid_schedule_is_rejected(interval: float) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EngineSettings.model_validate({**engine().settings.model_dump(), "interval_minutes": interval})


def test_batch_snapshot_keeps_feedback_identity_and_only_current_evidence() -> None:
    from litellm.proxy.engine.state import snapshot_finding

    original: Final = engine()
    dismissed: Final = merge_finding(original, finding("old-run"), 1, NOW).model_copy(
        update={"status": "dismissed", "reason": "Expected recovery"}
    )
    saved: Final = original.model_copy(update={"findings": (dismissed,)})
    draft: Final = finding("new-run").model_copy(
        update={"title": "Updated wording", "existing_finding_id": dismissed.id}
    )
    snapshot: Final = snapshot_finding(saved, draft, 2, NOW + timedelta(days=1))
    assert snapshot.id == dismissed.id
    assert snapshot.status == "dismissed"
    assert snapshot.reason == "Expected recovery"
    assert snapshot.occurrences == ("new-run",)
    assert snapshot.title == "Updated wording"
    assert snapshot.evidence == draft.evidence
    assert snapshot.revision == 2
