from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.lens.models import (
    AgentTestCase,
    Check,
    Evidence,
    FindingDraft,
    IssueBrief,
    Lens,
    LensSettings,
    Scope,
    Worker,
)
from litellm.proxy.lens.state import can_access, claim_job, current_job, merge_finding, queue_job, renew_budget

NOW: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)


def lens() -> Lens:
    return Lens(
        id="lens",
        scope=Scope(team_id="alpha"),
        settings=LensSettings(
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
    original: Final = lens()
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
    original: Final = lens()
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
    settings: Final = LensSettings(name="Behavior", model="analysis", context="Answer using cited sources")
    assert tuple(c.id for c in settings.analysis_checks) == ("expected_behavior",)
    assert settings.sample_size is None
    assert settings.sample_percent == 100


@pytest.mark.parametrize(
    "field,value",
    (
        ("sample_percent", 0),
        ("sample_percent", 101),
        ("sample_size", 0),
        ("concurrency", 0),
        ("lookback_hours", 0),
        ("lookback_hours", 8761),
    ),
)
def test_invalid_selection_and_parallelism_are_rejected(field: str, value: int) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LensSettings.model_validate({**lens().settings.model_dump(), field: value})


def test_lease_prevents_double_claim_and_expires_with_bounded_retries() -> None:
    queued: Final = queue_job(lens(), NOW, "job")
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
    from litellm.proxy.lens.state import snapshot_finding

    original: Final = lens()
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
    spent: Final = queue_job(lens(), NOW, "job").model_copy(update={"spent": 12})
    renewed: Final = renew_budget(spent, datetime(2026, 2, 1, tzinfo=timezone.utc))
    assert renewed.spent == 0
    assert renewed.jobs == spent.jobs
    assert renew_budget(spent, NOW) is spent


@pytest.mark.parametrize("hours", (24, 168, 720, 4800, 8760))
def test_every_scan_uses_the_configured_lookback_window(hours: int) -> None:
    original: Final = lens()
    configured: Final = original.model_copy(
        update={"settings": LensSettings.model_validate({**original.settings.model_dump(), "lookback_hours": hours})}
    )
    first: Final = queue_job(configured, NOW, "first")
    assert first.jobs[0].start == NOW - timedelta(hours=hours)
    resumed: Final = configured.model_copy(update={"last_scan_at": NOW - timedelta(hours=1)})
    assert queue_job(resumed, NOW, "next").jobs[0].start == NOW - timedelta(hours=hours)


def test_finding_keeps_uncertainty_separate_from_the_main_summary() -> None:
    draft: Final = finding("run1").model_copy(update={"limitation": "The final response was not recorded."})
    saved: Final = merge_finding(lens(), draft, 1, NOW)
    assert saved.limitation == draft.limitation
    assert saved.description == draft.description


def issue_brief(problem: str) -> IssueBrief:
    return IssueBrief(
        problem=problem,
        user_goal="Open a pull request",
        what_happened="The agent replied that it lacked repository access",
        test_cases=(AgentTestCase(input="Open a PR fixing the typo", expected="A PR URL is returned"),),
    )


def test_issue_brief_survives_merges_and_refreshes_only_when_a_new_one_is_found() -> None:
    draft: Final = finding("run1").model_copy(update={"brief": issue_brief("No repo tool")})
    first: Final = merge_finding(lens(), draft, 1, NOW)
    assert first.brief == issue_brief("No repo tool")
    reviewed: Final = lens().model_copy(update={"findings": (first,)})
    assert merge_finding(reviewed, finding("run2"), 2, NOW).brief == first.brief
    refreshed: Final = finding("run2").model_copy(update={"brief": issue_brief("Token expired")})
    assert merge_finding(reviewed, refreshed, 2, NOW).brief == refreshed.brief


def test_issue_brief_requires_a_test_case() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        IssueBrief.model_validate({**issue_brief("No repo tool").model_dump(), "test_cases": ()})


@pytest.mark.parametrize("interval", (1, 2, 37, 90, 10080))
def test_custom_schedule_does_not_overlap_an_active_scan(interval: int) -> None:
    original: Final = lens()
    settings: Final = LensSettings.model_validate({**original.settings.model_dump(), "interval_minutes": interval})
    configured: Final = original.model_copy(update={"settings": settings})
    running: Final = claim_job(queue_job(configured, NOW, "first"), worker(), NOW)
    assert queue_job(running, NOW + timedelta(minutes=interval), "second") is running


@pytest.mark.parametrize("interval", (0, -1, 10081, 1.5))
def test_invalid_schedule_is_rejected(interval: float) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LensSettings.model_validate({**lens().settings.model_dump(), "interval_minutes": interval})


def test_batch_snapshot_keeps_feedback_identity_and_only_current_evidence() -> None:
    from litellm.proxy.lens.state import snapshot_finding

    original: Final = lens()
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


@pytest.mark.parametrize("explicit_reference", (False, True))
def test_issue_and_pattern_with_same_title_keep_independent_feedback(explicit_reference: bool) -> None:
    from litellm.proxy.lens.state import snapshot_finding

    original: Final = lens()
    issue: Final = merge_finding(original, finding("old"), 1, NOW).model_copy(
        update={"status": "dismissed", "reason": "Expected retry"}
    )
    reviewed: Final = original.model_copy(update={"findings": (issue,)})
    draft: Final = finding("new").model_copy(
        update={"kind": "pattern", "existing_finding_id": issue.id if explicit_reference else None}
    )
    pattern: Final = merge_finding(reviewed, draft, 1, NOW)
    assert pattern.id != issue.id
    assert pattern.kind == "pattern"
    assert pattern.status == "open" and pattern.reason == ""
    assert pattern.occurrences == ("new",)
    assert snapshot_finding(reviewed, draft, 1, NOW).id == pattern.id
    both: Final = reviewed.model_copy(update={"findings": (issue, pattern)})
    assert merge_finding(both, finding("again"), 1, NOW).id == issue.id
    assert merge_finding(both, finding("again"), 1, NOW).status == "dismissed"


def test_legacy_finding_identity_preserves_feedback_only_for_same_kind_and_check() -> None:
    import hashlib

    original: Final = lens()
    draft: Final = finding("old")
    legacy_id: Final = hashlib.sha256(f"{original.id}:{draft.check_id}:{draft.title.lower()}".encode()).hexdigest()[:24]
    legacy: Final = merge_finding(original, draft, 1, NOW).model_copy(
        update={"id": legacy_id, "status": "dismissed", "reason": "Accepted"}
    )
    reviewed: Final = original.model_copy(update={"findings": (legacy,)})
    repeated: Final = merge_finding(reviewed, finding("new"), 2, NOW)
    assert repeated.id == legacy_id
    assert repeated.status == "dismissed" and repeated.reason == "Accepted"
    other: Final = finding("new").model_copy(update={"check_id": "different", "existing_finding_id": legacy_id})
    separate: Final = merge_finding(reviewed, other, 2, NOW)
    assert separate.id != legacy_id
    assert separate.status == "open" and separate.reason == ""
