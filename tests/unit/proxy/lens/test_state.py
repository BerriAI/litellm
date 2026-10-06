from datetime import datetime, timedelta, timezone
from functools import reduce
from typing import Final, Literal

import pytest

from litellm.proxy.lens.models import (
    MAX_REVIEWS,
    MAX_STEPS,
    Activity,
    AgentTestCase,
    Check,
    Coverage,
    Evidence,
    Execution,
    FindingDraft,
    InFlight,
    IssueBrief,
    Job,
    Lens,
    LensSettings,
    MetadataFilter,
    Progress,
    Result,
    Review,
    RunAssessment,
    Sample,
    Scope,
    Step,
    Worker,
)
from litellm.proxy.lens.state import (
    add_review,
    add_step,
    apply_progress,
    can_access,
    cancel_job,
    claim_job,
    current_job,
    end_job,
    merge_finding,
    next_scan_start,
    queue_job,
    renew_budget,
    replace_job,
    result_status,
    reviews_after,
    summarized,
)

NOW: Final = datetime(2026, 1, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("has_finding", "assessable", "error", "expected"),
    (
        (True, False, "One candidate exhausted its retries", "completed"),
        (False, True, "One review exhausted its retries", "completed"),
        (False, False, "Every review exhausted its retries", "failed"),
        (False, False, "", "completed"),
    ),
)
def test_partial_results_are_completed_while_total_failure_remains_failed(
    has_finding: bool, assessable: bool, error: str, expected: str
) -> None:
    result: Final = Result(
        findings=(finding("run"),) if has_finding else (),
        assessments=(RunAssessment(execution_id="run", cannot_assess=not assessable),),
        coverage=Coverage(screened=1, unassessable=int(not assessable)),
        error=error,
    )
    assert result_status(result) == expected


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
def test_first_scan_covers_the_configured_lookback_window(hours: int) -> None:
    original: Final = lens()
    configured: Final = original.model_copy(
        update={"settings": LensSettings.model_validate({**original.settings.model_dump(), "lookback_hours": hours})}
    )
    first: Final = queue_job(configured, NOW, "first")
    assert first.jobs[0].start == NOW - timedelta(hours=hours)
    assert first.jobs[0].trigger == "schedule"


def test_later_scheduled_scans_only_cover_traces_since_the_last_scan() -> None:
    resumed: Final = lens().model_copy(update={"last_scan_at": NOW - timedelta(hours=1)})
    job: Final = queue_job(resumed, NOW, "next").jobs[0]
    assert job.start == NOW - timedelta(hours=1)
    assert job.end == NOW - timedelta(minutes=2)


def test_a_scan_after_a_long_outage_never_reaches_past_the_lookback_window() -> None:
    stale: Final = lens().model_copy(update={"last_scan_at": NOW - timedelta(days=400)})
    assert queue_job(stale, NOW, "next").jobs[0].start == NOW - timedelta(hours=stale.settings.lookback_hours)


def test_run_now_with_an_exact_window_scans_that_window_and_is_marked_manual() -> None:
    window: Final = (NOW - timedelta(hours=5), NOW - timedelta(hours=3))
    job: Final = queue_job(lens(), NOW, "manual", window=window, trigger="manual").jobs[0]
    assert (job.start, job.end) == window
    assert job.trigger == "manual"


def test_steps_keep_only_the_most_recent_entries() -> None:
    job: Final = queue_job(lens(), NOW, "job").jobs[0]
    steps: Final = tuple(Step(at=NOW, kind="stage", label=f"step {i}") for i in range(MAX_STEPS + 5))
    grown: Final = reduce(add_step, steps, job)
    assert len(grown.steps) == MAX_STEPS
    assert grown.steps[0].label == "step 5"
    assert grown.steps[-1].label == f"step {MAX_STEPS + 4}"


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


@pytest.mark.parametrize("interval", (0, -1, 1.5))
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


def test_only_successful_scheduled_scans_move_the_next_scan_forward() -> None:
    previous: Final = lens().model_copy(update={"last_scan_at": NOW - timedelta(hours=3)})
    scheduled: Final = queue_job(previous, NOW, "scheduled").jobs[0]
    manual: Final = queue_job(
        previous, NOW, "manual", window=(NOW - timedelta(hours=2), NOW - timedelta(hours=1)), trigger="manual"
    ).jobs[0]
    assert next_scan_start(previous, scheduled, failed=False) == scheduled.end
    assert next_scan_start(previous, scheduled, failed=True) == previous.last_scan_at
    assert next_scan_start(previous, manual, failed=False) == previous.last_scan_at


@pytest.mark.parametrize("field", ("lookback_hours", "interval_minutes"))
def test_calendar_overflow_is_rejected_without_the_old_history_and_interval_caps(field: str) -> None:
    from pydantic import ValidationError

    accepted: Final = LensSettings.model_validate({**lens().settings.model_dump(), field: 100000})
    assert getattr(accepted, field) == 100000
    with pytest.raises(ValidationError, match="supported calendar range"):
        LensSettings.model_validate({**lens().settings.model_dump(), field: 10**30})


def review(index: int) -> Review:
    return Review(
        execution_id=f"run-{index}", trace_id="t", agent="support", name="task", model="analysis", duration_ms=1, at=NOW
    )


def test_reviews_keep_the_newest_window_while_counting_every_review() -> None:
    job: Final = queue_job(lens(), NOW, "job").jobs[0]
    grown: Final = reduce(add_review, tuple(review(i) for i in range(MAX_REVIEWS + 3)), job)
    assert grown.reviewed == MAX_REVIEWS + 3
    assert len(grown.reviews) == MAX_REVIEWS
    assert grown.reviews[0].execution_id == "run-3"
    assert grown.reviews[-1].execution_id == f"run-{MAX_REVIEWS + 2}"


def test_reclaimed_run_starts_its_review_history_over() -> None:
    queued: Final = queue_job(lens(), NOW, "job")
    first: Final = claim_job(queued, worker(), NOW)
    reviewed: Final = replace_job(first, reduce(add_review, (review(0), review(1)), first.jobs[0]))
    stalled: Final = reviewed.jobs[0].model_copy(
        update={"reading": (InFlight(execution_id="run-2", trace_id="t", agent="support", started_at=NOW),)}
    )
    reclaimed: Final = claim_job(replace_job(reviewed, stalled), worker(identity="other"), NOW + timedelta(minutes=6))
    job: Final = reclaimed.jobs[0]
    assert job.worker_id == "other"
    assert (job.reviews, job.reviewed, job.reading) == ((), 0, ())
    replayed: Final = reduce(add_review, (review(0), review(1)), job)
    assert replayed.reviewed == len(replayed.reviews) == 2


def test_progress_without_a_review_leaves_the_review_history_alone() -> None:
    job: Final = add_review(queue_job(lens(), NOW, "job").jobs[0], review(0))
    assert add_review(job, None) == job


def reviewed_job() -> Job:
    execution: Final = Execution(
        id="run-0",
        source="traces",
        trace_id="t",
        team_id="alpha",
        name="task",
        start_time="2026-01-15 00:00:00",
        span_count=3,
        service="support",
        metadata=(MetadataFilter(key="gen_ai.agent.name", value="support"),),
    )
    job: Final = (
        queue_job(lens(), NOW, "job")
        .jobs[0]
        .model_copy(update={"sample": Sample(executions=(execution,), eligible=4, selected=1)})
    )
    timed: Final = tuple(review(i).model_copy(update={"at": NOW + timedelta(seconds=i)}) for i in range(3))
    return reduce(add_review, timed, job)


def test_summary_drops_reviews_and_run_attributes_but_keeps_counts_and_run_identity() -> None:
    job: Final = reviewed_job()
    listed: Final = summarized(lens().model_copy(update={"jobs": (job,)})).jobs[0]
    assert listed.reviews == ()
    assert listed.reviewed == job.reviewed == 3
    assert listed.sample is not None and job.sample is not None
    assert listed.sample.executions[0].metadata == ()
    assert (
        listed.sample.executions[0].model_copy(update={"metadata": job.sample.executions[0].metadata})
        == (job.sample.executions[0])
    )
    assert listed.model_copy(update={"reviews": job.reviews, "sample": job.sample}) == job


def test_review_polling_returns_only_reviews_after_the_cursor_even_when_they_finished_out_of_order() -> None:
    job: Final = reduce(add_review, (review(5).model_copy(update={"at": NOW - timedelta(hours=1)}),), reviewed_job())
    assert reviews_after(job, 0).reviews == job.reviews
    assert [r.execution_id for r in reviews_after(job, 2).reviews] == ["run-2", "run-5"]
    assert reviews_after(job, 4).reviews == ()
    assert reviews_after(job, 4).reviewed == 4


def test_review_polling_after_the_window_moved_on_returns_what_is_still_kept() -> None:
    job: Final = reduce(add_review, tuple(review(i) for i in range(MAX_REVIEWS + 10)), reviewed_job())
    page: Final = reviews_after(job, 5)
    assert page.reviews == job.reviews
    assert page.reviewed == MAX_REVIEWS + 13
    assert [r.execution_id for r in reviews_after(job, page.reviewed - 2).reviews] == [
        f"run-{MAX_REVIEWS + 8}",
        f"run-{MAX_REVIEWS + 9}",
    ]


def in_flight(execution: str) -> InFlight:
    return InFlight(execution_id=execution, trace_id="t", agent="support", started_at=NOW)


def reading_job() -> Job:
    running: Final = claim_job(queue_job(lens(), NOW, "job"), worker(), NOW).jobs[0]
    return apply_progress(running, Progress(stage=running.stage, reading=(in_flight("a"), in_flight("b"))), NOW)


def test_progress_replaces_the_in_flight_runs_and_old_workers_leave_them_alone() -> None:
    job: Final = reading_job()
    assert [r.execution_id for r in job.reading] == ["a", "b"]
    finished: Final = apply_progress(job, Progress(stage=job.stage, review=review(0), reading=(in_flight("b"),)), NOW)
    assert [r.execution_id for r in finished.reading] == ["b"]
    assert finished.reviewed == 1
    assert apply_progress(job, Progress(stage=job.stage, review=review(1)), NOW).reading == job.reading
    assert apply_progress(job, Progress(stage=job.stage, reading=()), NOW).reading == ()


@pytest.mark.parametrize("status", ("completed", "failed", "cancelled"))
def test_finished_jobs_stop_showing_runs_in_flight(status: Literal["completed", "failed", "cancelled"]) -> None:
    ended: Final = end_job(reading_job(), status, NOW)
    assert ended.status == status
    assert ended.finished_at == NOW
    assert ended.reading == ()


def test_cancel_and_repeated_disconnects_clear_runs_in_flight() -> None:
    reading: Final = replace_job(queue_job(lens(), NOW, "job"), reading_job())
    cancelled: Final = cancel_job(reading, NOW).jobs[0]
    assert (cancelled.status, cancelled.reading) == ("cancelled", ())
    abandoned: Final = reading.model_copy(update={"jobs": (reading.jobs[0].model_copy(update={"attempts": 3}),)})
    expired: Final = claim_job(abandoned, worker(), NOW + timedelta(minutes=10)).jobs[0]
    assert (expired.status, expired.reading) == ("failed", ())


def test_activity_updates_preserve_coverage_reviews_and_other_concurrent_lanes() -> None:
    initial: Final = add_review(reading_job(), review(0))
    first: Final = Activity(id="review:one", phase="review", label="Review one", execution_ids=("one",), started_at=NOW)
    second: Final = Activity(id="group:one", phase="group", label="Compare batch", started_at=NOW)
    started: Final = apply_progress(
        apply_progress(initial, Progress(activity=first), NOW), Progress(activity=second), NOW
    )
    reading: Final = first.model_copy(update={"operations": ("python",)})
    updated: Final = apply_progress(started, Progress(activity=reading), NOW)
    assert updated.activities == (reading, second)
    assert (updated.stage, updated.coverage, updated.reviews, updated.reading) == (
        initial.stage,
        initial.coverage,
        initial.reviews,
        initial.reading,
    )
    assert updated.reviewed == initial.reviewed
    finished: Final = apply_progress(updated, Progress(activity=reading.model_copy(update={"finished": True})), NOW)
    assert finished.activities == (second,)
    assert end_job(updated, "cancelled", NOW).activities == ()
    expired: Final = replace_job(queue_job(lens(), NOW, "job"), updated.model_copy(update={"lease_until": NOW}))
    assert claim_job(expired, worker(), NOW).jobs[0].activities == ()
