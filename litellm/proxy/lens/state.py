from datetime import datetime, timedelta
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal
from uuid import uuid4

from litellm.proxy.lens.models import (
    MAX_REVIEWS,
    MAX_STEPS,
    Activity,
    Finding,
    FindingDraft,
    Job,
    Lens,
    LensSettings,
    Progress,
    Result,
    Review,
    ReviewPage,
    Sample,
    Scope,
    Step,
    Worker,
)
from litellm.proxy.lens.reviews import criteria_key


def can_access(viewer: Scope, target: Scope) -> bool:
    return viewer.all_teams or (
        not target.all_teams
        and viewer.team_id == target.team_id
        and (bool(viewer.team_id) or viewer.api_key_hash == target.api_key_hash)
    )


def current_job(lens: Lens) -> Job | None:
    return next((job for job in lens.jobs if job.status in ("queued", "running")), None)


def replace_job(lens: Lens, job: Job) -> Lens:
    return lens.model_copy(
        update=MappingProxyType({"jobs": tuple(job if old.id == job.id else old for old in lens.jobs)})
    )


SETTLE_DELAY = timedelta(minutes=2)


def scheduled_window(lens: Lens, now: datetime) -> tuple[datetime, datetime]:
    end: Final = now - SETTLE_DELAY
    floor: Final = now - timedelta(hours=lens.settings.lookback_hours)
    start: Final = max(lens.last_scan_at, floor) if lens.last_scan_at else floor
    return min(start, end), end


def next_scan_start(lens: Lens, job: Job, failed: bool) -> datetime | None:
    if failed or job.trigger == "manual" or criteria_key(lens.settings) != criteria_key(job.settings):
        return lens.last_scan_at
    return max(lens.last_scan_at or job.end, job.end)


def queue_job(
    lens: Lens,
    now: datetime,
    job_id: str,
    lookback_hours: int | None = None,
    settings: LensSettings | None = None,
    window: tuple[datetime, datetime] | None = None,
    trigger: Literal["schedule", "manual"] = "schedule",
) -> Lens:
    if current_job(lens):
        return lens
    selected: Final = settings or lens.settings
    hours: Final = lookback_hours if lookback_hours is not None else (selected.lookback_hours if settings else None)
    start, end = window or (
        (now - timedelta(hours=hours), now - SETTLE_DELAY) if hours is not None else scheduled_window(lens, now)
    )
    job: Final = Job(
        id=job_id,
        created_at=now,
        start=start,
        end=end,
        settings=selected,
        revision=lens.revision,
        trigger=trigger,
    )
    return lens.model_copy(update=MappingProxyType({"jobs": (job,)}))


def add_step(job: Job, step: Step) -> Job:
    return job.model_copy(update=MappingProxyType({"steps": (*job.steps, step)[-MAX_STEPS:]}))


def result_status(result: Result) -> Literal["completed", "failed"]:
    if result.error and not result.findings and not any(not item.cannot_assess for item in result.assessments):
        return "failed"
    return "completed"


def end_job(job: Job, status: Literal["completed", "failed", "cancelled"], now: datetime) -> Job:
    stage: Final = {"completed": "Complete", "failed": "Failed", "cancelled": "Cancelled"}[status]
    return job.model_copy(
        update=MappingProxyType({"status": status, "stage": stage, "finished_at": now, "reading": (), "activities": ()})
    )


def cancel_job(lens: Lens, now: datetime) -> Lens:
    job: Final = current_job(lens)
    if job is None:
        return lens
    return replace_job(lens, end_job(job, "cancelled", now)).model_copy(
        update=MappingProxyType({"next_run_at": now + timedelta(minutes=lens.settings.interval_minutes)})
    )


def apply_progress(job: Job, progress: Progress, now: datetime) -> Job:
    updates: Final = MappingProxyType(
        {
            "stage": job.stage if progress.stage is None else progress.stage,
            "coverage": job.coverage if progress.coverage is None else progress.coverage,
            "lease_until": now + timedelta(minutes=5),
            "reading": job.reading if progress.reading is None else progress.reading,
            "activities": update_activity(job.activities, progress.activity),
        }
    )
    renewed: Final = add_review(job.model_copy(update=updates), progress.review)
    if renewed.stage == job.stage:
        return renewed
    return add_step(renewed, Step(at=now, kind="stage", label=renewed.stage))


def update_activity(activities: tuple[Activity, ...], activity: Activity | None) -> tuple[Activity, ...]:
    if activity is None:
        return activities
    if activity.finished:
        return tuple(item for item in activities if item.id != activity.id)
    if any(item.id == activity.id for item in activities):
        return tuple(activity if item.id == activity.id else item for item in activities)
    return (*activities, activity)


def add_review(job: Job, review: Review | None) -> Job:
    if review is None:
        return job
    summary: Final = review.model_copy(update=MappingProxyType({"extraction": None, "content_version": ""}))
    return job.model_copy(
        update=MappingProxyType({"reviews": (*job.reviews, summary)[-MAX_REVIEWS:], "reviewed": job.reviewed + 1})
    )


def claim_job(lens: Lens, worker: Worker, now: datetime) -> Lens:
    job: Final = current_job(lens)
    if job is None or not can_access(worker.scope, lens.scope):
        return lens
    if job.status == "running" and job.lease_until is not None and job.lease_until > now:
        return lens
    if job.attempts >= 3:
        return replace_job(
            lens,
            end_job(job, "failed", now).model_copy(
                update=MappingProxyType({"error": "Worker disconnected repeatedly"})
            ),
        ).model_copy(update=MappingProxyType({"next_run_at": now + timedelta(minutes=lens.settings.interval_minutes)}))
    return replace_job(
        lens,
        job.model_copy(
            update=MappingProxyType(
                {
                    "status": "running",
                    "stage": "Collecting executions",
                    "worker_id": worker.id,
                    "lease_until": now + timedelta(minutes=5),
                    "attempts": job.attempts + 1,
                    "reviews": (),
                    "reviewed": 0,
                    "reading": (),
                    "activities": (),
                }
            )
        ),
    )


def renew_budget(lens: Lens, now: datetime) -> Lens:
    month: Final = now.strftime("%Y-%m")
    if lens.budget_month == month:
        return lens
    return lens.model_copy(update=MappingProxyType({"budget_month": month, "spent": 0}))


def merge_finding(
    lens: Lens,
    draft: FindingDraft,
    revision: int,
    now: datetime,
    job_id: str | None = None,
    *,
    match_titles: bool = True,
) -> Finding:
    identities: Final = frozenset((draft.existing_finding_id, *draft.merged_finding_ids))
    matches: Final = tuple(
        sorted(
            (
                finding
                for finding in lens.findings
                if finding.kind == draft.kind
                and (
                    finding.id in identities
                    or bool(identities.intersection(finding.merged_finding_ids))
                    or (
                        match_titles
                        and draft.existing_finding_id is None
                        and finding.title.casefold() == draft.title.casefold()
                        and finding.check_id == draft.check_id
                    )
                )
            ),
            key=lambda finding: (finding.first_seen, finding.id),
        )
    )
    previous: Final = tuple(
        finding for finding in matches if (finding.status, finding.reason) == (matches[0].status, matches[0].reason)
    )
    occurrences: Final = frozenset(quote.execution_id for quote in draft.evidence if quote.role == "support")
    prior_occurrences: Final = frozenset(chain.from_iterable(finding.occurrences for finding in previous))
    new_occurrence: Final = bool(occurrences - prior_occurrences)
    first: Final = previous[0] if previous else None
    checks: Final = tuple(
        sorted(frozenset(chain.from_iterable((finding.check_id, *finding.check_ids) for finding in (*previous, draft))))
    )
    return Finding(
        **draft.model_copy(
            update=MappingProxyType(
                {
                    "check_ids": checks,
                    "brief": draft.brief or (first.brief if first else None),
                    "evidence": tuple(
                        dict.fromkeys(chain.from_iterable(finding.evidence for finding in (*previous, draft)))
                    ),
                    "merged_finding_ids": tuple(
                        sorted(
                            frozenset(
                                chain.from_iterable((finding.id, *finding.merged_finding_ids) for finding in previous)
                            )
                            - ({first.id} if first else set())
                        )
                    ),
                }
            )
        ).model_dump(),
        id=first.id if first else str(uuid4()),
        status="open" if first is None or (first.status == "resolved" and new_occurrence) else first.status,
        reason=first.reason if first else "",
        first_seen=first.first_seen if first else now,
        last_seen=now if new_occurrence else max((finding.last_seen for finding in previous), default=now),
        occurrences=tuple(sorted(prior_occurrences | occurrences)),
        revision=revision,
        investigation_runs=tuple(
            dict.fromkeys(
                (
                    *chain.from_iterable(finding.investigation_runs for finding in previous),
                    *((job_id,) if job_id and new_occurrence else ()),
                )
            )
        ),
    )


def snapshot_finding(lens: Lens, draft: FindingDraft, revision: int, now: datetime) -> Finding:
    merged: Final = merge_finding(lens, draft, revision, now)
    return Finding.model_validate(
        MappingProxyType(
            {
                **merged.model_dump(),
                **draft.model_dump(),
                "revision": revision,
                "first_seen": now,
                "last_seen": now,
                "occurrences": tuple(sorted(frozenset(e.execution_id for e in draft.evidence if e.role == "support"))),
            }
        )
    )


def without_attributes(sample: Sample) -> Sample:
    executions: Final = tuple(e.model_copy(update=MappingProxyType({"metadata": ()})) for e in sample.executions)
    return sample.model_copy(update=MappingProxyType({"executions": executions}))


def summarized_job(job: Job) -> Job:
    sample: Final = without_attributes(job.sample) if job.sample else None
    return job.model_copy(update=MappingProxyType({"reviews": (), "sample": sample}))


def summarized(lens: Lens) -> Lens:
    return lens.model_copy(update=MappingProxyType({"jobs": tuple(summarized_job(job) for job in lens.jobs)}))


def reviews_after(job: Job, after: int) -> ReviewPage:
    first_kept: Final = job.reviewed - len(job.reviews)
    return ReviewPage(reviews=job.reviews[max(0, after - first_kept) :], reviewed=job.reviewed)
