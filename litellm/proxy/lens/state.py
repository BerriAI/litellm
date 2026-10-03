import hashlib
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final, Literal

from litellm.proxy.lens.models import (
    MAX_STEPS,
    Finding,
    FindingDraft,
    Job,
    Lens,
    LensSettings,
    Scope,
    Step,
    Worker,
)


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
    if failed or job.trigger == "manual":
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


def claim_job(lens: Lens, worker: Worker, now: datetime) -> Lens:
    job: Final = current_job(lens)
    if job is None or not can_access(worker.scope, lens.scope):
        return lens
    if job.status == "running" and job.lease_until is not None and job.lease_until > now:
        return lens
    if job.attempts >= 3:
        return replace_job(
            lens,
            job.model_copy(
                update=MappingProxyType(
                    {
                        "status": "failed",
                        "stage": "Failed",
                        "error": "Worker disconnected repeatedly",
                        "finished_at": now,
                    }
                )
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
                }
            )
        ),
    )


def renew_budget(lens: Lens, now: datetime) -> Lens:
    month: Final = now.strftime("%Y-%m")
    if lens.budget_month == month:
        return lens
    return lens.model_copy(update=MappingProxyType({"budget_month": month, "spent": 0}))


def merge_finding(lens: Lens, draft: FindingDraft, revision: int, now: datetime) -> Finding:
    legacy_identity: Final = hashlib.sha256(f"{lens.id}:{draft.check_id}:{draft.title.lower()}".encode()).hexdigest()[
        :24
    ]
    identity: Final = hashlib.sha256(
        f"{lens.id}:{draft.check_id}:{draft.kind}:{draft.title.lower()}".encode()
    ).hexdigest()[:24]
    identities: Final = (draft.existing_finding_id, identity, legacy_identity)
    previous: Final = next(
        (f for f in lens.findings if f.id in identities and f.kind == draft.kind and f.check_id == draft.check_id),
        None,
    )
    occurrences: Final = tuple(sorted(frozenset(e.execution_id for e in draft.evidence if e.role == "support")))
    if previous is None:
        return Finding(
            title=draft.title,
            description=draft.description,
            check_id=draft.check_id,
            kind=draft.kind,
            priority=draft.priority,
            suggestion=draft.suggestion,
            limitation=draft.limitation,
            brief=draft.brief,
            evidence=draft.evidence,
            existing_finding_id=draft.existing_finding_id,
            id=identity,
            first_seen=now,
            last_seen=now,
            occurrences=occurrences,
            revision=revision,
        )
    new_occurrence: Final = bool(frozenset(occurrences) - frozenset(previous.occurrences))
    return previous.model_copy(
        update=MappingProxyType(
            {
                "last_seen": now if new_occurrence else previous.last_seen,
                "occurrences": tuple(sorted(frozenset((*previous.occurrences, *occurrences)))),
                "evidence": tuple(
                    MappingProxyType(
                        {(e.execution_id, e.span_id, e.quote): e for e in (*previous.evidence, *draft.evidence)}
                    ).values()
                )[-20:],
                "status": "open" if previous.status == "resolved" and new_occurrence else previous.status,
                "brief": draft.brief or previous.brief,
            }
        )
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
