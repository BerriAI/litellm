import hashlib
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final

from litellm.proxy.engine.models import Engine, EngineSettings, Finding, FindingDraft, Job, Scope, Worker


def can_access(viewer: Scope, target: Scope) -> bool:
    return viewer.all_teams or (
        not target.all_teams
        and viewer.team_id == target.team_id
        and (bool(viewer.team_id) or viewer.api_key_hash == target.api_key_hash)
    )


def current_job(engine: Engine) -> Job | None:
    return next((job for job in engine.jobs if job.status in ("queued", "running")), None)


def replace_job(engine: Engine, job: Job) -> Engine:
    return engine.model_copy(
        update=MappingProxyType({"jobs": tuple(job if old.id == job.id else old for old in engine.jobs)})
    )


def queue_job(
    engine: Engine,
    now: datetime,
    job_id: str,
    lookback_hours: int | None = None,
    settings: EngineSettings | None = None,
) -> Engine:
    if current_job(engine):
        return engine
    selected: Final = settings or engine.settings
    job: Final = Job(
        id=job_id,
        created_at=now,
        start=now - timedelta(hours=lookback_hours if lookback_hours is not None else selected.lookback_hours),
        end=now - timedelta(minutes=2),
        settings=selected,
        revision=engine.revision,
    )
    return engine.model_copy(update=MappingProxyType({"jobs": (job,)}))


def claim_job(engine: Engine, worker: Worker, now: datetime) -> Engine:
    job: Final = current_job(engine)
    if job is None or not can_access(worker.scope, engine.scope):
        return engine
    if job.status == "running" and job.lease_until is not None and job.lease_until > now:
        return engine
    if job.attempts >= 3:
        return replace_job(
            engine,
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
        ).model_copy(
            update=MappingProxyType({"next_run_at": now + timedelta(minutes=engine.settings.interval_minutes)})
        )
    return replace_job(
        engine,
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


def renew_budget(engine: Engine, now: datetime) -> Engine:
    month: Final = now.strftime("%Y-%m")
    if engine.budget_month == month:
        return engine
    return engine.model_copy(update=MappingProxyType({"budget_month": month, "spent": 0}))


def merge_finding(engine: Engine, draft: FindingDraft, revision: int, now: datetime) -> Finding:
    identity: Final = hashlib.sha256(f"{engine.id}:{draft.check_id}:{draft.title.lower()}".encode()).hexdigest()[:24]
    previous: Final = next((f for f in engine.findings if f.id == (draft.existing_finding_id or identity)), None)
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
            }
        )
    )


def snapshot_finding(engine: Engine, draft: FindingDraft, revision: int, now: datetime) -> Finding:
    merged: Final = merge_finding(engine, draft, revision, now)
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
