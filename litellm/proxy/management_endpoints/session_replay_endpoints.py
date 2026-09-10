"""Admin endpoints to replay a recorded session against several model arms and judge it."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Annotated, Final, Protocol

import fastapi
from fastapi import APIRouter, Depends, HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm.integrations.shadow_eval_logger import (
    _request_mutating_guardrail_ran,  # pyright: ignore[reportPrivateUsage]  # one owner for the guardrail-mode reading
)
from litellm.litellm_core_utils.internal_call_metadata import sanitized_forwardable_call_metadata
from litellm.proxy._types import CommonProxyErrors, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.auto_router_endpoints import (
    _is_unique_violation,  # pyright: ignore[reportPrivateUsage]  # one owner for the unique-violation read
)
from litellm.proxy.session_replay.executor import replay_session
from litellm.proxy.session_replay.transcript import RecordedBody, ReplayTranscript, build_transcript
from litellm.types.management_endpoints.session_replay_endpoints import (
    SessionReplayArmResponse,
    SessionReplayArmSpec,
    SessionReplayFidelityResponse,
    SessionReplayJobListResponse,
    SessionReplayJobResponse,
    SessionReplayVerdictResponse,
    StartSessionReplayRequest,
)
from litellm.types.utils import SESSION_REPLAY_ARM_CALL_ORIGIN, SESSION_REPLAY_JUDGE_CALL_ORIGIN

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router

router: Final = APIRouter()

MAX_CONCURRENT_SESSION_REPLAYS: Final = 4
STALE_JOB_AFTER: Final = timedelta(hours=3)
_REPLAYABLE_CALL_TYPE: Final = "anthropic_messages"


class _SessionReplayJobRow(Protocol):
    @property
    def id(self) -> str: ...


class _SessionReplayJobTable(Protocol):
    async def create(self, data: Mapping[str, object]) -> _SessionReplayJobRow: ...

    async def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...


def _json_column(value: object) -> object:
    """prisma is an optional dependency of the base install, so it stays a lazy import."""
    from prisma import Json

    return Json(value)


def _session_replay_jobs(prisma_client: PrismaClient) -> _SessionReplayJobTable:
    return prisma_client.db.litellm_sessionreplayjob  # pyright: ignore[reportAttributeAccessIssue]  # generated client


async def _query_raw(prisma_client: PrismaClient, query: str, *args: object) -> Sequence[Mapping[str, object]]:
    return await prisma_client.db.query_raw(query, *args)


@dataclass(frozen=True, slots=True)
class _SourceRequest:
    request_id: str
    body: RecordedBody
    recorded_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class _SourceUnavailable:
    status_code: int
    reason: str


def _as_mapping(value: object) -> Mapping[str, object] | None:
    """Read a jsonb column back, which query_raw hands over as a dict or as raw JSON text."""
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded: Final = json.loads(value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


async def _load_source_request(prisma_client: PrismaClient, session_id: str) -> _SourceRequest | _SourceUnavailable:
    """Pick the recorded request whose stored body carries the fullest transcript.

    Turn N's body already embeds turns 1..N-1, so the row with the most prompt tokens is
    the whole conversation and every earlier row is a prefix of it.
    """
    rows: Final = await _query_raw(
        prisma_client,
        """
        SELECT request_id, proxy_server_request, prompt_tokens, metadata
        FROM "LiteLLM_SpendLogs"
        WHERE session_id = $1 AND call_type = $2
        ORDER BY prompt_tokens DESC
        LIMIT 1
        """,
        session_id,
        _REPLAYABLE_CALL_TYPE,
    )
    if not rows:
        return _SourceUnavailable(
            status_code=status.HTTP_404_NOT_FOUND,
            reason=f"no {_REPLAYABLE_CALL_TYPE} spend logs found for session {session_id}",
        )
    row: Final = rows[0]
    recorded: Final = _as_mapping(row.get("proxy_server_request"))
    if recorded is None or not recorded.get("messages"):
        return _SourceUnavailable(
            status_code=status.HTTP_409_CONFLICT,
            reason=(
                "the recorded request body was not stored, so there is nothing to replay. "
                "Set general_settings.store_prompts_in_spend_logs to record it"
            ),
        )
    metadata: Final = _as_mapping(row.get("metadata"))
    if metadata is not None and _request_mutating_guardrail_ran(metadata):
        return _SourceUnavailable(
            status_code=status.HTTP_409_CONFLICT,
            reason="a request-mutating guardrail ran on this session, so the stored body predates the rewrite",
        )
    prompt_tokens: Final = row.get("prompt_tokens")
    return _SourceRequest(
        request_id=str(row.get("request_id") or ""),
        body=RecordedBody.model_validate(recorded),
        recorded_prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else 0,
    )


def _arm_caller(llm_router: Router, parent_metadata: Mapping[str, object]):
    async def call(request: Mapping[str, object]) -> Mapping[str, object]:
        return await llm_router.aanthropic_messages(
            **request,
            metadata=sanitized_forwardable_call_metadata(parent_metadata, SESSION_REPLAY_ARM_CALL_ORIGIN),
            num_retries=0,
            fallbacks=[],  # mutable-ok: SDK kwarg
        )

    return call


def _judge_caller(llm_router: Router, judge_model: str, parent_metadata: Mapping[str, object]):
    async def call(system_prompt: str, user_prompt: str) -> str:
        response: Final = await llm_router.aanthropic_messages(
            model=judge_model,
            max_tokens=800,
            system=[{"type": "text", "text": system_prompt}],  # mutable-ok: SDK kwarg
            messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],  # mutable-ok: SDK kwarg
            stream=False,
            metadata=sanitized_forwardable_call_metadata(parent_metadata, SESSION_REPLAY_JUDGE_CALL_ORIGIN),
            num_retries=0,
            fallbacks=[],  # mutable-ok: SDK kwarg
        )
        blocks: Final = response.get("content") if isinstance(response, Mapping) else None
        if not isinstance(blocks, Sequence):
            return ""
        return "".join(
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "text"
        )

    return call


async def _reap_stale_jobs(prisma_client: PrismaClient) -> None:
    """Fail jobs whose pod died mid-run, so the partial unique index does not lock a session out forever.

    The heartbeat only ticks between turns, and a single arm or judge call can legitimately run
    for the router's whole timeout, so the window sits well above one call rather than near it.
    Reaping is still a guess, which is why the terminal writes below are status-guarded: a job
    reaped while it was in fact alive cannot overwrite the row a later replay now owns.
    """
    cutoff: Final = datetime.now(timezone.utc) - STALE_JOB_AFTER
    await _session_replay_jobs(prisma_client).update_many(
        where={"status": "running", "updated_at": {"lt": cutoff}},  # mutable-ok: Prisma filter
        data={  # mutable-ok: Prisma update input
            "status": "failed",
            "error": "abandoned: no progress recorded",
            "finished_at": datetime.now(timezone.utc),
        },
    )


async def _run_job(
    job_id: str,
    transcript: ReplayTranscript,
    arms: Sequence[SessionReplayArmSpec],
    request: StartSessionReplayRequest,
    recorded_prompt_tokens: int,
    llm_router: Router,
    prisma_client: PrismaClient,
    parent_metadata: Mapping[str, object],
) -> None:
    jobs: Final = _session_replay_jobs(prisma_client)

    async def on_progress(turns_completed: int) -> None:
        await jobs.update_many(
            where={"id": job_id, "status": "running"},  # mutable-ok: Prisma filter
            data={"turns_completed": turns_completed},  # mutable-ok: Prisma update input
        )

    try:
        outcome: Final = await replay_session(
            transcript=transcript,
            arms=arms,
            max_turns=request.max_turns,
            session_id=request.session_id,
            recorded_prompt_tokens=recorded_prompt_tokens,
            call_model=_arm_caller(llm_router, parent_metadata),
            call_judge=_judge_caller(llm_router, request.judge_model, parent_metadata),
            rng=random.Random(),
            on_progress=on_progress,
        )
    except Exception as exc:  # noqa: BLE001  # the job records its own failure rather than dying silently
        verbose_proxy_logger.exception("session replay job %s failed", job_id)
        await jobs.update_many(
            where={"id": job_id, "status": "running"},  # mutable-ok: Prisma filter
            data={  # mutable-ok: Prisma update input
                "status": "failed",
                "error": str(exc)[:1000],
                "finished_at": datetime.now(timezone.utc),
            },
        )
        return
    await jobs.update_many(
        where={"id": job_id, "status": "running"},  # mutable-ok: Prisma filter
        data={  # mutable-ok: Prisma update input
            "status": "completed",
            "result": _json_column(outcome.model_dump()),
            "turns_completed": sum(len(run.turns) for run in outcome.arms),
            "finished_at": datetime.now(timezone.utc),
        },
    )


def _require_admin_writer(user_api_key_dict: UserAPIKeyAuth, action: str) -> None:
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail=f"Only a proxy admin can {action}")


def _require_admin_viewer(user_api_key_dict: UserAPIKeyAuth, action: str) -> None:
    if user_api_key_dict.user_role not in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    ):
        raise HTTPException(status_code=403, detail=f"Only proxy admin roles can {action}")


def _job_response(row: Mapping[str, object]) -> SessionReplayJobResponse:
    payload: Final = _as_mapping(row.get("result")) or {}  # mutable-ok: stand-in for an unfinished job's result
    verdict: Final = payload.get("verdict")
    fidelity: Final = payload.get("fidelity")
    arms: Final = payload.get("arms")
    human_asks: Final = payload.get("human_asks")
    return SessionReplayJobResponse(
        job_id=str(row.get("id") or ""),
        session_id=str(row.get("session_id") or ""),
        status=str(row.get("status") or "running"),  # pyright: ignore[reportArgumentType]  # column is the literal set
        judge_model=str(row.get("judge_model") or ""),
        max_turns=int(row.get("max_turns") or 0),  # pyright: ignore[reportArgumentType]  # numeric column
        source_request_id=str(row["source_request_id"]) if row.get("source_request_id") else None,
        turns_completed=int(row.get("turns_completed") or 0),  # pyright: ignore[reportArgumentType]  # numeric column
        created_by=str(row["created_by"]) if row.get("created_by") else None,
        created_at=str(row["created_at"]) if row.get("created_at") else None,
        finished_at=str(row["finished_at"]) if row.get("finished_at") else None,
        error=str(row["error"]) if row.get("error") else None,
        human_asks=tuple(str(ask) for ask in human_asks) if isinstance(human_asks, Sequence) else (),
        arms=tuple(SessionReplayArmResponse.model_validate(arm) for arm in arms) if isinstance(arms, Sequence) else (),
        verdict=SessionReplayVerdictResponse.model_validate(verdict) if isinstance(verdict, Mapping) else None,
        fidelity=SessionReplayFidelityResponse.model_validate(fidelity) if isinstance(fidelity, Mapping) else None,
    )


@router.post(
    "/auto_router/session_replay/start",
    tags=("auto router",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=SessionReplayJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_session_replay(
    data: StartSessionReplayRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SessionReplayJobResponse:
    """
    Replay a recorded session against several model arms and judge the end results blind.

    The recorded user and tool_result turns are the fixed environment; each arm generates
    its own assistant turns, so the arms diverge from the recording and from each other.
    Judging compares the final answer each arm reached against the human turns the
    recording actually contained.

    Spend logs truncate stored prompts, so the replayed request is smaller than the one
    that was recorded. Every job reports that gap under `fidelity`; read a verdict against
    it rather than as a clean comparison.

    A replay issues one billable call per recorded turn per arm, so `max_turns` bounds it
    and one session can have only one running job.
    """
    from litellm.proxy.proxy_server import llm_router, prisma_client

    _require_admin_writer(user_api_key_dict, "start a session replay")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    if llm_router is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.no_llm_router.value)

    await _reap_stale_jobs(prisma_client)
    running: Final = await _query_raw(
        prisma_client,
        'SELECT COUNT(*)::int AS running FROM "LiteLLM_SessionReplayJob" WHERE status = $1',
        "running",
    )
    in_flight: Final = running[0].get("running") if running else 0
    if isinstance(in_flight, int) and in_flight >= MAX_CONCURRENT_SESSION_REPLAYS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"{in_flight} session replays already running, limit is {MAX_CONCURRENT_SESSION_REPLAYS}",
        )

    source: Final = await _load_source_request(prisma_client, data.session_id)
    if isinstance(source, _SourceUnavailable):
        raise HTTPException(status_code=source.status_code, detail=source.reason)

    transcript: Final = build_transcript(source.body)
    if not transcript.user_turns:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="the recorded session has no user turns to replay",
        )

    try:
        created: Final = await _session_replay_jobs(prisma_client).create(
            data={  # mutable-ok: Prisma create input
                "session_id": data.session_id,
                "status": "running",
                "arms": _json_column([arm.model_dump() for arm in data.arms]),  # mutable-ok: Prisma Json column payload
                "judge_model": data.judge_model,
                "max_turns": data.max_turns,
                "source_request_id": source.request_id,
                "created_by": user_api_key_dict.user_id,
            }
        )
    except Exception as exc:  # noqa: BLE001  # separate the lost race from every other create failure
        if _is_unique_violation(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"a replay is already running for session {data.session_id}",
            ) from exc
        verbose_proxy_logger.exception("could not create session replay job")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"could not create the replay job: {exc}",
        ) from exc

    parent_metadata: Final = {  # mutable-ok: forwardable call metadata, an SDK kwarg
        "user_api_key_user_id": user_api_key_dict.user_id
    }  # mutable-ok: forwardable call metadata, an SDK kwarg
    asyncio.create_task(  # detached job; its own handler records terminal state on the row
        _run_job(
            job_id=created.id,
            transcript=transcript,
            arms=data.arms,
            request=data,
            recorded_prompt_tokens=source.recorded_prompt_tokens,
            llm_router=llm_router,
            prisma_client=prisma_client,
            parent_metadata=parent_metadata,
        )
    )
    return SessionReplayJobResponse(
        job_id=created.id,
        session_id=data.session_id,
        status="running",
        judge_model=data.judge_model,
        max_turns=data.max_turns,
        source_request_id=source.request_id,
        created_by=user_api_key_dict.user_id,
        human_asks=transcript.human_asks,
        fidelity=SessionReplayFidelityResponse(
            truncated_strings=transcript.truncated_strings,
            recorded_prompt_tokens=source.recorded_prompt_tokens,
        ),
    )


@router.get(
    "/auto_router/session_replay/{job_id}",
    tags=("auto router",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=SessionReplayJobResponse,
)
async def get_session_replay(
    job_id: str,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SessionReplayJobResponse:
    """Read one session replay job, including its verdict once it has finished."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin_viewer(user_api_key_dict, "read a session replay")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    rows: Final = await _query_raw(
        prisma_client,
        'SELECT * FROM "LiteLLM_SessionReplayJob" WHERE id = $1',
        job_id,
    )
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"no session replay job {job_id}")
    return _job_response(rows[0])


@router.get(
    "/auto_router/session_replay",
    tags=("auto router",),
    dependencies=(Depends(user_api_key_auth),),
    response_model=SessionReplayJobListResponse,
)
async def list_session_replays(
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    session_id: str | None = fastapi.Query(default=None, description="Only jobs for this session"),
    limit: int = fastapi.Query(default=25, ge=1, le=100),
) -> SessionReplayJobListResponse:
    """List session replay jobs, newest first."""
    from litellm.proxy.proxy_server import prisma_client

    _require_admin_viewer(user_api_key_dict, "list session replays")
    if prisma_client is None:
        raise HTTPException(status_code=500, detail=CommonProxyErrors.db_not_connected_error.value)
    rows: Final = (
        await _query_raw(
            prisma_client,
            'SELECT * FROM "LiteLLM_SessionReplayJob" WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2',
            session_id,
            limit,
        )
        if session_id
        else await _query_raw(
            prisma_client,
            'SELECT * FROM "LiteLLM_SessionReplayJob" ORDER BY created_at DESC LIMIT $1',
            limit,
        )
    )
    return SessionReplayJobListResponse(jobs=tuple(_job_response(row) for row in rows))
