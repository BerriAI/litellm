"""
Session CRUD endpoints — POST/GET/DELETE /v2/sessions{,/<id>}, plus
``/followup`` (smart inject vs. new-run) and ``/conversation`` (stateless
snapshot of all events across runs).

Sessions are VM-backed. ``POST /v2/sessions``:
  1. validates the parent agent exists + caller owns it
  2. resolves repos/env_vars (overlay caller-provided over agent defaults)
  3. inserts the session row in ``provisioning`` status
  4. mints a daemon JWT, stores its hash for revocation
  5. spawns the VM provider call as a background task (no client wait)
  6. returns the session JSON immediately so the client can subscribe

The daemon JWT is returned exactly once on create — subsequent reads
return ``daemon_token=null``. Callers that lose it must DELETE and recreate.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import prisma
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import ORJSONResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_session_endpoints.auth import (
    hash_daemon_token,
    mint_daemon_token,
)
from litellm.proxy.agent_session_endpoints.constants import (
    DEFAULT_MAX_SESSION_MINUTES,
    EVENT_TYPE_RUN_CANCELLED,
    EVENT_TYPE_USER_MESSAGE,
    RUN_ACTIVE_STATUSES,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_QUEUED,
    SESSION_STATUS_ERROR,
    SESSION_STATUS_PROVISIONING,
    SESSION_STATUS_TERMINATED,
    SESSION_TERMINAL_STATUSES,
)
from litellm.proxy.agent_session_endpoints.ids import new_run_id, new_session_id
from litellm.proxy.agent_session_endpoints.ownership import (
    assert_caller_can_mutate,
    assert_caller_owns_agent,
    assert_caller_owns_session,
    caller_api_key_hash,
    owner_filter_for_caller,
)
from litellm.proxy.agent_session_endpoints.schemas import (
    FollowupCreate,
    FollowupResponse,
    SessionCreate,
    SessionResponse,
)
from litellm.proxy.agent_session_endpoints.serialization import (
    event_row_to_message,
    session_row_to_response,
)
from litellm.proxy.agent_session_endpoints.session_status import (
    refresh_session_status_from_runs,
)
from litellm.managed_agents.vms import (
    ProvisionContext,
    Repo,
    VMHandle,
    get_vm_provider,
)
from litellm.managed_agents.vms.warm_pool.attach import (
    AttachResult,
    WarmPoolEmptyError,
    attach_warm_vm,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

DEFAULT_VM_PROVIDER_NAME = "noop"


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _has_active_run(prisma_client, session_id: str) -> bool:
    """True iff ``session_id`` has any run in queued/running.

    Mirrors the ``_has_active_run`` helper in ``run_endpoints.py``. We
    duplicate it here (instead of importing) to keep ``session_endpoints``
    free of any dependency on ``run_endpoints``.
    """
    existing = await prisma_client.db.litellm_agentrun.find_first(
        where={
            "session_id": session_id,
            "status": {"in": list(RUN_ACTIVE_STATUSES)},
        }
    )
    return existing is not None


async def _get_prisma_client_or_503():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return prisma_client


def _resolve_repos(
    body_repos: Optional[List[Any]],
    agent_default_repos: Any,
) -> List[Dict[str, Any]]:
    """Caller-provided repos override agent defaults entirely (whole-list
    replace, not merge). If caller passes nothing, fall back to defaults.
    """
    if body_repos is not None:
        return [r.model_dump(exclude_none=True) for r in body_repos]
    if isinstance(agent_default_repos, list):
        return [r for r in agent_default_repos if isinstance(r, dict)]
    return []


def _resolve_env_vars(
    body_env_vars: Optional[Dict[str, str]],
    agent_default_env_vars: Any,
) -> Optional[Dict[str, str]]:
    """Merge: agent defaults first, caller overrides on top, key by key.

    This matches CRT-style env_var resolution and lets agent-level secrets
    (e.g. NPM_TOKEN) sit alongside per-session overrides without re-typing.
    """
    if body_env_vars is None and not isinstance(agent_default_env_vars, dict):
        return None
    merged: Dict[str, str] = {}
    if isinstance(agent_default_env_vars, dict):
        merged.update({str(k): str(v) for k, v in agent_default_env_vars.items()})
    if body_env_vars:
        merged.update({str(k): str(v) for k, v in body_env_vars.items()})
    return merged or None


def _proxy_base_url() -> str:
    """Best-effort proxy base URL for the daemon to call back into.

    Checks ``LITELLM_PROXY_BASE_URL`` env var first; falls back to localhost.
    Production deploys MUST set the env var.
    """
    import os

    return os.environ.get("LITELLM_PROXY_BASE_URL", "http://localhost:4000")


def _build_provision_repos(
    repos: List[Dict[str, Any]],
) -> List[Repo]:
    """Convert raw dicts (legacy schema) into typed ``Repo`` instances."""
    out: List[Repo] = []
    for r in repos or []:
        if isinstance(r, dict):
            out.append(
                Repo(
                    url=r.get("url", ""),
                    ref=r.get("ref"),
                    path=r.get("path"),
                )
            )
        else:
            out.append(r)
    return out


async def _provision_in_background(
    session_id: str,
    agent_id: str,
    team_id: Optional[str],
    repos: List[Dict[str, Any]],
    env_vars: Optional[Dict[str, str]],
    daemon_token: str,
    provider_name: str,
) -> None:
    """Background task: call provider.provision and update the session row.

    Failure paths flip status to ``error`` so the cleanup sweeper can chase
    the row. We never raise — this runs detached and a raise would crash
    the event loop's exception handler.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        verbose_proxy_logger.error(
            "session.provision failed: prisma_client is None (session_id=%s)",
            session_id,
        )
        return

    try:
        provider = get_vm_provider(provider_name)
        ctx = ProvisionContext(
            session_id=session_id,
            team_id=team_id or "",
            agent_id=agent_id,
            repos=_build_provision_repos(repos),
            env_vars=dict(env_vars or {}),
            secrets={},  # Epic G secrets injected by a separate hook (not yet wired)
            runtime_config={},
            aws_creds=None,  # populated by team_config.get_team_vm_config when ec2
            daemon_jwt=daemon_token,
            daemon_base_url=_proxy_base_url(),
            mode="session",
        )
        handle: VMHandle = await provider.provision(ctx)
        await prisma_client.db.litellm_agentsession.update(
            where={"id": session_id},
            data={
                "vm_id": handle.vm_id,
                "vm_provider": provider_name,
                "updated_at": _now(),
            },
        )
        verbose_proxy_logger.info(
            "session.provision ok session_id=%s vm_id=%s", session_id, handle.vm_id
        )
    except Exception as exc:
        verbose_proxy_logger.exception(
            "session.provision failed session_id=%s: %s", session_id, exc
        )
        try:
            await prisma_client.db.litellm_agentsession.update(
                where={"id": session_id},
                data={
                    "status": SESSION_STATUS_ERROR,
                    "updated_at": _now(),
                    "terminated_at": _now(),
                },
            )
        except Exception as inner:
            verbose_proxy_logger.exception(
                "session.provision: failed to mark session=%s as error: %s",
                session_id,
                inner,
            )


async def _try_warm_attach(
    *,
    prisma_client: Any,
    team_id: Optional[str],
    session_id: str,
    agent_row: Any,
    daemon_token: str,
    expires_at: datetime,
    repos: List[Dict[str, Any]],
    env_vars: Optional[Dict[str, str]],
) -> Optional[AttachResult]:
    """Best-effort warm-pool attach. Returns ``None`` to fall through to cold boot.

    Never raises: warm attach failures fall through silently so the session
    creation itself is not gated on warm-pool health. The cold-boot path
    serves as the always-available fallback.
    """
    if not team_id:
        return None
    try:
        return await attach_warm_vm(
            prisma_client=prisma_client,
            team_id=team_id,
            session_id=session_id,
            agent_id=agent_row.id,
            jwt=daemon_token,
            jwt_expires_at=expires_at,
            repos=repos,
            env_vars=env_vars,
            agent_row=agent_row,
        )
    except WarmPoolEmptyError:
        return None
    except Exception as exc:
        verbose_proxy_logger.warning(
            "session.create: warm attach failed; falling back to cold boot session=%s: %s",
            session_id,
            exc,
        )
        return None


async def _find_idempotent_session(user_api_key_hash: str, idempotency_key: str):
    """Return the existing session row for ``(user, idempotency_key)`` if any."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None
    return await prisma_client.db.litellm_agentsession.find_first(
        where={
            "user_api_key_hash": user_api_key_hash,
            "idempotency_key": idempotency_key,
        }
    )


@router.post(
    "/v2/sessions",
    response_class=ORJSONResponse,
    response_model=SessionResponse,
    tags=["sessions"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_session(
    body: SessionCreate,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    assert_caller_can_mutate(user_api_key_dict)
    prisma_client = await _get_prisma_client_or_503()

    # Idempotency: same (caller, key) returns same session — no daemon
    # token re-mint, no second provision call.
    user_hash = caller_api_key_hash(user_api_key_dict)
    if idempotency_key:
        existing = await _find_idempotent_session(user_hash, idempotency_key)
        if existing is not None:
            return session_row_to_response(existing, daemon_token=None)

    # Validate parent agent + ownership.
    agent_row = await prisma_client.db.litellm_agent.find_unique(
        where={"id": body.agent_id}
    )
    assert_caller_owns_agent(user_api_key_dict, agent_row)

    # Resolve repos/env_vars (overlay caller over agent defaults).
    resolved_repos = _resolve_repos(body.repos, agent_row.default_repos)
    resolved_env_vars = _resolve_env_vars(body.env_vars, agent_row.default_env_vars)

    # Compute expiry — default 4h, capped 24h via Pydantic validator.
    max_minutes = body.max_session_minutes or DEFAULT_MAX_SESSION_MINUTES
    expires_at = _now() + timedelta(minutes=max_minutes)

    session_id = new_session_id()
    daemon_token = mint_daemon_token(
        session_id=session_id,
        agent_id=body.agent_id,
        expires_at_epoch=int(expires_at.timestamp()),
    )
    # Prisma create payload notes:
    # - JSON columns (``repos``, ``env_vars``) must be wrapped in
    #   ``prisma.Json(...)``; bare dict/list values trigger
    #   ``MissingRequiredValueError``.
    # - Relation fields (``agent``) must use ``{"connect": {"id": ...}}``;
    #   bare ``agent_id`` strings are rejected by the generated client
    #   even though the underlying SQL column is ``agent_id`` (a TEXT FK).
    payload: Dict[str, Any] = {
        "id": session_id,
        "agent": {"connect": {"id": body.agent_id}},
        "user_api_key_hash": user_hash,
        "team_id": user_api_key_dict.team_id,
        "vm_provider": DEFAULT_VM_PROVIDER_NAME,
        "repos": prisma.Json(resolved_repos or []),
        "status": SESSION_STATUS_PROVISIONING,
        "daemon_token_hash": hash_daemon_token(daemon_token),
        "expires_at": expires_at,
        "idempotency_key": idempotency_key,
        "updated_at": _now(),
    }
    if resolved_env_vars is not None:
        payload["env_vars"] = prisma.Json(resolved_env_vars)
    row = await prisma_client.db.litellm_agentsession.create(data=payload)

    # Try warm-pool attach first. Per LIT-2890, this is the primary path
    # when ``LiteLLM_AgentVMConfig.warm_pool_enabled=true`` for the team —
    # P95 target <3s vs ~60s cold boot. If no warm VM is available we
    # fall through to the existing cold-boot path.
    attach_result = await _try_warm_attach(
        prisma_client=prisma_client,
        team_id=user_api_key_dict.team_id,
        session_id=session_id,
        agent_row=agent_row,
        daemon_token=daemon_token,
        expires_at=expires_at,
        repos=resolved_repos,
        env_vars=resolved_env_vars,
    )
    if attach_result is not None:
        row = await prisma_client.db.litellm_agentsession.update(
            where={"id": session_id},
            data={
                "vm_id": attach_result.vm_id,
                "vm_provider": "ec2",
                "status": SESSION_STATUS_READY,
                "updated_at": _now(),
            },
        )
        verbose_proxy_logger.info(
            "session.create warm_attach=ok id=%s vm_id=%s expires_at=%s",
            session_id,
            attach_result.vm_id,
            expires_at.isoformat(),
        )
        return session_row_to_response(row, daemon_token=daemon_token)

    # Cold boot path: fire-and-forget VM provisioning. The client polls /
    # subscribes for status flips.
    asyncio.create_task(
        _provision_in_background(
            session_id=session_id,
            agent_id=body.agent_id,
            team_id=user_api_key_dict.team_id,
            repos=resolved_repos,
            env_vars=resolved_env_vars,
            daemon_token=daemon_token,
            provider_name=DEFAULT_VM_PROVIDER_NAME,
        )
    )

    verbose_proxy_logger.info(
        "session.create warm_attach=miss id=%s agent_id=%s expires_at=%s",
        session_id,
        body.agent_id,
        expires_at.isoformat(),
    )
    return session_row_to_response(row, daemon_token=daemon_token)


@router.get(
    "/v2/sessions/{session_id}",
    response_class=ORJSONResponse,
    response_model=SessionResponse,
    tags=["sessions"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_session(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    prisma_client = await _get_prisma_client_or_503()
    row = await prisma_client.db.litellm_agentsession.find_unique(
        where={"id": session_id}
    )
    assert_caller_owns_session(user_api_key_dict, row)
    return session_row_to_response(row, daemon_token=None)


@router.get(
    "/v2/sessions",
    response_class=ORJSONResponse,
    tags=["sessions"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_sessions(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    agent_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    prisma_client = await _get_prisma_client_or_503()
    where: Dict[str, Any] = {}
    owner = owner_filter_for_caller(user_api_key_dict)
    if owner:
        where.update(owner)
    if agent_id:
        where["agent_id"] = agent_id
    rows = await prisma_client.db.litellm_agentsession.find_many(
        where=where or None,
        order={"created_at": "desc"},
        take=limit,
        skip=offset,
    )
    return {
        "data": [
            session_row_to_response(r, daemon_token=None).model_dump() for r in rows
        ]
    }


@router.delete(
    "/v2/sessions/{session_id}",
    response_class=ORJSONResponse,
    tags=["sessions"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_session(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    assert_caller_can_mutate(user_api_key_dict)
    prisma_client = await _get_prisma_client_or_503()
    row = await prisma_client.db.litellm_agentsession.find_unique(
        where={"id": session_id}
    )
    assert_caller_owns_session(user_api_key_dict, row)

    if row.status not in SESSION_TERMINAL_STATUSES:
        await _terminate_session_internal(session_id, reason="user_delete")

    return {"id": session_id, "deleted": True}


async def _next_event_seq(prisma_client, run_id: str) -> int:
    """Return ``MAX(seq) + 1`` for a run, or 1 if no events yet.

    The endpoint that calls this still relies on the DB unique constraint
    ``(run_id, seq)`` for correctness — this lookup is just a best-effort
    starting point so retries collide and increment quickly.
    """
    last = await prisma_client.db.litellm_agentrunevent.find_first(
        where={"run_id": run_id},
        order={"seq": "desc"},
    )
    if last is None:
        return 1
    return last.seq + 1


async def _terminate_session_internal(session_id: str, reason: str) -> None:
    """Internal helper: cancel runs, mark session terminated, fire provider.terminate.

    Used by:
      - DELETE /v2/sessions/{id}
      - DELETE /v2/agents/{id}    (cascade)
      - cleanup sweeper
      - daemon-dead detector
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return

    session = await prisma_client.db.litellm_agentsession.find_unique(
        where={"id": session_id}
    )
    if session is None:
        return
    if session.status in SESSION_TERMINAL_STATUSES:
        return

    # 1. Cancel any non-terminal runs and emit run_cancelled events.
    active_runs = await prisma_client.db.litellm_agentrun.find_many(
        where={
            "session_id": session_id,
            "status": {"in": list(RUN_ACTIVE_STATUSES)},
        }
    )
    now = _now()
    for run in active_runs:
        await prisma_client.db.litellm_agentrun.update(
            where={"id": run.id},
            data={
                "status": RUN_STATUS_CANCELLED,
                "terminated_at": now,
                "updated_at": now,
            },
        )
        next_seq = await _next_event_seq(prisma_client, run.id)
        try:
            await prisma_client.db.litellm_agentrunevent.create(
                data={
                    "run_id": run.id,
                    "seq": next_seq,
                    "event_type": EVENT_TYPE_RUN_CANCELLED,
                    "payload": {"reason": reason},
                }
            )
        except Exception as exc:
            # Swallow seq-race; the daemon may have just emitted run_finished.
            verbose_proxy_logger.warning(
                "session.terminate: skipped run_cancelled emit run=%s seq=%s: %s",
                run.id,
                next_seq,
                exc,
            )

    # 2. Mark session terminated.
    await prisma_client.db.litellm_agentsession.update(
        where={"id": session_id},
        data={
            "status": SESSION_STATUS_TERMINATED,
            "terminated_at": now,
            "updated_at": now,
        },
    )

    # 3. Fire provider.terminate (best-effort; never blocks API caller).
    try:
        provider = get_vm_provider(session.vm_provider or DEFAULT_VM_PROVIDER_NAME)
        if session.vm_id:
            handle = VMHandle(
                vm_id=session.vm_id,
                provider=session.vm_provider or DEFAULT_VM_PROVIDER_NAME,
                metadata={"session_id": session_id},
            )
            # NoopProvider.terminate accepts both signatures; EC2Provider
            # requires aws_creds — wired in Epic C when team_config is fully
            # plumbed through.
            await provider.terminate(handle)
    except Exception as exc:
        verbose_proxy_logger.exception(
            "session.terminate: provider.terminate failed session=%s: %s",
            session_id,
            exc,
        )


@router.post(
    "/v2/sessions/{session_id}/followup",
    response_class=ORJSONResponse,
    response_model=FollowupResponse,
    tags=["sessions"],
    dependencies=[Depends(user_api_key_auth)],
)
async def followup(
    session_id: str,
    body: FollowupCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Smart followup: if the latest run is active, append a user_message
    event to it; if terminal or no runs, start a new run.

    Matches Cursor's ``/followup`` semantics. Daemon picks up the
    ``user_message`` event via the events stream and weaves it into the
    in-flight LLM turn.
    """
    assert_caller_can_mutate(user_api_key_dict)
    prisma_client = await _get_prisma_client_or_503()
    session = await prisma_client.db.litellm_agentsession.find_unique(
        where={"id": session_id}
    )
    assert_caller_owns_session(user_api_key_dict, session)

    if session.status in SESSION_TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Session is {session.status}; cannot followup",
        )

    latest_run = await prisma_client.db.litellm_agentrun.find_first(
        where={"session_id": session_id},
        order={"created_at": "desc"},
    )

    if latest_run is not None and latest_run.status in RUN_ACTIVE_STATUSES:
        # Inject as a user_message event on the active run.
        next_seq = await _next_event_seq(prisma_client, latest_run.id)
        await prisma_client.db.litellm_agentrunevent.create(
            data={
                "run_id": latest_run.id,
                "seq": next_seq,
                "event_type": EVENT_TYPE_USER_MESSAGE,
                "payload": body.prompt,
            }
        )
        return FollowupResponse(run_id=latest_run.id, action="queued")

    # Concurrency guard: matches POST /runs. Without this, two concurrent
    # /followup requests on an idle session both pass the
    # ``latest_run.status in RUN_ACTIVE_STATUSES`` check above and both
    # fall through to ``create``, breaking the "one active run at a time"
    # invariant. Re-check for any active run RIGHT before insert and
    # return 409 ``run_busy`` instead, just like POST /runs does.
    if await _has_active_run(prisma_client, session_id):
        raise HTTPException(
            status_code=409,
            detail="run_busy: another run is queued/running for this session",
        )

    # Else start a fresh run.
    try:
        new_run = await prisma_client.db.litellm_agentrun.create(
            data={
                "id": new_run_id(),
                "session_id": session_id,
                "status": RUN_STATUS_QUEUED,
                "prompt": body.prompt,
                "updated_at": _now(),
            }
        )
    except Exception as exc:
        # Last line of defense: another /followup or POST /runs may have
        # raced past the busy check and won the insert. Surface the same
        # 409 so the client can retry deterministically.
        active_other = await prisma_client.db.litellm_agentrun.find_first(
            where={
                "session_id": session_id,
                "status": {"in": list(RUN_ACTIVE_STATUSES)},
            }
        )
        if active_other is not None:
            raise HTTPException(
                status_code=409,
                detail="run_busy: another run is queued/running for this session",
            ) from exc
        raise
    # Run was just queued — flip session ``ready`` -> ``busy`` so SDK
    # consumers see the right status without waiting for the daemon to
    # claim the run.
    await refresh_session_status_from_runs(prisma_client, session_id)
    return FollowupResponse(run_id=new_run.id, action="new_run")


@router.get(
    "/v2/sessions/{session_id}/conversation",
    response_class=ORJSONResponse,
    tags=["sessions"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_conversation(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Stateless snapshot: every event across every run for this session,
    in the order the daemon emitted them. Used by SDK consumers that don't
    need the SSE stream — read-once-and-render."""
    prisma_client = await _get_prisma_client_or_503()
    session = await prisma_client.db.litellm_agentsession.find_unique(
        where={"id": session_id}
    )
    assert_caller_owns_session(user_api_key_dict, session)

    runs = await prisma_client.db.litellm_agentrun.find_many(
        where={"session_id": session_id},
        order={"created_at": "asc"},
    )
    if not runs:
        return {"session_id": session_id, "messages": []}

    run_ids = [r.id for r in runs]
    events = await prisma_client.db.litellm_agentrunevent.find_many(
        where={"run_id": {"in": run_ids}},
        order=[{"created_at": "asc"}, {"seq": "asc"}],
    )
    return {
        "session_id": session_id,
        "messages": [event_row_to_message(e).model_dump() for e in events],
    }
