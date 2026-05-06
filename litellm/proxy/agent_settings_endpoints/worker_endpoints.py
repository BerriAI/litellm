"""
`/v2/agent-workers` endpoints (LIT-2891 / Screen 3).

Self-hosted worker registration flow:

1. Operator clicks "Add Machine" → UI calls `POST /v2/agent-workers/pair-token`.
   We mint a 32-byte urlsafe token, persist only its sha256, return the raw
   token + an install one-liner. TTL = 15 min, single use.
2. Operator runs the install command on the worker box. The worker calls
   `POST /v2/agent-workers/register` with the raw token + its hostname. We
   re-hash, atomically mark the pairing-token row as consumed, mint a long-
   lived worker JWT, persist its sha256, and return the raw JWT to the worker.
   Both raw values are returned exactly once and never persisted.
3. The dashboard lists workers via `GET /v2/agent-workers` and revokes them
   via `DELETE /v2/agent-workers/{id}`.

The actual long-poll / hydrate transport is owned by Epic B2 — this module
only owns the auth handshake and CRUD surface.
"""

import os
import secrets as _stdlib_secrets
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.agent_settings_endpoints.pair_tokens import (
    build_install_command,
    hash_pair_token,
    hash_worker_jwt,
    is_expired,
    issue_pair_token,
)
from litellm.proxy.agent_settings_endpoints.types import (
    AgentWorkerListResponse,
    AgentWorkerRegisterRequest,
    AgentWorkerRegisterResponse,
    AgentWorkerResponse,
    PairTokenResponse,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


def _resolve_team_id(user_api_key_dict: UserAPIKeyAuth) -> str:
    """Pick the team to scope this request to. Raise 400 if missing."""
    team_id = user_api_key_dict.team_id or (user_api_key_dict.metadata or {}).get(
        "team_id"
    )
    if not team_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cloud Agent workers are scoped to a team. Pick a team from the "
                "header switcher and try again."
            ),
        )
    return team_id


def _row_to_worker_response(row: Dict[str, Any]) -> AgentWorkerResponse:
    """Map a Prisma worker row to the public response shape."""
    last_seen = row.get("last_seen_at")
    return AgentWorkerResponse(
        id=row["id"],
        hostname=row.get("hostname") or "",
        status=row.get("status") or "offline",
        last_seen_at=str(last_seen) if last_seen is not None else None,
        cpu_pct=row.get("cpu_pct"),
        mem_gb=row.get("mem_gb"),
        active_sessions=int(row.get("active_sessions") or 0),
    )


def _resolve_proxy_url(request: Request) -> str:
    """Resolve the public proxy URL embedded in the install one-liner.

    SECURITY: this URL ends up in `--proxy <url>` of the curl-pipe-sh
    install command. If an attacker can influence it, they can redirect
    the worker (and the freshly-issued pair token) to a host they control.
    Resolution order:

    1. `LITELLM_CLOUD_AGENT_PROXY_BASE_URL` env var — operator-configured,
       fully trusted. This is the recommended path for production.
    2. `X-Forwarded-Host` / `X-Forwarded-Proto` — only honored when the
       operator opts in via `LITELLM_TRUST_PROXY_HEADERS=1`. Required for
       deployments behind a reverse proxy that terminates TLS.
    3. The request's direct `Host` header + URL scheme. Safe by default
       because it reflects the actual TCP-layer destination of the
       request, not an attacker-controlled hop hint.

    `localhost:4000` is the last-resort fallback for tests/local dev.
    """
    configured = os.getenv("LITELLM_CLOUD_AGENT_PROXY_BASE_URL")
    if configured:
        return configured.rstrip("/")

    if os.getenv("LITELLM_TRUST_PROXY_HEADERS", "0") == "1":
        forwarded_proto = request.headers.get("x-forwarded-proto")
        forwarded_host = request.headers.get("x-forwarded-host")
        if forwarded_host:
            scheme = forwarded_proto or request.url.scheme or "https"
            return f"{scheme}://{forwarded_host}"

    host = request.headers.get("host") or "localhost:4000"
    scheme = request.url.scheme or "https"
    return f"{scheme}://{host}"


@router.get(
    "/v2/agent-workers",
    dependencies=[Depends(user_api_key_auth)],
    response_model=AgentWorkerListResponse,
    tags=["cloud agents"],
)
async def list_agent_workers(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> AgentWorkerListResponse:
    """List the team's self-hosted workers."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    rows = await prisma_client.db.litellm_agentworker.find_many(
        where={"team_id": team_id},
        order={"created_at": "desc"},
    )
    workers: List[AgentWorkerResponse] = [
        _row_to_worker_response(dict(r) if not isinstance(r, dict) else r) for r in rows
    ]
    return AgentWorkerListResponse(workers=workers)


@router.post(
    "/v2/agent-workers/pair-token",
    dependencies=[Depends(user_api_key_auth)],
    response_model=PairTokenResponse,
    tags=["cloud agents"],
)
async def create_pair_token(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PairTokenResponse:
    """Mint a single-use 15-minute pairing token. Raw token returned ONCE."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    issued = issue_pair_token()

    try:
        await prisma_client.db.litellm_agentworkerpairingtoken.create(
            data={
                "token_hash": issued.token_hash,
                "team_id": team_id,
                "created_by": user_api_key_dict.user_id or "unknown",
                "expires_at": issued.expires_at,
            }
        )
    except Exception as exc:
        verbose_proxy_logger.exception(
            "Failed to persist pair token for team=%s: %s", team_id, exc
        )
        raise HTTPException(status_code=500, detail="Failed to mint pair token.")

    install_command = build_install_command(
        proxy_url=_resolve_proxy_url(request),
        raw_token=issued.raw_token,
    )
    return PairTokenResponse(
        token=issued.raw_token,
        expires_at=issued.expires_at.isoformat(),
        install_command=install_command,
    )


@router.post(
    "/v2/agent-workers/register",
    response_model=AgentWorkerRegisterResponse,
    tags=["cloud agents"],
)
async def register_agent_worker(
    request: Request,
    body: AgentWorkerRegisterRequest,
) -> AgentWorkerRegisterResponse:
    """Worker exchanges its pair token for a long-lived JWT.

    NOTE: this endpoint is NOT behind `user_api_key_auth` — the worker has
    no API key yet, the pair token *is* its proof of authorization. We
    enforce single-use atomically by checking `used_at IS NULL` on update.

    Defense-in-depth: the pair token is 256 bits of entropy with a 15-min
    TTL, so brute-force is not a credible threat. We log each failed
    attempt with the source IP so operators running this behind a WAF or
    fail2ban can detect and rate-limit abusive callers at the network
    layer (the proxy itself doesn't ship a built-in per-IP limiter).
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    client_ip = request.client.host if request.client is not None else "unknown"

    token_hash = hash_pair_token(body.pair_token)
    pair_row = await prisma_client.db.litellm_agentworkerpairingtoken.find_unique(
        where={"token_hash": token_hash}
    )
    if pair_row is None:
        verbose_proxy_logger.warning(
            "agent-workers/register: invalid pair token from ip=%s hostname=%s",
            client_ip,
            body.hostname,
        )
        raise HTTPException(status_code=401, detail="Invalid pairing token.")

    pair = dict(pair_row) if not isinstance(pair_row, dict) else pair_row
    if pair.get("used_at") is not None:
        verbose_proxy_logger.warning(
            "agent-workers/register: replay of consumed pair token from ip=%s hostname=%s team=%s",
            client_ip,
            body.hostname,
            pair.get("team_id"),
        )
        raise HTTPException(
            status_code=401, detail="Pairing token has already been used."
        )
    if is_expired(pair.get("expires_at")):
        verbose_proxy_logger.warning(
            "agent-workers/register: expired pair token from ip=%s hostname=%s team=%s",
            client_ip,
            body.hostname,
            pair.get("team_id"),
        )
        raise HTTPException(status_code=401, detail="Pairing token has expired.")

    team_id = pair["team_id"]

    # Atomically consume the pair token. `update_many` with the `used_at: None`
    # filter is the closest Prisma gives us to a CAS — if a second register
    # raced us, the second one matches 0 rows and we 401.
    consumed = await prisma_client.db.litellm_agentworkerpairingtoken.update_many(
        where={"token_hash": token_hash, "used_at": None},
        data={"used_at": datetime.now(timezone.utc)},
    )
    if not consumed:
        raise HTTPException(
            status_code=401, detail="Pairing token has already been used."
        )

    # Mint the worker JWT. This is a short opaque urlsafe string; the daemon
    # presents it on every long-poll. We persist only its sha256.
    raw_jwt = _stdlib_secrets.token_urlsafe(48)
    jwt_hash = hash_worker_jwt(raw_jwt)

    try:
        worker = await prisma_client.db.litellm_agentworker.create(
            data={
                "team_id": team_id,
                "hostname": body.hostname,
                "status": "online",
                "last_seen_at": datetime.now(timezone.utc),
                "active_sessions": 0,
                "worker_jwt_hash": jwt_hash,
            }
        )
    except Exception as exc:
        verbose_proxy_logger.exception(
            "Failed to create agent worker hostname=%s team=%s: %s",
            body.hostname,
            team_id,
            exc,
        )
        raise HTTPException(status_code=500, detail="Failed to register worker.")

    worker_dict = dict(worker) if not isinstance(worker, dict) else worker
    return AgentWorkerRegisterResponse(
        worker_id=worker_dict["id"],
        worker_jwt=raw_jwt,
    )


@router.delete(
    "/v2/agent-workers/{worker_id}",
    dependencies=[Depends(user_api_key_auth)],
    tags=["cloud agents"],
)
async def delete_agent_worker(
    request: Request,
    worker_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """Revoke a worker. Idempotent — 404 if not found, no-op if already gone."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    team_id = _resolve_team_id(user_api_key_dict)
    existing = await prisma_client.db.litellm_agentworker.find_unique(
        where={"id": worker_id}
    )
    if existing is None:
        raise HTTPException(status_code=404, detail="Worker not found.")

    existing_dict = dict(existing) if not isinstance(existing, dict) else existing
    if existing_dict.get("team_id") != team_id:
        # Don't leak existence cross-team; same status code as not-found.
        raise HTTPException(status_code=404, detail="Worker not found.")

    await prisma_client.db.litellm_agentworker.delete(where={"id": worker_id})
    return {"deleted": True, "id": worker_id}


# Re-exported for tests + B2 hydrate path: given a raw worker JWT, look up the
# worker row. Centralized here so the auth scheme lives in exactly one place.
async def find_worker_by_jwt(
    prisma_client: Any, raw_jwt: str
) -> Optional[Dict[str, Any]]:
    """Look up a worker row by its (raw) JWT. Returns None if not found.

    `worker_jwt_hash` is indexed (see migration), so this is an index
    lookup — fine for B2's per-heartbeat call pattern.
    """
    jwt_hash = hash_worker_jwt(raw_jwt)
    worker = await prisma_client.db.litellm_agentworker.find_first(
        where={"worker_jwt_hash": jwt_hash}
    )
    if worker is None:
        return None
    return dict(worker) if not isinstance(worker, dict) else worker
