"""
JWT minting + validation for the daemon-side internal endpoints.

Daemon tokens are minted at session creation and scoped to a single session.
The token's SHA-256 hash is stored on the session row so revoking the
session also revokes the token without rotating signing keys.
"""

import hashlib
import os
import time
from typing import Any, Dict, Optional

import jwt
from fastapi import Header, HTTPException

from litellm.proxy.agent_session_endpoints.constants import (
    AGENT_JWT_ALGORITHM,
    AGENT_JWT_SECRET_ENV,
    AGENT_RUNTIME_SCOPE,
    SESSION_TERMINAL_STATUSES,
)


class AgentDaemonTokenError(Exception):
    """Raised when a daemon token is invalid (used internally by validators)."""


class AgentJWTSecretNotConfiguredError(RuntimeError):
    """Raised when ``LITELLM_AGENT_JWT_SECRET`` is unset.

    The daemon JWT secret MUST be a separate credential from the proxy
    master key. Falling back to ``LITELLM_MASTER_KEY`` would conflate two
    distinct auth surfaces — a captured daemon JWT could then be used to
    mint regular API keys with master-key authority.
    """


def is_agent_jwt_secret_configured() -> bool:
    """Return True iff a non-empty ``LITELLM_AGENT_JWT_SECRET`` is present.

    Used at startup by ``proxy_server.py`` to decide whether to mount the
    agent_session_endpoints routers. Mounting the routers when this is
    False would silently expose an unsigned auth surface — refuse instead.
    """
    return bool(os.environ.get(AGENT_JWT_SECRET_ENV))


def _get_signing_secret() -> str:
    """Resolve the JWT signing secret.

    The daemon JWT secret is a SEPARATE credential from the proxy master
    key. There is no fallback — if ``LITELLM_AGENT_JWT_SECRET`` is unset,
    we refuse to mint or validate any token. Mounting the agent_session
    routers without this env var is itself a startup error (see
    ``proxy_server.py``).
    """
    secret = os.environ.get(AGENT_JWT_SECRET_ENV)
    if not secret:
        raise AgentJWTSecretNotConfiguredError(
            f"{AGENT_JWT_SECRET_ENV} is not set; cannot mint or validate "
            "daemon JWTs. This env var must be a dedicated random secret, "
            "distinct from LITELLM_MASTER_KEY."
        )
    return secret


def hash_daemon_token(token: str) -> str:
    """Return ``sha256(token)`` as hex — what we store on the session row."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def mint_daemon_token(
    session_id: str,
    agent_id: str,
    expires_at_epoch: int,
) -> str:
    """Mint a session-scoped JWT for the daemon.

    Claims:
      sub:      session id (the only session this token can act on)
      agent_id: parent agent id (informational)
      iat:      issued-at
      exp:      epoch seconds
      scope:    "agent_runtime_internal" — gated by a dedicated dependency
                so this token CANNOT be used as a normal user virtual key.
    """
    now = int(time.time())
    payload: Dict[str, Any] = {
        "sub": session_id,
        "agent_id": agent_id,
        "iat": now,
        "exp": expires_at_epoch,
        "scope": AGENT_RUNTIME_SCOPE,
    }
    return jwt.encode(payload, _get_signing_secret(), algorithm=AGENT_JWT_ALGORITHM)


def decode_daemon_token(token: str) -> Dict[str, Any]:
    """Decode + verify signature + verify ``exp`` and ``scope``.

    Raises HTTPException(401) for any failure so callers can re-raise
    directly.
    """
    try:
        payload = jwt.decode(
            token,
            _get_signing_secret(),
            algorithms=[AGENT_JWT_ALGORITHM],
        )
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=401, detail="Daemon token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=401, detail="Invalid daemon token") from exc

    if payload.get("scope") != AGENT_RUNTIME_SCOPE:
        raise HTTPException(
            status_code=401,
            detail="Daemon token has wrong scope",
        )
    return payload


def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401, detail="Authorization header must be 'Bearer <token>'"
        )
    return parts[1].strip()


async def daemon_token_auth(
    session_id: str,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    """FastAPI dependency: validate the daemon JWT for a path-scoped session.

    Validation steps (in order):
      1. Decode + verify signature, ``exp``, ``scope``.
      2. Confirm ``sub == session_id`` (cross-session abuse).
      3. Look up session, confirm it's not terminated.
      4. Confirm ``sha256(token) == session.daemon_token_hash`` (revoked tokens).

    Returns the validated payload dict (with ``_session_row`` injected) so
    downstream code can avoid re-fetching the session.
    """
    token = _extract_bearer(authorization)
    payload = decode_daemon_token(token)

    if payload.get("sub") != session_id:
        raise HTTPException(status_code=403, detail="Token not valid for this session")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=503, detail="Database unavailable for daemon token check"
        )

    session = await prisma_client.db.litellm_agentsession.find_unique(
        where={"id": session_id}
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if session.status in SESSION_TERMINAL_STATUSES:
        # Stored hash check would also fail after termination, but we want
        # a 410 Gone so the daemon's systemd unit knows to shutdown.
        raise HTTPException(status_code=410, detail="Session terminated")

    expected_hash = session.daemon_token_hash
    if not expected_hash or hash_daemon_token(token) != expected_hash:
        raise HTTPException(status_code=401, detail="Daemon token revoked")

    return {**payload, "_session_row": session}
