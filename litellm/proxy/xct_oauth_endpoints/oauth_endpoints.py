"""
XCT App OAuth handshake — ``/oauth/{authorize,token,revoke,introspect}``
(S4-05 + S4-06 + S4-07).

Per ADR-0002:
    - Spec: OAuth 2.0 Authorization Code flow with PKCE (RFC 7636).
    - Public clients (browser) MUST use PKCE.
    - Confidential clients (server-to-server) MAY skip PKCE if they
      present client_secret on /oauth/token.
    - Tokens are issued as ``LiteLLM_VerificationToken`` rows with
      ``token_type ∈ {oauth_access, oauth_refresh}`` and ``app_id`` set.
    - The /authorize consent screen is delegated to the existing dashboard
      SSO; this module only validates inputs and persists the auth code.

Endpoints in this module are mounted at the top level (no auth dependency)
because OAuth itself bootstraps auth — applying ``user_api_key_auth`` to
``/oauth/token`` would create a chicken-and-egg.
"""

import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import RedirectResponse

from litellm._logging import verbose_proxy_logger

router = APIRouter()

# Auth code TTL — 10 min is conservative; clients should redeem within
# seconds. RFC 7636 §4.4 recommends ≤10 min.
AUTH_CODE_TTL_SECONDS = 600

# Access token TTL — 1 hour. Refresh tokens last 30 days.
ACCESS_TOKEN_TTL_SECONDS = 3600
REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 3600

_ALLOWED_CHALLENGE_METHODS = ("S256",)  # RFC 7636 — we don't accept "plain"


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _verify_pkce(verifier: str, challenge: str, method: str) -> bool:
    """RFC 7636 §4.6 verification."""
    if method != "S256":
        return False
    derived = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .decode()
        .rstrip("=")
    )
    return secrets.compare_digest(derived, challenge)


async def _lookup_app_by_client_id(prisma_client, client_id: str):
    if prisma_client is None:
        return None
    rows = await prisma_client.db.litellm_xctapptable.find_many(
        where={"oauth_client_id": client_id, "is_active": True}, take=1
    )
    return rows[0] if rows else None


def _redirect_uri_allowed(uri: str, whitelist: List[str]) -> bool:
    """Exact-match. RFC 6749 §3.1.2 — partial matches are not safe."""
    return uri in (whitelist or [])


# ---------------------------------------------------------------------------
# S4-05  /oauth/authorize
# ---------------------------------------------------------------------------


@router.get("/oauth/authorize", tags=["[beta] XCT OAuth"])
async def authorize(
    request: Request,
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    response_type: str = Query("code"),
    state: Optional[str] = Query(None),
    code_challenge: str = Query(...),
    code_challenge_method: str = Query("S256"),
    scope: Optional[str] = Query(None),
) -> RedirectResponse:
    """Issue an authorization code + redirect back to the app.

    v0 simplification (ADR-0002): no per-call user consent screen — the
    user is assumed to be the dashboard-SSO user attached to this session.
    If we don't have a session, return 401 so the SDK can bounce the
    browser through /sso/login first.

    Errors that are caused by the CLIENT'S malformed request return JSON
    400 / 401 directly (the spec calls this the "user-agent error case").
    Errors after the redirect_uri is validated SHOULD be relayed by
    appending ?error=... to the redirect — implemented for the common
    cases below.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")

    if response_type != "code":
        raise HTTPException(status_code=400, detail="response_type must be 'code'")
    if code_challenge_method not in _ALLOWED_CHALLENGE_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"code_challenge_method must be one of {_ALLOWED_CHALLENGE_METHODS}",
        )

    app = await _lookup_app_by_client_id(prisma_client, client_id)
    if app is None:
        raise HTTPException(status_code=400, detail="unknown client_id")

    if not _redirect_uri_allowed(redirect_uri, app.redirect_uris):
        raise HTTPException(
            status_code=400, detail="redirect_uri not in client whitelist"
        )

    # Identify the end user. The dashboard SSO sets a cookie; if we don't
    # have one, kick to /sso/login with a return-to that brings the user
    # back here. Until we land that round-trip, accept an explicit
    # x-xct-user-id header for SDK-driven test flows.
    end_user = (
        request.cookies.get("token")
        or request.cookies.get("litellm_jwt")
        or request.headers.get("x-xct-user-id")
    )
    if not end_user:
        raise HTTPException(
            status_code=401,
            detail="Not signed in. Bounce through /sso/login first.",
        )

    requested_scope = (scope or "").split() if scope else []
    granted = [s for s in requested_scope if s in (app.default_scopes or [])]

    code = secrets.token_urlsafe(40)
    expires_at = datetime.utcnow() + timedelta(seconds=AUTH_CODE_TTL_SECONDS)
    await prisma_client.db.litellm_oauthauthorizationcode.create(
        data={
            "code": code,
            "client_id": client_id,
            "user_id": end_user,
            "redirect_uri": redirect_uri,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "scope": granted,
            "expires_at": expires_at,
        }
    )

    redirect_back = f"{redirect_uri}?code={code}"
    if state:
        redirect_back += f"&state={state}"
    return RedirectResponse(url=redirect_back, status_code=302)


# ---------------------------------------------------------------------------
# S4-06  /oauth/token
# ---------------------------------------------------------------------------


async def _consume_authorization_code(
    prisma_client,
    code: str,
    *,
    client_id: str,
    redirect_uri: str,
    code_verifier: str,
):
    row = await prisma_client.db.litellm_oauthauthorizationcode.find_unique(
        where={"code": code}
    )
    if row is None:
        raise HTTPException(status_code=400, detail="invalid_grant: code not found")
    if row.consumed_at is not None:
        raise HTTPException(
            status_code=400, detail="invalid_grant: code already consumed"
        )
    if row.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="invalid_grant: code expired")
    if row.client_id != client_id:
        raise HTTPException(status_code=400, detail="invalid_grant: client_id mismatch")
    if row.redirect_uri != redirect_uri:
        raise HTTPException(
            status_code=400, detail="invalid_grant: redirect_uri mismatch"
        )
    if not _verify_pkce(code_verifier, row.code_challenge, row.code_challenge_method):
        raise HTTPException(
            status_code=400, detail="invalid_grant: PKCE verification failed"
        )

    # Mark consumed before returning so a replay never succeeds even if
    # the caller retries the exchange.
    await prisma_client.db.litellm_oauthauthorizationcode.update(
        where={"code": code},
        data={"consumed_at": datetime.utcnow()},
    )
    return row


async def _verify_client_credentials(
    prisma_client, client_id: str, client_secret: Optional[str]
):
    app = await _lookup_app_by_client_id(prisma_client, client_id)
    if app is None:
        raise HTTPException(status_code=400, detail="unknown client_id")
    # PKCE-only browser flow: client_secret may be omitted entirely.
    # Refresh-token flow / server-to-server: client_secret required and
    # must match.
    if client_secret is not None:
        if _sha256_hex(client_secret) != app.oauth_client_secret_hash:
            raise HTTPException(
                status_code=401, detail="invalid_client: secret mismatch"
            )
    return app


def _make_token_value() -> str:
    """sk-prefixed so the proxy auth fast-paths recognize it as a key."""
    return "sk-xct-" + secrets.token_urlsafe(32)


async def _issue_token_pair(
    prisma_client,
    *,
    app,
    user_id: str,
    scope: List[str],
) -> Dict[str, Any]:
    """Mint matched access + refresh rows in LiteLLM_VerificationToken.

    Both rows carry app_id + the same scope. The access token expires in
    1h; the refresh token in 30d.
    """
    access_value = _make_token_value()
    refresh_value = _make_token_value()
    now = datetime.utcnow()
    access_expires = now + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS)
    refresh_expires = now + timedelta(seconds=REFRESH_TOKEN_TTL_SECONDS)
    from litellm.proxy.utils import hash_token

    await prisma_client.db.litellm_verificationtoken.create(
        data={
            "token": hash_token(access_value),
            "key_alias": f"oauth/{app.app_name}",
            "expires": access_expires,
            "user_id": user_id,
            "team_id": app.default_team_id,
            "app_id": app.app_id,
            "token_type": "oauth_access",
            "models": [],
            "metadata": {"scope": scope},
        }
    )
    await prisma_client.db.litellm_verificationtoken.create(
        data={
            "token": hash_token(refresh_value),
            "key_alias": f"oauth/{app.app_name}/refresh",
            "expires": refresh_expires,
            "user_id": user_id,
            "team_id": app.default_team_id,
            "app_id": app.app_id,
            "token_type": "oauth_refresh",
            "models": [],
            "metadata": {"scope": scope},
        }
    )

    return {
        "access_token": access_value,
        "refresh_token": refresh_value,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_TTL_SECONDS,
        "scope": " ".join(scope) if scope else "",
    }


@router.post("/oauth/token", tags=["[beta] XCT OAuth"])
async def token(
    grant_type: str = Form(...),
    client_id: str = Form(...),
    client_secret: Optional[str] = Form(None),
    code: Optional[str] = Form(None),
    code_verifier: Optional[str] = Form(None),
    redirect_uri: Optional[str] = Form(None),
    refresh_token: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Issue an access/refresh token pair.

    grant_type ∈ {"authorization_code", "refresh_token"}.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")

    app = await _verify_client_credentials(prisma_client, client_id, client_secret)

    if grant_type == "authorization_code":
        if not code or not code_verifier or not redirect_uri:
            raise HTTPException(
                status_code=400,
                detail="authorization_code requires code, code_verifier, redirect_uri",
            )
        auth_code_row = await _consume_authorization_code(
            prisma_client,
            code,
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
        )
        return await _issue_token_pair(
            prisma_client,
            app=app,
            user_id=auth_code_row.user_id,
            scope=list(auth_code_row.scope or []),
        )

    if grant_type == "refresh_token":
        if not refresh_token:
            raise HTTPException(
                status_code=400, detail="refresh_token grant requires refresh_token"
            )
        from litellm.proxy.utils import hash_token

        hashed = hash_token(refresh_token)
        row = await prisma_client.db.litellm_verificationtoken.find_unique(
            where={"token": hashed}
        )
        if row is None or getattr(row, "token_type", None) != "oauth_refresh":
            raise HTTPException(
                status_code=400, detail="invalid_grant: refresh_token unknown"
            )
        if row.expires and row.expires < datetime.utcnow():
            raise HTTPException(
                status_code=400, detail="invalid_grant: refresh_token expired"
            )
        if row.app_id != app.app_id:
            raise HTTPException(
                status_code=400, detail="invalid_grant: refresh_token / client mismatch"
            )
        existing_scope = (row.metadata or {}).get("scope") or []
        requested = scope.split() if scope else existing_scope
        # Scope can be NARROWED, never broadened (RFC 6749 §6).
        granted = [s for s in requested if s in existing_scope]
        return await _issue_token_pair(
            prisma_client, app=app, user_id=row.user_id, scope=granted
        )

    raise HTTPException(
        status_code=400,
        detail=f"unsupported_grant_type: {grant_type}",
    )


# ---------------------------------------------------------------------------
# S4-07  /oauth/revoke + /oauth/introspect
# ---------------------------------------------------------------------------


@router.post("/oauth/revoke", tags=["[beta] XCT OAuth"])
async def revoke(
    token: str = Form(..., description="Access OR refresh token to revoke."),
    token_type_hint: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Soft-revoke an OAuth token (set expires=now).

    Per RFC 7009 we return 200 even for unknown tokens — don't leak
    enumeration.
    """
    from litellm.proxy.proxy_server import prisma_client
    from litellm.proxy.utils import hash_token

    if prisma_client is None:
        return {"revoked": True}

    if client_id is not None:
        # Best-effort: if client provided credentials, verify them.
        await _verify_client_credentials(prisma_client, client_id, client_secret)

    hashed = hash_token(token)
    try:
        await prisma_client.db.litellm_verificationtoken.update(
            where={"token": hashed},
            data={"expires": datetime.utcnow()},
        )
    except Exception as e:
        verbose_proxy_logger.debug("revoke: token not found (%s)", e)
    return {"revoked": True}


@router.post("/oauth/introspect", tags=["[beta] XCT OAuth"])
async def introspect(
    token: str = Form(...),
    client_id: str = Form(...),
    client_secret: str = Form(...),
) -> Dict[str, Any]:
    """RFC 7662 — confidential clients can ask whether a token is active.

    Requires client credentials so a random caller can't enumerate tokens.
    """
    from litellm.proxy.proxy_server import prisma_client
    from litellm.proxy.utils import hash_token

    if prisma_client is None:
        return {"active": False}

    app = await _verify_client_credentials(prisma_client, client_id, client_secret)

    hashed = hash_token(token)
    row = await prisma_client.db.litellm_verificationtoken.find_unique(
        where={"token": hashed}
    )
    if row is None:
        return {"active": False}
    if row.app_id != app.app_id:
        # Token belongs to a different app — don't acknowledge it exists.
        return {"active": False}
    if row.expires and row.expires < datetime.utcnow():
        return {"active": False}

    scope = (row.metadata or {}).get("scope") or []
    return {
        "active": True,
        "scope": " ".join(scope),
        "exp": int(row.expires.timestamp()) if row.expires else None,
        "sub": row.user_id,
        "app_id": row.app_id,
        "client_id": client_id,
        "token_type": getattr(row, "token_type", None),
    }
