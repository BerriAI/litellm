from __future__ import annotations

import secrets
from typing import cast

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse

from litellm.proxy.auth_v2 import errors
from litellm.proxy.auth_v2.authenticators import apply_role_policy
from litellm.proxy.auth_v2.models import AuthMethod
from ..services.redirects import safe_relay_state
from litellm.proxy.auth_v2.resolvers import ProvisioningStore
from litellm.proxy.auth_v2.security import AuthSecurity
from litellm.proxy.auth_v2.sessions.types import OAuthTransaction, SessionState

from ..services.oidc import mapped_claims, providers_by_key, user_from_userinfo
from .dependencies import get_auth, get_oauth_registry

router = APIRouter(prefix="/auth/oidc", tags=["oidc"])


@router.get("/{provider}/login")
async def login(
    provider: str,
    request: Request,
    auth: AuthSecurity = Depends(get_auth),
    oauth: OAuth = Depends(get_oauth_registry),
) -> RedirectResponse:
    session = auth.config.session
    client = oauth.create_client(provider)
    if client is None:
        raise errors.unknown_provider()
    redirect_uri = str(request.url_for("oidc_callback", provider=provider))
    relay = safe_relay_state(request.query_params.get("next"), session.default_redirect_path)
    authorization = await client.create_authorization_url(redirect_uri)
    txn_id = secrets.token_urlsafe(32)
    await auth.oauth_txn_store.set(
        txn_id,
        OAuthTransaction(
            provider=provider,
            state=authorization["state"],
            redirect_uri=redirect_uri,
            relay=relay,
            nonce=authorization.get("nonce"),
            code_verifier=authorization.get("code_verifier"),
        ),
    )
    response = RedirectResponse(authorization["url"], status_code=303)
    response.set_cookie(
        session.login_cookie,
        txn_id,
        httponly=True,
        samesite="lax",
        secure=session.secure,
        max_age=session.login_state_ttl,
    )
    return response


@router.get("/{provider}/callback", name="oidc_callback")
async def callback(
    provider: str,
    request: Request,
    auth: AuthSecurity = Depends(get_auth),
    oauth: OAuth = Depends(get_oauth_registry),
) -> RedirectResponse:
    session = auth.config.session
    client = oauth.create_client(provider)
    if client is None:
        raise errors.unknown_provider()
    txn_id = request.cookies.get(session.login_cookie)
    txn = await auth.oauth_txn_store.pop(txn_id) if txn_id else None
    if txn is None or txn["provider"] != provider:
        raise errors.invalid_login_state()
    returned_state = request.query_params.get("state")
    if not returned_state or returned_state != txn["state"]:
        raise errors.state_mismatch()
    error = request.query_params.get("error")
    if error:
        raise errors.oidc_provider_error(error)
    code = request.query_params.get("code")
    if not code:
        raise errors.missing_authorization_code()

    token = await client.fetch_access_token(
        redirect_uri=txn["redirect_uri"],
        code=code,
        code_verifier=txn["code_verifier"],
        state=txn["state"],
    )
    if token.get("id_token"):
        userinfo = await client.parse_id_token(token, nonce=txn["nonce"])
    else:
        userinfo = await client.userinfo(token=token)

    info = dict(userinfo)
    provider_config = providers_by_key(auth.config.oidc_providers)[provider]

    store = cast(ProvisioningStore, auth.resolver)
    await store.upsert_user(user_from_userinfo(info))

    claims = mapped_claims(info)
    apply_role_policy(claims, provider_config)
    session_id = secrets.token_urlsafe(32)
    await auth.session_store.set(
        session_id,
        SessionState(
            method=AuthMethod.OIDC.value,
            subject=info.get("sub", ""),
            issuer=info.get("iss") or provider_config.issuer,
            claims=claims,
        ),
    )
    target = safe_relay_state(txn["relay"], session.default_redirect_path)
    response = RedirectResponse(target, status_code=303)
    response.set_cookie(
        session.cookie,
        session_id,
        httponly=True,
        samesite="lax",
        secure=session.secure,
    )
    return response
