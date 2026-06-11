from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, cast

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from scim2_models import User as ScimUser

from .config import OIDCProviderConfig
from ..resolver import ProvisioningStore
from ..session import safe_relay_state

if TYPE_CHECKING:
    from ..security import AuthSecurity

_CLAIM_KEYS = ("email", "preferred_username", "name", "groups", "roles")


def _provider_key(provider: OIDCProviderConfig) -> str:
    return re.sub(r"[^a-z0-9]+", "-", provider.issuer.lower()).strip("-")


def _user_from_userinfo(userinfo: Dict[str, Any]) -> ScimUser:
    return ScimUser(
        external_id=userinfo.get("sub"),
        user_name=userinfo.get("preferred_username") or userinfo.get("email"),
        display_name=userinfo.get("name"),
    )


def _mapped_claims(userinfo: Dict[str, Any]) -> Dict[str, Any]:
    return {key: userinfo[key] for key in _CLAIM_KEYS if userinfo.get(key) is not None}


def build_oidc_router(auth: AuthSecurity) -> APIRouter:
    session = auth.config.session
    issuers = {_provider_key(p): p.issuer for p in auth.config.oidc_providers}
    oauth = OAuth()
    for provider in auth.config.oidc_providers:
        oauth.register(
            name=_provider_key(provider),
            server_metadata_url=f"{provider.issuer.rstrip('/')}/.well-known/openid-configuration",
            client_id=provider.client_id,
            client_secret=(
                provider.client_secret.get_secret_value()
                if provider.client_secret
                else None
            ),
            client_kwargs={
                "scope": " ".join(provider.login_scopes),
                "code_challenge_method": "S256",
            },
        )

    router = APIRouter(prefix="/auth/oidc", tags=["oidc"])

    @router.get("/{provider}/login")
    async def login(provider: str, request: Request) -> RedirectResponse:
        client = oauth.create_client(provider)
        if client is None:
            raise HTTPException(status_code=404, detail="unknown provider")
        redirect_uri = str(request.url_for("oidc_callback", provider=provider))
        relay = safe_relay_state(
            request.query_params.get("next"), session.default_redirect_path
        )
        authorization = await client.create_authorization_url(redirect_uri)
        txn_id = auth.oauth_txn_store.create_session(
            {
                "provider": provider,
                "state": authorization["state"],
                "nonce": authorization.get("nonce"),
                "code_verifier": authorization.get("code_verifier"),
                "redirect_uri": redirect_uri,
                "relay": relay,
            }
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
    async def callback(provider: str, request: Request) -> RedirectResponse:
        client = oauth.create_client(provider)
        if client is None:
            raise HTTPException(status_code=404, detail="unknown provider")
        txn_id = request.cookies.get(session.login_cookie)
        txn = auth.oauth_txn_store.pop(txn_id) if txn_id else None
        if txn is None or txn.get("provider") != provider:
            raise HTTPException(
                status_code=400, detail="invalid or expired login state"
            )
        returned_state = request.query_params.get("state")
        if not returned_state or returned_state != txn["state"]:
            raise HTTPException(status_code=400, detail="state mismatch")
        error = request.query_params.get("error")
        if error:
            raise HTTPException(status_code=400, detail=error)
        code = request.query_params.get("code")
        if not code:
            raise HTTPException(status_code=400, detail="missing authorization code")

        token = await client.fetch_access_token(
            redirect_uri=txn["redirect_uri"],
            code=code,
            code_verifier=txn.get("code_verifier"),
            state=txn["state"],
        )
        if token.get("id_token"):
            userinfo = await client.parse_id_token(token, nonce=txn.get("nonce"))
        else:
            userinfo = await client.userinfo(token=token)
        info = dict(userinfo)

        store = cast(ProvisioningStore, auth.resolver)
        await store.upsert_user(_user_from_userinfo(info))

        session_id = auth.session_store.create_session(
            {
                "method": "oidc",
                "subject": info.get("sub"),
                "issuer": info.get("iss") or issuers.get(provider),
                "claims": _mapped_claims(info),
            }
        )
        target = safe_relay_state(txn.get("relay"), session.default_redirect_path)
        response = RedirectResponse(target, status_code=303)
        response.set_cookie(
            session.cookie,
            session_id,
            httponly=True,
            samesite="lax",
            secure=session.secure,
        )
        return response

    return router
