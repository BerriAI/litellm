from __future__ import annotations

import re
from typing import Any, Dict

from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from scim2_models import User as ScimUser

from .config import AuthConfig, OidcProviderConfig
from .resolver import ProvisioningStore


def _provider_key(provider: OidcProviderConfig) -> str:
    return re.sub(r"[^a-z0-9]+", "-", provider.issuer.lower()).strip("-")


def _user_from_userinfo(userinfo: Dict[str, Any]) -> ScimUser:
    return ScimUser(
        external_id=userinfo.get("sub"),
        user_name=userinfo.get("preferred_username") or userinfo.get("email"),
        display_name=userinfo.get("name"),
    )


def build_oidc_router(config: AuthConfig) -> APIRouter:
    oauth = OAuth()
    for provider in config.oidc_providers:
        oauth.register(
            name=_provider_key(provider),
            server_metadata_url=f"{provider.issuer.rstrip('/')}/.well-known/openid-configuration",
            client_id=provider.client_id,
            client_secret=(
                provider.client_secret.get_secret_value()
                if provider.client_secret
                else None
            ),
            client_kwargs={"scope": " ".join(provider.login_scopes)},
        )

    router = APIRouter(prefix="/auth/oidc", tags=["oidc"])

    @router.get("/{provider}/login")
    async def login(provider: str, request: Request) -> Any:
        client = oauth.create_client(provider)
        if client is None:
            raise HTTPException(status_code=404, detail="unknown provider")
        redirect_uri = request.url_for("oidc_callback", provider=provider)
        return await client.authorize_redirect(request, str(redirect_uri))

    @router.get("/{provider}/callback", name="oidc_callback")
    async def callback(provider: str, request: Request) -> JSONResponse:
        client = oauth.create_client(provider)
        if client is None:
            raise HTTPException(status_code=404, detail="unknown provider")
        token = await client.authorize_access_token(request)
        userinfo = token.get("userinfo")
        if userinfo is None:
            userinfo = await client.userinfo(token=token)
        store: ProvisioningStore = request.app.state.auth_v2.resolver
        stored = await store.upsert_user(_user_from_userinfo(dict(userinfo)))
        return JSONResponse(content=stored.model_dump())

    return router
