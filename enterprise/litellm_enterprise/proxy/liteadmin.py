from __future__ import annotations

import hashlib
import hmac
import html
import os
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated, Final
from urllib.parse import urlencode, urlsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._experimental.mcp_server.oauth_utils import get_request_base_url
from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.types.proxy.auth.auth_checks import UserNotFoundError

router: Final = APIRouter()
_PREFIX: Final = "/liteadmin/slack/connect/"
_COOKIE: Final = "__Host-litellm-slack-connect-"
_HEADERS: Final = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "same-origin",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
}


class LinkDetails(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    workspace_id: str = Field(min_length=1, max_length=64)
    slack_user_id: str = Field(min_length=1, max_length=64)
    email: str = Field(min_length=1, max_length=320)


class AdminSession(BaseModel):
    model_config = ConfigDict(frozen=True)
    user_id: str
    credential: SecretStr
    expires_at: float


@dataclass(frozen=True, slots=True)
class NativeAdminContext:
    worker_url: str
    service_token: SecretStr
    client: httpx.AsyncClient
    session_user: Callable[[Request], Awaitable[str | None]]
    load_user: Callable[[str], Awaitable[LiteLLM_UserTable | None]]
    mint_session: Callable[[LiteLLM_UserTable], AdminSession]

    async def worker_request(self, token: str, session: AdminSession | None = None) -> httpx.Response:
        if re.fullmatch(r"[A-Za-z0-9_-]{43}", token) is None:
            raise HTTPException(410, "Connection link expired. Send connect in Slack for a new link")
        try:
            response: Final = await self.client.request(
                "GET" if session is None else "POST",
                f"{self.worker_url}/internal/liteadmin/links/{token}",
                headers={"X-LiteLLM-Admin-Agent-Token": self.service_token.get_secret_value()},
                json=None
                if session is None
                else {
                    "user_id": session.user_id,
                    "credential": session.credential.get_secret_value(),
                    "expires_at": session.expires_at,
                },
                timeout=15,
                follow_redirects=False,
            )
        except httpx.HTTPError:
            raise HTTPException(503, "LiteAdmin is temporarily unavailable") from None
        if response.status_code == 410:
            raise HTTPException(410, "Connection link expired. Send connect in Slack for a new link")
        if response.status_code == 403:
            raise HTTPException(403, "Connect your own active LiteLLM proxy-admin account with the same email as Slack")
        if response.status_code != 200:
            raise HTTPException(503, "LiteAdmin could not verify this connection")
        return response

    async def details(self, token: str) -> LinkDetails:
        response: Final = await self.worker_request(token)
        try:
            return LinkDetails.model_validate_json(response.content)
        except ValidationError:
            raise HTTPException(503, "LiteAdmin could not verify this connection") from None

    async def admin(self, user_id: str, details: LinkDetails) -> LiteLLM_UserTable:
        user: Final = await self.load_user(user_id)
        if (
            user is None
            or user.user_role != LitellmUserRoles.PROXY_ADMIN.value
            or not user.user_email
            or user.user_email.strip().casefold() != details.email.strip().casefold()
        ):
            raise HTTPException(403, "Connect your own active LiteLLM proxy-admin account with the same email as Slack")
        return user


def _page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f'<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
        "<style>body{font:17px system-ui;color:#18252f;max-width:560px;margin:10vh auto;padding:24px}"
        "p{line-height:1.6}button{font:inherit;border:0;border-radius:8px;padding:14px 20px;background:#5b3fd1;"
        "color:white;cursor:pointer}small{color:#556}</style>"
        f"<main><h1>{html.escape(title)}</h1>{body}</main></html>",
        headers=_HEADERS,
    )


def _cookie_name(token: str) -> str:
    return _COOKIE + hashlib.sha256(token.encode()).hexdigest()[:16]


async def _session_user(request: Request) -> str | None:
    from litellm.proxy._experimental.mcp_server.byok_oauth_endpoints import (
        get_authenticated_browser_user_id,
    )

    return await get_authenticated_browser_user_id(request)


async def _load_user(user_id: str) -> LiteLLM_UserTable | None:
    from litellm.proxy.auth.auth_checks import get_user_object
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(503, "LiteAdmin requires a database")
    try:
        return await get_user_object(
            user_id=user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            check_db_only=True,
        )
    except UserNotFoundError:
        return None
    except Exception:
        raise HTTPException(503, "LiteAdmin could not verify your current permissions") from None


def mint_admin_session(user: LiteLLM_UserTable) -> AdminSession:
    from litellm.proxy.auth.auth_checks import LITELLM_SESSION_TOKEN_PREFIX
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_bearer_token

    expires: Final = datetime.now(timezone.utc) + timedelta(hours=24)
    auth: Final = UserAPIKeyAuth(
        token="liteadmin-" + secrets.token_urlsafe(24),
        key_name="LiteAdmin Slack",
        key_alias="LiteAdmin Slack",
        user_id=user.user_id,
        user_role=LitellmUserRoles.PROXY_ADMIN,
        models=TypeAdapter(list[str]).validate_python(user.model_dump().get("models", [])),
        expires=expires,
        is_session_token=True,
    )
    return AdminSession(
        user_id=user.user_id,
        credential=SecretStr(
            encrypt_bearer_token(auth.model_dump_json(exclude_none=True), LITELLM_SESSION_TOKEN_PREFIX)
        ),
        expires_at=expires.timestamp(),
    )


def validate_native_configuration(
    worker_url: str, service_token: str, enterprise: bool, database_available: bool
) -> None:
    if not worker_url:
        raise HTTPException(404, "LiteAdmin Slack is not enabled")
    if not enterprise:
        raise HTTPException(403, "LiteAdmin Slack requires LiteLLM Enterprise")
    if not database_available:
        raise HTTPException(503, "LiteAdmin requires a database")
    try:
        parsed: Final = urlsplit(worker_url)
        port: Final = parsed.port
    except ValueError:
        raise HTTPException(503, "LiteAdmin worker configuration is invalid") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or port == 0
        or parsed.username
        or parsed.password
        or parsed.path
        or parsed.query
        or parsed.fragment
        or len(service_token) < 32
        or any(character.isspace() for character in service_token)
    ):
        raise HTTPException(503, "LiteAdmin worker configuration is invalid")


async def native_admin_context() -> NativeAdminContext:
    from litellm.proxy.proxy_server import premium_user, prisma_client

    worker_url: Final = os.getenv("LITELLM_ADMIN_AGENT_URL", "").rstrip("/")
    service_token: Final = os.getenv("ADMIN_AGENT_SERVICE_TOKEN", "")
    validate_native_configuration(worker_url, service_token, premium_user is True, prisma_client is not None)
    client: Final = get_async_httpx_client(
        llm_provider="liteadmin_native", params={"timeout": 15.0, "follow_redirects": False}
    ).client
    return NativeAdminContext(
        worker_url, SecretStr(service_token), client, _session_user, _load_user, mint_admin_session
    )


@router.get(_PREFIX + "{token}", include_in_schema=False, response_class=HTMLResponse)
async def connect_page(
    request: Request,
    token: str,
    context: Annotated[NativeAdminContext, Depends(native_admin_context)],
) -> Response:
    details: Final = await context.details(token)
    base_url: Final = get_request_base_url(request)
    parsed_base: Final = urlsplit(base_url)
    if parsed_base.scheme != "https":
        raise HTTPException(400, "LiteAdmin account connections require HTTPS")
    user_id: Final = await context.session_user(request)
    if user_id is None:
        return RedirectResponse(
            base_url + "/sso/key/generate?" + urlencode({"return_to": parsed_base.path + _PREFIX + token}),
            status_code=303,
            headers=_HEADERS,
        )
    await context.admin(user_id, details)
    csrf: Final = secrets.token_urlsafe(32)
    page: Final = _page(
        "Connect LiteAdmin to Slack",
        f"<p>Connect <strong>{html.escape(details.email)}</strong> to LiteAdmin in your Slack workspace?</p>"
        "<p>Model requests and administrative actions will use your own LiteLLM account and current permissions</p>"
        f'<form method="post"><input type="hidden" name="csrf" value="{csrf}">'
        '<button type="submit">Connect account</button></form>'
        "<p><small>This connection lasts 24 hours. Send disconnect in Slack to remove the saved session</small></p>",
    )
    page.set_cookie(_cookie_name(token), csrf, max_age=600, secure=True, httponly=True, samesite="strict", path="/")
    return page


@router.post(_PREFIX + "{token}", include_in_schema=False, response_class=HTMLResponse)
async def connect_account(
    request: Request,
    token: str,
    context: Annotated[NativeAdminContext, Depends(native_admin_context)],
) -> Response:
    base_url: Final = get_request_base_url(request)
    parsed_base: Final = urlsplit(base_url)
    origin: Final = f"{parsed_base.scheme}://{parsed_base.netloc}"
    if parsed_base.scheme != "https" or request.headers.get("Origin") != origin:
        raise HTTPException(403, "Reopen your private Slack connection link")
    if request.headers.get("Content-Type", "").split(";", 1)[0] != "application/x-www-form-urlencoded":
        raise HTTPException(400, "Expected a connection form")
    form: Final = await request.form(max_fields=1, max_files=0, max_part_size=1024)
    supplied: Final = form.get("csrf")
    expected: Final = request.cookies.get(_cookie_name(token), "")
    if (
        not isinstance(supplied, str)
        or len(expected) != 43
        or len(supplied) != 43
        or not hmac.compare_digest(supplied.encode(), expected.encode())
    ):
        raise HTTPException(403, "Reopen your private Slack connection link")
    user_id: Final = await context.session_user(request)
    if user_id is None:
        raise HTTPException(401, "Your login expired. Reopen your private Slack connection link")
    details: Final = await context.details(token)
    user: Final = await context.admin(user_id, details)
    await context.worker_request(token, context.mint_session(user))
    page: Final = _page(
        "Account connected", "<p>Return to Slack and ask LiteAdmin to list your teams or check a budget</p>"
    )
    page.delete_cookie(_cookie_name(token), path="/", secure=True, httponly=True, samesite="strict")
    return page
