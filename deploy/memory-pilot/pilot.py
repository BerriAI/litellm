"""An isolated office pilot that preserves upstream gateway credentials."""

import asyncio
import hashlib
import os
import secrets
from contextvars import ContextVar
from typing import Final

import httpx
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy._types import UI_TEAM_ID, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken
from litellm.proxy.auth.user_api_key_auth import _get_bearer_token_or_received_api_key
from litellm.repositories.verification_token_repository import VerificationTokenRepository
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.utils import CallTypesLiteral

_UPSTREAM: Final = os.environ["UPSTREAM_LITELLM_BASE_URL"].rstrip("/")
_CREDENTIAL: Final[ContextVar[str | None]] = ContextVar("memory_pilot_credential", default=None)
_INFERENCE: Final = frozenset(
    ("/chat/completions", "/v1/chat/completions", "/responses", "/v1/responses", "/v1/messages")
)
_SELF_SERVICE: Final = frozenset(("/v2/memory/status", "/v2/memory/preference", "/v2/memory/entries"))


class ForwardCredential(CustomLogger):
    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict[str, object], call_type: CallTypesLiteral
    ) -> dict[str, object]:
        credential: Final = _CREDENTIAL.get()
        if credential is None:
            raise HTTPException(status_code=403, detail="Use your upstream gateway key for model calls")
        return {**data, "api_key": credential, "api_base": _UPSTREAM}


forward_credential: Final = ForwardCredential()


class PilotGateway:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.validation_slots = asyncio.Semaphore(16)
        self.upstream = get_async_httpx_client(
            httpxSpecialProvider.PassThroughEndpoint,
            params={"timeout": 20, "client_alias": "memory-pilot-upstream"},
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            try:
                await self.app(scope, receive, send)
            finally:
                await self.upstream.close()
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request: Final = Request(scope, receive)
        credential: Final = _get_bearer_token_or_received_api_key(
            request.headers.get("x-litellm-api-key")
            or request.headers.get("authorization")
            or request.headers.get("x-api-key")
            or ""
        )
        from litellm.proxy.proxy_server import master_key, prisma_client

        if not credential or master_key and secrets.compare_digest(credential, master_key):
            await self.app(scope, receive, send)
            return
        if prisma_client is None:
            await JSONResponse({"error": "Pilot database unavailable"}, status_code=503)(scope, receive, send)
            return
        digest: Final = hashlib.sha256(credential.encode()).hexdigest()
        tokens: Final = VerificationTokenRepository(prisma_client)
        local_key: Final = await tokens.find_by_id(digest)
        if local_key and local_key.team_id == UI_TEAM_ID:
            await self.app(scope, receive, send)
            return
        if (
            not credential.startswith("sk-")
            and ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(credential) is not None
        ):
            await self.app(scope, receive, send)
            return
        path: Final = request.url.path.rstrip("/")
        if path not in _INFERENCE | _SELF_SERVICE | {"/models", "/v1/models"} and not path.startswith(
            "/v2/memory/entries/"
        ):
            await JSONResponse(
                {"error": "Upstream keys can only use inference and their own memories"}, status_code=403
            )(scope, receive, send)
            return
        try:
            await asyncio.wait_for(self.validation_slots.acquire(), timeout=0.05)
        except TimeoutError:
            await JSONResponse(
                {"error": "Pilot credential validation is busy; retry shortly"},
                status_code=503,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
            return
        try:
            models: Final = await self.upstream.get(
                _UPSTREAM + "/v1/models", headers={"Authorization": "Bearer " + credential}
            )
        except httpx.HTTPError:
            await JSONResponse({"error": "Upstream gateway unavailable"}, status_code=503)(scope, receive, send)
            return
        finally:
            self.validation_slots.release()
        if models.is_error:
            await JSONResponse({"error": "Upstream gateway rejected this key"}, status_code=models.status_code)(
                scope, receive, send
            )
            return
        if path in ("/models", "/v1/models"):
            await JSONResponse(models.json())(scope, receive, send)
            return
        await tokens.table.upsert(
            where={"token": digest},
            data={
                "create": {"token": digest, "models": [], "key_alias": "Memory pilot " + digest[:8]},
                "update": {},
            },
        )
        token: Final = _CREDENTIAL.set(credential)
        try:
            await self.app(scope, receive, send)
        finally:
            _CREDENTIAL.reset(token)


def create_app() -> PilotGateway:
    from litellm.proxy.proxy_server import app

    return PilotGateway(app)
