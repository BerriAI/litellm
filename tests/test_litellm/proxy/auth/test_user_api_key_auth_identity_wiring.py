import contextlib
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath("../../../.."))

import jwt
import pytest
from fastapi import Request
from starlette.datastructures import URL

import litellm.proxy.proxy_server as _proxy_server_mod
from litellm.caching.dual_cache import DualCache
from litellm.identity import IdentityContext
from litellm.identity.principal import ApiKeyPrincipal
from litellm.proxy._types import LiteLLM_JWTAuth, LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import _user_api_key_auth_builder

MASTER_KEY = "sk-master-key"
JWT_TOKEN = jwt.encode({"sub": "user1"}, "test-secret", algorithm="HS256")


def _master_path_attrs():
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.internal_usage_cache = MagicMock()
    proxy_logging_obj.internal_usage_cache.dual_cache = AsyncMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)

    user_api_key_cache = MagicMock()
    user_api_key_cache.async_get_cache = AsyncMock(return_value=None)
    user_api_key_cache.async_set_cache = AsyncMock()

    return {
        "prisma_client": None,
        "user_api_key_cache": user_api_key_cache,
        "proxy_logging_obj": proxy_logging_obj,
        "master_key": MASTER_KEY,
        "general_settings": {},
        "llm_model_list": [],
        "llm_router": None,
        "open_telemetry_logger": None,
        "model_max_budget_limiter": MagicMock(),
        "user_custom_auth": None,
        "jwt_handler": None,
        "litellm_proxy_admin_name": "admin",
    }


@contextmanager
def _proxy_attrs(attrs):
    originals = {attr: getattr(_proxy_server_mod, attr, None) for attr in attrs}
    try:
        for attr, value in attrs.items():
            setattr(_proxy_server_mod, attr, value)
        yield
    finally:
        for attr, value in originals.items():
            setattr(_proxy_server_mod, attr, value)


def _http_request():
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [(b"authorization", f"Bearer {MASTER_KEY}".encode())],
            "client": ("1.2.3.4", 0),
        }
    )
    request._url = URL(url="/chat/completions")
    return request


async def _run_master_builder(request, request_data):
    return await _user_api_key_auth_builder(
        request=request,
        api_key=f"Bearer {MASTER_KEY}",
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        request_data=request_data,
    )


@pytest.mark.asyncio
async def test_identity_context_populated_on_request_state():
    request = _http_request()
    with (
        _proxy_attrs(_master_path_attrs()),
        patch(
            "litellm.proxy.auth.user_api_key_auth._cache_key_object",
            new_callable=AsyncMock,
        ),
    ):
        await _run_master_builder(request, {"model": "gpt-4o-mini"})

    ctx = request.state.identity_context
    assert isinstance(ctx, IdentityContext)
    assert ctx.principal.kind == "api_key"


@pytest.mark.asyncio
async def test_end_user_id_read_from_identity_context_when_present():
    pinned = IdentityContext(
        principal=ApiKeyPrincipal(token_hash="hash"),
        end_user_id="eu-from-context",
    )
    request = _http_request()
    with (
        _proxy_attrs(_master_path_attrs()),
        patch(
            "litellm.proxy.auth.user_api_key_auth._cache_key_object",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.resolve_identity",
            new_callable=AsyncMock,
            return_value=pinned,
        ),
    ):
        result = await _run_master_builder(
            request, {"model": "gpt-4o-mini", "user": "eu-from-body"}
        )

    assert result.end_user_id == "eu-from-context"


@pytest.mark.asyncio
async def test_parity_uak_for_api_key_request_matches_pre_integration():
    request_data = {"model": "gpt-4o-mini"}

    request_wired = _http_request()
    with (
        _proxy_attrs(_master_path_attrs()),
        patch(
            "litellm.proxy.auth.user_api_key_auth._cache_key_object",
            new_callable=AsyncMock,
        ),
    ):
        wired = await _run_master_builder(request_wired, request_data)

    request_legacy = _http_request()
    with (
        _proxy_attrs(_master_path_attrs()),
        patch(
            "litellm.proxy.auth.user_api_key_auth._cache_key_object",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.resolve_identity",
            new_callable=AsyncMock,
            side_effect=RuntimeError("resolve_identity unavailable"),
        ),
    ):
        legacy = await _run_master_builder(request_legacy, request_data)

    assert request_legacy.state.identity_context is None
    assert wired.user_role == legacy.user_role == LitellmUserRoles.PROXY_ADMIN
    assert wired.user_id == legacy.user_id
    assert wired.api_key == legacy.api_key
    assert wired.end_user_id == legacy.end_user_id


def _jwt_result():
    return {
        "is_proxy_admin": True,
        "team_object": None,
        "user_object": None,
        "end_user_object": None,
        "org_object": None,
        "token": JWT_TOKEN,
        "team_id": "jwt-team",
        "user_id": "jwt-human-user",
        "end_user_id": None,
        "org_id": None,
        "team_membership": None,
        "jwt_claims": {"sub": "user1"},
    }


async def _run_jwt_auth(resolve_raises):
    request = MagicMock()
    request.url.path = "/v1/chat/completions"
    request.method = "POST"
    request.headers = {"authorization": f"Bearer {JWT_TOKEN}"}
    request.query_params = {}
    request.state = SimpleNamespace()

    patches = [
        patch("litellm.proxy.proxy_server.general_settings", {"enable_jwt_auth": True}),
        patch("litellm.proxy.proxy_server.premium_user", True),
        patch("litellm.proxy.proxy_server.master_key", "sk-master"),
        patch("litellm.proxy.proxy_server.prisma_client", None),
        patch(
            "litellm.proxy.auth.user_api_key_auth.JWTAuthManager.auth_builder",
            new_callable=AsyncMock,
            return_value=_jwt_result(),
        ),
    ]
    if resolve_raises:
        patches.append(
            patch(
                "litellm.proxy.auth.user_api_key_auth.resolve_identity",
                new_callable=AsyncMock,
                side_effect=RuntimeError("resolve_identity unavailable"),
            )
        )

    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        _proxy_server_mod.jwt_handler.update_environment(
            prisma_client=None,
            user_api_key_cache=DualCache(),
            litellm_jwtauth=LiteLLM_JWTAuth(),
        )
        result = await _user_api_key_auth_builder(
            request=request,
            api_key=f"Bearer {JWT_TOKEN}",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={"model": "gpt-4o-mini"},
        )
    return request, result


@pytest.mark.asyncio
async def test_parity_uak_for_jwt_request_matches_pre_integration():
    request_wired, wired = await _run_jwt_auth(resolve_raises=False)
    request_legacy, legacy = await _run_jwt_auth(resolve_raises=True)

    assert isinstance(request_wired.state.identity_context, IdentityContext)
    assert request_wired.state.identity_context.principal.kind == "jwt"
    assert request_legacy.state.identity_context is None

    assert wired.user_id == legacy.user_id == "jwt-human-user"
    assert wired.team_id == legacy.team_id == "jwt-team"
    assert wired.user_role == legacy.user_role == LitellmUserRoles.PROXY_ADMIN
