import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from starlette.datastructures import URL

import litellm.proxy.proxy_server as proxy_server
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy._types import UserAPIKeyAuth, hash_token
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class FakeTTLCache:
    def __init__(self) -> None:
        self.now = 0.0
        self._values = {}

    async def async_get_cache(self, key, **kwargs):
        item = self._values.get(key)
        if item is None:
            return None

        value, expires_at = item
        if expires_at is not None and expires_at <= self.now:
            self._values.pop(key, None)
            return None
        return value

    async def async_set_cache(self, key, value, ttl: Optional[float] = None, **kwargs):
        expires_at = None if ttl is None else self.now + ttl
        self._values[key] = (value, expires_at)

    def set_cache(self, key, value, ttl: Optional[float] = None, **kwargs):
        expires_at = None if ttl is None else self.now + ttl
        self._values[key] = (value, expires_at)

    def delete_cache(self, key):
        self._values.pop(key, None)

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakePrismaClient:
    def __init__(self, hashed_token: str, token_data: Optional[dict]) -> None:
        self.hashed_token = hashed_token
        self.token_data = token_data

    async def get_data(
        self,
        token: str,
        table_name: str,
        parent_otel_span=None,
        proxy_logging_obj=None,
    ):
        assert table_name == "combined_view"
        if token != self.hashed_token or self.token_data is None:
            return None
        return UserAPIKeyAuth(**self.token_data)

    def update_token(self, **updates) -> None:
        assert self.token_data is not None
        self.token_data = {**self.token_data, **updates}

    def delete_token(self) -> None:
        self.token_data = None


def _request() -> Request:
    request = Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": "/chat/completions",
            "headers": [],
        }
    )
    request._url = URL(url="/chat/completions")
    request._body = b"{}"
    return request


async def _authenticate(
    *,
    api_key: str,
    cache: FakeTTLCache,
    prisma_client: FakePrismaClient,
) -> UserAPIKeyAuth:
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.post_call_failure_hook = AsyncMock(return_value=None)
    proxy_logging_obj.service_logging_obj = MagicMock()
    proxy_logging_obj.service_logging_obj.async_service_success_hook = AsyncMock()
    proxy_logging_obj.service_logging_obj.async_service_failure_hook = AsyncMock()
    model_max_budget_limiter = MagicMock()
    model_max_budget_limiter.is_key_within_model_budget = AsyncMock()
    model_max_budget_limiter.is_end_user_within_model_budget = AsyncMock()

    with (
        patch.object(proxy_server, "user_api_key_cache", cache),
        patch.object(proxy_server, "prisma_client", prisma_client),
        patch.object(proxy_server, "master_key", "sk-master"),
        patch.object(proxy_server, "general_settings", {}),
        patch.object(proxy_server, "llm_model_list", []),
        patch.object(proxy_server, "llm_router", None),
        patch.object(proxy_server, "proxy_logging_obj", proxy_logging_obj),
        patch.object(
            proxy_server, "model_max_budget_limiter", model_max_budget_limiter
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ),
    ):
        result = await user_api_key_auth(
            request=_request(),
            api_key=f"Bearer {api_key}",
        )

    await asyncio.sleep(0)
    return result


@pytest.mark.asyncio
async def test_should_refresh_rate_limits_after_cache_ttl_while_key_keeps_calling():
    api_key = "sk-onsite-rate-limit"
    hashed_token = hash_token(api_key)
    cache = FakeTTLCache()
    prisma_client = FakePrismaClient(
        hashed_token=hashed_token,
        token_data={
            "token": hashed_token,
            "models": ["gpt-4o"],
            "rpm_limit": 100,
        },
    )

    first_auth = await _authenticate(
        api_key=api_key,
        cache=cache,
        prisma_client=prisma_client,
    )
    assert first_auth.rpm_limit == 100

    prisma_client.update_token(rpm_limit=1)

    cache.advance(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL - 1)
    stale_auth_before_ttl = await _authenticate(
        api_key=api_key,
        cache=cache,
        prisma_client=prisma_client,
    )
    assert stale_auth_before_ttl.rpm_limit == 100

    cache.advance(2)
    refreshed_auth = await _authenticate(
        api_key=api_key,
        cache=cache,
        prisma_client=prisma_client,
    )

    assert refreshed_auth.rpm_limit == 1


@pytest.mark.asyncio
async def test_should_stop_serving_deleted_key_after_cache_ttl_while_key_keeps_calling():
    api_key = "sk-onsite-deleted-key"
    hashed_token = hash_token(api_key)
    cache = FakeTTLCache()
    prisma_client = FakePrismaClient(
        hashed_token=hashed_token,
        token_data={
            "token": hashed_token,
            "models": ["gpt-4o"],
            "rpm_limit": 100,
        },
    )

    await _authenticate(
        api_key=api_key,
        cache=cache,
        prisma_client=prisma_client,
    )
    prisma_client.delete_token()

    cache.advance(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL - 1)
    stale_auth_before_ttl = await _authenticate(
        api_key=api_key,
        cache=cache,
        prisma_client=prisma_client,
    )
    assert stale_auth_before_ttl.rpm_limit == 100

    cache.advance(2)
    with pytest.raises(Exception):
        await _authenticate(
            api_key=api_key,
            cache=cache,
            prisma_client=prisma_client,
        )
