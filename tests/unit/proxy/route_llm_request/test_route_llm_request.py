# Cache-key and cache-control stripping at the proxy request boundary
# (fork hardening R2-4, R2-5). route_request is the boundary where the
# client body becomes provider kwargs.

import pytest
from unittest.mock import AsyncMock, MagicMock

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.route_llm_request import route_request, strip_client_cache_authority_fields


def _router() -> MagicMock:
    router = MagicMock()
    router.acompletion = AsyncMock(return_value={"ok": True})
    router.model_names = []
    router.model_list = None
    return router


@pytest.mark.asyncio
async def test_route_request_strips_client_cache_key_and_overrides(monkeypatch):
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "shared prompt"}],
        "cache_key": "spend:team_member:admin:team-1",
        "cache": {"ttl": 31536000, "namespace": "tenant-a"},
        "metadata": {"redis_namespace": "tenant-a"},
    }
    router = _router()
    monkeypatch.setattr(
        "litellm.proxy.route_llm_request.allow_client_cache_control_params",
        lambda: False,
    )

    await route_request(
        data=data, llm_router=router, user_model=None, route_type="acompletion"
    )

    kwargs = router.acompletion.call_args.kwargs
    assert kwargs.get("cache_key") is None
    assert "ttl" not in (kwargs.get("cache") or {})
    assert "namespace" not in (kwargs.get("cache") or {})
    assert "redis_namespace" not in kwargs.get("metadata", {})


@pytest.mark.asyncio
async def test_route_request_keeps_cache_control_when_opted_in(monkeypatch):
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "shared prompt"}],
        "cache": {"ttl": 60, "namespace": "tenant-a"},
    }
    router = _router()
    monkeypatch.setattr(
        "litellm.proxy.route_llm_request.allow_client_cache_control_params",
        lambda: True,
    )

    await route_request(
        data=data, llm_router=router, user_model=None, route_type="acompletion"
    )

    kwargs = router.acompletion.call_args.kwargs
    assert kwargs["cache"] == {"ttl": 60, "namespace": "tenant-a"}


@pytest.mark.asyncio
async def test_route_request_keeps_harmless_cache_headers(monkeypatch):
    """s-maxage/no-store only bound stale acceptance, never a read/write key."""
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "shared prompt"}],
        "cache": {"s-maxage": 60, "no-cache": True},
    }
    router = _router()
    monkeypatch.setattr(
        "litellm.proxy.route_llm_request.allow_client_cache_control_params",
        lambda: False,
    )

    await route_request(
        data=data, llm_router=router, user_model=None, route_type="acompletion"
    )

    kwargs = router.acompletion.call_args.kwargs
    assert kwargs["cache"] == {"s-maxage": 60, "no-cache": True}


async def test_strip_client_cache_authority_fields_removes_only_authority_keys():
    data = {
        "cache_key": "attacker-chosen",
        "cache": {"ttl": 1, "namespace": "n", "s-maxage": 5},
        "metadata": {"redis_namespace": "x"},
    }
    strip_client_cache_authority_fields(data)
    assert "cache_key" not in data
    assert data["cache"] == {"s-maxage": 5}
    assert "redis_namespace" not in data["metadata"]
