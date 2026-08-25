"""Regression tests for `SearchAPIRouter.async_search_with_fallbacks_helper`.

Covers #37538: provider-specific keys in a `search_tools` entry's
`litellm_params` (e.g. `tool_name` for the agentcore provider) used to be
silently dropped, because the helper forwarded only `search_provider`,
`api_key`, and `api_base` to the underlying search call. Every other yaml key
was ignored with no warning. These tests are provider-free: they capture the
kwargs handed to the search function via a fake `original_generic_function`.
"""

from typing import Any

import pytest

from litellm.router_utils.search_api_router import SearchAPIRouter


class _FakeRouter:
    """Minimal stand-in exposing only the `search_tools` the helper reads."""

    def __init__(self, search_tools: list[dict[str, Any]]):
        self.search_tools = search_tools


def _make_router(litellm_params: dict[str, Any], name: str = "agentcore-search") -> _FakeRouter:
    return _FakeRouter(
        [{"search_tool_name": name, "litellm_params": {**litellm_params}}]
    )


@pytest.mark.asyncio
async def test_provider_specific_litellm_params_are_forwarded():
    """A yaml `tool_name` (and any other non-credential key) must reach the call."""
    captured: dict[str, Any] = {}

    async def fake_search(**kwargs: Any):
        captured.update(kwargs)
        return {"object": "search", "results": []}

    router = _make_router(
        {
            "search_provider": "agentcore",
            "api_base": "https://gateway.example/mcp",
            "tool_name": "MyTarget___WebSearch",
        }
    )

    await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="agentcore-search",
        original_generic_function=fake_search,
        query="anything",
        max_results=2,
    )

    assert captured["search_provider"] == "agentcore"
    assert captured["api_base"] == "https://gateway.example/mcp"
    # The previously-dropped key now reaches the provider.
    assert captured["tool_name"] == "MyTarget___WebSearch"
    # And the request-body params are still forwarded untouched.
    assert captured["query"] == "anything"
    assert captured["max_results"] == 2


@pytest.mark.asyncio
async def test_request_body_params_win_over_yaml_on_collision():
    """Per-request kwargs must override colliding yaml `litellm_params` keys."""
    captured: dict[str, Any] = {}

    async def fake_search(**kwargs: Any):
        captured.update(kwargs)
        return {"object": "search", "results": []}

    router = _make_router(
        {
            "search_provider": "agentcore",
            "tool_name": "yaml___WebSearch",
        }
    )

    await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="agentcore-search",
        original_generic_function=fake_search,
        query="anything",
        tool_name="request___WebSearch",  # collides with yaml
    )

    assert captured["tool_name"] == "request___WebSearch"


@pytest.mark.asyncio
async def test_credentials_are_not_double_passed():
    """`api_key`/`api_base` are resolved explicitly and must not also leak in as
    duplicate kwargs (which would raise `TypeError: multiple values`)."""
    captured: dict[str, Any] = {}

    async def fake_search(*, search_provider, api_key, api_base, **kwargs: Any):
        captured.update(
            {"search_provider": search_provider, "api_key": api_key, "api_base": api_base, **kwargs}
        )
        return {"object": "search", "results": []}

    router = _make_router(
        {
            "search_provider": "agentcore",
            "api_key": "sk-secret",
            "api_base": "https://gateway.example/mcp",
            "tool_name": "MyTarget___WebSearch",
        }
    )

    # Must not raise TypeError for duplicate api_key/api_base.
    await SearchAPIRouter.async_search_with_fallbacks_helper(
        router_instance=router,
        model="agentcore-search",
        original_generic_function=fake_search,
        query="anything",
    )

    assert captured["api_key"] == "sk-secret"
    assert captured["api_base"] == "https://gateway.example/mcp"
    assert captured["tool_name"] == "MyTarget___WebSearch"


@pytest.mark.asyncio
async def test_missing_search_provider_still_raises():
    """The existing guard for a missing `search_provider` must be preserved."""

    async def fake_search(**kwargs: Any):  # pragma: no cover - should not be called
        raise AssertionError("search should not be invoked")

    router = _make_router({"tool_name": "MyTarget___WebSearch"})

    with pytest.raises(ValueError, match="search_provider not found"):
        await SearchAPIRouter.async_search_with_fallbacks_helper(
            router_instance=router,
            model="agentcore-search",
            original_generic_function=fake_search,
            query="anything",
        )
