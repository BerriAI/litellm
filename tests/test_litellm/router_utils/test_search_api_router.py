"""Regression tests for `SearchAPIRouter.async_search_with_fallbacks_helper`.

Covers #37538: provider-specific keys in a `search_tools` entry's
`litellm_params` (e.g. `tool_name` for the agentcore provider) used to be
silently dropped, because the helper forwarded only `search_provider`,
`api_key`, and `api_base` to the underlying search call. Every other yaml key
was ignored with no warning. These tests are provider-free: they capture the
kwargs handed to the search function via a fake `original_generic_function`.
"""

import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from litellm.llms.base_llm.search.transformation import (
    BaseSearchConfig,
    SearchResponse,
)
from litellm.llms.bedrock.search.transformation import AgentCoreSearchConfig
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.router_utils.search_api_router import SearchAPIRouter

search_module = importlib.import_module("litellm.search.main")


class _FakeRouter:
    """Minimal stand-in exposing only the `search_tools` the helper reads."""

    def __init__(self, search_tools: list[dict[str, Any]]):
        self.search_tools = search_tools


def _make_router(litellm_params: dict[str, Any], name: str = "agentcore-search") -> _FakeRouter:
    return _FakeRouter([{"search_tool_name": name, "litellm_params": {**litellm_params}}])


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
        captured.update({"search_provider": search_provider, "api_key": api_key, "api_base": api_base, **kwargs})
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


def test_agentcore_auth_params_are_excluded_from_logging(monkeypatch: pytest.MonkeyPatch):
    config = AgentCoreSearchConfig()
    monkeypatch.setattr(
        search_module.ProviderConfigManager,
        "get_provider_search_config",
        lambda provider: config,
    )
    monkeypatch.setattr(config, "validate_environment", lambda **kwargs: {})
    monkeypatch.setattr(
        config,
        "get_complete_url",
        lambda **kwargs: "https://gateway.example/mcp",
    )

    captured: dict[str, Any] = {}

    def fake_search(**kwargs: Any):
        captured.update(kwargs)
        return SearchResponse(results=[])

    monkeypatch.setattr(search_module.base_llm_http_handler, "search", fake_search)
    logging_obj = MagicMock()

    search_module.search.__wrapped__(
        query="test",
        search_provider="agentcore",
        litellm_logging_obj=logging_obj,
        tool_name="target___WebSearch",
        aws_access_key_id="access",
        aws_secret_access_key="secret",
        aws_session_token="token",
        aws_region_name="us-east-1",
    )

    assert captured["optional_params"] == {"tool_name": "target___WebSearch"}
    assert captured["auth_params"] == {
        "aws_access_key_id": "access",
        "aws_secret_access_key": "secret",
        "aws_session_token": "token",
        "aws_region_name": "us-east-1",
    }
    logged = logging_obj.update_from_kwargs.call_args.kwargs
    assert logged["kwargs"] == {"tool_name": "target___WebSearch"}
    assert logged["optional_params"] == {"tool_name": "target___WebSearch"}


def test_search_handler_only_exposes_auth_params_to_signing():
    class CapturingConfig(BaseSearchConfig):
        def __init__(self):
            self.transformed_params: dict[str, object] | None = None
            self.signing_params: dict[str, object] | None = None

        def validate_environment(self, **kwargs):
            return {}

        def transform_search_request(self, query, optional_params, **kwargs):
            self.transformed_params = optional_params
            return {}

        def get_complete_url(self, **kwargs):
            return "https://gateway.example/mcp"

        def sign_request(self, *, optional_params, **kwargs):
            self.signing_params = optional_params
            raise RuntimeError("stop before network")

    config = CapturingConfig()
    with pytest.raises(RuntimeError, match="stop before network"):
        BaseLLMHTTPHandler().search(
            query="test",
            optional_params={"tool_name": "target___WebSearch"},
            auth_params={"aws_secret_access_key": "secret"},
            timeout=1,
            logging_obj=MagicMock(),
            api_key=None,
            api_base="https://gateway.example/mcp",
            custom_llm_provider="agentcore",
            provider_config=config,
        )

    assert config.transformed_params == {"tool_name": "target___WebSearch"}
    assert config.signing_params == {
        "tool_name": "target___WebSearch",
        "aws_secret_access_key": "secret",
    }
