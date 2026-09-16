"""Gateway coverage for Parallel AI Search."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

import litellm
from litellm import Router
from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import LlmProviders

PARALLEL_SEARCH_URL: Final = "https://api.parallel.ai/v1/search"


@pytest.fixture
def client() -> TestClient:
    return TestClient(proxy_server.app, raise_server_exceptions=False)


@pytest.fixture
def auth_as() -> Iterator[None]:
    async def _authorized_request() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key="hashed-sk-test",
            user_id="parallel-test-user",
        )

    previous: Final = proxy_server.app.dependency_overrides.get(user_api_key_auth)
    proxy_server.app.dependency_overrides[user_api_key_auth] = _authorized_request
    try:
        yield
    finally:
        if previous is None:
            proxy_server.app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            proxy_server.app.dependency_overrides[user_api_key_auth] = previous


def _parallel_search_body() -> dict[str, object]:
    return {
        "search_id": "search_parallel_gateway",
        "results": [
            {
                "url": "https://example.com/parallel",
                "title": "Parallel result",
                "publish_date": "2026-08-13",
                "excerpts": ["First excerpt", "Second excerpt"],
            }
        ],
        "usage": [{"name": "sku_search", "count": 1}],
    }


def _parallel_router(mode: str = "turbo") -> Router:
    return Router(
        model_list=[],
        search_tools=[
            {
                "search_tool_name": "parallel-search",
                "litellm_params": {
                    "search_provider": "parallel_ai",
                    "api_key": "parallel-search-key",
                    "mode": mode,
                },
            }
        ],
        num_retries=0,
    )


def _mock_async_post(
    monkeypatch,
    *,
    url: str,
    response_body: dict[str, object],
) -> AsyncMock:
    response = httpx.Response(
        status_code=200,
        json=response_body,
        request=httpx.Request("POST", url),
    )
    mock_post = AsyncMock(return_value=response)
    monkeypatch.setattr(AsyncHTTPHandler, "post", mock_post)
    return mock_post


def test_parallel_search_gateway_route(client, auth_as, monkeypatch):
    """The named search route selects its configured Parallel Search tool.

    The tool-level `mode` must survive the router hop, so the upstream request
    is sent as `turbo` rather than falling back to the adapter default.
    """
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(proxy_server, "llm_router", _parallel_router())
    mock_post = _mock_async_post(
        monkeypatch,
        url=PARALLEL_SEARCH_URL,
        response_body=_parallel_search_body(),
    )

    response = client.post(
        "/v1/search/parallel-search",
        json={"query": "Parallel AI news", "max_results": 3},
    )

    assert response.status_code == 200, response.text
    assert response.json()["results"] == [
        {
            "title": "Parallel result",
            "url": "https://example.com/parallel",
            "snippet": "First excerpt ... Second excerpt",
            "date": "2026-08-13",
            "last_updated": None,
            "excerpts": ["First excerpt", "Second excerpt"],
        }
    ]

    request_kwargs = mock_post.await_args.kwargs
    assert request_kwargs["url"] == PARALLEL_SEARCH_URL
    assert request_kwargs["headers"]["x-api-key"] == "parallel-search-key"
    assert request_kwargs["json"] == {
        "objective": "Parallel AI news",
        "search_queries": ["Parallel AI news"],
        "mode": "turbo",
        "advanced_settings": {"max_results": 3},
    }


@pytest.mark.asyncio
async def test_web_search_interception_executes_parallel_search(monkeypatch):
    """An intercepted web-search call uses the configured Parallel Search tool."""
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(proxy_server, "llm_router", _parallel_router(mode="fast"))
    mock_post = _mock_async_post(
        monkeypatch,
        url=PARALLEL_SEARCH_URL,
        response_body=_parallel_search_body(),
    )
    logger = WebSearchInterceptionLogger(
        enabled_providers=[LlmProviders.OPENAI],
        search_tool_name="parallel-search",
    )

    plan = await logger.async_build_responses_agentic_loop_plan(
        tools={
            "tool_calls": [
                {
                    "id": "fc_parallel",
                    "call_id": "fc_parallel",
                    "type": "function_call",
                    "name": "litellm_web_search",
                    "arguments": '{"query":"Parallel AI news"}',
                    "input": {"query": "Parallel AI news"},
                }
            ]
        },
        model="gpt-5",
        messages=[{"role": "user", "content": "Research Parallel"}],
        response=None,
        optional_params={"tools": [{"type": "function", "name": "litellm_web_search"}]},
        logging_obj=None,
        stream=False,
        kwargs={"custom_llm_provider": "openai"},
    )

    assert plan.run_agentic_loop is True
    assert plan.request_patch is not None
    assert plan.request_patch.messages[-1] == {
        "type": "function_call_output",
        "call_id": "fc_parallel",
        "output": (
            "Title: Parallel result\nURL: https://example.com/parallel\nSnippet: First excerpt ... Second excerpt"
        ),
    }

    request_kwargs = mock_post.await_args.kwargs
    assert request_kwargs["url"] == PARALLEL_SEARCH_URL
    assert request_kwargs["headers"]["x-api-key"] == "parallel-search-key"
    assert request_kwargs["json"]["mode"] == "fast"
