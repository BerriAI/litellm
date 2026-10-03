"""
Tests for domain-limit passthrough in web search interception.

Covers bug #44188: ``allowed_domains`` / ``blocked_domains`` set on an
Anthropic-native ``web_search_*`` tool must survive the conversion to the
standard LiteLLM web search tool and be applied to the downstream
``litellm.asearch()`` call as ``search_domain_filter``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from litellm.integrations.websearch_interception.handler import (
    WEBSEARCH_DOMAIN_FILTER_KEY,
    WebSearchInterceptionLogger,
    _extract_web_search_domain_filters,
)
from litellm.llms.base_llm.search.transformation import SearchResponse, SearchResult


def _make_search_response() -> SearchResponse:
    return SearchResponse(
        results=[
            SearchResult(
                title="LiteLLM Docs",
                url="https://docs.litellm.ai/",
                snippet="Unified interface for LLMs.",
                date=None,
            )
        ]
    )


class TestExtractWebSearchDomainFilters:
    def test_collects_allowed_and_blocked(self):
        tools = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["docs.litellm.ai"],
                "blocked_domains": ["twitter.com", "x.com"],
            }
        ]
        assert _extract_web_search_domain_filters(tools) == {
            "allowed_domains": ["docs.litellm.ai"],
            "blocked_domains": ["twitter.com", "x.com"],
        }

    def test_returns_none_without_domain_limits(self):
        tools = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
        assert _extract_web_search_domain_filters(tools) is None

    def test_ignores_domains_on_non_web_search_tools(self):
        tools = [
            {"name": "bash", "allowed_domains": ["example.com"]},
        ]
        assert _extract_web_search_domain_filters(tools) is None

    def test_ignores_non_string_entries(self):
        tools = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "allowed_domains": ["docs.litellm.ai", 42, None],
            }
        ]
        assert _extract_web_search_domain_filters(tools) == {"allowed_domains": ["docs.litellm.ai"]}


class TestDeploymentHookStashesDomainFilters:
    @pytest.mark.asyncio
    async def test_stashes_filters_for_native_tool(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs = {
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "allowed_domains": ["docs.litellm.ai"],
                    "blocked_domains": ["twitter.com"],
                }
            ],
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }
        out = await logger.async_pre_call_deployment_hook(kwargs, None)
        assert out is not None
        assert kwargs[WEBSEARCH_DOMAIN_FILTER_KEY] == {
            "allowed_domains": ["docs.litellm.ai"],
            "blocked_domains": ["twitter.com"],
        }

    @pytest.mark.asyncio
    async def test_no_stash_without_domain_limits(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs = {
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
            "litellm_params": {"custom_llm_provider": "bedrock"},
        }
        await logger.async_pre_call_deployment_hook(kwargs, None)
        assert WEBSEARCH_DOMAIN_FILTER_KEY not in kwargs


class TestExecuteSearchAppliesDomainFilter:
    @pytest.mark.asyncio
    async def test_forwards_search_domain_filter(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        kwargs = {
            WEBSEARCH_DOMAIN_FILTER_KEY: {
                "allowed_domains": ["docs.litellm.ai"],
                "blocked_domains": ["twitter.com"],
            }
        }
        asearch = AsyncMock(return_value=_make_search_response())
        with patch("litellm.asearch", asearch):
            await logger._execute_search("what is litellm", kwargs=kwargs)

        assert asearch.await_count == 1
        assert asearch.await_args.kwargs.get("search_domain_filter") == [
            "docs.litellm.ai",
            "-twitter.com",
        ]

    @pytest.mark.asyncio
    async def test_no_filter_when_kwargs_empty(self):
        logger = WebSearchInterceptionLogger(enabled_providers=["bedrock"])
        asearch = AsyncMock(return_value=_make_search_response())
        with patch("litellm.asearch", asearch):
            await logger._execute_search("what is litellm", kwargs={})

        assert asearch.await_count == 1
        assert asearch.await_args.kwargs.get("search_domain_filter") is None
