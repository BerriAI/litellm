"""
Unit tests for the rich web-search input shape (objective + search_queries).

The intercepted web search tool exposes optional `objective` and
`search_queries` fields alongside the required single `query` string. The
handler forwards the richer shape only to search providers whose config
reports supports_rich_search_input(); every other provider keeps receiving
the single query string the model also provided.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.websearch_interception.handler import (
    WebSearchInterceptionLogger,
)
from litellm.integrations.websearch_interception.tools import (
    get_litellm_web_search_tool,
    get_litellm_web_search_tool_openai,
    get_litellm_web_search_tool_responses,
)
from litellm.llms.base_llm.search.transformation import BaseSearchConfig, SearchResponse
from litellm.llms.parallel_ai.search.transformation import ParallelAISearchConfig

RICH_INPUT = {
    "query": "stripe node sdk v14 authentication",
    "objective": "Find the current authentication flow for the Stripe Node SDK v14",
    "search_queries": ["stripe node sdk v14 auth", "stripe api key rotation node"],
}


def _search_response() -> SearchResponse:
    return SearchResponse(object="search", results=[])


def _mock_router(search_provider: str) -> MagicMock:
    """Router stub exposing one configured search tool."""
    router = MagicMock()
    router.search_tools = [
        {
            "search_tool_name": "test-search",
            "litellm_params": {
                "search_provider": search_provider,
                "api_key": "sk-test",
            },
        }
    ]
    return router


class TestToolSchema:
    def test_all_formats_expose_rich_fields_and_keep_query_required(self):
        anthropic_schema = get_litellm_web_search_tool()["input_schema"]
        openai_schema = get_litellm_web_search_tool_openai()["function"]["parameters"]
        responses_schema = get_litellm_web_search_tool_responses()["parameters"]

        for schema in (anthropic_schema, openai_schema, responses_schema):
            assert schema["required"] == ["query"]
            assert "objective" in schema["properties"]
            assert "search_queries" in schema["properties"]
            assert schema["properties"]["search_queries"]["type"] == "array"


class TestRichInputExtraction:
    def test_extracts_objective_and_queries(self):
        rich = WebSearchInterceptionLogger._rich_search_input(RICH_INPUT)
        assert rich == {
            "objective": RICH_INPUT["objective"],
            "search_queries": RICH_INPUT["search_queries"],
        }

    def test_returns_none_when_only_query_present(self):
        assert WebSearchInterceptionLogger._rich_search_input({"query": "plain"}) is None

    def test_returns_none_for_non_mapping_input(self):
        assert WebSearchInterceptionLogger._rich_search_input(None) is None
        assert WebSearchInterceptionLogger._rich_search_input("query") is None

    def test_drops_invalid_queries_and_caps_at_five(self):
        rich = WebSearchInterceptionLogger._rich_search_input(
            {
                "query": "q",
                "search_queries": ["a", "", 3, "b", "c", "d", "e", "f"],
            }
        )
        assert rich == {"search_queries": ["a", "b", "c", "d", "e"]}

    def test_ignores_string_valued_search_queries(self):
        # A string is a Sequence; it must not be treated as a list of queries.
        assert WebSearchInterceptionLogger._rich_search_input({"query": "q", "search_queries": "not a list"}) is None


class TestProviderSupport:
    def test_parallel_ai_supports_rich_input(self):
        assert ParallelAISearchConfig().supports_rich_search_input() is True

    def test_base_config_defaults_to_unsupported(self):
        assert BaseSearchConfig().supports_rich_search_input() is False

    def test_unknown_provider_is_unsupported(self):
        assert WebSearchInterceptionLogger._provider_supports_rich_search(None) is False
        assert WebSearchInterceptionLogger._provider_supports_rich_search("not_a_provider") is False


class TestExecuteSearchShape:
    @pytest.mark.asyncio
    async def test_rich_shape_reaches_supporting_provider(self, monkeypatch):
        """Parallel AI receives the query list plus objective."""
        import litellm
        from litellm.proxy import proxy_server

        logger = WebSearchInterceptionLogger()
        mock_asearch = AsyncMock(return_value=_search_response())
        monkeypatch.setattr(proxy_server, "llm_router", _mock_router("parallel_ai"))
        monkeypatch.setattr(litellm, "asearch", mock_asearch)

        rich = WebSearchInterceptionLogger._rich_search_input(RICH_INPUT)
        await logger._execute_search(RICH_INPUT["query"], rich=rich)

        call_kwargs = mock_asearch.await_args.kwargs
        assert call_kwargs["query"] == RICH_INPUT["search_queries"]
        assert call_kwargs["objective"] == RICH_INPUT["objective"]
        assert call_kwargs["search_provider"] == "parallel_ai"

    @pytest.mark.asyncio
    async def test_string_only_provider_keeps_single_query(self, monkeypatch):
        """A provider without rich support receives the plain query string."""
        import litellm
        from litellm.proxy import proxy_server

        logger = WebSearchInterceptionLogger()
        mock_asearch = AsyncMock(return_value=_search_response())
        monkeypatch.setattr(proxy_server, "llm_router", _mock_router("perplexity"))
        monkeypatch.setattr(litellm, "asearch", mock_asearch)

        rich = WebSearchInterceptionLogger._rich_search_input(RICH_INPUT)
        await logger._execute_search(RICH_INPUT["query"], rich=rich)

        call_kwargs = mock_asearch.await_args.kwargs
        assert call_kwargs["query"] == RICH_INPUT["query"]
        assert "objective" not in call_kwargs

    @pytest.mark.asyncio
    async def test_single_string_callers_unchanged(self, monkeypatch):
        """No rich input: behavior is identical to before for any provider."""
        import litellm
        from litellm.proxy import proxy_server

        logger = WebSearchInterceptionLogger()
        mock_asearch = AsyncMock(return_value=_search_response())
        monkeypatch.setattr(proxy_server, "llm_router", _mock_router("parallel_ai"))
        monkeypatch.setattr(litellm, "asearch", mock_asearch)

        await logger._execute_search("plain query")

        call_kwargs = mock_asearch.await_args.kwargs
        assert call_kwargs["query"] == "plain query"
        assert "objective" not in call_kwargs

    @pytest.mark.asyncio
    async def test_configured_objective_not_overwritten(self, monkeypatch):
        """An objective set on the search tool's litellm_params wins over the model's."""
        import litellm
        from litellm.proxy import proxy_server

        logger = WebSearchInterceptionLogger()
        router = _mock_router("parallel_ai")
        router.search_tools[0]["litellm_params"]["objective"] = "configured objective"
        mock_asearch = AsyncMock(return_value=_search_response())
        monkeypatch.setattr(proxy_server, "llm_router", router)
        monkeypatch.setattr(litellm, "asearch", mock_asearch)

        rich = WebSearchInterceptionLogger._rich_search_input(RICH_INPUT)
        await logger._execute_search(RICH_INPUT["query"], rich=rich)

        call_kwargs = mock_asearch.await_args.kwargs
        assert call_kwargs["objective"] == "configured objective"


class TestCallSiteWiring:
    """Drive the patch builders end to end so regressions in the tool-call ->
    _rich_search_input wiring are caught, not just _execute_search itself."""

    @pytest.mark.asyncio
    async def test_anthropic_tool_call_forwards_rich_shape(self, monkeypatch):
        import litellm
        from litellm.proxy import proxy_server

        logger = WebSearchInterceptionLogger()
        mock_asearch = AsyncMock(return_value=_search_response())
        monkeypatch.setattr(proxy_server, "llm_router", _mock_router("parallel_ai"))
        monkeypatch.setattr(litellm, "asearch", mock_asearch)

        tool_calls = [{"id": "toolu_1", "name": "litellm_web_search", "input": dict(RICH_INPUT)}]
        await logger._build_anthropic_request_patch(
            model="claude",
            messages=[{"role": "user", "content": "hi"}],
            tool_calls=tool_calls,
            thinking_blocks=[],
            anthropic_messages_optional_request_params={},
            logging_obj=None,
            kwargs={},
        )

        call_kwargs = mock_asearch.await_args.kwargs
        assert call_kwargs["query"] == RICH_INPUT["search_queries"]
        assert call_kwargs["objective"] == RICH_INPUT["objective"]

    @pytest.mark.asyncio
    async def test_chat_completion_tool_call_forwards_rich_shape(self, monkeypatch):
        import json

        import litellm
        from litellm.proxy import proxy_server

        logger = WebSearchInterceptionLogger()
        mock_asearch = AsyncMock(return_value=_search_response())
        monkeypatch.setattr(proxy_server, "llm_router", _mock_router("parallel_ai"))
        monkeypatch.setattr(litellm, "asearch", mock_asearch)

        # The normalized shape transform_request produces for OpenAI responses:
        # function.arguments (raw) plus top-level name/input (parsed).
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "name": "litellm_web_search",
                "function": {
                    "name": "litellm_web_search",
                    "arguments": json.dumps(RICH_INPUT),
                },
                "input": dict(RICH_INPUT),
            }
        ]
        await logger._build_chat_completion_request_patch(
            model="claude",
            messages=[{"role": "user", "content": "hi"}],
            tool_calls=tool_calls,
            optional_params={},
            kwargs={},
        )

        call_kwargs = mock_asearch.await_args.kwargs
        assert call_kwargs["query"] == RICH_INPUT["search_queries"]
        assert call_kwargs["objective"] == RICH_INPUT["objective"]
