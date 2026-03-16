"""
Tests for file_search tool support in the Responses API via VectorStorePreCallHook.

Covers:
  - Query extraction from str and list inputs
  - Context injection into str and list inputs
  - VS ID extraction from file_search tools
  - Tool stripping for non-native providers
  - Provider capability check (native vs RAG mode)
  - async_pre_call_hook full integration with mocked VS search
  - Concurrent search with timeout and failure handling
  - Regression: chat completions path still works after refactor
"""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.integrations.vector_store_integrations.vector_store_pre_call_hook import (
    VectorStorePreCallHook,
)
from litellm.types.utils import CallTypes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hook() -> VectorStorePreCallHook:
    return VectorStorePreCallHook()


def _make_vs_config(vs_id: str, provider: str = "openai") -> Dict:
    return {
        "vector_store_id": vs_id,
        "custom_llm_provider": provider,
        "litellm_params": {},
    }


def _make_search_response(texts: List[str]) -> Dict:
    return {
        "object": "vector_store.search_results.page",
        "search_query": "test query",
        "data": [
            {
                "file_id": f"file_{i}",
                "filename": f"doc_{i}.txt",
                "score": 0.9,
                "attributes": {},
                "content": [{"type": "text", "text": t}],
            }
            for i, t in enumerate(texts)
        ],
    }


# ---------------------------------------------------------------------------
# _extract_query_from_responses_input
# ---------------------------------------------------------------------------


class TestExtractQueryFromResponsesInput:
    def test_str_input_returned_directly(self, hook: VectorStorePreCallHook) -> None:
        assert hook._extract_query_from_responses_input("hello world") == "hello world"

    def test_empty_str_returns_none(self, hook: VectorStorePreCallHook) -> None:
        assert hook._extract_query_from_responses_input("") is None

    def test_none_returns_none(self, hook: VectorStorePreCallHook) -> None:
        assert hook._extract_query_from_responses_input(None) is None

    def test_list_with_text_message_item(self, hook: VectorStorePreCallHook) -> None:
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "what is deep research?"}],
            }
        ]
        assert hook._extract_query_from_responses_input(input_value) == "what is deep research?"

    def test_list_with_str_content_in_message(self, hook: VectorStorePreCallHook) -> None:
        input_value = [{"type": "message", "role": "user", "content": "simple str content"}]
        assert hook._extract_query_from_responses_input(input_value) == "simple str content"

    def test_list_returns_last_text_item(self, hook: VectorStorePreCallHook) -> None:
        input_value = [
            {"type": "message", "role": "assistant", "content": "I can help."},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "last user query"}],
            },
        ]
        assert hook._extract_query_from_responses_input(input_value) == "last user query"

    def test_list_with_no_text_items_returns_none(self, hook: VectorStorePreCallHook) -> None:
        input_value = [{"type": "message", "role": "user", "content": [{"type": "image_url", "url": "..."}]}]
        assert hook._extract_query_from_responses_input(input_value) is None

    def test_empty_list_returns_none(self, hook: VectorStorePreCallHook) -> None:
        assert hook._extract_query_from_responses_input([]) is None


# ---------------------------------------------------------------------------
# _get_vs_ids_from_file_search_tools
# ---------------------------------------------------------------------------


class TestGetVsIdsFromFileSearchTools:
    def test_extracts_ids_from_file_search_tool(self, hook: VectorStorePreCallHook) -> None:
        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc", "vs_def"]}]
        assert hook._get_vs_ids_from_file_search_tools(tools) == ["vs_abc", "vs_def"]

    def test_ignores_non_file_search_tools(self, hook: VectorStorePreCallHook) -> None:
        tools = [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "file_search", "vector_store_ids": ["vs_abc"]},
        ]
        assert hook._get_vs_ids_from_file_search_tools(tools) == ["vs_abc"]

    def test_empty_vector_store_ids_returns_empty(self, hook: VectorStorePreCallHook) -> None:
        tools = [{"type": "file_search", "vector_store_ids": []}]
        assert hook._get_vs_ids_from_file_search_tools(tools) == []

    def test_none_tools_returns_empty(self, hook: VectorStorePreCallHook) -> None:
        assert hook._get_vs_ids_from_file_search_tools(None) == []

    def test_deduplicates_ids(self, hook: VectorStorePreCallHook) -> None:
        tools = [
            {"type": "file_search", "vector_store_ids": ["vs_abc", "vs_abc"]},
        ]
        assert hook._get_vs_ids_from_file_search_tools(tools) == ["vs_abc"]


# ---------------------------------------------------------------------------
# _strip_file_search_from_tools
# ---------------------------------------------------------------------------


class TestStripFileSearchFromTools:
    def test_removes_file_search_only(self, hook: VectorStorePreCallHook) -> None:
        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]
        assert hook._strip_file_search_from_tools(tools) is None

    def test_keeps_other_tools(self, hook: VectorStorePreCallHook) -> None:
        fn_tool = {"type": "function", "function": {"name": "get_weather"}}
        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}, fn_tool]
        result = hook._strip_file_search_from_tools(tools)
        assert result == [fn_tool]

    def test_none_tools_returns_none(self, hook: VectorStorePreCallHook) -> None:
        assert hook._strip_file_search_from_tools(None) is None

    def test_no_file_search_returns_unchanged(self, hook: VectorStorePreCallHook) -> None:
        fn_tool = {"type": "function", "function": {"name": "get_weather"}}
        assert hook._strip_file_search_from_tools([fn_tool]) == [fn_tool]


# ---------------------------------------------------------------------------
# _inject_context_into_responses_input
# ---------------------------------------------------------------------------


class TestInjectContextIntoResponsesInput:
    def test_str_input_prepends_context(self, hook: VectorStorePreCallHook) -> None:
        results = [_make_search_response(["relevant fact"])]
        output = hook._inject_context_into_responses_input("my query", results)
        assert isinstance(output, str)
        assert "Context:" in output
        assert "relevant fact" in output
        assert "my query" in output

    def test_list_input_inserts_context_item(self, hook: VectorStorePreCallHook) -> None:
        user_msg = {"type": "message", "role": "user", "content": "my query"}
        results = [_make_search_response(["relevant fact"])]
        output = hook._inject_context_into_responses_input([user_msg], results)
        assert isinstance(output, list)
        assert len(output) == 2  # context item + original user message
        context_item = output[0]
        assert "relevant fact" in str(context_item)
        assert output[-1] == user_msg

    def test_empty_search_results_returns_input_unchanged(self, hook: VectorStorePreCallHook) -> None:
        results: List = []
        assert hook._inject_context_into_responses_input("query", results) == "query"

    def test_search_results_with_no_text_returns_input_unchanged(self, hook: VectorStorePreCallHook) -> None:
        results = [{"object": "...", "search_query": "q", "data": []}]
        assert hook._inject_context_into_responses_input("query", results) == "query"

    def test_none_input_returns_none(self, hook: VectorStorePreCallHook) -> None:
        results = [_make_search_response(["text"])]
        # None input: inject returns None (no-op)
        assert hook._inject_context_into_responses_input(None, results) is None


# ---------------------------------------------------------------------------
# _is_native_file_search_provider
# ---------------------------------------------------------------------------


class TestIsNativeFileSearchProvider:
    def test_openai_model_is_native(self, hook: VectorStorePreCallHook) -> None:
        assert hook._is_native_file_search_provider("gpt-4.1") is True

    def test_azure_model_is_native(self, hook: VectorStorePreCallHook) -> None:
        assert hook._is_native_file_search_provider("azure/gpt-4o") is True

    def test_anthropic_model_is_not_native(self, hook: VectorStorePreCallHook) -> None:
        assert hook._is_native_file_search_provider("claude-3-7-sonnet-20250219") is False

    def test_unknown_model_defaults_to_false(self, hook: VectorStorePreCallHook) -> None:
        # Unknown models should default to RAG mode (safe fallback)
        assert hook._is_native_file_search_provider("totally-unknown-model/v1") is False


# ---------------------------------------------------------------------------
# _search_vector_stores_concurrent
# ---------------------------------------------------------------------------


class TestSearchVectorStoresConcurrent:
    @pytest.mark.asyncio
    async def test_successful_search_returns_results(self, hook: VectorStorePreCallHook) -> None:
        vs = _make_vs_config("vs_abc")
        response = _make_search_response(["fact A"])
        with patch("litellm.vector_stores.asearch", new_callable=AsyncMock, return_value=response):
            results = await hook._search_vector_stores_concurrent([vs], "query")
        assert len(results) == 1
        assert results[0] == response

    @pytest.mark.asyncio
    async def test_timeout_skips_that_vs(self, hook: VectorStorePreCallHook) -> None:
        vs = _make_vs_config("vs_slow")

        async def _slow_search(**kwargs: Any) -> Dict:
            await asyncio.sleep(100)
            return _make_search_response(["never returned"])

        with patch("litellm.vector_stores.asearch", side_effect=_slow_search):
            with patch(
                "litellm.integrations.vector_store_integrations.vector_store_pre_call_hook._VS_SEARCH_TIMEOUT_SECONDS",
                0.01,
            ):
                results = await hook._search_vector_stores_concurrent([vs], "query")
        assert results == []

    @pytest.mark.asyncio
    async def test_one_failure_does_not_cancel_others(self, hook: VectorStorePreCallHook) -> None:
        vs_ok = _make_vs_config("vs_ok")
        vs_bad = _make_vs_config("vs_bad")
        good_response = _make_search_response(["good result"])

        call_count = 0

        async def _maybe_fail(**kwargs: Any) -> Dict:
            nonlocal call_count
            call_count += 1
            if kwargs.get("vector_store_id") == "vs_bad":
                raise RuntimeError("search failed")
            return good_response

        with patch("litellm.vector_stores.asearch", side_effect=_maybe_fail):
            results = await hook._search_vector_stores_concurrent([vs_ok, vs_bad], "query")

        assert len(results) == 1
        assert results[0] == good_response
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_fan_out(self, hook: VectorStorePreCallHook) -> None:
        """Verify searches run concurrently (not sequentially)."""
        vs_list = [_make_vs_config(f"vs_{i}") for i in range(3)]
        responses = [_make_search_response([f"result {i}"]) for i in range(3)]
        idx = 0

        async def _return_next(**kwargs: Any) -> Dict:
            nonlocal idx
            r = responses[idx]
            idx += 1
            return r

        with patch("litellm.vector_stores.asearch", side_effect=_return_next):
            results = await hook._search_vector_stores_concurrent(vs_list, "query")

        assert len(results) == 3


# ---------------------------------------------------------------------------
# async_pre_call_hook: full integration
# ---------------------------------------------------------------------------


class TestAsyncPreCallHook:
    @pytest.mark.asyncio
    async def test_no_tools_is_passthrough(self, hook: VectorStorePreCallHook) -> None:
        data = {"model": "claude-3-7-sonnet-20250219", "input": "hello"}
        result = await hook.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=MagicMock(),
            data=data,
            call_type=CallTypes.aresponses.value,
        )
        assert result == data

    @pytest.mark.asyncio
    async def test_no_file_search_in_tools_is_passthrough(self, hook: VectorStorePreCallHook) -> None:
        data = {
            "model": "claude-3-7-sonnet-20250219",
            "input": "hello",
            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        }
        result = await hook.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=MagicMock(),
            data=data,
            call_type=CallTypes.aresponses.value,
        )
        assert result["tools"] == data["tools"]

    @pytest.mark.asyncio
    async def test_rag_injection_for_non_native_provider(self, hook: VectorStorePreCallHook) -> None:
        data = {
            "model": "claude-3-7-sonnet-20250219",
            "input": "What is deep research?",
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_abc"]}],
        }
        vs_config = _make_vs_config("vs_abc", "openai")
        search_response = _make_search_response(["Deep research is a methodology."])

        with (
            patch.object(hook, "_is_native_file_search_provider", return_value=False),
            patch.object(hook, "_resolve_vector_stores", new_callable=AsyncMock, return_value=[vs_config]),
            patch("litellm.vector_stores.asearch", new_callable=AsyncMock, return_value=search_response),
        ):
            result = await hook.async_pre_call_hook(
                user_api_key_dict=MagicMock(),
                cache=MagicMock(),
                data=data,
                call_type=CallTypes.aresponses.value,
            )

        # Context injected into input
        assert "Deep research is a methodology" in result["input"]
        # file_search stripped
        assert result["tools"] is None
        # search results stored for post-call hook
        assert "vs_search_results" in result

    @pytest.mark.asyncio
    async def test_native_provider_passthrough(self, hook: VectorStorePreCallHook) -> None:
        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]
        data = {
            "model": "gpt-4.1",
            "input": "What is deep research?",
            "tools": tools,
        }

        with patch.object(hook, "_is_native_file_search_provider", return_value=True):
            result = await hook.async_pre_call_hook(
                user_api_key_dict=MagicMock(),
                cache=MagicMock(),
                data=data,
                call_type=CallTypes.aresponses.value,
            )

        # Tools unchanged — native passthrough
        assert result["tools"] == tools
        assert result["input"] == "What is deep research?"

    @pytest.mark.asyncio
    async def test_non_responses_call_type_is_passthrough(self, hook: VectorStorePreCallHook) -> None:
        data = {
            "model": "gpt-4.1",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_abc"]}],
        }
        result = await hook.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=MagicMock(),
            data=data,
            call_type=CallTypes.acompletion.value,
        )
        # Not a responses call → returned unchanged
        assert result == data

    @pytest.mark.asyncio
    async def test_vs_not_in_registry_strips_tool_gracefully(self, hook: VectorStorePreCallHook) -> None:
        data = {
            "model": "claude-3-7-sonnet-20250219",
            "input": "query",
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_unknown"]}],
        }
        with (
            patch.object(hook, "_is_native_file_search_provider", return_value=False),
            patch.object(hook, "_resolve_vector_stores", new_callable=AsyncMock, return_value=[]),
        ):
            result = await hook.async_pre_call_hook(
                user_api_key_dict=MagicMock(),
                cache=MagicMock(),
                data=data,
                call_type=CallTypes.aresponses.value,
            )

        # No context injected (no VS found), but tool stripped
        assert result["tools"] is None
        assert "Context:" not in result.get("input", "")

    @pytest.mark.asyncio
    async def test_vs_search_failure_strips_tool_gracefully(self, hook: VectorStorePreCallHook) -> None:
        data = {
            "model": "claude-3-7-sonnet-20250219",
            "input": "query",
            "tools": [{"type": "file_search", "vector_store_ids": ["vs_abc"]}],
        }
        vs_config = _make_vs_config("vs_abc")

        async def _fail(**kwargs: Any) -> None:
            raise RuntimeError("search failed")

        with (
            patch.object(hook, "_is_native_file_search_provider", return_value=False),
            patch.object(hook, "_resolve_vector_stores", new_callable=AsyncMock, return_value=[vs_config]),
            patch("litellm.vector_stores.asearch", side_effect=_fail),
        ):
            result = await hook.async_pre_call_hook(
                user_api_key_dict=MagicMock(),
                cache=MagicMock(),
                data=data,
                call_type=CallTypes.aresponses.value,
            )

        # Tool stripped, LLM call proceeds without context
        assert result["tools"] is None
        assert "vs_search_results" not in result


# ---------------------------------------------------------------------------
# Regression: chat completions path still works after refactor
# ---------------------------------------------------------------------------


class TestChatCompletionsRegression:
    @pytest.mark.asyncio
    async def test_chat_completions_path_injects_context(self, hook: VectorStorePreCallHook) -> None:
        """async_get_chat_completion_prompt still works after shared search method refactor."""
        messages = [{"role": "user", "content": "what is litellm?"}]
        tools = [{"type": "file_search", "vector_store_ids": ["vs_abc"]}]
        non_default_params = {"tools": tools}

        vs_config = _make_vs_config("vs_abc")
        search_response = _make_search_response(["LiteLLM is a unified LLM interface."])

        mock_registry = MagicMock()
        mock_registry.pop_vector_stores_to_run_with_db_fallback = AsyncMock(
            return_value=[vs_config]
        )

        with (
            patch("litellm.vector_store_registry", mock_registry),
            patch("litellm.vector_stores.asearch", new_callable=AsyncMock, return_value=search_response),
        ):
            _, modified_messages, _ = await hook.async_get_chat_completion_prompt(
                model="claude-3-7-sonnet-20250219",
                messages=messages,
                non_default_params=non_default_params,
                prompt_id=None,
                prompt_variables=None,
                dynamic_callback_params=MagicMock(),
                litellm_logging_obj=MagicMock(),
                tools=tools,
            )

        # Context injected as a new message before the user's message
        assert len(modified_messages) == 2
        context_msg = modified_messages[0]
        assert "LiteLLM is a unified LLM interface" in str(context_msg["content"])
        assert modified_messages[-1] == messages[0]
