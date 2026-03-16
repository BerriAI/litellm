"""
Unit tests for FileSearchResponsesAPIUtils — file_search tool interception in the
Responses API for non-native providers.
"""

import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.responses.file_search_utils import (
    CONTEXT_PREFIX,
    FileSearchResponsesAPIUtils,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_file_search_tool(vector_store_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    tool: Dict[str, Any] = {"type": "file_search"}
    if vector_store_ids is not None:
        tool["vector_store_ids"] = vector_store_ids
    return tool


def _make_function_tool(name: str = "my_func") -> Dict[str, Any]:
    return {"type": "function", "name": name}


def _make_search_response(texts: List[str]) -> Dict[str, Any]:
    return {
        "data": [
            {
                "id": f"result_{i}",
                "score": 0.9,
                "content": [{"type": "text", "text": text}],
            }
            for i, text in enumerate(texts)
        ]
    }


# ---------------------------------------------------------------------------
# has_file_search_tools
# ---------------------------------------------------------------------------

class TestHasFileSearchTools:
    def test_detects_file_search_tool(self):
        tools = [_make_file_search_tool(["vs_abc"])]
        assert FileSearchResponsesAPIUtils.has_file_search_tools(tools) is True

    def test_ignores_non_file_search_tool(self):
        tools = [_make_function_tool()]
        assert FileSearchResponsesAPIUtils.has_file_search_tools(tools) is False

    def test_mixed_tools_detected(self):
        tools = [_make_function_tool(), _make_file_search_tool(["vs_abc"])]
        assert FileSearchResponsesAPIUtils.has_file_search_tools(tools) is True

    def test_empty_list(self):
        assert FileSearchResponsesAPIUtils.has_file_search_tools([]) is False


# ---------------------------------------------------------------------------
# split_file_search_tools
# ---------------------------------------------------------------------------

class TestSplitFileSearchTools:
    def test_splits_correctly(self):
        fs = _make_file_search_tool(["vs_1"])
        fn = _make_function_tool()
        file_search_tools, other_tools = FileSearchResponsesAPIUtils.split_file_search_tools([fs, fn])
        assert file_search_tools == [fs]
        assert other_tools == [fn]

    def test_all_file_search(self):
        tools = [_make_file_search_tool(["vs_1"]), _make_file_search_tool(["vs_2"])]
        fs, others = FileSearchResponsesAPIUtils.split_file_search_tools(tools)
        assert len(fs) == 2
        assert others == []

    def test_no_file_search(self):
        tools = [_make_function_tool("a"), _make_function_tool("b")]
        fs, others = FileSearchResponsesAPIUtils.split_file_search_tools(tools)
        assert fs == []
        assert others == tools

    def test_empty(self):
        fs, others = FileSearchResponsesAPIUtils.split_file_search_tools([])
        assert fs == []
        assert others == []


# ---------------------------------------------------------------------------
# extract_query_from_responses_input
# ---------------------------------------------------------------------------

class TestExtractQueryFromResponsesInput:
    def test_string_input(self):
        assert (
            FileSearchResponsesAPIUtils.extract_query_from_responses_input("what is rag?")
            == "what is rag?"
        )

    def test_list_input_last_user_message(self):
        input_items = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "what is rag?"},
        ]
        assert (
            FileSearchResponsesAPIUtils.extract_query_from_responses_input(input_items)
            == "what is rag?"
        )

    def test_list_input_structured_content(self):
        input_items = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "explain vector search"}],
            }
        ]
        assert (
            FileSearchResponsesAPIUtils.extract_query_from_responses_input(input_items)
            == "explain vector search"
        )

    def test_empty_list_returns_none(self):
        assert (
            FileSearchResponsesAPIUtils.extract_query_from_responses_input([]) is None
        )

    def test_no_user_message_returns_none(self):
        input_items = [{"role": "assistant", "content": "hi"}]
        assert (
            FileSearchResponsesAPIUtils.extract_query_from_responses_input(input_items)
            is None
        )

    def test_non_list_non_str_returns_none(self):
        assert (
            FileSearchResponsesAPIUtils.extract_query_from_responses_input(None) is None  # type: ignore
        )


# ---------------------------------------------------------------------------
# inject_context_into_responses_input
# ---------------------------------------------------------------------------

class TestInjectContextIntoResponsesInput:
    def test_string_input_becomes_list(self):
        result = FileSearchResponsesAPIUtils.inject_context_into_responses_input(
            "user question", "some context"
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["content"] == "some context"
        assert result[1]["content"] == "user question"

    def test_list_input_context_before_last(self):
        input_items = [
            {"role": "user", "content": "first message"},
            {"role": "user", "content": "last message"},
        ]
        result = FileSearchResponsesAPIUtils.inject_context_into_responses_input(
            input_items, "ctx"
        )
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[1]["content"] == "ctx"
        assert result[2]["content"] == "last message"

    def test_single_item_list_context_prepended(self):
        input_items = [{"role": "user", "content": "question"}]
        result = FileSearchResponsesAPIUtils.inject_context_into_responses_input(
            input_items, "ctx"
        )
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["content"] == "ctx"
        assert result[1]["content"] == "question"

    def test_empty_list_context_appended(self):
        result = FileSearchResponsesAPIUtils.inject_context_into_responses_input([], "ctx")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["content"] == "ctx"

    def test_original_input_not_mutated(self):
        original = [{"role": "user", "content": "q"}]
        FileSearchResponsesAPIUtils.inject_context_into_responses_input(original, "ctx")
        assert len(original) == 1


# ---------------------------------------------------------------------------
# asearch_and_inject_context — unit / integration (mocked)
# ---------------------------------------------------------------------------

class TestAsearchAndInjectContext:
    @pytest.mark.asyncio
    async def test_searches_and_injects_context(self):
        search_response = _make_search_response(["result text one", "result text two"])

        with patch(
            "litellm.vector_stores.asearch",
            new=AsyncMock(return_value=search_response),
        ):
            modified_input, remaining_tools = (
                await FileSearchResponsesAPIUtils.asearch_and_inject_context(
                    input="what is rag?",
                    tools=[_make_file_search_tool(["vs_abc"])],
                )
            )

        assert isinstance(modified_input, list)
        assert remaining_tools == []
        context_item = modified_input[0]
        assert context_item["role"] == "user"
        assert "result text one" in context_item["content"]
        assert "result text two" in context_item["content"]

    @pytest.mark.asyncio
    async def test_file_search_tool_stripped_from_tools(self):
        search_response = _make_search_response(["some context"])
        fn_tool = _make_function_tool("other_fn")

        with patch(
            "litellm.vector_stores.asearch",
            new=AsyncMock(return_value=search_response),
        ):
            _, remaining_tools = await FileSearchResponsesAPIUtils.asearch_and_inject_context(
                input="query",
                tools=[_make_file_search_tool(["vs_1"]), fn_tool],
            )

        assert fn_tool in remaining_tools
        assert not any(
            t.get("type") == "file_search" for t in remaining_tools
        )

    @pytest.mark.asyncio
    async def test_multiple_vector_stores_searched(self):
        search_response = _make_search_response(["doc"])

        mock_search = AsyncMock(return_value=search_response)
        with patch("litellm.vector_stores.asearch", new=mock_search):
            await FileSearchResponsesAPIUtils.asearch_and_inject_context(
                input="query",
                tools=[_make_file_search_tool(["vs_1", "vs_2"])],
            )

        assert mock_search.call_count == 2
        called_ids = {call.kwargs["vector_store_id"] for call in mock_search.call_args_list}
        assert called_ids == {"vs_1", "vs_2"}

    @pytest.mark.asyncio
    async def test_no_query_returns_unchanged(self):
        original_input: List[Dict[str, Any]] = []
        tools = [_make_file_search_tool(["vs_1"])]

        modified_input, remaining_tools = (
            await FileSearchResponsesAPIUtils.asearch_and_inject_context(
                input=original_input,
                tools=tools,
            )
        )

        assert modified_input == original_input
        # file_search tools stripped even when no query
        assert not any(t.get("type") == "file_search" for t in remaining_tools)

    @pytest.mark.asyncio
    async def test_no_vector_store_ids_strips_tool_only(self):
        tools = [_make_file_search_tool()]  # no vector_store_ids key

        modified_input, remaining_tools = (
            await FileSearchResponsesAPIUtils.asearch_and_inject_context(
                input="query",
                tools=tools,
            )
        )

        assert modified_input == "query"
        assert remaining_tools == []

    @pytest.mark.asyncio
    async def test_search_error_does_not_raise(self):
        with patch(
            "litellm.vector_stores.asearch",
            new=AsyncMock(side_effect=Exception("timeout")),
        ):
            modified_input, remaining_tools = (
                await FileSearchResponsesAPIUtils.asearch_and_inject_context(
                    input="query",
                    tools=[_make_file_search_tool(["vs_err"])],
                )
            )

        # Falls back gracefully: input unchanged, file_search stripped
        assert modified_input == "query"
        assert remaining_tools == []

    @pytest.mark.asyncio
    async def test_stores_results_in_logging_obj(self):
        search_response = _make_search_response(["relevant doc"])
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}

        with patch(
            "litellm.vector_stores.asearch",
            new=AsyncMock(return_value=search_response),
        ):
            await FileSearchResponsesAPIUtils.asearch_and_inject_context(
                input="query",
                tools=[_make_file_search_tool(["vs_abc"])],
                litellm_logging_obj=logging_obj,
            )

        assert "search_results" in logging_obj.model_call_details
        assert logging_obj.model_call_details["search_results"] == [search_response]


# ---------------------------------------------------------------------------
# supports_native_file_search — capability flag tests
# ---------------------------------------------------------------------------

class TestSupportsNativeFileSearch:
    def test_base_config_returns_false(self):
        """All non-overriding providers default to False."""

        class _MinimalConfig(BaseResponsesAPIConfig):
            @property
            def custom_llm_provider(self):
                return "test_provider"

            def get_supported_openai_params(self, model):
                return []

            def map_openai_params(self, response_api_optional_params, model, drop_params):
                return {}

            def validate_environment(self, headers, model, litellm_params):
                return {}

            def get_complete_url(self, api_base, litellm_params):
                return api_base or ""

            def transform_responses_api_request(self, model, input, response_api_optional_request_params, litellm_params, headers):
                return {}

            def transform_response_api_response(self, model, raw_response, logging_obj):
                return MagicMock()

            def transform_streaming_response(self, model, parsed_chunk, logging_obj):
                return MagicMock()

            def transform_delete_response_api_request(self, response_id, api_base, litellm_params, headers):
                return "", {}

            def transform_delete_response_api_response(self, raw_response, logging_obj):
                return MagicMock()

            def transform_get_response_api_request(self, response_id, api_base, litellm_params, headers):
                return "", {}

            def transform_get_response_api_response(self, raw_response, logging_obj):
                return MagicMock()

            def transform_list_input_items_request(self, response_id, api_base, litellm_params, headers, after=None, before=None, include=None, limit=20, order="desc"):
                return "", {}

            def transform_list_input_items_response(self, raw_response, logging_obj):
                return {}

            def transform_cancel_response_api_request(self, response_id, api_base, litellm_params, headers):
                return "", {}

            def transform_cancel_response_api_response(self, raw_response, logging_obj):
                return MagicMock()

            def transform_compact_response_api_request(self, model, input, response_api_optional_request_params, api_base, litellm_params, headers):
                return "", {}

            def transform_compact_response_api_response(self, raw_response, logging_obj):
                return MagicMock()

        assert _MinimalConfig().supports_native_file_search() is False

    def test_openai_config_returns_true(self):
        assert OpenAIResponsesAPIConfig().supports_native_file_search() is True

    def test_azure_config_inherits_true(self):
        from litellm.llms.azure.responses.transformation import (
            AzureOpenAIResponsesAPIConfig,
        )

        assert AzureOpenAIResponsesAPIConfig().supports_native_file_search() is True
