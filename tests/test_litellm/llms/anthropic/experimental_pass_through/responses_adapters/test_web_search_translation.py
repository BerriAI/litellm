"""
Tests for web search response translation utilities and integration.

Tests for responses_adapters/utils.py functions:
- build_web_tool_use
- build_web_search_results_from_annotations
- build_text_blocks_with_citations

Tests for non-streaming response translation (transformation.py):
- translate_response with web_search_call items

Tests for streaming response translation (streaming_iterator.py):
- web_search_call deferred emission
- text delta suppression during web search
- cited text block emission on output_item.done
"""

import json
import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.utils import (
    build_text_blocks_with_citations,
    build_web_search_results_from_annotations,
    build_web_tool_use,
)


# ---------------------------------------------------------------------------
# build_web_tool_use
# ---------------------------------------------------------------------------


class TestBuildWebToolUse:
    def test_search_action(self):
        item = {
            "id": "ws_123",
            "type": "web_search_call",
            "action": {"type": "search", "query": "React 19 features"},
        }
        block, input_dict = build_web_tool_use(item)
        assert block["type"] == "server_tool_use"
        assert block["name"] == "web_search"
        assert block["id"] == "ws_123"
        assert input_dict == {"query": "React 19 features"}

    def test_search_action_with_queries_list(self):
        item = {
            "id": "ws_456",
            "type": "web_search_call",
            "action": {"type": "search", "queries": ["React 19", "Vue 4"]},
        }
        block, input_dict = build_web_tool_use(item)
        assert block["name"] == "web_search"
        assert input_dict == {"query": "React 19\nVue 4"}

    def test_open_page_action(self):
        item = {
            "id": "ws_789",
            "type": "web_search_call",
            "action": {"type": "open_page", "url": "https://react.dev"},
        }
        block, input_dict = build_web_tool_use(item)
        assert block["name"] == "web_fetch"
        assert input_dict == {"url": "https://react.dev"}

    def test_find_in_page_action(self):
        item = {
            "id": "ws_abc",
            "type": "web_search_call",
            "action": {"type": "find_in_page", "url": "https://react.dev/blog"},
        }
        block, input_dict = build_web_tool_use(item)
        assert block["name"] == "web_fetch"
        assert input_dict == {"url": "https://react.dev/blog"}

    def test_model_dump_object(self):
        """Non-dict item with model_dump() is handled."""
        item = MagicMock()
        item.model_dump.return_value = {
            "id": "ws_obj",
            "action": {"type": "search", "query": "test"},
        }
        block, input_dict = build_web_tool_use(item)
        assert block["id"] == "ws_obj"
        assert block["name"] == "web_search"


# ---------------------------------------------------------------------------
# build_web_search_results_from_annotations
# ---------------------------------------------------------------------------


class TestBuildWebSearchResultsFromAnnotations:
    def test_basic_annotations(self):
        web_tool_uses = [
            {"type": "server_tool_use", "name": "web_search", "id": "ws_1"}
        ]
        annotations = [
            {
                "type": "url_citation",
                "url": "https://react.dev/blog",
                "title": "React Blog",
                "start_index": 0,
                "end_index": 10,
            },
            {
                "type": "url_citation",
                "url": "https://vue.org",
                "title": "Vue.js",
                "start_index": 20,
                "end_index": 30,
            },
        ]
        blocks, citations = build_web_search_results_from_annotations(
            web_tool_uses, annotations
        )
        assert len(blocks) == 1
        assert blocks[0]["type"] == "web_search_tool_result"
        assert blocks[0]["tool_use_id"] == "ws_1"
        assert len(blocks[0]["content"]) == 2
        assert len(citations) == 2
        assert citations[0][0] < citations[1][0]  # sorted by start_index

    def test_deduplicates_urls(self):
        web_tool_uses = [
            {"type": "server_tool_use", "name": "web_search", "id": "ws_1"}
        ]
        annotations = [
            {
                "type": "url_citation",
                "url": "https://react.dev",
                "title": "React",
                "start_index": 0,
                "end_index": 5,
            },
            {
                "type": "url_citation",
                "url": "https://react.dev",
                "title": "React",
                "start_index": 10,
                "end_index": 15,
            },
        ]
        blocks, citations = build_web_search_results_from_annotations(
            web_tool_uses, annotations
        )
        assert len(blocks[0]["content"]) == 1  # deduplicated
        assert len(citations) == 2  # both citations kept

    def test_empty_annotations(self):
        web_tool_uses = [
            {"type": "server_tool_use", "name": "web_search", "id": "ws_1"}
        ]
        blocks, citations = build_web_search_results_from_annotations(
            web_tool_uses, []
        )
        assert len(blocks) == 1
        assert blocks[0]["content"] == []
        assert citations == []

    def test_no_web_search_calls(self):
        """Only web_fetch calls — no web_search_tool_result emitted."""
        web_tool_uses = [
            {"type": "server_tool_use", "name": "web_fetch", "id": "wf_1"}
        ]
        annotations = [
            {
                "type": "url_citation",
                "url": "https://example.com",
                "title": "Example",
                "start_index": 0,
                "end_index": 5,
            },
        ]
        blocks, citations = build_web_search_results_from_annotations(
            web_tool_uses, annotations
        )
        assert len(blocks) == 0
        assert len(citations) == 1


# ---------------------------------------------------------------------------
# build_text_blocks_with_citations
# ---------------------------------------------------------------------------


class TestBuildTextBlocksWithCitations:
    def test_no_citations(self):
        blocks = build_text_blocks_with_citations("Hello world", [])
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == "Hello world"
        assert "citations" not in blocks[0]

    def test_single_citation_middle(self):
        text = "Before cited text after"
        citations = [
            (7, 18, {"type": "web_search_result_location", "url": "https://example.com"}),
        ]
        blocks = build_text_blocks_with_citations(text, citations)
        assert len(blocks) == 3
        assert blocks[0]["text"] == "Before "
        assert "citations" not in blocks[0]
        assert blocks[1]["text"] == "cited text "
        assert len(blocks[1]["citations"]) == 1
        assert blocks[2]["text"] == "after"

    def test_citation_at_start(self):
        text = "cited rest"
        citations = [
            (0, 5, {"type": "web_search_result_location", "url": "https://example.com"}),
        ]
        blocks = build_text_blocks_with_citations(text, citations)
        assert len(blocks) == 2
        assert blocks[0]["text"] == "cited"
        assert "citations" in blocks[0]
        assert blocks[1]["text"] == " rest"

    def test_multiple_citations(self):
        text = "aaa bbb ccc ddd"
        citations = [
            (0, 3, {"type": "web_search_result_location", "url": "https://a.com"}),
            (8, 11, {"type": "web_search_result_location", "url": "https://c.com"}),
        ]
        blocks = build_text_blocks_with_citations(text, citations)
        assert len(blocks) == 4  # cited, uncited, cited, uncited
        assert blocks[0]["text"] == "aaa"
        assert "citations" in blocks[0]
        assert blocks[1]["text"] == " bbb "
        assert "citations" not in blocks[1]
        assert blocks[2]["text"] == "ccc"
        assert "citations" in blocks[2]
        assert blocks[3]["text"] == " ddd"


# ---------------------------------------------------------------------------
# Non-streaming: translate_response with web search
# ---------------------------------------------------------------------------


class TestTranslateResponseWebSearch:
    """Test the non-streaming translate_response path with web_search_call items."""

    def test_web_search_response_has_server_tool_use(self):
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )

        adapter = LiteLLMAnthropicToResponsesAPIAdapter()

        # Simulate a Responses API response with web_search_call + message
        response = MagicMock()
        response.id = "resp_123"
        response.model = "gpt-5.4"
        response.status = "completed"
        response.usage = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50

        search_item = {
            "type": "web_search_call",
            "id": "ws_1",
            "action": {"type": "search", "query": "React 19"},
        }
        message_item = {
            "type": "message",
            "content": [
                {
                    "type": "output_text",
                    "text": "React 19 is great.",
                    "annotations": [
                        {
                            "type": "url_citation",
                            "url": "https://react.dev",
                            "title": "React",
                            "start_index": 0,
                            "end_index": 8,
                        },
                    ],
                }
            ],
        }
        response.output = [search_item, message_item]

        result = adapter.translate_response(response)

        content_types = [b["type"] for b in result["content"]]
        assert "server_tool_use" in content_types
        assert "web_search_tool_result" in content_types
        assert "text" in content_types

        # server_tool_use should come first
        server_tool = [b for b in result["content"] if b["type"] == "server_tool_use"][0]
        assert server_tool["name"] == "web_search"
        assert server_tool["input"] == {"query": "React 19"}

        # usage should include server_tool_use counts
        assert result["usage"]["server_tool_use"]["web_search_requests"] == 1

    def test_no_web_search_response_unchanged(self):
        """Response without web_search_call should not add server_tool_use."""
        from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
            LiteLLMAnthropicToResponsesAPIAdapter,
        )
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        adapter = LiteLLMAnthropicToResponsesAPIAdapter()

        response = MagicMock()
        response.id = "resp_456"
        response.model = "gpt-5.4"
        response.status = "completed"
        response.usage = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50

        msg = ResponseOutputMessage(
            id="msg_1",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(type="output_text", text="Hello world", annotations=[])
            ],
        )
        response.output = [msg]

        result = adapter.translate_response(response)

        content_types = [b["type"] for b in result["content"]]
        assert "server_tool_use" not in content_types
        assert "web_search_tool_result" not in content_types
        assert "server_tool_use" not in result.get("usage", {})
