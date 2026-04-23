"""
Tests for Gemini context circulation (server-side tool invocations).

When includeServerSideToolInvocations=true is set, Gemini returns toolCall/toolResponse
parts for server-side tools (e.g. Google Search). These must be:
1. Extracted from the response into provider_specific_fields["server_side_tool_invocations"]
2. Re-injected as raw toolCall/toolResponse parts when converting messages back to Gemini format
3. The includeServerSideToolInvocations flag must be passed through to toolConfig
"""

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)
from litellm.types.llms.vertex_ai import HttpxPartType


# --- Response extraction tests ---


class TestExtractServerSideToolInvocations:
    """Test _extract_server_side_tool_invocations from response parts."""

    def test_extracts_tool_call_and_response(self):
        """Basic case: one toolCall + one toolResponse with same id."""
        parts: List[HttpxPartType] = [
            {
                "thoughtSignature": "sig_call_1",
                "toolCall": {
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "id": "abc123",
                    "args": {"queries": ["weather Buenos Aires"]},
                },
            },
            {
                "thoughtSignature": "sig_resp_1",
                "toolResponse": {
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "id": "abc123",
                    "response": {"weather": "Sunny, 20°C"},
                },
            },
            {
                "text": "The weather in Buenos Aires is sunny.",
                "thoughtSignature": "sig_text",
            },
        ]

        result = VertexGeminiConfig._extract_server_side_tool_invocations(parts)

        assert result is not None
        assert len(result) == 1
        assert result[0]["tool_type"] == "GOOGLE_SEARCH_WEB"
        assert result[0]["id"] == "abc123"
        assert result[0]["args"] == {"queries": ["weather Buenos Aires"]}
        assert result[0]["response"] == {"weather": "Sunny, 20°C"}
        assert result[0]["thought_signature"] == "sig_call_1"

    def test_returns_none_when_no_server_side_tools(self):
        """No toolCall/toolResponse parts → returns None."""
        parts: List[HttpxPartType] = [
            {"text": "Hello world", "thoughtSignature": "sig1"},
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Paris"},
                },
                "thoughtSignature": "sig2",
            },
        ]

        result = VertexGeminiConfig._extract_server_side_tool_invocations(parts)
        assert result is None

    def test_multiple_server_side_invocations(self):
        """Multiple toolCall/toolResponse pairs."""
        parts: List[HttpxPartType] = [
            {
                "toolCall": {
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "id": "search1",
                    "args": {"queries": ["query1"]},
                },
                "thoughtSignature": "sig1",
            },
            {
                "toolResponse": {"toolType": "GOOGLE_SEARCH_WEB", "id": "search1", "response": "result1"},
                "thoughtSignature": "sig2",
            },
            {
                "toolCall": {
                    "toolType": "GOOGLE_SEARCH_WEB",
                    "id": "search2",
                    "args": {"queries": ["query2"]},
                },
                "thoughtSignature": "sig3",
            },
            {
                "toolResponse": {"toolType": "GOOGLE_SEARCH_WEB", "id": "search2", "response": "result2"},
                "thoughtSignature": "sig4",
            },
        ]

        result = VertexGeminiConfig._extract_server_side_tool_invocations(parts)

        assert result is not None
        assert len(result) == 2
        assert result[0]["id"] == "search1"
        assert result[0]["response"] == "result1"
        assert result[1]["id"] == "search2"
        assert result[1]["response"] == "result2"

    def test_tool_call_without_response(self):
        """toolCall without matching toolResponse is still captured."""
        parts: List[HttpxPartType] = [
            {
                "toolCall": {
                    "toolType": "CODE_EXECUTION",
                    "id": "exec1",
                    "args": {"code": "print('hello')"},
                },
            },
        ]

        result = VertexGeminiConfig._extract_server_side_tool_invocations(parts)

        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "exec1"
        assert "response" not in result[0]


# --- Input re-injection tests ---


class TestReInjectServerSideToolInvocations:
    """Test that server_side_tool_invocations are re-injected into Gemini parts."""

    def test_roundtrip_single_invocation(self):
        """Server-side invocations from assistant message are converted back to Gemini parts."""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "It's sunny in Buenos Aires.",
                "provider_specific_fields": {
                    "server_side_tool_invocations": [
                        {
                            "tool_type": "GOOGLE_SEARCH_WEB",
                            "id": "abc123",
                            "args": {"queries": ["weather Buenos Aires"]},
                            "response": {"weather": "Sunny, 20°C"},
                            "thought_signature": "sig_abc",
                        }
                    ]
                },
            },
            {"role": "user", "content": "Thanks!"},
        ]

        contents = _gemini_convert_messages_with_history(messages)

        # Find the model turn
        model_turn = [c for c in contents if c["role"] == "model"]
        assert len(model_turn) == 1

        parts = model_turn[0]["parts"]
        # Should have: text part + toolCall part + toolResponse part
        tool_call_parts = [p for p in parts if "toolCall" in p]
        tool_response_parts = [p for p in parts if "toolResponse" in p]

        assert len(tool_call_parts) == 1
        assert tool_call_parts[0]["toolCall"]["toolType"] == "GOOGLE_SEARCH_WEB"
        assert tool_call_parts[0]["toolCall"]["id"] == "abc123"
        assert tool_call_parts[0]["toolCall"]["args"] == {"queries": ["weather Buenos Aires"]}
        assert tool_call_parts[0]["thoughtSignature"] == "sig_abc"

        assert len(tool_response_parts) == 1
        assert tool_response_parts[0]["toolResponse"]["id"] == "abc123"
        assert tool_response_parts[0]["toolResponse"]["toolType"] == "GOOGLE_SEARCH_WEB"
        assert tool_response_parts[0]["toolResponse"]["response"] == {"weather": "Sunny, 20°C"}

    def test_no_invocations_no_extra_parts(self):
        """Without server_side_tool_invocations, no extra parts are added."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Bye"},
        ]

        contents = _gemini_convert_messages_with_history(messages)
        model_turn = [c for c in contents if c["role"] == "model"]
        assert len(model_turn) == 1

        parts = model_turn[0]["parts"]
        assert len(parts) == 1
        assert "text" in parts[0]
        assert "toolCall" not in parts[0]


# --- toolConfig flag tests ---


class TestIncludeServerSideToolInvocationsConfig:
    """Test that the flag is passed through to toolConfig."""

    def test_flag_added_to_tool_config(self):
        """include_server_side_tool_invocations=True should be mapped to optional_params."""
        config = VertexGeminiConfig()
        non_default_params = {"include_server_side_tool_invocations": True}
        optional_params: Dict[str, Any] = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-3-flash-preview",
            drop_params=False,
        )

        assert result["include_server_side_tool_invocations"] is True

    def test_flag_in_supported_params(self):
        """include_server_side_tool_invocations should be in supported params."""
        config = VertexGeminiConfig()
        supported = config.get_supported_openai_params(model="gemini-3-flash-preview")
        assert "include_server_side_tool_invocations" in supported
