"""
Tests for AI Usage Chat module.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat import (
    GET_USAGE_DATA_TOOL,
    SYSTEM_PROMPT,
    _summarise_usage_data,
    stream_usage_ai_chat,
)


SAMPLE_AGGREGATED_RESPONSE = {
    "results": [
        {
            "date": "2025-01-15",
            "metrics": {
                "spend": 50.25,
                "prompt_tokens": 20000,
                "completion_tokens": 10000,
                "total_tokens": 30000,
                "api_requests": 500,
                "successful_requests": 480,
                "failed_requests": 20,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
            "breakdown": {
                "models": {
                    "gpt-4": {
                        "metrics": {
                            "spend": 40.0,
                            "api_requests": 300,
                            "total_tokens": 25000,
                            "prompt_tokens": 15000,
                            "completion_tokens": 10000,
                            "successful_requests": 290,
                            "failed_requests": 10,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                        "metadata": {},
                        "api_key_breakdown": {},
                    },
                    "gpt-3.5-turbo": {
                        "metrics": {
                            "spend": 10.25,
                            "api_requests": 200,
                            "total_tokens": 5000,
                            "prompt_tokens": 3000,
                            "completion_tokens": 2000,
                            "successful_requests": 190,
                            "failed_requests": 10,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                        "metadata": {},
                        "api_key_breakdown": {},
                    },
                },
                "providers": {
                    "openai": {
                        "metrics": {
                            "spend": 50.25,
                            "api_requests": 500,
                            "total_tokens": 30000,
                            "prompt_tokens": 20000,
                            "completion_tokens": 10000,
                            "successful_requests": 480,
                            "failed_requests": 20,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                        "metadata": {},
                        "api_key_breakdown": {},
                    },
                },
                "api_keys": {
                    "sk-test123": {
                        "metrics": {
                            "spend": 50.25,
                            "api_requests": 500,
                            "total_tokens": 30000,
                            "prompt_tokens": 20000,
                            "completion_tokens": 10000,
                            "successful_requests": 480,
                            "failed_requests": 20,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                        },
                        "metadata": {"key_alias": "Production Key"},
                    },
                },
                "model_groups": {},
                "mcp_servers": {},
                "entities": {},
            },
        },
    ],
    "metadata": {
        "total_spend": 50.25,
        "total_prompt_tokens": 20000,
        "total_completion_tokens": 10000,
        "total_tokens": 30000,
        "total_api_requests": 500,
        "total_successful_requests": 480,
        "total_failed_requests": 20,
        "total_cache_read_input_tokens": 0,
        "total_cache_creation_input_tokens": 0,
        "page": 1,
        "total_pages": 1,
        "has_more": False,
    },
}


class TestToolSchema:
    def test_tool_schema_is_valid(self):
        assert GET_USAGE_DATA_TOOL["type"] == "function"
        func = GET_USAGE_DATA_TOOL["function"]
        assert func["name"] == "get_usage_data"
        params = func["parameters"]
        assert "start_date" in params["properties"]
        assert "end_date" in params["properties"]
        assert "user_id" in params["properties"]
        assert params["required"] == ["start_date", "end_date"]

    def test_system_prompt_mentions_tool(self):
        assert "get_usage_data" in SYSTEM_PROMPT
        assert "usage" in SYSTEM_PROMPT.lower()


class TestSummariseUsageData:
    def test_summarise_includes_totals(self):
        summary = _summarise_usage_data(SAMPLE_AGGREGATED_RESPONSE)
        assert "$50.25" in summary
        assert "500" in summary
        assert "480" in summary
        assert "20" in summary

    def test_summarise_includes_models(self):
        summary = _summarise_usage_data(SAMPLE_AGGREGATED_RESPONSE)
        assert "gpt-4" in summary
        assert "gpt-3.5-turbo" in summary
        assert "$40.0" in summary

    def test_summarise_includes_providers(self):
        summary = _summarise_usage_data(SAMPLE_AGGREGATED_RESPONSE)
        assert "openai" in summary

    def test_summarise_includes_api_keys(self):
        summary = _summarise_usage_data(SAMPLE_AGGREGATED_RESPONSE)
        assert "Production Key" in summary

    def test_summarise_includes_daily(self):
        summary = _summarise_usage_data(SAMPLE_AGGREGATED_RESPONSE)
        assert "2025-01-15" in summary

    def test_summarise_handles_empty_data(self):
        empty = {"results": [], "metadata": {}}
        summary = _summarise_usage_data(empty)
        assert "no data" in summary.lower()


class TestStreamUsageAiChat:
    @pytest.mark.asyncio
    async def test_stream_with_tool_call(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function.name = "get_usage_data"
        mock_tool_call.function.arguments = json.dumps({
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
        })

        mock_first_response = MagicMock()
        mock_first_response.choices = [MagicMock()]
        mock_first_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_first_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_usage_data",
                        "arguments": '{"start_date":"2025-01-01","end_date":"2025-01-31"}',
                    },
                }
            ],
        }

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Your total spend is $50.25"
            yield chunk

        with patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm, \
             patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_usage_data", new_callable=AsyncMock) as mock_fetch:

            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_first_response,
                mock_stream(),
            ])
            mock_fetch.return_value = SAMPLE_AGGREGATED_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "What is my total spend?"}],
                model="gpt-4o-mini",
                user_id="user-123",
                is_admin=True,
            ):
                events.append(event)

            assert len(events) >= 2
            chunk_event = json.loads(events[0].replace("data: ", "").strip())
            assert chunk_event["type"] == "chunk"
            assert "$50.25" in chunk_event["content"]

            done_event = json.loads(events[-1].replace("data: ", "").strip())
            assert done_event["type"] == "done"

    @pytest.mark.asyncio
    async def test_stream_without_tool_call(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "I need more context."

        with patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "Hello"}],
            ):
                events.append(event)

            assert len(events) >= 2
            chunk_event = json.loads(events[0].replace("data: ", "").strip())
            assert chunk_event["type"] == "chunk"
            assert "context" in chunk_event["content"]

    @pytest.mark.asyncio
    async def test_stream_handles_error(self):
        with patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=Exception("LLM error"))

            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "test"}],
            ):
                events.append(event)

            assert len(events) == 1
            error_event = json.loads(events[0].replace("data: ", "").strip())
            assert error_event["type"] == "error"
            assert "LLM error" in error_event["message"]

    @pytest.mark.asyncio
    async def test_non_admin_enforces_user_id(self):
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_456"
        mock_tool_call.function.name = "get_usage_data"
        mock_tool_call.function.arguments = json.dumps({
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
            "user_id": "other-user",
        })

        mock_first_response = MagicMock()
        mock_first_response.choices = [MagicMock()]
        mock_first_response.choices[0].message.tool_calls = [mock_tool_call]
        mock_first_response.choices[0].message.model_dump.return_value = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "get_usage_data",
                        "arguments": '{"start_date":"2025-01-01","end_date":"2025-01-31","user_id":"other-user"}',
                    },
                }
            ],
        }

        async def mock_stream():
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = "Here is your data."
            yield chunk

        with patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat.litellm") as mock_litellm, \
             patch("litellm.proxy.management_endpoints.usage_endpoints.ai_usage_chat._fetch_usage_data", new_callable=AsyncMock) as mock_fetch:

            mock_litellm.acompletion = AsyncMock(side_effect=[
                mock_first_response,
                mock_stream(),
            ])
            mock_fetch.return_value = SAMPLE_AGGREGATED_RESPONSE

            events = []
            async for event in stream_usage_ai_chat(
                messages=[{"role": "user", "content": "Show me other user data"}],
                model="gpt-4o-mini",
                user_id="my-user-id",
                is_admin=False,
            ):
                events.append(event)

            mock_fetch.assert_called_once_with(
                start_date="2025-01-01",
                end_date="2025-01-31",
                user_id="my-user-id",
            )
