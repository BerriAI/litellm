"""DD LLM Observability intake-schema mapping: tool calls into
meta.output.messages[].tool_calls (DD ToolCall shape) and prompt-cache token
counts into top-level span metrics."""

import os
import sys
from datetime import datetime
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from litellm.types.utils import StandardLoggingPayload


def create_standard_logging_payload_with_tool_calls() -> StandardLoggingPayload:
    """Create a StandardLoggingPayload object with tool calls for testing"""
    return {
        "id": "test-request-id-tool-calls",
        "trace_id": "test-trace-id-tool-calls",
        "call_type": "completion",
        "stream": None,
        "response_cost": 0.05,
        "response_cost_failure_debug_info": None,
        "status": "success",
        "custom_llm_provider": "openai",
        "total_tokens": 50,
        "prompt_tokens": 20,
        "completion_tokens": 30,
        "startTime": 1234567890.0,
        "endTime": 1234567891.0,
        "completionStartTime": 1234567890.5,
        "response_time": 1.0,
        "model_map_information": {"model_map_key": "gpt-4", "model_map_value": None},
        "model": "gpt-4",
        "model_id": "model-123",
        "model_group": "openai-gpt",
        "api_base": "https://api.openai.com",
        "metadata": {
            "user_api_key_hash": "test_hash",
            "user_api_key_org_id": None,
            "user_api_key_alias": "test_alias",
            "user_api_key_team_id": "test_team",
            "user_api_key_user_id": "test_user",
            "user_api_key_team_alias": "test_team_alias",
            "user_api_key_user_email": None,
            "user_api_key_end_user_id": None,
            "user_api_key_request_route": None,
            "spend_logs_metadata": None,
            "requester_ip_address": "127.0.0.1",
            "requester_metadata": None,
            "requester_custom_headers": None,
            "prompt_management_metadata": None,
            "mcp_tool_call_metadata": None,
            "vector_store_request_metadata": None,
            "applied_guardrails": None,
            "usage_object": None,
            "cold_storage_object_key": None,
        },
        "cache_hit": False,
        "cache_key": None,
        "saved_cache_cost": 0.0,
        "request_tags": [],
        "end_user": None,
        "requester_ip_address": "127.0.0.1",
        "messages": [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": "I'll check the weather for you.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": '{"temperature": 72, "condition": "sunny"}',
            },
        ],
        "response": {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "It's 72°F and sunny in NYC!",
                        "tool_calls": [
                            {
                                "id": "call_456",
                                "type": "function",
                                "function": {
                                    "name": "format_response",
                                    "arguments": '{"temp": 72, "condition": "sunny"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        "error_str": None,
        "error_information": None,
        "model_parameters": {"temperature": 0.7},
        "hidden_params": {
            "model_id": "model-123",
            "cache_key": None,
            "api_base": "https://api.openai.com",
            "response_cost": "0.05",
            "litellm_overhead_time_ms": None,
            "additional_headers": None,
            "batch_models": None,
            "litellm_model_name": None,
            "usage_object": None,
        },
        "guardrail_information": None,
        "standard_built_in_tools_params": None,
    }  # type: ignore


class TestDataDogLLMObsLoggerToolCalls:
    """Simple test suite for DataDog LLM Observability Logger tool call handling"""

    @pytest.fixture
    def mock_env_vars(self):
        """Mock environment variables for DataDog"""
        with patch.dict(os.environ, {"DD_API_KEY": "test_api_key", "DD_SITE": "us5.datadoghq.com"}):
            yield

    def test_tool_call_span_kind_mapping(self, mock_env_vars):
        """Test that tool call operations are correctly mapped to 'tool' span kind"""
        with (
            patch("litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"),
            patch("asyncio.create_task"),
        ):
            logger = DataDogLLMObsLogger()

            # Test MCP tool call mapping
            from litellm.types.utils import CallTypes

            assert logger._get_datadog_span_kind(CallTypes.call_mcp_tool.value, "123") == "tool"

    def test_tool_call_payload_creation(self, mock_env_vars):
        """Test that tool call payloads are created correctly"""
        with (
            patch("litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"),
            patch("asyncio.create_task"),
        ):
            logger = DataDogLLMObsLogger()

            standard_payload = create_standard_logging_payload_with_tool_calls()

            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {}},
            }

            start_time = datetime.now()
            end_time = datetime.now()

            payload = logger.create_llm_obs_payload(kwargs, start_time, end_time)

            # Verify basic payload structure
            assert payload.get("name") == "litellm_llm_call"
            assert payload.get("status") == "ok"
            assert payload.get("meta", {}).get("kind") == "llm"  # Regular completion, not tool call

            # Verify metrics
            metrics = payload.get("metrics", {})
            assert metrics.get("input_tokens") == 20
            assert metrics.get("output_tokens") == 30
            assert metrics.get("total_tokens") == 50

    def test_tool_call_messages_preserved(self, mock_env_vars):
        """Test that tool call messages are preserved in the payload"""
        with (
            patch("litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"),
            patch("asyncio.create_task"),
        ):
            logger = DataDogLLMObsLogger()

            standard_payload = create_standard_logging_payload_with_tool_calls()

            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {}},
            }

            start_time = datetime.now()
            end_time = datetime.now()

            payload = logger.create_llm_obs_payload(kwargs, start_time, end_time)

            # Verify input messages include tool calls
            meta = payload.get("meta", {})
            input_meta = meta.get("input", {})
            input_messages = input_meta.get("messages", [])
            assert len(input_messages) == 3

            # Check assistant message has tool calls
            assistant_msg = input_messages[1]
            assert assistant_msg.get("role") == "assistant"
            assert "tool_calls" in assistant_msg
            tool_calls = assistant_msg.get("tool_calls", [])
            assert len(tool_calls) == 1
            tool_call = tool_calls[0]
            function_info = tool_call.get("function", {})
            assert function_info.get("name") == "get_weather"

            # Check tool message
            tool_msg = input_messages[2]
            assert tool_msg.get("role") == "tool"
            assert tool_msg.get("tool_call_id") == "call_123"

    def test_tool_call_response_handling(self, mock_env_vars):
        """Test that tool calls in response are handled correctly"""
        with (
            patch("litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"),
            patch("asyncio.create_task"),
        ):
            logger = DataDogLLMObsLogger()

            standard_payload = create_standard_logging_payload_with_tool_calls()

            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {}},
            }

            start_time = datetime.now()
            end_time = datetime.now()

            payload = logger.create_llm_obs_payload(kwargs, start_time, end_time)

            meta = payload.get("meta", {})
            output_meta = meta.get("output", {})
            output_messages = output_meta.get("messages", [])
            assert len(output_messages) == 1

            output_msg = output_messages[0]
            assert output_msg.get("role") == "assistant"
            assert output_msg.get("content") == "It's 72°F and sunny in NYC!"
            assert "tool_calls" in output_msg
            output_tool_calls = output_msg.get("tool_calls", [])
            assert len(output_tool_calls) == 1
            dd_tool_call = output_tool_calls[0]
            assert dd_tool_call == {
                "name": "format_response",
                "arguments": {"temp": 72, "condition": "sunny"},
                "tool_id": "call_456",
                "type": "function",
            }

    def test_output_tool_call_with_unparseable_arguments(self, mock_env_vars):
        """Malformed JSON arguments are kept as the raw string, not dropped"""
        with (
            patch("litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"),
            patch("asyncio.create_task"),
        ):
            logger = DataDogLLMObsLogger()

            standard_payload = create_standard_logging_payload_with_tool_calls()
            standard_payload["response"]["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
                "{not valid json"
            )

            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {}},
            }

            payload = logger.create_llm_obs_payload(kwargs, datetime.now(), datetime.now())

            tool_call = payload["meta"]["output"]["messages"][0]["tool_calls"][0]
            assert tool_call["name"] == "format_response"
            assert tool_call["arguments"] == "{not valid json"

    def test_output_message_without_tool_calls_unchanged(self, mock_env_vars):
        """Plain responses keep role/content and gain no tool_calls key"""
        with (
            patch("litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"),
            patch("asyncio.create_task"),
        ):
            logger = DataDogLLMObsLogger()

            standard_payload = create_standard_logging_payload_with_tool_calls()
            standard_payload["response"] = {"choices": [{"message": {"role": "assistant", "content": "Hi!"}}]}

            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {}},
            }

            payload = logger.create_llm_obs_payload(kwargs, datetime.now(), datetime.now())

            output_msg = payload["meta"]["output"]["messages"][0]
            assert output_msg == {"role": "assistant", "content": "Hi!"}


class TestDataDogLLMObsCacheTokenMetrics:
    """Prompt-cache token counts must land in top-level span metrics"""

    @pytest.fixture
    def mock_env_vars(self):
        with patch.dict(os.environ, {"DD_API_KEY": "test_api_key", "DD_SITE": "us5.datadoghq.com"}):
            yield

    def _payload_with_usage_object(self, usage_object):
        standard_payload = create_standard_logging_payload_with_tool_calls()
        standard_payload["metadata"]["usage_object"] = usage_object
        return standard_payload

    def test_cache_tokens_forwarded_to_span_metrics(self, mock_env_vars):
        """cache_read/cache_creation tokens map to DD span metrics fields"""
        with (
            patch("litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"),
            patch("asyncio.create_task"),
        ):
            logger = DataDogLLMObsLogger()

            standard_payload = self._payload_with_usage_object(
                {
                    "cache_creation_input_tokens": 176,
                    "cache_read_input_tokens": 16695,
                    "prompt_tokens": 16872,
                    "completion_tokens": 704,
                }
            )
            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {}},
            }

            payload = logger.create_llm_obs_payload(kwargs, datetime.now(), datetime.now())

            metrics = payload["metrics"]
            assert metrics["cache_read_input_tokens"] == 16695.0
            assert metrics["cache_write_input_tokens"] == 176.0
            assert metrics["non_cached_input_tokens"] == 177.0

    def test_no_cache_metrics_when_usage_object_absent(self, mock_env_vars):
        """Without cache activity the new metrics keys are not emitted"""
        with (
            patch("litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"),
            patch("asyncio.create_task"),
        ):
            logger = DataDogLLMObsLogger()

            standard_payload = self._payload_with_usage_object(None)
            kwargs = {
                "standard_logging_object": standard_payload,
                "litellm_params": {"metadata": {}},
            }

            payload = logger.create_llm_obs_payload(kwargs, datetime.now(), datetime.now())

            metrics = payload["metrics"]
            assert "cache_read_input_tokens" not in metrics
            assert "cache_write_input_tokens" not in metrics
            assert "non_cached_input_tokens" not in metrics
            assert metrics["input_tokens"] == 20.0


def test_parse_tool_call_arguments_survives_deeply_nested_json():
    """A hostile/degenerate arguments string must fall back to the raw
    string, not raise RecursionError and drop the span."""
    from litellm.integrations.datadog.datadog_llm_obs import (
        _parse_tool_call_arguments,
    )

    hostile = "[" * 50000
    assert _parse_tool_call_arguments(hostile) == hostile
