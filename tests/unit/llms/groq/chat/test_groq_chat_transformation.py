import logging
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import (
    StandardBuiltInToolCostTracking,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.groq.chat.transformation import (
    GroqChatCompletionStreamingHandler,
    GroqChatConfig,
)
from litellm.utils import get_optional_params

WEB_SEARCH_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-safeguard-20b",
)

COMPOUND_MODELS = ("compound", "compound-mini", "groq/compound", "groq/compound-mini")


class TestGroqWebSearchOptions:
    @pytest.mark.parametrize("model", WEB_SEARCH_MODELS + COMPOUND_MODELS)
    def test_supported_on_search_capable_models(self, model: str):
        assert "web_search_options" in GroqChatConfig().get_supported_openai_params(model)

    def test_not_supported_on_other_models(self):
        assert "web_search_options" not in GroqChatConfig().get_supported_openai_params("llama-3.3-70b-versatile")

    @pytest.mark.parametrize("web_search_options", [{"search_context_size": "high"}, {}])
    def test_translates_to_browser_search_tool(self, web_search_options: dict):
        optional_params = get_optional_params(
            model="openai/gpt-oss-20b",
            custom_llm_provider="groq",
            web_search_options=web_search_options,
        )
        assert optional_params["tools"] == [{"type": "browser_search"}]
        assert "web_search_options" not in optional_params

    def test_no_duplicate_browser_search_tool(self):
        optional_params = get_optional_params(
            model="openai/gpt-oss-20b",
            custom_llm_provider="groq",
            web_search_options={"search_context_size": "high"},
            tools=[{"type": "browser_search"}],
        )
        assert optional_params["tools"] == [{"type": "browser_search"}]

    def test_caller_function_tools_preserved(self):
        function_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        optional_params = get_optional_params(
            model="openai/gpt-oss-20b",
            custom_llm_provider="groq",
            web_search_options={},
            tools=[function_tool],
        )
        assert optional_params["tools"] == [function_tool, {"type": "browser_search"}]

    def test_unsupported_model_drops_param_with_drop_params(self):
        optional_params = get_optional_params(
            model="llama-3.3-70b-versatile",
            custom_llm_provider="groq",
            web_search_options={"search_context_size": "high"},
            drop_params=True,
        )
        assert "web_search_options" not in optional_params
        assert "tools" not in optional_params

    def test_unsupported_model_raises_without_drop_params(self):
        with pytest.raises(litellm.UnsupportedParamsError):
            get_optional_params(
                model="llama-3.3-70b-versatile",
                custom_llm_provider="groq",
                web_search_options={"search_context_size": "high"},
                drop_params=False,
            )

    @pytest.mark.parametrize("model", COMPOUND_MODELS)
    def test_compound_injects_no_tool(self, model: str):
        optional_params = get_optional_params(
            model=model,
            custom_llm_provider="groq",
            web_search_options={"search_context_size": "high"},
        )
        assert "web_search_options" not in optional_params
        assert "tools" not in optional_params

    def test_ignored_fields_logged_as_info(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO, logger="LiteLLM"):
            get_optional_params(
                model="openai/gpt-oss-20b",
                custom_llm_provider="groq",
                web_search_options={"search_context_size": "high", "user_location": {"type": "approximate"}},
            )
        ignored_fields_records = tuple(
            record
            for record in caplog.records
            if "search_context_size" in record.message and "user_location" in record.message
        )
        assert len(ignored_fields_records) == 1
        assert ignored_fields_records[0].levelno == logging.INFO
        assert "enabled" in ignored_fields_records[0].message

    def test_empty_options_log_nothing(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO, logger="LiteLLM"):
            get_optional_params(
                model="openai/gpt-oss-20b",
                custom_llm_provider="groq",
                web_search_options={},
            )
        assert not [record for record in caplog.records if "web_search_options" in record.message]


def _searched_groq_response(executed_tools: list | None) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-oss-20b",
        "service_tier": "auto",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Top headline: example",
                    **({"executed_tools": executed_tools} if executed_tools is not None else {}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
    }


EXECUTED_TOOLS_THREE_SEARCHES_TWO_OPENS = [
    {"name": "browser.search", "type": "browser_search"},
    {"name": "browser.open", "type": "function"},
    {"name": "browser.search", "type": "browser_search"},
    {"type": "browser_search"},
    {"name": "browser.open", "type": "browser_search"},
    {"name": "browser.find", "type": "browser.find"},
]

EXECUTED_TOOLS_OPENS_ONLY = [
    {"name": "browser.open", "type": "browser.open"},
    {"name": "browser.open", "type": "function"},
    {"name": "browser.find", "type": "browser.find"},
]


def _groq_completion_with_mocked_response(response_json: dict) -> litellm.ModelResponse:
    client = HTTPHandler()
    fake_response = httpx.Response(
        status_code=200,
        json=response_json,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    with patch.object(client, "post", return_value=fake_response):
        return litellm.completion(
            model="groq/openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "hi"}],
            web_search_options={"search_context_size": "high"},
            api_key="fake-key",
            client=client,
        )


class TestGroqWebSearchUsageSignal:
    @pytest.mark.parametrize(
        "executed_tools, expected_searches, expected_opens",
        [
            (EXECUTED_TOOLS_THREE_SEARCHES_TWO_OPENS, 3, 2),
            (EXECUTED_TOOLS_OPENS_ONLY, 0, 2),
        ],
    )
    def test_counts_actions_into_usage(self, executed_tools: list, expected_searches: int, expected_opens: int):
        response = _groq_completion_with_mocked_response(_searched_groq_response(executed_tools))
        assert response.usage.server_tool_use.web_search_requests == expected_searches
        assert response.usage.server_tool_use.browser_open_requests == expected_opens

    def test_no_signal_without_executed_tools(self):
        response = _groq_completion_with_mocked_response(_searched_groq_response(None))
        assert getattr(response.usage, "server_tool_use", None) is None

    def test_malformed_executed_tools_skips_billing_without_breaking_response(self):
        response = _groq_completion_with_mocked_response(
            _searched_groq_response(["not-a-dict", {"name": {"nested": "junk"}}])
        )
        assert response.choices[0].message.content == "Top headline: example"
        assert getattr(response.usage, "server_tool_use", None) is None

    def test_response_without_usage_is_left_untouched(self):
        model_response = litellm.ModelResponse()
        GroqChatConfig()._add_web_search_usage(model_response=model_response)
        assert getattr(model_response, "usage", None) is None


@pytest.mark.parametrize(
    "model",
    ["groq/qwen/qwen3.8-27b", "groq/openai/gpt-oss-20b", "groq/openai/gpt-oss-120b"],
)
def test_reasoning_effort_in_supported_params(model):
    """Test that reasoning_effort is in the list of supported parameters for Groq"""
    supported_params = GroqChatConfig().get_supported_openai_params(model=model)
    assert "reasoning_effort" in supported_params


class TestGroqStructuredOutputs:
    """
    Tests for Groq structured outputs handling.
    Related issues:
    - https://github.com/BerriAI/litellm/issues/11001
    - https://github.com/openai/openai-agents-python/issues/2140
    """

    def test_structured_output_with_tools_raises_error_for_non_native_models(self):
        """
        Test that using structured outputs + tools with models that don't support
        native json_schema raises a clear error message.

        Groq does not support structured outputs + tools together.
        See: https://console.groq.com/docs/structured-outputs
        "Streaming and tool use are not currently supported with Structured Outputs"
        """
        config = GroqChatConfig()


        model = "llama-3.3-70b-versatile"

        non_default_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            },
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        with pytest.raises(litellm.BadRequestError) as exc_info:
            config.map_openai_params(
                non_default_params=non_default_params,
                optional_params={},
                model=model,
                drop_params=False,
            )

        assert "does not support native structured outputs" in str(exc_info.value)
        assert "incompatible with user-provided tools" in str(exc_info.value)

    def test_structured_output_without_tools_uses_workaround_for_non_native_models(
        self,
    ):
        """
        Test that structured outputs without tools works using the json_tool_call workaround
        for models that don't support native json_schema.
        """
        config = GroqChatConfig()

        model = "llama-3.3-70b-versatile"

        non_default_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            }
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )


        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "json_tool_call"
        assert result["tool_choice"]["function"]["name"] == "json_tool_call"
        assert result.get("json_mode") is True

    def test_structured_output_passes_through_for_native_models(self):
        """
        Test that structured outputs pass through directly for models that
        support native json_schema (e.g., gpt-oss-120b).
        """
        config = GroqChatConfig()


        model = "openai/gpt-oss-120b"

        non_default_params = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "test",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            }
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )



        assert result.get("json_mode") is not True

        if "tools" in result:
            tool_names = [t.get("function", {}).get("name") for t in result["tools"]]
            assert "json_tool_call" not in tool_names


class TestGroqReasoning:
    """
    Tests for Groq reasoning field mapping.

    Groq returns 'reasoning' field in delta, but LiteLLM expects 'reasoning_content'.
    """

    def test_reasoning_field_mapping_in_streaming_chunks(self):
        """
        Test that Groq's 'reasoning' field in streaming chunks is properly mapped
        to LiteLLM's 'reasoning_content' field.
        """
        handler = GroqChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )


        groq_chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1769511767,
            "model": "qwen/qwen3-32b",
            "choices": [
                {
                    "delta": {
                        "reasoning": "This is reasoning content",
                        "role": None,
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        }


        parsed_chunk = handler.chunk_parser(groq_chunk)


        assert (
            parsed_chunk.choices[0].delta.reasoning_content
            == "This is reasoning content"
        )

        assert not hasattr(parsed_chunk.choices[0].delta, "reasoning")

    def test_reasoning_field_not_present(self):
        """
        Test that chunks without reasoning field still work correctly.
        """
        handler = GroqChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )


        groq_chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1769511767,
            "model": "qwen/qwen3-32b",
            "choices": [
                {
                    "delta": {
                        "content": "Regular content",
                        "role": "assistant",
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        }


        parsed_chunk = handler.chunk_parser(groq_chunk)


        assert parsed_chunk.choices[0].delta.content == "Regular content"
        assert parsed_chunk.choices[0].delta.role == "assistant"

        assert not hasattr(parsed_chunk.choices[0].delta, "reasoning_content")

    def test_reasoning_with_tool_calls(self):
        """
        Test that reasoning field is properly mapped even when tool_calls are present.
        """
        handler = GroqChatCompletionStreamingHandler(
            streaming_response=None, sync_stream=True
        )


        groq_chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1769511767,
            "model": "qwen/qwen3-32b",
            "choices": [
                {
                    "delta": {
                        "reasoning": "Reasoning before tool call",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {
                                    "name": "test_function",
                                    "arguments": "{}",
                                },
                                "type": "function",
                            }
                        ],
                    },
                    "finish_reason": None,
                    "index": 0,
                }
            ],
        }


        parsed_chunk = handler.chunk_parser(groq_chunk)


        assert (
            parsed_chunk.choices[0].delta.reasoning_content
            == "Reasoning before tool call"
        )

        assert parsed_chunk.choices[0].delta.tool_calls is not None
        assert len(parsed_chunk.choices[0].delta.tool_calls) == 1
        assert (
            parsed_chunk.choices[0].delta.tool_calls[0]["function"]["name"]
            == "test_function"
        )
