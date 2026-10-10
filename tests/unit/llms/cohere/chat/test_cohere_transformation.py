import datetime
import json
from collections.abc import Callable, Iterator
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.prompt_templates.factory import cohere_messages_pt_v2
from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from litellm.llms.custom_httpx.http_handler import HTTPHandler

COHERE_V1_CHAT_URL: Final = "https://api.cohere.ai/v1/chat"
COHERE_V2_CHAT_URL: Final = "https://api.cohere.com/v2/chat"


def _cohere_logging_obj(messages: list[object]) -> Logging:
    return Logging(
        model="command-r-plus",
        messages=messages,
        stream=False,
        call_type="completion",
        start_time=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="cohere-test-call",
        function_id="cohere-test-function",
    )


@pytest.fixture
def _cohere_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    monkeypatch.setattr(litellm, "sync_transport", None, raising=False)
    yield
    client_cache.flush_cache()


def _mock_post_response(mock_post: MagicMock, response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    def _respond(request: httpx.Request) -> httpx.Response:
        mock_post(request)
        return response

    return _respond


class TestCohereTransform:
    def setup_method(self):
        self.config = CohereChatConfig()
        self.model = "command-r-plus-latest"
        self.logging_obj = MagicMock()

    def test_map_cohere_params(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
            "max_completion_tokens": 256,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_completion_tokens to max_tokens and override max_tokens
        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_cohere_max_tokens_backward_compat(self):
        """Test that parameters are correctly mapped"""
        test_params = {
            "temperature": 0.7,
            "max_tokens": 200,
        }

        result = self.config.map_openai_params(
            non_default_params=test_params,
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        # The function should properly map max_tokens if max_completion_tokens is not provided
        assert result == {"temperature": 0.7, "max_tokens": 200}


class TestCohereV2Transform:
    def setup_method(self):
        self.config = CohereV2ChatConfig()
        self.model = "command-r"

    def test_v2_supports_max_completion_tokens(self):
        """max_completion_tokens must be advertised so get_optional_params does not reject it"""
        assert "max_completion_tokens" in self.config.get_supported_openai_params(self.model)

    def test_v2_max_tokens_only_still_maps(self):
        """max_tokens alone maps to cohere max_tokens when max_completion_tokens is absent"""
        result = self.config.map_openai_params(
            non_default_params={"temperature": 0.7, "max_tokens": 200},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7, "max_tokens": 200}


def test_cohere_v1_response_parses_citations_tool_calls_and_billed_units() -> None:
    config: Final = CohereChatConfig()
    model_response: Final = litellm.ModelResponse()
    raw_response: Final = httpx.Response(
        200,
        json={
            "text": "Searching",
            "citations": [{"start": 0, "end": 9, "text": "Searching", "document_ids": ["doc-1"]}],
            "tool_calls": [{"name": "lookup", "generation_id": "call-1", "parameters": {"query": "weather"}}],
            "meta": {"billed_units": {"input_tokens": 7, "output_tokens": 3}},
        },
    )

    response: Final = config.transform_response(
        model="command-r-plus",
        raw_response=raw_response,
        model_response=model_response,
        logging_obj=_cohere_logging_obj([]),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.choices[0].message.tool_calls[0].function.name == "lookup"
    assert response.choices[0].message.tool_calls[0].function.arguments == '{"query": "weather"}'
    assert getattr(response, "citations") == [{"start": 0, "end": 9, "text": "Searching", "document_ids": ["doc-1"]}]
    assert response.usage.prompt_tokens == 7
    assert response.usage.completion_tokens == 3


def test_cohere_v2_response_parses_annotations_and_tool_calls() -> None:
    config: Final = CohereV2ChatConfig()
    model_response: Final = litellm.ModelResponse()
    raw_response: Final = httpx.Response(
        200,
        json={
            "id": "response-1",
            "finish_reason": "COMPLETE",
            "message": {
                "content": [{"type": "text", "text": "The weather is clear."}],
                "citations": [
                    {
                        "start": 0,
                        "end": 21,
                        "sources": [
                            {
                                "type": "document",
                                "id": "source-1",
                                "url": "https://example.com/weather",
                                "document": {"title": "Weather report"},
                            }
                        ],
                    }
                ],
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            "usage": {"tokens": {"input_tokens": 11, "output_tokens": 5}},
        },
    )

    response: Final = config.transform_response(
        model="command-r-plus",
        raw_response=raw_response,
        model_response=model_response,
        logging_obj=_cohere_logging_obj([]),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.choices[0].message.tool_calls[0].function.name == "lookup"
    assert response.choices[0].message.tool_calls[0].function.arguments == '{"city":"Paris"}'
    assert response.choices[0].message.annotations[0]["url_citation"]["url"] == "https://example.com/weather"
    assert response.choices[0].message.annotations[0]["url_citation"]["title"] == "Weather report"
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 5


def test_cohere_v2_request_preserves_conversation_messages_and_parameters() -> None:
    config: Final = CohereV2ChatConfig()
    messages: Final = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "What language is spoken there?"},
    ]
    optional_params: Final = config.map_openai_params(
        non_default_params={
            "max_completion_tokens": 128,
            "top_p": 0.7,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
            "stop": ["END"],
        },
        optional_params={},
        model="command-r-plus",
        drop_params=False,
    )

    request: Final = config.transform_request(
        model="command-r-plus",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert request["messages"] == messages
    assert request["max_tokens"] == 128
    assert request["p"] == 0.7
    assert request["frequency_penalty"] == 0.3
    assert request["presence_penalty"] == 0.2
    assert request["stop_sequences"] == ["END"]


@pytest.mark.respx(assert_all_called=True)
def test_cohere_v2_completion_sends_mapped_request_and_parses_response(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(COHERE_V2_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "response-1",
                "finish_reason": "COMPLETE",
                "message": {"content": [{"type": "text", "text": "Paris is the capital."}]},
                "usage": {"tokens": {"input_tokens": 8, "output_tokens": 4}},
            },
        )
    )
    response: Final = litellm.completion(
        model="cohere_chat/command-r-plus",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        api_key="test-api-key",
        client=HTTPHandler(),
        max_completion_tokens=64,
        top_p=0.7,
    )

    assert response.choices[0].message.content == "Paris is the capital."
    assert response.usage.prompt_tokens == 8
    assert response.usage.completion_tokens == 4
    assert route.called
    request_body: Final = json.loads(route.calls.last.request.content)
    assert request_body["messages"] == [{"role": "user", "content": "What is the capital of France?"}]
    assert request_body["model"] == "command-r-plus"
    assert request_body["max_tokens"] == 64
    assert request_body["p"] == 0.7
    assert "top_p" not in request_body


def test_cohere_v1_request_separates_conversation_history_from_latest_message() -> None:
    messages: Final = [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4."},
        {"role": "user", "content": "And 3 + 3?"},
    ]

    latest_message, history = cohere_messages_pt_v2(
        messages=messages.copy(),
        model="command-r-plus",
        llm_provider="cohere_chat",
    )

    assert latest_message == "And 3 + 3?"
    assert history == [
        {"role": "USER", "message": "What is 2 + 2?"},
        {"role": "CHATBOT", "message": "4.", "tool_calls": []},
    ]


def test_cohere_v2_stream_parser_emits_text_tool_calls_and_usage() -> None:
    config: Final = CohereV2ChatConfig()
    response_iterator: Final = config.get_model_response_iterator(
        iter(
            [
                json.dumps(
                    {
                        "type": "content-delta",
                        "delta": {"message": {"content": {"text": "Searching"}}},
                    }
                ),
                json.dumps(
                    {
                        "type": "tool-call-delta",
                        "delta": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "name": "lookup",
                                    "arguments": '{"city":"Paris"}',
                                }
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "event": "message-end",
                        "data": {
                            "delta": {
                                "finish_reason": "COMPLETE",
                                "usage": {"tokens": {"input_tokens": 8, "output_tokens": 3}},
                            }
                        },
                    }
                ),
            ]
        ),
        sync_stream=True,
    )

    chunks: Final = list(response_iterator)

    assert chunks[0]["text"] == "Searching"
    assert chunks[1]["tool_use"] is not None
    assert chunks[1]["tool_use"]["id"] == "call-1"
    assert chunks[1]["tool_use"]["function"]["name"] == "lookup"
    assert chunks[2]["is_finished"]
    assert chunks[2]["finish_reason"] == "COMPLETE"
    assert chunks[2]["usage"]["prompt_tokens"] == 8
    assert chunks[2]["usage"]["completion_tokens"] == 3


def test_cohere_v2_error_class_preserves_status_and_message() -> None:
    config: Final = CohereV2ChatConfig()

    error: Final = config.get_error_class(
        error_message="rate limit exceeded",
        status_code=429,
        headers={},
    )

    assert error.status_code == 429
    assert error.message == "rate limit exceeded"

    def test_v2_map_max_completion_tokens_overrides_max_tokens(self):
        """max_completion_tokens maps to cohere max_tokens and overrides max_tokens, matching v1"""
        result = self.config.map_openai_params(
            non_default_params={
                "temperature": 0.7,
                "max_tokens": 200,
                "max_completion_tokens": 256,
            },
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert result == {"temperature": 0.7, "max_tokens": 256}

    def test_v2_max_completion_tokens_precedence_is_order_independent(self):
        """max_completion_tokens wins over max_tokens regardless of dict ordering"""
        max_tokens_first = self.config.map_openai_params(
            non_default_params={"max_tokens": 200, "max_completion_tokens": 256},
            optional_params={},
            model=self.model,
            drop_params=False,
        )
        max_completion_first = self.config.map_openai_params(
            non_default_params={"max_completion_tokens": 256, "max_tokens": 200},
            optional_params={},
            model=self.model,
            drop_params=False,
        )

        assert max_tokens_first == {"max_tokens": 256}
        assert max_completion_first == {"max_tokens": 256}

    def test_v2_default_route_accepts_max_completion_tokens(self):
        """The default cohere_chat route resolves to v2; max_completion_tokens must not raise"""
        optional_params = litellm.get_optional_params(
            model=self.model,
            custom_llm_provider="cohere_chat",
            max_completion_tokens=256,
        )

        assert optional_params["max_tokens"] == 256


@pytest.mark.asyncio
async def test_cohere_request_body_with_allowed_params():
    """
    Test to validate that when allowed_openai_params is provided, the request body contains
    the correct response_format and reasoning_effort values.
    """
    # Define test parameters
    test_response_format = {"type": "json"}
    test_reasoning_effort = "low"
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "I am Command, a language model developed by Cohere.",
        "generation_id": "mock-generation-id",
        "finish_reason": "COMPLETE",
    }

    # Mock the AsyncHTTPHandler.post method at the module level
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        try:
            await litellm.acompletion(
                model="cohere/v1/command",
                messages=[{"content": "what llm are you", "role": "user"}],
                allowed_openai_params=["tools", "response_format", "reasoning_effort"],
                response_format=test_response_format,
                reasoning_effort=test_reasoning_effort,
                tools=test_tools,
            )
        except Exception:
            pass  # We only care about the request body validation

        # Verify the API call was made
        mock_post.assert_called_once()

        # Get and parse the request body
        request_data = json.loads(mock_post.call_args.kwargs["data"])

        # Validate request contains our specified parameters
        assert "allowed_openai_params" not in request_data
        assert request_data["response_format"] == test_response_format
        assert request_data["reasoning_effort"] == test_reasoning_effort


@pytest.mark.asyncio
async def test_cohere_documents_options_in_request_body():
    """
    Test that documents parameters is properly included
    in the request body after transformation (sent via extra_body).
    """
    # Create a mock response
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Test response with citations",
        "generation_id": "mock-generation-id",
        "finish_reason": "COMPLETE",
    }

    # Mock the AsyncHTTPHandler.post method
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        try:
            # Test documents and citation_options parameters
            test_documents = [
                {
                    "data": {
                        "title": "Test Document 1",
                        "snippet": "This is test content 1",
                    }
                },
                {
                    "data": {
                        "title": "Test Document 2",
                        "snippet": "This is test content 2",
                    }
                },
            ]
            await litellm.acompletion(
                model="cohere_chat/command-a-03-2025",
                messages=[{"role": "user", "content": "Test message"}],
                documents=test_documents,
            )
        except Exception:
            pass  # We only care about the request body validation

        # Verify the API call was made
        mock_post.assert_called_once()

        # Get and parse the request body
        request_data = json.loads(mock_post.call_args.kwargs["data"])

        # Validate that documents and citation_options are in the request body
        assert "documents" in request_data
        assert request_data["documents"] == test_documents
