import json
from collections.abc import Callable, Iterator
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.cohere.chat.transformation import CohereChatConfig
from litellm.llms.cohere.chat.v2_transformation import CohereV2ChatConfig
from unittest.mock import AsyncMock, patch

COHERE_V1_CHAT_URL: Final = "https://api.cohere.ai/v1/chat"
COHERE_V2_CHAT_URL: Final = "https://api.cohere.com/v2/chat"


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
        print(f"request_data: {request_data}")

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
        print(f"Request body: {request_data}")

        # Validate that documents and citation_options are in the request body
        assert "documents" in request_data
        assert request_data["documents"] == test_documents


PENGUIN_CITATIONS: Final = [{"start": 0, "end": 16, "text": "Emperor penguins", "document_ids": ["doc_0"]}]
WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


def _cohere_v2_response(message: dict[str, object], finish_reason: str = "COMPLETE") -> dict[str, object]:
    return {
        "id": "c14c80c3-18eb-4519-9460-6c92edd8cfb4",
        "finish_reason": finish_reason,
        "message": {"role": "assistant", **message},
        "usage": {
            "billed_units": {"input_tokens": 17, "output_tokens": 9},
            "tokens": {"input_tokens": 211, "output_tokens": 9},
        },
    }


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("stream", [True, False])
async def test_chat_completion_cohere_citations(respx_mock: respx.MockRouter, stream: bool) -> None:
    documents: Final = [
        {"title": "Tall penguins", "text": "Emperor penguins are the tallest."},
        {"title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."},
    ]
    final_response: Final = {
        "response_id": "5b3a4a52-2f1c-4b0f-9b43-0b5e2b0f3a5c",
        "text": "Emperor penguins are the tallest.",
        "generation_id": "0c2cf4b3-7ab0-4b0c-8f6b-3a0f2d3c1b7e",
        "citations": PENGUIN_CITATIONS,
        "finish_reason": "COMPLETE",
        "meta": {"api_version": {"version": "1"}, "billed_units": {"input_tokens": 5, "output_tokens": 6}},
    }
    stream_events: Final = (
        {"is_finished": False, "event_type": "stream-start", "generation_id": final_response["generation_id"]},
        {"is_finished": False, "event_type": "text-generation", "text": "Emperor penguins"},
        {"is_finished": False, "event_type": "citation-generation", "citations": PENGUIN_CITATIONS},
        {"is_finished": True, "event_type": "stream-end", "finish_reason": "COMPLETE", "response": final_response},
    )
    route: Final = respx_mock.post(COHERE_V1_CHAT_URL).mock(
        return_value=(
            httpx.Response(200, content="".join(json.dumps(event) + "\n" for event in stream_events).encode())
            if stream
            else httpx.Response(200, json=final_response)
        )
    )

    response: Final = await litellm.acompletion(
        model="cohere_chat/v1/command-r",
        messages=[{"role": "user", "content": "Which penguins are the tallest?"}],
        documents=documents,
        stream=stream,
        api_key="cohere-test-key",
    )

    assert json.loads(route.calls.last.request.content)["documents"] == documents
    if stream:
        chunks: Final = [chunk async for chunk in response]
        assert [getattr(chunk, "citations", None) for chunk in chunks] == [None, PENGUIN_CITATIONS, None]
    else:
        assert response.citations == PENGUIN_CITATIONS


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
def test_completion_cohere_command_r_plus_function_call(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(COHERE_V1_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "response_id": "8d1a1e0e-4f4e-4c9a-9a52-3b8d3f0f6c11",
                "text": "",
                "generation_id": "f7a1a4f0-6c1f-4f1d-9c1e-2d6c4a3b2e10",
                "finish_reason": "COMPLETE",
                "tool_calls": [
                    {"name": "get_current_weather", "parameters": {"location": "Boston, MA", "unit": "fahrenheit"}}
                ],
                "meta": {"api_version": {"version": "1"}, "billed_units": {"input_tokens": 30, "output_tokens": 20}},
            },
        )
    )

    response: Final = litellm.completion(
        model="cohere_chat/v1/command-r-plus",
        messages=[{"role": "user", "content": "What's the weather like in Boston today in Fahrenheit?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        api_key="cohere-test-key",
    )

    assert json.loads(route.calls.last.request.content)["tools"] == [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameter_definitions": {
                "location": {
                    "description": "The city and state, e.g. San Francisco, CA",
                    "type": "string",
                    "required": True,
                },
                "unit": {"description": "", "type": "string", "required": False},
            },
        }
    ]
    tool_call: Final = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_current_weather"
    assert json.loads(tool_call.function.arguments) == {"location": "Boston, MA", "unit": "fahrenheit"}


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_cohere_v2_chat_completion(respx_mock: respx.MockRouter, sync_mode: bool) -> None:
    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
    ]
    route: Final = respx_mock.post(COHERE_V2_CHAT_URL).mock(
        return_value=httpx.Response(
            200, json=_cohere_v2_response({"content": [{"type": "text", "text": "3+3 equals 6."}]})
        )
    )
    kwargs: Final = {
        "model": "cohere_chat/v2/command-a-03-2025",
        "messages": messages,
        "max_tokens": 50,
        "api_key": "cohere-test-key",
    }

    response: Final = litellm.completion(**kwargs) if sync_mode else await litellm.acompletion(**kwargs)

    assert json.loads(route.calls.last.request.content) == {
        "model": "command-a-03-2025",
        "messages": messages,
        "max_tokens": 50,
    }
    assert response.choices[0].message.content == "3+3 equals 6."
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 211
    assert response.usage.completion_tokens == 9
    assert response.usage.total_tokens == 220


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("stream", [True, False])
async def test_cohere_v2_streaming(respx_mock: respx.MockRouter, stream: bool) -> None:
    stream_events: Final = (
        {
            "type": "message-start",
            "id": "29f14a5a-11de-4cae-9800-25e4747408ea",
            "delta": {"message": {"role": "assistant"}},
        },
        {"type": "content-start", "index": 0, "delta": {"message": {"content": {"type": "text", "text": ""}}}},
        {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": "Once upon"}}}},
        {"type": "content-delta", "index": 0, "delta": {"message": {"content": {"text": " a time"}}}},
        {"type": "content-end", "index": 0},
    )
    respx_mock.post(COHERE_V2_CHAT_URL).mock(
        return_value=(
            httpx.Response(200, content="".join(json.dumps(event) + "\n" for event in stream_events).encode())
            if stream
            else httpx.Response(
                200, json=_cohere_v2_response({"content": [{"type": "text", "text": "Once upon a time"}]})
            )
        )
    )

    response: Final = await litellm.acompletion(
        model="cohere_chat/v2/command-a-03-2025",
        messages=[{"role": "user", "content": "Tell me a short story about a robot."}],
        max_tokens=100,
        stream=stream,
        api_key="cohere-test-key",
    )

    if stream:
        chunks: Final = [chunk async for chunk in response]
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Once upon a time"
        assert chunks[-1].choices[0].finish_reason == "stop"
    else:
        assert response.choices[0].message.content == "Once upon a time"


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
def test_cohere_v2_tool_calling(respx_mock: respx.MockRouter) -> None:
    tool: Final = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "The city and state"}},
                "required": ["location"],
            },
        },
    }
    route: Final = respx_mock.post(COHERE_V2_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_cohere_v2_response(
                {
                    "tool_plan": "I will look up the weather in New York.",
                    "tool_calls": [
                        {
                            "id": "get_weather_k9q2xw1n",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location":"New York, NY"}'},
                        }
                    ],
                },
                finish_reason="TOOL_CALL",
            ),
        )
    )

    response: Final = litellm.completion(
        model="cohere_chat/v2/command-a-03-2025",
        messages=[{"role": "user", "content": "What's the weather like in New York?"}],
        tools=[tool],
        tool_choice="auto",
        max_tokens=100,
        api_key="cohere-test-key",
    )

    assert json.loads(route.calls.last.request.content)["tools"] == [tool]
    tool_calls: Final = response.choices[0].message.tool_calls
    assert [(call.id, call.function.name, call.function.arguments) for call in tool_calls] == [
        ("get_weather_k9q2xw1n", "get_weather", '{"location":"New York, NY"}')
    ]


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
async def test_cohere_v2_annotations(respx_mock: respx.MockRouter) -> None:
    documents: Final = [
        {"data": {"title": "Renewable Energy Benefits Document", "snippet": "Solar and wind provide clean power."}},
        {"data": {"title": "Environmental Impact Study", "snippet": "Renewables reduce carbon footprint."}},
    ]
    route: Final = respx_mock.post(COHERE_V2_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            json=_cohere_v2_response(
                {
                    "content": [{"type": "text", "text": "Renewables provide clean power and cut emissions."}],
                    "citations": [
                        {
                            "start": 0,
                            "end": 32,
                            "text": "Renewables provide clean power",
                            "type": "TEXT_CONTENT",
                            "sources": [
                                {
                                    "type": "document",
                                    "id": "doc:0",
                                    "document": {"id": "doc:0", "title": "Renewable Energy Benefits Document"},
                                },
                                {
                                    "type": "document",
                                    "id": "doc:1",
                                    "url": "https://example.com/impact-study",
                                    "document": {"id": "doc:1", "title": "Environmental Impact Study"},
                                },
                            ],
                        }
                    ],
                }
            ),
        )
    )

    response: Final = await litellm.acompletion(
        model="cohere_chat/v2/command-a-03-2025",
        messages=[{"role": "user", "content": "What are the benefits of renewable energy?"}],
        documents=documents,
        max_tokens=100,
        api_key="cohere-test-key",
    )

    assert json.loads(route.calls.last.request.content)["documents"] == documents
    assert response.choices[0].message.annotations == [
        {
            "type": "url_citation",
            "url_citation": {
                "start_index": 0,
                "end_index": 32,
                "title": "Renewable Energy Benefits Document",
                "url": "source:doc:0",
            },
        },
        {
            "type": "url_citation",
            "url_citation": {
                "start_index": 0,
                "end_index": 32,
                "title": "Environmental Impact Study",
                "url": "https://example.com/impact-study",
            },
        },
    ]
    assert not hasattr(response, "citations")


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
def test_cohere_v2_parameter_mapping(respx_mock: respx.MockRouter) -> None:
    messages: Final = [{"role": "user", "content": "Generate a creative story."}]
    route: Final = respx_mock.post(COHERE_V2_CHAT_URL).mock(
        return_value=httpx.Response(
            200, json=_cohere_v2_response({"content": [{"type": "text", "text": "A story."}]})
        )
    )

    litellm.completion(
        model="cohere_chat/v2/command-a-03-2025",
        messages=messages,
        temperature=0.7,
        max_tokens=50,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        stop=["END", "STOP"],
        seed=42,
        api_key="cohere-test-key",
    )

    assert json.loads(route.calls.last.request.content) == {
        "model": "command-a-03-2025",
        "messages": messages,
        "temperature": 0.7,
        "p": 0.9,
        "stop_sequences": ["END", "STOP"],
        "max_tokens": 50,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.1,
        "seed": 42,
    }


@pytest.mark.usefixtures("_cohere_httpx_transport")
@pytest.mark.respx(assert_all_called=True)
def test_cohere_v2_error_handling(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(COHERE_V2_CHAT_URL).mock(
        return_value=httpx.Response(
            400, json={"id": "7f0c", "message": "invalid request: model 'invalid-model' not found"}
        )
    )

    with pytest.raises(litellm.BadRequestError) as error:
        litellm.completion(
            model="cohere_chat/v2/invalid-model",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10,
            num_retries=0,
            api_key="cohere-test-key",
        )

    assert error.value.status_code == 400
    assert error.value.llm_provider == "cohere"
    assert "invalid request: model 'invalid-model' not found" in error.value.message
