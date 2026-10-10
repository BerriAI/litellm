"""
Tests that client headers are forwarded to the provider on the OpenAI
text completion path.

Regression tests for https://github.com/BerriAI/litellm/issues/27410
"""


import json
from typing import Final

import pytest
import respx
from httpx import Response


import litellm
from litellm import atext_completion, text_completion


@pytest.fixture(autouse=True)
def setup_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")


@pytest.fixture
def mock_completions_endpoint():
    return respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "cmpl-test123",
                "object": "text_completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [
                    {
                        "text": "hi",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )
    )


@respx.mock
def test_completion_forwards_client_headers_to_provider(mock_completions_endpoint):
    text_completion(
        model="gpt-3.5-turbo-instruct",
        prompt="hello",
        max_tokens=5,
        headers={"x-mycorp-llmcall-id": "abc-123"},
    )

    request_headers = mock_completions_endpoint.calls.last.request.headers
    assert request_headers["x-mycorp-llmcall-id"] == "abc-123"


@respx.mock
def test_completion_forwards_extra_headers_to_provider(mock_completions_endpoint):
    text_completion(
        model="gpt-3.5-turbo-instruct",
        prompt="hello",
        max_tokens=5,
        extra_headers={"x-mycorp-llmcall-id": "abc-123"},
    )

    request_headers = mock_completions_endpoint.calls.last.request.headers
    assert request_headers["x-mycorp-llmcall-id"] == "abc-123"


@respx.mock
async def test_acompletion_forwards_client_headers_to_provider(
    mock_completions_endpoint, monkeypatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    await atext_completion(
        model="gpt-3.5-turbo-instruct",
        prompt="hello",
        max_tokens=5,
        headers={"x-mycorp-llmcall-id": "abc-123"},
    )

    request_headers = mock_completions_endpoint.calls.last.request.headers
    assert request_headers["x-mycorp-llmcall-id"] == "abc-123"


@pytest.fixture
def rejected_completions_endpoint():
    return respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            400,
            json={"error": {"message": "bad prompt", "type": "invalid_request_error"}},
            headers={"x-request-id": "req-1"},
        )
    )


@respx.mock
def test_completion_surfaces_the_status_and_headers_of_a_rejected_request(rejected_completions_endpoint):
    with pytest.raises(litellm.BadRequestError) as rejected:
        text_completion(model="gpt-3.5-turbo-instruct", prompt="hello")

    assert rejected.value.status_code == 400
    assert rejected.value.litellm_response_headers["x-request-id"] == "req-1"


@respx.mock
def test_streaming_completion_surfaces_the_status_and_headers_of_a_rejected_request(rejected_completions_endpoint):
    with pytest.raises(litellm.BadRequestError) as rejected:
        list(text_completion(model="gpt-3.5-turbo-instruct", prompt="hello", stream=True))

    assert rejected.value.status_code == 400
    assert rejected.value.litellm_response_headers["x-request-id"] == "req-1"


@respx.mock
async def test_acompletion_surfaces_the_status_and_headers_of_a_rejected_request(
    rejected_completions_endpoint, monkeypatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with pytest.raises(litellm.BadRequestError) as rejected:
        await atext_completion(model="gpt-3.5-turbo-instruct", prompt="hello")

    assert rejected.value.status_code == 400
    assert rejected.value.litellm_response_headers["x-request-id"] == "req-1"


@respx.mock
async def test_async_streaming_completion_reports_an_error_event_sent_mid_stream(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            200,
            text='data: {"error": {"message": "upstream overloaded", "type": "server_error"}}\n\n',
            headers={"content-type": "text/event-stream"},
        )
    )
    with pytest.raises(litellm.InternalServerError) as failed:
        async for _ in await atext_completion(model="gpt-3.5-turbo-instruct", prompt="hello", stream=True):
            pass

    assert failed.value.status_code == 500
    assert "upstream overloaded" in str(failed.value)


@respx.mock
def test_completion_openai_prompt_array_sends_both_prompts() -> None:
    route: Final = respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "cmpl-test123",
                "object": "text_completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [
                    {"text": "first answer", "index": 0, "logprobs": None, "finish_reason": "stop"},
                    {"text": "second answer", "index": 1, "logprobs": None, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )
    )

    response: Final = text_completion(
        model="gpt-3.5-turbo-instruct",
        prompt=["first prompt", "second prompt"],
        max_tokens=5,
    )

    body: Final = json.loads(route.calls.last.request.read())
    assert body["prompt"] == ["first prompt", "second prompt"]
    assert [choice.text for choice in response.choices] == ["first answer", "second answer"]


@respx.mock
def test_completion_text_003_token_prompt_array_returns_one_choice_per_prompt() -> None:
    token_prompts: Final = [[2061, 338, 262, 6193, 287, 14362, 30], [2437, 318, 9502, 30]]
    route: Final = respx.post("https://api.fireworks.ai/inference/v1/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "cmpl-token-prompts",
                "object": "text_completion",
                "created": 1677652288,
                "model": "accounts/fireworks/models/glm-5p3-flash",
                "choices": [
                    {"text": "sunny", "index": 0, "logprobs": None, "finish_reason": "length"},
                    {"text": "rainy", "index": 1, "logprobs": None, "finish_reason": "length"},
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            },
        )
    )

    response: Final = text_completion(
        model="text-completion-openai/accounts/fireworks/models/glm-5p3-flash",
        api_base="https://api.fireworks.ai/inference/v1",
        api_key="fireworks-test-key",
        prompt=token_prompts,
        max_tokens=5,
    )

    body: Final = json.loads(route.calls.last.request.read())
    assert body["prompt"] == token_prompts
    assert body["model"] == "accounts/fireworks/models/glm-5p3-flash"
    assert [(choice.index, choice.text) for choice in response.choices] == [(0, "sunny"), (1, "rainy")]


@respx.mock
def test_text_completion_with_echo_returns_prompt_and_token_logprobs() -> None:
    route: Final = respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "cmpl-test123",
                "object": "text_completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [
                    {
                        "text": "hello world",
                        "index": 0,
                        "logprobs": {"tokens": ["hello", " world"], "token_logprobs": [-0.1, -0.2]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    response: Final = text_completion(
        model="gpt-3.5-turbo-instruct",
        prompt="hello",
        max_tokens=1,
        echo=True,
        logprobs=1,
    )

    body: Final = json.loads(route.calls.last.request.read())
    assert body["echo"] is True
    assert response.choices[0].text == "hello world"
    assert response.choices[0].logprobs.token_logprobs == [-0.1, -0.2]


@respx.mock
def test_completion_openai_engine_selects_the_engine_model(mock_completions_endpoint) -> None:
    response: Final = text_completion(engine="gpt-3.5-turbo-instruct", prompt="hello", max_tokens=5)

    request_body: Final = json.loads(mock_completions_endpoint.calls.last.request.read())
    assert request_body["model"] == "gpt-3.5-turbo-instruct"
    assert response.model == "gpt-3.5-turbo-instruct"


@respx.mock
def test_completion_openai_engine_and_model_uses_the_explicit_model(mock_completions_endpoint) -> None:
    response: Final = text_completion(
        model="gpt-3.5-turbo-instruct",
        engine="unused-engine",
        prompt="hello",
        max_tokens=5,
    )

    request_body: Final = json.loads(mock_completions_endpoint.calls.last.request.read())
    assert request_body["model"] == "gpt-3.5-turbo-instruct"
    assert response.model == "gpt-3.5-turbo-instruct"


@respx.mock
async def test_async_text_completion_stream_returns_text_and_one_finish_reason(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            200,
            text=(
                'data: {"id":"cmpl-stream","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"index":0,"text":"hello","finish_reason":null}]}\n\n'
                'data: {"id":"cmpl-stream","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"index":0,"text":"","finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    stream: Final = await atext_completion(
        model="gpt-3.5-turbo-instruct", prompt="say hello", stream=True, max_tokens=5
    )
    chunks: Final = [chunk async for chunk in stream]

    assert [chunk.choices[0].text for chunk in chunks if chunk.choices[0].text] == ["hello"]
    assert tuple(
        chunk.choices[0].finish_reason
        for chunk in chunks
        if chunk.choices[0].finish_reason is not None
    ) == ("stop",)
    assert len(route.calls) == 1


@respx.mock
async def test_async_text_completion_chat_model_stream_builds_chat_chunks(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=Response(
            200,
            text=(
                'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}\n\n'
                'data: {"id":"chatcmpl-stream","object":"chat.completion.chunk","created":1,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    stream: Final = await atext_completion(
        model="gpt-3.5-turbo", prompt="say hello", stream=True, max_tokens=5
    )
    chunks: Final = [chunk async for chunk in stream]
    response: Final = litellm.stream_chunk_builder(chunks=chunks)

    assert response.choices[0].text == "hello"
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (0, 1)
    assert litellm.completion_cost(
        completion_response=response,
        custom_cost_per_token={"input_cost_per_token": 5e-07, "output_cost_per_token": 1.5e-06},
    ) == pytest.approx(1 * 1.5e-06)
    assert tuple(
        chunk.choices[0].finish_reason
        for chunk in chunks
        if chunk.choices[0].finish_reason is not None
    ) == ("stop",)
    assert len(route.calls) == 1


@respx.mock
def test_text_completion_stream_forwards_stream_options_and_usage(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx.post("https://api.openai.com/v1/completions").mock(
        return_value=Response(
            200,
            text=(
                'data: {"id":"cmpl-stream","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[{"index":0,"text":"hello","finish_reason":null}]}\n\n'
                'data: {"id":"cmpl-stream","object":"text_completion","created":1,"model":"gpt-3.5-turbo-instruct","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
                "data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )
    )

    chunks: Final = list(
        text_completion(
            model="gpt-3.5-turbo-instruct",
            prompt="say hello",
            stream=True,
            stream_options={"include_usage": True},
        )
    )

    assert tuple(chunk.choices[0].text for chunk in chunks) == ("hello", None, None)
    assert tuple(chunk.usage for chunk in chunks[:-1]) == (None, None)
    assert (chunks[-1].usage.prompt_tokens, chunks[-1].usage.completion_tokens, chunks[-1].usage.total_tokens) == (
        2,
        1,
        3,
    )
    assert json.loads(route.calls[0].request.content)["stream_options"] == {
        "include_usage": True
    }
