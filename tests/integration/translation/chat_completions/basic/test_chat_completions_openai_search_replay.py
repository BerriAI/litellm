import json
from collections.abc import Callable
from typing import Final

from integration._support.client import Gateway
from integration._support.openai_wire import responses_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

TOKEN: Final = "scripted-openai-search-key"
SEARCH_ANNOTATION: Final[dict[str, JsonValue]] = {
    "type": "url_citation",
    "url_citation": {
        "start_index": 0,
        "end_index": 11,
        "url": "https://news.example.test/positive-story",
        "title": "Positive news",
    },
}


def _responses_peer(identity: str, model: str, text: str) -> Callable[[Request], Reply]:
    def respond(_: Request) -> Reply:
        return responses_reply(identity, model, text, stream=True)

    return respond


def _single_upstream_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1
    return received[0]


def test_gpt_5_reasoning_streaming_sends_reasoning_effort(gateway: Gateway) -> None:
    provider_body: Final = {
        "model": "gpt-5-mini",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Think of a poem, and then write it."}],
            }
        ],
        "reasoning": {"effort": "low"},
        "stream": True,
    }

    with wire_server(_responses_peer("reasoning-stream", "gpt-5-mini", "A small poem.")) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/responses/gpt-5-mini",
                api_key=TOKEN,
                api_base=wire.url,
            )
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Think of a poem, and then write it."}],
                    "reasoning_effort": "low",
                    "stream": True,
                    "cache": {"no-cache": True},
                },
            )
            request: Final = _single_upstream_request(wire)

    assert response.status_code == 200, response.text
    assert (request.method, request.target) == ("POST", "/responses")
    assert json.loads(request.body) == provider_body
    assert "A small poem." in response.text


def test_gpt_5_web_search_maps_builtin_tool(gateway: Gateway) -> None:
    provider_body: Final = {
        "model": "gpt-5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "get price of nvda"}],
            }
        ],
        "temperature": 1,
        "max_output_tokens": 8192,
        "tools": [{"type": "web_search"}],
        "stream": True,
    }

    with wire_server(_responses_peer("web-search", "gpt-5", "NVDA is listed at $123.")) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model="openai/responses/gpt-5", api_key=TOKEN, api_base=wire.url)
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "get price of nvda"}],
                    "stream": True,
                    "temperature": 1,
                    "max_tokens": 8192,
                    "tools": [{"type": "web_search"}],
                    "cache": {"no-cache": True},
                },
            )
            request: Final = _single_upstream_request(wire)

    assert response.status_code == 200, response.text
    assert (request.method, request.target) == ("POST", "/responses")
    assert json.loads(request.body) == provider_body
    assert "NVDA is listed at $123." in response.text


def test_openai_web_search_response_preserves_url_citation(gateway: Gateway) -> None:
    provider_body: Final = {
        "model": "gpt-5-search-api",
        "messages": [{"role": "user", "content": "What was a positive news story from today?"}],
    }
    provider_reply: Final = {
        "id": "chatcmpl-search",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5-search-api",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "A positive story was reported today.",
                    "annotations": [SEARCH_ANNOTATION],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
    }

    def search_peer(_: Request) -> Reply:
        return Reply(body=json.dumps(provider_reply).encode())

    with wire_server(search_peer) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model="openai/gpt-5-search-api", api_key=TOKEN, api_base=wire.url)
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "What was a positive news story from today?"}],
                    "cache": {"no-cache": True},
                },
            )
            request: Final = _single_upstream_request(wire)

    assert response.status_code == 200, response.text
    assert (request.method, request.target) == ("POST", "/chat/completions")
    assert json.loads(request.body) == provider_body
    annotations: Final = response.json()["choices"][0]["message"]["annotations"]
    assert annotations == [SEARCH_ANNOTATION]


def test_openai_web_search_streaming_preserves_url_citation(gateway: Gateway) -> None:
    provider_body: Final = {
        "model": "gpt-5-search-api",
        "messages": [{"role": "user", "content": "What was a positive news story from today?"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    stream_body: Final = b"".join(
        (
            b"data: "
            + json.dumps(
                {
                    "id": "chatcmpl-search-stream",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-5-search-api",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "A positive story.",
                                "annotations": [SEARCH_ANNOTATION],
                            },
                            "finish_reason": None,
                        }
                    ],
                }
            ).encode()
            + b"\n\n",
            b'data: {"id":"chatcmpl-search-stream","object":"chat.completion.chunk","created":1,'
            b'"model":"gpt-5-search-api","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        )
    )

    def search_peer(_: Request) -> Reply:
        return Reply(content_type="text/event-stream", body=stream_body)

    with wire_server(search_peer) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model="openai/gpt-5-search-api", api_key=TOKEN, api_base=wire.url)
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "What was a positive news story from today?"}],
                    "stream": True,
                    "cache": {"no-cache": True},
                },
            )
            request: Final = _single_upstream_request(wire)

    assert response.status_code == 200, response.text
    assert (request.method, request.target) == ("POST", "/chat/completions")
    assert json.loads(request.body) == provider_body
    chunks: Final = tuple(
        json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: {")
    )
    annotations: Final = chunks[0]["choices"][0]["delta"]["annotations"]
    assert annotations == [SEARCH_ANNOTATION]


def test_openai_codex_stream_uses_responses_endpoint(gateway: Gateway) -> None:
    provider_body: Final = {
        "model": "gpt-5.3-codex",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hey!"}],
            }
        ],
        "stream": True,
    }

    with wire_server(_responses_peer("codex-stream", "gpt-5.3-codex", "Hello.")) as wire:
        with gateway.scenario() as scenario:
            model: Final = scenario.model(model="openai/gpt-5.3-codex", api_key=TOKEN, api_base=wire.url)
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hey!"}],
                    "stream": True,
                    "cache": {"no-cache": True},
                },
            )
            request: Final = _single_upstream_request(wire)

    assert response.status_code == 200, response.text
    assert (request.method, request.target) == ("POST", "/responses")
    assert json.loads(request.body) == provider_body
    assert "Hello." in response.text
