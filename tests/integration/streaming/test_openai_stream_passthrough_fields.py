import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

import httpx
import openai

from tests.integration._support.client import Gateway
from tests.integration._support.wire import Reply, Request, wire_server

_CHAT_CHUNKS: Final = (
    {
        "id": "chatcmpl-logprobs",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "hello"},
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": [
                                {"token": "hello", "logprob": -0.1, "bytes": [104, 101, 108, 108, 111]},
                                {"token": "hi", "logprob": -2.3, "bytes": [104, 105]},
                            ],
                        }
                    ]
                },
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "chatcmpl-logprobs",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {}, "logprobs": None, "finish_reason": "stop"}],
    },
)
_COMPLETION_CHUNKS: Final = (
    {
        "id": "cmpl-usage",
        "object": "text_completion",
        "created": 1,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{"text": "done", "index": 0, "logprobs": None, "finish_reason": None}],
    },
    {
        "id": "cmpl-usage",
        "object": "text_completion",
        "created": 1,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}],
    },
    {
        "id": "cmpl-usage",
        "object": "text_completion",
        "created": 1,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    },
)


def _event_stream(chunks: tuple[dict[str, object], ...]) -> Reply:
    return Reply(
        chunks=(*(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks), b"data: [DONE]\n\n"),
        content_type="text/event-stream",
    )


@contextmanager
def _client(gateway: Gateway) -> Iterator[openai.OpenAI]:
    with (
        httpx.Client(timeout=15, trust_env=False) as transport,
        openai.OpenAI(
            api_key=gateway.key, base_url=str(gateway.client.base_url.join("/v1")), max_retries=0, http_client=transport
        ) as client,
    ):
        yield client


def test_chat_stream_relays_the_provider_logprobs(gateway: Gateway) -> None:
    def upstream(request: Request) -> Reply:
        return _event_stream(_CHAT_CHUNKS)

    with wire_server(upstream) as provider, gateway.scenario() as scenario, _client(gateway) as client:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=f"{provider.url}/v1")
        chunks: Final = tuple(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello!"}],
                logprobs=True,
                top_logprobs=2,
                stream=True,
            )
        )
        sent: Final = provider.drain()

    assert len(sent) == 1
    outbound: Final = json.loads(sent[0].body)
    assert (outbound["logprobs"], outbound["top_logprobs"], outbound["stream"]) == (True, 2, True)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "hello"
    relayed: Final = tuple(
        chunk.choices[0].logprobs.model_dump(exclude_none=True)
        for chunk in chunks
        if chunk.choices and chunk.choices[0].logprobs is not None
    )
    assert relayed == (_CHAT_CHUNKS[0]["choices"][0]["logprobs"],)


def test_completion_stream_reports_the_provider_usage_when_include_usage_is_set(gateway: Gateway) -> None:
    def upstream(request: Request) -> Reply:
        return _event_stream(_COMPLETION_CHUNKS)

    with wire_server(upstream) as provider, gateway.scenario() as scenario, _client(gateway) as client:
        model: Final = scenario.model(model="openai/gpt-3.5-turbo-instruct", api_base=f"{provider.url}/v1")
        chunks: Final = tuple(
            client.completions.create(
                model=model,
                prompt="hey",
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=4,
            )
        )
        sent: Final = provider.drain()

    assert len(sent) == 1
    outbound: Final = json.loads(sent[0].body)
    assert (outbound["prompt"], outbound["stream_options"], outbound["max_tokens"]) == (
        "hey",
        {"include_usage": True},
        4,
    )
    assert "".join(choice.text or "" for chunk in chunks for choice in chunk.choices) == "done"
    usages: Final = tuple(
        (chunk.usage.prompt_tokens, chunk.usage.completion_tokens, chunk.usage.total_tokens)
        for chunk in chunks
        if chunk.usage is not None
    )
    assert usages == ((7, 3, 10),)
