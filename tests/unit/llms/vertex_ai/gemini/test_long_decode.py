import json
from collections.abc import Callable, Mapping, Sequence
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

MODEL: Final = "gemini/gemini-3.8-flash"
MESSAGES: Final = ({"role": "user", "content": "Prove it"},)

HOP_1: Final = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "hello"}]},
            "finishReason": "MAX_TOKENS",
            "continuationToken": "dG9rLTE=",
        }
    ],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 30, "totalTokenCount": 40},
}
HOP_2: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": " world"}]}, "finishReason": "STOP"}],
    "usageMetadata": {
        "promptTokenCount": 10,
        "cachedContentTokenCount": 9,
        "candidatesTokenCount": 20,
        "totalTokenCount": 30,
    },
}
HOP_2_TRUNCATED: Final = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": " world"}]},
            "finishReason": "MAX_TOKENS",
            "continuationToken": "dG9rLTI=",
        }
    ],
    "usageMetadata": HOP_2["usageMetadata"],
}


HOP_END_FINISH_REASONS: Final = ("MAX_TOKENS", "CONTINUATION")


def _ending_with(body: Mapping[str, object], finish_reason: str) -> dict[str, object]:
    candidates: Final = body["candidates"]
    assert isinstance(candidates, list)
    return {**body, "candidates": [{**candidates[0], "finishReason": finish_reason}, *candidates[1:]]}


def _replay(hops: Sequence[httpx.Response], sent: list[dict[str, object]]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return hops[len(sent) - 1]

    return handler


def _sse(*chunks: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        content="".join(f"data: {json.dumps(chunk)}\r\n\r\n" for chunk in chunks).encode(),
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize("hop_end", HOP_END_FINISH_REASONS)
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_unary_long_decode_resends_the_original_request_with_the_token_and_bills_every_hop(
    is_async: bool, hop_end: str
):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (httpx.Response(200, json=_ending_with(HOP_1, hop_end)), httpx.Response(200, json=HOP_2))
    kwargs: Final = {"model": MODEL, "messages": list(MESSAGES), "max_tokens": 100_000, "api_key": "fake-gemini-key"}

    response: Final = (
        await litellm.acompletion(**kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent))))
        if is_async
        else litellm.completion(
            **kwargs, client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_replay(hops, sent))))
        )
    )

    assert sent[1] == {**sent[0], "continuationToken": "dG9rLTE="}
    assert response.choices[0].message.content == "hello world"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens == 20
    assert response.usage.completion_tokens == 50
    assert response.usage.prompt_tokens_details.cached_tokens == 9


@pytest.mark.parametrize("hop_end", HOP_END_FINISH_REASONS)
@pytest.mark.parametrize(
    "max_tokens, expected_requests, expected_token",
    [(None, 1, "dG9rLTE="), (30, 1, "dG9rLTE="), (50, 2, "dG9rLTI=")],
)
@pytest.mark.asyncio
async def test_unary_long_decode_stops_at_the_callers_cap_and_returns_the_token_to_resume(
    max_tokens: int | None, expected_requests: int, expected_token: str, hop_end: str
):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        httpx.Response(200, json=_ending_with(HOP_1, hop_end)),
        httpx.Response(200, json=_ending_with(HOP_2_TRUNCATED, hop_end)),
    )

    response: Final = await litellm.acompletion(
        model=MODEL,
        messages=list(MESSAGES),
        max_tokens=max_tokens,
        api_key="fake-gemini-key",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent))),
    )

    assert len(sent) == expected_requests
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.provider_specific_fields["continuation_token"] == expected_token


@pytest.mark.asyncio
async def test_unary_failed_continuation_hop_returns_the_answer_so_far_with_its_token():
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (httpx.Response(200, json=HOP_1), httpx.Response(429, json={"error": {"code": 429}}))

    response: Final = await litellm.acompletion(
        model=MODEL,
        messages=list(MESSAGES),
        max_tokens=100_000,
        api_key="fake-gemini-key",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent))),
    )

    assert len(sent) == 2
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.provider_specific_fields["continuation_token"] == "dG9rLTE="
    assert response.usage.completion_tokens == 30


@pytest.mark.asyncio
async def test_continuation_token_param_is_sent_at_the_top_level_of_generate_content():
    sent: Final[list[dict[str, object]]] = []

    await litellm.acompletion(
        model=MODEL,
        messages=list(MESSAGES),
        continuation_token="dG9rLTE=",
        api_key="fake-gemini-key",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay((httpx.Response(200, json=HOP_2),), sent))),
    )

    assert sent[0]["continuationToken"] == "dG9rLTE="
    assert "continuation_token" not in json.dumps(sent[0])


def _stream_hops(hop_end: str = "MAX_TOKENS") -> tuple[httpx.Response, httpx.Response]:
    return (
        _sse(
            {"candidates": [{"content": {"role": "model", "parts": [{"text": "hel"}]}}]},
            _ending_with(
                {
                    "candidates": [{**HOP_1["candidates"][0], "content": {"role": "model", "parts": [{"text": "lo"}]}}],
                    "usageMetadata": HOP_1["usageMetadata"],
                },
                hop_end,
            ),
        ),
        _sse(HOP_2),
    )


async def _collect_stream(is_async: bool, hops: Sequence[httpx.Response], sent: list[dict[str, object]]) -> list:
    kwargs: Final = {
        "model": MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 100_000,
        "stream": True,
        "stream_options": {"include_usage": True},
        "api_key": "fake-gemini-key",
    }
    if is_async:
        stream: Final = await litellm.acompletion(
            **kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent)))
        )
        return [chunk async for chunk in stream]
    return list(
        litellm.completion(
            **kwargs, client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_replay(hops, sent))))
        )
    )


def _delta_fields(chunks: list) -> list[Mapping[str, object]]:
    return [
        choice.delta.provider_specific_fields
        for chunk in chunks
        for choice in chunk.choices  # comprehension-ok: flattening stream chunks into their choices
        if choice.delta.provider_specific_fields
    ]


@pytest.mark.parametrize("hop_end", HOP_END_FINISH_REASONS)
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_long_decode_chains_hops_into_one_stream_with_one_finish_and_summed_usage(
    is_async: bool, hop_end: str
):
    sent: Final[list[dict[str, object]]] = []

    chunks: Final = await _collect_stream(is_async, _stream_hops(hop_end), sent)

    assert sent[1] == {**sent[0], "continuationToken": "dG9rLTE="}
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "hello world"
    assert [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason] == ["stop"]
    assert not any("continuation_token" in fields for fields in _delta_fields(chunks))
    assert chunks[-1].usage.prompt_tokens == 20
    assert chunks[-1].usage.completion_tokens == 50


@pytest.mark.asyncio
async def test_streaming_checkpoint_token_reaches_the_client():
    sent: Final[list[dict[str, object]]] = []
    checkpoint: Final = {
        "candidates": [{"content": {"role": "model", "parts": [{"text": " wor"}]}, "continuationToken": "Y2twdC0x"}]
    }
    hops: Final = (_stream_hops()[0], _sse(checkpoint, HOP_2))

    chunks: Final = await _collect_stream(True, hops, sent)

    assert {"continuation_token": "Y2twdC0x"} in _delta_fields(chunks)


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_failed_continuation_hop_ends_the_stream_with_length_and_the_token(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (_stream_hops()[0], httpx.Response(500, json={"error": {"code": 500}}))

    chunks: Final = await _collect_stream(is_async, hops, sent)

    assert len(sent) == 2
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "hello"
    assert [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason] == ["length"]
    assert {"continuation_token": "dG9rLTE="} in _delta_fields(chunks)
