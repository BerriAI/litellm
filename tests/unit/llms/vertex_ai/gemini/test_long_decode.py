import json
import sys
from collections.abc import Callable, Mapping, Sequence
from itertools import chain, repeat
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.vertex_ai.gemini.long_decode import (
    collect_hops_async,
    collect_hops_sync,
    initial_prompt_tokens_of,
    with_initial_prompt_tokens,
)
from litellm.types.llms.vertex_ai import RequestBody
from litellm.types.utils import Usage

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
VERTEX_PUBLISHER_BASE: Final = "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google"

HOP_1_SEARCHES: Final = ("argon launch date", "argon output limit", "argon pricing")
HOP_2_SEARCHES: Final = ("argon pricing", "argon context window")
STREAM_HOP_1_END: Final = {
    "candidates": [{**HOP_1["candidates"][0], "content": {"role": "model", "parts": [{"text": "lo"}]}}],
    "usageMetadata": HOP_1["usageMetadata"],
}


def _ending_with(body: Mapping[str, object], finish_reason: str) -> dict[str, object]:
    candidates: Final = body["candidates"]
    assert isinstance(candidates, list)
    return {**body, "candidates": [{**candidates[0], "finishReason": finish_reason}, *candidates[1:]]}


def _searching(body: Mapping[str, object], queries: Sequence[str]) -> Mapping[str, object]:
    if not queries:
        return body
    candidates: Final = body["candidates"]
    assert isinstance(candidates, list)
    grounded: Final = {**candidates[0], "groundingMetadata": {"webSearchQueries": list(queries)}}
    return {**body, "candidates": [grounded, *candidates[1:]]}


def _replay(hops: Sequence[httpx.Response], sent: list[dict[str, object]]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return hops[len(sent) - 1]

    return handler


def _recording_paths(hops: Sequence[httpx.Response], paths: list[str]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return hops[len(paths) - 1]

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
    assert response.usage.prompt_tokens_details.initial_prompt_tokens == 10


@pytest.mark.parametrize("hop_2_searches", [(), HOP_2_SEARCHES])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_unary_long_decode_bills_the_searches_of_every_hop(is_async: bool, hop_2_searches: Sequence[str]):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        httpx.Response(200, json=_searching(HOP_1, HOP_1_SEARCHES)),
        httpx.Response(200, json=_searching(HOP_2, hop_2_searches)),
    )
    kwargs: Final = {"model": MODEL, "messages": list(MESSAGES), "max_tokens": 100_000, "api_key": "fake-gemini-key"}

    response: Final = (
        await litellm.acompletion(**kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent))))
        if is_async
        else litellm.completion(
            **kwargs, client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_replay(hops, sent))))
        )
    )

    assert len(sent) == 2
    assert response.choices[0].message.content == "hello world"
    assert response.usage.prompt_tokens_details.web_search_requests == len({*HOP_1_SEARCHES, *hop_2_searches})
    assert [item["webSearchQueries"] for item in response.vertex_ai_grounding_metadata] == [
        list(searches) for searches in (HOP_1_SEARCHES, hop_2_searches) if searches
    ]


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
    assert response.usage.prompt_tokens_details.initial_prompt_tokens == (10 if expected_requests > 1 else None)


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


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_unary_long_decode_follows_more_hops_than_the_interpreter_allows_nested_calls(is_async: bool):
    hop_count: Final = sys.getrecursionlimit() + 50
    one_token_hop: Final = {**HOP_1, "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1}}
    responses: Final = chain(
        repeat(httpx.Response(200, json=one_token_hop), hop_count - 2),
        (httpx.Response(200, json=HOP_2),),
    )
    request_body: Final = RequestBody(contents=[], generationConfig={"max_output_tokens": hop_count * 100})

    async def post_async(_body: RequestBody) -> httpx.Response:
        return next(responses)

    hops: Final = (
        await collect_hops_async(one_token_hop, request_body, post_async)
        if is_async
        else collect_hops_sync(one_token_hop, request_body, lambda _body: next(responses))
    )

    assert len(hops) == hop_count
    assert hops[-1] == HOP_2


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
            _ending_with(STREAM_HOP_1_END, hop_end),
        ),
        _sse(HOP_2),
    )


async def _stream_chunks(
    is_async: bool,
    handler: Callable[[httpx.Request], httpx.Response],
    api_base: str | None = None,
    continuation_token: str | None = None,
) -> list:
    kwargs: Final = {
        "model": MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 100_000,
        "stream": True,
        "stream_options": {"include_usage": True},
        "api_key": "fake-gemini-key",
        "api_base": api_base,
        "continuation_token": continuation_token,
    }
    if is_async:
        stream: Final = await litellm.acompletion(
            **kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(handler))
        )
        return [chunk async for chunk in stream]
    return list(
        litellm.completion(**kwargs, client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler))))
    )


async def _collect_stream(is_async: bool, hops: Sequence[httpx.Response], sent: list[dict[str, object]]) -> list:
    return await _stream_chunks(is_async, _replay(hops, sent))


def _stream_text(chunks: list) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)


def _stream_finish_reasons(chunks: list) -> list[str]:
    return [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]


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
    assert chunks[-1].usage.prompt_tokens_details.initial_prompt_tokens == 10


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_long_decode_reads_every_chunk_of_each_continuation_stream(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        _stream_hops()[0],
        _sse(
            {
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": " wor"}]}, "continuationToken": "Y2twdC0x"}
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            },
            {
                **HOP_2_TRUNCATED,
                "candidates": [{**HOP_2_TRUNCATED["candidates"][0], "content": {"parts": [{"text": "ld"}]}}],
            },
        ),
        _sse(
            {"candidates": [{"content": {"role": "model", "parts": [{"text": " ag"}]}}]},
            {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "ain"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 7, "totalTokenCount": 17},
            },
        ),
    )

    chunks: Final = await _collect_stream(is_async, hops, sent)

    assert [body.get("continuationToken") for body in sent] == [None, "dG9rLTE=", "dG9rLTI="]
    assert _stream_text(chunks) == "hello world again"
    assert _stream_finish_reasons(chunks) == ["stop"]
    assert chunks[-1].usage.prompt_tokens == 30
    assert chunks[-1].usage.completion_tokens == 57


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_initial_prompt_tokens_is_the_prompt_count_the_first_stream_ends_with(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    first_stream: Final = _sse(
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hel"}]}}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10},
        },
        STREAM_HOP_1_END,
    )

    chunks: Final = await _collect_stream(is_async, (first_stream, _sse(HOP_2)), sent)

    assert chunks[-1].usage.prompt_tokens_details.initial_prompt_tokens == 10
    assert chunks[-1].usage.prompt_tokens == 20


def _api_versions(paths: Sequence[str]) -> list[str]:
    return [path.split("/projects/")[0] for path in paths]


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_unary_continuation_request_to_a_vertex_url_goes_to_the_api_version_that_accepts_the_token(
    is_async: bool,
):
    paths: Final[list[str]] = []
    hops: Final = (httpx.Response(200, json=HOP_1), httpx.Response(200, json=HOP_2))
    transport: Final = httpx.MockTransport(_recording_paths(hops, paths))
    kwargs: Final = {
        "model": MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 100_000,
        "api_key": "fake-gemini-key",
        "api_base": VERTEX_PUBLISHER_BASE,
    }

    response: Final = (
        await litellm.acompletion(**kwargs, client=AsyncHTTPHandler(transport=transport))
        if is_async
        else litellm.completion(**kwargs, client=HTTPHandler(client=httpx.Client(transport=transport)))
    )

    assert response.choices[0].message.content == "hello world"
    # Vertex AI answered 400 for continuationToken on v1 and 200 on v1beta1, probed at location global on 2026-10-13
    assert _api_versions(paths) == ["/v1", "/v1beta1"]


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_continuation_request_to_a_vertex_url_goes_to_the_api_version_that_accepts_the_token(
    is_async: bool,
):
    paths: Final[list[str]] = []

    chunks: Final = await _stream_chunks(is_async, _recording_paths(_stream_hops(), paths), VERTEX_PUBLISHER_BASE)

    assert _stream_text(chunks) == "hello world"
    assert _api_versions(paths) == ["/v1", "/v1beta1"]


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_request_the_client_continues_on_a_vertex_url_goes_to_the_api_version_that_accepts_the_token(
    is_async: bool,
):
    paths: Final[list[str]] = []

    chunks: Final = await _stream_chunks(
        is_async, _recording_paths((_sse(HOP_2),), paths), VERTEX_PUBLISHER_BASE, continuation_token="dG9rLTE="
    )

    assert _stream_text(chunks) == " world"
    assert _api_versions(paths) == ["/v1beta1"]


@pytest.mark.parametrize("hop_2_searches", [(), HOP_2_SEARCHES])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_long_decode_bills_the_searches_of_every_hop(is_async: bool, hop_2_searches: Sequence[str]):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        _sse(_searching(STREAM_HOP_1_END, HOP_1_SEARCHES)),
        _sse(_searching(HOP_2, hop_2_searches)),
    )

    chunks: Final = await _collect_stream(is_async, hops, sent)

    assert len(sent) == 2
    assert chunks[-1].usage.prompt_tokens_details.web_search_requests == len({*HOP_1_SEARCHES, *hop_2_searches})


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


def test_initial_prompt_tokens_is_absent_on_a_single_hop_and_the_first_hops_count_otherwise():
    assert initial_prompt_tokens_of((HOP_1,)) is None
    assert initial_prompt_tokens_of((HOP_1, HOP_2)) == 10


def test_with_initial_prompt_tokens_adds_the_details_block_and_keeps_the_counts():
    usage: Final = Usage(prompt_tokens=20, completion_tokens=50, total_tokens=70)

    marked: Final = with_initial_prompt_tokens(usage, 10)

    assert marked.prompt_tokens_details.initial_prompt_tokens == 10
    assert (marked.prompt_tokens, marked.completion_tokens, marked.total_tokens) == (20, 50, 70)
    assert usage.prompt_tokens_details is None
