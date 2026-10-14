import json
import sys
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain, count, repeat
from typing import Final

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

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
            "finishReason": "CONTINUATION",
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
            "finishReason": "CONTINUATION",
            "continuationToken": "dG9rLTI=",
        }
    ],
    "usageMetadata": HOP_2["usageMetadata"],
}

TWO_CHOICES_CUT: Final = {
    "candidates": [{**HOP_1["candidates"][0], "index": 0}, {**HOP_2_TRUNCATED["candidates"][0], "index": 1}],
    "usageMetadata": HOP_1["usageMetadata"],
}
GOOGLE_REJECTION: Final = {"error": {"code": 400, "status": "INVALID_ARGUMENT"}}

VERTEX_PUBLISHER_BASE: Final = "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google"

HOP_1_SEARCHES: Final = ("argon launch date", "argon output limit", "argon pricing")
HOP_2_SEARCHES: Final = ("argon pricing", "argon context window")
STREAM_HOP_1_END: Final = {
    "candidates": [{**HOP_1["candidates"][0], "content": {"role": "model", "parts": [{"text": "lo"}]}}],
    "usageMetadata": HOP_1["usageMetadata"],
}


def _answered_as(body: Mapping[str, object], response_id: str) -> dict[str, object]:
    return {**body, "responseId": response_id}


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


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_unary_long_decode_resends_the_original_request_with_the_token_and_bills_every_hop(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (httpx.Response(200, json=HOP_1), httpx.Response(200, json=HOP_2))
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


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_unary_long_decode_answers_under_the_first_requests_response_id(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        httpx.Response(200, json=_answered_as(HOP_1, "resp-1")),
        httpx.Response(200, json=_answered_as(HOP_2, "resp-2")),
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
    assert response.id == "resp-1"


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("sent_as_a_string", [{"n": "1"}, {"max_tokens": "100000"}])
@pytest.mark.asyncio
async def test_request_whose_n_or_max_tokens_is_a_string_is_sent_once_and_never_continued(
    stream: bool, sent_as_a_string: Mapping[str, str]
):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        _sse(STREAM_FIRST_CHUNK, STREAM_HOP_1_END) if stream else httpx.Response(200, json=HOP_1),
        httpx.Response(400, json=GOOGLE_REJECTION),
    )
    kwargs: Final = {
        "model": MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 100_000,
        "stream": stream,
        "api_key": "fake-gemini-key",
        **sent_as_a_string,
    }

    response: Final = await litellm.acompletion(
        **kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent)))
    )
    text: Final = (
        "".join([chunk.choices[0].delta.content or "" async for chunk in response])
        if stream
        else response.choices[0].message.content
    )

    assert len(sent) == 1
    assert text == "hello"


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


HOP_1_GROUNDED: Final = "Kórnik Arboretum reached Level IV."
HOP_1_CITED: Final = " Python is easy to learn."
HOP_2_GROUNDED: Final = " The garden opened in 1826."
URL_CONTEXT: Final = {
    "urlMetadata": [
        {"retrievedUrl": "https://www.python.org/about/", "urlRetrievalStatus": "URL_RETRIEVAL_STATUS_SUCCESS"}
    ]
}


def _byte_span(text: str, part: str) -> tuple[int, int]:
    start: Final = text.encode().index(part.encode())
    return start, start + len(part.encode())


def _support(text: str, part: str) -> dict[str, object]:
    start, end = _byte_span(text, part)
    segment: Final = {"startIndex": start, "endIndex": end, "text": part} if start else {"endIndex": end, "text": part}
    return {"segment": segment, "groundingChunkIndices": [0]}


def _citation(text: str, part: str) -> dict[str, object]:
    start, end = _byte_span(text, part)
    return {"startIndex": start, "endIndex": end, "uri": f"https://example.com/{len(part)}"}


def _citing(
    body: Mapping[str, object], parts: Sequence[Mapping[str, object]], metadata: Mapping[str, object]
) -> dict[str, object]:
    candidates: Final = body["candidates"]
    assert isinstance(candidates, list)
    return {**body, "candidates": [{**candidates[0], "content": {"role": "model", "parts": list(parts)}, **metadata}]}


def _hops_citing_sources(citation_key: str, stream: bool = False) -> tuple[httpx.Response, httpx.Response]:
    thought: Final = {"text": "Planning the report", "thought": True}
    hop_1_text: Final = HOP_1_GROUNDED + HOP_1_CITED
    hop_1_metadata: Final = {
        "groundingMetadata": {
            "groundingChunks": [{"web": {"uri": "https://example.com/arboretum", "title": "arboretum"}}],
            "groundingSupports": [_support(hop_1_text, HOP_1_GROUNDED)],
        },
        "citationMetadata": {citation_key: [_citation(hop_1_text, HOP_1_CITED.strip())]},
        "urlContextMetadata": URL_CONTEXT,
    }
    hop_2: Final = _citing(
        HOP_2,
        [{"text": HOP_2_GROUNDED}],
        {
            "groundingMetadata": {
                "groundingChunks": [{"web": {"uri": "https://example.com/garden", "title": "garden"}}],
                "groundingSupports": [_support(HOP_2_GROUNDED, HOP_2_GROUNDED)],
            },
            "citationMetadata": {citation_key: [_citation(HOP_2_GROUNDED, "1826")]},
        },
    )
    if stream:
        opening: Final = {"candidates": [{"content": {"role": "model", "parts": [thought, {"text": HOP_1_GROUNDED}]}}]}
        return _sse(opening, _citing(HOP_1, [{"text": HOP_1_CITED}], hop_1_metadata)), _sse(hop_2)
    hop_1: Final = _citing(HOP_1, [thought, {"text": hop_1_text}], hop_1_metadata)
    return httpx.Response(200, json=hop_1), httpx.Response(200, json=hop_2)


def _supported_spans(grounding: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [support["segment"] for support in chain.from_iterable(item["groundingSupports"] for item in grounding)]


def _cited_spans(citation_metadata: Sequence[Mapping[str, object]], citation_key: str) -> list[Mapping[str, object]]:
    return list(chain.from_iterable(item[citation_key] for item in citation_metadata))


def _bytes_at(answer: str, span: Mapping[str, object]) -> str:
    return answer.encode()[span.get("startIndex", 0) : span["endIndex"]].decode()


@pytest.mark.parametrize("citation_key", ["citations", "citationSources"])
@pytest.mark.asyncio
async def test_unary_long_decode_keeps_every_hops_citations_and_points_them_into_the_whole_answer(citation_key: str):
    sent: Final[list[dict[str, object]]] = []

    response: Final = await litellm.acompletion(
        model=MODEL,
        messages=list(MESSAGES),
        max_tokens=100_000,
        api_key="fake-gemini-key",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(_hops_citing_sources(citation_key), sent))),
    )
    answer: Final = response.choices[0].message.content
    supported: Final = _supported_spans(response.vertex_ai_grounding_metadata)
    cited: Final = _cited_spans(response.vertex_ai_citation_metadata, citation_key)

    assert len(sent) == 2
    assert answer == HOP_1_GROUNDED + HOP_1_CITED + HOP_2_GROUNDED
    assert [_bytes_at(answer, span) for span in supported] == [span["text"] for span in supported]
    assert [span["text"] for span in supported] == [HOP_1_GROUNDED, HOP_2_GROUNDED]
    assert [_bytes_at(answer, span) for span in cited] == [HOP_1_CITED.strip(), "1826"]
    assert response.vertex_ai_url_context_metadata == [URL_CONTEXT]


NO_CITATIONS_OR_URL_CONTEXT: Final = (
    pytest.param({}, id="missing"),
    pytest.param({"citationMetadata": None, "urlContextMetadata": None}, id="null"),
    pytest.param({"citationMetadata": {}, "urlContextMetadata": {}}, id="empty"),
)


def _carrying(body: Mapping[str, object], fields: Mapping[str, object]) -> dict[str, object]:
    candidates: Final = body["candidates"]
    assert isinstance(candidates, list)
    return {**body, "candidates": [{**candidates[0], **fields}, *candidates[1:]]}


async def _answer(hops: Sequence[httpx.Response], sent: list[dict[str, object]]):
    return await litellm.acompletion(
        model=MODEL,
        messages=list(MESSAGES),
        max_tokens=100_000,
        api_key="fake-gemini-key",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent))),
    )


@pytest.mark.parametrize("fields", NO_CITATIONS_OR_URL_CONTEXT)
@pytest.mark.parametrize(
    "bodies, answer", [((HOP_2,), " world"), ((HOP_1, HOP_2), "hello world")], ids=["one_request", "continued"]
)
@pytest.mark.asyncio
async def test_unary_answer_whose_candidates_carry_no_citations_or_url_context_is_returned_whole(
    bodies: Sequence[Mapping[str, object]], answer: str, fields: Mapping[str, object]
):
    sent: Final[list[dict[str, object]]] = []

    response: Final = await _answer(tuple(httpx.Response(200, json=_carrying(body, fields)) for body in bodies), sent)

    assert len(sent) == len(bodies)
    assert response.choices[0].message.content == answer
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.parametrize("fields", NO_CITATIONS_OR_URL_CONTEXT)
@pytest.mark.asyncio
async def test_unary_long_decode_keeps_earlier_citations_and_url_context_when_a_follow_up_carries_none(
    fields: Mapping[str, object],
):
    sent: Final[list[dict[str, object]]] = []
    citation: Final = _citation("hello", "hello")
    hop_1: Final = _carrying(HOP_1, {"citationMetadata": {"citations": [citation]}, "urlContextMetadata": URL_CONTEXT})

    response: Final = await _answer(
        (httpx.Response(200, json=hop_1), httpx.Response(200, json=_carrying(HOP_2, fields))), sent
    )

    assert response.choices[0].message.content == "hello world"
    assert [item for item in response.vertex_ai_url_context_metadata if item != {}] == [URL_CONTEXT]
    assert _cited_spans([item for item in response.vertex_ai_citation_metadata if item != {}], "citations") == [
        citation
    ]


def _hop_2_spanning(span: Mapping[str, object]) -> dict[str, object]:
    return _carrying(
        HOP_2,
        {
            "groundingMetadata": {
                "groundingChunks": [{"web": {"uri": "https://example.com/world", "title": "world"}}],
                "groundingSupports": [{"segment": {**span, "text": " wo"}, "groundingChunkIndices": [0]}],
            },
            "citationMetadata": {"citations": [{**span, "uri": "https://example.com/world"}]},
        },
    )


def _chunk_metadata(chunks: Sequence[object], field: str) -> list[Mapping[str, object]]:
    return list(chain.from_iterable(getattr(chunk, field, None) or () for chunk in chunks))


async def _answer_with_spans(
    stream: bool, hop_2: Mapping[str, object], sent: list[dict[str, object]]
) -> tuple[str, list[Mapping[str, object]], list[Mapping[str, object]]]:
    if stream:
        chunks: Final = await _collect_stream(True, (_sse(STREAM_FIRST_CHUNK, STREAM_HOP_1_END), _sse(hop_2)), sent)
        return (
            _stream_text(chunks),
            _supported_spans(_chunk_metadata(chunks, "vertex_ai_grounding_metadata")),
            _cited_spans(_chunk_metadata(chunks, "vertex_ai_citation_metadata"), "citations"),
        )
    response: Final = await _answer((httpx.Response(200, json=HOP_1), httpx.Response(200, json=hop_2)), sent)
    return (
        response.choices[0].message.content,
        _supported_spans(response.vertex_ai_grounding_metadata),
        _cited_spans(response.vertex_ai_citation_metadata, "citations"),
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_long_decode_counts_a_follow_ups_null_start_index_from_the_start_of_its_text(stream: bool):
    sent: Final[list[dict[str, object]]] = []

    answer, supported, cited = await _answer_with_spans(
        stream, _hop_2_spanning({"startIndex": None, "endIndex": len(" wo")}), sent
    )

    assert answer == "hello world"
    assert [_bytes_at(answer, s) for s in supported] == [" wo"]
    assert [_bytes_at(answer, s) for s in cited] == [" wo"]


@pytest.mark.parametrize(
    "span",
    [
        pytest.param({}, id="no_offsets"),
        pytest.param({"startIndex": 1}, id="missing_end"),
        pytest.param({"startIndex": 1, "endIndex": None}, id="null_end"),
    ],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_long_decode_returns_a_follow_ups_span_with_no_end_index_as_google_sent_it(
    stream: bool, span: Mapping[str, object]
):
    sent: Final[list[dict[str, object]]] = []

    answer, supported, cited = await _answer_with_spans(stream, _hop_2_spanning(span), sent)

    assert answer == "hello world"
    assert supported == [{**span, "text": " wo"}]
    assert cited == [{**span, "uri": "https://example.com/world"}]


@pytest.mark.parametrize(
    "max_tokens, expected_requests, expected_token",
    [(None, 1, "dG9rLTE="), (30, 1, "dG9rLTE="), (50, 2, "dG9rLTI=")],
)
@pytest.mark.asyncio
async def test_unary_long_decode_stops_at_the_callers_cap_and_returns_the_token_to_resume(
    max_tokens: int | None, expected_requests: int, expected_token: str
):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (httpx.Response(200, json=HOP_1), httpx.Response(200, json=HOP_2_TRUNCATED))

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


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_unary_answer_cut_at_the_callers_cap_is_returned_with_its_token_in_one_request(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    # The Gemini API answered 400 "max_output_tokens (16000) has already been exhausted" to a resend
    # after MAX_TOKENS on 2026-10-13
    hops: Final = (
        httpx.Response(200, json=_ending_with(HOP_1, "MAX_TOKENS")),
        httpx.Response(400, json=GOOGLE_REJECTION),
    )
    kwargs: Final = {
        "model": MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 100_000,
        "continuation_token": "cHJldg==",
        "api_key": "fake-gemini-key",
    }

    response: Final = (
        await litellm.acompletion(**kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent))))
        if is_async
        else litellm.completion(
            **kwargs, client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_replay(hops, sent))))
        )
    )

    assert len(sent) == 1
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.provider_specific_fields["continuation_token"] == "dG9rLTE="


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_unary_request_for_more_than_one_choice_is_never_continued(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    # Vertex AI answered 400 "Candidate count greater than 1 is not supported when continuation_token is set"
    # on 2026-10-13
    hops: Final = (httpx.Response(200, json=TWO_CHOICES_CUT), httpx.Response(400, json=GOOGLE_REJECTION))
    kwargs: Final = {
        "model": MODEL,
        "messages": list(MESSAGES),
        "max_tokens": 100_000,
        "n": 2,
        "api_key": "fake-gemini-key",
    }

    response: Final = (
        await litellm.acompletion(**kwargs, client=AsyncHTTPHandler(transport=httpx.MockTransport(_replay(hops, sent))))
        if is_async
        else litellm.completion(
            **kwargs, client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_replay(hops, sent))))
        )
    )

    assert len(sent) == 1
    assert [choice.message.content for choice in response.choices] == ["hello", " world"]
    assert [choice.finish_reason for choice in response.choices] == ["length", "length"]
    assert [choice.message.provider_specific_fields["continuation_token"] for choice in response.choices] == [
        "dG9rLTE=",
        "dG9rLTI=",
    ]


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


STREAM_FIRST_CHUNK: Final = {"candidates": [{"content": {"role": "model", "parts": [{"text": "hel"}]}}]}


def _stream_hops() -> tuple[httpx.Response, httpx.Response]:
    return (_sse(STREAM_FIRST_CHUNK, STREAM_HOP_1_END), _sse(HOP_2))


async def _stream_chunks(
    is_async: bool,
    handler: Callable[[httpx.Request], httpx.Response],
    api_base: str | None = None,
    continuation_token: str | None = None,
    n: int | None = None,
    timeout: float | None = None,
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
        "n": n,
        "timeout": timeout,
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


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_long_decode_chains_hops_into_one_stream_with_one_finish_and_summed_usage(is_async: bool):
    sent: Final[list[dict[str, object]]] = []

    chunks: Final = await _collect_stream(is_async, _stream_hops(), sent)

    assert sent[1] == {**sent[0], "continuationToken": "dG9rLTE="}
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "hello world"
    assert [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason] == ["stop"]
    assert not any("continuation_token" in fields for fields in _delta_fields(chunks))
    assert chunks[-1].usage.prompt_tokens == 20
    assert chunks[-1].usage.completion_tokens == 50
    assert chunks[-1].usage.prompt_tokens_details.initial_prompt_tokens == 10


@pytest.mark.parametrize("fields", NO_CITATIONS_OR_URL_CONTEXT)
@pytest.mark.parametrize(
    "hop_chunks, answer",
    [(((HOP_2,),), " world"), (((STREAM_FIRST_CHUNK, STREAM_HOP_1_END), (HOP_2,)), "hello world")],
    ids=["one_request", "continued"],
)
@pytest.mark.asyncio
async def test_streamed_answer_whose_chunks_carry_no_citations_or_url_context_is_returned_whole(
    hop_chunks: Sequence[Sequence[Mapping[str, object]]], answer: str, fields: Mapping[str, object]
):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = tuple(_sse(*(_carrying(chunk, fields) for chunk in hop)) for hop in hop_chunks)

    chunks: Final = await _collect_stream(True, hops, sent)

    assert len(sent) == len(hop_chunks)
    assert _stream_text(chunks) == answer
    assert _stream_finish_reasons(chunks) == ["stop"]


@pytest.mark.parametrize("citation_key", ["citations", "citationSources"])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_long_decode_points_every_hops_citations_into_the_whole_answer(
    is_async: bool, citation_key: str
):
    sent: Final[list[dict[str, object]]] = []

    chunks: Final = await _collect_stream(is_async, _hops_citing_sources(citation_key, stream=True), sent)
    answer: Final = _stream_text(chunks)
    supported: Final = _supported_spans(_chunk_metadata(chunks, "vertex_ai_grounding_metadata"))
    cited: Final = _cited_spans(_chunk_metadata(chunks, "vertex_ai_citation_metadata"), citation_key)

    assert len(sent) == 2
    assert answer == HOP_1_GROUNDED + HOP_1_CITED + HOP_2_GROUNDED
    assert [span["text"] for span in supported] == [HOP_1_GROUNDED, HOP_2_GROUNDED]
    assert [_bytes_at(answer, span) for span in supported] == [HOP_1_GROUNDED, HOP_2_GROUNDED]
    assert [_bytes_at(answer, span) for span in cited] == [HOP_1_CITED.strip(), "1826"]


@pytest.mark.parametrize("next_request_fails", [False, True])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_long_decode_keeps_the_first_requests_response_id_on_every_chunk(
    is_async: bool, next_request_fails: bool
):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        _sse(_answered_as(STREAM_FIRST_CHUNK, "resp-1"), _answered_as(STREAM_HOP_1_END, "resp-1")),
        httpx.Response(500, json={"error": {"code": 500}})
        if next_request_fails
        else _sse(_answered_as(HOP_2, "resp-2")),
    )

    chunks: Final = await _collect_stream(is_async, hops, sent)

    assert len(sent) == 2
    assert _stream_finish_reasons(chunks) == ["length" if next_request_fails else "stop"]
    assert {chunk.id for chunk in chunks} == {"resp-1"}


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_answer_cut_at_the_callers_cap_ends_with_length_and_its_token_in_one_request(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        _sse(STREAM_FIRST_CHUNK, _ending_with(STREAM_HOP_1_END, "MAX_TOKENS")),
        httpx.Response(400, json=GOOGLE_REJECTION),
    )

    chunks: Final = await _collect_stream(is_async, hops, sent)

    assert len(sent) == 1
    assert _stream_text(chunks) == "hello"
    assert _stream_finish_reasons(chunks) == ["length"]
    assert {"continuation_token": "dG9rLTE="} in _delta_fields(chunks)


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_request_for_more_than_one_choice_is_never_continued(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (_sse(TWO_CHOICES_CUT), httpx.Response(400, json=GOOGLE_REJECTION))

    chunks: Final = await _stream_chunks(is_async, _replay(hops, sent), n=2)

    assert len(sent) == 1
    assert _stream_finish_reasons(chunks) == ["length"]
    assert {"continuation_token": "dG9rLTE="} in _delta_fields(chunks)


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
async def test_streaming_continuation_stream_is_read_whole_after_a_stream_that_ended_on_an_unfinished_frame(
    is_async: bool,
):
    sent: Final[list[dict[str, object]]] = []
    first_stream, next_stream = _stream_hops()
    cut_off_first_stream: Final = httpx.Response(
        200,
        content=first_stream.content + b'data: {"candidates": [{"content": {"parts": [{"text": "lo\r\n\r\n',
        headers=first_stream.headers,
    )

    chunks: Final = await _collect_stream(is_async, (cut_off_first_stream, next_stream), sent)

    assert len(sent) == 2
    assert _stream_text(chunks) == "hello world"
    assert _stream_finish_reasons(chunks) == ["stop"]


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


def _checkpoint(text: str, token: str) -> dict[str, object]:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "continuationToken": token}]}


def _client_tokens(chunks: list) -> list[object]:
    return [fields["continuation_token"] for fields in _delta_fields(chunks) if "continuation_token" in fields]


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_mid_stream_checkpoint_tokens_never_reach_the_client(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        _sse(_checkpoint("hel", "Y2twdC0x"), STREAM_HOP_1_END),
        _sse(_checkpoint(" big", "Y2twdC0y"), _checkpoint(" wide", "Y2twdC0z"), HOP_2),
    )

    chunks: Final = await _collect_stream(is_async, hops, sent)

    assert len(sent) == 2
    assert _stream_text(chunks) == "hello big wide world"
    assert _stream_finish_reasons(chunks) == ["stop"]
    assert _client_tokens(chunks) == []


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_cut_off_stream_gives_the_client_only_its_final_token(is_async: bool):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (
        _sse(_checkpoint("hel", "Y2twdC0x"), _ending_with(STREAM_HOP_1_END, "MAX_TOKENS")),
        httpx.Response(400, json=GOOGLE_REJECTION),
    )

    chunks: Final = await _collect_stream(is_async, hops, sent)

    assert len(sent) == 1
    assert _stream_text(chunks) == "hello"
    assert _stream_finish_reasons(chunks) == ["length"]
    assert _client_tokens(chunks) == ["dG9rLTE="]


class _BodyThatDropsBeforeItsFirstFrame(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    def __aiter__(self) -> AsyncIterator[bytes]:
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


@pytest.mark.parametrize(
    "failed_hop",
    [
        lambda: httpx.Response(500, json={"error": {"code": 500}}),
        lambda: httpx.Response(200, content=b"<html>not json</html>"),
        lambda: _sse({"error": {"code": 503, "message": "The model is overloaded.", "status": "UNAVAILABLE"}}, HOP_2),
        lambda: httpx.Response(200, stream=_BodyThatDropsBeforeItsFirstFrame()),
    ],
    ids=["refused", "not_json", "error_frame", "dropped"],
)
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_failed_continuation_hop_ends_the_stream_with_length_and_the_token(
    is_async: bool, failed_hop: Callable[[], httpx.Response]
):
    sent: Final[list[dict[str, object]]] = []
    hops: Final = (_stream_hops()[0], failed_hop())

    chunks: Final = await _collect_stream(is_async, hops, sent)

    assert len(sent) == 2
    assert "".join(c.choices[0].delta.content or "" for c in chunks if c.choices) == "hello"
    assert [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason] == ["length"]
    assert {"continuation_token": "dG9rLTE="} in _delta_fields(chunks)


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_streaming_continuation_request_is_sent_with_the_callers_timeout(is_async: bool):
    read_timeouts: Final[list[object]] = []
    hops: Final = _stream_hops()

    def handler(request: httpx.Request) -> httpx.Response:
        read_timeouts.append(request.extensions["timeout"]["read"])
        return hops[len(read_timeouts) - 1]

    chunks: Final = await _stream_chunks(is_async, handler, timeout=7.5)

    assert _stream_text(chunks) == "hello world"
    assert read_timeouts == [7.5, 7.5]


@dataclass(frozen=True, slots=True)
class _TokenEndpoint:
    credentials: str
    down: threading.Event


def _service_account_json(token_uri: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "p",
            "private_key_id": "k",
            "private_key": private_key,
            "client_email": "sa@p.iam.gserviceaccount.com",
            "token_uri": token_uri,
        }
    )


@pytest.fixture
def token_endpoint() -> Iterator[_TokenEndpoint]:
    down: Final = threading.Event()
    minted: Final = count(1)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            refused: Final = down.is_set()
            payload: Final = json.dumps(
                {"error": "invalid_grant"} if refused else {"access_token": f"token-{next(minted)}", "expires_in": 0}
            ).encode()
            self.send_response(400 if refused else 200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            return None

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield _TokenEndpoint(_service_account_json(f"http://127.0.0.1:{server.server_port}/token"), down)
    server.shutdown()
    server.server_close()


def _google_taking_each_access_token_once(
    hops: Sequence[httpx.Response], used: list[str], on_answer: Callable[[], object]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        token: Final = request.headers["authorization"]
        if token in used:
            return httpx.Response(401, json={"error": {"code": 401, "status": "UNAUTHENTICATED"}})
        used.append(token)
        on_answer()
        return hops[len(used) - 1]

    return handler


async def _vertex_answer(
    is_async: bool,
    stream: bool,
    handler: Callable[[httpx.Request], httpx.Response],
    credentials: str,
    extra_headers: Mapping[str, str] | None = None,
) -> tuple[str, list[str], list[object]]:
    kwargs: Final = {
        "model": "vertex_ai/gemini-3.8-flash",
        "extra_headers": extra_headers,
        "messages": list(MESSAGES),
        "max_tokens": 100_000,
        "stream": stream,
        "vertex_project": "p",
        "vertex_location": "global",
        "vertex_credentials": credentials,
    }
    transport: Final = httpx.MockTransport(handler)
    response: Final = (
        await litellm.acompletion(**kwargs, client=AsyncHTTPHandler(transport=transport))
        if is_async
        else litellm.completion(**kwargs, client=HTTPHandler(client=httpx.Client(transport=transport)))
    )
    if stream:
        chunks: Final = [chunk async for chunk in response] if is_async else list(response)
        return _stream_text(chunks), _stream_finish_reasons(chunks), _client_tokens(chunks)
    choice: Final = response.choices[0]
    fields: Final = choice.message.provider_specific_fields or {}
    return choice.message.content, [choice.finish_reason], [fields["continuation_token"]] if fields else []


def _two_hops(stream: bool) -> tuple[httpx.Response, httpx.Response]:
    return _stream_hops() if stream else (httpx.Response(200, json=HOP_1), httpx.Response(200, json=HOP_2))


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_vertex_long_decode_finishes_when_each_continuation_outlives_the_access_token_before_it(
    is_async: bool, stream: bool, token_endpoint: _TokenEndpoint
):
    used: Final[list[str]] = []
    google: Final = _google_taking_each_access_token_once(_two_hops(stream), used, lambda: None)

    answer: Final = await _vertex_answer(is_async, stream, google, token_endpoint.credentials)

    assert answer == ("hello world", ["stop"], [])
    assert len(used) == 2


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_vertex_long_decode_returns_the_answer_so_far_with_its_token_when_no_new_access_token_can_be_had(
    is_async: bool, stream: bool, token_endpoint: _TokenEndpoint
):
    used: Final[list[str]] = []
    google: Final = _google_taking_each_access_token_once(_two_hops(stream), used, token_endpoint.down.set)

    answer: Final = await _vertex_answer(is_async, stream, google, token_endpoint.credentials)

    assert answer == ("hello", ["length"], ["dG9rLTE="])
    assert len(used) == 1


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_vertex_long_decode_keeps_the_callers_own_authorization_header_on_every_continuation(
    is_async: bool, stream: bool, token_endpoint: _TokenEndpoint
):
    hops: Final = _two_hops(stream)
    seen: Final[list[str]] = []

    def google(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return hops[len(seen) - 1]

    answer: Final = await _vertex_answer(
        is_async, stream, google, token_endpoint.credentials, {"Authorization": "Bearer mine"}
    )

    assert answer == ("hello world", ["stop"], [])
    assert seen == ["Bearer mine", "Bearer mine"]


def test_initial_prompt_tokens_is_absent_on_a_single_hop_and_the_first_hops_count_otherwise():
    assert initial_prompt_tokens_of((HOP_1,)) is None
    assert initial_prompt_tokens_of((HOP_1, HOP_2)) == 10


def test_with_initial_prompt_tokens_adds_the_details_block_and_keeps_the_counts():
    usage: Final = Usage(prompt_tokens=20, completion_tokens=50, total_tokens=70)

    marked: Final = with_initial_prompt_tokens(usage, 10)

    assert marked.prompt_tokens_details.initial_prompt_tokens == 10
    assert (marked.prompt_tokens, marked.completion_tokens, marked.total_tokens) == (20, 50, 70)
    assert usage.prompt_tokens_details is None
