"""Life of a streaming request around the chunks themselves.

These contracts cover what happens at the edges of a stream: when the response headers are released
relative to the provider's first chunk, which `x-litellm-*` headers a stream carries, what a cache hit
looks like when the caller asked to stream, what a `mock_response` deployment streams, what a caller
sees when the first deployment fails before any chunk and a fallback answers, and what is recorded when
the caller walks away mid stream.

The shared proxy runs with the Redis response cache on, so the cache contracts need no extra setup and
every other test uses a unique prompt.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, held, object_value
from integration._support.wire import Reply
from integration.streaming.dec1_stream_support import (
    Rig,
    chunks_of,
    completion_json,
    content_of,
    finish_reasons,
    marker,
    payloads,
    scripted,
    serving,
    sse,
    text_frames,
    usage_chunks,
)


def test_nothing_is_sent_to_the_caller_until_the_provider_sends_its_first_chunk(gateway: Gateway) -> None:
    """The provider accepts the request and holds its first chunk. For a full second the caller's request
    is still pending with no status line. When the provider releases the chunk the caller gets 200 and
    the whole stream. The proxy buffers the first chunk so that a provider error on open can still become
    a real HTTP error; a rewrite that sends 200 immediately loses that."""
    gate: Final = threading.Event()
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("held ", "back"))
    with scripted(gateway, lambda request: sse(frames, gate_before_first=gate)) as rig, ThreadPoolExecutor(1) as pool:
        pending: Final = pool.submit(rig.chat)
        held(pending.done, lambda done: not done, holding=1)
        gate.set()
        response: Final = pending.result(timeout=30)
        assert response.status_code == 200
        assert content_of(chunks_of(response.text)) == "held back"


def test_headers_and_first_chunk_arrive_while_the_provider_is_still_streaming(gateway: Gateway) -> None:
    """The provider sends its first chunk and then holds the rest. The caller already has status 200, the
    `x-litellm-call-id` header and the first content delta while the stream is open, and receives the
    remaining text and `[DONE]` once the provider continues. Buffering the whole answer before the
    first byte would pass every non timing test and still break every chat UI."""
    gate: Final = threading.Event()
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("early ", "late"))
    with scripted(gateway, lambda request: sse(frames, gate_after_first=gate)) as rig:
        with rig.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json=rig.chat_body(),
            headers={"Authorization": f"Bearer {rig.key}"},
            timeout=30,
        ) as response:
            assert response.status_code == 200
            assert response.headers["x-litellm-call-id"]
            lines: Final = response.iter_lines()
            first: Final = next(line for line in lines if line.startswith("data: "))
            assert content_of([json.loads(first.removeprefix("data: "))]) == "early "
            gate.set()
            rest: Final = [line.removeprefix("data: ") for line in lines if line.startswith("data: ")]
        assert rest[-1] == "[DONE]"
        assert content_of([json.loads(item) for item in rest[:-1]]) == "late"


def test_stream_response_headers_identify_the_call_the_deployment_and_the_version(gateway: Gateway) -> None:
    """A streamed chat response carries `x-litellm-call-id` (non empty), `x-litellm-model-id` equal to the
    deployment's `model_info.id`, `x-litellm-model-api-base` equal to the deployment's `api_base`,
    `x-litellm-version`, and `x-litellm-key-spend`. These are set from the first chunk's hidden
    parameters in `_stream_response_headers`; operators use them to see which deployment served a
    stream."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("hea", "ders")))) as rig:
        response: Final = rig.chat()
        entries: Final = [object_value(entry) for entry in gateway.get("/model/info")["data"]]  # type: ignore[union-attr]
        deployment: Final = next(entry for entry in entries if entry["model_name"] == rig.model)
        assert response.headers["x-litellm-call-id"]
        assert response.headers["x-litellm-model-id"] == object_value(deployment["model_info"])["id"]
        assert response.headers["x-litellm-model-api-base"] == rig.wire.url + "/v1"
        assert response.headers["x-litellm-version"]
        assert "x-litellm-key-spend" in response.headers


def test_stream_response_headers_never_contain_the_provider_key(gateway: Gateway) -> None:
    """No response header of a streamed call contains the deployment's provider API key or the string
    `Bearer`. Header building copies provider and hidden parameters, which is where a leak would
    happen."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("no ", "leak")))) as rig:
        headers: Final = json.dumps(dict(rig.chat().headers))
        assert "synthetic-provider-key" not in headers
        assert "Bearer" not in headers


def test_provider_connection_is_closed_after_a_completed_stream(gateway: Gateway) -> None:
    """One stream runs to completion against a peer that does not keep connections alive. Afterwards the
    peer has zero open connections from the proxy. A stream wrapper that is never closed keeps the
    socket (and in Python the logging context) alive until garbage collection."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("close ", "me")))) as rig:
        assert rig.chat().status_code == 200
        assert eventually(rig.wire.open_connections, lambda count: count == 0, seconds=10) == 0


def test_identical_stream_request_is_replayed_from_cache_without_calling_the_provider(gateway: Gateway) -> None:
    """Two identical streaming requests are sent one after the other. The provider receives exactly one
    call. The second caller still gets an event stream (not a JSON body) with the same text, a `stop`
    finish and `[DONE]`."""
    prompt: Final = f"dec1 cached stream {marker()}"
    with scripted(gateway, lambda request: sse(text_frames(f"chatcmpl-{marker()}", ("cached ", "answer")))) as rig:
        first: Final = rig.chat(prompt)
        rig.spend_rows(1)
        second: Final = rig.chat(prompt)
        assert second.headers["content-type"].split(";")[0] == "text/event-stream"
        assert content_of(chunks_of(first.text)) == content_of(chunks_of(second.text)) == "cached answer"
        assert finish_reasons(chunks_of(second.text)) == ["stop"]
        assert payloads(second.text)[-1] == "[DONE]"
        assert len(rig.upstream_bodies()) == 1


def test_cache_hit_stream_is_logged_as_a_cache_hit_and_costs_nothing(gateway: Gateway) -> None:
    """A priced deployment answers one stream (0.019) and the identical second stream is a cache hit. Two
    SpendLogs rows exist: the first with `cache_hit == "False"` and spend 0.019, the second with
    `cache_hit == "True"` and spend 0. Both record the provider's token counts."""
    prompt: Final = f"dec1 cached billing {marker()}"
    with scripted(
        gateway,
        lambda request: sse(text_frames(f"chatcmpl-{marker()}", ("free ", "replay"))),
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    ) as rig:
        assert rig.chat(prompt).status_code == 200
        rig.spend_rows(1)
        assert rig.chat(prompt).status_code == 200
        rows: Final = rig.spend_rows(2)
        assert [row["cache_hit"] for row in rows] == ["False", "True"]
        assert float(str(rows[0]["spend"])) == pytest.approx(0.019)
        assert float(str(rows[1]["spend"])) == 0.0
        assert {(row["prompt_tokens"], row["completion_tokens"]) for row in rows} == {(11, 4)}


def test_cache_hit_stream_response_carries_the_cache_key_header(gateway: Gateway) -> None:
    """The second of two identical streams, served from cache, has a non empty `x-litellm-cache-key`
    response header. UNVERIFIED: `_stream_response_headers` passes `hidden_params["cache_key"]`, which
    the caching handler sets on a hit."""
    prompt: Final = f"dec1 cache header {marker()}"
    with scripted(gateway, lambda request: sse(text_frames(f"chatcmpl-{marker()}", ("key",)))) as rig:
        assert rig.chat(prompt).status_code == 200
        rig.spend_rows(1)
        assert rig.chat(prompt).headers.get("x-litellm-cache-key")


def test_cache_hit_stream_honours_include_usage(gateway: Gateway) -> None:
    """The first stream is sent without `include_usage`, the identical second one with it. The second is
    served from cache and still ends with a `choices: []` usage chunk reporting 11, 4 and 15 before
    `[DONE]`. UNVERIFIED that `stream_options` is excluded from the cache key; if it is part of the key
    the provider is called twice and the assertion on one upstream call fails, which is the signal."""
    prompt: Final = f"dec1 cache usage {marker()}"
    with scripted(gateway, lambda request: sse(text_frames(f"chatcmpl-{marker()}", ("use",)))) as rig:
        assert rig.chat(prompt).status_code == 200
        rig.spend_rows(1)
        chunks: Final = chunks_of(rig.chat(prompt, stream_options={"include_usage": True}).text)
        usage: Final = object_value(usage_chunks(chunks)[-1]["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 4, 15)
        assert len(rig.upstream_bodies()) == 1


def test_no_cache_request_control_sends_both_streams_to_the_provider(gateway: Gateway) -> None:
    """Two identical streaming requests both carry `cache: {"no-cache": true}`. The provider is called
    twice and neither SpendLogs row is a cache hit."""
    prompt: Final = f"dec1 no cache {marker()}"
    with scripted(gateway, lambda request: sse(text_frames(f"chatcmpl-{marker()}", ("fresh",)))) as rig:
        for _ in range(2):
            assert rig.chat(prompt, cache={"no-cache": True}).status_code == 200
        assert [row["cache_hit"] for row in rig.spend_rows(2)] == ["False", "False"]
        assert len(rig.upstream_bodies()) == 2


def test_failed_stream_is_not_cached(gateway: Gateway) -> None:
    """The first stream for a prompt is cut by the provider mid answer; the identical second request gets
    a healthy provider. The second caller receives the full healthy text, which proves the truncated
    first answer was not stored in the cache, and the provider was called twice."""
    prompt: Final = f"dec1 poisoned cache {marker()}"
    calls: Final = []

    def respond(request):  # type: ignore[no-untyped-def]
        calls.append(1)
        frames: Final = text_frames(f"chatcmpl-{marker()}", ("whole ", "answer", " here"))
        return sse(frames, abort_after=1) if len(calls) == 1 else sse(frames)

    with scripted(gateway, respond) as rig:
        assert any('"error"' in item for item in payloads(rig.chat(prompt).text))
        rig.spend_rows(1)
        assert content_of(chunks_of(rig.chat(prompt).text)) == "whole answer here"
        assert len(calls) == 2


def test_stream_then_non_stream_of_the_same_request_do_not_share_a_cache_entry(gateway: Gateway) -> None:
    """A prompt is first answered as a stream, then requested again with `stream: false`. The second caller
    gets the provider's JSON answer, not the streamed text, and the provider is called again: `stream` is an
    OpenAI chat parameter, so its value is part of the cache key (`Cache.get_cache_key`,
    `litellm/caching/caching.py:428`, over `ModelParamHelper.get_all_llm_api_params`). A third request that
    streams again is served the stored stream without a provider call, which proves the first entry exists and
    the miss came from the key. Pinned current behaviour: a rewrite that drops `stream` from the key would
    start serving one mode's answer to the other.
    The streamed and the JSON answer carry different ids: a SpendLogs row is keyed by the response id
    (`request_id @id`, `litellm/proxy-extras/litellm_proxy_extras/schema.prisma:723`), so two answers sharing
    one id leave one row and run 43314424795 waited for a second row that could never be written."""
    prompt: Final = f"dec1 cross mode cache {marker()}"
    streamed: Final = f"chatcmpl-{marker()}"
    answered: Final = f"chatcmpl-{marker()}"
    with scripted(
        gateway,
        serving(text_frames(streamed, ("shared ", "entry")), json_body=completion_json(answered, "from provider")),
    ) as rig:
        assert rig.chat(prompt).status_code == 200
        rig.spend_rows(1)
        second: Final = rig.chat(prompt, stream=False)
        assert second.json()["choices"][0]["message"]["content"] == "from provider"
        rig.spend_rows(2)
        assert content_of(chunks_of(rig.chat(prompt).text)) == "shared entry"
        assert [body.get("stream") is True for body in rig.upstream_bodies()] == [True, False]


def test_mock_response_deployment_streams_its_text_without_calling_a_provider(gateway: Gateway) -> None:
    """A deployment configured with `mock_response: "mocked dec1 reply"` is asked to stream. The caller
    gets chat chunks whose content concatenates to that text, `role: assistant` on the first, a `stop`
    finish, and `[DONE]`. The provider peer receives no request. Load tests and health probes rely on
    mock deployments streaming like real ones."""
    with scripted(gateway, lambda request: Reply(status=500), mock_response="mocked dec1 reply") as rig:
        body: Final = rig.chat().text
        chunks: Final = chunks_of(body)
        assert content_of(chunks) == "mocked dec1 reply"
        assert finish_reasons(chunks) == ["stop"]
        assert payloads(body)[-1] == "[DONE]"
        assert rig.upstream_bodies() == []


def test_fallback_before_the_first_chunk_streams_the_fallback_answer_under_the_fallback_name(gateway: Gateway) -> None:
    """The requested deployment fails on stream open with a 500 and the request names a second deployment
    in `fallbacks`. The caller gets status 200 and the fallback's text; every chunk's `model` is the
    fallback's public name, because `_restamp_streaming_chunk_model` stamps the model that actually
    answered once a fallback was attempted; and the `x-litellm-attempted-fallbacks` header is `1`.
    UNVERIFIED on the header value being readable on the stream response."""
    failing: Final = Reply(status=500, body=b'{"error": {"message": "primary down", "type": "server_error"}}')
    with (
        scripted(gateway, lambda request: failing) as primary,
        scripted(gateway, lambda request: sse(text_frames(f"chatcmpl-{marker()}", ("from ", "fallback")))) as fallback,
    ):
        response: Final = primary.post(
            "/v1/chat/completions", primary.chat_body(fallbacks=[fallback.model]), key=gateway.key
        )
        assert response.status_code == 200, response.text
        chunks: Final = chunks_of(response.text)
        assert content_of(chunks) == "from fallback"
        assert {item["model"] for item in chunks} == {fallback.model}
        assert response.headers.get("x-litellm-attempted-fallbacks") == "1"


def test_fallback_stream_is_billed_once_against_the_deployment_that_answered(gateway: Gateway) -> None:
    """A stream falls back before its first chunk from a failing deployment to a priced healthy one. The
    caller's key ends with exactly one `success` SpendLogs row carrying the fallback's tokens (11 and 4)
    and its price (0.019). Any `failure` row for the first attempt carries no spend."""
    failing: Final = Reply(status=500, body=b'{"error": {"message": "primary down", "type": "server_error"}}')
    with (
        gateway.scenario() as scenario,
        scripted(gateway, lambda request: failing) as primary,
        scripted(
            gateway,
            lambda request: sse(text_frames(f"chatcmpl-{marker()}", ("paid ", "once"))),
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        ) as fallback,
    ):
        key: Final = scenario.key(models=[primary.model, fallback.model])
        caller: Final = Rig(gateway, primary.wire, primary.model, key)
        assert caller.chat(fallbacks=[fallback.model]).status_code == 200
        rows: Final = _rows(caller)
        succeeded: Final = [row for row in rows if row["status"] == "success"]
        assert len(succeeded) == 1
        assert (succeeded[0]["prompt_tokens"], succeeded[0]["completion_tokens"]) == (11, 4)
        assert sum(float(str(row["spend"])) for row in rows) == pytest.approx(0.019)


def _rows(rig):  # type: ignore[no-untyped-def]
    """All SpendLogs rows for the rig's key once at least one success row has landed and writes settle."""
    from hashlib import sha256

    from integration._support.database import read_rows

    digest: Final = sha256(rig.key.encode()).hexdigest()
    query: Final = 'SELECT status, spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s'

    def read() -> list:
        return read_rows(query, (digest,))

    eventually(read, lambda rows: any(row["status"] == "success" for row in rows), seconds=70)
    return held(read, lambda rows: any(row["status"] == "success" for row in rows), holding=3)


def test_caller_disconnect_mid_stream_still_records_the_partial_call(gateway: Gateway) -> None:
    """The provider streams slowly. The caller reads the first content chunk and closes the connection.
    One SpendLogs row still lands for the key with a positive `completion_tokens` count and, on this
    priced deployment, a positive spend: tokens the provider already generated are paid for. The row's
    metadata records the disconnect as `error_information.error_code == "499"`. UNVERIFIED on the
    metadata reaching the row: read from `_apply_client_disconnect_metadata` and
    `_bill_partial_streamed_spend_on_disconnect` in `litellm/proxy/common_request_processing.py`."""
    frames: Final = text_frames(f"chatcmpl-{marker()}", tuple(f"piece{number} " for number in range(40)), usage=None)
    with scripted(
        gateway,
        lambda request: sse(frames, pause_between_chunks=0.1),
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    ) as rig:
        with rig.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json=rig.chat_body(),
            headers={"Authorization": f"Bearer {rig.key}"},
            timeout=30,
        ) as response:
            assert response.status_code == 200
            next(line for line in response.iter_lines() if line.startswith("data: "))
        row: Final = rig.spend_rows(1, settle=5)[0]
        assert int(str(row["completion_tokens"])) > 0
        assert float(str(row["spend"])) > 0
        information: Final = object_value(object_value(row["metadata"]).get("error_information") or {})
        assert information.get("error_code") == "499"


def test_caller_disconnect_mid_stream_stops_the_provider_stream(gateway: Gateway) -> None:
    """The provider streams 200 chunks 50 ms apart (10 seconds in all). The caller closes after the first
    chunk. Within a few seconds the provider peer sees its connection from the proxy closed, long before
    the stream would have ended on its own. Continuing to pull tokens nobody reads costs real money."""
    frames: Final = text_frames(f"chatcmpl-{marker()}", tuple(f"p{number} " for number in range(200)), usage=None)
    with scripted(gateway, lambda request: sse(frames, pause_between_chunks=0.05)) as rig:
        with rig.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json=rig.chat_body(),
            headers={"Authorization": f"Bearer {rig.key}"},
            timeout=30,
        ) as response:
            next(line for line in response.iter_lines() if line.startswith("data: "))
        assert eventually(rig.wire.open_connections, lambda count: count == 0, seconds=5) == 0


def test_next_stream_after_a_caller_disconnect_is_served_normally(gateway: Gateway) -> None:
    """A caller abandons a stream after its first chunk; the same key then streams again. The second
    stream is complete with a `stop` finish and `[DONE]`. A disconnect must release the key's parallel
    request slot and leave no half closed state behind (`finalize_streaming_generator_cleanup`)."""
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("still ", "works", " fine"), usage=None)
    with scripted(
        gateway, lambda request: sse(frames, pause_between_chunks=0.2), key_fields={"max_parallel_requests": 1}
    ) as rig:
        with rig.gateway.client.stream(
            "POST",
            "/v1/chat/completions",
            json=rig.chat_body(),
            headers={"Authorization": f"Bearer {rig.key}"},
            timeout=30,
        ) as response:
            next(line for line in response.iter_lines() if line.startswith("data: "))
        second: Final = eventually(rig.chat, lambda answer: answer.status_code == 200, seconds=20)
        assert content_of(chunks_of(second.text)) == "still works fine"
        assert payloads(second.text)[-1] == "[DONE]"
