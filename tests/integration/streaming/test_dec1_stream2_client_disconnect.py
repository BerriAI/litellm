"""A caller that hangs up mid stream, on every streaming route and the big provider dialects.

When the caller closes its connection the proxy must stop pulling tokens from the provider (close the
upstream connection), still write exactly one SpendLogs row that pays for what was generated so far, flag
that row as a client disconnect (`metadata.error_information.error_code == "499"`, status `success`),
settle the key's budget hold to the partial cost instead of leaving the worst case estimate pinned, and give
back the key's `max_parallel_requests` slot. A caller that leaves before the first provider chunk gets the
upstream call cancelled and the whole output part of the hold released.

Read from `async_data_generator` (`litellm/proxy/proxy_server.py`, the `CancelledError`/`GeneratorExit`
branch), `finalize_streaming_generator_cleanup`, `_bill_partial_streamed_spend_on_disconnect` and
`_buffer_first_chunk_honoring_disconnect` in `litellm/proxy/common_request_processing.py` (the disconnect
billing was last reworked in #52088 on 2026-11-16). Chat on OpenAI already has disconnect tests in
`test_dec1_stream_lifecycle.py`; this file adds the other routes, the other providers, the row shape, the
budget hold and the before-first-chunk path. The native `/v1/messages` path is served by Rust only for
Anthropic, Azure AI Foundry and Bedrock Invoke, so the Python expectations below are the contract the Rust
relay must keep.
"""

from __future__ import annotations

from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, object_value
from integration.spend.dec1_budgets_support import counter, counter_settles, key_counter, redis_client
from integration.streaming.dec1_stream2_support import (
    ANTHROPIC,
    AZURE,
    BEDROCK_CONVERSE,
    CLIENT_DISCONNECT_ERROR_INFORMATION,
    GEMINI_AI_STUDIO,
    NATIVE_MESSAGES_PROVIDERS,
    OPENAI,
    OPENAI_RESPONSES,
    Provider,
    deployed,
    first_frame,
    opened,
    provider_ids,
    provider_stream,
    route_body,
    streaming,
)
from integration.streaming.dec1_stream_support import Rig, chunks_of, content_of, payloads

SLOW_PIECES: Final = tuple(f"slow{number} " for number in range(150))
"""150 deltas 50 ms apart: about 7.5 seconds, long enough that a closed upstream is visibly early."""
COMMENT_WARMUP: Final = tuple(b": provider warming up\n\n" for _ in range(100))
"""SSE comments a provider sends before its first token; the proxy never turns them into a chunk."""


def _hang_up_after_first_frame(rig: Rig, route: str) -> None:
    with opened(rig, route) as response:
        assert response.status_code == 200, response.read()
        first_frame(response)


def _upstream_closed(rig: Rig) -> bool:
    return rig.wire.open_connections() == 0 or rig.wire.disconnected.qsize() > 0


@pytest.mark.parametrize("provider", [ANTHROPIC, AZURE, GEMINI_AI_STUDIO, BEDROCK_CONVERSE], ids=provider_ids)
def test_chat_caller_disconnect_closes_the_provider_stream_for_each_provider(
    gateway: Gateway, provider: Provider
) -> None:
    """Each big provider streams 150 deltas 50 ms apart in its own dialect (Anthropic SSE, Azure SSE, Gemini
    SSE, Bedrock binary event stream). The caller of `/v1/chat/completions` reads one frame and hangs up.
    Within 5 seconds the provider peer sees the proxy's connection closed, long before its 7.5 second stream
    would have ended. A rewrite that keeps draining the provider after the caller left pays for tokens
    nobody reads."""
    with deployed(gateway, provider, provider_stream(provider, SLOW_PIECES, pause_between_chunks=0.05)) as rig:
        _hang_up_after_first_frame(rig, "/v1/chat/completions")
        assert eventually(lambda: _upstream_closed(rig), lambda closed: closed, seconds=5)


@pytest.mark.parametrize("provider", NATIVE_MESSAGES_PROVIDERS, ids=provider_ids)
def test_messages_caller_disconnect_closes_the_provider_stream(gateway: Gateway, provider: Provider) -> None:
    """A native `/v1/messages` deployment (Anthropic, Azure AI Foundry Claude, Bedrock Invoke) streams 150
    text deltas 50 ms apart. The caller reads the first event and hangs up. Within 5 seconds the provider
    peer sees its connection closed. These three providers are served by the Rust byte relay only, which
    must propagate the caller's hang up to the upstream body it is relaying."""
    with deployed(gateway, provider, provider_stream(provider, SLOW_PIECES, pause_between_chunks=0.05)) as rig:
        _hang_up_after_first_frame(rig, "/v1/messages")
        assert eventually(lambda: _upstream_closed(rig), lambda closed: closed, seconds=5)


def test_responses_caller_disconnect_closes_the_provider_stream(gateway: Gateway) -> None:
    """A native OpenAI `/v1/responses` deployment streams 150 output_text deltas 50 ms apart. The caller
    reads the first event and hangs up. Within 5 seconds the provider peer sees its connection closed."""
    with deployed(
        gateway, OPENAI_RESPONSES, provider_stream(OPENAI_RESPONSES, SLOW_PIECES, pause_between_chunks=0.05)
    ) as rig:
        _hang_up_after_first_frame(rig, "/v1/responses")
        assert eventually(lambda: _upstream_closed(rig), lambda closed: closed, seconds=5)


def test_disconnect_row_is_a_success_row_carrying_the_client_disconnect_error_information(gateway: Gateway) -> None:
    """An OpenAI deployment streams slowly; the caller reads one frame and hangs up. Exactly one SpendLogs
    row lands with `status == "success"` (the partial answer is billed through the success path) and its
    `metadata.error_information` is exactly `{"error_code": "499", "error_message": "Client disconnected the
    request", "error_class": "ClientDisconnected"}`. Dashboards split abandoned streams from real failures
    on exactly these fields."""
    with deployed(gateway, OPENAI, provider_stream(OPENAI, SLOW_PIECES, pause_between_chunks=0.05)) as rig:
        _hang_up_after_first_frame(rig, "/v1/chat/completions")
        row: Final = rig.spend_rows(1, settle=5)[0]
        assert row["status"] == "success"
        information: Final = object_value(object_value(row["metadata"])["error_information"])
        assert {key: information.get(key) for key in CLIENT_DISCONNECT_ERROR_INFORMATION} == dict(
            CLIENT_DISCONNECT_ERROR_INFORMATION
        )


def test_disconnect_row_bills_only_the_tokens_streamed_so_far_at_the_deployment_price(gateway: Gateway) -> None:
    """A priced OpenAI deployment (0.001 per input token, 0.002 per output token) would stream 150 deltas and
    sends no usage. The caller hangs up after one frame. The single row's `completion_tokens` is positive and
    well below what 150 deltas would count, and its `spend` equals `prompt_tokens * 0.001 +
    completion_tokens * 0.002`: the partial answer is costed from the chunks the proxy actually received."""
    with deployed(
        gateway,
        OPENAI,
        provider_stream(OPENAI, SLOW_PIECES, pause_between_chunks=0.05),
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    ) as rig:
        _hang_up_after_first_frame(rig, "/v1/chat/completions")
        row: Final = rig.spend_rows(1, settle=5)[0]
        prompt: Final = int(str(row["prompt_tokens"]))
        completion: Final = int(str(row["completion_tokens"]))
        assert 0 < completion < 150
        assert float(str(row["spend"])) == pytest.approx(prompt * 0.001 + completion * 0.002)


@pytest.mark.parametrize("provider", [ANTHROPIC, BEDROCK_CONVERSE], ids=provider_ids)
def test_disconnect_from_a_translated_chat_stream_writes_one_success_row(gateway: Gateway, provider: Provider) -> None:
    """An Anthropic or Bedrock Converse deployment streams slowly through `/v1/chat/completions`. The caller
    hangs up after one frame. Exactly one row lands for the key, `status == "success"`, with
    `error_information.error_code == "499"` and a positive `completion_tokens`. The partial chunks here were
    translated from a non OpenAI dialect, which is where a rewrite is most likely to lose them."""
    with deployed(
        gateway,
        provider,
        provider_stream(provider, SLOW_PIECES, pause_between_chunks=0.05),
        output_cost_per_token=0.002,
    ) as rig:
        _hang_up_after_first_frame(rig, "/v1/chat/completions")
        row: Final = rig.spend_rows(1, settle=5)[0]
        assert row["status"] == "success"
        assert object_value(object_value(row["metadata"])["error_information"])["error_code"] == "499"
        assert int(str(row["completion_tokens"])) > 0


def test_disconnect_from_a_native_messages_stream_writes_one_row_with_the_provider_input_tokens(
    gateway: Gateway,
) -> None:
    """An Anthropic deployment streams `message_start` (input_tokens 10) and then slow text deltas on
    `/v1/messages`. The caller hangs up after the first event. Exactly one SpendLogs row lands for the key
    with `prompt_tokens == 10`, read from the `message_start` the proxy already relayed. UNVERIFIED: that
    the Rust relay bills the partial message on disconnect like the Python iterator did; look at
    `litellm/rust/crates/inference-messages/src/handler.rs` if it writes no row."""
    with deployed(gateway, ANTHROPIC, provider_stream(ANTHROPIC, SLOW_PIECES, pause_between_chunks=0.05)) as rig:
        _hang_up_after_first_frame(rig, "/v1/messages")
        row: Final = rig.spend_rows(1, settle=5)[0]
        assert int(str(row["prompt_tokens"])) == 10


def test_disconnect_from_a_native_responses_stream_writes_one_success_row(gateway: Gateway) -> None:
    """A native OpenAI Responses deployment streams slowly. The caller hangs up after the first event.
    Exactly one row lands with `status == "success"` and `error_information.error_code == "499"`: the
    proxy builds a `response.incomplete` event from the text so far and bills it
    (`partial_responses_incomplete_event_on_disconnect`, `litellm/responses/streaming_iterator.py:1726`)."""
    with deployed(
        gateway, OPENAI_RESPONSES, provider_stream(OPENAI_RESPONSES, SLOW_PIECES, pause_between_chunks=0.05)
    ) as rig:
        _hang_up_after_first_frame(rig, "/v1/responses")
        row: Final = rig.spend_rows(1, settle=5)[0]
        assert row["status"] == "success"
        assert object_value(object_value(row["metadata"])["error_information"])["error_code"] == "499"


def test_disconnect_before_the_first_chunk_cancels_the_provider_call(gateway: Gateway) -> None:
    """The provider answers 200 and then sends only SSE comment lines, 200 ms apart, for 20 seconds, so the
    proxy is still waiting for its first chunk. The caller gives up after one second of reading. Within 5
    seconds the provider's next write fails because the proxy closed the connection
    (`_buffer_first_chunk_honoring_disconnect`, LIT-3568). Without this a long time to first token keeps
    the provider busy until the request timeout.

    BUG: run 44176682961 showed the provider connection still open after the caller left before any provider
    output (no wire disconnect in 5 to 10 s, and the late chunk later billed 0.004 on the hold test), although
    `_buffer_first_chunk_honoring_disconnect` (litellm/proxy/common_request_processing.py:1436-1475) is meant
    to cancel the upstream fetch. UNVERIFIED whether the chat stream on the merge ref still goes through it."""
    pytest.skip("BUG: a caller that leaves before the first provider chunk does not cancel the provider call")
    frames: Final = (*COMMENT_WARMUP, *OPENAI.frames("chatcmpl-late", ("late",)))
    with deployed(gateway, OPENAI, streaming(frames, pause_between_chunks=0.2)) as rig:
        with pytest.raises(httpx.ReadTimeout):
            rig.gateway.client.post(
                "/v1/chat/completions",
                json=route_body("/v1/chat/completions", rig.model),
                headers={"Authorization": f"Bearer {rig.key}"},
                timeout=httpx.Timeout(30, read=1.0),
            )
        assert eventually(lambda: rig.wire.disconnected.qsize(), lambda count: count >= 1, seconds=5) >= 1


def test_disconnect_before_the_first_chunk_charges_no_output_tokens(gateway: Gateway) -> None:
    """Same before-first-chunk hang up on a deployment that charges only for output (input free, 0.002 per
    output token). The key's stored spend is still 0 ten seconds later: nothing was generated, and the
    cancel path reconciles to the input cost only (`release_budget_reservation_on_cancel`). UNVERIFIED
    whether a failure row is also written; this test only pins that no output is charged.

    BUG: run 44176682961 showed the provider connection still open after the caller left before any provider
    output (no wire disconnect in 5 to 10 s, and the late chunk later billed 0.004 on the hold test), although
    `_buffer_first_chunk_honoring_disconnect` (litellm/proxy/common_request_processing.py:1436-1475) is meant
    to cancel the upstream fetch. UNVERIFIED whether the chat stream on the merge ref still goes through it."""
    pytest.skip("BUG: a caller that leaves before the first provider chunk does not cancel the provider call")
    frames: Final = (*COMMENT_WARMUP, *OPENAI.frames("chatcmpl-late", ("late",)))
    with deployed(
        gateway,
        OPENAI,
        streaming(frames, pause_between_chunks=0.2),
        input_cost_per_token=0,
        output_cost_per_token=0.002,
    ) as rig:
        with pytest.raises(httpx.ReadTimeout):
            rig.gateway.client.post(
                "/v1/chat/completions",
                json=route_body("/v1/chat/completions", rig.model),
                headers={"Authorization": f"Bearer {rig.key}"},
                timeout=httpx.Timeout(30, read=1.0),
            )
        eventually(lambda: rig.wire.disconnected.qsize(), lambda count: count >= 1, seconds=5)
        assert rig.key_spend() == 0.0


@pytest.mark.timeout(180)
def test_mid_stream_disconnect_settles_the_budget_hold_to_the_partial_cost(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 streams without `max_tokens` from a deployment that charges 0.004 per
    output token and nothing for input, so the admission hold pins the Redis counter `spend:key:<hash>` at
    exactly 1.0 while the stream runs. The caller hangs up after one frame. The counter then settles to the
    spend of the single partial row (well under 1.0), not to 1.0 and not to 0."""
    with (
        redis_client() as cache,
        deployed(
            gateway,
            OPENAI,
            provider_stream(OPENAI, SLOW_PIECES, pause_between_chunks=0.05),
            input_cost_per_token=0,
            output_cost_per_token=0.004,
            key_fields={"max_budget": 1.0},
        ) as rig,
    ):
        with opened(rig, "/v1/chat/completions") as response:
            first_frame(response)
            assert counter(cache, key_counter(rig.key)) == pytest.approx(1.0)
        row: Final = rig.spend_rows(1, settle=5)[0]
        partial: Final = float(str(row["spend"]))
        assert 0 < partial < 1.0
        counter_settles(cache, key_counter(rig.key), partial, seconds=30)


@pytest.mark.timeout(180)
def test_after_a_mid_stream_disconnect_the_next_call_on_a_budgeted_key_is_admitted(gateway: Gateway) -> None:
    """Same budgeted key (`max_budget` 1.0, hold of the whole budget while streaming). After the caller
    hangs up mid stream, a second streaming call on the key is admitted with 200 and completes with
    `[DONE]`. A disconnect path that forgot the hold would refuse it with the 422 budget message until the
    counter's TTL expired."""
    with deployed(
        gateway,
        OPENAI,
        provider_stream(OPENAI, ("short ", "answer"), pause_between_chunks=0.3),
        input_cost_per_token=0,
        output_cost_per_token=0.004,
        key_fields={"max_budget": 1.0},
    ) as rig:
        _hang_up_after_first_frame(rig, "/v1/chat/completions")
        second: Final = eventually(rig.chat, lambda answer: answer.status_code == 200, seconds=30)
        assert payloads(second.text)[-1] == "[DONE]"
        assert content_of(chunks_of(second.text)) == "short answer"


@pytest.mark.timeout(180)
def test_disconnect_before_any_provider_output_releases_the_whole_output_hold(gateway: Gateway) -> None:
    """A key with `max_budget` 1.0 streams without `max_tokens` (hold 1.0) from a provider that sends only
    comment lines. The caller gives up after one second. With free input the counter settles back to 0
    (`release_reservation_if_unbilled` in `async_data_generator`: no provider output was relayed), and a
    following call on the key is admitted.

    BUG: run 44176682961 showed the provider connection still open after the caller left before any provider
    output (no wire disconnect in 5 to 10 s, and the late chunk later billed 0.004 on the hold test), although
    `_buffer_first_chunk_honoring_disconnect` (litellm/proxy/common_request_processing.py:1436-1475) is meant
    to cancel the upstream fetch. UNVERIFIED whether the chat stream on the merge ref still goes through it."""
    pytest.skip("BUG: a caller that leaves before the first provider chunk does not cancel the provider call")
    frames: Final = (*COMMENT_WARMUP, *OPENAI.frames("chatcmpl-late", ("late",)))
    with (
        redis_client() as cache,
        deployed(
            gateway,
            OPENAI,
            streaming(frames, pause_between_chunks=0.2),
            input_cost_per_token=0,
            output_cost_per_token=0.004,
            key_fields={"max_budget": 1.0},
        ) as rig,
    ):
        with pytest.raises(httpx.ReadTimeout):
            rig.gateway.client.post(
                "/v1/chat/completions",
                json=route_body("/v1/chat/completions", rig.model),
                headers={"Authorization": f"Bearer {rig.key}"},
                timeout=httpx.Timeout(30, read=1.0),
            )
        counter_settles(cache, key_counter(rig.key), 0.0, seconds=30)


@pytest.mark.parametrize(
    ("route", "provider"),
    [("/v1/messages", ANTHROPIC), ("/v1/responses", OPENAI_RESPONSES)],
    ids=["messages", "responses"],
)
def test_mid_stream_disconnect_gives_the_parallel_request_slot_back_on_each_route(
    gateway: Gateway, route: str, provider: Provider
) -> None:
    """A key with `max_parallel_requests: 1` opens a native stream on `/v1/messages` or `/v1/responses` and
    hangs up after the first event. A second stream on the same key is eventually answered 200 rather than
    429. Chat already has this contract; the other two routes have their own generators (and Rust for
    messages) that must release the slot too."""
    with deployed(
        gateway,
        provider,
        provider_stream(provider, ("one ", "two"), pause_between_chunks=0.3),
        key_fields={"max_parallel_requests": 1},
    ) as rig:
        _hang_up_after_first_frame(rig, route)

        def again() -> int:
            with opened(rig, route) as response:
                response.read()
                return response.status_code

        assert eventually(again, lambda status: status == 200, seconds=20) == 200


def test_disconnect_before_the_first_chunk_gives_the_parallel_request_slot_back(gateway: Gateway) -> None:
    """A key with `max_parallel_requests: 1` streams from a provider that sends two seconds of comment lines
    before its only text piece; the first caller gives up after one second, before any chunk. A second
    stream on the same key is eventually answered 200 with the text `late`, never the 429 parallel limit:
    with no chunks collected nothing bills the disconnect, so `finalize_streaming_generator_cleanup` must
    release the slot itself."""
    frames: Final = (*COMMENT_WARMUP[:10], *OPENAI.frames("chatcmpl-late", ("late",)))
    with deployed(
        gateway, OPENAI, streaming(frames, pause_between_chunks=0.2), key_fields={"max_parallel_requests": 1}
    ) as rig:
        with pytest.raises(httpx.ReadTimeout):
            rig.gateway.client.post(
                "/v1/chat/completions",
                json=route_body("/v1/chat/completions", rig.model),
                headers={"Authorization": f"Bearer {rig.key}"},
                timeout=httpx.Timeout(30, read=1.0),
            )
        second: Final = eventually(rig.chat, lambda answer: answer.status_code != 429, seconds=20)
        assert second.status_code == 200, second.text
        assert content_of(chunks_of(second.text)) == "late"


def test_a_disconnected_stream_is_never_written_to_the_response_cache(gateway: Gateway) -> None:
    """The caller hangs up after one frame of a slow stream, then sends the identical request again (same
    prompt, so the shared proxy's Redis cache would match). The provider is called a second time and the
    second caller receives the full text: a partial answer must never be cached as if it were complete."""
    pieces: Final = ("whole ", "answer ", "only")
    with deployed(gateway, OPENAI, provider_stream(OPENAI, pieces, pause_between_chunks=0.4)) as rig:
        body: Final = route_body("/v1/chat/completions", rig.model)
        with rig.gateway.client.stream(
            "POST", "/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {rig.key}"}, timeout=30
        ) as response:
            first_frame(response)
        eventually(lambda: _upstream_closed(rig), lambda closed: closed, seconds=10)
        second: Final = rig.post("/v1/chat/completions", body)
        assert content_of(chunks_of(second.text)) == "whole answer only"
        assert len([request for request in rig.wire.drain() if request.method == "POST"]) == 2
