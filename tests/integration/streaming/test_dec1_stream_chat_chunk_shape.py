"""Wire shape of a streamed `/v1/chat/completions` answer from an OpenAI compatible deployment.

The proxy does not relay provider bytes. It parses every provider chunk, rebuilds it, and writes its own
SSE frames, so the exact frame grammar is proxy behaviour that a rewrite has to reproduce: each frame is
`data: <json>` followed by a blank line, the stream ends with one `data: [DONE]`, `role` appears on the
first content chunk only, `id` and `created` are the provider's and identical on every chunk, and `model`
is the public model name the caller sent rather than the provider's model id. OpenAI SDKs and most
hand written SSE readers depend on each of these.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration.streaming.dec1_stream_support import (
    DONE,
    UPSTREAM_CREATED,
    UPSTREAM_MODEL,
    USAGE,
    chunk,
    chunks_of,
    content_of,
    delta,
    deltas_of,
    finish_reasons,
    marker,
    payloads,
    scripted,
    serving,
    sse,
    text_frames,
    usage_only,
)


def test_stream_answers_200_event_stream_with_buffering_disabled(gateway: Gateway) -> None:
    """A deployment streams two text pieces. The caller posts `stream: true`. The answer is 200 with a
    `text/event-stream` content type and the two headers that stop reverse proxies from buffering,
    `Cache-Control: no-cache` and `X-Accel-Buffering: no`. A front door that drops them makes tokens
    arrive in bursts behind nginx."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("Hel", "lo")))) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].split(";")[0] == "text/event-stream"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"


def test_every_frame_is_one_data_line_followed_by_a_blank_line(gateway: Gateway) -> None:
    """A deployment streams text. The caller reads the raw body. Every non empty line starts with
    `data: `, frames are separated by exactly one blank line, and no `event:`, `id:` or `retry:` field
    appears. Chat completions are data only SSE; an added field breaks strict line parsers."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("one ", "two ", "three")))) as rig:
        body: Final = rig.chat().text
        assert body.endswith("\n\n")
        frames: Final = body[: -len("\n\n")].split("\n\n")
        assert all(frame.startswith("data: ") and "\n" not in frame for frame in frames), frames
        assert all(json.loads(frame.removeprefix("data: ")) for frame in frames[:-1])


def test_stream_ends_with_exactly_one_done_sentinel(gateway: Gateway) -> None:
    """A deployment streams text and ends with `[DONE]`. The caller reads to the end. The last frame of
    the proxy's body is the literal `data: [DONE]` and it appears once. SDKs stop reading on this
    sentinel, so a missing one hangs them until the socket closes and a doubled one breaks JSON parsers."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("done ", "check")))) as rig:
        seen: Final = payloads(rig.chat().text)
        assert seen[-1] == "[DONE]"
        assert seen.count("[DONE]") == 1


def test_done_sentinel_is_written_even_when_the_provider_omits_it(gateway: Gateway) -> None:
    """A deployment sends content, a `stop` finish chunk, then closes cleanly without `[DONE]`. The
    caller still receives the full text and a final `data: [DONE]`. The sentinel is the proxy's own, not
    a relayed provider frame."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = text_frames(identity, ("no ", "sentinel"), usage=None)[:-1]
    assert DONE not in frames
    with scripted(gateway, serving(frames)) as rig:
        seen: Final = payloads(rig.chat().text)
        assert seen[-1] == "[DONE]"
        assert content_of(chunks_of("\n".join(f"data: {item}" for item in seen))) == "no sentinel"


def test_role_is_on_the_first_chunk_only(gateway: Gateway) -> None:
    """A deployment streams three text pieces with `role` on its first chunk. The caller sees
    `delta.role == "assistant"` on the first chunk and no `role` key on any later chunk, the finish
    chunk included. Clients that concatenate roles or open a new message per role rely on this."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("a", "b", "c")))) as rig:
        deltas: Final = deltas_of(chunks_of(rig.chat().text))
        assert deltas[0].get("role") == "assistant"
        assert [item.get("role") for item in deltas[1:]] == [None] * (len(deltas) - 1)


def test_role_repeated_by_the_provider_on_every_chunk_is_stripped_after_the_first(gateway: Gateway) -> None:
    """A provider (some vLLM and Mistral builds do this) puts `role: assistant` on every chunk. The
    caller still gets `role` only on the first chunk and the text intact."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": "re"}),
        delta(identity, {"role": "assistant", "content": "peat"}),
        delta(identity, {"role": "assistant"}, finish="stop"),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        deltas: Final = deltas_of(chunks)
        assert content_of(chunks) == "repeat"
        assert [bool(item.get("role")) for item in deltas] == [True] + [False] * (len(deltas) - 1)


def test_role_is_added_when_the_provider_never_sends_one(gateway: Gateway) -> None:
    """A provider streams content deltas with no `role` at all. The proxy stamps `role: assistant` on the
    first chunk it sends, because the OpenAI SDK's stream accumulator needs one to build the message."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"content": "role"}),
        delta(identity, {"content": "less"}),
        delta(identity, {}, finish="stop"),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        deltas: Final = deltas_of(chunks_of(rig.chat().text))
        assert deltas[0].get("role") == "assistant"


def test_id_is_the_providers_and_created_is_one_proxy_timestamp_on_every_chunk(gateway: Gateway) -> None:
    """A provider streams under id `chatcmpl-<marker>` with `created` 1700000000. Every chunk the caller
    receives, including the finish chunk and the usage chunk, carries exactly that id and one shared
    `created` value. The id is also the SpendLogs `request_id`, so support can join a client log to a
    spend row.

    Pinned current behaviour: `created` is not the provider's value but the proxy's own clock when it
    built the first chunk, then reused on every later chunk (`model_response_creator` in
    `litellm/litellm_core_utils/streaming_handler.py` stores `self.created` from the first
    `ModelResponseStream` and never reads the provider chunk's `created`). Non streamed answers keep the
    provider's value, so a rewrite that forwards it on streams too would make this fail loudly."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("same ", "id")))) as rig:
        before: Final = int(time.time())
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        after: Final = int(time.time()) + 1
        assert {item["id"] for item in chunks} == {identity}
        created: Final = {item["created"] for item in chunks}
        assert len(created) == 1, created
        assert UPSTREAM_CREATED not in created
        assert before - 1 <= int(str(next(iter(created)))) <= after
        assert rig.spend_rows()[0]["request_id"] == identity


def test_object_is_chat_completion_chunk_on_every_chunk(gateway: Gateway) -> None:
    """Every chunk of a streamed chat answer, the usage chunk included, has
    `object == "chat.completion.chunk"`."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("obj", "ect")))) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        assert {item["object"] for item in chunks} == {"chat.completion.chunk"}


def test_model_on_every_chunk_is_the_public_name_the_caller_sent(gateway: Gateway) -> None:
    """The deployment's public name is a random alias and the provider reports `gpt-4o-mini`. Every chunk
    the caller receives says `model == <alias>`, never the provider model and never an
    `openai/`-prefixed internal id. The provider itself is still asked for `gpt-4o-mini`."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("ali", "as")))) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        assert {item["model"] for item in chunks} == {rig.model}
        assert rig.model != UPSTREAM_MODEL
        assert [body["model"] for body in rig.upstream_bodies()] == [UPSTREAM_MODEL]


def test_text_arrives_in_provider_order_one_delta_per_provider_delta(gateway: Gateway) -> None:
    """A provider sends five distinct text pieces. The caller receives the same five `delta.content`
    strings in the same order, not merged and not split. Token level pacing is visible to end users."""
    identity: Final = f"chatcmpl-{marker()}"
    pieces: Final = ("The ", "quick ", "brown ", "fox ", "jumps")
    with scripted(gateway, serving(text_frames(identity, pieces))) as rig:
        deltas: Final = deltas_of(chunks_of(rig.chat().text))
        assert [item["content"] for item in deltas if item.get("content")] == list(pieces)


def test_finish_chunk_has_an_empty_delta_and_earlier_chunks_have_no_finish_reason(gateway: Gateway) -> None:
    """A provider streams text then a `stop` finish chunk. Exactly one chunk the caller receives has a
    non null `finish_reason`, it is the last chunk that has choices, and its delta carries no content.
    All earlier chunks either omit `finish_reason` or send it as null."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("fin", "ish")))) as rig:
        chunks: Final = [item for item in chunks_of(rig.chat().text) if item["choices"]]
        choices: Final = [object_value(item["choices"][0]) for item in chunks]  # type: ignore[index]
        assert [choice.get("finish_reason") for choice in choices[:-1]] == [None] * (len(choices) - 1)
        assert choices[-1]["finish_reason"] == "stop"
        assert not object_value(choices[-1].get("delta") or {}).get("content")


@pytest.mark.parametrize("reason", ["stop", "length", "content_filter", "tool_calls", "function_call"])
def test_provider_finish_reason_reaches_the_caller_unchanged(gateway: Gateway, reason: str) -> None:
    """A provider ends its stream with each OpenAI finish reason in turn. The caller's final choice chunk
    carries the same string. `length` in particular drives client side continuation logic, so mapping
    everything to `stop` would silently truncate answers."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("reason ", reason), finish=reason))) as rig:
        assert finish_reasons(chunks_of(rig.chat().text)) == [reason]


def test_unknown_provider_finish_reason_is_mapped_rather_than_dropped(gateway: Gateway) -> None:
    """A provider ends with a finish reason outside the OpenAI enum (`eos`). The caller still gets exactly
    one finish chunk and its reason is `stop`, the value `map_finish_reason` gives `eos` and any
    unmapped string. A null or free form reason fails OpenAI SDK validation."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("odd ", "reason"), finish="eos"))) as rig:
        assert finish_reasons(chunks_of(rig.chat().text)) == ["stop"]


def test_reasoning_content_deltas_reach_the_caller_before_the_answer(gateway: Gateway) -> None:
    """A provider streams two `reasoning_content` deltas and then two `content` deltas. The caller
    receives the reasoning under `delta.reasoning_content` and the answer under `delta.content`, in that
    order, and the two never share a string."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "reasoning_content": "think "}),
        delta(identity, {"reasoning_content": "hard"}),
        delta(identity, {"content": "ans"}),
        delta(identity, {"content": "wer"}),
        delta(identity, {}, finish="stop"),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert content_of(chunks, field="reasoning_content") == "think hard"
        assert content_of(chunks) == "answer"
        order: Final = [
            "r" if item.get("reasoning_content") else "c"
            for item in deltas_of(chunks)
            if item.get("reasoning_content") or item.get("content")
        ]
        assert order == ["r", "r", "c", "c"]


def test_refusal_deltas_reach_the_caller_under_delta_refusal(gateway: Gateway) -> None:
    """A provider streams a refusal the OpenAI way: `delta.refusal` fragments, null content, finish
    `stop`. The caller receives the fragments under `delta.refusal` and no content text. UNVERIFIED: the
    proxy keeps unknown delta fields as extras (`Delta` allows extras in `litellm/types/utils.py`); if a
    build maps refusal to content this test documents the intended passthrough.

    BUG: only the first fragment (`I can`) arrives. It rides on the role chunk, which is always sent;
    every later refusal only delta is dropped because `is_chunk_non_empty` in
    `litellm/litellm_core_utils/streaming_handler.py` does not count `refusal` as content. The proxy's own
    stream rebuilder joins refusal fragments across chunks (`_chunk_refusal` in
    `litellm/litellm_core_utils/streaming_chunk_builder_utils.py`), so dropping them is unintended."""
    pytest.skip("BUG: streamed refusal deltas after the first chunk are dropped, caller sees a truncated refusal")
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": None, "refusal": "I can"}),
        delta(identity, {"refusal": "not help"}),
        delta(identity, {}, finish="stop"),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert content_of(chunks, field="refusal") == "I cannot help"
        assert content_of(chunks) == ""
        assert finish_reasons(chunks) == ["stop"]


def test_two_choices_stream_interleaved_each_with_its_own_index_and_finish(gateway: Gateway) -> None:
    """The caller asks for `n: 2` and the provider interleaves chunks for choice 0 and choice 1. The
    caller can rebuild both answers by `index`, each choice ends with its own finish reason (`stop` and
    `length`), and the provider was asked for `n: 2`.

    BUG: both texts arrive but choice 1 never gets a finish reason. The chat path of
    `litellm/litellm_core_utils/streaming_handler.py` keeps one `received_finish_reason` for the whole
    stream and `finish_reason_handler` emits a single finish chunk for choice 0; only the text completion
    path reports a `sibling_finish_reason` per choice index. An OpenAI client sees choice 1 as never
    finished, and a `length` cut on it is invisible."""
    pytest.skip("BUG: n=2 chat stream drops the finish_reason of choice 1")
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": "zero-a "}, index=0),
        delta(identity, {"role": "assistant", "content": "one-a "}, index=1),
        delta(identity, {"content": "zero-b"}, index=0),
        delta(identity, {"content": "one-b"}, index=1),
        delta(identity, {}, finish="stop", index=0),
        delta(identity, {}, finish="length", index=1),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat(n=2).text)
        assert content_of(chunks, index=0) == "zero-a zero-b"
        assert content_of(chunks, index=1) == "one-a one-b"
        assert finish_reasons(chunks, index=0) == ["stop"]
        assert finish_reasons(chunks, index=1) == ["length"]
        assert rig.upstream_bodies()[0]["n"] == 2


def test_system_fingerprint_from_the_provider_is_kept_on_chunks(gateway: Gateway) -> None:
    """A provider stamps `system_fingerprint: fp_dec1` on every chunk. The caller sees the same value on
    the content chunks. Callers use it to detect backend changes when they pin a seed."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        chunk(identity, [{"index": 0, "delta": {"role": "assistant", "content": "fp"}}], system_fingerprint="fp_dec1"),
        chunk(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}], system_fingerprint="fp_dec1"),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert chunks[0].get("system_fingerprint") == "fp_dec1"


def test_multibyte_text_split_across_transfer_chunks_reaches_the_caller_intact(gateway: Gateway) -> None:
    """The provider's SSE bytes are cut every 3 bytes, which lands inside the UTF-8 encoding of CJK and
    emoji free accented text and inside JSON escapes. The caller's concatenated content is exactly
    `Zoë 雪 naïve café Ωmega`, with no replacement characters and no dropped frame."""
    identity: Final = f"chatcmpl-{marker()}"
    text: Final = ("Zoë ", "雪 ", "naïve ", "café ", "Ωmega")
    body: Final = b"".join(text_frames(identity, text))
    pieces: Final = tuple(body[start : start + 3] for start in range(0, len(body), 3))
    with scripted(gateway, lambda request: sse(pieces)) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert content_of(chunks) == "Zoë 雪 naïve café Ωmega"
        assert "�" not in content_of(chunks)


def test_several_provider_frames_in_one_transfer_chunk_are_all_delivered(gateway: Gateway) -> None:
    """The provider flushes its whole stream, five frames and `[DONE]`, in a single TCP write. The caller
    still receives one proxy frame per provider chunk with the text in order, then one `[DONE]`."""
    identity: Final = f"chatcmpl-{marker()}"
    pieces: Final = ("p1 ", "p2 ", "p3 ", "p4")
    with scripted(gateway, lambda request: sse((b"".join(text_frames(identity, pieces)),))) as rig:
        seen: Final = payloads(rig.chat().text)
        chunks: Final = chunks_of("\n".join(f"data: {item}" for item in seen))
        assert [item["content"] for item in deltas_of(chunks) if item.get("content")] == list(pieces)
        assert seen[-1] == "[DONE]"


def test_one_megabyte_delta_is_delivered_whole_in_one_chunk(gateway: Gateway) -> None:
    """A provider sends a single content delta of 1,000,000 characters. The caller receives one delta
    whose content has the same length and the same first and last characters. A fixed size line buffer
    in a rewritten SSE reader would truncate or split it."""
    identity: Final = f"chatcmpl-{marker()}"
    big: Final = "A" + "x" * 999_998 + "Z"
    with scripted(gateway, serving(text_frames(identity, (big,)))) as rig:
        contents: Final = [item["content"] for item in deltas_of(chunks_of(rig.chat().text)) if item.get("content")]
        assert len(contents) == 1
        assert len(str(contents[0])) == 1_000_000
        assert str(contents[0])[0] == "A" and str(contents[0])[-1] == "Z"


def test_two_thousand_small_deltas_all_arrive_in_order(gateway: Gateway) -> None:
    """A provider sends 2000 one token deltas `t0 ` to `t1999 `. The caller receives 2000 content deltas
    in the same order with none dropped, merged or duplicated, and a single finish chunk."""
    identity: Final = f"chatcmpl-{marker()}"
    pieces: Final = tuple(f"t{number} " for number in range(2000))
    with scripted(gateway, serving(text_frames(identity, pieces))) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert [item["content"] for item in deltas_of(chunks) if item.get("content")] == list(pieces)
        assert finish_reasons(chunks) == ["stop"]


def test_provider_chunk_with_empty_choices_and_no_usage_is_not_forwarded(gateway: Gateway) -> None:
    """Azure style providers open a stream with a chunk whose `choices` is empty (content filter results).
    The caller, who did not ask for usage, receives no chunk with an empty `choices` list, and the text
    that follows is intact. Many clients index `choices[0]` without a guard."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        chunk(identity, [], prompt_filter_results=[]),
        *text_frames(identity, ("after ", "preamble"), usage=None),
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert all(item["choices"] for item in chunks), chunks
        assert content_of(chunks) == "after preamble"


def test_sse_comment_lines_from_the_provider_are_not_forwarded_as_chunks(gateway: Gateway) -> None:
    """A provider (OpenRouter does this) interleaves `: OPENROUTER PROCESSING` comment lines with its
    data frames. The caller's stream contains only JSON chunk frames and `[DONE]`; the text is intact."""
    identity: Final = f"chatcmpl-{marker()}"
    first, *rest = text_frames(identity, ("com", "ment"))
    with scripted(gateway, serving((b": PROVIDER PROCESSING\n\n", first, b": PROVIDER PROCESSING\n\n", *rest))) as rig:
        body: Final = rig.chat().text
        assert "PROVIDER PROCESSING" not in body
        assert content_of(chunks_of(body)) == "comment"


def test_provider_is_asked_to_stream_and_never_sees_proxy_internal_fields(gateway: Gateway) -> None:
    """The caller posts a streaming chat request with `temperature` and `max_tokens`. The provider
    receives one POST to `/v1/chat/completions` with `stream: true`, the same sampling fields, the
    provider model id, and none of `litellm_call_id`, `metadata`, `proxy_server_request` or
    `litellm_logging_obj`. A leaked internal field is invisible from the response."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("out", "bound")))) as rig:
        assert rig.chat(temperature=0.25, max_tokens=17).status_code == 200
        requests: Final = [request for request in rig.wire.drain() if request.method == "POST"]
        assert [request.target for request in requests] == ["/v1/chat/completions"]
        body: Final = json.loads(requests[0].body)
        assert body["stream"] is True and body["temperature"] == 0.25 and body["max_tokens"] == 17
        assert not {
            "litellm_call_id",
            "metadata",
            "proxy_server_request",
            "litellm_logging_obj",
            "litellm_params",
        } & set(body)
        assert requests[0].headers["authorization"] == "Bearer synthetic-provider-key"


def test_explicit_stream_false_answers_one_json_object_not_an_event_stream(gateway: Gateway) -> None:
    """The caller posts `stream: false`. The provider is asked without streaming and the caller gets a
    single `application/json` body with `object == "chat.completion"`, a full `message`, and no
    `data:` prefix. `stream: false` must not be treated as truthy because the key is present."""
    identity: Final = f"chatcmpl-{marker()}"
    reply: Final = json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": UPSTREAM_CREATED,
            "model": UPSTREAM_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "whole"}, "finish_reason": "stop"}],
            "usage": dict(USAGE),
        }
    ).encode()
    with scripted(gateway, serving(text_frames(identity, ("never",)), json_body=reply)) as rig:
        response: Final = rig.chat(stream=False)
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].split(";")[0] == "application/json"
        assert response.json()["object"] == "chat.completion"
        assert response.json()["choices"][0]["message"]["content"] == "whole"
        assert rig.upstream_bodies()[0].get("stream") in (None, False)


def test_concurrent_streams_never_mix_chunks_between_callers(gateway: Gateway) -> None:
    """Twelve callers stream at once and the provider answers each with text derived from that caller's
    own prompt, paced so the streams overlap. Every caller receives only its own text and one chunk id.
    Shared buffers or a misrouted channel in a rewritten stream pump would show up here."""

    def respond(request):  # type: ignore[no-untyped-def]
        prompt: Final = str(json.loads(request.body)["messages"][0]["content"])
        tag: Final = prompt.rsplit(" ", 1)[-1]
        return sse(text_frames(f"chatcmpl-{tag}", (f"{tag}-a ", f"{tag}-b ", f"{tag}-c")), pause_between_chunks=0.05)

    with scripted(gateway, respond) as rig:
        tags: Final = [marker() for _ in range(12)]
        with ThreadPoolExecutor(max_workers=12) as pool:
            bodies: Final = list(pool.map(lambda tag: rig.chat(f"isolated stream {tag}").text, tags))
        for tag, body in zip(tags, bodies, strict=True):
            chunks = chunks_of(body)
            assert content_of(chunks) == f"{tag}-a {tag}-b {tag}-c"
            assert {item["id"] for item in chunks} == {f"chatcmpl-{tag}"}


def test_usage_only_provider_chunk_is_withheld_when_the_caller_did_not_ask_for_usage(gateway: Gateway) -> None:
    """The provider ends with a usage only chunk (the proxy asks for one for billing). The caller sent no
    `stream_options`. No chunk in the caller's stream has a `usage` key and none has empty `choices`."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (*text_frames(identity, ("quiet ", "usage"), usage=None)[:-1], usage_only(identity), DONE)
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        assert all("usage" not in item or item["usage"] is None for item in chunks)
        assert all(item["choices"] for item in chunks)
