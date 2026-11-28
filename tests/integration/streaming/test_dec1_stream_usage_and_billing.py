"""Usage chunks and the spend record of a streamed call.

Two separate things are pinned here. First, what the caller sees: with `stream_options.include_usage`
the proxy sends one extra chunk after the finish chunk with `choices: []` and a `usage` object, directly
before `[DONE]`; without it no chunk has `usage`. Second, what gets billed: whatever the caller asked,
the proxy asks the provider for usage, and the `LiteLLM_SpendLogs` row written after the stream carries
the provider's token counts, the configured price times those counts, and the same identifiers the
caller saw. A streamed call has to cost exactly what the same call costs without streaming.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration.streaming.dec1_stream_support import (
    DETAILED_USAGE,
    DONE,
    USAGE,
    chunk,
    chunks_of,
    completion_json,
    delta,
    marker,
    payloads,
    scripted,
    serving,
    text_frames,
    usage_chunks,
)

_INPUT_PRICE: Final = 0.001
_OUTPUT_PRICE: Final = 0.002
_CACHE_READ_PRICE: Final = 0.0001


def test_include_usage_adds_one_choiceless_usage_chunk_directly_before_done(gateway: Gateway) -> None:
    """The caller streams with `stream_options: {"include_usage": true}` and the provider reports 11
    prompt and 4 completion tokens. The second to last frame is a chunk with `choices == []` and
    `usage == {prompt_tokens: 11, completion_tokens: 4, total_tokens: 15}`, the last frame is `[DONE]`,
    and the usage chunk comes after the chunk that carries `finish_reason`."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("use", "age")))) as rig:
        body: Final = rig.chat(stream_options={"include_usage": True}).text
        chunks: Final = chunks_of(body)
        assert payloads(body)[-1] == "[DONE]"
        assert chunks[-1]["choices"] == []
        usage: Final = object_value(chunks[-1]["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (11, 4, 15)
        assert object_value(chunks[-2]["choices"][0])["finish_reason"] == "stop"  # type: ignore[index]
        assert len(usage_chunks(chunks)) == 1


def test_usage_is_absent_from_every_chunk_before_the_usage_chunk(gateway: Gateway) -> None:
    """With `include_usage` on, only the final chunk carries `usage`. Content chunks and the finish chunk
    have no `usage` key (the proxy omits it instead of sending `usage: null`). UNVERIFIED on the null
    versus absent detail: chunks are serialised with `exclude_none=True` in `_serialize_chat_chunk`, so
    absent is expected; OpenAI itself sends `usage: null` on those chunks."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("only ", "last")))) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        assert all(item.get("usage") is None for item in chunks[:-1])
        assert chunks[-1].get("usage") is not None


def test_include_usage_false_sends_no_usage_chunk(gateway: Gateway) -> None:
    """The caller streams with `stream_options: {"include_usage": false}`. No chunk has `usage` and no
    chunk has empty `choices`, while the row in SpendLogs still records 11 prompt and 4 completion
    tokens because the proxy asked the provider for usage on its own."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("no ", "usage")))) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": False}).text)
        assert usage_chunks(chunks) == []
        assert all(item["choices"] for item in chunks)
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 4)


def test_usage_chunk_carries_cached_and_reasoning_token_details(gateway: Gateway) -> None:
    """The provider reports `prompt_tokens_details.cached_tokens: 8` and
    `completion_tokens_details.reasoning_tokens: 3`. The caller's usage chunk carries both nested values
    unchanged next to the totals 20, 9 and 29. Cost dashboards built on the client side read these."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("de", "tail"), usage=DETAILED_USAGE))) as rig:
        usage: Final = object_value(chunks_of(rig.chat(stream_options={"include_usage": True}).text)[-1]["usage"])
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (20, 9, 29)
        assert object_value(usage["prompt_tokens_details"])["cached_tokens"] == 8
        assert object_value(usage["completion_tokens_details"])["reasoning_tokens"] == 3


def test_usage_sent_on_the_provider_finish_chunk_is_moved_to_its_own_final_chunk(gateway: Gateway) -> None:
    """Some providers attach `usage` to the same chunk that carries `finish_reason` instead of sending a
    separate chunk. With `include_usage` on, the caller still gets the finish chunk without usage and
    then a separate `choices: []` chunk with the totals 11, 4 and 15."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": "merged"}),
        chunk(identity, [{"index": 0, "delta": {}, "finish_reason": "stop"}], usage=dict(USAGE)),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        finishing: Final = [item for item in chunks if item["choices"] and object_value(item["choices"][0]).get("finish_reason")]  # type: ignore[index]
        assert len(finishing) == 1 and finishing[0].get("usage") is None
        assert chunks[-1]["choices"] == []
        assert object_value(chunks[-1]["usage"])["total_tokens"] == 15


def test_usage_is_estimated_when_the_provider_reports_none(gateway: Gateway) -> None:
    """A provider ignores `include_usage` and never reports usage. The caller asked for usage and still
    receives a final usage chunk whose `completion_tokens` and `prompt_tokens` are positive and whose
    `total_tokens` is their sum; the proxy counts tokens itself. The SpendLogs row carries the same
    counts. A rewrite that only relays provider usage would bill zero for such providers."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("count ", "these ", "tokens"), usage=None))) as rig:
        chunks: Final = chunks_of(rig.chat(stream_options={"include_usage": True}).text)
        usage: Final = object_value(chunks[-1]["usage"])
        assert int(str(usage["prompt_tokens"])) > 0 and int(str(usage["completion_tokens"])) > 0
        assert usage["total_tokens"] == int(str(usage["prompt_tokens"])) + int(str(usage["completion_tokens"]))
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (usage["prompt_tokens"], usage["completion_tokens"])


def test_provider_is_asked_for_usage_even_when_the_caller_did_not(gateway: Gateway) -> None:
    """The caller sends no `stream_options`. The provider's request body still has
    `stream_options.include_usage == true`, added by the proxy so the stream can be billed from provider
    counts. The caller's own request is otherwise unchanged."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("ask ", "anyway")))) as rig:
        assert rig.chat().status_code == 200
        assert object_value(rig.upstream_bodies()[0]["stream_options"])["include_usage"] is True


def test_spend_row_of_a_stream_has_provider_tokens_and_configured_price(gateway: Gateway) -> None:
    """A deployment priced at 0.001 per input token and 0.002 per output token streams an answer whose
    provider usage is 11 and 4. One SpendLogs row lands with `prompt_tokens 11`, `completion_tokens 4`,
    `total_tokens 15` and `spend == 11 * 0.001 + 4 * 0.002 == 0.019`."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("pri", "ced"))), input_cost_per_token=_INPUT_PRICE, output_cost_per_token=_OUTPUT_PRICE) as rig:
        assert rig.chat().status_code == 200
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (11, 4, 15)
        assert float(str(row["spend"])) == pytest.approx(11 * _INPUT_PRICE + 4 * _OUTPUT_PRICE)


def test_cached_prompt_tokens_in_a_stream_are_billed_at_the_cache_read_price(gateway: Gateway) -> None:
    """The provider reports 20 prompt tokens of which 8 were cached, and 9 completion tokens. With a
    cache read price of 0.0001 the row's spend is `12 * 0.001 + 8 * 0.0001 + 9 * 0.002 == 0.0308`, not
    the 0.038 that billing all 20 prompt tokens at full price would give."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(
        gateway,
        serving(text_frames(identity, ("cache ", "read"), usage=DETAILED_USAGE)),
        input_cost_per_token=_INPUT_PRICE,
        output_cost_per_token=_OUTPUT_PRICE,
        cache_read_input_token_cost=_CACHE_READ_PRICE,
    ) as rig:
        assert rig.chat().status_code == 200
        row: Final = rig.spend_rows()[0]
        assert float(str(row["spend"])) == pytest.approx(12 * _INPUT_PRICE + 8 * _CACHE_READ_PRICE + 9 * _OUTPUT_PRICE)


def test_streamed_and_non_streamed_calls_write_equivalent_spend_rows(gateway: Gateway) -> None:
    """The same deployment answers one call with `stream: true` and one with `stream: false`, the provider
    reporting the same usage both times. The two SpendLogs rows agree on `call_type`, `status`, `model`,
    `model_group`, `custom_llm_provider`, the three token counts and `spend`. Streaming is a transport
    choice; a rewrite must not let it change what gets billed or how the row is labelled."""
    stream_id: Final = f"chatcmpl-{marker()}"
    json_id: Final = f"chatcmpl-{marker()}"
    with scripted(
        gateway,
        serving(text_frames(stream_id, ("same ", "bill")), json_body=completion_json(json_id, "same bill")),
        input_cost_per_token=_INPUT_PRICE,
        output_cost_per_token=_OUTPUT_PRICE,
    ) as rig:
        assert rig.chat().status_code == 200
        assert rig.chat(stream=False).status_code == 200
        rows: Final = {row["request_id"]: row for row in rig.spend_rows(2)}
        compared: Final = ("call_type", "status", "model", "model_group", "custom_llm_provider", "prompt_tokens", "completion_tokens", "total_tokens")
        assert [rows[stream_id][name] for name in compared] == [rows[json_id][name] for name in compared]
        assert float(str(rows[stream_id]["spend"])) == pytest.approx(float(str(rows[json_id]["spend"])))


def test_spend_row_labels_a_streamed_chat_call(gateway: Gateway) -> None:
    """After a successful stream the SpendLogs row has `call_type == "acompletion"`,
    `status == "success"`, `model_group ==` the public model name, `custom_llm_provider == "openai"`,
    `cache_hit == "False"`, and a `completionStartTime` (first token time) that is set and not after
    `endTime`. Latency dashboards compute time to first token from these two columns."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("label ", "me")))) as rig:
        assert rig.chat().status_code == 200
        row: Final = rig.spend_rows()[0]
        assert row["call_type"] == "acompletion"
        assert row["status"] == "success"
        assert row["model_group"] == rig.model
        assert row["custom_llm_provider"] == "openai"
        assert row["cache_hit"] == "False"
        assert row["completionStartTime"] is not None
        assert str(row["startTime"]) <= str(row["completionStartTime"]) <= str(row["endTime"])


def test_spend_row_call_id_matches_the_call_id_response_header(gateway: Gateway) -> None:
    """A streamed response carries `x-litellm-call-id`. The SpendLogs row for that call has the same value
    in `litellm_call_id`. Operators trace a user reported stream to its spend row with this header."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("trace ", "me")))) as rig:
        response: Final = rig.chat()
        call_id: Final = response.headers["x-litellm-call-id"]
        assert call_id
        assert rig.spend_rows()[0]["litellm_call_id"] == call_id


def test_caller_supplied_call_id_header_is_echoed_and_logged(gateway: Gateway) -> None:
    """The caller sends `x-litellm-call-id: <uuid>` on a streaming request. The response header echoes
    that value and the SpendLogs row stores it (`resolve_litellm_call_id` accepts any non empty client
    id up to its length cap). Callers use this to correlate their own request ids with proxy logs."""
    import uuid

    identity: Final = f"chatcmpl-{marker()}"
    supplied: Final = str(uuid.uuid4())
    with scripted(gateway, serving(text_frames(identity, ("own ", "id")))) as rig:
        response: Final = rig.chat(headers={"x-litellm-call-id": supplied})
        assert response.headers["x-litellm-call-id"] == supplied
        assert rig.spend_rows()[0]["litellm_call_id"] == supplied


def test_key_spend_rises_by_the_cost_of_the_stream(gateway: Gateway) -> None:
    """A fresh key streams one priced answer (cost 0.019). The key's `spend` in
    `LiteLLM_VerificationToken` converges to 0.019. Budgets are enforced from this column, so a stream
    whose cost is logged but never added to the key lets a key overspend."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("add ", "up"))), input_cost_per_token=_INPUT_PRICE, output_cost_per_token=_OUTPUT_PRICE) as rig:
        assert rig.chat().status_code == 200
        spent: Final = eventually(rig.key_spend, lambda value: value > 0, seconds=70)
        assert spent == pytest.approx(0.019)


def test_exactly_one_spend_row_per_stream(gateway: Gateway) -> None:
    """Three streams are sent one after another with `include_usage` on. Exactly three SpendLogs rows
    exist for the key once writes settle, one per chunk id. The usage chunk, the finish chunk and the
    end of the generator are three places a stream can accidentally log from."""
    identities: Final = [f"chatcmpl-{marker()}" for _ in range(3)]
    queue: Final = list(identities)
    with scripted(gateway, lambda request: serving(text_frames(queue.pop(0), ("row ", "once")))(request)) as rig:
        for _ in identities:
            assert rig.chat(stream_options={"include_usage": True}).status_code == 200
        assert sorted(str(row["request_id"]) for row in rig.spend_rows(3)) == sorted(identities)


def test_spend_metadata_usage_object_keeps_token_details(gateway: Gateway) -> None:
    """The provider reports cached and reasoning token details on a stream. The SpendLogs row's
    `metadata.usage_object` holds `prompt_tokens_details.cached_tokens == 8` and
    `completion_tokens_details.reasoning_tokens == 3`. The key name comes from the spend logs metadata
    builder in `litellm/proxy/spend_tracking/spend_tracking_utils.py`."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(text_frames(identity, ("meta ", "data"), usage=DETAILED_USAGE))) as rig:
        assert rig.chat().status_code == 200
        usage: Final = object_value(object_value(rig.spend_rows()[0]["metadata"])["usage_object"])
        assert object_value(usage["prompt_tokens_details"])["cached_tokens"] == 8
        assert object_value(usage["completion_tokens_details"])["reasoning_tokens"] == 3


def test_text_completion_stream_chunks_use_the_text_completion_shape(gateway: Gateway) -> None:
    """The caller streams `/v1/completions` with a prompt. The provider answers on its
    `/v1/completions` route with `text` fragments. Every chunk the caller gets has
    `object == "text_completion"`, the text under `choices[0].text` (no `delta`), the final choice has
    `finish_reason == "stop"`, and the stream ends with `[DONE]`. The SpendLogs row's `call_type` is
    `atext_completion`."""
    identity: Final = f"cmpl-{marker()}"

    def piece(text: str, finish: str | None = None) -> bytes:
        from integration.streaming.dec1_stream_support import data

        return data({"id": identity, "object": "text_completion", "created": 1700000000, "model": "gpt-3.5-turbo-instruct", "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}]})

    frames: Final = (piece("plain "), piece("text"), piece("", "stop"), DONE)
    with scripted(gateway, serving(frames), model="text-completion-openai/gpt-3.5-turbo-instruct") as rig:
        response: Final = rig.post("/v1/completions", {"model": rig.model, "prompt": f"complete {marker()}", "stream": True})
        assert response.status_code == 200, response.text
        chunks: Final = chunks_of(response.text)
        assert {item["object"] for item in chunks} == {"text_completion"}
        choices: Final = [object_value(item["choices"][0]) for item in chunks if item["choices"]]  # type: ignore[index]
        assert "".join(str(choice.get("text") or "") for choice in choices) == "plain text"
        assert all("delta" not in choice for choice in choices)
        assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == ["stop"]
        assert payloads(response.text)[-1] == "[DONE]"
        assert rig.spend_rows()[0]["call_type"] == "atext_completion"
