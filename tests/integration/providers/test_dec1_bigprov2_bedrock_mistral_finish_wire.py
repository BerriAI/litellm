"""Bedrock Mistral on InvokeModel: the finish reason, streamed and not streamed.

A Mistral model on Bedrock InvokeModel answers `{"outputs": [{"text": ..., "stop_reason": "stop" | "length"}]}`
and streams `chunk` frames of the same shape, with `stop_reason` inside `outputs[0]` and `null` until the
last frame. The non streamed path reads `outputs[0].stop_reason`
(`litellm/llms/bedrock/chat/invoke_transformations/amazon_mistral_transformation.py:109`). The streamed path
(`AWSEventStreamDecoder._chunk_parser`, `litellm/llms/bedrock/chat/invoke_handler.py:820`) reads
`chunk_data.get("stop_reason")` at the top level of the chunk instead, so it never sees the reason; the stream
then ends without one and the end-of-stream handler falls back to `stop`
(`litellm/litellm_core_utils/streaming_handler.py:2331`). A truncated answer (`length`) therefore reaches a
streaming caller as a normal `stop`, which hides truncation from agents that continue on `length`.

The round 1 Bedrock suite found this and left it unpinned (`index/test_dec1_bedrock_.md`, "Suspected bugs").
Confirmed from the code here; the streamed `length` test is skipped as a BUG and the controls pin what works.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request
from integration.providers.dec1_bedrock_support import (
    MISTRAL_7B,
    bedrock,
    invoke_frames,
    path_of,
    sse_events,
    streamed_text,
)
from pydantic import JsonValue

MODEL: Final = f"bedrock/{MISTRAL_7B}"


def _finish_reasons(events: list[dict[str, JsonValue]]) -> list[str]:
    reasons: Final[list[str]] = []
    for event in events:
        for choice in event.get("choices", []) or []:  # pyright: ignore[reportUnknownVariableType]
            reason = choice.get("finish_reason")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if isinstance(reason, str):
                reasons.append(reason)
    return reasons


def _mistral_document(stop_reason: str) -> Reply:
    return Reply(body=json.dumps({"outputs": [{"text": "truncat", "stop_reason": stop_reason}]}).encode())


def _mistral_stream(stop_reason: str) -> Reply:
    return invoke_frames(
        [
            {"outputs": [{"text": "trun", "stop_reason": None}]},
            {"outputs": [{"text": "cat", "stop_reason": stop_reason}]},
        ]
    )


@pytest.mark.parametrize("stop_reason", ["stop", "length"])
def test_a_non_streamed_mistral_answer_keeps_the_outputs_stop_reason(gateway: Gateway, stop_reason: str) -> None:
    """Control: Bedrock answers `{"outputs": [{"text": "truncat", "stop_reason": "<reason>"}]}` to `/invoke`.

    The caller's `choices[0].finish_reason` must be exactly that reason (`stop` or `length`), read from
    `outputs[0].stop_reason`.
    """
    with bedrock(gateway, lambda _r: _mistral_document(stop_reason), model=MODEL) as rig:
        response: Final = rig.chat(max_tokens=3)
        assert response.status_code == 200, response.text
        assert response.json()["choices"][0]["finish_reason"] == stop_reason


def test_a_streamed_mistral_stop_ends_with_a_stop_finish_reason(gateway: Gateway) -> None:
    """Control: a Mistral `invoke-with-response-stream` whose last frame has `outputs[0].stop_reason: "stop"`.

    The caller's text is `truncat` and exactly one chunk carries a finish reason, `stop`; the stream ends with
    `[DONE]` and no in-band error. This passes today only because `stop` is also the fallback.
    """
    with bedrock(gateway, lambda _r: _mistral_stream("stop"), model=MODEL) as rig:
        response: Final = rig.chat(stream=True)
        assert response.status_code == 200, response.text
        assert path_of(rig.one_post()) == f"/model/{MISTRAL_7B}/invoke-with-response-stream"
        events: Final = sse_events(response)
        assert streamed_text(events) == "truncat"
        assert _finish_reasons(events) == ["stop"]
        assert not any("error" in event for event in events)
        assert response.text.rstrip().endswith("data: [DONE]")


def test_a_streamed_mistral_length_stop_reaches_the_caller_as_length(gateway: Gateway) -> None:
    """A Mistral stream whose last frame has `outputs[0].stop_reason: "length"` (the answer hit `max_tokens`).

    The caller's one finish reason must be `length`, the same as the non streamed call. Today it is `stop`,
    because `_chunk_parser` reads `stop_reason` at the top level of the chunk (`invoke_handler.py:823`).
    """
    pytest.skip("BUG: Bedrock Mistral streams read stop_reason outside outputs[0], so a length stop streams as stop")

    def respond(_: Request) -> Reply:
        return _mistral_stream("length")

    with bedrock(gateway, respond, model=MODEL) as rig:
        response: Final = rig.chat(stream=True, max_tokens=3)
        assert response.status_code == 200, response.text
        assert _finish_reasons(sse_events(response)) == ["length"]


def test_a_streamed_mistral_length_stop_is_logged_with_the_length_finish_reason(gateway: Gateway) -> None:
    """The same truncated Mistral stream, with `stream_options.include_usage`. The final usage chunk and the
    finish chunk must agree: the chunk carrying `finish_reason` says `length`. Skipped with the same cause."""
    pytest.skip("BUG: Bedrock Mistral streams read stop_reason outside outputs[0], so a length stop streams as stop")
    with bedrock(gateway, lambda _r: _mistral_stream("length"), model=MODEL) as rig:
        response: Final = rig.chat(stream=True, stream_options={"include_usage": True})
        assert response.status_code == 200, response.text
        assert "length" in _finish_reasons(sse_events(response))
