"""Differential parity for sagemaker_chat streaming (wave-2b-beta), pinned
at the AWS event-stream PARSED-EVENT seam (the bedrock precedent: botocore
framing is transport).

v1 side: each decoded JSON event through
``AWSEventStreamDecoder(model="", is_messages_api=True).
_chunk_parser_messages_api`` (a VALIDATED StreamingChatCompletionChunk)
into ``CustomStreamWrapper("sagemaker_chat")``'s default openai branch.
v2 side: ``fold_events`` with the shared openai chunk parser and the
"openai" dialect — the validated-chunk materialization (service_tier None,
refusal None) IS the SDK-dump shape that parser normalizes.
"""

import copy
import json
import time

import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.sagemaker.common_utils import AWSEventStreamDecoder

from litellm.translation.engine.stream import fold_events
from litellm.translation.inbound.openai_chat.stream import initial_state
from litellm.translation.providers.sagemaker_chat import parse_event
from litellm.translation_seam import to_model_response_stream

MODEL = "my-endpoint"

_USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def _chunk(
    delta: dict | None = None,
    finish: str | None = None,
    usage: dict | None = None,
    choices: list | None = None,
) -> dict:
    payload: dict = {
        "id": "chunk-1",
        "object": "chat.completion.chunk",
        "created": 1718000000,
        "model": "tgi",
    }
    payload["choices"] = (
        choices
        if choices is not None
        else [
            {
                "index": 0,
                "delta": delta if delta is not None else {},
                "finish_reason": finish,
            }
        ]
    )
    if usage is not None:
        payload["usage"] = usage
    return payload


STREAMS = {
    "text": [
        _chunk({"role": "assistant", "content": "He"}),
        _chunk({"content": "llo"}),
        _chunk({}, finish="stop"),
    ],
    "tools": [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "f", "arguments": ""},
                    }
                ],
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
        _chunk({}, finish="tool_calls"),
    ],
}

USAGE_STREAM = [
    _chunk({"role": "assistant", "content": "Hi"}),
    _chunk({}, finish="stop"),
    _chunk(choices=[], usage=_USAGE),
]


def _v1_chunks(events: list, stream_options: dict | None = None) -> list:
    decoder = AWSEventStreamDecoder(model="", is_messages_api=True)
    chunks = [
        decoder._chunk_parser_messages_api(chunk_data=copy.deepcopy(event))
        for event in events
    ]
    logging = Logging(
        model=MODEL,
        messages=[{"role": "user", "content": "stream"}],
        stream=True,
        call_type="completion",
        start_time=time.time(),
        litellm_call_id="diff-sagemaker-stream",
        function_id="diff-sagemaker-stream",
    )
    wrapper = CustomStreamWrapper(
        completion_stream=iter(chunks),
        model=MODEL,
        custom_llm_provider="sagemaker_chat",
        logging_obj=logging,
        stream_options=stream_options,
    )
    return [chunk.model_dump() for chunk in wrapper]


def _v2_chunks(events: list) -> list:
    folded = fold_events(
        copy.deepcopy(events), parse_event, initial_state(MODEL, dialect="openai")
    )
    assert folded.is_ok(), folded.error.summary
    return [
        to_model_response_stream(chunk, "chatcmpl-AMBIENT").model_dump()
        for chunk in folded.ok
    ]


def _norm(chunks: list) -> str:
    return json.dumps(chunks, sort_keys=True, default=str)


@pytest.mark.parametrize("name", sorted(STREAMS))
def test_v2_stream_matches_v1(name: str, frozen_ambient) -> None:
    events = STREAMS[name]
    assert _norm(_v2_chunks(events)) == _norm(_v1_chunks(events))


def test_usage_tail_pins_the_seam_contract(frozen_ambient) -> None:
    v1 = _v1_chunks(USAGE_STREAM, stream_options={"include_usage": True})
    v2 = _v2_chunks(USAGE_STREAM)
    assert len(v1) == len(v2)
    assert _norm(v2[:-1]) == _norm(v1[: len(v2) - 1])
    assert v2[-1]["choices"] == []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert v1[-1]["usage"][key] == v2[-1]["usage"][key], key


def test_usage_tail_swallowed_without_stream_options(frozen_ambient) -> None:
    v1 = _v1_chunks(USAGE_STREAM, None)
    v2 = _v2_chunks(USAGE_STREAM)
    assert len(v1) == len(v2) - 1
    assert _norm(v1) == _norm(v2[:-1])


def test_invalid_chunk_raises_on_both_sides(frozen_ambient) -> None:
    """Decoder validation RAISES out of v1 (unlike watsonx's swallowing
    iterator); v2's parser is a loud error value."""
    bad = _chunk({"content": 5})
    result = parse_event(copy.deepcopy(bad))
    assert result.is_error()
    with pytest.raises(Exception):
        _v1_chunks([bad])
