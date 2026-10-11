from dataclasses import replace
import json
from typing import Final

import pytest
from integration._support.client import Gateway, JSON_OBJECT
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import TranslationTestCase
from integration.translation.chat_completions.bases.anthropic import CLAUDE_SONNET_5_5_TEST_CASE
from integration.translation.runner import assert_translation
from pydantic import JsonValue

CLAUDE_SONNET_5_5_BASIC_REPLAY_TEST_CASE: Final[TranslationTestCase] = replace(
    CLAUDE_SONNET_5_5_TEST_CASE,
    scenario="basic_completion_replay",
)

ANTHROPIC_STREAM_REPLY: Final = b"".join(
    (
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_stream","type":"message",'
        b'"role":"assistant","model":"claude-sonnet-5-5","content":[],"usage":{"input_tokens":8,"output_tokens":1}}}\n\n',
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        b'"content_block":{"type":"text","text":""}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        b'"delta":{"type":"text_delta","text":"Hello"}}\n\n',
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        b'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
        b'"stop_sequence":null},"usage":{"output_tokens":2}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )
)


@pytest.mark.parametrize("case", [CLAUDE_SONNET_5_5_BASIC_REPLAY_TEST_CASE], ids=lambda case: case.id)
def test_anthropic_basic_completion_replay(
    case: TranslationTestCase, gateway: Gateway, provider: SharedProvider
) -> None:
    assert_translation(case, gateway, provider)


def test_anthropic_streaming_completion_replay(gateway: Gateway, provider: SharedProvider) -> None:
    request_body: Final[dict[str, JsonValue]] = {
        "model": "anthropic/claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "stream": True,
        "cache": {"no-cache": True},
    }
    expected_provider_body: Final[dict[str, JsonValue]] = {
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}],
        "max_tokens": 16,
        "stream": True,
    }
    provider.expect(Reply(content_type="text/event-stream", body=ANTHROPIC_STREAM_REPLY))

    response: Final = gateway.request("POST", "/v1/chat/completions", request_body)
    received: Final = provider.received()

    assert response.status_code == 200, response.text
    assert len(received) == 1
    assert received[0].target == "/v1/messages"
    assert JSON_OBJECT.validate_json(received[0].body) == expected_provider_body
    data_lines: Final = tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )
    choices: Final = tuple(
        JSON_OBJECT.validate_python(choice)
        for chunk in data_lines
        for choice in chunk["choices"]
    )
    contents: Final = tuple(
        JSON_OBJECT.validate_python(choice["delta"]).get("content")
        for choice in choices
    )

    assert contents.count("Hello") == 1
    assert sum(choice["delta"].get("role") == "assistant" for choice in choices) == 1
    assert choices[-1]["finish_reason"] == "stop"
