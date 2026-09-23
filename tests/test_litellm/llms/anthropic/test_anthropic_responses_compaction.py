import json

import pytest

from litellm.llms.anthropic import responses_compaction
from litellm.types.llms.openai import ChatCompletionResponseMessage


def _replayed_turn(token_value: str) -> ChatCompletionResponseMessage:
    block = {"type": "compaction", "content": token_value, "encrypted_content": token_value}
    token = responses_compaction.encode({"compaction_blocks": [block]})
    assert token is not None
    replayed = responses_compaction.replay_message(token)
    assert replayed is not None
    return replayed


def test_encode_replays_only_the_newest_block_with_every_field_intact():
    older = {"type": "compaction", "content": "first pass", "encrypted_content": "tok-1"}
    newer = {"type": "compaction", "content": "final", "encrypted_content": "tok-2", "signature": "sig"}

    token = responses_compaction.encode({"compaction_blocks": [older, newer]})

    assert token is not None
    replayed = responses_compaction.replay_message(token)
    assert replayed is not None
    assert replayed["provider_specific_fields"] == {"compaction_blocks": [newer]}
    assert responses_compaction.inspectable_text(token) == "final"


def test_failed_compaction_has_nothing_to_inspect_but_still_replays():
    block = {"type": "compaction", "content": None, "encrypted_content": "tok"}

    token = responses_compaction.encode({"compaction_blocks": [block]})

    assert token is not None
    assert responses_compaction.inspectable_text(token) is None
    replayed = responses_compaction.replay_message(token)
    assert replayed is not None
    assert replayed["provider_specific_fields"] == {"compaction_blocks": [block]}


@pytest.mark.parametrize(
    "token",
    [
        "opaque-blob-from-another-provider",
        json.dumps({"type": "reasoning", "summary": []}),
        json.dumps(["compaction"]),
        "",
    ],
)
def test_tokens_this_codec_did_not_issue_are_neither_replayed_nor_inspected(token):
    assert responses_compaction.replay_message(token) is None
    assert responses_compaction.inspectable_text(token) is None


def test_only_the_newest_replayed_compaction_turn_survives():
    messages = [
        _replayed_turn("tok-1"),
        {"role": "user", "content": "a turn"},
        _replayed_turn("tok-2"),
        {"role": "user", "content": "next"},
    ]

    assert responses_compaction.superseded_replay_indices(messages) == frozenset({0})
    assert responses_compaction.superseded_replay_indices(messages[1:]) == frozenset()


@pytest.mark.parametrize(
    ("provider_specific_fields", "expected"),
    [
        ({"compaction_delta": {"type": "compaction_delta", "content": "s"}}, True),
        ({"compaction_start": {"type": "compaction", "content": None}}, True),
        ({"thinking_blocks": [{"type": "thinking", "thinking": "t"}]}, False),
        (None, False),
    ],
)
def test_streamed_compaction_chunks_are_recognized(provider_specific_fields, expected):
    assert responses_compaction.is_streaming_compaction(provider_specific_fields) is expected
