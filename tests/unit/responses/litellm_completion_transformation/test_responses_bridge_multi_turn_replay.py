"""
Replay-invariant contract tests for the Responses -> chat completions bridge.

A stateless Responses client has no ``previous_response_id`` to lean on, so
every turn it appends the bridge's own ``output`` items back onto ``input``
verbatim and resends the whole transcript. That is the only option against a
provider whose Messages API is stateless, and it is the case single-turn
translation tests do not cover: each turn in isolation can translate perfectly
while the accumulated transcript still degrades.

The invariants pinned here, from the discussion on #42005:

* every ``function_call`` the client replays reaches the provider as a native
  ``tool_calls`` entry on an assistant message, not as prose;
* a ``reasoning`` item never becomes visible ``content`` on any message, and
  never arrives with ``role="user"`` -- replaying the model's own hidden
  chain-of-thought as something the user said is worse than dropping it;
* every ``tool`` result is preceded by the assistant message carrying its
  ``tool_call_id``, so no result is orphaned;
* none of the above decays as the transcript grows.

The upstream is deliberately absent. These assert the request the bridge builds,
which makes them provider-independent and deterministic -- no Kimi, no network,
no recorded cassette to rot.
"""

import json

import pytest

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)

VAULT_TOOL = {
    "type": "function",
    "name": "vault_token",
    "description": "Return an opaque random token for a slot. Never guess it.",
    "parameters": {
        "type": "object",
        "properties": {"slot": {"type": "integer"}},
        "required": ["slot"],
    },
}


def _field(message, key, default=None):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def _tool_call_ids(message):
    ids = []
    for call in _field(message, "tool_calls") or []:
        ids.append(_field(call, "id"))
    return ids


def _visible_text(message):
    """Text a provider would read as prompt content, flattened to one string."""
    content = _field(message, "content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _reasoning_item(turn, *, summary=True, encrypted=False, content_blocks=False):
    item = {"type": "reasoning", "id": f"rs_{turn}"}
    secret = f"SECRET-CHAIN-OF-THOUGHT-{turn}"
    if summary:
        item["summary"] = [{"type": "summary_text", "text": secret}]
    else:
        item["summary"] = []
    if content_blocks:
        item["content"] = [{"type": "reasoning_text", "text": secret}]
    if encrypted:
        item["encrypted_content"] = "gAAAAABopaque=="
    return item


def _function_call_item(turn):
    return {
        "type": "function_call",
        "id": f"fc_{turn}",
        "call_id": f"call_{turn}",
        "name": "vault_token",
        "arguments": json.dumps({"slot": turn}),
        "status": "completed",
    }


def _function_call_output_item(turn):
    return {
        "type": "function_call_output",
        "call_id": f"call_{turn}",
        "output": f"tok-{turn}-opaque",
    }


def _assistant_text_item(turn):
    return {
        "type": "message",
        "id": f"msg_{turn}",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": f"Fetching slot {turn}.", "annotations": []}],
    }


def _build_transcript(turns, shape):
    """The `input` a stateless client holds after `turns` completed turns."""
    transcript = [
        {
            "role": "user",
            "content": "Collect vault tokens one at a time using vault_token, starting at slot 1.",
        }
    ]
    for turn in range(1, turns + 1):
        transcript.extend(shape(turn))
    return transcript


def _bridge(transcript):
    return LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="openai/kimi-k3",
        input=transcript,
        responses_api_request={"tools": [VAULT_TOOL], "max_output_tokens": 500},
        custom_llm_provider="openai",
    )


# Every shape a real `output` array takes across providers. Each one is replayed
# verbatim, because that is what the client is told to do.
SHAPES = {
    "reasoning_summary": lambda t: [
        _reasoning_item(t),
        _function_call_item(t),
        _function_call_output_item(t),
    ],
    "reasoning_empty_summary": lambda t: [
        _reasoning_item(t, summary=False),
        _function_call_item(t),
        _function_call_output_item(t),
    ],
    "reasoning_encrypted_only": lambda t: [
        _reasoning_item(t, summary=False, encrypted=True),
        _function_call_item(t),
        _function_call_output_item(t),
    ],
    "reasoning_content_blocks": lambda t: [
        _reasoning_item(t, summary=False, content_blocks=True),
        _function_call_item(t),
        _function_call_output_item(t),
    ],
    "assistant_text_beside_the_call": lambda t: [
        _assistant_text_item(t),
        _function_call_item(t),
        _function_call_output_item(t),
    ],
    "reasoning_and_assistant_text": lambda t: [
        _reasoning_item(t),
        _assistant_text_item(t),
        _function_call_item(t),
        _function_call_output_item(t),
    ],
    "no_reasoning_item": lambda t: [
        _function_call_item(t),
        _function_call_output_item(t),
    ],
}

# Eight turns because the reported degradation appeared on the fifth; one turn
# is the control that says a failure at depth is about accumulation.
TURN_COUNTS = [1, 2, 5, 8]


@pytest.mark.parametrize("shape_name", sorted(SHAPES))
@pytest.mark.parametrize("turns", TURN_COUNTS)
class TestResponsesBridgeMultiTurnReplay:
    def test_every_replayed_call_reaches_the_provider_as_a_native_tool_call(self, shape_name, turns):
        """A replayed function_call must not degrade into prose."""
        request = _bridge(_build_transcript(turns, SHAPES[shape_name]))

        sent = set()
        for message in request["messages"]:
            sent.update(_tool_call_ids(message))

        expected = {f"call_{turn}" for turn in range(1, turns + 1)}
        assert expected <= sent, (
            f"{sorted(expected - sent)} were replayed by the client but reach the provider as "
            f"no tool call at all after {turns} turns"
        )

    def test_reasoning_never_becomes_visible_content(self, shape_name, turns):
        """The model's hidden chain-of-thought must not re-enter as prompt text."""
        request = _bridge(_build_transcript(turns, SHAPES[shape_name]))

        for message in request["messages"]:
            text = _visible_text(message)
            assert "SECRET-CHAIN-OF-THOUGHT" not in text, (
                f"reasoning surfaced as visible {_field(message, 'role')!r} content "
                f"after {turns} turns: {text[:120]!r}"
            )

    def test_reasoning_is_never_replayed_as_something_the_user_said(self, shape_name, turns):
        """A reasoning item has no `role`, so a naive default makes it a user turn."""
        request = _bridge(_build_transcript(turns, SHAPES[shape_name]))

        user_messages = [m for m in request["messages"] if _field(m, "role") == "user"]
        assert len(user_messages) == 1, (
            f"the transcript holds one real user turn, but {len(user_messages)} user messages "
            f"reach the provider after {turns} turns"
        )

    def test_no_tool_result_is_orphaned(self, shape_name, turns):
        """A tool result whose call never precedes it is rejected by most providers."""
        messages = _bridge(_build_transcript(turns, SHAPES[shape_name]))["messages"]

        announced = set()
        for message in messages:
            if _field(message, "role") == "tool":
                call_id = _field(message, "tool_call_id")
                assert call_id in announced, (
                    f"tool result {call_id!r} arrives before any assistant message announces it, "
                    f"after {turns} turns"
                )
            announced.update(_tool_call_ids(message))

    def test_the_tools_survive_the_whole_transcript(self, shape_name, turns):
        """A transcript that loses its tools explains a model answering in prose."""
        request = _bridge(_build_transcript(turns, SHAPES[shape_name]))

        tools = request.get("tools") or []
        assert len(tools) == 1, f"the tool definition is gone from the request after {turns} turns"


@pytest.mark.parametrize("shape_name", sorted(SHAPES))
def test_replay_grows_by_exactly_one_exchange_per_turn(shape_name):
    """
    Message count must grow linearly.

    Both real failure modes show up here before they show up anywhere else: a
    turn that stops producing its assistant message, and a merge that folds two
    turns into one and loses a call on the way.
    """
    shape = SHAPES[shape_name]
    counts = [len(_bridge(_build_transcript(turns, shape))["messages"]) for turns in (1, 2, 3, 4)]
    deltas = {later - earlier for earlier, later in zip(counts, counts[1:])}

    assert len(deltas) == 1, (
        f"messages per replayed turn is not constant across the transcript: counts {counts}"
    )


@pytest.mark.parametrize("turns", TURN_COUNTS)
def test_two_calls_in_one_turn_are_both_replayed(turns):
    """Parallel tool calls share a turn; neither may be dropped by the merge."""

    def shape(turn):
        return [
            _reasoning_item(turn),
            _function_call_item(turn),
            {
                "type": "function_call",
                "id": f"fc_{turn}b",
                "call_id": f"call_{turn}b",
                "name": "vault_token",
                "arguments": json.dumps({"slot": turn + 100}),
                "status": "completed",
            },
            _function_call_output_item(turn),
            {
                "type": "function_call_output",
                "call_id": f"call_{turn}b",
                "output": f"tok-{turn}b-opaque",
            },
        ]

    messages = _bridge(_build_transcript(turns, shape))["messages"]

    sent = set()
    for message in messages:
        sent.update(_tool_call_ids(message))

    expected = {f"call_{turn}" for turn in range(1, turns + 1)}
    expected |= {f"call_{turn}b" for turn in range(1, turns + 1)}
    assert expected <= sent, f"{sorted(expected - sent)} were dropped after {turns} turns"
