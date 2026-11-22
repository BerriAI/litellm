from typing import Final

import pytest

from litellm.responses.litellm_completion_transformation.repeated_tool_calls import without_repeated_tool_calls
from litellm.types.utils import ChatCompletionMessageToolCall, Message

_USER: Final = {"role": "user", "content": "Weather in Paris, Rome and Oslo?"}
_ANSWER: Final = {"role": "assistant", "content": "14 everywhere."}


def _call(call_id: str) -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}


def _asks(*call_ids: str) -> dict[str, object]:
    return {"role": "assistant", "content": None, "tool_calls": [_call(call_id) for call_id in call_ids]}


def _output(call_id: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": "14"}


_STORED_TURN: Final = Message(
    role="assistant", content=None, tool_calls=[ChatCompletionMessageToolCall(**_call(i)) for i in ("c1", "c2", "c3")]
)


@pytest.mark.parametrize("asked", [_asks("c1", "c2", "c3"), _STORED_TURN], ids=["message-dict", "stored-message"])
@pytest.mark.parametrize("output_order", [("c1", "c2", "c3"), ("c3", "c1", "c2")], ids=["in-order", "reordered"])
def test_calls_a_turn_already_made_are_not_made_again_before_their_outputs(
    asked: dict[str, object] | Message, output_order: tuple[str, ...]
) -> None:
    repeated: Final = [message for call_id in output_order for message in (_asks(call_id), _output(call_id))]
    once: Final = [_output(call_id) for call_id in output_order]

    replayed: Final = without_repeated_tool_calls([_USER, asked, *repeated, _ANSWER, _USER])

    assert list(replayed) == [_USER, asked, *once, _ANSWER, _USER]


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([], id="nothing"),
        pytest.param([_USER, _asks("c1"), _output("c1"), _asks("c1"), _output("c1")], id="id-reused-by-the-next-step"),
        pytest.param([_asks("c1", "c2"), _output("c1"), _asks("c1"), _output("c2")], id="call-already-answered"),
        pytest.param([_USER, _asks("c1"), _USER, _asks("c1"), _output("c1")], id="id-reused-after-a-user-turn"),
        pytest.param([_USER, _asks("c1"), _ANSWER, _asks("c1"), _output("c1")], id="id-reused-after-an-answer"),
        pytest.param([_USER, _asks("c1"), _asks("c2"), _output("c1"), _output("c2")], id="call-made-nowhere-else"),
        pytest.param([_asks("c1", "c2"), _asks("c1", "c2"), _output("c1")], id="several-calls-at-once"),
        pytest.param([_asks("c1"), {**_asks("c1"), "content": "Checking."}, _output("c1")], id="text"),
        pytest.param([_asks("c1"), {**_asks("c1"), "thinking_blocks": [{}]}, _output("c1")], id="thinking"),
        pytest.param([_asks("c1"), {**_asks("c1"), "reasoning_content": "So"}, _output("c1")], id="reasoning"),
        pytest.param([_asks("c1"), {"role": "assistant", "tool_calls": [{"type": "function"}]}], id="no-id"),
    ],
)
def test_every_other_message_stays(messages: list[dict[str, object]]) -> None:
    assert list(without_repeated_tool_calls(messages)) == messages
