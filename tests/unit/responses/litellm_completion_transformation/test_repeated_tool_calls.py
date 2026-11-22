from typing import Final

import pytest

from litellm.litellm_core_utils.db_truncation import truncated_for_db
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR
from litellm.responses.litellm_completion_transformation.repeated_tool_calls import without_repeated_tool_calls
from litellm.types.utils import ChatCompletionMessageToolCall, Message

_USER: Final = {"role": "user", "content": "Weather in Paris, Rome and Oslo?"}
_ANSWER: Final = {"role": "assistant", "content": "14 everywhere."}
_WHOLE: Final = f'{{"note": "{"Paris " * 500}"}}'
_CUT: Final = truncated_for_db(_WHOLE, 2048)
_EPHEMERAL: Final = {"type": "ephemeral"}


def _call(call_id: str, arguments: str = "{}") -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": arguments}}


def _asks(*call_ids: str, arguments: str = "{}") -> dict[str, object]:
    return {"role": "assistant", "content": None, "tool_calls": [_call(call_id, arguments) for call_id in call_ids]}


def _output(call_id: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": "14"}


def _stored(asked: dict[str, object]) -> Message:
    return Message(**{**asked, "tool_calls": [ChatCompletionMessageToolCall(**call) for call in asked["tool_calls"]]})


def _arguments(message: dict[str, object] | Message) -> list[str]:
    return [call["function"]["arguments"] for call in message["tool_calls"]]


@pytest.mark.parametrize("as_stored", [dict, _stored], ids=["message-dict", "stored-message"])
@pytest.mark.parametrize("output_order", [("c1", "c2", "c3"), ("c3", "c1", "c2"), ("c1",)], ids="-".join)
def test_calls_a_turn_already_made_are_not_made_again_before_their_outputs(as_stored, output_order: tuple[str, ...]):
    asked: Final = as_stored(_asks(*sorted(output_order)))
    repeats: Final = [
        {"role": "assistant", "tool_calls": [{**_call(call_id), "index": index}]}
        for index, call_id in enumerate(output_order)
    ]
    once: Final = [_output(call_id) for call_id in output_order]
    repeated: Final = [message for pair in zip(repeats, once) for message in pair]

    replayed: Final = without_repeated_tool_calls([_USER, asked, *repeated, _ANSWER, _USER])

    assert list(replayed) == [_USER, asked, *once, _ANSWER, _USER]


def test_a_call_repeated_twice_before_its_output_is_made_once():
    asked: Final = _asks("c1", "c2")

    replayed: Final = without_repeated_tool_calls([asked, _asks("c1"), _asks("c1"), _output("c1"), _output("c2")])

    assert list(replayed) == [asked, _output("c1"), _output("c2")]


@pytest.mark.parametrize("signed", [0, 1], ids=["stored-id-lost-its-signature", "stored-id-kept-its-signature"])
def test_a_call_whose_id_lost_its_signature_is_the_same_call(signed: int):
    ids: Final = ("call_1", f"call_1{THOUGHT_SIGNATURE_SEPARATOR}EmwKAhgB==")
    asked: Final = _asks(ids[signed], "call_2")

    replayed: Final = without_repeated_tool_calls([asked, _asks(ids[1 - signed]), _output(ids[1 - signed])])

    assert list(replayed) == [asked, _output(ids[1 - signed])]


@pytest.mark.parametrize("as_stored", [dict, _stored], ids=["message-dict", "stored-message"])
@pytest.mark.parametrize("stored_arguments", [_CUT, "{}"], ids=["cut", "redacted"])
@pytest.mark.parametrize("call_ids", [("c1",), ("c1", "c2")], ids="-".join)
def test_a_repeated_call_gives_back_the_arguments_storage_lost(as_stored, stored_arguments: str, call_ids):
    signed: Final = {"provider_specific_fields": {"thought_signature": "EmwKAhgB=="}}
    asked: Final = _asks(*call_ids, arguments=stored_arguments)
    stored_turn: Final = as_stored({**asked, "tool_calls": [{**call, **signed} for call in asked["tool_calls"]]})

    question, turn, *outputs = without_repeated_tool_calls(
        [_USER, stored_turn, *(m for i in call_ids for m in (_asks(i, arguments=_WHOLE), _output(i)))]
    )

    assert question == _USER
    assert outputs == [_output(call_id) for call_id in call_ids]
    assert _arguments(turn) == [_WHOLE for _ in call_ids]
    assert [call["id"] for call in turn["tool_calls"]] == list(call_ids)
    assert [call["provider_specific_fields"] for call in turn["tool_calls"]] == [
        signed["provider_specific_fields"] for _ in call_ids
    ]
    assert _arguments(stored_turn) == [stored_arguments for _ in call_ids]


def test_only_the_call_that_was_repeated_gets_its_arguments_back():
    turn, output = without_repeated_tool_calls(
        [_asks("c1", "c2", arguments=_CUT), _asks("c1", arguments=_WHOLE), _output("c1")]
    )

    assert _arguments(turn) == [_WHOLE, _CUT]
    assert output == _output("c1")


@pytest.mark.parametrize(
    ("stored_arguments", "repeated_arguments", "extra"),
    [
        pytest.param('{"city": "Paris"}', '{"city": "Rome"}', {}, id="whole-arguments"),
        pytest.param(_CUT, '{"city": "Rome"}', {}, id="arguments-of-another-call"),
        pytest.param(_CUT, None, {}, id="nothing-repeated"),
        pytest.param(_CUT, _WHOLE, {"thinking_blocks": [{"type": "thinking", "signature": "EmwKA"}]}, id="signed-turn"),
    ],
)
def test_the_stored_arguments_stay_unless_the_repeat_is_what_storage_cut(
    stored_arguments: str, repeated_arguments: str | None, extra: dict[str, object]
):
    asked: Final = {**_asks("c1", "c2", arguments=stored_arguments), **extra}

    replayed: Final = without_repeated_tool_calls([asked, _asks("c1", arguments=repeated_arguments), _output("c1")])

    assert list(replayed) == [asked, _output("c1")]


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([], id="nothing"),
        pytest.param([_USER, _asks("c1"), _output("c1"), _asks("c1"), _output("c1")], id="id-reused-by-the-next-step"),
        pytest.param([_asks("c1", "c2"), _output("c1"), _asks("c1"), _output("c2")], id="call-already-answered"),
        pytest.param([_USER, _asks("c1"), _USER, _asks("c1"), _output("c1")], id="id-reused-after-a-user-turn"),
        pytest.param([_USER, _asks("c1"), _ANSWER, _asks("c1"), _output("c1")], id="id-reused-after-an-answer"),
        pytest.param([_asks("c1", "c2"), _output("c1"), _stored(_asks("c2"))], id="id-reused-by-a-stored-turn"),
        pytest.param([_USER, _asks("c1"), _asks("c2"), _output("c1"), _output("c2")], id="call-made-nowhere-else"),
        pytest.param([_asks("c1", "c2"), _asks("c1", "c2"), _output("c1")], id="several-calls-at-once"),
        pytest.param([_asks("c1"), {**_asks("c1"), "content": "Checking."}, _output("c1")], id="text"),
        pytest.param([_asks("c1"), {**_asks("c1"), "thinking_blocks": [{}]}, _output("c1")], id="thinking"),
        pytest.param([_asks("c1"), {**_asks("c1"), "reasoning_content": "So"}, _output("c1")], id="reasoning"),
        pytest.param([_asks("c1"), {**_asks("c1"), "cache_control": _EPHEMERAL}, _output("c1")], id="marked-message"),
        pytest.param(
            [_asks("c1"), {**_asks("c1"), "tool_calls": [{**_call("c1"), "cache_control": _EPHEMERAL}]}],
            id="marked-call",
        ),
        pytest.param(
            [_asks("c1"), {**_asks("c1"), "tool_calls": [{**_call("c1"), "function": {"name": "a", "strict": True}}]}],
            id="function-saying-more",
        ),
        pytest.param([_asks(""), _asks(""), _output("")], id="empty-ids"),
        pytest.param([{"role": "assistant", "tool_calls": [{"type": "function", "function": {}}]}] * 2, id="no-ids"),
        pytest.param([_asks("c1"), {"role": "assistant", "tool_calls": [{"id": "c1"}]}], id="no-function"),
        pytest.param([_asks("c1"), {"role": "assistant", "tool_calls": ["c1"]}], id="no-call"),
    ],
)
def test_every_other_message_stays(messages: list[dict[str, object]]):
    assert list(without_repeated_tool_calls(messages)) == messages
