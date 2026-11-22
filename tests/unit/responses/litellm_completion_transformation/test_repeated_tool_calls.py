from collections.abc import Callable, Sequence
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
_OTHER_WHOLE: Final = f'{{"note": "{"Rome " * 600}"}}'
_OTHER_CUT: Final = truncated_for_db(_OTHER_WHOLE, 2048)
_EPHEMERAL: Final = {"type": "ephemeral"}
_FIELDS: Final = {"provider_specific_fields": {"trace": "t1"}}


def _call(call_id: str, arguments: str | None = "{}", name: str = "get_weather") -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _asks(*call_ids: str, arguments: str | None = "{}", name: str = "get_weather") -> dict[str, object]:
    calls: Final = [_call(call_id, arguments, name) for call_id in call_ids]
    return {"role": "assistant", "content": None, "tool_calls": calls}


def _output(call_id: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": "14"}


def _stored(asked: dict[str, object]) -> Message:
    return Message(**{**asked, "tool_calls": [ChatCompletionMessageToolCall(**call) for call in asked["tool_calls"]]})


def _arguments(message: dict[str, object] | Message) -> list[str]:
    return [call["function"]["arguments"] for call in message["tool_calls"]]


def _replayed(messages: Sequence[object]) -> list[object]:
    return list(without_repeated_tool_calls(messages, keep_stored_arguments=lambda: False))


def _target_is_not_looked_up() -> bool:
    pytest.fail("the target was looked up with no arguments to give back")


_AS_STORED: Final[tuple[Callable[[dict[str, object]], dict[str, object] | Message], ...]] = (dict, _stored)


@pytest.mark.parametrize("as_stored", _AS_STORED, ids=["message-dict", "stored-message"])
@pytest.mark.parametrize("output_order", [("c1", "c2", "c3"), ("c3", "c1", "c2"), ("c1",)], ids="-".join)
def test_calls_a_turn_already_made_are_not_made_again_before_their_outputs(
    as_stored: Callable[[dict[str, object]], dict[str, object] | Message], output_order: tuple[str, ...]
) -> None:
    asked: Final = as_stored(_asks(*sorted(output_order)))
    repeats: Final = [
        {"role": "assistant", "tool_calls": [{**_call(call_id), "index": index}]}
        for index, call_id in enumerate(output_order)
    ]
    once: Final = [_output(call_id) for call_id in output_order]
    repeated: Final = [message for pair in zip(repeats, once) for message in pair]

    assert _replayed([_USER, asked, *repeated, _ANSWER, _USER]) == [_USER, asked, *once, _ANSWER, _USER]


def test_a_call_repeated_twice_before_its_output_is_made_once() -> None:
    asked: Final = _asks("c1", "c2")

    replayed: Final = _replayed([asked, _asks("c1"), _asks("c1"), _output("c1"), _output("c2")])

    assert replayed == [asked, _output("c1"), _output("c2")]


@pytest.mark.parametrize("signed", [0, 1], ids=["stored-id-lost-its-signature", "stored-id-kept-its-signature"])
def test_a_call_whose_id_lost_its_signature_is_the_same_call(signed: int) -> None:
    ids: Final = ("call_1", f"call_1{THOUGHT_SIGNATURE_SEPARATOR}EmwKAhgB==")
    asked: Final = _asks(ids[signed], "call_2")

    replayed: Final = _replayed([asked, _asks(ids[1 - signed]), _output(ids[1 - signed])])

    assert replayed == [asked, _output(ids[1 - signed])]


@pytest.mark.parametrize("as_stored", _AS_STORED, ids=["message-dict", "stored-message"])
@pytest.mark.parametrize("call_ids", [("c1",), ("c1", "c2")], ids="-".join)
def test_a_repeated_call_gives_back_the_arguments_storage_cut(
    as_stored: Callable[[dict[str, object]], dict[str, object] | Message], call_ids: tuple[str, ...]
) -> None:
    signed: Final = {"provider_specific_fields": {"thought_signature": "EmwKAhgB=="}}

    def turn(arguments: str) -> dict[str, object] | Message:
        asked: Final = _asks(*call_ids, arguments=arguments)
        return as_stored({**asked, "tool_calls": [{**call, **signed} for call in asked["tool_calls"]]})

    stored_turn: Final = turn(_CUT)
    repeats: Final = [m for i in call_ids for m in (_asks(i, arguments=_WHOLE), _output(i))]

    assert _replayed([_USER, stored_turn, *repeats]) == [_USER, turn(_WHOLE), *map(_output, call_ids)]
    assert stored_turn == turn(_CUT)


def test_only_the_call_that_was_repeated_gets_its_arguments_back() -> None:
    turn, output = _replayed([_asks("c1", "c2", arguments=_CUT), _asks("c1", arguments=_WHOLE), _output("c1")])

    assert _arguments(turn) == [_WHOLE, _CUT]
    assert output == _output("c1")


@pytest.mark.parametrize("next_id", ["c1", "c2"], ids=["id-reused", "next-id"])
def test_every_step_gets_back_the_arguments_of_its_own_call(next_id: str) -> None:
    step: Final = [_asks("c1", arguments=_CUT), _asks("c1", arguments=_WHOLE), _output("c1")]
    next_step: Final = [_asks(next_id, arguments=_OTHER_CUT), _asks(next_id, arguments=_OTHER_WHOLE), _output(next_id)]

    turn, output, next_turn, next_output = _replayed([*step, *next_step])

    assert (output, next_output) == (_output("c1"), _output(next_id))
    assert [_arguments(turn), _arguments(next_turn)] == [[_WHOLE], [_OTHER_WHOLE]]


@pytest.mark.parametrize(
    ("stored_arguments", "repeated"),
    [
        pytest.param('{"city": "Paris"}', _asks("c1", arguments='{"city": "Rome"}'), id="whole-arguments"),
        pytest.param("{}", _asks("c1", arguments='{"city": "Rome"}'), id="no-arguments"),
        pytest.param(_CUT, _asks("c1", arguments='{"city": "Rome"}'), id="arguments-of-another-call"),
        pytest.param(_CUT, _asks("c1", arguments=_WHOLE, name="get_time"), id="call-of-another-function"),
        pytest.param(_CUT, _asks("c1", arguments=None), id="nothing-repeated"),
        pytest.param(None, _asks("c1", arguments=_WHOLE), id="nothing-stored"),
    ],
)
def test_the_stored_arguments_stay_unless_the_repeat_is_what_storage_cut(
    stored_arguments: str | None, repeated: dict[str, object]
) -> None:
    asked: Final = _asks("c1", "c2", arguments=stored_arguments)

    replayed: Final = without_repeated_tool_calls(
        [asked, repeated, _output("c1")], keep_stored_arguments=_target_is_not_looked_up
    )

    assert list(replayed) == [asked, _output("c1")]


def test_the_stored_arguments_stay_when_asked_to_keep_them() -> None:
    asked: Final = _asks("c1", "c2", arguments=_CUT)
    repeated: Final = [m for i in ("c1", "c2") for m in (_asks(i, arguments=_WHOLE), _output(i))]

    replayed: Final = without_repeated_tool_calls([asked, *repeated], keep_stored_arguments=lambda: True)

    assert list(replayed) == [asked, _output("c1"), _output("c2")]


def test_a_resent_call_that_says_more_stays_and_the_next_repeat_is_still_one() -> None:
    asked: Final = _asks("c1", "c2", arguments=_CUT)
    marked: Final = {**_asks("c1", arguments=_WHOLE), "cache_control": _EPHEMERAL}

    replayed: Final = _replayed([asked, marked, _output("c1"), _asks("c2"), _output("c2")])

    assert replayed == [asked, marked, _output("c1"), _output("c2")]


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([], id="nothing"),
        pytest.param([_USER, _asks("c1"), _output("c1"), _asks("c1"), _output("c1")], id="id-reused-by-the-next-step"),
        pytest.param([_asks("c1", "c2"), _output("c1"), _asks("c1"), _output("c2")], id="call-already-answered"),
        pytest.param(
            [_asks("c1", "c2"), _output(f"c1{THOUGHT_SIGNATURE_SEPARATOR}Em=="), _asks("c1"), _output("c2")],
            id="call-already-answered-by-its-signed-id",
        ),
        pytest.param([_USER, _asks("c1"), _USER, _asks("c1"), _output("c1")], id="id-reused-after-a-user-turn"),
        pytest.param([_USER, _asks("c1"), _ANSWER, _asks("c1"), _output("c1")], id="id-reused-after-an-answer"),
        pytest.param([_asks("c1", "c2"), _output("c1"), _stored(_asks("c2"))], id="id-reused-by-a-stored-turn"),
        pytest.param([_USER, _asks("c1"), _asks("c2"), _output("c1"), _output("c2")], id="call-made-nowhere-else"),
        pytest.param([_asks("c1", "c2"), _asks("c1", "c2"), _output("c1")], id="several-calls-at-once"),
        pytest.param([_asks("c1"), {**_asks("c1"), "content": "Checking."}, _output("c1")], id="text"),
        pytest.param([_asks("c1"), {**_asks("c1"), "thinking_blocks": [{}]}, _output("c1")], id="thinking"),
        pytest.param([_asks("c1"), {**_asks("c1"), "reasoning_content": "So"}, _output("c1")], id="reasoning"),
        pytest.param([_asks("c1"), {**_asks("c1"), "cache_control": _EPHEMERAL}, _output("c1")], id="marked-message"),
        pytest.param([_asks("c1"), {**_asks("c1"), "refusal": "No."}, _output("c1")], id="refusal"),
        pytest.param([_asks("c1"), {**_asks("c1"), **_FIELDS}, _output("c1")], id="message-saying-more"),
        pytest.param(
            [_asks("c1"), {**_asks("c1"), "tool_calls": [{**_call("c1"), "cache_control": _EPHEMERAL}]}],
            id="marked-call",
        ),
        pytest.param([_asks("c1"), {**_asks("c1"), "tool_calls": [{**_call("c1"), **_FIELDS}]}], id="call-saying-more"),
        pytest.param(
            [_asks("c1"), {**_asks("c1"), "tool_calls": [{**_call("c1"), "function": {"name": "a", **_FIELDS}}]}],
            id="function-saying-more",
        ),
        pytest.param([_asks(""), _asks(""), _output("")], id="empty-ids"),
        pytest.param([{"role": "assistant", "tool_calls": [{"type": "function", "function": {}}]}] * 2, id="no-ids"),
        pytest.param([_asks("c1"), {"role": "assistant", "tool_calls": [{"id": "c1"}]}], id="no-function"),
        pytest.param([_asks("c1"), {"role": "assistant", "tool_calls": ["c1"]}], id="no-call"),
    ],
)
def test_every_other_message_stays(messages: list[dict[str, object]]) -> None:
    assert _replayed(messages) == messages
