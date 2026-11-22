from collections.abc import Callable, Sequence
from itertools import chain
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.litellm_core_utils.db_truncation import truncated_for_db
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR
from litellm.responses.litellm_completion_transformation.repeated_tool_calls import without_repeated_tool_calls
from litellm.types.utils import ChatCompletionMessageToolCall, Message

_USER: Final = {"role": "user", "content": "Weather in Paris, Rome and Oslo?"}
_ANSWER: Final = {"role": "assistant", "content": "14 everywhere."}
_DEVELOPER: Final = {"role": "developer", "content": "Answer in French."}
_SYSTEM: Final = {"role": "system", "content": "Answer in French."}
_WHOLE: Final = f'{{"note": "{"Paris " * 500}"}}'
_CUT: Final = truncated_for_db(_WHOLE, 2048)
_OTHER_WHOLE: Final = f'{{"note": "{"Rome " * 600}"}}'
_OTHER_CUT: Final = truncated_for_db(_OTHER_WHOLE, 2048)
_EPHEMERAL: Final = {"type": "ephemeral"}
_FIELDS: Final = {"provider_specific_fields": {"trace": "t1"}}
_SIGNED: Final = f"c1{THOUGHT_SIGNATURE_SEPARATOR}EmwKAhgB"
_OTHERWISE_SIGNED: Final = f"c1{THOUGHT_SIGNATURE_SEPARATOR}EmwK"
_ONLY_SIGNED: Final = f"{THOUGHT_SIGNATURE_SEPARATOR}EmwK"


def _call(call_id: str, arguments: str | None = "{}", name: str = "get_weather") -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _asks(*call_ids: str, arguments: str | None = "{}", name: str = "get_weather") -> dict[str, object]:
    calls: Final = [_call(call_id, arguments, name) for call_id in call_ids]
    return {"role": "assistant", "content": None, "tool_calls": calls}


def _output(call_id: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call_id, "content": "14"}


def _resent_one_by_one(*call_ids: str, arguments: str = "{}", name: str = "get_weather") -> list[dict[str, object]]:
    resent: Final = ((_asks(call_id, arguments=arguments, name=name), _output(call_id)) for call_id in call_ids)
    return list(chain.from_iterable(resent))


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
    repeated: Final = chain.from_iterable(zip(repeats, once))

    assert _replayed([_USER, asked, *repeated, _ANSWER, _USER]) == [_USER, asked, *once, _ANSWER, _USER]


@pytest.mark.parametrize("as_stored", _AS_STORED, ids=["message-dict", "stored-message"])
@pytest.mark.parametrize("resent", [("c1", "c2", "c3"), ("c2", "c3"), ("c3", "c1")], ids="-".join)
def test_calls_resent_together_before_their_outputs_are_not_made_again(
    as_stored: Callable[[dict[str, object]], dict[str, object] | Message], resent: tuple[str, ...]
) -> None:
    asked: Final = as_stored(_asks("c1", "c2", "c3"))
    once: Final = [_output(call_id) for call_id in resent]

    assert _replayed([_USER, asked, _asks(*resent), *once, _ANSWER]) == [_USER, asked, *once, _ANSWER]


@pytest.mark.parametrize("empty", [{"content": ""}, {"thinking_blocks": []}, {"cache_control": None}], ids=str)
def test_a_repeat_whose_other_fields_are_empty_is_still_a_repeat(empty: dict[str, object]) -> None:
    asked: Final = _asks("c1", "c2")
    repeat: Final = {**_asks("c1"), "tool_calls": [{**_call("c1"), **empty}], **empty}

    assert _replayed([asked, repeat, _output("c1"), _output("c2")]) == [asked, _output("c1"), _output("c2")]


@pytest.mark.parametrize("first_resent", [("c2",), ("c1", "c2")], ids=["alone", "together"])
def test_a_call_repeated_twice_before_its_output_is_made_once(first_resent: tuple[str, ...]) -> None:
    asked: Final = _asks("c1", "c2")

    replayed: Final = _replayed([asked, _asks(*first_resent), _output("c1"), _asks("c2"), _output("c2")])

    assert replayed == [asked, _output("c1"), _output("c2")]


@pytest.mark.parametrize(
    ("stored_id", "resent_id"),
    [
        pytest.param("c1", _SIGNED, id="stored-id-lost-its-signature"),
        pytest.param(_SIGNED, "c1", id="resent-id-lost-its-signature"),
        pytest.param(_SIGNED, _SIGNED, id="both-ids-kept-it"),
    ],
)
def test_a_call_whose_id_carries_its_signature_or_lost_it_is_the_same_call(stored_id: str, resent_id: str) -> None:
    asked: Final = _asks("c2", stored_id, "c3", arguments=_CUT)

    turn, output = _replayed([asked, _asks(resent_id, arguments=_WHOLE), _output(resent_id)])

    assert [call["id"] for call in turn["tool_calls"]] == ["c2", stored_id, "c3"]
    assert _arguments(turn) == [_CUT, _WHOLE, _CUT]
    assert output == _output(resent_id)


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
    outputs: Final = [_output(call_id) for call_id in call_ids]
    one_by_one: Final = _resent_one_by_one(*call_ids, arguments=_WHOLE)
    together: Final = [_asks(*call_ids, arguments=_WHOLE), *outputs]

    assert _replayed([_USER, stored_turn, *one_by_one]) == [_USER, turn(_WHOLE), *outputs]
    assert _replayed([_USER, stored_turn, *together]) == [_USER, turn(_WHOLE), *outputs]
    assert stored_turn == turn(_CUT)


def test_a_call_resent_under_its_namespaced_name_gives_back_its_arguments_under_the_stored_name() -> None:
    stored: Final = _asks("c1", "c2", arguments=_CUT, name="search")
    resent: Final = _resent_one_by_one("c1", "c2", arguments=_WHOLE, name="crm__search")

    turn, *outputs = _replayed([stored, *resent])

    assert turn == _asks("c1", "c2", arguments=_WHOLE, name="search")
    assert outputs == [_output("c1"), _output("c2")]


def test_a_stored_call_of_an_unknown_kind_keeps_what_storage_holds() -> None:
    call: Final = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_weather", arguments=_CUT))
    asked: Final = {"role": "assistant", "tool_calls": [call]}

    assert _replayed([asked, _asks("c1", arguments=_WHOLE), _output("c1")]) == [asked, _output("c1")]


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
    repeated: Final = _resent_one_by_one("c1", "c2", arguments=_WHOLE)

    replayed: Final = without_repeated_tool_calls([asked, *repeated], keep_stored_arguments=lambda: True)

    assert list(replayed) == [asked, _output("c1"), _output("c2")]


@pytest.mark.parametrize(
    "stored_ids",
    [(_SIGNED, _OTHERWISE_SIGNED), ("c1", _SIGNED), (_SIGNED, "c1")],
    ids=["both-signed", "second-signed", "first-signed"],
)
def test_calls_that_are_one_id_once_their_signature_is_taken_off_stay_as_sent(stored_ids: tuple[str, str]) -> None:
    asked: Final = _asks(*stored_ids, arguments=_CUT)
    one_by_one: Final = _resent_one_by_one(*stored_ids, arguments=_WHOLE)
    together: Final = [_asks(*stored_ids, arguments=_WHOLE), *(_output(i) for i in stored_ids)]

    assert _replayed([asked, *one_by_one]) == [asked, *one_by_one]
    assert _replayed([asked, *together]) == [asked, *together]


def test_bare_repeats_after_a_resent_turn_that_says_more_are_left_out() -> None:
    asked: Final = _asks("c1", "c2")
    marked: Final = {**_asks("c1", "c2"), "cache_control": _EPHEMERAL}

    assert _replayed([asked, marked, *_resent_one_by_one("c1", "c2")]) == [asked, marked, _output("c1"), _output("c2")]


def test_a_call_is_looked_for_only_in_the_newest_turn() -> None:
    marked: Final = {**_asks(_SIGNED), "cache_control": _EPHEMERAL}

    assert _replayed([_asks("c1"), marked, _asks("c1"), _output("c1")]) == [_asks("c1"), marked, _output("c1")]


def test_the_last_arguments_resent_for_a_call_are_the_ones_given_back() -> None:
    asked: Final = _asks("c1", arguments=_CUT)
    other: Final = _asks("c1", arguments='{"city": "Rome"}')

    assert _replayed([asked, other, _asks("c1", arguments=_WHOLE), _output("c1")]) == [
        _asks("c1", arguments=_WHOLE),
        _output("c1"),
    ]


def test_a_call_made_next_to_one_id_made_twice_is_still_not_made_again() -> None:
    asked: Final = _asks("c1", "c1", "c2")

    assert _replayed([asked, _asks("c2"), _output("c2")]) == [asked, _output("c2")]


def test_a_repeat_before_a_resent_call_that_says_more_is_still_one() -> None:
    asked: Final = _asks("c1", "c2")
    marked: Final = {**_asks("c2"), "cache_control": _EPHEMERAL}

    replayed: Final = _replayed([asked, _asks("c1"), _output("c1"), marked, _output("c2")])

    assert replayed == [asked, _output("c1"), marked, _output("c2")]


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
        pytest.param(
            [_asks(_SIGNED, "c2"), _output(_OTHERWISE_SIGNED), _asks("c1"), _output("c1")],
            id="call-already-answered-by-its-id-signed-otherwise",
        ),
        pytest.param([_USER, _asks("c1"), _USER, _asks("c1"), _output("c1")], id="id-reused-after-a-user-turn"),
        pytest.param([_USER, _asks("c1"), _ANSWER, _asks("c1"), _output("c1")], id="id-reused-after-an-answer"),
        pytest.param([_asks("c1", "c2"), _DEVELOPER, _asks("c1"), _output("c1")], id="id-reused-after-instructions"),
        pytest.param([_asks("c1", "c2"), _SYSTEM, _asks("c1"), _output("c1")], id="id-reused-after-a-system-message"),
        pytest.param([_asks("c1", "c2"), _output("c1"), _stored(_asks("c2"))], id="id-reused-by-a-stored-turn"),
        pytest.param([_USER, _asks("c1"), _asks("c2"), _output("c1"), _output("c2")], id="call-made-nowhere-else"),
        pytest.param(
            [_asks("c1", "c2"), _asks("c1", "c3"), _output("c1"), _output("c3"), _asks("c2"), _output("c2")],
            id="a-new-call-among-the-repeated",
        ),
        pytest.param(
            [_asks(_SIGNED, "c2"), _asks(_OTHERWISE_SIGNED, "c2"), _output(_OTHERWISE_SIGNED), _output("c2")],
            id="an-id-otherwise-signed-among-the-repeated",
        ),
        pytest.param([_asks(_SIGNED), _asks(_OTHERWISE_SIGNED), _output(_OTHERWISE_SIGNED)], id="id-otherwise-signed"),
        pytest.param(
            [_asks(_SIGNED, _OTHERWISE_SIGNED), _asks("c1"), _output("c1")], id="id-of-two-calls-without-its-signature"
        ),
        pytest.param(
            [_asks("c1", "c1", arguments=_CUT), _asks("c1", arguments=_WHOLE), _output("c1")], id="id-made-twice"
        ),
        pytest.param([_asks("c1", "c1", "c2"), _asks("c1"), _output("c1")], id="id-made-twice-next-to-another"),
        pytest.param([_asks("c1"), _asks("c1", "c1"), _asks("c1"), _output("c1")], id="id-made-twice-by-a-later-turn"),
        pytest.param([_asks("c1", "c1", "c1"), _asks("c1"), _output("c1")], id="id-made-three-times"),
        pytest.param(
            [_asks("c1", "c1", arguments=_CUT), _asks("c1", "c1", arguments=_WHOLE), _output("c1"), _output("c1")],
            id="id-made-twice-resent-together",
        ),
        pytest.param([_asks("c1", "c2"), _asks("c1", "c1"), _output("c1")], id="one-call-resent-twice-at-once"),
        pytest.param(
            [_asks(_SIGNED, "c2"), _asks(_SIGNED, "c1"), _output(_SIGNED)], id="one-call-resent-at-once-under-two-ids"
        ),
        pytest.param([_asks("c1", "c2"), _asks("c12"), _output("c12")], id="id-that-starts-with-another"),
        pytest.param([_asks("c12", "c2"), _asks("c1"), _output("c1")], id="id-that-starts-another"),
        pytest.param(
            [_asks("c1", "c2"), {**_asks("c1"), "cache_control": _EPHEMERAL}, _output("c1"), _asks("c2")],
            id="after-a-resent-call-that-says-more",
        ),
        pytest.param(
            [
                _asks("c1", "c2", "c3"),
                {"role": "assistant", "tool_calls": [{**_call("c1"), **_FIELDS}, _call("c2")]},
                _output("c1"),
                _output("c2"),
                _asks("c3"),
                _output("c3"),
            ],
            id="one-of-the-repeated-calls-saying-more",
        ),
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
        pytest.param([_asks(_ONLY_SIGNED), _asks(_ONLY_SIGNED), _output(_ONLY_SIGNED)], id="ids-of-only-a-signature"),
        pytest.param([{"role": "assistant", "tool_calls": [{"type": "function", "function": {}}]}] * 2, id="no-ids"),
        pytest.param([_asks("c1"), {"role": "assistant", "tool_calls": [{"id": "c1"}]}], id="no-function"),
        pytest.param([_asks("c1"), {"role": "assistant", "tool_calls": ["c1"]}], id="no-call"),
        pytest.param(
            [_asks("c1"), {"role": "assistant", "tool_calls": [{**_call("c1"), "id": ["c1"]}]}], id="id-of-no-text"
        ),
    ],
)
def test_every_other_message_stays(messages: list[dict[str, object]]) -> None:
    assert _replayed(messages) == messages
