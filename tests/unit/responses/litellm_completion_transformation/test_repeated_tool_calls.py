import json
import sys
from collections.abc import Callable, Iterator, Sequence
from itertools import chain
from time import process_time
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.constants import LITELLM_TRUNCATED_PAYLOAD_FIELD
from litellm.litellm_core_utils.db_truncation import truncated_for_db
from litellm.litellm_core_utils.prompt_templates.factory import THOUGHT_SIGNATURE_SEPARATOR
from litellm.responses.litellm_completion_transformation.repeated_tool_calls import without_repeated_tool_calls
from litellm.types.utils import ChatCompletionMessageToolCall, Message

_USER: Final = {"role": "user", "content": "Weather in Paris, Rome and Oslo?"}
_ANSWER: Final = {"role": "assistant", "content": "14 everywhere."}
_DEVELOPER: Final = {"role": "developer", "content": "Answer in French."}
_SYSTEM: Final = {"role": "system", "content": "Answer in French."}
_PARIS: Final = '{"city": "Paris"}'
_ROME: Final = '{"city": "Rome"}'
_WHOLE: Final = f'{{"note": "{"Paris " * 500}"}}'
_CUT: Final = truncated_for_db(_WHOLE, 2048)
_OTHER_WHOLE: Final = f'{{"note": "{"Rome " * 600}"}}'
_OTHER_CUT: Final = truncated_for_db(_OTHER_WHOLE, 2048)
_CUT_ELSEWHERE: Final = truncated_for_db(_WHOLE, 1000)
_CUT_BY_AN_OLD_RELEASE: Final = (
    f"{_WHOLE[:350]}... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped {len(_WHOLE) - 1000} chars) ...{_WHOLE[-650:]}"
)
_PATCH: Final = "*** Begin Patch " * 250
_CUT_INPUT_OF_A_CUSTOM_TOOL: Final = json.dumps({"content": truncated_for_db(_PATCH, 2048)})
_CUT_ARGUMENTS_OF_A_CUSTOM_TOOL: Final = truncated_for_db(json.dumps({"content": _PATCH}), 2048)
_TEXT_WITH_THE_MARKER: Final = f"The row says ... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped 5 chars) ... there."
_PARIS_MANY_TIMES: Final = f"{{{', '.join([_PARIS[1:-1]] * 20000)}}}"
_FULL_OF_MARKERS: Final = truncated_for_db(
    f"... ({LITELLM_TRUNCATED_PAYLOAD_FIELD} skipped 1 chars) ..." * 9000, 300000
)
_JUST_OVER_THE_LIMIT: Final = f'{{"note": "{"x" * 2040}"}}'
_EPHEMERAL: Final = {"type": "ephemeral"}
_FIELDS: Final = {"provider_specific_fields": {"trace": "t1"}}
_SIGNED: Final = f"c1{THOUGHT_SIGNATURE_SEPARATOR}EmwKAhgB"
_OTHERWISE_SIGNED: Final = f"c1{THOUGHT_SIGNATURE_SEPARATOR}EmwK"
_ONLY_SIGNED: Final = f"{THOUGHT_SIGNATURE_SEPARATOR}EmwK"


def _call(call_id: str, arguments: object = "{}", name: str | None = "get_weather") -> dict[str, object]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def _asks(*call_ids: str, arguments: object = "{}", name: str | None = "get_weather") -> dict[str, object]:
    calls: Final = [_call(call_id, arguments, name) for call_id in call_ids]
    return {"role": "assistant", "content": None, "tool_calls": calls}


def _remembers(call_id: str, arguments: str = "{}", name: str = "get_weather") -> dict[str, object]:
    return {"role": "assistant", "tool_calls": [{**_call(call_id, arguments, name), "index": 0}]}


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


def _seconds_to_replay(messages: Sequence[object]) -> float:
    started: Final = process_time()
    _replayed(messages)
    return process_time() - started


def _target_is_not_looked_up() -> bool:
    pytest.fail("the target was looked up with no arguments to give back")


_AS_STORED: Final[tuple[Callable[[dict[str, object]], dict[str, object] | Message], ...]] = (dict, _stored)


@pytest.mark.parametrize("as_stored", _AS_STORED, ids=["message-dict", "stored-message"])
@pytest.mark.parametrize("output_order", [("c1", "c2", "c3"), ("c3", "c1", "c2"), ("c1",)], ids="-".join)
def test_calls_a_turn_already_made_are_not_made_again_before_their_outputs(
    as_stored: Callable[[dict[str, object]], dict[str, object] | Message], output_order: tuple[str, ...]
) -> None:
    asked: Final = as_stored(_asks(*sorted(output_order)))
    repeats: Final = [_remembers(call_id) for call_id in output_order]
    once: Final = [_output(call_id) for call_id in output_order]
    repeated: Final = chain.from_iterable(zip(repeats, once))

    assert _replayed([_USER, asked, *repeated, _ANSWER, _USER]) == [_USER, asked, *once, _ANSWER, _USER]


@pytest.mark.parametrize(
    ("stored", "remembered", "name"),
    [
        pytest.param(_PARIS, _ROME, "get_weather", id="other-arguments"),
        pytest.param(_PARIS, _PARIS, "delete_all", id="another-tool"),
        pytest.param("{}", _ROME, "get_weather", id="stored-redacted"),
        pytest.param("", _ROME, "get_weather", id="empty-stored"),
        pytest.param(None, _ROME, "get_weather", id="nothing-stored"),
        pytest.param(_CUT, _OTHER_WHOLE, "get_weather", id="stored-cut-from-other-arguments"),
        pytest.param(_WHOLE, _CUT, "get_weather", id="remembered-cut"),
        pytest.param("{}", _CUT, "get_weather", id="stored-redacted-remembered-cut"),
    ],
)
@pytest.mark.parametrize("as_stored", _AS_STORED, ids=["message-dict", "stored-message"])
def test_a_call_remembered_from_another_conversation_neither_is_made_nor_changes_the_stored_one(
    as_stored: Callable[[dict[str, object]], dict[str, object] | Message],
    stored: str | None,
    remembered: str,
    name: str,
) -> None:
    asked: Final = as_stored(_asks("c1", "c2", arguments=stored))

    replayed: Final = without_repeated_tool_calls(
        [asked, _remembers("c1", remembered, name), _output("c1")], keep_stored_arguments=_target_is_not_looked_up
    )

    assert list(replayed) == [asked, _output("c1")]


def test_a_remembered_call_gives_back_the_arguments_storage_cut() -> None:
    turn, output = _replayed([_asks("c1", "c2", arguments=_CUT), _remembers("c1", _WHOLE), _output("c1")])

    assert _arguments(turn) == [_WHOLE, _CUT]
    assert output == _output("c1")


@pytest.mark.parametrize("stored", ["{}", _CUT], ids=["redacted", "cut"])
def test_what_the_client_resent_stays_over_what_this_process_remembers(stored: str) -> None:
    resent: Final = _asks("c1", arguments=_WHOLE)

    turn, output = _replayed(
        [_asks("c1", arguments=stored), resent, _remembers("c1", _OTHER_WHOLE, "delete_all"), _output("c1")]
    )

    assert _arguments(turn) == [_WHOLE]
    assert output == _output("c1")


def test_a_call_remembered_after_the_client_edited_it_is_not_made_again() -> None:
    asked: Final = _asks("c1", arguments=_PARIS)
    edited: Final = _asks("c1", arguments=_ROME)

    assert _replayed([asked, edited, _remembers("c1", _PARIS), _output("c1")]) == [asked, edited, _output("c1")]


@pytest.mark.parametrize(
    "resent",
    [_CUT, _CUT_ELSEWHERE, _CUT_BY_AN_OLD_RELEASE, _CUT_INPUT_OF_A_CUSTOM_TOOL, _CUT_ARGUMENTS_OF_A_CUSTOM_TOOL],
    ids=["cut", "cut-elsewhere", "cut-by-an-old-release", "cut-custom-tool-input", "cut-custom-tool-arguments"],
)
@pytest.mark.parametrize(
    "stored",
    ["{}", "", None, _PARIS, _WHOLE, _CUT, truncated_for_db(_WHOLE, 500)],
    ids=["redacted", "empty", "nothing", "other", "whole", "cut", "cut-to-less"],
)
def test_a_resent_call_whose_arguments_storage_cut_leaves_the_stored_ones(stored: str | None, resent: str) -> None:
    asked: Final = _asks("c1", "c2", arguments=stored)

    replayed: Final = without_repeated_tool_calls(
        [asked, _asks("c1", "c2", arguments=resent), _output("c1"), _output("c2")],
        keep_stored_arguments=_target_is_not_looked_up,
    )

    assert list(replayed) == [asked, _output("c1"), _output("c2")]


@pytest.mark.parametrize("stored", ["{}", _CUT_ELSEWHERE], ids=["redacted", "cut"])
@pytest.mark.parametrize("resent", [(_CUT, _WHOLE), (_WHOLE, _CUT)], ids=["cut-then-whole", "whole-then-cut"])
def test_arguments_resent_whole_are_given_back_next_to_a_copy_storage_cut(stored: str, resent: tuple[str, str]) -> None:
    copies: Final = [_asks("c1", arguments=arguments) for arguments in resent]

    assert _replayed([_asks("c1", arguments=stored), *copies, _output("c1")]) == [
        _asks("c1", arguments=_WHOLE),
        _output("c1"),
    ]


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


@pytest.mark.parametrize(
    ("stored_name", "resent_name"),
    [
        ("search", "crm__search"),
        ("crm__search", "search"),
        ("search", "eu__crm__search"),
        ("eu__crm__search", "search"),
        ("crm__search", "eu__crm__search"),
        ("eu__crm__search", "crm__search"),
    ],
)
def test_a_call_resent_with_or_without_its_namespace_gives_back_its_arguments_under_the_stored_name(
    stored_name: str, resent_name: str
) -> None:
    stored: Final = _asks("c1", "c2", arguments=_CUT, name=stored_name)
    resent: Final = _resent_one_by_one("c1", "c2", arguments=_WHOLE, name=resent_name)

    turn, *outputs = _replayed([stored, *resent])

    assert turn == _asks("c1", "c2", arguments=_WHOLE, name=stored_name)
    assert outputs == [_output("c1"), _output("c2")]


@pytest.mark.parametrize("stored", ["{}", _CUT, _WHOLE], ids=["redacted", "cut", "whole"])
def test_every_copy_of_a_call_is_compared_with_the_name_storage_holds(stored: str) -> None:
    copies: Final = [_asks("c1", arguments=_WHOLE, name=name) for name in ("crm__search", "erp__search")]

    assert _replayed([_asks("c1", arguments=stored, name="search"), *copies, _output("c1")]) == [
        _asks("c1", arguments=_WHOLE, name="search"),
        _output("c1"),
    ]


def test_a_copy_without_the_signature_is_left_out_between_calls_signed_otherwise() -> None:
    asked: Final = _asks(_SIGNED)
    otherwise: Final = [_asks(_OTHERWISE_SIGNED), _output(_OTHERWISE_SIGNED)]

    assert _replayed([asked, _asks("c1"), *otherwise]) == [asked, *otherwise]


def test_arguments_shorter_than_their_cut_text_are_given_back() -> None:
    stored: Final = truncated_for_db(_JUST_OVER_THE_LIMIT, 2048)
    assert len(stored) > len(_JUST_OVER_THE_LIMIT)

    turn, output = _replayed(
        [_asks("c1", arguments=stored), _asks("c1", arguments=_JUST_OVER_THE_LIMIT), _output("c1")]
    )

    assert _arguments(turn) == [_JUST_OVER_THE_LIMIT]
    assert output == _output("c1")


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


@pytest.mark.parametrize("stored", ["{}", "{ }", "", None], ids=["redacted", "spaced", "empty", "nothing"])
@pytest.mark.parametrize("resent", [_PARIS, "Paris"], ids=["json", "text"])
@pytest.mark.parametrize("as_stored", _AS_STORED, ids=["message-dict", "stored-message"])
def test_a_repeated_call_gives_back_the_arguments_storage_does_not_hold(
    as_stored: Callable[[dict[str, object]], dict[str, object] | Message], stored: str | None, resent: str
) -> None:
    asked: Final = as_stored(_asks("c1", arguments=stored))

    user, turn, output = _replayed([_USER, asked, _asks("c1", arguments=resent), _output("c1")])

    assert user == _USER
    assert _arguments(turn) == [resent]
    assert output == _output("c1")


@pytest.mark.parametrize("stored", ["{}", _CUT], ids=["redacted", "cut"])
def test_the_stored_arguments_stay_when_asked_to_keep_them(stored: str) -> None:
    asked: Final = _asks("c1", "c2", arguments=stored)
    repeated: Final = _resent_one_by_one("c1", "c2", arguments=_WHOLE)

    replayed: Final = without_repeated_tool_calls([asked, *repeated], keep_stored_arguments=lambda: True)

    assert list(replayed) == [asked, _output("c1"), _output("c2")]


@pytest.mark.parametrize(
    ("stored", "repeated"),
    [
        pytest.param(_PARIS, _PARIS, id="as-stored"),
        pytest.param(_PARIS, '{"city":"Paris"}', id="no-spaces"),
        pytest.param('{"a": 1, "b": "é"}', '{"b": "\\u00e9", "a": 1}', id="other-order-and-escapes"),
        pytest.param('{"a": [{"b": 1, "c": 2}]}', '{"a": [{"c": 2, "b": 1}]}', id="other-order-inside"),
        pytest.param("[1, 2]", "[1,2]", id="list"),
        pytest.param("Paris", "Paris", id="no-json"),
        pytest.param("{}", "", id="no-arguments"),
        pytest.param(None, "{ }", id="nothing-stored"),
    ],
)
def test_the_same_arguments_spelled_otherwise_are_the_same_call(stored: str | None, repeated: str) -> None:
    asked: Final = _asks("c1", "c2", arguments=stored)

    replayed: Final = without_repeated_tool_calls(
        [asked, _asks("c1", arguments=repeated), _output("c1")], keep_stored_arguments=_target_is_not_looked_up
    )

    assert list(replayed) == [asked, _output("c1")]


@pytest.mark.parametrize("resent", ["{}", "", None, {}], ids=["no-arguments", "empty", "nothing", "no-text"])
def test_a_stored_call_that_names_no_arguments_stays_as_stored(resent: object) -> None:
    asked: Final = {
        **_asks("c1"),
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather"}}],
    }

    replayed: Final = without_repeated_tool_calls(
        [asked, _asks("c1", arguments=resent), _output("c1")], keep_stored_arguments=_target_is_not_looked_up
    )

    assert list(replayed) == [asked, _output("c1")]


@pytest.mark.parametrize(
    ("stored", "resent"),
    [
        pytest.param(_PARIS, '{"city": "paris"}', id="another-case"),
        pytest.param(_PARIS, '{"city": " Paris "}', id="padded-value"),
        pytest.param(_PARIS, '{"city": "Paris", "day": null}', id="null-member"),
        pytest.param(_PARIS, '{"city": "Paris", "day": ""}', id="empty-member"),
        pytest.param('{"city": "Paris", "day": null}', _PARIS, id="null-member-taken-off"),
        pytest.param(_PARIS, json.dumps(_PARIS), id="encoded-twice"),
        pytest.param(json.dumps(_PARIS), _PARIS, id="stored-encoded-twice"),
        pytest.param(_PARIS, "", id="emptied"),
        pytest.param("Paris", '"Paris"', id="text-in-quotes"),
        pytest.param("Paris", "paris", id="text-in-another-case"),
        pytest.param("Paris", " Paris ", id="padded-text"),
        pytest.param("[]", _PARIS, id="stored-list"),
        pytest.param(" ", _PARIS, id="stored-space"),
        pytest.param("{}x", _PARIS, id="stored-text-that-starts-like-none"),
        pytest.param(_PARIS, _TEXT_WITH_THE_MARKER, id="text-with-the-marker"),
        pytest.param(_PARIS, json.dumps({"content": _TEXT_WITH_THE_MARKER}), id="custom-tool-input-with-the-marker"),
        pytest.param(_PARIS, json.dumps({"log": _CUT}), id="cut-text-in-another-field"),
        pytest.param(_PARIS, json.dumps({"content": [_CUT]}), id="cut-text-in-a-list"),
        pytest.param(_PARIS, json.dumps([{"content": _CUT}]), id="cut-text-in-a-list-of-fields"),
        pytest.param(_PARIS, json.dumps({"content": _CUT, "day": "today"}), id="cut-text-next-to-another-field"),
        pytest.param(
            _PARIS,
            f'{{"{LITELLM_TRUNCATED_PAYLOAD_FIELD}": {"[" * 60000}{"]" * 60000}}}',
            id="marker-and-too-deep-to-read",
        ),
    ],
)
def test_a_call_resent_with_other_arguments_stays_as_sent(stored: str, resent: str) -> None:
    messages: Final = [_asks("c1", arguments=stored), _asks("c1", arguments=resent), _output("c1")]

    assert _replayed(messages) == messages


@pytest.mark.parametrize(
    ("stored_id", "resent_id"), [("c1", "c1"), (_SIGNED, "c1"), ("c1", _SIGNED)], ids=["bare", "stored", "resent"]
)
@pytest.mark.parametrize(
    ("stored_name", "resent_name"),
    [
        ("get_weather", "delete_all"),
        ("get_weather", ""),
        ("", "get_weather"),
        ("get_weather", None),
        (None, "get_weather"),
        ("crm__search", "search__crm"),
        ("crm__search", "crm"),
        ("crm__None", None),
    ],
)
def test_a_call_resent_to_another_tool_stays_as_sent(
    stored_id: str, resent_id: str, stored_name: str | None, resent_name: str | None
) -> None:
    messages: Final = [
        _asks(stored_id, arguments=_PARIS, name=stored_name),
        _asks(resent_id, arguments=_PARIS, name=resent_name),
        _output(resent_id),
    ]

    assert _replayed(messages) == messages


@pytest.fixture
def few_digits_can_be_printed() -> Iterator[None]:
    digits: Final = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(640)
    yield
    sys.set_int_max_str_digits(digits)


@pytest.mark.usefixtures("few_digits_can_be_printed")
def test_a_call_with_a_number_too_long_to_print_stays_as_sent_when_spelled_otherwise() -> None:
    number: Final = "1" * 1000
    asked: Final = _asks("c1", arguments=f'{{"n": {number}}}')
    respelled: Final = [asked, _asks("c1", arguments=f'{{"n":{number}}}'), _output("c1")]

    assert _replayed(respelled) == respelled
    assert _replayed([asked, asked, _output("c1")]) == [asked, _output("c1")]


@pytest.mark.parametrize(
    ("stored", "resent"),
    [
        pytest.param(_PARIS_MANY_TIMES, _PARIS, id="arguments-slow-to-read"),
        pytest.param(_FULL_OF_MARKERS, _CUT, id="cut-arguments-slow-to-search"),
    ],
)
@pytest.mark.parametrize("together", [False, True], ids=["one-by-one", "together"])
def test_a_call_is_not_read_again_for_every_copy_of_it(stored: str, resent: str, together: bool) -> None:
    ids: Final = ["c1"] * 200
    copies: Final = (
        [_asks(*ids, arguments=resent)] if together else [_asks(call_id, arguments=resent) for call_id in ids]
    )
    read_once: Final = _seconds_to_replay([_asks("c1", arguments=stored), _asks("c1", arguments=resent)])
    light: Final = _seconds_to_replay([_asks("c1", arguments=resent), *copies])

    assert _seconds_to_replay([_asks("c1", arguments=stored), *copies]) < 10 * (read_once + light)


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


@pytest.mark.parametrize(
    ("second_copy", "stays"),
    [(_PARIS, False), ('{"city":"Paris"}', False), (_ROME, True)],
    ids=["as-the-first", "spelled-otherwise", "edited"],
)
def test_a_second_copy_is_compared_with_the_arguments_the_first_gave_back(second_copy: str, stays: bool) -> None:
    second: Final = _asks("c1", arguments=second_copy)

    replayed: Final = _replayed([_asks("c1", arguments="{}"), _asks("c1", arguments=_PARIS), second, _output("c1")])

    assert replayed == [_asks("c1", arguments=_PARIS), *([second] if stays else []), _output("c1")]


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
                {"role": "assistant", "content": None, "tool_calls": [{**_call("c1"), **_FIELDS}, _call("c2")]},
                _output("c1"),
                _output("c2"),
                _asks("c3"),
                _output("c3"),
            ],
            id="one-of-the-repeated-calls-saying-more",
        ),
        pytest.param(
            [_asks("c1", arguments=_PARIS), _asks("c1", arguments=_ROME), _output("c1")], id="edited-arguments"
        ),
        pytest.param(
            [_asks("c1", arguments=_PARIS), _asks("c1", arguments="{}"), _output("c1")], id="arguments-taken-off"
        ),
        pytest.param(
            [_asks("c1", arguments=_PARIS), _asks("c1", arguments=None), _output("c1")], id="no-arguments-resent"
        ),
        pytest.param(
            [_asks("c1", arguments='{"a": 1}'), _asks("c1", arguments='{"a": true}'), _output("c1")],
            id="edited-to-another-type",
        ),
        pytest.param(
            [_asks("c1", arguments='{"a": 1}'), _asks("c1", arguments='{"a": 1.0}'), _output("c1")],
            id="edited-to-another-number",
        ),
        pytest.param(
            [_asks("c1", arguments='{"a": [1, 2]}'), _asks("c1", arguments='{"a": [2, 1]}'), _output("c1")],
            id="edited-order",
        ),
        pytest.param([_asks("c1", arguments="Paris"), _asks("c1", arguments="Rome"), _output("c1")], id="edited-text"),
        pytest.param([_asks("c1", arguments="a b"), _asks("c1", arguments="ab"), _output("c1")], id="edited-spaces"),
        pytest.param([_asks("c1", arguments="null"), _asks("c1", arguments=_PARIS), _output("c1")], id="edited-null"),
        pytest.param(
            [_asks("c1", arguments=_CUT), _asks("c1", arguments=None), _output("c1")],
            id="no-arguments-resent-where-storage-cut",
        ),
        pytest.param(
            [_asks("c1", arguments=_CUT), _asks("c1", arguments=_ROME), _output("c1")], id="edited-where-storage-cut"
        ),
        pytest.param(
            [_asks("c1", arguments=_WHOLE), _asks("c1", arguments=_CUT, name="delete_all"), _output("c1")],
            id="another-tool-whose-arguments-storage-cut",
        ),
        pytest.param(
            [
                _asks("c1", arguments=_WHOLE),
                _asks("c1", arguments=_CUT_INPUT_OF_A_CUSTOM_TOOL, name="delete_all"),
                _output("c1"),
            ],
            id="another-tool-whose-input-storage-cut",
        ),
        pytest.param(
            [_asks("c1"), _asks("c1", arguments=_PARIS, name="delete_all"), _output("c1")],
            id="another-tool-where-storage-redacted",
        ),
        pytest.param(
            [_asks("c1", arguments="{}"), _asks("c1", arguments={"city": "Rome"})],
            id="arguments-of-no-text",
        ),
        pytest.param(
            [
                _asks("c1", arguments={"city": "Paris"}),
                _asks("c1", arguments=_PARIS),
                _output("c1"),
            ],
            id="stored-arguments-of-no-text",
        ),
        pytest.param(
            [_asks("c1", arguments=_PARIS), _asks("c1", arguments=_PARIS, name="delete_all"), _output("c1")],
            id="another-tool",
        ),
        pytest.param(
            [_asks("c1", name="crm__search"), _asks("c1", name="erp__search"), _output("c1")],
            id="tool-of-another-namespace",
        ),
        pytest.param(
            [_asks("c1", name="Search"), _asks("c1", name="search"), _output("c1")], id="tool-in-another-case"
        ),
        pytest.param(
            [_asks("c1", name="weather"), _asks("c1", name="get_weather"), _output("c1")],
            id="tool-whose-name-ends-alike",
        ),
        pytest.param(
            [_asks("c1", name="get_weather"), _asks("c1", name="weather"), _output("c1")],
            id="tool-whose-resent-name-ends-alike",
        ),
        pytest.param(
            [_asks("c1", name="crm__get_weather_now"), _asks("c1", name="get_weather"), _output("c1")],
            id="tool-whose-name-holds-the-resent-one",
        ),
        pytest.param(
            [_asks("c1", name="get_weather"), _asks("c1", name="crm__get_weather_now"), _output("c1")],
            id="tool-whose-resent-name-holds-it",
        ),
        pytest.param(
            [
                _asks("c1", "c2", arguments=_PARIS),
                {"role": "assistant", "content": None, "tool_calls": [_call("c1", _PARIS), _call("c2", _ROME)]},
                _output("c1"),
                _output("c2"),
            ],
            id="one-edited-among-the-repeated",
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
            [
                _asks("c1"),
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{**_call("c1"), "function": {"name": "get_weather", "arguments": "{}", **_FIELDS}}],
                },
            ],
            id="function-saying-more",
        ),
        pytest.param([_asks(""), _asks(""), _output("")], id="empty-ids"),
        pytest.param([_asks(_ONLY_SIGNED), _asks(_ONLY_SIGNED), _output(_ONLY_SIGNED)], id="ids-of-only-a-signature"),
        pytest.param([{**_asks("c1"), "tool_calls": [{"type": "function", "function": {}}]}] * 2, id="no-ids"),
        pytest.param([_remembers("c1"), _output("c1")] * 2, id="a-call-made-only-from-memory"),
        pytest.param([_asks("c1"), _remembers("c2"), _output("c2")], id="another-call-from-memory"),
        pytest.param([_asks("c1"), {**_asks("c1"), "tool_calls": [{"id": "c1"}]}], id="no-function"),
        pytest.param([_asks("c1"), {**_asks("c1"), "tool_calls": ["c1"]}], id="no-call"),
        pytest.param([_asks("c1"), {**_asks("c1"), "tool_calls": [{**_call("c1"), "id": ["c1"]}]}], id="id-of-no-text"),
    ],
)
def test_every_other_message_stays(messages: list[dict[str, object]]) -> None:
    assert _replayed(messages) == messages
