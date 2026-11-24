import json
from collections.abc import Mapping, Sequence
from typing import Final, cast
from unittest.mock import Mock

import pytest

import litellm
from litellm.completion_extras.litellm_responses_transformation.configuration import (
    CONFIGURATION_FIELD,
    SUPPORTS_CONFIGURATION_UPDATE,
)
from litellm.completion_extras.litellm_responses_transformation.history import retained_replay_fields
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    BridgedRequest,
    LiteLLMResponsesTransformationHandler,
    OpenAiResponsesToChatCompletionStreamIterator,
)
from litellm.types.llms.openai import AllMessageValues, ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.utils import ModelResponse

_CONFIGURATION_ITEM_TYPES: Final = frozenset({"configuration_update", "additional_tools"})


def _function_tool(name: str, description: str = "") -> Mapping[str, object]:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": {}}},
    }


def _responses_function_tool(name: str, description: str = "") -> Mapping[str, object]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
        "strict": None,
    }


_SCHEMA_FORMAT: Final = {
    "type": "json_schema",
    "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}},
}


def _bridged(
    messages: Sequence[Mapping[str, object]],
    optional_params: Mapping[str, object],
    *,
    model: str = "gpt-6.2",
    litellm_params: Mapping[str, object] | None = None,
) -> BridgedRequest:
    return LiteLLMResponsesTransformationHandler().transform_request_with_configuration(
        model=model,
        messages=cast(list[AllMessageValues], [dict(message) for message in messages]),  # cast-ok: chat test messages
        optional_params=dict(optional_params),
        litellm_params=dict(litellm_params or {"custom_llm_provider": "openai"}),
        headers={},
        litellm_logging_obj=Mock(),
    )


def _wire(bridged: BridgedRequest) -> Mapping[str, object]:
    return {key: value for key, value in bridged.request_data.items() if key != "litellm_logging_obj"}


def _top_level(bridged: BridgedRequest) -> Mapping[str, object]:
    return {key: bridged.request_data.get(key) for key in ("reasoning", "text", "tools")}


def _configuration_items(bridged: BridgedRequest) -> list[object]:
    items: Final = cast(list[Mapping[str, object]], bridged.request_data["input"])
    return [item for item in items if item.get("type") in _CONFIGURATION_ITEM_TYPES]


def _reply(
    bridged: BridgedRequest,
    optional_params: Mapping[str, object],
    *,
    text: str,
    call_id: str | None = None,
) -> Mapping[str, object]:
    message_item: Final = {
        "type": "message",
        "id": f"msg_{text}",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    call_item: Final = {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "get_weather",
        "arguments": "{}",
        "status": "completed",
    }
    raw_response: Final = ResponsesAPIResponse(
        id=f"resp_{text}",
        created_at=1760144904,
        model="gpt-6.2",
        object="response",
        output=[message_item, call_item] if call_id else [message_item],
        status="completed",
        usage=ResponseAPIUsage(input_tokens=1, output_tokens=1, total_tokens=2),
    )
    response: Final = LiteLLMResponsesTransformationHandler().transform_response(
        model="gpt-6.2",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=Mock(),
        request_data=bridged.request_data,
        messages=[],
        optional_params=dict(optional_params),
        litellm_params={"custom_llm_provider": "openai"},
        encoding=None,
        configuration_note=bridged.configuration_note,
    )
    return response.choices[0].message.model_dump(exclude_none=True)


def _user(text: str) -> Mapping[str, object]:
    return {"role": "user", "content": text}


def test_mid_conversation_changes_go_inline_and_every_turn_extends_the_previous_input() -> None:
    turn_1_params: Final = {"reasoning_effort": "low", "tools": [_function_tool("get_weather")]}
    turn_2_params: Final = {**turn_1_params, "reasoning_effort": "high"}
    turn_3_params: Final = {
        **turn_2_params,
        "tools": [_function_tool("get_weather"), _function_tool("get_time")],
        "response_format": _SCHEMA_FORMAT,
    }

    turn_1: Final = _bridged([_user("one")], turn_1_params)
    history_1: Final = [_user("one"), _reply(turn_1, turn_1_params, text="first"), _user("two")]
    turn_2: Final = _bridged(history_1, turn_2_params)
    history_2: Final = [*history_1, _reply(turn_2, turn_2_params, text="second"), _user("three")]
    turn_3: Final = _bridged(history_2, turn_3_params)
    history_3: Final = [*history_2, _reply(turn_3, turn_3_params, text="third"), _user("four")]
    turn_4: Final = _bridged(history_3, turn_3_params)

    inputs: Final = [cast(list[object], turn["input"]) for turn in map(_wire, (turn_1, turn_2, turn_3, turn_4))]
    for earlier, later in zip(inputs, inputs[1:]):
        assert later[: len(earlier)] == earlier, "a later turn must resend the earlier input byte for byte"
    for turn in (turn_2, turn_3, turn_4):
        assert _top_level(turn) == _top_level(turn_1), "request-level settings stay at the first turn's values"
    assert _top_level(turn_1) == {
        "reasoning": {"effort": "low"},
        "text": None,
        "tools": [_responses_function_tool("get_weather")],
    }
    effort_update: Final = {"type": "configuration_update", "reasoning": {"effort": "high"}}
    rewritten_text: Final = _bridged([_user("one")], turn_3_params).request_data["text"]
    format_update: Final = {"type": "configuration_update", "text": rewritten_text}
    added_tools: Final = {
        "type": "additional_tools",
        "role": "developer",
        "tools": [_responses_function_tool("get_time")],
    }
    assert inputs[1][-2:] == [effort_update, inputs[1][-1]]
    assert inputs[2][-3:-1] == [format_update, added_tools]
    assert _configuration_items(turn_4) == [effort_update, format_update, added_tools]


def test_a_change_during_a_tool_continuation_lands_after_the_tool_output_and_stays_there() -> None:
    low: Final = {"reasoning_effort": "low", "tools": [_function_tool("get_weather")]}
    high: Final = {**low, "reasoning_effort": "high"}
    turn_1: Final = _bridged([_user("one")], low)
    tool_result: Final = {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
    continuation_history: Final = [_user("one"), _reply(turn_1, low, text="checking", call_id="call_1"), tool_result]

    continuation: Final = _bridged(continuation_history, high)
    next_turn: Final = _bridged(
        [*continuation_history, _reply(continuation, high, text="sunny today"), _user("thanks")], high
    )

    continuation_input: Final = cast(list[Mapping[str, object]], continuation.request_data["input"])
    next_input: Final = cast(list[Mapping[str, object]], next_turn.request_data["input"])
    assert [item.get("type") for item in continuation_input[-2:]] == ["function_call_output", "configuration_update"]
    assert next_input[: len(continuation_input)] == continuation_input
    assert _configuration_items(next_turn) == [{"type": "configuration_update", "reasoning": {"effort": "high"}}]


_LOW: Final = {"reasoning_effort": "low", "verbosity": "low", "tools": [_function_tool("get_weather")]}


@pytest.mark.parametrize(
    "turn_2_params",
    [
        pytest.param({**_LOW, "tools": []}, id="removed_tool"),
        pytest.param({**_LOW, "tools": [_function_tool("get_weather", "edited")]}, id="edited_tool"),
        pytest.param({**_LOW, "tools": [*_LOW["tools"], {"type": "web_search"}]}, id="hosted_tool"),
        pytest.param({**_LOW, "verbosity": "high"}, id="verbosity"),
        pytest.param({**_LOW, "response_format": {"type": "json_object"}}, id="json_object"),
        pytest.param({**_LOW, "reasoning_effort": "high", "truncation": "auto"}, id="truncation_auto"),
    ],
)
def test_a_change_no_item_can_express_rewrites_the_request_level_settings_and_becomes_the_new_baseline(
    turn_2_params: Mapping[str, object],
) -> None:
    turn_1: Final = _bridged([_user("one")], _LOW)
    history: Final = [_user("one"), _reply(turn_1, _LOW, text="first"), _user("two")]
    assistant_fields: Final = cast(Mapping[str, object], history[1]["provider_specific_fields"])
    without_note: Final = [
        history[0],
        {
            **history[1],
            "provider_specific_fields": {k: v for k, v in assistant_fields.items() if k != CONFIGURATION_FIELD},
        },
        history[2],
    ]
    turn_3_params: Final = {
        **{key: value for key, value in turn_2_params.items() if key != "truncation"},
        "reasoning_effort": "medium",
    }

    turn_2: Final = _bridged(history, turn_2_params)
    turn_3: Final = _bridged([*history, _reply(turn_2, turn_2_params, text="second"), _user("three")], turn_3_params)

    assert _wire(turn_2) == _wire(_bridged(without_note, turn_2_params)), (
        "a rewrite sends what a note-less caller sends"
    )
    assert _configuration_items(turn_2) == []
    assert turn_2.configuration_note is not None and turn_2.configuration_note.effective is None, (
        "a rewrite makes the requested settings the note's base"
    )
    assert {**_top_level(turn_3), "reasoning": None} == {**_top_level(turn_2), "reasoning": None}
    assert turn_3.request_data["reasoning"] == {"effort": cast(Mapping[str, object], turn_2_params)["reasoning_effort"]}
    assert _configuration_items(turn_3)[-1:] == [{"type": "configuration_update", "reasoning": {"effort": "medium"}}]


@pytest.mark.parametrize(
    "broken_note",
    [pytest.param(None, id="dropped"), pytest.param({"version": 2}, id="malformed")],
)
def test_history_without_a_readable_note_sends_the_new_settings_at_request_level(broken_note: object) -> None:
    turn_1: Final = _bridged([_user("one")], _LOW)
    reply: Final = _reply(turn_1, _LOW, text="first")
    fields: Final = cast(Mapping[str, object], reply["provider_specific_fields"])
    broken_fields: Final = {
        **{key: value for key, value in fields.items() if key != CONFIGURATION_FIELD},
        **({CONFIGURATION_FIELD: broken_note} if broken_note is not None else {}),
    }
    high: Final = {**_LOW, "reasoning_effort": "high"}

    turn_2: Final = _bridged([_user("one"), {**reply, "provider_specific_fields": broken_fields}, _user("two")], high)

    assert _configuration_items(turn_2) == []
    assert turn_2.request_data["reasoning"] == {"effort": "high"}


@pytest.mark.parametrize(
    ("model", "litellm_params", "inlines"),
    [
        pytest.param("gpt-6.2", {"custom_llm_provider": "openai"}, True, id="gpt_6_openai"),
        pytest.param("prod-deploy", {"custom_llm_provider": "azure", "base_model": "gpt-6.1"}, True, id="azure_base"),
        pytest.param(
            "prod-deploy",
            {"custom_llm_provider": "azure", "model_info": {"id": "d1", "base_model": "azure/gpt-6.1"}},
            True,
            id="azure_model_info_base",
        ),
        pytest.param("prod-deploy", {"custom_llm_provider": "azure"}, False, id="azure_unknown_deployment"),
        pytest.param(
            "gpt-6.2-sol",
            {"custom_llm_provider": "openai", "model_info": {"id": "d1", SUPPORTS_CONFIGURATION_UPDATE: False}},
            False,
            id="deployment_flag_off",
        ),
        pytest.param(
            "qa-configuration-alias",
            {"custom_llm_provider": "openai", "base_model": "qa-configuration-flagged"},
            True,
            id="cost_map_flag_via_base_model",
        ),
        pytest.param(
            "qa-gpt-6.9-unflagged", {"custom_llm_provider": "openai"}, False, id="cost_map_entry_without_flag"
        ),
        pytest.param("gpt-7.1-preview", {"custom_llm_provider": "openai"}, True, id="unlisted_newer_name"),
        pytest.param("gpt-5.5", {"custom_llm_provider": "openai"}, False, id="gpt_5_5"),
        pytest.param("gpt-6.2", {"custom_llm_provider": "azure_ai"}, False, id="other_provider"),
    ],
)
def test_only_gpt_6_and_newer_on_openai_or_azure_get_inline_updates_and_a_note(
    model: str, litellm_params: Mapping[str, object], inlines: bool
) -> None:
    litellm.register_model(
        {
            "qa-configuration-flagged": {
                "litellm_provider": "openai",
                "mode": "chat",
                SUPPORTS_CONFIGURATION_UPDATE: True,
            },
            "qa-gpt-6.9-unflagged": {"litellm_provider": "openai", "mode": "chat"},
        }
    )
    turn_1: Final = _bridged([_user("one")], _LOW)
    history: Final = [_user("one"), _reply(turn_1, _LOW, text="first"), _user("two")]

    turn_2: Final = _bridged(history, {**_LOW, "reasoning_effort": "high"}, model=model, litellm_params=litellm_params)

    assert bool(_configuration_items(turn_2)) is inlines
    assert (turn_2.configuration_note is not None) is inlines


def test_the_note_rides_on_the_final_stream_chunk_and_survives_the_tool_loop() -> None:
    note: Final = _bridged([_user("one")], _LOW).configuration_note
    completed: Final = {
        "type": "response.completed",
        "response": {"id": "resp_1", "status": "completed", "output": []},
    }

    chunk: Final = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
        completed, replays_openai_history=True, configuration_note=note
    )

    fields: Final = chunk.choices[0].delta.provider_specific_fields
    assert note is not None
    assert fields == {
        CONFIGURATION_FIELD: {
            "version": 1,
            "base": note.base.model_dump(mode="json", exclude_defaults=True),
            "starts": True,
        }
    }
    assert retained_replay_fields(fields) == fields


def test_history_trimmed_past_an_inline_change_sends_the_current_settings_at_request_level() -> None:
    high: Final = {**_LOW, "reasoning_effort": "high"}
    turn_1: Final = _bridged([_user("one")], _LOW)
    history: Final = [_user("one"), _reply(turn_1, _LOW, text="first"), _user("two")]
    turn_2: Final = _bridged(history, high)

    trimmed_turn: Final = _bridged([_user("two"), _reply(turn_2, high, text="second"), _user("three")], high)

    assert _configuration_items(trimmed_turn) == []
    assert trimmed_turn.request_data["reasoning"] == {"effort": "high"}


def _with_note(reply: Mapping[str, object], note: object) -> Mapping[str, object]:
    fields: Final = cast(Mapping[str, object], reply["provider_specific_fields"])
    kept: Final = {key: value for key, value in fields.items() if key != CONFIGURATION_FIELD}
    return {**reply, "provider_specific_fields": {**kept, **({CONFIGURATION_FIELD: note} if note is not None else {})}}


def _two_turns_at(params: Mapping[str, object]) -> list[Mapping[str, object]]:
    turn_1: Final = _bridged([_user("one")], params)
    history_1: Final = [_user("one"), _reply(turn_1, params, text="first"), _user("two")]
    turn_2: Final = _bridged(history_1, params)
    return [*history_1, _reply(turn_2, params, text="second"), _user("three")]


def _assert_rewritten(bridged: BridgedRequest, params: Mapping[str, object]) -> None:
    assert _configuration_items(bridged) == []
    assert _top_level(bridged) == _top_level(_bridged([_user("one")], params)), (
        "request-level settings are the new ones"
    )
    assert bridged.configuration_note is not None and bridged.configuration_note.effective is None


def test_context_management_rewrites_the_request_level_settings() -> None:
    history: Final = _two_turns_at(_LOW)
    params: Final = {**_LOW, "reasoning_effort": "high", "context_management": [{"type": "compaction"}]}

    _assert_rewritten(_bridged(history, params), params)


def test_a_deferred_tool_is_never_added_inline() -> None:
    searchable: Final = {**_LOW, "tools": [*_LOW["tools"], {"type": "tool_search"}]}
    deferred: Final = {**_function_tool("get_time"), "defer_loading": True}
    params: Final = {**searchable, "tools": [*searchable["tools"], deferred]}

    _assert_rewritten(_bridged(_two_turns_at(searchable), params), params)


def test_a_malformed_note_inside_the_segment_rewrites() -> None:
    history: Final = _two_turns_at(_LOW)
    corrupted: Final = [*history[:1], _with_note(history[1], {"version": 1, "base": "broken"}), *history[2:]]
    params: Final = {**_LOW, "reasoning_effort": "high"}

    _assert_rewritten(_bridged(corrupted, params), params)


def test_an_unstamped_assistant_between_stamped_ones_rewrites() -> None:
    history: Final = _two_turns_at(_LOW)
    turn_3: Final = _bridged(history, _LOW)
    stamped_history: Final = [*history, _reply(turn_3, _LOW, text="third"), _user("four")]
    gapped: Final = [*stamped_history[:3], _with_note(stamped_history[3], None), *stamped_history[4:]]
    params: Final = {**_LOW, "reasoning_effort": "high"}

    _assert_rewritten(_bridged(gapped, params), params)


def test_duplicate_tool_names_rewrite() -> None:
    duplicated: Final = {**_LOW, "tools": [_function_tool("get_weather"), _function_tool("get_weather", "other")]}
    params: Final = {**duplicated, "reasoning_effort": "high"}

    _assert_rewritten(_bridged(_two_turns_at(duplicated), params), params)


def test_dropping_reasoning_effort_rewrites_instead_of_sending_an_empty_effort() -> None:
    params: Final = {key: value for key, value in _LOW.items() if key != "reasoning_effort"}

    bridged: Final = _bridged(_two_turns_at(_LOW), params)

    _assert_rewritten(bridged, params)
    assert bridged.request_data.get("reasoning") is None


def test_a_reasoning_summary_change_rewrites() -> None:
    summarized: Final = {**_LOW, "reasoning_effort": {"effort": "low", "summary": "auto"}}
    params: Final = {**_LOW, "reasoning_effort": {"effort": "high", "summary": "detailed"}}

    _assert_rewritten(_bridged(_two_turns_at(summarized), params), params)


def test_a_rewrite_back_to_earlier_settings_starts_a_new_stretch_that_later_changes_extend_inline() -> None:
    with_time_tool: Final = {**_LOW, "tools": [*_LOW["tools"], _function_tool("get_time")]}
    high: Final = {**_LOW, "reasoning_effort": "high"}
    turn_1: Final = _bridged([_user("one")], _LOW)
    history_1: Final = [_user("one"), _reply(turn_1, _LOW, text="first"), _user("two")]
    turn_2: Final = _bridged(history_1, with_time_tool)
    history_2: Final = [*history_1, _reply(turn_2, with_time_tool, text="second"), _user("three")]
    removed_tool: Final = _bridged(history_2, _LOW)
    history_3: Final = [*history_2, _reply(removed_tool, _LOW, text="third"), _user("four")]

    turn_4: Final = _bridged(history_3, high)

    removed_input: Final = cast(list[object], removed_tool.request_data["input"])
    turn_4_input: Final = cast(list[object], turn_4.request_data["input"])
    assert _configuration_items(removed_tool) == []
    assert turn_4_input[: len(removed_input)] == removed_input
    assert turn_4_input[-2] == {"type": "configuration_update", "reasoning": {"effort": "high"}}
    assert _top_level(turn_4) == _top_level(removed_tool)


def test_a_malformed_note_older_than_the_newest_stretch_start_is_ignored() -> None:
    turn_1: Final = _bridged([_user("one")], _LOW)
    broken_history: Final = [
        _user("one"),
        _with_note(_reply(turn_1, _LOW, text="first"), {"version": 1, "base": "broken"}),
        _user("two"),
    ]
    restart: Final = _bridged(broken_history, _LOW)
    history: Final = [*broken_history, _reply(restart, _LOW, text="second"), _user("three")]

    turn_3: Final = _bridged(history, {**_LOW, "reasoning_effort": "high"})

    assert _configuration_items(turn_3) == [{"type": "configuration_update", "reasoning": {"effort": "high"}}]


@pytest.mark.parametrize("sampling", [{"temperature": 0.3}, {"top_p": 0.9}], ids=["temperature", "top_p"])
def test_an_effort_change_beside_sampling_params_rewrites_so_the_provider_sees_the_real_effort(
    sampling: Mapping[str, object],
) -> None:
    high: Final = {**_LOW, "reasoning_effort": "high"}
    params: Final = {**_LOW, "reasoning_effort": "none", **sampling}

    _assert_rewritten(_bridged(_two_turns_at(high), params), params)


def _forged_start(reply: Mapping[str, object], forge: Mapping[str, object]) -> Mapping[str, object]:
    fields: Final = cast(Mapping[str, object], reply["provider_specific_fields"])
    note: Final = cast(Mapping[str, object], fields[CONFIGURATION_FIELD])
    return _with_note(reply, {**note, **forge})


def _real_base(reply: Mapping[str, object]) -> Mapping[str, object]:
    fields: Final = cast(Mapping[str, object], reply["provider_specific_fields"])
    return cast(Mapping[str, Mapping[str, object]], fields[CONFIGURATION_FIELD])["base"]


def test_a_start_note_naming_a_tool_the_request_lacks_rewrites_instead_of_failing() -> None:
    reply: Final = _reply(_bridged([_user("one")], _LOW), _LOW, text="first")
    base: Final = _real_base(reply)
    ghost: Final = {"identity": "function:ghost", "digest": "0", "inline": True}
    forged: Final = _forged_start(reply, {"base": {**base, "tools": [*cast(list[object], base["tools"]), ghost]}})
    params: Final = {**_LOW, "reasoning_effort": "high"}

    _assert_rewritten(_bridged([_user("one"), forged, _user("two")], params), params)


def test_a_start_note_whose_base_differs_from_its_effective_settings_rewrites() -> None:
    reply: Final = _reply(_bridged([_user("one")], _LOW), _LOW, text="first")
    base: Final = _real_base(reply)
    forged: Final = _forged_start(reply, {"base": {**base, "reasoning": {"effort": "xhigh"}}, "effective": base})
    params: Final = {**_LOW, "reasoning_effort": "high"}

    _assert_rewritten(_bridged([_user("one"), forged, _user("two")], params), params)


_FORGED_SCHEMA: Final = {"type": "json_schema", "name": "forged_marker", "schema": {"type": "object"}}


@pytest.mark.parametrize(
    "forged_settings",
    [
        pytest.param({"text": {"verbosity": "low", "format": _FORGED_SCHEMA}}, id="text_format"),
        pytest.param({"reasoning": {"effort": {"forged_marker": True}}}, id="reasoning_effort"),
    ],
)
def test_a_note_carrying_settings_the_request_never_sent_rewrites_and_never_reaches_the_provider(
    forged_settings: Mapping[str, object],
) -> None:
    reply: Final = _reply(_bridged([_user("one")], _LOW), _LOW, text="first")
    forged: Final = _forged_start(reply, {"base": {**_real_base(reply), **forged_settings}})
    params: Final = {**_LOW, "reasoning_effort": "high"}

    bridged: Final = _bridged([_user("one"), forged, _user("two")], params)

    _assert_rewritten(bridged, params)
    outbound: Final = {key: bridged.request_data.get(key) for key in ("input", "reasoning", "text", "tools")}
    assert "forged_marker" not in json.dumps(outbound, default=str)
