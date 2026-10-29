import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.completion_extras.litellm_responses_transformation import history
from litellm.exceptions import MidStreamFallbackError

OPENAI_RESPONSES_URL: Final = "https://api.openai.com/v1/responses"
AZURE_API_BASE: Final = "https://litellm-history-test.openai.azure.com"
MODEL: Final = "gpt-5.6"
TOOLS: Final = (
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
)
USER: Final = {"role": "user", "content": "Weather in Paris? Narrate as you go."}

COMMENTARY: Final = {
    "type": "message",
    "id": "msg_commentary",
    "status": "completed",
    "role": "assistant",
    "phase": "commentary",
    "content": [{"type": "output_text", "text": "Checking the weather.", "annotations": []}],
}
FIRST_REASONING: Final = {
    "type": "reasoning",
    "id": "rs_first",
    "summary": [{"type": "summary_text", "text": "Paris is likely wet."}],
    "encrypted_content": "enc_first",
}
SECOND_REASONING: Final = {"type": "reasoning", "id": "rs_second", "summary": [], "encrypted_content": "enc_second"}
WEATHER_CALL: Final = {
    "type": "function_call",
    "id": "fc_weather",
    "call_id": "call_weather_paris",
    "name": "get_weather",
    "arguments": '{"city":"Paris"}',
    "status": "completed",
}
FINAL_ANSWER: Final = {
    "type": "message",
    "id": "msg_final",
    "status": "completed",
    "role": "assistant",
    "phase": "final_answer",
    "content": [
        {"type": "output_text", "text": "Paris: ", "annotations": []},
        {"type": "output_text", "text": "light rain.", "annotations": []},
    ],
}
OUTPUT: Final = (COMMENTARY, FIRST_REASONING, SECOND_REASONING, WEATHER_CALL, FINAL_ANSWER)
REPLAYED: Final = (
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Checking the weather."}],
        "phase": "commentary",
    },
    {
        "type": "reasoning",
        "id": "rs_first",
        "summary": [{"type": "summary_text", "text": "Paris is likely wet."}],
        "encrypted_content": "enc_first",
    },
    {"type": "reasoning", "id": "rs_second", "summary": [], "encrypted_content": "enc_second"},
    {"type": "function_call", "call_id": "call_weather_paris", "name": "get_weather", "arguments": '{"city":"Paris"}'},
    {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Paris: "}, {"type": "output_text", "text": "light rain."}],
        "phase": "final_answer",
    },
)
TOOL_RESULT: Final = {"role": "tool", "tool_call_id": "call_weather_paris", "content": "light rain"}
FOLLOW_UP: Final = {
    "type": "message",
    "id": "msg_follow_up",
    "status": "completed",
    "role": "assistant",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
}


def _response(output: Sequence[Mapping[str, object]], status: str = "completed") -> dict[str, object]:
    return {
        "id": "resp_history",
        "object": "response",
        "created_at": 1789788253,
        "status": status,
        "model": MODEL,
        "output": list(output),
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


def _added_item(item: Mapping[str, object]) -> Mapping[str, object]:
    match item.get("type"):
        case "message":
            return {**item, "status": "in_progress", "content": []}
        case "function_call":
            return {**item, "status": "in_progress", "arguments": ""}
        case "custom_tool_call":
            return {**item, "status": "in_progress", "input": ""}
        case _:
            return item


def _delta_events(index: int, item: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    match item.get("type"):
        case "message":
            parts: Final = item["content"]
            assert isinstance(parts, list)
            return tuple(
                {
                    "type": "response.output_text.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "content_index": part_index,
                    "delta": part["text"],
                }
                for part_index, part in enumerate(parts)
            )
        case "function_call":
            return (
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "delta": item["arguments"],
                },
            )
        case "custom_tool_call":
            return (
                {
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": item["id"],
                    "output_index": index,
                    "delta": item["input"],
                },
            )
        case _:
            return ()


def _item_events(index: int, item: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return (
        {"type": "response.output_item.added", "output_index": index, "item": _added_item(item)},
        *_delta_events(index, item),
        {"type": "response.output_item.done", "output_index": index, "item": item},
    )


def _sse(output: Sequence[Mapping[str, object]], terminal_event: str | None = "response.completed") -> httpx.Response:
    item_events: Final = tuple(event for index, item in enumerate(output) for event in _item_events(index, item))
    terminal: Final = (
        ()
        if terminal_event is None
        else (
            {
                "type": terminal_event,
                "response": _response(output, "completed" if terminal_event == "response.completed" else "incomplete"),
            },
        )
    )
    events: Final = (
        {"type": "response.created", "response": {**_response((), "in_progress"), "usage": None}},
        *item_events,
        *terminal,
    )
    body: Final = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})


def _assistant_turn(
    output: Sequence[Mapping[str, object]],
    *,
    stream: bool,
    model: str = f"openai/responses/{MODEL}",
    terminal_event: str | None = "response.completed",
    status: str = "completed",
    response_format: Mapping[str, object] | None = None,
) -> dict[str, object]:
    reply: Final = httpx.Response(
        200, content=json.dumps(_response(output, status)), headers={"content-type": "application/json"}
    )
    with respx.mock(assert_all_called=True) as router:
        router.post(url__startswith=OPENAI_RESPONSES_URL if model.startswith("openai/") else AZURE_API_BASE).mock(
            return_value=_sse(output, terminal_event) if stream else reply
        )
        response: Final = litellm.completion(
            model=model,
            messages=[USER],
            tools=list(TOOLS),
            api_key="sk-test",
            api_base=None if model.startswith("openai/") else AZURE_API_BASE,
            api_version=None if model.startswith("openai/") else "2025-04-01-preview",
            stream=stream,
            response_format=response_format,
        )
        message: Final = (
            (litellm.stream_chunk_builder(list(response), messages=[USER]) if stream else response).choices[0].message
        )
    return json.loads(json.dumps(message.model_dump()))


def _next_turn_input(
    assistant: Mapping[str, object],
    model: str = f"openai/responses/{MODEL}",
    tools: Sequence[Mapping[str, object]] = TOOLS,
    drop_params: bool = False,
) -> list[Mapping[str, object]]:
    with respx.mock(assert_all_called=True) as router:
        upstream: Final = router.post(
            url__startswith=OPENAI_RESPONSES_URL if model.startswith("openai/") else AZURE_API_BASE
        ).mock(return_value=httpx.Response(200, json=_response((FOLLOW_UP,))))
        litellm.completion(
            model=model,
            messages=[USER, dict(assistant), TOOL_RESULT],
            tools=list(tools),
            api_key="sk-test",
            api_base=None if model.startswith("openai/") else AZURE_API_BASE,
            api_version=None if model.startswith("openai/") else "2025-04-01-preview",
            drop_params=drop_params,
        )
    return json.loads(upstream.calls[0].request.content)["input"]


def _replayed_fields(message: Mapping[str, object]) -> Mapping[str, object]:
    fields: Final = message.get("provider_specific_fields")
    return fields if isinstance(fields, Mapping) else {}


def _types(items: Sequence[Mapping[str, object]]) -> list[object]:
    return [item.get("type") for item in items]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_next_turn_replays_every_item_phase_and_boundary_once_in_order(stream: bool) -> None:
    assistant: Final = _assistant_turn(OUTPUT, stream=stream)

    sent: Final = _next_turn_input(assistant)

    assert assistant["content"] == "Checking the weather.Paris: light rain."
    assert _replayed_fields(assistant)["phase"] == "final_answer"
    assert sent[1:-1] == list(REPLAYED)
    assert sent[-1]["type"] == "function_call_output"
    assert sent[-1]["call_id"] == "call_weather_paris"


def _edit_content(message: Mapping[str, object]) -> Mapping[str, object]:
    return {**message, "content": "Checking the weather. Paris: heavy rain."}


def _edit_phase(message: Mapping[str, object]) -> Mapping[str, object]:
    return {**message, "provider_specific_fields": {**_replayed_fields(message), "phase": "commentary"}}


def _edit_arguments(message: Mapping[str, object]) -> Mapping[str, object]:
    tool_calls: Final = message["tool_calls"]
    assert isinstance(tool_calls, list)
    edited_call: Final = {**tool_calls[0], "function": {**tool_calls[0]["function"], "arguments": '{"city":"Lyon"}'}}
    return {**message, "tool_calls": [edited_call]}


@pytest.mark.parametrize(
    ("edit", "expected_text", "expected_phase", "expected_arguments"),
    [
        (_edit_content, "Checking the weather. Paris: heavy rain.", "final_answer", '{"city":"Paris"}'),
        (_edit_phase, "Checking the weather.Paris: light rain.", "commentary", '{"city":"Paris"}'),
        (_edit_arguments, "Checking the weather.Paris: light rain.", "final_answer", '{"city":"Lyon"}'),
    ],
    ids=["content", "phase", "arguments"],
)
def test_an_edited_chat_field_wins_over_the_recorded_items(
    edit: Callable[[Mapping[str, object]], Mapping[str, object]],
    expected_text: str,
    expected_phase: str,
    expected_arguments: str,
) -> None:
    sent: Final = _next_turn_input(edit(_assistant_turn(OUTPUT, stream=False)))

    assert _types(sent[1:-1]) == ["reasoning", "reasoning", "message", "function_call"]
    assert sent[3]["content"] == [{"type": "output_text", "text": expected_text}]
    assert sent[3]["phase"] == expected_phase
    assert sent[4]["arguments"] == expected_arguments


def _without_replay(fields: Mapping[str, object]) -> Mapping[str, object]:
    return {key: value for key, value in fields.items() if key != "openai_responses_replay"}


def _with_replay(fields: Mapping[str, object], replay: object) -> Mapping[str, object]:
    return {**fields, "openai_responses_replay": replay}


def _replay_of(fields: Mapping[str, object]) -> Mapping[str, object]:
    replay: Final = fields["openai_responses_replay"]
    assert isinstance(replay, Mapping)
    return replay


@pytest.mark.parametrize(
    "corrupt",
    [
        _without_replay,
        lambda fields: _with_replay(fields, {**_replay_of(fields), "checksum": "0" * 64}),
        lambda fields: _with_replay(fields, {**_replay_of(fields), "version": 2}),
        lambda fields: _with_replay(fields, {**_replay_of(fields), "items": [{"type": "message", "parts": [-1]}]}),
        lambda fields: _with_replay(fields, "not a replay"),
    ],
    ids=["missing", "checksum", "version", "items", "not-a-mapping"],
)
def test_missing_or_corrupt_replay_metadata_falls_back_to_the_chat_fields(
    corrupt: Callable[[Mapping[str, object]], Mapping[str, object]],
) -> None:
    assistant: Final = _assistant_turn(OUTPUT, stream=False)

    sent: Final = _next_turn_input({**assistant, "provider_specific_fields": corrupt(_replayed_fields(assistant))})

    assert _types(sent[1:-1]) == ["reasoning", "reasoning", "message", "function_call"]
    assert sent[3]["content"] == [{"type": "output_text", "text": "Checking the weather.Paris: light rain."}]
    assert sent[3]["phase"] == "final_answer"


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_an_unphased_answer_after_commentary_is_never_labeled_commentary(stream: bool) -> None:
    unphased_answer: Final = {key: value for key, value in FINAL_ANSWER.items() if key != "phase"}
    assistant: Final = _assistant_turn((COMMENTARY, WEATHER_CALL, unphased_answer), stream=stream)

    sent: Final = _next_turn_input(_edit_content(assistant))

    assert "phase" not in _replayed_fields(assistant)
    assert _types(sent[1:-1]) == ["message", "function_call"]
    assert "phase" not in sent[1]


def _with_text(item: Mapping[str, object], text: str) -> Mapping[str, object]:
    return {**item, "content": [{"type": "output_text", "text": text, "annotations": []}]}


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_a_structured_output_turn_carries_the_phase_of_the_message_it_returns(stream: bool) -> None:
    output: Final = (_with_text(COMMENTARY, '{"city":"Paris"}'), _with_text(FINAL_ANSWER, '{"city":"Lyon"}'))

    assistant: Final = _assistant_turn(output, stream=stream, response_format={"type": "json_object"})

    assert assistant["content"] == '{"city":"Paris"}'
    assert _replayed_fields(assistant)["phase"] == "commentary"


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_a_lone_surrogate_in_the_reply_still_gets_a_record(stream: bool) -> None:
    assistant: Final = _assistant_turn((_with_text(COMMENTARY, "Checking \ud83d"), WEATHER_CALL), stream=stream)

    assert "openai_responses_replay" in _replayed_fields(assistant)


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_a_last_message_the_bridge_cannot_read_leaves_the_turn_without_a_phase(stream: bool) -> None:
    unreadable: Final = {**_with_text(FINAL_ANSWER, "It is raining in Paris."), "phase": "summary"}
    assistant: Final = _assistant_turn((COMMENTARY, WEATHER_CALL, unreadable), stream=stream)

    assert "phase" not in _replayed_fields(assistant)


def test_a_phase_openai_does_not_define_is_dropped_under_drop_params() -> None:
    assistant: Final = _assistant_turn(OUTPUT, stream=False)

    sent: Final = _next_turn_input({**assistant, "provider_specific_fields": {"phase": "analysis"}}, drop_params=True)

    assert _types(sent[1:-1]) == ["reasoning", "reasoning", "message", "function_call"]
    assert "phase" not in sent[3]


def test_a_phase_openai_does_not_define_is_rejected_without_drop_params() -> None:
    assistant: Final = _assistant_turn(OUTPUT, stream=False)

    with respx.mock(assert_all_called=False) as router:
        upstream: Final = router.post(url__startswith=OPENAI_RESPONSES_URL).mock(
            return_value=httpx.Response(200, json=_response((FOLLOW_UP,)))
        )
        with pytest.raises(litellm.BadRequestError, match="'analysis'"):
            litellm.completion(
                model=f"openai/responses/{MODEL}",
                messages=[USER, {**assistant, "phase": "analysis"}, TOOL_RESULT],
                tools=list(TOOLS),
                api_key="sk-test",
            )

    assert not upstream.called


@pytest.mark.parametrize("fields", [None, ["not", "a", "mapping"]], ids=["no-fields", "malformed-fields"])
def test_an_assistant_message_the_app_wrote_replays_without_a_phase(fields: object) -> None:
    sent: Final = _next_turn_input(
        {
            "role": "assistant",
            "content": "Checking Paris.",
            "tool_calls": [
                {
                    "id": "call_weather_paris",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
            "provider_specific_fields": fields,
        }
    )

    assert _types(sent[1:-1]) == ["message", "function_call"]
    assert "phase" not in sent[1]


SEARCH_CALL: Final = {
    "type": "tool_search_call",
    "id": "tsc_weather",
    "arguments": {"paths": ["get_weather"]},
    "status": "completed",
}
SEARCH_OUTPUT: Final = {
    "type": "tool_search_output",
    "id": "tso_weather",
    "status": "completed",
    "tools": [{"name": "get_weather"}],
}


@pytest.mark.parametrize(
    ("tools", "expected"),
    [
        ((*TOOLS, {"type": "tool_search"}), ["tool_search_call", "tool_search_output", *_types(REPLAYED)]),
        (TOOLS, ["reasoning", "reasoning", "message", "function_call"]),
    ],
    ids=["searching", "not-searching"],
)
def test_tool_search_items_replay_only_when_the_next_request_searches(
    tools: Sequence[Mapping[str, object]], expected: Sequence[str]
) -> None:
    assistant: Final = _assistant_turn((SEARCH_CALL, SEARCH_OUTPUT, *OUTPUT), stream=False)

    sent: Final = _next_turn_input(assistant, tools=tools)

    assert _types(sent[1:-1]) == list(expected)


WEB_SEARCH: Final = {"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search"}}


@pytest.mark.parametrize(
    ("output", "stream", "terminal_event", "status"),
    [
        (OUTPUT, False, "response.completed", "incomplete"),
        (OUTPUT, True, "response.incomplete", "incomplete"),
        ((WEB_SEARCH, *OUTPUT), False, "response.completed", "completed"),
        ((WEB_SEARCH, *OUTPUT), True, "response.completed", "completed"),
    ],
    ids=["incomplete-json", "incomplete-stream", "unsupported-item-json", "unsupported-item-stream"],
)
def test_an_incomplete_or_unsupported_output_records_no_replay(
    output: Sequence[Mapping[str, object]], stream: bool, terminal_event: str, status: str
) -> None:
    assistant: Final = _assistant_turn(output, stream=stream, terminal_event=terminal_event, status=status)

    sent: Final = _next_turn_input(assistant)

    assert "openai_responses_replay" not in _replayed_fields(assistant)
    assert _types(sent[1:-1]) == ["reasoning", "reasoning", "message", "function_call"]


def _chunks_before_error(stream: Iterator[object]) -> Iterator[object]:
    try:
        yield from stream
    except MidStreamFallbackError:
        return


def test_an_interrupted_stream_records_no_replay() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.post(OPENAI_RESPONSES_URL).mock(return_value=_sse(OUTPUT, terminal_event=None))
        stream: Final = litellm.completion(
            model=f"openai/responses/{MODEL}", messages=[USER], tools=list(TOOLS), api_key="sk-test", stream=True
        )
        received: Final = tuple(_chunks_before_error(iter(stream)))
    assistant: Final = litellm.stream_chunk_builder(list(received), messages=[USER]).choices[0].message.model_dump()

    assert assistant["content"] == "Checking the weather.Paris: light rain."
    assert _replayed_fields(assistant) == {}


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_a_non_openai_route_records_no_phase_or_replay(stream: bool) -> None:
    assistant: Final = _assistant_turn(OUTPUT, stream=stream, model=f"azure/responses/{MODEL}")

    assert _replayed_fields(assistant) == {}


def test_a_non_openai_route_ignores_a_recorded_replay() -> None:
    sent: Final = _next_turn_input(_assistant_turn(OUTPUT, stream=False), model=f"azure/responses/{MODEL}")

    assert _types(sent[1:-1]) == ["reasoning", "reasoning", "message", "function_call"]
    assert all("phase" not in item for item in sent)


SECOND_THOUGHT: Final = {**SECOND_REASONING, "summary": [{"type": "summary_text", "text": "Lyon is likely dry."}]}
LYON_CALL: Final = {**WEATHER_CALL, "id": "fc_lyon", "call_id": "call_weather_lyon", "arguments": '{"city":"Lyon"}'}


@pytest.mark.parametrize(
    ("model", "expected_ids", "expected_text"),
    [
        (f"openai/responses/{MODEL}", ["rs_first", "rs_second"], "Paris is likely wet. Lyon is likely dry."),
        (f"azure/responses/{MODEL}", ["rs_second"], "Lyon is likely dry."),
    ],
    ids=["openai", "azure"],
)
def test_only_the_openai_route_keeps_every_reasoning_item_of_a_tool_call_turn(
    model: str, expected_ids: list[str], expected_text: str
) -> None:
    assistant: Final = _assistant_turn(
        (FIRST_REASONING, WEATHER_CALL, SECOND_THOUGHT, LYON_CALL), stream=False, model=model
    )

    reasoning_items: Final = assistant["reasoning_items"]
    assert isinstance(reasoning_items, list)
    assert [item["id"] for item in reasoning_items] == expected_ids
    assert assistant["reasoning_content"] == expected_text


@pytest.mark.parametrize(
    ("model", "expected_content"),
    [(f"openai/responses/{MODEL}", None), (f"azure/responses/{MODEL}", "")],
    ids=["openai", "azure"],
)
def test_only_the_openai_route_ends_a_streamed_tool_call_turn_with_null_content(
    model: str, expected_content: str | None
) -> None:
    assistant: Final = _assistant_turn((FIRST_REASONING, WEATHER_CALL), stream=True, model=model)

    assert assistant["content"] == expected_content


def _resealed(message: Mapping[str, object], items: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    chat: Final = history._chat_message(message)
    assert chat is not None
    entries: Final = history._Replay.model_validate({"version": 1, "items": items, "checksum": ""}).items
    replay: Final = {"version": 1, "items": list(items), "checksum": history._checksum(chat, entries)}
    return {**message, "provider_specific_fields": _with_replay(_replayed_fields(message), replay)}


@pytest.mark.parametrize(
    "items",
    [
        (
            {"type": "message", "parts": [21]},
            {"type": "function_call", "index": 1},
            {"type": "message", "parts": [7, 11]},
        ),
        (
            {"type": "message", "parts": [5]},
            {"type": "function_call", "index": 0},
            {"type": "message", "parts": [7, 11]},
        ),
    ],
    ids=["tool-call-index", "text-lengths"],
)
def test_a_resealed_record_that_does_not_fit_the_chat_fields_falls_back(items: Sequence[Mapping[str, object]]) -> None:
    assistant: Final = _assistant_turn((COMMENTARY, WEATHER_CALL, FINAL_ANSWER), stream=False)

    sent: Final = _next_turn_input(_resealed(assistant, items))

    assert _types(sent[1:-1]) == ["message", "function_call"]
    assert sent[1]["content"] == [{"type": "output_text", "text": "Checking the weather.Paris: light rain."}]


PATCH_CALL: Final = {
    "type": "custom_tool_call",
    "id": "ctc_patch",
    "call_id": "call_patch_readme",
    "name": "apply_patch",
    "input": "*** Begin Patch\n*** End Patch",
    "status": "completed",
}


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_a_custom_tool_call_replays_with_its_custom_output(stream: bool) -> None:
    assistant: Final = _assistant_turn((COMMENTARY, PATCH_CALL), stream=stream)

    with respx.mock(assert_all_called=True) as router:
        upstream: Final = router.post(OPENAI_RESPONSES_URL).mock(
            return_value=httpx.Response(200, json=_response((FOLLOW_UP,)))
        )
        litellm.completion(
            model=f"openai/responses/{MODEL}",
            messages=[USER, assistant, {"role": "tool", "tool_call_id": "call_patch_readme", "content": "applied"}],
            tools=[{"type": "custom", "custom": {"name": "apply_patch"}}],
            api_key="sk-test",
        )
    sent: Final = json.loads(upstream.calls[0].request.content)["input"]

    assert sent[1:] == [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Checking the weather."}],
            "phase": "commentary",
        },
        {
            "type": "custom_tool_call",
            "call_id": "call_patch_readme",
            "name": "apply_patch",
            "input": PATCH_CALL["input"],
        },
        {"type": "custom_tool_call_output", "call_id": "call_patch_readme", "output": "applied"},
    ]
