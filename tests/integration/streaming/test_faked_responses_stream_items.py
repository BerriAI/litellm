import json
import uuid
from typing import Final

import httpx
import openai
import pytest
from integration._support.client import JSON_OBJECT, Gateway, list_value, object_value, string_value
from integration._support.wire import Reply, Request, wire_server
from integration.streaming.faked_responses_case import (
    ANSWER,
    ARGUMENTS,
    NO_CACHE,
    RESPONSES_TOOL_PARAMS,
    SECOND_ARGUMENTS,
    SQL,
    SUMMARIES,
    answering,
    assert_each_value_arrives_once,
    assert_one_unstreamed_upstream_call,
    completed_output,
    custom_tool_call_item,
    data_events,
    every_item_kind,
    faked_model,
    function_call_item,
    items_by_index,
    joined,
    message_item,
    reasoning_item,
    responses_request,
    spend_statuses,
    stream_events,
    values,
)
from pydantic import JsonValue

_TARGET: Final = "/v1/responses"
_LARGE: Final = 5000


def _openai(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def _async_openai(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=str(gateway.client.base_url) + "/v1", api_key=gateway.key, max_retries=0)


def test_raw_stream_opens_every_item_empty_and_each_value_arrives_once(gateway: Gateway) -> None:
    with wire_server(answering(every_item_kind)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        call_id, events = stream_events(gateway, _TARGET, responses_request(model, cache=NO_CACHE))
        assert_one_unstreamed_upstream_call(wire, _TARGET)
        added: Final = items_by_index(events, "response.output_item.added")
        assert [added[index]["type"] for index in sorted(added)] == [
            "reasoning",
            "message",
            "function_call",
            "custom_tool_call",
            "message",
        ], added
        assert added[0]["summary"] == [], added[0]
        assert (added[1]["content"], added[1]["status"]) == ([], "in_progress"), added[1]
        assert (added[2]["arguments"], added[2]["status"]) == ("", "in_progress"), added[2]
        assert added[3]["input"] == "", added[3]
        assert (added[4]["content"], added[4]["status"]) == ([], "in_progress"), added[4]
        summary_parts: Final = [
            (event["summary_index"], object_value(event["part"])["text"])
            for event in events
            if event["type"] == "response.reasoning_summary_part.added"
        ]
        assert summary_parts == [(0, ""), (1, "")], summary_parts
        assert [
            joined(events, "response.reasoning_summary_text.delta", 0, summary_index=index) for index in (0, 1)
        ] == [*SUMMARIES]
        assert joined(events, "response.output_text.delta", 1) == ANSWER
        assert joined(events, "response.function_call_arguments.delta", 2) == ARGUMENTS
        assert joined(events, "response.custom_tool_call_input.delta", 3) == SQL
        custom_done: Final = [event for event in events if event["type"] == "response.custom_tool_call_input.done"]
        assert [(event["output_index"], event["input"]) for event in custom_done] == [(3, SQL)], custom_done
        identity: Final = string_value(object_value(added[1])["id"]).removeprefix("msg_")
        assert_each_value_arrives_once(events, every_item_kind(identity))
        assert spend_statuses(call_id) == ["success"]


def test_openai_sdk_stream_snapshots_hold_each_value_once(gateway: Gateway) -> None:
    with wire_server(answering(lambda identity: (message_item(identity), function_call_item(identity)))) as wire:
        with gateway.scenario() as scenario:
            model: Final = faked_model(scenario, wire)
            with _openai(gateway).responses.stream(
                model=model,
                input=f"weather {uuid.uuid4().hex}",
                tools=RESPONSES_TOOL_PARAMS,
                extra_body={"cache": NO_CACHE},
            ) as stream:
                events: Final = list(stream)
                final: Final = stream.get_final_response()
            text_snapshots: Final = [event.snapshot for event in events if event.type == "response.output_text.delta"]
            argument_snapshots: Final = [
                event.snapshot for event in events if event.type == "response.function_call_arguments.delta"
            ]
            assert text_snapshots and text_snapshots[-1] == ANSWER, text_snapshots[-1:]
            assert argument_snapshots and argument_snapshots[-1] == ARGUMENTS, argument_snapshots[-1:]
            assert final.output_text == ANSWER
            calls: Final = [item for item in final.output if item.type == "function_call"]
            assert [call.arguments for call in calls] == [ARGUMENTS], final.output
            assert_one_unstreamed_upstream_call(wire, _TARGET)


async def test_async_openai_sdk_stream_snapshots_keep_two_calls_and_a_message_apart(gateway: Gateway) -> None:
    def output(identity: str) -> tuple[dict[str, JsonValue], ...]:
        return (
            function_call_item(identity),
            function_call_item(identity, SECOND_ARGUMENTS, "get_forecast"),
            message_item(identity),
        )

    with wire_server(answering(output)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        async with _async_openai(gateway).responses.stream(
            model=model,
            input=f"weather {uuid.uuid4().hex}",
            tools=RESPONSES_TOOL_PARAMS,
            extra_body={"cache": NO_CACHE},
        ) as stream:
            events: Final = [event async for event in stream]
            final: Final = await stream.get_final_response()
        last_arguments: Final = {
            event.output_index: event.snapshot
            for event in events
            if event.type == "response.function_call_arguments.delta"
        }
        assert last_arguments == {0: ARGUMENTS, 1: SECOND_ARGUMENTS}, last_arguments
        last_text: Final = [event.snapshot for event in events if event.type == "response.output_text.delta"]
        assert last_text and last_text[-1] == ANSWER, last_text[-1:]
        assert final.output_text == ANSWER
        assert_one_unstreamed_upstream_call(wire, _TARGET)


def test_non_stream_answer_carries_the_upstream_output_unchanged(gateway: Gateway) -> None:
    with wire_server(answering(every_item_kind)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        response: Final = gateway.request("POST", _TARGET, responses_request(model, cache=NO_CACHE))
        assert response.status_code == 200, response.text
        output: Final = tuple(map(object_value, list_value(JSON_OBJECT.validate_json(response.content)["output"])))
        identity: Final = string_value(output[1]["id"]).removeprefix("msg_")
        assert [(item["type"], item["id"], values(item)) for item in output] == [
            (item["type"], item["id"], values(item)) for item in every_item_kind(identity)
        ], output
        assert_one_unstreamed_upstream_call(wire, _TARGET)


def test_an_item_type_the_fake_stream_does_not_split_opens_as_its_done_value(gateway: Gateway) -> None:
    def output(identity: str) -> tuple[dict[str, JsonValue], ...]:
        search: Final[dict[str, JsonValue]] = {
            "type": "web_search_call",
            "id": f"ws_{identity}",
            "status": "completed",
            "action": {"type": "search", "query": "paris weather"},
        }
        return (search,)

    with wire_server(answering(output)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        _, events = stream_events(gateway, _TARGET, responses_request(model, cache=NO_CACHE))
        added: Final = items_by_index(events, "response.output_item.added")
        done: Final = items_by_index(events, "response.output_item.done")
        assert list(added) == list(done) == [0], (added, done)
        assert added[0] == done[0], (added[0], done[0])
        action: Final = object_value(done[0]["action"])
        assert (action["type"], action["query"]) == ("search", "paris weather"), action
        assert [item["id"] for item in completed_output(events)] == [done[0]["id"]]


@pytest.mark.parametrize("status", (400, 401, 404))
def test_upstream_error_reaches_the_caller_and_the_next_stream_is_served(gateway: Gateway, status: int) -> None:
    message: Final = f"scripted upstream failure {uuid.uuid4().hex}"
    failure: Final = Reply(
        status=status,
        body=json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": None}}).encode(),
    )
    healthy: Final = answering(lambda identity: (message_item(identity),))

    def respond(request: Request) -> Reply:
        return failure if b"fail-now" in request.body else healthy(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        with gateway.client.stream(
            "POST",
            _TARGET,
            json={**responses_request(model, cache=NO_CACHE), "input": "fail-now", "stream": True},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as failed:
            failed.read()
        assert failed.status_code == status, failed.text
        assert message in failed.text, failed.text
        _, events = stream_events(gateway, _TARGET, responses_request(model, cache=NO_CACHE))
        identity: Final = string_value(items_by_index(events, "response.output_item.done")[0]["id"]).removeprefix(
            "msg_"
        )
        assert_each_value_arrives_once(events, (message_item(identity),))
        assert [request.target for request in wire.drain() if request.method == "POST"] == [_TARGET, _TARGET]


def test_empty_values_open_empty_and_close_empty(gateway: Gateway) -> None:
    def output(identity: str) -> tuple[dict[str, JsonValue], ...]:
        return (
            reasoning_item(identity, ()),
            message_item(identity, ""),
            function_call_item(identity, ""),
            custom_tool_call_item(identity, ""),
        )

    with wire_server(answering(output)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        _, events = stream_events(gateway, _TARGET, responses_request(model, cache=NO_CACHE))
        identity: Final = string_value(items_by_index(events, "response.output_item.done")[1]["id"]).removeprefix(
            "msg_"
        )
        assert_each_value_arrives_once(events, output(identity))
        assert joined(events, "response.function_call_arguments.delta", 2) == ""
        assert_one_unstreamed_upstream_call(wire, _TARGET)


def test_five_kilobyte_arguments_and_text_arrive_once(gateway: Gateway) -> None:
    arguments: Final = json.dumps({"notes": ("détail " * _LARGE)[: _LARGE - 13]}, ensure_ascii=False)
    text: Final = ("Long answer, part " * _LARGE)[:_LARGE]

    def output(identity: str) -> tuple[dict[str, JsonValue], ...]:
        return (message_item(identity, text), function_call_item(identity, arguments))

    with wire_server(answering(output)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        call_id, events = stream_events(gateway, _TARGET, responses_request(model, cache=NO_CACHE))
        assert len(arguments) == _LARGE and len(text) == _LARGE
        assert joined(events, "response.output_text.delta", 0) == text
        assert joined(events, "response.function_call_arguments.delta", 1) == arguments
        identity: Final = string_value(items_by_index(events, "response.output_item.done")[0]["id"]).removeprefix(
            "msg_"
        )
        assert_each_value_arrives_once(events, output(identity))
        assert_one_unstreamed_upstream_call(wire, _TARGET)
        assert spend_statuses(call_id) == ["success"]


def test_arguments_sent_as_an_object_never_break_the_stream_or_the_next_request(gateway: Gateway) -> None:
    malformed: Final[dict[str, JsonValue]] = {"city": "Paris"}

    def output(identity: str) -> tuple[dict[str, JsonValue], ...]:
        return (function_call_item(identity, malformed),)

    with wire_server(answering(output)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        with gateway.client.stream(
            "POST",
            _TARGET,
            json={**responses_request(model, cache=NO_CACHE), "stream": True},
            headers={"Authorization": f"Bearer {gateway.key}"},
        ) as response:
            body: Final = response.read().decode()
        assert response.status_code in (200, 400, 500), body
        if response.status_code == 200:
            events: Final = data_events(iter(body.splitlines()))
            assert items_by_index(events, "response.output_item.done")[0]["arguments"] == malformed, body
        healthy: Final = gateway.request("POST", _TARGET, responses_request(model, cache=NO_CACHE))
        assert healthy.status_code == 200, healthy.text
        assert healthy.json()["output"][0]["arguments"] == malformed, healthy.text
        assert len([request for request in wire.drain() if request.method == "POST"]) == 2


def test_identical_uncached_streams_each_reach_the_upstream_and_log_once(gateway: Gateway) -> None:
    with wire_server(answering(lambda identity: (message_item(identity),))) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        body: Final = responses_request(model, cache=NO_CACHE)
        first_call, first = stream_events(gateway, _TARGET, body)
        second_call, second = stream_events(gateway, _TARGET, body)
        assert first_call != second_call
        first_id: Final = string_value(items_by_index(first, "response.output_item.done")[0]["id"])
        second_id: Final = string_value(items_by_index(second, "response.output_item.done")[0]["id"])
        assert first_id != second_id
        assert_each_value_arrives_once(first, (message_item(first_id.removeprefix("msg_")),))
        assert_each_value_arrives_once(second, (message_item(second_id.removeprefix("msg_")),))
        assert [request.target for request in wire.drain() if request.method == "POST"] == [_TARGET, _TARGET]
        assert spend_statuses(first_call) == ["success"]
        assert spend_statuses(second_call) == ["success"]


def test_client_that_drops_the_stream_after_the_first_event_leaves_the_next_stream_whole(gateway: Gateway) -> None:
    with wire_server(answering(every_item_kind)) as wire, gateway.scenario() as scenario:
        model: Final = faked_model(scenario, wire)
        with httpx.Client(base_url=str(gateway.client.base_url), timeout=60, trust_env=False) as client:
            with client.stream(
                "POST",
                _TARGET,
                json={**responses_request(model, cache=NO_CACHE), "stream": True},
                headers={"Authorization": f"Bearer {gateway.key}"},
            ) as dropped:
                assert dropped.status_code == 200
                first: Final = next(line for line in dropped.iter_lines() if line.startswith("data: {"))
                assert json.loads(first.removeprefix("data: "))["type"] == "response.created", first
        _, events = stream_events(gateway, _TARGET, responses_request(model, cache=NO_CACHE))
        identity: Final = string_value(items_by_index(events, "response.output_item.done")[1]["id"]).removeprefix(
            "msg_"
        )
        assert_each_value_arrives_once(events, every_item_kind(identity))
        assert [request.target for request in wire.drain() if request.method == "POST"] == [_TARGET, _TARGET]
        assert gateway.client.get("/health/liveliness").status_code == 200
