import json
import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

MODEL: Final = "claude-haiku-4-5"
TEXT: Final = {"type": "text", "text": "alpha bravo"}
EXPLANATION: Final = "This request was declined under the usage policy."
PENDING_SEARCH: Final = {
    "type": "server_tool_use",
    "id": "srvtoolu_pending",
    "name": "web_search",
    "input": {"query": "next city"},
}
TERMINAL_EVENTS: Final = ("response.completed", "response.incomplete", "response.failed")


def _sse(event: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _message(identity: str, stop_reason: str | None) -> dict[str, JsonValue]:
    return {
        "id": "msg_" + identity,
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": [TEXT],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 199878, "output_tokens": 12799},
    }


def _block_events(index: int, block: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    if block["type"] == "text":
        return (
            _sse({"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}}),
            _sse(
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]},
                }
            ),
            _sse({"type": "content_block_stop", "index": index}),
        )
    return (
        _sse({"type": "content_block_start", "index": index, "content_block": {**block, "input": {}}}),
        _sse(
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
            }
        ),
        _sse({"type": "content_block_stop", "index": index}),
    )


def _turn(
    identity: str,
    stop_reason: str,
    content: tuple[Mapping[str, JsonValue], ...] = (TEXT,),
    stop_details: Mapping[str, JsonValue] | None = None,
):
    stop: Final = {"stop_reason": stop_reason, "stop_sequence": None}
    details: Final = {"stop_details": stop_details} if stop_details is not None else {}
    message: Final = {**_message(identity, stop_reason), "content": list(content), **details}
    stream: Final = (
        _sse({"type": "message_start", "message": {**_message(identity, None), "content": []}}),
        *(
            event
            for index, block in enumerate(content)
            for event in _block_events(index, block)  # comprehension-ok: block -> events flatten
        ),
        _sse({"type": "message_delta", "delta": {**stop, **details}, "usage": {"output_tokens": 12799}}),
        _sse({"type": "message_stop"}),
    )

    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        if json.loads(request.body).get("stream"):
            return Reply(content_type="text/event-stream", chunks=stream)
        return Reply(body=json.dumps(message).encode())

    return respond


def _events(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    with gateway.client.stream(
        "POST", path, json={**body, "stream": True}, headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return tuple(
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )


def _chat_choices(gateway: Gateway, body: Mapping[str, JsonValue], *, stream: bool) -> tuple[dict[str, JsonValue], ...]:
    if stream:
        return tuple(
            choice
            for chunk in _events(gateway, "/v1/chat/completions", body)
            for choice in chunk["choices"]  # comprehension-ok: chunk -> choices flatten
        )
    response: Final = gateway.request("POST", "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    return tuple(response.json()["choices"])


def _chat_body(model: str) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 64000, "messages": [{"role": "user", "content": "copy the list"}]}


def _chat_finish_reason(gateway: Gateway, model: str, *, stream: bool) -> JsonValue:
    reasons: Final = tuple(
        choice["finish_reason"]
        for choice in _chat_choices(gateway, _chat_body(model), stream=stream)
        if choice.get("finish_reason")
    )
    assert len(reasons) == 1, reasons
    return reasons[0]


def _chat_messages(gateway: Gateway, model: str, *, stream: bool) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        choice.get("delta") or choice.get("message") or {}
        for choice in _chat_choices(gateway, _chat_body(model), stream=stream)
    )


def _responses_body(model: str) -> dict[str, JsonValue]:
    return {"model": model, "max_output_tokens": 64000, "input": "copy the list", "tools": [{"type": "web_search"}]}


def _responses_result(gateway: Gateway, model: str, *, stream: bool) -> dict[str, JsonValue]:
    if stream:
        terminal: Final = tuple(
            event["response"]
            for event in _events(gateway, "/v1/responses", _responses_body(model))
            if event["type"] in TERMINAL_EVENTS
        )
        assert len(terminal) == 1, terminal
        return terminal[0]
    response: Final = gateway.request("POST", "/v1/responses", _responses_body(model))
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize(
    ("stop_reason", "finish_reason"),
    [
        ("model_context_window_exceeded", "length"),
        ("max_tokens", "length"),
        ("pause_turn", "length"),
        ("end_turn", "stop"),
    ],
)
def test_chat_maps_anthropic_stop_reason(gateway: Gateway, stop_reason: str, finish_reason: str, stream: bool) -> None:
    identity: Final = uuid.uuid4().hex
    with wire_server(_turn(identity, stop_reason)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key")
        assert _chat_finish_reason(gateway, model, stream=stream) == finish_reason


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize(
    ("stop_reason", "status"),
    [
        ("model_context_window_exceeded", "incomplete"),
        ("max_tokens", "incomplete"),
        ("pause_turn", "completed"),
        ("end_turn", "completed"),
    ],
)
def test_responses_maps_anthropic_stop_reason(gateway: Gateway, stop_reason: str, status: str, stream: bool) -> None:
    identity: Final = uuid.uuid4().hex
    with wire_server(_turn(identity, stop_reason)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key")
        assert _responses_result(gateway, model, stream=stream)["status"] == status


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_chat_keeps_pause_turn_so_the_client_knows_to_resend_the_turn(gateway: Gateway, stream: bool) -> None:
    identity: Final = uuid.uuid4().hex
    paused: Final = _turn(identity, "pause_turn", (PENDING_SEARCH,))
    with wire_server(paused) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key")
        stop_reasons: Final = tuple(
            (message.get("provider_specific_fields") or {}).get("stop_reason")
            for message in _chat_messages(gateway, model, stream=stream)
        )
        assert "pause_turn" in stop_reasons, stop_reasons


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_responses_list_a_paused_turn_unrun_search_as_an_in_progress_web_search(gateway: Gateway, stream: bool) -> None:
    identity: Final = uuid.uuid4().hex
    paused: Final = _turn(identity, "pause_turn", (PENDING_SEARCH,))
    with wire_server(paused) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key")
        result: Final = _responses_result(gateway, model, stream=stream)
        assert (result["status"], result.get("incomplete_details")) == ("completed", None)
        assert [(item["type"], item["status"], item["action"]["query"]) for item in result["output"]] == [
            ("web_search_call", "in_progress", PENDING_SEARCH["input"]["query"])
        ]


_REFUSAL: Final = {"type": "refusal", "category": "cyber", "explanation": EXPLANATION}


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_chat_returns_the_refusal_explanation_as_the_message_refusal(gateway: Gateway, stream: bool) -> None:
    identity: Final = uuid.uuid4().hex
    with wire_server(_turn(identity, "refusal", (), _REFUSAL)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key")
        refusal: Final = "".join(
            message["refusal"] for message in _chat_messages(gateway, model, stream=stream) if message.get("refusal")
        )
        assert refusal == EXPLANATION


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize(
    ("content", "expected_parts"),
    [
        ((), [("refusal", EXPLANATION)]),
        ((TEXT,), [("output_text", TEXT["text"]), ("refusal", EXPLANATION)]),
    ],
    ids=["refusal_only", "answer_then_refusal"],
)
def test_responses_return_the_refusal_explanation_as_a_refusal_part(
    gateway: Gateway,
    content: tuple[Mapping[str, JsonValue], ...],
    expected_parts: list[tuple[str, str]],
    stream: bool,
) -> None:
    identity: Final = uuid.uuid4().hex
    with wire_server(_turn(identity, "refusal", content, _REFUSAL)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{MODEL}", api_base=wire.url, api_key="scripted-anthropic-key")
        result: Final = _responses_result(gateway, model, stream=stream)
        assert (result["status"], result["incomplete_details"]) == ("incomplete", {"reason": "content_filter"})
        assert [
            [(part["type"], part.get("text") or part.get("refusal")) for part in item["content"]]
            for item in result["output"]
        ] == [expected_parts]
