import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Final, cast

import httpx
import openai
import pytest
from integration._support import bedrock_runtime_peer
from integration._support import prompt_cache_breakpoint as pcb
from integration._support.client import Gateway, Scenario, object_value
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import (
    ANTHROPIC_MODEL,
    ASK,
    ASK_LABEL,
    BEDROCK_MODEL,
    CITIES,
    EPHEMERAL,
    POINTS,
    PROVIDER_KEY,
    SYSTEM,
    SYSTEM_LABEL,
    anthropic_labels,
    anthropic_marks,
    anthropic_peer,
    bedrock_labels,
    bedrock_peer,
    call_id,
    client_marked,
    final_label,
    final_text,
    new_marker,
    responses_body,
)
from openai.types.responses import ResponseInputParam
from pydantic import JsonValue, TypeAdapter

from litellm.utils import get_prompt_cache_min_tokens

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_HOUR: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
_CHECK: Final = "I will check."
_ASSISTANT_LABEL: Final = f"assistant:text:{_CHECK}"
_RESULT_LABEL: Final = f"user:tool_result:{call_id(CITIES[0])}"
_BEDROCK_SYSTEM_LABEL: Final = f"system:text:{SYSTEM}"
_BEDROCK_RESULT_LABEL: Final = f"user:toolResult:{call_id(CITIES[0])}"
_TOOL: Final[dict[str, JsonValue]] = {"type": "function", "name": "lookup_weather", "parameters": {"type": "object"}}
_GEMINI: Final = "gemini/gemini-3.8-flash"
_GEMINI_CACHE_WRITE: Final = [
    ("GET", "/models/gemini-3.8-flash:cachedContents"),
    ("POST", "/models/gemini-3.8-flash:cachedContents"),
    ("POST", "/models/gemini-3.8-flash:generateContent"),
]
_GEMINI_REPLY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "sunny"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1600, "candidatesTokenCount": 1, "totalTokenCount": 1601},
    }
).encode()
_OPENAI_COMPATIBLE: Final = "openai/gpt-5.4"


@dataclass(frozen=True, slots=True)
class _Answer:
    status: int
    text: str
    call_id: str


@pytest.fixture(scope="module")
def spend() -> Iterator[pcb.SpendLogs]:
    with pcb.spend_logs() as logs:
        yield logs


def _item(role: str, text: str, mark: JsonValue = EPHEMERAL) -> dict[str, JsonValue]:
    return {"role": role, "content": text, "cache_control": mark}


def _responses(model: str, items: Sequence[JsonValue], **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_output_tokens": 64, "input": list(items), **fields}


def _marked_system(model: str, marker: str, mark: JsonValue = EPHEMERAL, **fields: JsonValue) -> dict[str, JsonValue]:
    return _responses(model, [_item("system", SYSTEM, mark), {"role": "user", "content": final_text(marker)}], **fields)


def _role_items(role: str, marker: str) -> list[JsonValue]:
    match role:
        case "system":
            return [_item("system", SYSTEM), {"role": "user", "content": final_text(marker)}]
        case "user":
            return [_item("user", ASK), {"role": "user", "content": final_text(marker)}]
        case _:
            return [
                {"role": "user", "content": ASK},
                _item("assistant", _CHECK),
                {"role": "user", "content": final_text(marker)},
            ]


def _tool_items(marker: str, *, output_mark: JsonValue = None, call_mark: JsonValue = None) -> list[JsonValue]:
    call: Final[dict[str, JsonValue]] = {
        "type": "function_call",
        "call_id": call_id(CITIES[0]),
        "name": "lookup_weather",
        "arguments": json.dumps({"city": CITIES[0]}),
    }
    output: Final[dict[str, JsonValue]] = {
        "type": "function_call_output",
        "call_id": call_id(CITIES[0]),
        "output": "sunny.",
    }
    return [
        {"role": "user", "content": ASK},
        {**call, "cache_control": call_mark} if call_mark is not None else call,
        {**output, "cache_control": output_mark} if output_mark is not None else output,
        {"role": "user", "content": final_text(marker)},
    ]


def _call_id_of(response: httpx.Response) -> str:
    return response.headers["x-litellm-call-id"] if "x-litellm-call-id" in response.headers else ""


def _send(gateway: Gateway, body: dict[str, JsonValue], *, key: str | None = None) -> _Answer:
    headers: Final = {"Authorization": f"Bearer {gateway.key if key is None else key}"}
    with gateway.client.stream("POST", "/v1/responses", json=body, headers=headers) as response:
        return _Answer(response.status_code, response.read().decode(), _call_id_of(response))


def _send_raw(gateway: Gateway, raw: bytes) -> _Answer:
    headers: Final = {"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"}
    response: Final = gateway.client.post("/v1/responses", content=raw, headers=headers)
    return _Answer(response.status_code, response.text, _call_id_of(response))


def _assert_completed(answer: _Answer, *, stream: bool = False) -> None:
    assert answer.status == 200, answer.text
    assert "sunny" in answer.text, answer.text
    if stream:
        assert '"type":"response.completed"' in answer.text, answer.text
        return
    assert _JSON_OBJECT.validate_json(answer.text)["status"] == "completed", answer.text


def _only_request(wire: Wire) -> Request:
    received: Final = wire.drain()
    assert len(received) == 1, [request.target for request in received]
    return received[0]


def _anthropic_deployment(scenario: Scenario, wire: Wire, *, inject: bool = False) -> str:
    if inject:
        return scenario.model(
            model=f"anthropic/{ANTHROPIC_MODEL}",
            api_base=wire.url,
            api_key=PROVIDER_KEY,
            cache_control_injection_points=POINTS,
        )
    return scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=PROVIDER_KEY)


def _bedrock_peer(request: Request) -> Reply:
    if request.target.endswith("/converse-stream"):
        return bedrock_runtime_peer.respond(request)
    return bedrock_peer(request)


def _bedrock_deployment(scenario: Scenario, wire: Wire, route: str = "converse") -> str:
    return scenario.model(
        model=f"bedrock/{route}/{BEDROCK_MODEL}",
        api_key=PROVIDER_KEY,
        api_base=wire.url,
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=wire.url,
    )


def _blocks_of(body: dict[str, JsonValue]) -> Iterator[JsonValue]:
    system: Final = body.get("system")
    yield from system if isinstance(system, list) else []
    messages: Final = body.get("messages")
    for message in messages if isinstance(messages, list) else []:
        yield from _content_blocks(object_value(message).get("content"))


def _content_blocks(content: JsonValue) -> Iterator[JsonValue]:
    yield from content if isinstance(content, list) else []


def _bedrock_ttls(request: Request) -> list[JsonValue]:
    blocks: Final = list(_blocks_of(_JSON_OBJECT.validate_json(request.body)))
    return [
        object_value(block["cachePoint"]).get("ttl")
        for block in blocks
        if isinstance(block, dict) and "cachePoint" in block
    ]


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
@pytest.mark.parametrize(
    ("role", "expected"),
    (
        pytest.param("system", SYSTEM_LABEL, id="system"),
        pytest.param("user", ASK_LABEL, id="user"),
        pytest.param("assistant", _ASSISTANT_LABEL, id="assistant"),
    ),
)
def test_marked_item_reaches_anthropic_as_a_cache_point(
    gateway: Gateway, spend: pcb.SpendLogs, role: str, expected: str, stream: bool
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        answer: Final = _send(gateway, _responses(model, _role_items(role, marker), stream=stream))
        _assert_completed(answer, stream=stream)
        assert anthropic_labels(_only_request(wire)) == [expected]
        spend.landed(model, answer.call_id, None)


def test_list_content_message_mark_lands_on_the_last_block(gateway: Gateway) -> None:
    marker: Final = new_marker()
    content: Final[list[JsonValue]] = [{"type": "input_text", "text": "first"}, {"type": "input_text", "text": ASK}]
    items: Final[list[JsonValue]] = [
        {"role": "user", "content": content, "cache_control": EPHEMERAL},
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, _responses(model, items)))
        assert anthropic_labels(_only_request(wire)) == [ASK_LABEL]


def test_block_mark_outranks_the_message_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    items: Final[list[JsonValue]] = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": ASK, "cache_control": _HOUR}],
            "cache_control": EPHEMERAL,
        },
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, _responses(model, items)))
        marks: Final = anthropic_marks(_only_request(wire))
    assert [(mark.label, mark.ttl) for mark in marks] == [(ASK_LABEL, "1h")]


def test_function_call_output_mark_reaches_anthropic_as_a_tool_result_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        body: Final = _responses(model, _tool_items(marker, output_mark=EPHEMERAL), tools=[_TOOL])
        _assert_completed(_send(gateway, body))
        assert anthropic_labels(_only_request(wire)) == [_RESULT_LABEL]


def test_function_call_marks_reach_anthropic_as_tool_use_marks(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, responses_body(model, marker)))
        assert anthropic_labels(_only_request(wire)) == client_marked()


@pytest.mark.parametrize("stream", (False, True), ids=("json", "stream"))
@pytest.mark.parametrize(
    ("role", "expected"),
    (
        pytest.param("system", _BEDROCK_SYSTEM_LABEL, id="system"),
        pytest.param("user", f"user:text:{ASK}", id="user"),
        pytest.param("assistant", f"assistant:text:{_CHECK}", id="assistant"),
    ),
)
def test_marked_item_reaches_bedrock_converse_as_a_cache_point(
    gateway: Gateway, role: str, expected: str, stream: bool
) -> None:
    marker: Final = new_marker()
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        answer: Final = _send(gateway, _responses(model, _role_items(role, marker), stream=stream))
        assert answer.status == 200, answer.text
        assert f"marker-{marker}" in answer.text or "sunny" in answer.text, answer.text
        request: Final = _only_request(wire)
    assert request.target.endswith("/converse-stream" if stream else "/converse"), request.target
    assert bedrock_labels(request) == [expected]


def test_function_call_output_mark_reaches_bedrock_as_a_tool_result_cache_point(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        body: Final = _responses(model, _tool_items(marker, output_mark=EPHEMERAL), tools=[_TOOL])
        answer: Final = _send(gateway, body)
        assert answer.status == 200, answer.text
        assert bedrock_labels(_only_request(wire)) == [_BEDROCK_RESULT_LABEL]


def test_one_hour_mark_keeps_its_ttl_on_anthropic(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, _marked_system(model, marker, _HOUR)))
        marks: Final = anthropic_marks(_only_request(wire))
    assert [(mark.label, mark.ttl) for mark in marks] == [(SYSTEM_LABEL, "1h")]


def test_one_hour_mark_keeps_its_ttl_on_bedrock(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        answer: Final = _send(gateway, _marked_system(model, marker, _HOUR))
        assert answer.status == 200, answer.text
        request: Final = _only_request(wire)
    assert bedrock_labels(request) == [_BEDROCK_SYSTEM_LABEL]
    assert _bedrock_ttls(request) == ["1h"]


def _openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def _async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def _sdk_input(marker: str) -> ResponseInputParam:
    return cast(ResponseInputParam, _role_items("system", marker))


def test_openai_sdk_sync_responses_carries_the_system_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        with _openai_client(gateway) as client:
            response: Final = client.responses.create(model=model, input=_sdk_input(marker), max_output_tokens=64)
        assert response.status == "completed", response.model_dump()
        assert response.output_text == "sunny", response.model_dump()
        assert anthropic_labels(_only_request(wire)) == [SYSTEM_LABEL]


async def test_openai_sdk_async_stream_responses_carries_the_system_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        async with _async_openai_client(gateway) as client:
            stream: Final = await client.responses.create(
                model=model, input=_sdk_input(marker), max_output_tokens=64, stream=True
            )
            events: Final = [event async for event in stream]
        assert [event.type for event in events][-1] == "response.completed", events
        assert anthropic_labels(_only_request(wire)) == [SYSTEM_LABEL]


def _gemini_peer(request: Request) -> Reply:
    if "cachedContents" in request.target and request.method == "GET":
        return Reply(body=b'{"cachedContents":[]}')
    if "cachedContents" in request.target:
        return Reply(
            body=json.dumps(
                {
                    "name": "cachedContents/bridge",
                    "model": "models/gemini-3.8-flash",
                    "expireTime": "2099-01-01T00:00:00Z",
                }
            ).encode()
        )
    return Reply(body=_GEMINI_REPLY)


def test_gemini_marked_system_item_writes_a_context_cache(gateway: Gateway) -> None:
    marker: Final = new_marker()
    long_system: Final = "lorem " * (2 * get_prompt_cache_min_tokens(_GEMINI))
    items: Final[list[JsonValue]] = [_item("system", long_system), {"role": "user", "content": final_text(marker)}]
    with wire_server(_gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI, api_base=wire.url, api_key=PROVIDER_KEY)
        _assert_completed(_send(gateway, _responses(model, items)))
        targets: Final = [(request.method, request.target.split("?")[0]) for request in wire.drain()]
    assert targets == _GEMINI_CACHE_WRITE


def test_azure_ai_claude_marked_system_item_reaches_the_anthropic_route(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"azure_ai/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=PROVIDER_KEY)
        _assert_completed(_send(gateway, _marked_system(model, marker)))
        request: Final = _only_request(wire)
    assert request.target == "/anthropic/v1/messages", request.target
    assert anthropic_labels(request) == [SYSTEM_LABEL]


def test_bedrock_invoke_marked_system_item_reaches_the_invoke_body(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire, route="invoke")
        _assert_completed(_send(gateway, _marked_system(model, marker)))
        request: Final = _only_request(wire)
    assert request.target.endswith("/invoke"), request.target
    assert anthropic_labels(request) == [SYSTEM_LABEL]


def _openai_message_marks(request: Request) -> list[str]:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    messages: Final = body["messages"] if isinstance(body["messages"], list) else []
    return [
        f"cache_control@{object_value(message).get('role')}:message"
        for message in messages
        if "cache_control" in object_value(message)
    ]


def test_openai_compatible_chat_bridge_keeps_the_message_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    reply: Final = chat_reply(f"chatcmpl-{marker}", "gpt-5.4", "sunny", stream=False)
    with wire_server(answering_model_discovery(lambda request: reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_OPENAI_COMPATIBLE, api_base=f"{wire.url}/v1", api_key=PROVIDER_KEY, use_chat_completions_api=True
        )
        _assert_completed(_send(gateway, _marked_system(model, marker)))
        posted: Final = [request for request in wire.drain() if request.method == "POST"]
    assert [request.target for request in posted] == ["/v1/chat/completions"], [request.target for request in posted]
    assert _openai_message_marks(posted[0]) == ["cache_control@system:message"]


def test_injection_points_keep_the_caller_one_hour_system_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, inject=True)
        _assert_completed(_send(gateway, _marked_system(model, marker, _HOUR)))
        marks: Final = anthropic_marks(_only_request(wire))
    assert [(mark.label, mark.ttl) for mark in marks] == [(SYSTEM_LABEL, "1h"), (final_label(marker), None)]


@pytest.mark.parametrize(
    "mark",
    (
        pytest.param({}, id="empty-object"),
        pytest.param(None, id="null"),
        pytest.param("", id="empty-string"),
        pytest.param(0, id="zero"),
        pytest.param(False, id="false"),
    ),
)
def test_falsy_marks_are_dropped(gateway: Gateway, mark: JsonValue) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, _marked_system(model, marker, mark)))
        assert anthropic_labels(_only_request(wire)) == []


def test_missing_mark_is_dropped(gateway: Gateway) -> None:
    marker: Final = new_marker()
    items: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, _responses(model, items)))
        assert anthropic_labels(_only_request(wire)) == []


def test_response_cache_serves_the_marked_request_once(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        body: Final = _marked_system(model, marker)
        answers: Final = tuple(_send(gateway, body) for _ in range(2))
        received: Final = wire.drain()
    for answer in answers:
        _assert_completed(answer)
    assert [anthropic_labels(request) for request in received] == [[SYSTEM_LABEL]]


def _chat_with_message_mark(model: str, marker: str, mark: JsonValue = EPHEMERAL) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": SYSTEM, "cache_control": mark},
            {"role": "user", "content": final_text(marker)},
        ],
    }


def _messages_with_system_mark(model: str, marker: str) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "system": [{"type": "text", "text": SYSTEM, "cache_control": EPHEMERAL}],
        "messages": [{"role": "user", "content": final_text(marker)}],
    }


@pytest.mark.parametrize("path", ("/v1/chat/completions", "/v1/messages"))
def test_message_marks_on_the_other_endpoints_are_unchanged(gateway: Gateway, path: str) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        body: Final = (
            _chat_with_message_mark(model, marker)
            if path == "/v1/chat/completions"
            else _messages_with_system_mark(model, marker)
        )
        response: Final = gateway.request("POST", path, body)
        assert response.status_code == 200, response.text
        assert anthropic_labels(_only_request(wire)) == [SYSTEM_LABEL]


def test_block_mark_on_responses_input_is_unchanged(gateway: Gateway) -> None:
    marker: Final = new_marker()
    items: Final[list[JsonValue]] = [
        {"role": "user", "content": [{"type": "input_text", "text": ASK, "cache_control": EPHEMERAL}]},
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, _responses(model, items, instructions=SYSTEM)))
        assert anthropic_labels(_only_request(wire)) == [ASK_LABEL]


def _non_object_marks(request: Request) -> list[JsonValue]:
    blocks: Final = list(_blocks_of(_JSON_OBJECT.validate_json(request.body)))
    return [
        block["cache_control"]
        for block in blocks
        if isinstance(block, dict) and "cache_control" in block and not isinstance(block["cache_control"], dict)
    ]


def _strict_anthropic_peer(request: Request) -> Reply:
    odd: Final = _non_object_marks(request)
    if odd:
        return Reply(
            status=400,
            body=json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": f"cache_control: Input should be a valid dictionary, got {json.dumps(odd[0])[:64]}",
                    },
                }
            ).encode(),
        )
    return anthropic_peer(request)


@pytest.mark.parametrize(
    "mark",
    (
        pytest.param("ephemeral", id="string"),
        pytest.param(1, id="int"),
        pytest.param(["ephemeral"], id="list"),
        pytest.param("x" * 5120, id="5kb-string"),
    ),
)
def test_non_object_marks_reach_anthropic_as_they_do_on_chat_completions(gateway: Gateway, mark: JsonValue) -> None:
    marker: Final = new_marker()
    with wire_server(_strict_anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        on_chat: Final = gateway.request("POST", "/v1/chat/completions", _chat_with_message_mark(model, marker, mark))
        on_responses: Final = _send(gateway, _marked_system(model, marker, mark))
        received: Final = wire.drain()
    assert (on_chat.status_code, on_responses.status) == (400, 400), (on_chat.text, on_responses.text)
    assert [_non_object_marks(request) for request in received] == [[mark], [mark]]


def test_string_mark_reaches_bedrock_as_it_does_on_chat_completions(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = _bedrock_deployment(scenario, wire)
        on_chat: Final = gateway.request(
            "POST", "/v1/chat/completions", _chat_with_message_mark(model, marker, "ephemeral")
        )
        assert on_chat.status_code == 200, on_chat.text
        on_responses: Final = _send(gateway, _marked_system(model, marker, "ephemeral"))
        assert on_responses.status == 200, on_responses.text
        received: Final = wire.drain()
    assert len(received) == 2, [request.target for request in received]
    assert bedrock_labels(received[1]) == bedrock_labels(received[0])


def test_unknown_mark_type_is_forwarded_like_a_block_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, _marked_system(model, marker, {"type": "persistent"})))
        assert anthropic_labels(_only_request(wire)) == [SYSTEM_LABEL]


def test_duplicate_mark_keys_keep_the_last_one(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        raw: Final = (
            f'{{"model":"{model}","max_output_tokens":64,"input":['
            f'{{"role":"system","content":{json.dumps(SYSTEM)},"cache_control":{{"type":"ephemeral"}},'
            f'"cache_control":{{"type":"ephemeral","ttl":"1h"}}}},'
            f'{{"role":"user","content":{json.dumps(final_text(marker))}}}]}}'
        ).encode()
        _assert_completed(_send_raw(gateway, raw))
        marks: Final = anthropic_marks(_only_request(wire))
    assert [(mark.label, mark.ttl) for mark in marks] == [(SYSTEM_LABEL, "1h")]


def _many_marks(marker: str, *, assistant_marked: bool) -> list[JsonValue]:
    assistant: Final[dict[str, JsonValue]] = {"role": "assistant", "content": _CHECK}
    return [
        _item("system", SYSTEM),
        _item("user", ASK),
        {**assistant, "cache_control": EPHEMERAL} if assistant_marked else assistant,
        {
            "type": "function_call",
            "call_id": call_id(CITIES[0]),
            "name": "lookup_weather",
            "arguments": json.dumps({"city": CITIES[0]}),
        },
        {
            "type": "function_call_output",
            "call_id": call_id(CITIES[0]),
            "output": "sunny.",
            "cache_control": EPHEMERAL,
        },
        _item("user", final_text(marker)),
    ]


def test_four_marks_reach_anthropic(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        body: Final = _responses(model, _many_marks(marker, assistant_marked=False), tools=[_TOOL])
        _assert_completed(_send(gateway, body))
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == [SYSTEM_LABEL, ASK_LABEL, _RESULT_LABEL, final_label(marker)]


def test_five_marks_are_refused_by_the_vendor(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        body: Final = _responses(model, _many_marks(marker, assistant_marked=True), tools=[_TOOL])
        answer: Final = _send(gateway, body)
        labels: Final = anthropic_labels(_only_request(wire))
    assert answer.status == 400, answer.text
    assert "A maximum of 4 blocks with cache_control" in answer.text, answer.text
    assert len(labels) == 5, labels


def test_ttl_order_violation_is_the_vendors_verdict(gateway: Gateway) -> None:
    marker: Final = new_marker()
    items: Final[list[JsonValue]] = [
        _item("system", SYSTEM),
        _item("user", ASK, _HOUR),
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        answer: Final = _send(gateway, _responses(model, items))
        marks: Final = anthropic_marks(_only_request(wire))
    assert answer.status == 400, answer.text
    assert "ttl='1h' cache_control block must not come after" in answer.text, answer.text
    assert [(mark.label, mark.ttl) for mark in marks] == [(SYSTEM_LABEL, None), (ASK_LABEL, "1h")]


def test_unauthenticated_marked_request_never_reaches_the_provider(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        answer: Final = _send(gateway, _marked_system(model, marker), key=f"sk-not-a-key-{marker}")
        received: Final = wire.drain()
    assert answer.status == 401, answer.text
    assert "error" in _JSON_OBJECT.validate_json(answer.text), answer.text
    assert received == ()


def _overloaded_peer(request: Request) -> Reply:
    return Reply(
        status=503,
        body=json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}).encode(),
    )


def test_provider_error_on_a_marked_request_reaches_the_caller(gateway: Gateway, spend: pcb.SpendLogs) -> None:
    marker: Final = new_marker()
    with wire_server(_overloaded_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        answer: Final = _send(gateway, _marked_system(model, marker))
        labels: Final = anthropic_labels(_only_request(wire))
        assert answer.status == 500, answer.text
        assert "overloaded_error" in answer.text, answer.text
        assert labels == [SYSTEM_LABEL]
        spend.landed(model, answer.call_id, None, status="failure")
