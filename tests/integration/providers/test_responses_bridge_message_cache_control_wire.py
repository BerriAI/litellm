import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Final, cast

import httpx
import openai
import pytest
from integration._support import bedrock_runtime_peer
from integration._support import prompt_cache_breakpoint as pcb
from integration._support.client import Gateway, Scenario, object_value
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.vertex import service_account_json
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
    marker_of,
    new_marker,
    responses_body,
    tool_use_label,
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
_GEMINI_GENERATE_ONLY: Final = [("POST", "/models/gemini-3.8-flash:generateContent")]
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
        case "system" | "developer":
            return [_item(role, SYSTEM), {"role": "user", "content": final_text(marker)}]
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
        pytest.param("developer", SYSTEM_LABEL, id="developer"),
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


@pytest.mark.parametrize("drop_params", (False, True), ids=("as-sent", "drop-params"))
def test_nova_tool_marks_follow_drop_params(gateway: Gateway, drop_params: bool) -> None:
    marker: Final = new_marker()
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/converse/us.amazon.nova-2-lite-v1:0",
            api_key=PROVIDER_KEY,
            api_base=wire.url,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            drop_params=drop_params,
        )
        items: Final = _tool_items(marker, output_mark=EPHEMERAL, call_mark=EPHEMERAL)
        answer: Final = _send(gateway, _responses(model, items, tools=[_TOOL]))
        assert answer.status == 200, answer.text
        labels: Final = bedrock_labels(_only_request(wire))
    assert (_BEDROCK_RESULT_LABEL in labels) is not drop_params, labels
    assert (len(labels) == 2) is not drop_params, labels


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


@pytest.mark.parametrize(
    "fields",
    (
        pytest.param({}, id="bare"),
        pytest.param({"instructions": SYSTEM, "tools": [_TOOL]}, id="instructions-and-tools"),
    ),
)
def test_gemini_marked_items_never_start_a_context_cache(gateway: Gateway, fields: dict[str, JsonValue]) -> None:
    marker: Final = new_marker()
    long_system: Final = "lorem " * (2 * get_prompt_cache_min_tokens(_GEMINI))
    items: Final[list[JsonValue]] = [
        _item("system", long_system),
        _item("developer", SYSTEM),
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(_gemini_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_GEMINI, api_base=wire.url, api_key=PROVIDER_KEY)
        _assert_completed(_send(gateway, _responses(model, items, **fields)))
        targets: Final = [(request.method, request.target.split("?")[0]) for request in wire.drain()]
    assert targets == _GEMINI_GENERATE_ONLY


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


def _openai_block_marks(request: Request) -> list[JsonValue]:
    return [object_value(block).get("cache_control") for block in _blocks_of(_JSON_OBJECT.validate_json(request.body))]


def test_databricks_claude_keeps_the_item_mark_like_its_chat_route(gateway: Gateway) -> None:
    marker: Final = new_marker()
    reply: Final = chat_reply(f"chatcmpl-{marker}", "databricks-claude-sonnet-4", "sunny", stream=False)
    with wire_server(answering_model_discovery(lambda request: reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="databricks/databricks-claude-sonnet-4", api_base=wire.url, api_key=PROVIDER_KEY
        )
        _assert_completed(_send(gateway, _marked_system(model, marker)))
        posted: Final = [request for request in wire.drain() if request.method == "POST"]
    assert [request.target for request in posted] == ["/chat/completions"], [request.target for request in posted]
    assert _openai_block_marks(posted[0]) == [EPHEMERAL]


def test_openai_compatible_chat_bridge_leaves_the_item_mark_off(gateway: Gateway) -> None:
    marker: Final = new_marker()
    reply: Final = chat_reply(f"chatcmpl-{marker}", "gpt-5.4", "sunny", stream=False)
    with wire_server(answering_model_discovery(lambda request: reply)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_OPENAI_COMPATIBLE, api_base=f"{wire.url}/v1", api_key=PROVIDER_KEY, use_chat_completions_api=True
        )
        _assert_completed(_send(gateway, _marked_system(model, marker)))
        posted: Final = [request for request in wire.drain() if request.method == "POST"]
    assert [request.target for request in posted] == ["/v1/chat/completions"], [request.target for request in posted]
    assert _openai_message_marks(posted[0]) == []


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


def test_five_marks_keep_the_newest_four_under_drop_params(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=PROVIDER_KEY, drop_params=True
        )
        body: Final = _responses(model, _many_marks(marker, assistant_marked=True), tools=[_TOOL])
        _assert_completed(_send(gateway, body))
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == [ASK_LABEL, _ASSISTANT_LABEL, _RESULT_LABEL, final_label(marker)]


def test_a_top_level_mark_takes_a_slot_under_drop_params(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=PROVIDER_KEY, drop_params=True
        )
        body: Final = _responses(
            model, _many_marks(marker, assistant_marked=False), tools=[_TOOL], cache_control=EPHEMERAL
        )
        _assert_completed(_send(gateway, body))
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == [ASK_LABEL, _RESULT_LABEL, final_label(marker), "request"]


@pytest.mark.parametrize(
    "mark",
    (
        pytest.param("ephemeral", id="string"),
        pytest.param({"type": "forever"}, id="unknown-type"),
    ),
)
def test_malformed_mark_is_dropped_under_drop_params(gateway: Gateway, mark: JsonValue) -> None:
    marker: Final = new_marker()
    with wire_server(_strict_anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=PROVIDER_KEY, drop_params=True
        )
        _assert_completed(_send(gateway, _marked_system(model, marker, mark)))
        assert anthropic_labels(_only_request(wire)) == []


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


_SAP_DISCOVERY: Final = ("/oauth/token", "/v2/lm/")
_SAP_COMPLETION: Final = "/sap-deployment/v2/completion"


def _only_post(wire: Wire) -> Request:
    posted: Final = [
        request
        for request in wire.drain()
        if request.method == "POST" and not request.target.startswith(_SAP_DISCOVERY)
    ]
    assert len(posted) == 1, [request.target for request in posted]
    return posted[0]


def _message_mark_labels(message: JsonValue) -> list[str]:
    fields: Final = object_value(message)
    content: Final = fields.get("content")
    blocks: Final = content if isinstance(content, list) else []
    block_labels: Final = [
        f"{fields.get('role')}:block" for block in blocks if object_value(block).get("cache_control") is not None
    ]
    return [f"{fields.get('role')}:message"] * int(fields.get("cache_control") is not None) + block_labels


def _openai_mark_labels(request: Request) -> list[str]:
    messages: Final = _JSON_OBJECT.validate_json(request.body)["messages"]
    return list(chain.from_iterable(map(_message_mark_labels, messages if isinstance(messages, list) else [])))


def _sap_mark_labels(request: Request) -> list[str]:
    config: Final = object_value(_JSON_OBJECT.validate_json(request.body)["config"])
    prompt: Final = object_value(object_value(object_value(config["modules"])["prompt_templating"])["prompt"])
    template: Final = prompt["template"]
    return list(chain.from_iterable(map(_message_mark_labels, template if isinstance(template, list) else [])))


def _route_mark_labels(request: Request) -> list[str]:
    if "/converse" in request.target:
        return bedrock_labels(request)
    if request.target.endswith("/chat/completions"):
        return _openai_mark_labels(request)
    if request.target == _SAP_COMPLETION:
        return _sap_mark_labels(request)
    return anthropic_labels(request)


def _sap_peer(request: Request) -> Reply:
    match request.target:
        case "/oauth/token":
            return Reply(body=json.dumps({"access_token": "scripted", "expires_in": 3600}).encode())
        case "/v2/lm/deployments":
            deployment: Final = {
                "scenarioId": "orchestration",
                "configurationId": "orchestration-config",
                "deploymentUrl": f"http://{request.headers['host']}/sap-deployment",
                "createdAt": "2026-01-01T00:00:00Z",
            }
            return Reply(body=json.dumps({"resources": [deployment]}).encode())
        case "/v2/lm/configurations/orchestration-config":
            return Reply(body=json.dumps({"executableId": "orchestration"}).encode())
        case _:
            completion: Final = chat_reply(f"chatcmpl-{marker_of(request)}", "scripted", "sunny", stream=False)
            return Reply(body=json.dumps({"final_result": json.loads(completion.body)}).encode())


def _route_peer(request: Request) -> Reply:
    if "/converse" in request.target:
        return bedrock_peer(request)
    if request.target.endswith("/chat/completions"):
        return chat_reply(f"chatcmpl-{marker_of(request)}", "scripted", "sunny", stream=False)
    if request.target.startswith((*_SAP_DISCOVERY, "/sap-deployment/")):
        return _sap_peer(request)
    return anthropic_peer(request)


def _route_deployment(gateway: Gateway, scenario: Scenario, wire: Wire, route: str) -> str:
    match route:
        case "vertex-claude":
            return scenario.model(
                model=f"vertex_ai/{ANTHROPIC_MODEL}",
                api_key=None,
                api_base=wire.url,
                vertex_project="scripted-project",
                vertex_location="us-east5",
                vertex_credentials=service_account_json("scripted-project", gateway.upstream_url.rstrip("/")),
            )
        case "bedrock-llama":
            return scenario.model(
                model="bedrock/converse/meta.llama3-3-70b-instruct-v1:0",
                api_key=PROVIDER_KEY,
                api_base=wire.url,
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=wire.url,
            )
        case "sap-claude":
            service_key: Final = {
                "clientid": "scripted",
                "clientsecret": PROVIDER_KEY,
                "url": wire.url,
                "serviceurls": {"AI_API_URL": wire.url},
            }
            return scenario.model(
                model="sap/anthropic--claude-4.5-sonnet", api_base=wire.url, api_key=json.dumps(service_key)
            )
        case _ if route.startswith("openrouter/"):
            return scenario.model(model=route, api_base=wire.url, api_key=PROVIDER_KEY, use_chat_completions_api=True)
        case _:
            return scenario.model(model=route, api_base=wire.url, api_key=PROVIDER_KEY)


def _chat_with_tool_mark(model: str, marker: str) -> dict[str, JsonValue]:
    tool_call: Final[dict[str, JsonValue]] = {
        "id": call_id(CITIES[0]),
        "type": "function",
        "function": {"name": "lookup_weather", "arguments": json.dumps({"city": CITIES[0]})},
    }
    return {
        "model": model,
        "max_tokens": 64,
        "tools": [{"type": "function", "function": {"name": "lookup_weather", "parameters": {"type": "object"}}}],
        "messages": [
            {"role": "user", "content": ASK},
            {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": call_id(CITIES[0]), "content": "sunny.", "cache_control": EPHEMERAL},
            {"role": "user", "content": final_text(marker)},
        ],
    }


def _shaped_requests(model: str, shape: str) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    match shape:
        case "tool":
            bridged: Final = _responses(model, _tool_items(new_marker(), output_mark=EPHEMERAL), tools=[_TOOL])
            return _chat_with_tool_mark(model, new_marker()), bridged
        case _:
            return _chat_with_message_mark(model, new_marker()), _marked_system(model, new_marker())


@pytest.mark.parametrize(
    ("route", "shape", "keeps"),
    (
        pytest.param("dashscope/qwen-plus", "system", True, id="dashscope"),
        pytest.param("minimax/MiniMax-M2", "system", True, id="minimax"),
        pytest.param("zai/glm-4.6", "system", True, id="zai"),
        pytest.param(f"openrouter/anthropic/{ANTHROPIC_MODEL}", "system", True, id="openrouter-claude"),
        pytest.param("openrouter/google/gemini-3.8-flash", "system", True, id="openrouter-gemini"),
        pytest.param("snowflake/claude-4-sonnet", "tool", True, id="snowflake-claude-tool-output"),
        pytest.param("vertex-claude", "system", True, id="vertex-claude"),
        pytest.param("sap-claude", "system", True, id="sap-claude"),
        pytest.param("anthropic/my-gateway-model", "system", True, id="anthropic-non-claude-name"),
        pytest.param("snowflake/claude-4-sonnet", "system", False, id="snowflake-claude-string-system"),
        pytest.param("openrouter/openai/gpt-5.4", "system", False, id="openrouter-gpt"),
        pytest.param("snowflake/llama3.1-70b", "system", False, id="snowflake-llama"),
        pytest.param("snowflake/llama-claude-distill", "system", False, id="snowflake-claude-inside-the-name"),
        pytest.param("databricks/databricks-meta-llama-3-3-70b-instruct", "tool", False, id="databricks-llama"),
        pytest.param("bedrock-llama", "system", False, id="bedrock-llama"),
    ),
)
def test_item_mark_reaches_each_route_the_way_a_chat_message_mark_does(
    gateway: Gateway, route: str, shape: str, keeps: bool
) -> None:
    with wire_server(answering_model_discovery(_route_peer)) as wire, gateway.scenario() as scenario:
        model: Final = _route_deployment(gateway, scenario, wire, route)
        on_chat_body, bridged_body = _shaped_requests(model, shape)
        on_chat: Final = gateway.request("POST", "/v1/chat/completions", on_chat_body)
        assert on_chat.status_code == 200, on_chat.text
        chat_request: Final = _only_post(wire)
        _assert_completed(_send(gateway, bridged_body))
        bridged_request: Final = _only_post(wire)
    bridged_labels: Final = _route_mark_labels(bridged_request)
    assert bridged_request.target == chat_request.target
    assert bridged_labels == (_route_mark_labels(chat_request) if keeps else [])
    assert bool(bridged_labels) is keeps, bridged_labels


@pytest.mark.parametrize(
    ("drop_params", "expected"),
    (
        pytest.param(False, [_BEDROCK_SYSTEM_LABEL], id="as-sent"),
        pytest.param(True, [], id="drop-params"),
    ),
)
def test_unmapped_bedrock_arn_keeps_the_item_mark_unless_drop_params(
    gateway: Gateway, drop_params: bool, expected: list[str]
) -> None:
    marker: Final = new_marker()
    with wire_server(_bedrock_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="bedrock/converse/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/scripted",
            api_key=PROVIDER_KEY,
            api_base=wire.url,
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
            drop_params=drop_params,
        )
        answer: Final = _send(gateway, _marked_system(model, marker))
        assert answer.status == 200, answer.text
        assert bedrock_labels(_only_request(wire)) == expected


@pytest.mark.parametrize("request_drop_params", (True, "true"), ids=("bool", "string"))
def test_request_drop_params_trims_five_marks_to_the_newest_four(
    gateway: Gateway, request_drop_params: JsonValue
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        items: Final = _many_marks(marker, assistant_marked=True)
        _assert_completed(_send(gateway, _responses(model, items, tools=[_TOOL], drop_params=request_drop_params)))
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == [ASK_LABEL, _ASSISTANT_LABEL, _RESULT_LABEL, final_label(marker)]


@pytest.mark.parametrize(
    ("deployment_fields", "request_drop_params"),
    (
        pytest.param({}, None, id="request-null"),
        pytest.param({}, "", id="request-empty-string"),
        pytest.param({"drop_params": True}, False, id="request-false-over-deployment-true"),
    ),
)
def test_request_without_drop_params_forwards_five_marks_as_sent(
    gateway: Gateway, deployment_fields: dict[str, JsonValue], request_drop_params: JsonValue
) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{ANTHROPIC_MODEL}", api_base=wire.url, api_key=PROVIDER_KEY, **deployment_fields
        )
        items: Final = _many_marks(marker, assistant_marked=True)
        answer: Final = _send(gateway, _responses(model, items, tools=[_TOOL], drop_params=request_drop_params))
        labels: Final = anthropic_labels(_only_request(wire))
    assert answer.status == 400, answer.text
    assert "A maximum of 4 blocks with cache_control" in answer.text, answer.text
    assert len(labels) == 5, labels


def _marked_function_call_items(marker: str) -> list[JsonValue]:
    calls: Final[list[JsonValue]] = [
        {
            "type": "function_call",
            "call_id": call_id(city),
            "name": "lookup_weather",
            "arguments": json.dumps({"city": city}),
            "cache_control": EPHEMERAL,
        }
        for city in CITIES
    ]
    outputs: Final[list[JsonValue]] = [
        {"type": "function_call_output", "call_id": call_id(city), "output": "sunny."} for city in CITIES
    ]
    return [{"role": "user", "content": ASK}, *calls, *outputs, {"role": "user", "content": final_text(marker)}]


def test_injection_points_count_a_marked_function_tool_toward_the_cap(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire, inject=True)
        body: Final = _responses(
            model,
            _marked_function_call_items(marker),
            instructions=SYSTEM,
            tools=[{**_TOOL, "cache_control": EPHEMERAL}],
        )
        _assert_completed(_send(gateway, body))
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == ["tool:lookup_weather", *(tool_use_label(city) for city in CITIES)]


def test_assistant_text_after_a_function_call_keeps_its_mark(gateway: Gateway) -> None:
    marker: Final = new_marker()
    items: Final[list[JsonValue]] = [
        {"role": "user", "content": ASK},
        {
            "type": "function_call",
            "call_id": call_id(CITIES[0]),
            "name": "lookup_weather",
            "arguments": json.dumps({"city": CITIES[0]}),
        },
        _item("assistant", _CHECK),
        {"type": "function_call_output", "call_id": call_id(CITIES[0]), "output": "sunny."},
        {"role": "user", "content": final_text(marker)},
    ]
    with wire_server(anthropic_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        _assert_completed(_send(gateway, _responses(model, items, tools=[_TOOL])))
        labels: Final = anthropic_labels(_only_request(wire))
    assert labels == [_ASSISTANT_LABEL]


_COUNTED_TOKENS: Final = 42


def _count_peer(request: Request) -> Reply:
    if request.target.endswith("/count_tokens"):
        return Reply(body=json.dumps({"input_tokens": _COUNTED_TOKENS}).encode())
    return anthropic_peer(request)


def test_input_tokens_counts_marked_items_without_their_marks(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(_count_peer) as wire, gateway.scenario() as scenario:
        model: Final = _anthropic_deployment(scenario, wire)
        body: Final = {"model": model, "input": _many_marks(marker, assistant_marked=True), "tools": [_TOOL]}
        response: Final = gateway.request("POST", "/v1/responses/input_tokens", body)
        counted: Final = _only_request(wire)
    assert response.status_code == 200, response.text
    assert _JSON_OBJECT.validate_json(response.text) == {
        "object": "response.input_tokens",
        "input_tokens": _COUNTED_TOKENS,
    }
    assert counted.target.endswith("/count_tokens"), counted.target
    assert anthropic_labels(counted) == []
