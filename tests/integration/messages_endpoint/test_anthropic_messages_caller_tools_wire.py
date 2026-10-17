import json
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, list_value, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

CLAUDE_MODEL: Final = "caller-tools-claude"
OPENAI_MODEL: Final = "caller-tools-openai"
GEMINI_MODEL: Final = "caller-tools-gemini"
DOWN_MODEL: Final = "caller-tools-gemini-down"
OPENAI_DOWN_MODEL: Final = "caller-tools-openai-down"
TOOL_RESULTS_GUARDRAIL: Final = "caller-tools-results-filter"
CLAUDE_TARGET: Final = "/anthropic/v1/messages"
RESPONSES_TARGET: Final = "/openai/v1/responses"
GEMINI_PREFIX: Final = "/gemini/"
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_NONCE: Final = re.compile(rb"nonce-([0-9a-f]{32})")


def strict_tool(name: str = "lint") -> dict[str, JsonValue]:
    return {
        "name": name,
        "description": "Lint a file",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                }
            },
            "required": ["target"],
            "additionalProperties": False,
        },
        "input_examples": [{"target": {"path": "a.py"}}],
    }


def tool_with_a_property_named_cache_control() -> dict[str, JsonValue]:
    return {
        "name": "tag",
        "description": "Tag a file",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "cache_control": {"type": "string"}},
            "required": ["path", "cache_control"],
        },
    }


def schema_nested(levels: int) -> dict[str, JsonValue]:
    return reduce(
        lambda inner, level: {"type": "object", "properties": {f"p{level}": inner}},
        range(levels),
        {"type": "string"},
    )


def parsed_json(raw: bytes) -> JsonValue:
    return json.loads(raw)


def json_mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict), f"Expected an object, received {type(value).__name__}"
    return value


def nonce() -> str:
    return f"nonce-{uuid.uuid4().hex}"


def nonce_hex(body: bytes) -> str:
    match: Final = _NONCE.search(body)
    assert match is not None, body[:200]
    return match.group(1).decode()


def claude_reply(request: Request) -> Reply:
    identity: Final = nonce_hex(request.body)
    message: Final[dict[str, JsonValue]] = {
        "id": f"msg_{identity}",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": f"linted {identity}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 12, "output_tokens": 3},
    }
    if json_mapping(parsed_json(request.body)).get("stream") is not True:
        return Reply(body=json.dumps(message).encode())
    events: Final = (
        ("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": f"linted {identity}"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode() for event, payload in events),
    )


def responses_reply(request: Request) -> Reply:
    identity: Final = nonce_hex(request.body)
    return Reply(
        body=json.dumps(
            {
                "id": f"resp_{identity}",
                "object": "response",
                "created_at": 1789788253,
                "status": "completed",
                "model": "gpt-5.4-mini",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{identity}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": f"linted {identity}", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
            }
        ).encode()
    )


def gemini_reply(request: Request) -> Reply:
    identity: Final = nonce_hex(request.body)
    return Reply(
        body=json.dumps(
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": f"linted {identity}"}], "role": "model"},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3, "totalTokenCount": 15},
                "modelVersion": "gemini-3.8-flash",
            }
        ).encode()
    )


def respond(request: Request) -> Reply:
    if request.method != "POST":
        return Reply(body=b'{"object":"list","data":[]}')
    if request.target.startswith(CLAUDE_TARGET):
        return claude_reply(request)
    if request.target.startswith(RESPONSES_TARGET):
        return responses_reply(request)
    if request.target.startswith(GEMINI_PREFIX):
        return gemini_reply(request)
    return Reply(status=404, body=json.dumps({"error": {"message": f"no route for {request.target}"}}).encode())


def deployment(name: str, model: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": api_base, "api_key": f"synthetic-{name}-key"},
    }


def write_config(directory: Path, wire_url: str) -> Path:
    path: Final = directory / f"caller-tools-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            {
                "model_list": [
                    deployment(CLAUDE_MODEL, "anthropic/claude-haiku-5-5", f"{wire_url}/anthropic"),
                    deployment(OPENAI_MODEL, "openai/gpt-5.4-mini", f"{wire_url}/openai/v1"),
                    deployment(GEMINI_MODEL, "gemini/gemini-3.8-flash", f"{wire_url}/gemini"),
                    deployment(DOWN_MODEL, "gemini/gemini-3.8-flash", "http://127.0.0.1:9"),
                    deployment(OPENAI_DOWN_MODEL, "openai/gpt-5.4-mini", "http://127.0.0.1:9"),
                ],
                "router_settings": {"fallbacks": [{DOWN_MODEL: [CLAUDE_MODEL]}, {OPENAI_DOWN_MODEL: [CLAUDE_MODEL]}]},
                "guardrails": [
                    {
                        "guardrail_name": TOOL_RESULTS_GUARDRAIL,
                        "litellm_params": {
                            "guardrail": "litellm_content_filter",
                            "mode": "pre_call",
                            "default_on": False,
                            "scan_only_tool_results": True,
                            "blocked_words": [{"keyword": "ignore previous instructions", "action": "BLOCK"}],
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_prompts_in_spend_logs": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                },
            }
        )
    )
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    wire: Wire

    def send(
        self, body: Mapping[str, JsonValue], *, streaming: bool = False, close_connection: bool = False
    ) -> tuple[int, bytes]:
        headers: Final = {
            "Authorization": f"Bearer {self.gateway.key}",
            **({"Connection": "close"} if close_connection else {}),
        }
        with self.gateway.client.stream(
            "POST", "/v1/messages", json={**body, "stream": streaming}, headers=headers, timeout=120
        ) as response:
            return response.status_code, response.read()

    def received(self, request_nonce: str) -> tuple[Request, ...]:
        return tuple(request for request in self.wire.drain() if request_nonce.encode() in request.body)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("caller-tools")
    with (
        gateway_from_environment() as shared,
        wire_server(respond) as wire,
        owned_proxy(shared, directory, {}, config=write_config(directory, wire.url), workers=2) as gateway,
    ):
        yield Rig(gateway, wire)


def messages_body(
    model: str,
    request_nonce: str,
    *,
    tools: JsonValue,
    guardrails: Sequence[str] = (TOOL_RESULTS_GUARDRAIL,),
) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": f"lint the repo {request_nonce}"}],
        "tools": tools,
        "guardrails": list(guardrails),
    }


def reply_text(status: int, raw: bytes, *, streaming: bool) -> str:
    assert status == 200, raw[:500]
    if not streaming:
        return "".join(
            str(object_value(block)["text"]) for block in list_value(_JSON_OBJECT.validate_json(raw)["content"])
        )
    deltas: Final = (
        object_value(_JSON_OBJECT.validate_json(line.removeprefix(b"data: "))["delta"])
        for line in raw.split(b"\n")
        if line.startswith(b"data: ") and b'"content_block_delta"' in line
    )
    return "".join(str(delta["text"]) for delta in deltas)


def sent_body(requests: Sequence[Request], target: str) -> dict[str, JsonValue]:
    assert [request.target.split("?")[0] for request in requests] == [target], [r.target for r in requests]
    return json_mapping(parsed_json(requests[0].body))


@pytest.mark.parametrize("streaming", [False, True], ids=["json", "stream"])
def test_claude_behind_a_tool_results_guardrail_receives_the_tools_as_sent(rig: Rig, streaming: bool) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(messages_body(CLAUDE_MODEL, request_nonce, tools=[strict_tool()]), streaming=streaming)

    assert reply_text(status, raw, streaming=streaming) == f"linted {request_nonce.removeprefix('nonce-')}"
    assert sent_body(rig.received(request_nonce), CLAUDE_TARGET)["tools"] == [strict_tool()]


@pytest.mark.parametrize("guardrails", [(), (TOOL_RESULTS_GUARDRAIL,)], ids=["no_guardrail", "tool_results_guardrail"])
def test_fallback_after_a_failed_gemini_call_sends_claude_the_tools_as_sent(
    rig: Rig, guardrails: tuple[str, ...]
) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(messages_body(DOWN_MODEL, request_nonce, tools=[strict_tool()], guardrails=guardrails))

    assert reply_text(status, raw, streaming=False) == f"linted {request_nonce.removeprefix('nonce-')}"
    assert sent_body(rig.received(request_nonce), CLAUDE_TARGET)["tools"] == [strict_tool()]


def test_fallback_after_a_failed_openai_call_sends_claude_a_property_named_cache_control(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(
        messages_body(
            OPENAI_DOWN_MODEL, request_nonce, tools=[tool_with_a_property_named_cache_control()], guardrails=()
        )
    )

    assert reply_text(status, raw, streaming=False) == f"linted {request_nonce.removeprefix('nonce-')}"
    assert sent_body(rig.received(request_nonce), CLAUDE_TARGET)["tools"] == [
        tool_with_a_property_named_cache_control()
    ]


def test_openai_bridge_behind_a_tool_results_guardrail_sends_the_schema_as_sent(rig: Rig) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(messages_body(OPENAI_MODEL, request_nonce, tools=[strict_tool()]))

    assert reply_text(status, raw, streaming=False) == f"linted {request_nonce.removeprefix('nonce-')}"
    tools: Final = list_value(sent_body(rig.received(request_nonce), RESPONSES_TARGET)["tools"])
    assert [object_value(tool)["name"] for tool in tools] == ["lint"]
    assert object_value(tools[0])["parameters"] == strict_tool()["input_schema"]


@pytest.mark.parametrize(
    ("model", "guardrails"),
    [(GEMINI_MODEL, ()), (CLAUDE_MODEL, (TOOL_RESULTS_GUARDRAIL,))],
    ids=["gemini_bridge", "claude_tool_results_guardrail"],
)
def test_spend_log_keeps_the_tools_the_caller_sent(rig: Rig, model: str, guardrails: tuple[str, ...]) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(messages_body(model, request_nonce, tools=[strict_tool()], guardrails=guardrails))

    assert reply_text(status, raw, streaming=False) == f"linted {request_nonce.removeprefix('nonce-')}"
    assert len(rig.received(request_nonce)) == 1
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT proxy_server_request FROM "LiteLLM_SpendLogs" WHERE strpos(proxy_server_request::text, %s) > 0',
            (request_nonce,),
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert object_value(rows[0]["proxy_server_request"])["tools"] == [strict_tool()]


@pytest.mark.parametrize("tools", [7, "x" * 5000, ["lint"], ""], ids=["int", "five_kb", "list_of_str", "empty_string"])
def test_a_malformed_tools_field_is_answered_with_an_error_and_the_proxy_keeps_serving(
    rig: Rig, tools: JsonValue
) -> None:
    request_nonce: Final = nonce()
    status, raw = rig.send(messages_body(CLAUDE_MODEL, request_nonce, tools=tools), close_connection=True)

    assert status >= 400, raw[:500]
    assert object_value(_JSON_OBJECT.validate_json(raw)["error"])["message"], raw[:500]
    assert rig.received(request_nonce) == ()
    control_nonce: Final = nonce()
    control_status, control_raw = rig.send(messages_body(CLAUDE_MODEL, control_nonce, tools=[strict_tool()]))
    assert reply_text(control_status, control_raw, streaming=False) == f"linted {control_nonce.removeprefix('nonce-')}"


@pytest.mark.parametrize(
    ("model", "target"),
    [(CLAUDE_MODEL, CLAUDE_TARGET), (OPENAI_MODEL, RESPONSES_TARGET)],
    ids=["claude", "openai_bridge"],
)
def test_a_schema_nested_past_the_recursion_limit_reaches_the_provider_as_sent(
    rig: Rig, model: str, target: str
) -> None:
    request_nonce: Final = nonce()
    schema: Final = schema_nested(420)
    status, raw = rig.send(
        messages_body(model, request_nonce, tools=[{"name": "deep", "description": "Deep", "input_schema": schema}])
    )

    assert reply_text(status, raw, streaming=False) == f"linted {request_nonce.removeprefix('nonce-')}"
    tool: Final = json_mapping(list_value(sent_body(rig.received(request_nonce), target)["tools"])[0])
    assert tool.get("input_schema", tool.get("parameters")) == schema


def test_the_same_request_twice_reaches_claude_with_the_tools_as_sent_both_times(rig: Rig) -> None:
    request_nonce: Final = nonce()
    body: Final = messages_body(CLAUDE_MODEL, request_nonce, tools=[strict_tool()])
    replies: Final = (rig.send(body), rig.send(body))

    assert [reply_text(status, raw, streaming=False) for status, raw in replies] == [
        f"linted {request_nonce.removeprefix('nonce-')}"
    ] * 2
    sent: Final = rig.received(request_nonce)
    assert [request.target for request in sent] == [CLAUDE_TARGET] * 2
    assert [_JSON_OBJECT.validate_json(request.body)["tools"] for request in sent] == [[strict_tool()]] * 2


@pytest.mark.parametrize("tools", [None, []], ids=["missing", "empty_list"])
def test_a_request_without_tools_reaches_claude_without_tools(rig: Rig, tools: list[JsonValue] | None) -> None:
    request_nonce: Final = nonce()
    body: Final = messages_body(CLAUDE_MODEL, request_nonce, tools=tools)
    status, raw = rig.send({key: value for key, value in body.items() if value is not None})

    assert reply_text(status, raw, streaming=False) == f"linted {request_nonce.removeprefix('nonce-')}"
    assert sent_body(rig.received(request_nonce), CLAUDE_TARGET).get("tools", []) == []


def burst_case(index: int) -> tuple[str, str, bool]:
    kinds: Final = (
        (CLAUDE_MODEL, CLAUDE_TARGET, False),
        (CLAUDE_MODEL, CLAUDE_TARGET, True),
        (OPENAI_MODEL, RESPONSES_TARGET, False),
        (DOWN_MODEL, CLAUDE_TARGET, False),
    )
    return kinds[index % len(kinds)]


def test_a_concurrent_mixed_burst_reaches_each_provider_with_its_own_tools(rig: Rig) -> None:
    nonces: Final = tuple(nonce() for _ in range(24))

    def send(index: int) -> tuple[int, bytes]:
        model, _, streaming = burst_case(index)
        return rig.send(messages_body(model, nonces[index], tools=[strict_tool(f"lint_{index}")]), streaming=streaming)

    with ThreadPoolExecutor(max_workers=12) as pool:
        replies: Final = tuple(pool.map(send, range(len(nonces))))

    assert [reply_text(status, raw, streaming=burst_case(index)[2]) for index, (status, raw) in enumerate(replies)] == [
        f"linted {request_nonce.removeprefix('nonce-')}" for request_nonce in nonces
    ]
    sent: Final = rig.wire.drain()
    assert [sent_tools(sent, request_nonce, burst_case(index)[1]) for index, request_nonce in enumerate(nonces)] == [
        (f"lint_{index}", strict_tool()["input_schema"]) for index in range(len(nonces))
    ]


def sent_tools(sent: Sequence[Request], request_nonce: str, target: str) -> tuple[str, JsonValue]:
    tools: Final = list_value(sent_body([r for r in sent if request_nonce.encode() in r.body], target)["tools"])
    assert len(tools) == 1, tools
    tool: Final = object_value(tools[0])
    return str(tool["name"]), tool.get("input_schema", tool.get("parameters"))
