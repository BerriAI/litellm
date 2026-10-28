import asyncio
import json
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import anthropic
import httpx
import pytest
from integration._support.anthropic_sse import sse
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.ports import allocate_port
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

from litellm.llms.anthropic.pass_through.adapters.transformation import truncate_tool_name

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_JSON_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])
_NAME_SOURCE: Final = (
    "mcp__plugin_support-kit_remote-workspace__getAccessibleWorkspaceResourcesForTheCurrentUser"
    "_including_archived_and_shared_workspaces_across_every_region_and_every_tenant_the_user_belongs_to"
    "_sorted_by_last_activity_with_owner_and_role_details"
)
_SEARCH: Final = "ToolSearch"
_SHORT_NAME: Final = "workspace_resources"
_EMAIL: Final = "jane.doe@example.com"
_MASK: Final = "<EMAIL>"
_FORCED_ID: Final = "toolu_forced"
_GUARDRAIL_NAME_LIMIT: Final = 64
_BLOCK_REASON: Final = "tool policy says no"


def _tool_name(length: int) -> str:
    name: Final = _NAME_SOURCE[:length]
    assert len(name) == length, name
    return name


def _tool(name: str) -> dict[str, JsonValue]:
    return {"name": name, "description": "Returns OK", "input_schema": {"type": "object", "properties": {}}}


def _every_carrier(model: str, name: str, guardrail: str, stream: bool) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 32,
        "stream": stream,
        "guardrails": [guardrail],
        "tools": [_tool(_SEARCH), {**_tool(name), "defer_loading": True}],
        "tool_choice": {"type": "tool", "name": name},
        "messages": [
            {"role": "user", "content": "find the workspace tool"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": _SEARCH, "input": {}}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": [{"type": "tool_reference", "tool_name": name}],
                    }
                ],
            },
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_02", "name": name, "input": {}}]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_02", "content": "workspace-alpha"},
                    {"type": "text", "text": f"share it with {_EMAIL}"},
                ],
            },
        ],
    }


def _blocks(body: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    messages: Final = _JSON_OBJECTS.validate_python(body["messages"])
    contents: Final = tuple(message["content"] for message in messages if isinstance(message["content"], list))
    return tuple(object_value(block) for content in contents for block in content)


def _tool_result_parts(body: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    contents: Final = tuple(block.get("content") for block in _blocks(body) if block.get("type") == "tool_result")
    return tuple(object_value(part) for content in contents if isinstance(content, list) for part in content)


def _named_tools(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(string_value(tool["name"]) for tool in _JSON_OBJECTS.validate_python(body.get("tools") or []))


def _tool_use_names(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    return tuple(string_value(block["name"]) for block in _blocks(body) if block.get("type") == "tool_use")


def _tool_references(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    parts: Final = _tool_result_parts(body)
    return tuple(string_value(part["tool_name"]) for part in parts if part.get("type") == "tool_reference")


def _forced_name(body: Mapping[str, JsonValue]) -> str | None:
    choice: Final = body.get("tool_choice")
    if not isinstance(choice, dict) or choice.get("type") != "tool":
        return None
    return string_value(choice["name"])


def _carriers(body: Mapping[str, JsonValue]) -> tuple[tuple[str, ...], str | None, tuple[str, ...], tuple[str, ...]]:
    return _named_tools(body), _forced_name(body), _tool_use_names(body), _tool_references(body)


def _anthropic_error(message: str) -> Reply:
    error: Final = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return Reply(status=400, body=json.dumps(error).encode())


def _message(model: str, forced: str | None) -> dict[str, JsonValue]:
    content: Final[list[JsonValue]] = (
        [{"type": "text", "text": "done"}]
        if forced is None
        else [{"type": "tool_use", "id": _FORCED_ID, "name": forced, "input": {}}]
    )
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": "end_turn" if forced is None else "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 3},
    }


def _sse_reply(message: Mapping[str, JsonValue]) -> Reply:
    block: Final = object_value(_JSON_OBJECTS.validate_python(message["content"])[0])
    delta: Final[dict[str, JsonValue]] = (
        {"type": "input_json_delta", "partial_json": "{}"}
        if block["type"] == "tool_use"
        else {"type": "text_delta", "text": string_value(block["text"])}
    )
    started: Final = {**block, "input": {}} if block["type"] == "tool_use" else {**block, "text": ""}
    events: Final = (
        sse("message_start", {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}),
        sse("content_block_start", {"type": "content_block_start", "index": 0, "content_block": started}),
        sse("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": message["stop_reason"]}, "usage": {"output_tokens": 3}},
        ),
        sse("message_stop", {"type": "message_stop"}),
    )
    return Reply(chunks=events, content_type="text/event-stream")


def _anthropic_peer(request: Request) -> Reply:
    assert request.target.endswith("/v1/messages"), request.target
    body: Final = _JSON_OBJECT.validate_json(request.body)
    available: Final = frozenset(_named_tools(body))
    missing: Final = [name for name in _tool_references(body) if name not in available]
    if missing:
        return _anthropic_error(f"Tool reference '{missing[0]}' not found in available tools")
    forced: Final = _forced_name(body)
    if forced is not None and forced not in available:
        return _anthropic_error(f"Tool '{forced}' not found in provided tools")
    message: Final = _message(string_value(body["model"]), forced)
    return _sse_reply(message) if body.get("stream") is True else Reply(body=json.dumps(message).encode())


def _chat_peer(request: Request) -> Reply:
    assert request.target.endswith("/chat/completions"), request.target
    body: Final = _JSON_OBJECT.validate_json(request.body)
    function: Final = object_value(_JSON_OBJECTS.validate_python(body["tools"])[0]["function"])
    call: Final = {"id": "call_1", "type": "function", "function": {"name": function["name"], "arguments": "{}"}}
    reply: Final = {
        "id": "chatcmpl-long-tool-names",
        "object": "chat.completion",
        "created": 1,
        "model": body["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
    }
    return Reply(body=json.dumps(reply).encode())


def _masked(content: JsonValue) -> JsonValue:
    if isinstance(content, str):
        return content.replace(_EMAIL, _MASK)
    if isinstance(content, list):
        return [_masked(part) for part in content]
    if isinstance(content, dict) and isinstance(text := content.get("text"), str):
        return {**content, "text": text.replace(_EMAIL, _MASK)}
    return content


def _masked_row(row: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {**row, "content": _masked(row["content"])} if "content" in row else dict(row)


def _guardrail_peer(request: Request) -> Reply:
    verdict: Final = request.target.split("/")[1]
    rows: Final = _JSON_OBJECTS.validate_python(
        _JSON_OBJECT.validate_json(request.body).get("structured_messages") or []
    )
    match verdict:
        case "block":
            return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": _BLOCK_REASON}).encode())
        case "rewrite" if rows:
            rewritten: Final = [_masked_row(row) for row in rows]
            return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "structured_messages": rewritten}).encode())
        case _:
            return Reply(body=json.dumps({"action": "NONE"}).encode())


def _delete_guardrail(gateway: Gateway, identity: str) -> None:
    deleted: Final = gateway.request("DELETE", f"/guardrails/{identity}")
    assert deleted.status_code == 200, deleted.text


def _guardrail(gateway: Gateway, scenario: Scenario, api_base: str, mode: str, **extra: JsonValue) -> str:
    name: Final = f"long-tool-names-{mode}-{uuid.uuid4().hex}"
    params: Final[dict[str, JsonValue]] = {
        "guardrail": "generic_guardrail_api",
        "mode": mode,
        "default_on": False,
        "api_base": api_base,
        "api_key": "synthetic-guardrail-key",
        **extra,
    }
    created: Final = gateway.post("/guardrails", {"guardrail": {"guardrail_name": name, "litellm_params": params}})
    scenario.cleanups.callback(_delete_guardrail, gateway, string_value(created["guardrail_id"]))
    return name


def _anthropic_model(scenario: Scenario, provider: Wire) -> str:
    return scenario.model(model="anthropic/claude-opus-5", api_base=provider.url, api_key="synthetic-anthropic-key")


def _sse_tool_use_names(text: str) -> tuple[str, ...]:
    events: Final = tuple(
        _JSON_OBJECT.validate_json(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    )
    starts: Final = tuple(
        object_value(event["content_block"]) for event in events if event.get("type") == "content_block_start"
    )
    return tuple(string_value(block["name"]) for block in starts if block.get("type") == "tool_use")


def _answered_tool_names(response: httpx.Response, stream: bool) -> tuple[str, ...]:
    if stream:
        return _sse_tool_use_names(response.text)
    content: Final = _JSON_OBJECTS.validate_python(_JSON_OBJECT.validate_json(response.content)["content"])
    return tuple(string_value(block["name"]) for block in content if block.get("type") == "tool_use")


def _vendor_tool_names(vendor_requests: tuple[Request, ...]) -> tuple[str, ...]:
    bodies: Final = tuple(_JSON_OBJECT.validate_json(request.body) for request in vendor_requests)
    tools: Final = tuple(
        object_value(tool) for body in bodies for tool in _JSON_OBJECTS.validate_python(body.get("tools") or [])
    )
    return tuple(string_value(object_value(tool["function"])["name"]) for tool in tools)


@pytest.mark.parametrize("stream", (False, True), ids=("json", "sse"))
@pytest.mark.parametrize("length", (64, 65, 128, 200))
@pytest.mark.parametrize("verdict", ("echo", "rewrite"))
@pytest.mark.parametrize("mode", ("pre_call", "during_call"))
def test_guardrail_forwards_every_tool_name_the_caller_sent(
    gateway: Gateway, mode: str, verdict: str, length: int, stream: bool
) -> None:
    name: Final = _tool_name(length)
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
    ):
        model: Final = _anthropic_model(scenario, provider)
        guardrail: Final = _guardrail(gateway, scenario, f"{vendor.url}/{verdict}", mode)
        short: Final = gateway.request("POST", "/v1/messages", _every_carrier(model, _SHORT_NAME, guardrail, stream))
        response: Final = gateway.request("POST", "/v1/messages", _every_carrier(model, name, guardrail, stream))
        sent: Final = provider.drain()
        screened: Final = vendor.drain()

    assert (short.status_code, response.status_code) == (200, 200), (short.text, response.text)
    assert len(sent) == 2, sent
    assert len(screened) == 2, screened
    outbound: Final = _JSON_OBJECT.validate_json(sent[1].body)
    assert _carriers(outbound) == ((_SEARCH, name), name, (_SEARCH, name), (name,)), outbound
    assert outbound == _JSON_OBJECT.validate_json(sent[0].body.decode().replace(_SHORT_NAME, name)), outbound
    assert _answered_tool_names(response, stream) == (name,), response.text
    assert all(len(seen) <= _GUARDRAIL_NAME_LIMIT for seen in _vendor_tool_names(screened)), screened
    rewritten: Final = mode == "pre_call" and verdict == "rewrite"
    assert (_MASK in sent[1].body.decode(), _EMAIL in sent[1].body.decode()) == (rewritten, not rewritten), outbound


@pytest.mark.parametrize("verdict", ("echo", "rewrite"))
def test_short_tool_named_like_a_shortened_history_tool_keeps_its_name(gateway: Gateway, verdict: str) -> None:
    history_only: Final = _tool_name(90)
    alias: Final = truncate_tool_name(history_only)
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
    ):
        body: Final[dict[str, JsonValue]] = {
            "model": _anthropic_model(scenario, provider),
            "max_tokens": 32,
            "guardrails": [_guardrail(gateway, scenario, f"{vendor.url}/{verdict}", "pre_call")],
            "tools": [_tool(alias)],
            "tool_choice": {"type": "tool", "name": alias},
            "messages": [
                {"role": "user", "content": "list my workspaces"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_01", "name": history_only, "input": {}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_01", "content": "workspace-alpha"},
                        {"type": "text", "text": f"share it with {_EMAIL}"},
                    ],
                },
            ],
        }
        response: Final = gateway.request("POST", "/v1/messages", body)
        sent: Final = provider.drain()

    assert response.status_code == 200, response.text
    assert len(sent) == 1, sent
    outbound: Final = _JSON_OBJECT.validate_json(sent[0].body)
    assert _carriers(outbound) == ((alias,), alias, (history_only,), ()), outbound
    assert _answered_tool_names(response, stream=False) == (alias,), response.text


def test_logging_only_guardrail_forwards_the_long_tool_names_untouched(gateway: Gateway) -> None:
    name: Final = _tool_name(90)
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
    ):
        body: Final = _every_carrier(
            _anthropic_model(scenario, provider),
            name,
            _guardrail(gateway, scenario, f"{vendor.url}/rewrite", "logging_only"),
            False,
        )
        response: Final = gateway.request("POST", "/v1/messages", body)
        sent: Final = provider.drain()
        eventually(vendor.received.qsize, lambda count: count >= 1)
        screened: Final = vendor.drain()

    assert response.status_code == 200, response.text
    assert len(sent) == 1, sent
    outbound: Final = _JSON_OBJECT.validate_json(sent[0].body)
    assert _carriers(outbound) == ((_SEARCH, name), name, (_SEARCH, name), (name,)), outbound
    assert _EMAIL in sent[0].body.decode() and _MASK not in sent[0].body.decode(), outbound
    assert _answered_tool_names(response, stream=False) == (name,), response.text
    assert all(len(seen) <= _GUARDRAIL_NAME_LIMIT for seen in _vendor_tool_names(screened)), screened


@pytest.mark.parametrize("fallback", ("fail_open", "fail_closed"))
def test_unreachable_guardrail_follows_its_fallback_with_long_tool_names(gateway: Gateway, fallback: str) -> None:
    name: Final = _tool_name(90)
    with wire_server(_anthropic_peer) as provider, gateway.scenario() as scenario:
        guardrail: Final = _guardrail(
            gateway, scenario, f"http://127.0.0.1:{allocate_port()}", "pre_call", unreachable_fallback=fallback
        )
        body: Final = _every_carrier(_anthropic_model(scenario, provider), name, guardrail, False)
        response: Final = gateway.request("POST", "/v1/messages", body)
        sent: Final = provider.drain()

    if fallback == "fail_closed":
        assert (response.status_code, sent) == (500, ()), response.text
        return
    assert response.status_code == 200, response.text
    assert len(sent) == 1, sent
    outbound: Final = _JSON_OBJECT.validate_json(sent[0].body)
    assert _carriers(outbound) == ((_SEARCH, name), name, (_SEARCH, name), (name,)), outbound


def test_blocking_guardrail_stops_a_long_tool_name_request_before_the_provider(gateway: Gateway) -> None:
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
    ):
        body: Final = _every_carrier(
            _anthropic_model(scenario, provider),
            _tool_name(90),
            _guardrail(gateway, scenario, f"{vendor.url}/block", "pre_call"),
            stream=False,
        )
        response: Final = gateway.request("POST", "/v1/messages", body)
        sent: Final = provider.drain()

    assert response.status_code == 400, response.text
    assert _BLOCK_REASON in response.text, response.text
    assert sent == (), sent


@pytest.mark.parametrize("length", (64, 90))
def test_bridged_route_answers_with_the_tool_name_the_caller_sent(gateway: Gateway, length: int) -> None:
    name: Final = _tool_name(length)
    with (
        wire_server(_chat_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(
            model="mistral/mistral-small-latest", api_base=provider.url, api_key="synthetic-mistral-key"
        )
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "max_tokens": 32,
            "guardrails": [_guardrail(gateway, scenario, f"{vendor.url}/echo", "pre_call")],
            "tools": [_tool(name)],
            "messages": [{"role": "user", "content": "list my workspaces"}],
        }
        response: Final = gateway.request("POST", "/v1/messages", body)
        sent: Final = provider.drain()

    assert response.status_code == 200, response.text
    assert len(sent) == 1, sent
    assert _answered_tool_names(response, stream=False) == (name,), response.text


_PARAMETERS: Final[dict[str, JsonValue]] = {"type": "object", "properties": {}}


def _openai_shaped_body(endpoint: str, model: str, name: str, guardrail: str) -> dict[str, JsonValue]:
    shared: Final[dict[str, JsonValue]] = {"model": model, "guardrails": [guardrail]}
    if endpoint == "/v1/chat/completions":
        return {
            **shared,
            "messages": [{"role": "user", "content": "list my workspaces"}],
            "tools": [{"type": "function", "function": {"name": name, "parameters": _PARAMETERS}}],
            "tool_choice": {"type": "function", "function": {"name": name}},
        }
    return {
        **shared,
        "input": "list my workspaces",
        "tools": [{"type": "function", "name": name, "parameters": _PARAMETERS}],
        "tool_choice": {"type": "function", "name": name},
    }


def _openai_shaped_tool_calls(endpoint: str, response: httpx.Response) -> tuple[str, ...]:
    body: Final = _JSON_OBJECT.validate_json(response.content)
    if endpoint == "/v1/chat/completions":
        message: Final = object_value(object_value(_JSON_OBJECTS.validate_python(body["choices"])[0])["message"])
        calls: Final = _JSON_OBJECTS.validate_python(message.get("tool_calls") or [])
        return tuple(string_value(object_value(call["function"])["name"]) for call in calls)
    items: Final = _JSON_OBJECTS.validate_python(body["output"])
    return tuple(string_value(item["name"]) for item in items if item.get("type") == "function_call")


@pytest.mark.parametrize("endpoint", ("/v1/chat/completions", "/v1/responses"))
def test_openai_shaped_routes_forward_the_long_tool_name_the_caller_sent(gateway: Gateway, endpoint: str) -> None:
    name: Final = _tool_name(90)
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
    ):
        body: Final = _openai_shaped_body(
            endpoint,
            _anthropic_model(scenario, provider),
            name,
            _guardrail(gateway, scenario, f"{vendor.url}/echo", "pre_call"),
        )
        response: Final = gateway.request("POST", endpoint, body)
        sent: Final = provider.drain()

    assert response.status_code == 200, response.text
    assert len(sent) == 1, sent
    assert _carriers(_JSON_OBJECT.validate_json(sent[0].body))[:2] == ((name,), name), sent
    assert _openai_shaped_tool_calls(endpoint, response) == (name,), response.text


def _forced_tool_kwargs(model: str, name: str, guardrail: str) -> dict[str, object]:
    return {
        "model": model,
        "max_tokens": 32,
        "tools": [_tool(name)],
        "tool_choice": {"type": "tool", "name": name},
        "messages": [{"role": "user", "content": "list my workspaces"}],
        "extra_body": {"guardrails": [guardrail]},
    }


def _origin(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def test_sync_anthropic_sdk_gets_the_forced_long_tool_back(gateway: Gateway) -> None:
    name: Final = _tool_name(90)
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
        httpx.Client(timeout=30, trust_env=False) as transport,
        anthropic.Anthropic(
            api_key=gateway.key, base_url=_origin(gateway), max_retries=0, http_client=transport
        ) as client,
    ):
        kwargs: Final = _forced_tool_kwargs(
            _anthropic_model(scenario, provider),
            name,
            _guardrail(gateway, scenario, f"{vendor.url}/echo", "pre_call"),
        )
        message: Final = client.messages.create(**kwargs)
        sent: Final = provider.drain()

    assert [block.name for block in message.content if block.type == "tool_use"] == [name], message
    assert _carriers(_JSON_OBJECT.validate_json(sent[0].body))[:2] == ((name,), name), sent


def test_async_anthropic_sdk_streams_the_forced_long_tool_back(gateway: Gateway) -> None:
    name: Final = _tool_name(90)

    async def streamed(client: anthropic.AsyncAnthropic, kwargs: dict[str, object]) -> tuple[str, ...]:
        async with client.messages.stream(**kwargs) as stream:
            final: Final = await stream.get_final_message()
        return tuple(block.name for block in final.content if block.type == "tool_use")

    async def run(kwargs: dict[str, object]) -> tuple[str, ...]:
        async with (
            httpx.AsyncClient(timeout=30, trust_env=False) as transport,
            anthropic.AsyncAnthropic(
                api_key=gateway.key, base_url=_origin(gateway), max_retries=0, http_client=transport
            ) as client,
        ):
            return await streamed(client, kwargs)

    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
    ):
        kwargs: Final = _forced_tool_kwargs(
            _anthropic_model(scenario, provider),
            name,
            _guardrail(gateway, scenario, f"{vendor.url}/echo", "pre_call"),
        )
        answered: Final = asyncio.run(run(kwargs))
        sent: Final = provider.drain()

    assert answered == (name,), answered
    assert _carriers(_JSON_OBJECT.validate_json(sent[0].body))[:2] == ((name,), name), sent


def test_concurrent_long_tool_name_requests_each_keep_their_own_names(gateway: Gateway) -> None:
    names: Final = tuple(f"{_tool_name(85)}_{index:02d}" for index in range(24))
    with (
        wire_server(_anthropic_peer) as provider,
        wire_server(_guardrail_peer) as vendor,
        gateway.scenario() as scenario,
    ):
        model: Final = _anthropic_model(scenario, provider)
        guardrail: Final = _guardrail(gateway, scenario, f"{vendor.url}/rewrite", "pre_call")
        with ThreadPoolExecutor(max_workers=12) as pool:
            responses: Final = tuple(
                pool.map(
                    lambda name: gateway.request("POST", "/v1/messages", _every_carrier(model, name, guardrail, False)),
                    names,
                )
            )
        sent: Final = provider.drain()

    assert [response.status_code for response in responses] == [200] * len(names), [r.text for r in responses]
    assert [_answered_tool_names(response, stream=False) for response in responses] == [(name,) for name in names]
    forwarded: Final = sorted(_carriers(_JSON_OBJECT.validate_json(request.body)) for request in sent)
    assert forwarded == sorted(((_SEARCH, name), name, (_SEARCH, name), (name,)) for name in names), forwarded
