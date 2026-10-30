import json
from collections.abc import Mapping
from itertools import count
from typing import Final, Literal

import pytest
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

Surface = Literal["responses", "messages", "chat/completions"]
URL: Final = "https://mcp.example.com/github"


def _policy(tools: JsonValue = None, url: str = URL) -> dict[str, JsonValue]:
    return {
        "mcp_servers": [
            {
                "name": "github",
                "url": url,
                "tools": ["search_code"] if tools is None else tools,
            }
        ]
    }


def _request(
    surface: Surface,
    model: str,
    names: JsonValue = None,
    url: str = URL,
    bounded: bool = True,
) -> dict[str, JsonValue]:
    scope: Final[JsonValue] = ["search_code"] if names is None else names
    if surface == "messages":
        return {
            "model": model,
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Use the search tool"}],
            "mcp_servers": [
                {
                    "type": "url",
                    "name": "github",
                    "url": url,
                    **(
                        {"tool_configuration": {"allowed_tools": scope}}
                        if bounded
                        else {}
                    ),
                }
            ],
        }
    attachment: Final[dict[str, JsonValue]] = {
        "type": "mcp",
        "server_label": "github",
        "server_url": url,
        "require_approval": "never",
        **({"allowed_tools": scope} if bounded else {}),
    }
    inputs: Final[dict[str, JsonValue]] = (
        {"input": "Use the search tool"}
        if surface == "responses"
        else {"messages": [{"role": "user", "content": "Use the search tool"}]}
    )
    return {"model": model, "tools": [attachment], **inputs}


def _response(surface: Surface, tool: str | None = None) -> dict[str, JsonValue]:
    if surface == "responses":
        item: Final[JsonValue] = (
            {
                "type": "mcp_call",
                "id": "mcp_call_1",
                "server_label": "github",
                "name": tool,
                "arguments": "{}",
                "output": "found",
                "error": None,
            }
            if tool is not None
            else {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "found", "annotations": []}
                ],
            }
        )
        return {
            "id": "resp_native_policy",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [item],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        }
    if surface == "messages":
        block: Final[JsonValue] = (
            {
                "type": "mcp_tool_use",
                "id": "mcp_call_1",
                "server_name": "github",
                "name": tool,
                "input": {},
            }
            if tool is not None
            else {"type": "text", "text": "found"}
        )
        return {
            "id": "msg_native_policy",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": [block],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }
    return {
        "id": "chatcmpl-native-policy",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "found"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _reply(surface: Surface, request: Request, tool: str | None = None) -> Reply:
    incoming: Final = JSON_OBJECT.validate_json(request.body)
    assert "native_mcp_policy" not in incoming
    payload: Final = _response(surface, tool)
    if not incoming.get("stream"):
        return Reply(body=json.dumps(payload).encode())
    if surface == "responses":
        output: Final = payload["output"]
        assert isinstance(output, list)
        events: Final[tuple[Mapping[str, JsonValue], ...]] = (
            {
                "type": "response.created",
                "response": {**payload, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": output[0],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": output[0]},
            {"type": "response.completed", "response": payload},
        )
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(
                f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': i})}\n\n".encode()
                for i, event in enumerate(events)
            ),
        )
    content: Final = payload["content"]
    assert isinstance(content, list)
    frames: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {
            "type": "message_start",
            "message": {**payload, "content": [], "stop_reason": None},
        },
        {"type": "content_block_start", "index": 0, "content_block": content[0]},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            for event in frames
        ),
    )


def _provider(surface: Surface) -> str:
    return (
        "anthropic/claude-haiku-4-5" if surface == "messages" else "openai/gpt-4o-mini"
    )


@pytest.mark.parametrize("surface", ("responses", "messages", "chat/completions"))
@pytest.mark.parametrize("case", ("tool", "url", "unbounded", "empty"))
def test_native_mcp_request_denial_never_reaches_provider(
    gateway: Gateway, surface: Surface, case: str
) -> None:
    with (
        wire_server(lambda request: _reply(surface, request)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_provider(surface), api_base=wire.url)
        key: Final = scenario.key(
            native_mcp_policy={} if case == "empty" else _policy()
        )
        offered: Final[JsonValue] = (
            ["delete_repo"] if case == "tool" else ["search_code"]
        )
        body: Final = _request(
            surface,
            model,
            offered,
            URL + "/spoofed" if case == "url" else URL,
            bounded=case != "unbounded",
        )
        denied: Final = gateway.request("POST", f"/v1/{surface}", body, key=key)
        assert denied.status_code == 400, denied.text
        assert (
            object_value(JSON_OBJECT.validate_json(denied.content)["error"])["type"]
            == "tool_access_denied"
        )
        assert wire.drain() == ()


@pytest.mark.parametrize("surface", ("responses", "messages", "chat/completions"))
def test_native_mcp_union_allows_both_grants_and_absent_policy_preserves_behavior(
    gateway: Gateway, surface: Surface
) -> None:
    with (
        wire_server(lambda request: _reply(surface, request)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_provider(surface), api_base=wire.url)
        team: Final = scenario.team(native_mcp_policy=_policy())
        key: Final = scenario.key(team_id=team, native_mcp_policy=_policy(["get_file"]))
        allowed: Final = gateway.request(
            "POST",
            f"/v1/{surface}",
            _request(surface, model, ["search_code", "get_file"]),
            key=key,
        )
        assert allowed.status_code == 200, allowed.text
        unrestricted: Final = scenario.key()
        unchanged: Final = gateway.request(
            "POST",
            f"/v1/{surface}",
            _request(surface, model, ["delete_repo"]),
            key=unrestricted,
        )
        assert unchanged.status_code == 200, unchanged.text
        sent: Final = wire.drain()
        assert len(sent) == 2
        assert (
            "search_code" in sent[0].body.decode()
            and "get_file" in sent[0].body.decode()
        )
        assert "delete_repo" in sent[1].body.decode()


def test_native_mcp_empty_policy_rejects_tools_supplied_by_model_configuration(
    gateway: Gateway,
) -> None:
    with (
        wire_server(lambda request: _reply("responses", request)) as wire,
        gateway.scenario() as scenario,
    ):
        attachment: Final = _request("responses", "unused", ["delete_repo"])["tools"]
        model: Final = scenario.model(
            model=_provider("responses"), api_base=wire.url, tools=attachment
        )
        key: Final = scenario.key(native_mcp_policy={})
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": "Use the configured tools"},
            key=key,
        )
        sent: Final = wire.drain()
        assert response.status_code == 400, (
            response.text,
            tuple(request.body.decode() for request in sent),
        )
        assert sent == ()


def test_native_mcp_empty_policy_preserves_ordinary_function_tools(
    gateway: Gateway,
) -> None:
    with (
        wire_server(lambda request: _reply("chat/completions", request)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy={})
        tools: Final[JsonValue] = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Check the weather"}],
                "tools": tools,
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        sent: Final = wire.drain()
        assert len(sent) == 1
        assert JSON_OBJECT.validate_json(sent[0].body)["tools"] == tools


@pytest.mark.parametrize("surface", ("responses", "messages", "chat/completions"))
def test_native_mcp_whole_server_grant_allows_unbounded_attachment(
    gateway: Gateway, surface: Surface
) -> None:
    with (
        wire_server(lambda request: _reply(surface, request)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_provider(surface), api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy=_policy("*"))
        response: Final = gateway.request(
            "POST", f"/v1/{surface}", _request(surface, model, bounded=False), key=key
        )
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1


def test_native_mcp_policy_revocation_reaches_a_warmed_peer(
    gateway: Gateway, peer: Gateway
) -> None:
    with (
        wire_server(lambda request: _reply("responses", request)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy=_policy())
        body: Final = _request("responses", model)
        assert (
            gateway.request("POST", "/v1/responses", body, key=key).status_code == 200
        )
        warmed: Final = eventually(
            lambda: peer.request("POST", "/v1/responses", body, key=key),
            lambda response: response.status_code == 200,
            seconds=60,
        )
        assert warmed.status_code == 200, warmed.text
        gateway.post("/key/update", {"key": key, "native_mcp_policy": {}})
        refused: Final = eventually(
            lambda: peer.request("POST", "/v1/responses", body, key=key),
            lambda response: response.status_code == 400,
        )
        assert "tool_access_denied" in refused.text
        wire.drain()
        assert peer.request("POST", "/v1/responses", body, key=key).status_code == 400
        assert wire.drain() == ()


def test_native_mcp_modern_toolset_honors_default_off_and_explicit_grants(
    gateway: Gateway,
) -> None:
    with (
        wire_server(lambda request: _reply("messages", request)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_provider("messages"), api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy=_policy())
        body: Final[dict[str, JsonValue]] = {
            **_request("messages", model, bounded=False),
            "tools": [
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": "github",
                    "default_config": {"enabled": False},
                    "configs": {
                        "search_code": {"enabled": True},
                        "delete_repo": {"enabled": False},
                    },
                }
            ],
        }
        response: Final = gateway.request("POST", "/v1/messages", body, key=key)
        assert response.status_code == 200, response.text
        sent: Final = wire.drain()
        assert len(sent) == 1
        assert JSON_OBJECT.validate_json(sent[0].body)["tools"] == body["tools"]


def test_native_mcp_response_checks_anthropic_tool_through_chat_translation(
    gateway: Gateway,
) -> None:
    with (
        wire_server(lambda request: _reply("messages", request, "delete_repo")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_provider("messages"), api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy=_policy())
        body: Final = _request("messages", model)
        response: Final = gateway.request("POST", "/v1/chat/completions", body, key=key)
        assert len(wire.drain()) == 1
        assert response.status_code == 400, response.text
        assert "tool_access_denied" in response.text


@pytest.mark.parametrize("surface", ("responses", "messages"))
@pytest.mark.parametrize("tool", ("search_code", "delete_repo"))
def test_native_mcp_response_checks_returned_tool(
    gateway: Gateway, surface: Surface, tool: str
) -> None:
    with (
        wire_server(lambda request: _reply(surface, request, tool)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_provider(surface), api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy=_policy())
        response: Final = gateway.request(
            "POST", f"/v1/{surface}", _request(surface, model), key=key
        )
        assert len(wire.drain()) == 1
        assert response.status_code == (200 if tool == "search_code" else 400), (
            response.text
        )
        assert tool in response.text
        if tool == "delete_repo":
            assert object_value(JSON_OBJECT.validate_json(response.content)["error"])[
                "type"
            ] == (
                "invalid_request_error"
                if surface == "messages"
                else "tool_access_denied"
            )


def test_native_mcp_allowed_response_remains_retrievable(gateway: Gateway) -> None:
    reply: Final = Reply(
        body=json.dumps(_response("responses", "search_code")).encode()
    )
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy=_policy())
        created: Final = gateway.post(
            "/v1/responses", _request("responses", model), key=key
        )
        fetched: Final = gateway.request(
            "GET", f"/v1/responses/{created['id']}", key=key
        )
        assert len(wire.drain()) == 2
        assert fetched.status_code == 200, fetched.text
        assert JSON_OBJECT.validate_json(fetched.content)["output"] == created["output"]


@pytest.mark.parametrize("surface", ("responses", "messages"))
@pytest.mark.parametrize("stream", (False,), ids=("json",))
def test_native_mcp_configured_grant_binds_returned_tools(
    gateway: Gateway, surface: Surface, stream: bool
) -> None:
    field: Final = "mcp_servers" if surface == "messages" else "tools"
    configured: Final = _request(surface, "unused")[field]
    with (
        wire_server(lambda request: _reply(surface, request, "search_code")) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = (
            scenario.model(
                model=_provider(surface), api_base=wire.url, mcp_servers=configured
            )
            if surface == "messages"
            else scenario.model(
                model=_provider(surface), api_base=wire.url, tools=configured
            )
        )
        key: Final = scenario.key(native_mcp_policy=_policy())
        body: Final = {
            name: value
            for name, value in _request(surface, model).items()
            if name != field
        }
        response: Final = gateway.request(
            "POST", f"/v1/{surface}", {**body, "stream": stream}, key=key
        )
        assert response.status_code == 200, response.text
        assert "search_code" in response.text
        assert "tool_access_denied" not in response.text
        if stream:
            assert (
                "response.completed" in response.text
                if surface == "responses"
                else "message_stop" in response.text
            )
        sent: Final = wire.drain()
        assert len(sent) == 1
        assert JSON_OBJECT.validate_json(sent[0].body)[field] == configured


def test_native_mcp_retry_preserves_request_binding(gateway: Gateway) -> None:
    attempts: Final = count()

    def upstream(request: Request) -> Reply:
        if next(attempts) == 0:
            return Reply(
                status=429,
                body=b'{"error":{"message":"retry control","type":"rate_limit_error"}}',
            )
        return _reply("responses", request, "search_code")

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url, num_retries=1)
        key: Final = scenario.key(native_mcp_policy=_policy())
        body: Final = _request("responses", model)
        response: Final = gateway.request("POST", "/v1/responses", body, key=key)
        assert response.status_code == 200, response.text
        assert "search_code" in response.text
        sent: Final = wire.drain()
        assert len(sent) == 2
        assert all(
            JSON_OBJECT.validate_json(request.body)["tools"] == body["tools"]
            for request in sent
        )


@pytest.mark.parametrize("granted", (True, False), ids=("allowed", "denied"))
def test_native_mcp_fallback_checks_its_configured_tools(
    gateway: Gateway, granted: bool
) -> None:
    def upstream(request: Request) -> Reply:
        if request.target.startswith("/primary/"):
            return Reply(
                status=503,
                body=b'{"error":{"message":"fallback control","type":"server_error"}}',
            )
        return _reply("responses", request, "search_code")

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        primary: Final = scenario.model(api_base=f"{wire.url}/primary", num_retries=0)
        configured: Final = _request("responses", "unused")["tools"]
        fallback: Final = scenario.model(
            api_base=f"{wire.url}/fallback", num_retries=0, tools=configured
        )
        key: Final = scenario.key(native_mcp_policy=_policy() if granted else {})
        response: Final = gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": primary,
                "input": "Use configured tools",
                "fallbacks": [fallback],
            },
            key=key,
        )
        sent: Final = wire.drain()
        assert response.status_code == (200 if granted else 400), (
            response.text,
            tuple(request.target for request in sent),
        )
        assert sum(request.target.startswith("/primary/") for request in sent) == 1
        fallback_requests: Final = tuple(
            request for request in sent if request.target.startswith("/fallback/")
        )
        if granted:
            assert len(fallback_requests) == 1
            assert (
                JSON_OBJECT.validate_json(fallback_requests[0].body)["tools"]
                == configured
            )
            assert "search_code" in response.text
        else:
            assert fallback_requests == ()
            assert "tool_access_denied" in response.text


def test_native_mcp_retrieval_applies_current_policy_after_revocation(
    gateway: Gateway,
) -> None:
    reply: Final = Reply(
        body=json.dumps(_response("responses", "search_code")).encode()
    )
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy=_policy())
        created: Final = gateway.post(
            "/v1/responses", _request("responses", model), key=key
        )
        control: Final = gateway.request(
            "GET", f"/v1/responses/{created['id']}", key=key
        )
        assert control.status_code == 200, control.text
        gateway.post("/key/update", {"key": key, "native_mcp_policy": {}})
        denied: Final = gateway.request(
            "GET", f"/v1/responses/{created['id']}", key=key
        )
        assert denied.status_code == 400, denied.text
        assert "tool_access_denied" in denied.text
        assert len(wire.drain()) in (2, 3)


def test_native_mcp_empty_policy_preserves_ordinary_response_retrieval(
    gateway: Gateway,
) -> None:
    reply: Final = Reply(body=json.dumps(_response("responses")).encode())
    with wire_server(lambda _: reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url)
        key: Final = scenario.key(native_mcp_policy={})
        created: Final = gateway.post(
            "/v1/responses", {"model": model, "input": "Answer without tools"}, key=key
        )
        fetched: Final = gateway.request(
            "GET", f"/v1/responses/{created['id']}", key=key
        )
        assert fetched.status_code == 200, fetched.text
        assert JSON_OBJECT.validate_json(fetched.content)["output"] == created["output"]
        assert len(wire.drain()) == 2
