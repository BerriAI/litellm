import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Literal

import httpx
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.mcp import McpPeer, mcp_peer, register_mcp, tool_calls
from integration._support.wire import Reply, Request, Wire, wire_server

Surface = Literal["chat", "responses", "messages", "messages_bridge"]
SURFACES: Final[tuple[Surface, ...]] = ("chat", "responses", "messages", "messages_bridge")
ADD: Final = {"a": 2, "b": 3}
ANSWER: Final = "the sum is 5"
GATEWAY_REF: Final = {"type": "mcp", "server_url": "litellm_proxy", "server_label": "litellm"}
AUTO: Final = {**GATEWAY_REF, "require_approval": "never"}


def _json(body: Mapping[str, object]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _has_tool_result(body: Mapping[str, object]) -> bool:
    messages: Final = body.get("messages")
    inputs: Final = body.get("input")
    if isinstance(messages, list):
        return any(
            isinstance(message, dict)
            and (
                message.get("role") == "tool"
                or any(
                    isinstance(block, dict) and block.get("type") == "tool_result"
                    for block in (message.get("content") if isinstance(message.get("content"), list) else ())
                )
            )
            for message in messages
        )
    if isinstance(inputs, list):
        return any(isinstance(item, dict) and item.get("type") == "function_call_output" for item in inputs)
    return False


def _model_double(tool: str) -> Callable[[Request], Reply]:
    arguments: Final = json.dumps(ADD)

    def respond(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        assert isinstance(body, dict), request.body
        done: Final = _has_tool_result(body)
        usage: Final = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        if request.target.endswith("/chat/completions"):
            message: Final = (
                {"role": "assistant", "content": ANSWER}
                if done
                else {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": tool, "arguments": arguments}}
                    ],
                }
            )
            finish: Final = "stop" if done else "tool_calls"
            if body.get("stream") is True:
                delta: Final = (
                    {**message, "tool_calls": [{**call, "index": 0} for call in message["tool_calls"]]}
                    if "tool_calls" in message
                    else message
                )
                chunk: Final = {
                    "id": "chatcmpl-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "gpt-4o-mini",
                }
                return Reply(
                    content_type="text/event-stream",
                    chunks=(
                        f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n".encode(),
                        f"data: {json.dumps({**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}], 'usage': usage})}\n\n".encode(),
                        b"data: [DONE]\n\n",
                    ),
                )
            return _json(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                    "usage": usage,
                }
            )
        if request.target.endswith("/messages"):
            content: Final = (
                [{"type": "text", "text": ANSWER}]
                if done
                else [{"type": "tool_use", "id": "toolu_1", "name": tool, "input": ADD}]
            )
            return _json(
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude",
                    "content": content,
                    "stop_reason": "end_turn" if done else "tool_use",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )
        assert request.target.endswith("/responses"), request.target
        output: Final = (
            [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": ANSWER, "annotations": []}],
                }
            ]
            if done
            else [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": tool,
                    "arguments": arguments,
                    "status": "completed",
                }
            ]
        )
        return _json(
            {
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": output,
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        )

    return respond


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    scenario: Scenario
    peer: McpPeer
    wire: Wire
    alias: str
    server_id: str
    model: str
    surface: Surface

    @property
    def tool(self) -> str:
        return f"{self.alias}-add"

    def send(self, key: str, tools: Sequence[Mapping[str, object]], **extra: object) -> httpx.Response:
        prompt: Final = f"add {self.alias}"
        headers: Final = {"Authorization": f"Bearer {key}"}
        if self.surface == "chat":
            body: Final = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "tools": list(tools)}
            return self.gateway.client.post("/v1/chat/completions", headers=headers, json={**body, **extra}, timeout=90)
        if self.surface == "responses":
            return self.gateway.client.post(
                "/v1/responses",
                headers=headers,
                json={"model": self.model, "input": prompt, "tools": list(tools), **extra},
                timeout=90,
            )
        return self.gateway.client.post(
            "/v1/messages",
            headers=headers,
            json={
                "model": self.model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt}],
                "tools": list(tools),
                **extra,
            },
            timeout=90,
        )

    def upstream_tools(self) -> tuple[tuple[str, ...], ...]:
        return tuple(_tool_names(json.loads(request.body)) for request in self.wire.drain())

    def final_text(self, body: Mapping[str, object]) -> str:
        if self.surface == "chat":
            choices: Final = body["choices"]
            assert isinstance(choices, list), body
            return str(choices[0]["message"]["content"])
        if self.surface == "responses":
            output: Final = body["output"]
            assert isinstance(output, list), body
            return "".join(
                str(block["text"])
                for item in output
                if isinstance(item, dict) and item.get("type") == "message"
                for block in item["content"]
                if isinstance(block, dict) and block.get("type") == "output_text"
            )
        content: Final = body["content"]
        assert isinstance(content, list), body
        return "".join(str(block["text"]) for block in content if block.get("type") == "text")


def _tool_names(body: Mapping[str, object]) -> tuple[str, ...]:
    tools: Final = body.get("tools")
    if not isinstance(tools, list):
        return ()
    return tuple(
        str(tool["name"] if "name" in tool else tool["function"]["name"]) for tool in tools if isinstance(tool, dict)
    )


def _upstream_model(surface: Surface) -> str:
    return {
        "chat": "openai/gpt-4o-mini",
        "responses": "openai/responses/gpt-4o-mini",
        "messages": "anthropic/claude-sonnet-4-5",
        "messages_bridge": "hosted_vllm/gpt-4o-mini",
    }[surface]


@contextmanager
def _rig(gateway: Gateway, surface: Surface) -> Iterator[Rig]:
    alias: Final = "llm" + uuid.uuid4().hex[:8]
    with (
        mcp_peer() as peer,
        wire_server(_model_double(f"{alias}-add")) as wire,
        gateway.scenario() as scenario,
    ):
        server_id: Final = register_mcp(scenario, peer, alias)
        model: Final = scenario.model(model=_upstream_model(surface), api_base=wire.url + "/v1")
        peer.drain()
        yield Rig(gateway, scenario, peer, wire, alias, server_id, model, surface)


def _granted_key(rig: Rig) -> str:
    return rig.scenario.key(object_permission={"mcp_servers": [rig.server_id]})


def _peer_add_calls(peer: McpPeer) -> tuple[dict[str, object], ...]:
    return tuple(
        call
        for call in tool_calls(peer.drain())
        if isinstance(call["body"], dict) and isinstance(call["body"].get("params"), dict)
    )


@pytest.mark.parametrize("surface", SURFACES)
def test_auto_approved_gateway_tool_is_listed_executed_once_and_fed_back(gateway: Gateway, surface: Surface) -> None:
    with _rig(gateway, surface) as rig:
        key: Final = _granted_key(rig)
        response: Final = rig.send(key, [AUTO])
        assert response.status_code == 200, response.text
        calls: Final = _peer_add_calls(rig.peer)
        requests: Final = rig.upstream_tools()
        assert [call["body"]["params"]["name"] for call in calls] == ["add"], calls
        assert calls[0]["body"]["params"]["arguments"] == ADD, calls
        assert len(requests) == 2, requests
        assert all(rig.tool in names for names in requests), requests
        assert rig.final_text(response.json()) == ANSWER, response.text


@pytest.mark.parametrize("surface", ("chat", "responses", "messages"))
def test_gateway_tool_without_auto_approval_returns_the_call_to_the_caller_and_never_hits_the_peer(
    gateway: Gateway, surface: Surface
) -> None:
    with _rig(gateway, surface) as rig:
        key: Final = _granted_key(rig)
        response: Final = rig.send(key, [GATEWAY_REF])
        assert response.status_code == 200, response.text
        assert rig.tool in response.text, response.text
        assert rig.final_text(response.json()) != ANSWER, response.text
        assert _peer_add_calls(rig.peer) == (), "tool ran without approval"
        requests: Final = rig.upstream_tools()
        assert len(requests) == 1 and rig.tool in requests[0], requests


@pytest.mark.parametrize("surface", ("chat", "responses", "messages"))
def test_ungranted_key_gets_no_gateway_tools_and_the_peer_is_never_reached(gateway: Gateway, surface: Surface) -> None:
    with _rig(gateway, surface) as rig:
        key: Final = rig.scenario.key()
        response: Final = rig.send(key, [AUTO])
        assert _peer_add_calls(rig.peer) == (), "denied caller reached the peer"
        requests: Final = rig.upstream_tools()
        assert requests and all(rig.tool not in names for names in requests), requests
        assert response.status_code == 200, response.text


@pytest.mark.parametrize("surface", ("chat", "responses", "messages"))
def test_allowed_tools_narrows_the_tool_list_handed_to_the_model(gateway: Gateway, surface: Surface) -> None:
    with _rig(gateway, surface) as rig:
        key: Final = _granted_key(rig)
        response: Final = rig.send(key, [{**AUTO, "allowed_tools": [rig.tool]}])
        assert response.status_code == 200, response.text
        requests: Final = rig.upstream_tools()
        assert requests and all(names == (rig.tool,) for names in requests), requests
        assert [call["body"]["params"]["name"] for call in _peer_add_calls(rig.peer)] == ["add"]


@pytest.mark.parametrize("surface", ("chat", "responses", "messages"))
def test_server_scoped_gateway_url_exposes_only_that_servers_tools(gateway: Gateway, surface: Surface) -> None:
    with _rig(gateway, surface) as rig, mcp_peer() as other_peer:
        other: Final = "oth" + uuid.uuid4().hex[:8]
        other_id: Final = register_mcp(rig.scenario, other_peer, other)
        key: Final = rig.scenario.key(object_permission={"mcp_servers": [rig.server_id, other_id]})
        response: Final = rig.send(key, [{**AUTO, "server_url": f"litellm_proxy/mcp/{rig.alias}"}])
        assert response.status_code == 200, response.text
        requests: Final = rig.upstream_tools()
        assert requests, "model was never called"
        assert all(rig.tool in names and not any(name.startswith(other) for name in names) for names in requests), (
            requests
        )
        assert _peer_add_calls(other_peer) == (), "unscoped server was called"
        assert [call["body"]["params"]["name"] for call in _peer_add_calls(rig.peer)] == ["add"]


def test_streaming_chat_executes_the_tool_once_and_streams_the_follow_up(gateway: Gateway) -> None:
    with _rig(gateway, "chat") as rig:
        key: Final = _granted_key(rig)
        response: Final = rig.send(key, [AUTO], stream=True)
        assert response.status_code == 200, response.text
        chunks: Final = tuple(
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        text: Final = "".join(
            str(chunk["choices"][0]["delta"].get("content") or "") for chunk in chunks if chunk.get("choices")
        )
        assert text == ANSWER, response.text
        assert [call["body"]["params"]["name"] for call in _peer_add_calls(rig.peer)] == ["add"]
        assert len(rig.upstream_tools()) == 2
