import asyncio
import json
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import Final, Literal, TypeVar

import httpx
import pytest
import yaml
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import read_rows
from integration._support.mcp import (
    McpPeer,
    ScriptedTool,
    mcp_peer,
    official_client_outcomes,
    register_mcp,
    scripted_peer,
    text_result,
    tool_calls,
)
from integration._support.mcp_grants import create_toolset
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from integration._support.workers import worker_services
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses.tool_param import Mcp
from pydantic import JsonValue, TypeAdapter


pytestmark: Final = [pytest.mark.xdist_group("mcp_llm_private_gateway"), pytest.mark.shares_worker_state]


@pytest.fixture(scope="session")
def _llm_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Mapping[str, str]]]:
    directory: Final = tmp_path_factory.mktemp("mcp-llm-gateway")
    stock: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    config: Final = directory / "stored-responses.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                **stock,
                "general_settings": {**object_value(stock["general_settings"]), "store_prompts_in_spend_logs": True},
                "litellm_settings": {
                    **object_value(stock["litellm_settings"]),
                    "cost_margin_config": {"hosted_vllm": {"fixed_amount": FIXED_MARGIN}},
                },
            }
        )
    )
    with (
        worker_services(directory) as services,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, services, config=config) as private,
    ):
        yield private, services


@pytest.fixture
def gateway(_llm_gateway: tuple[Gateway, Mapping[str, str]]) -> Iterator[Gateway]:
    private, services = _llm_gateway
    with pytest.MonkeyPatch.context() as environment:
        for name, value in services.items():
            environment.setenv(name, value)
        yield private


Surface = Literal["chat", "responses", "messages", "messages_bridge"]
SURFACES: Final[tuple[Surface, ...]] = ("chat", "responses", "messages", "messages_bridge")
ADD: Final = {"a": 2, "b": 3}
ANSWER: Final = "the sum is 5"
GATEWAY_REF: Final = {"type": "mcp", "server_url": "litellm_proxy", "server_label": "litellm"}
AUTO: Final = {**GATEWAY_REF, "require_approval": "never"}
OUTAGE: Final = "bridge-outage"
DIRECT: Final = "answer-without-tools"
FIXED_MARGIN: Final = 0.25
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


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


@dataclass(frozen=True, slots=True)
class Turn:
    tool: str
    arguments: str
    answer: str


@dataclass(frozen=True, slots=True)
class Billing:
    first: tuple[int, int] = (10, 5)
    follow_up: tuple[int, int] = (10, 5)
    follow_up_fails: bool = False


EVEN_ROUNDS: Final = Billing()
FOLLOW_UP_OUTAGE: Final = "scripted follow-up outage"


def _fixed_turn(tool: str) -> Callable[[Mapping[str, JsonValue]], Turn]:
    return lambda _: Turn(tool, json.dumps(ADD), ANSWER)


def _echoing_turn(body: Mapping[str, JsonValue]) -> Turn:
    names: Final = _tool_names(body)
    return Turn(names[0] if names else "", json.dumps({"query": _prompt(body)}), _tool_result_text(body) or "")


def _prompt(body: Mapping[str, JsonValue]) -> str:
    inputs: Final = body.get("input")
    if isinstance(inputs, str):
        return inputs
    items: Final = inputs if isinstance(inputs, list) else body.get("messages")
    first: Final = items[0] if isinstance(items, list) and items else None
    return str(first["content"]) if isinstance(first, dict) and isinstance(first.get("content"), str) else ""


def _tool_result_text(body: Mapping[str, JsonValue]) -> str | None:
    inputs: Final = body.get("input")
    messages: Final = body.get("messages")
    items: Final = inputs if isinstance(inputs, list) else messages if isinstance(messages, list) else ()
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output":
            return str(item["output"])
        if item.get("role") == "tool":
            return str(item["content"])
        if (block := _tool_result_block(item)) is not None:
            return block
    return None


def _tool_result_block(item: Mapping[str, JsonValue]) -> str | None:
    content: Final = item.get("content")
    for block in content if isinstance(content, list) else ():
        if isinstance(block, dict) and block.get("type") == "tool_result":
            return str(block["content"])
    return None


def _item_events(index: int, item: Mapping[str, JsonValue]) -> tuple[dict[str, JsonValue], ...]:
    return (
        {"type": "response.output_item.added", "output_index": index, "item": dict(item)},
        {"type": "response.output_item.done", "output_index": index, "item": dict(item)},
    )


def _responses_stream(response: Mapping[str, JsonValue], items: Sequence[Mapping[str, JsonValue]]) -> Reply:
    item_events: Final = tuple(chain.from_iterable(_item_events(index, item) for index, item in enumerate(items)))
    events: Final = (
        {"type": "response.created", "response": {**response, "status": "in_progress"}},
        {"type": "response.in_progress", "response": {**response, "status": "in_progress"}},
        *item_events,
        {"type": "response.completed", "response": response},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(
            f"event: {event['type']}\ndata: {json.dumps({**event, 'sequence_number': number})}\n\n".encode()
            for number, event in enumerate(events)
        ),
    )


def _block_ids(message: JsonValue, kind: str, field: str) -> frozenset[str]:
    content: Final = message.get("content") if isinstance(message, dict) else None
    blocks: Final = content if isinstance(content, list) else []
    return frozenset(str(block[field]) for block in blocks if isinstance(block, dict) and block.get("type") == kind)


def _unanswered_tool_uses(body: Mapping[str, JsonValue]) -> frozenset[str]:
    messages: Final = body.get("messages")
    turns: Final = messages if isinstance(messages, list) else []
    followers: Final = [*turns[1:], None]
    return frozenset().union(
        *(
            _block_ids(turn, "tool_use", "id") - _block_ids(follower, "tool_result", "tool_use_id")
            for turn, follower in zip(turns, followers)
        )
    )


def _orphan_tool_results(body: Mapping[str, JsonValue]) -> frozenset[str]:
    messages: Final = body.get("messages")
    turns: Final = messages if isinstance(messages, list) else []
    leaders: Final = [None, *turns[:-1]]
    return frozenset().union(
        *(
            _block_ids(turn, "tool_result", "tool_use_id") - _block_ids(leader, "tool_use", "id")
            for turn, leader in zip(turns, leaders)
        )
    )


def _anthropic_invalid_request(message: str) -> Reply:
    error: Final = {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
    return Reply(status=400, body=json.dumps(error).encode())


def _anthropic_tool_result_rule(body: Mapping[str, JsonValue]) -> Reply | None:
    unanswered: Final = _unanswered_tool_uses(body)
    orphans: Final = _orphan_tool_results(body)
    if unanswered:
        return _anthropic_invalid_request(
            f"`tool_use` ids were found without `tool_result` blocks immediately after: {', '.join(sorted(unanswered))}"
        )
    if orphans:
        return _anthropic_invalid_request(
            f"unexpected `tool_use_id` found in `tool_result` blocks: {', '.join(sorted(orphans))}"
        )
    return None


def _model_double(
    plan: Callable[[Mapping[str, JsonValue]], Turn], billing: Billing = EVEN_ROUNDS, parallel: int = 1
) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET" and request.target.endswith("/models"):
            return _json({"object": "list", "data": []})
        body: Final = JSON_OBJECT.validate_json(request.body)
        if OUTAGE in _prompt(body):
            outage: Final = {"error": {"message": "scripted provider outage", "type": "server_error", "code": None}}
            return Reply(status=500, body=json.dumps(outage).encode())
        turn: Final = plan(body)
        answered: Final = _has_tool_result(body)
        if answered and billing.follow_up_fails:
            failure: Final = {"error": {"message": FOLLOW_UP_OUTAGE, "type": "server_error", "code": None}}
            return Reply(status=500, body=json.dumps(failure).encode())
        done: Final = answered or DIRECT in _prompt(body)
        calls: Final = range(1, parallel + 1)
        prompt_tokens, completion_tokens = billing.follow_up if answered else billing.first
        identity: Final = uuid.uuid4().hex[:12]
        usage: Final = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        if request.target.endswith("/chat/completions"):
            requested: Final = tuple(
                {"id": f"call_{call}", "type": "function", "function": {"name": turn.tool, "arguments": turn.arguments}}
                for call in calls
            )
            message: Final = (
                {"role": "assistant", "content": turn.answer}
                if done
                else {"role": "assistant", "content": None, "tool_calls": [*requested]}
            )
            finish: Final = "stop" if done else "tool_calls"
            if body.get("stream") is True:
                delta: Final = (
                    message
                    if done
                    else {**message, "tool_calls": [{**call, "index": index} for index, call in enumerate(requested)]}
                )
                chunk: Final = {
                    "id": f"chatcmpl-{identity}",
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
                    "id": f"chatcmpl-{identity}",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                    "usage": usage,
                }
            )
        if request.target.endswith("/messages"):
            if (rejected := _anthropic_tool_result_rule(body)) is not None:
                return rejected
            content: Final = (
                [{"type": "text", "text": turn.answer}]
                if done
                else [
                    {"type": "tool_use", "id": f"toolu_{call}", "name": turn.tool, "input": json.loads(turn.arguments)}
                    for call in calls
                ]
            )
            return _json(
                {
                    "id": f"msg_{identity}",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude",
                    "content": content,
                    "stop_reason": "end_turn" if done else "tool_use",
                    "stop_sequence": None,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
                }
            )
        assert request.target.endswith("/responses"), request.target
        items: Final[tuple[dict[str, JsonValue], ...]] = (
            (
                {
                    "type": "message",
                    "id": f"msg_{identity}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": turn.answer, "annotations": []}],
                },
            )
            if done
            else tuple(
                {
                    "type": "function_call",
                    "id": f"fc_{identity}_{call}",
                    "call_id": f"call_{identity}_{call}",
                    "name": turn.tool,
                    "arguments": turn.arguments,
                    "status": "completed",
                }
                for call in calls
            )
        )
        response: Final[dict[str, JsonValue]] = {
            "id": f"resp_{identity}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [*items],
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return _responses_stream(response, items) if body.get("stream") is True else _json(response)

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

    def send(
        self, key: str, tools: Sequence[Mapping[str, object]], prompt: str | None = None, **extra: object
    ) -> httpx.Response:
        text: Final = prompt or f"add {self.alias}"
        headers: Final = {"Authorization": f"Bearer {key}"}
        if self.surface == "chat":
            body: Final = {"model": self.model, "messages": [{"role": "user", "content": text}], "tools": list(tools)}
            return self.gateway.client.post("/v1/chat/completions", headers=headers, json={**body, **extra}, timeout=90)
        if self.surface == "responses":
            return self.gateway.client.post(
                "/v1/responses",
                headers=headers,
                json={"model": self.model, "input": text, "tools": list(tools), **extra},
                timeout=90,
            )
        return self.gateway.client.post(
            "/v1/messages",
            headers=headers,
            json={
                "model": self.model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": text}],
                "tools": list(tools),
                **extra,
            },
            timeout=90,
        )

    def upstream_tools(self) -> tuple[tuple[str, ...], ...]:
        return tuple(_tool_names(json.loads(request.body)) for request in self.wire.drain() if request.method == "POST")

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


PRICES: Final[Mapping[str, JsonValue]] = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002, "num_retries": 0}


@contextmanager
def _rig(
    gateway: Gateway,
    surface: Surface,
    *,
    upstream: Surface | None = None,
    billing: Billing = EVEN_ROUNDS,
    prices: Mapping[str, JsonValue] | None = None,
    parallel: int = 1,
) -> Iterator[Rig]:
    alias: Final = "llm" + uuid.uuid4().hex[:8]
    with (
        mcp_peer() as peer,
        wire_server(_model_double(_fixed_turn(f"{alias}-add"), billing, parallel)) as wire,
        gateway.scenario() as scenario,
    ):
        server_id: Final = register_mcp(scenario, peer, alias)
        model: Final = scenario.model(
            model=_upstream_model(upstream or surface), api_base=wire.url + "/v1", **(prices or {})
        )
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
        assert response.status_code in (200, 400, 401, 403), response.text


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
def test_toolset_gateway_url_serves_a_team_granted_toolset_to_a_key_without_its_own_grant(
    gateway: Gateway, surface: Surface
) -> None:
    with _rig(gateway, surface) as rig:
        register_mcp(rig.scenario, rig.peer, "open" + uuid.uuid4().hex[:8], allow_all_keys=True)
        toolset_name: Final = "ts" + uuid.uuid4().hex[:8]
        toolset_id: Final = create_toolset(rig.scenario, ((rig.server_id, "add"),), toolset_name=toolset_name)
        sibling_id: Final = create_toolset(rig.scenario, ((rig.server_id, "multiply"),))
        team_id: Final = rig.scenario.team(object_permission={"mcp_toolsets": [toolset_id, sibling_id]})
        key: Final = rig.scenario.key(team_id=team_id)
        response: Final = rig.send(key, [{**AUTO, "server_url": f"litellm_proxy/mcp/{toolset_name}"}])
        assert response.status_code == 200, response.text
        requests: Final = rig.upstream_tools()
        assert requests, "model was never called"
        assert all(names == (rig.tool,) for names in requests), requests
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


@pytest.mark.parametrize("surface", ("chat", "responses", "messages"))
def test_proxy_path_url_of_a_served_server_is_listed_and_executed_by_the_gateway(
    gateway: Gateway, surface: Surface
) -> None:
    with _rig(gateway, surface) as rig:
        server_url: Final = f"{str(gateway.client.base_url).rstrip('/')}/mcp/{rig.alias}"
        response: Final = rig.send(_granted_key(rig), [{**AUTO, "server_url": server_url}])
        assert response.status_code == 200, response.text
        requests: Final = rig.upstream_tools()
        assert len(requests) == 2 and all(rig.tool in names for names in requests), requests
        peer_calls: Final = tuple(JSON_OBJECT.validate_python(call["body"]) for call in _peer_add_calls(rig.peer))
        assert [object_value(call["params"])["name"] for call in peer_calls] == ["add"], peer_calls
        assert rig.final_text(JSON_OBJECT.validate_json(response.content)) == ANSWER, response.text


LOOKUP_PARAMETERS: Final = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
CALLER_FUNCTION_TOOLS: Final[Mapping[Surface, Mapping[str, object]]] = {
    "chat": {"type": "function", "function": {"name": "lookup", "parameters": LOOKUP_PARAMETERS}},
    "responses": {"type": "function", "name": "lookup", "parameters": LOOKUP_PARAMETERS},
    "messages": {"name": "lookup", "input_schema": LOOKUP_PARAMETERS},
}


@pytest.mark.parametrize("surface", ("chat", "responses", "messages"))
def test_tools_that_do_not_name_the_gateway_reach_the_provider_unchanged_and_never_the_peer(
    gateway: Gateway, surface: Surface
) -> None:
    with _rig(gateway, surface) as rig:
        remote_mcp_tools: Final = [
            {
                "type": "mcp",
                "server_label": "docs",
                "server_url": "https://mcp.example.com/sse",
                "require_approval": "never",
            },
            {
                "type": "mcp",
                "server_label": "repos",
                "server_url": f"https://mcp.example.com/mcp/unserved{uuid.uuid4().hex[:8]}",
                "require_approval": "never",
            },
        ]
        response: Final = rig.send(_granted_key(rig), [CALLER_FUNCTION_TOOLS[surface], *remote_mcp_tools])
        assert response.status_code == 200, response.text
        bodies: Final = tuple(
            JSON_OBJECT.validate_json(request.body) for request in rig.wire.drain() if request.method == "POST"
        )
        assert len(bodies) == 1, bodies
        forwarded: Final = bodies[0].get("tools")
        assert isinstance(forwarded, list) and len(forwarded) == 3, forwarded
        assert [tool for tool in forwarded if isinstance(tool, dict) and tool.get("type") == "mcp"] == remote_mcp_tools
        assert "lookup" in json.dumps([tool for tool in forwarded if tool not in remote_mcp_tools]), forwarded
        assert tool_calls(rig.peer.drain()) == (), "a tool that does not name the gateway reached the peer"


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


def _mcp_call_items(
    events: Sequence[Mapping[str, JsonValue]], kind: str
) -> tuple[tuple[JsonValue, dict[str, JsonValue]], ...]:
    announced: Final = tuple(
        (event["output_index"], object_value(event["item"])) for event in events if event["type"] == kind
    )
    return tuple((index, item) for index, item in announced if item.get("type") == "mcp_call")


def _add_call_arguments(call: dict[str, object]) -> JsonValue:
    return object_value(JSON_OBJECT.validate_python(call["body"])["params"])["arguments"]


def test_streaming_responses_opens_the_gateway_tool_call_empty_and_streams_its_arguments_once(gateway: Gateway) -> None:
    with _rig(gateway, "responses") as rig:
        key: Final = _granted_key(rig)
        response: Final = rig.send(key, [AUTO], stream=True)
        assert response.status_code == 200, response.text
        events: Final = tuple(
            JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )
        opened: Final = _mcp_call_items(events, "response.output_item.added")
        closed: Final = _mcp_call_items(events, "response.output_item.done")
        assert [(index, item["arguments"], item["status"]) for index, item in opened] == [
            (closed[0][0], "", "in_progress")
        ], opened
        streamed: Final = "".join(
            string_value(event["delta"])
            for event in events
            if event["type"] == "response.mcp_call_arguments.delta" and event["output_index"] == opened[0][0]
        )
        assert json.loads(streamed) == ADD, streamed
        assert string_value(opened[0][1]["arguments"]) + streamed == closed[0][1]["arguments"], (opened, closed)
        assert [_add_call_arguments(call) for call in _peer_add_calls(rig.peer)] == [ADD]
        assert len(rig.upstream_tools()) == 2


def test_streaming_chat_through_a_toolset_gateway_url_serves_a_team_key_without_its_own_grant(
    gateway: Gateway,
) -> None:
    with _rig(gateway, "chat") as rig:
        register_mcp(rig.scenario, rig.peer, "open" + uuid.uuid4().hex[:8], allow_all_keys=True)
        toolset_name: Final = "ts" + uuid.uuid4().hex[:8]
        toolset_id: Final = create_toolset(rig.scenario, ((rig.server_id, "add"),), toolset_name=toolset_name)
        key: Final = rig.scenario.key(team_id=rig.scenario.team(object_permission={"mcp_toolsets": [toolset_id]}))
        response: Final = rig.send(key, [{**AUTO, "server_url": f"litellm_proxy/mcp/{toolset_name}"}], stream=True)
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
        requests: Final = rig.upstream_tools()
        assert len(requests) == 2 and all(names == (rig.tool,) for names in requests), requests


@pytest.mark.parametrize("surface", ("chat", "responses", "messages"))
def test_toolset_gateway_url_gives_a_key_of_an_ungranted_team_no_tools_and_never_reaches_the_peer(
    gateway: Gateway, surface: Surface
) -> None:
    with _rig(gateway, surface) as rig:
        toolset_name: Final = "ts" + uuid.uuid4().hex[:8]
        create_toolset(rig.scenario, ((rig.server_id, "add"),), toolset_name=toolset_name)
        key: Final = rig.scenario.key(team_id=rig.scenario.team())
        response: Final = rig.send(key, [{**AUTO, "server_url": f"litellm_proxy/mcp/{toolset_name}"}])
        assert _peer_add_calls(rig.peer) == (), "denied caller reached the peer"
        assert all(rig.tool not in names for names in rig.upstream_tools()), rig.upstream_tools()
        assert response.status_code in (200, 400, 401, 403), response.text


def test_an_open_server_in_another_gateway_cannot_widen_this_gateways_tools(gateway: Gateway) -> None:
    with gateway_from_environment() as shared:
        with (
            mcp_peer() as foreign,
            shared.scenario() as foreign_scenario,
            mcp_peer() as local,
            gateway.scenario() as scenario,
        ):
            foreign_alias: Final = "foreign" + uuid.uuid4().hex[:8]
            foreign_id: Final = register_mcp(foreign_scenario, foreign, foreign_alias)
            created: Final = shared.post("/key/generate", {"object_permission": {"mcp_servers": [foreign_id]}})
            key: Final = created["key"]
            assert isinstance(key, str)
            foreign_scenario.cleanups.callback(shared.post, "/key/delete", {"keys": [key]})
            register_mcp(scenario, local, "open" + uuid.uuid4().hex[:8], allow_all_keys=True)
            listed, called = official_client_outcomes(shared, key, "/mcp", f"{foreign_alias}-add", ADD)
            assert set(listed.tools) == {
                f"{foreign_alias}-add", f"{foreign_alias}-multiply", f"{foreign_alias}-fail"
            }, listed.tools
            assert called.ok and called.text == str(ADD["a"] + ADD["b"]), called
            assert [call["body"]["params"]["name"] for call in _peer_add_calls(foreign)] == ["add"]
            assert _peer_add_calls(local) == ()
        deleted: Final = shared.request("GET", "/key/info", params={"key": key})
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["info"]["status"] == "deleted", deleted.text


FIRST_ROUND: Final = (11, 3)
FOLLOW_UP_ROUND: Final = (17, 7)


def _price(*rounds: tuple[int, int]) -> float:
    input_price, output_price = PRICES["input_cost_per_token"], PRICES["output_cost_per_token"]
    assert isinstance(input_price, float) and isinstance(output_price, float)
    return sum(prompt * input_price + completion * output_price for prompt, completion in rounds)


def _spend_rows(call_id: str, call_type: str, status: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            "SELECT request_id, spend, prompt_tokens, completion_tokens, session_id "
            'FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s AND call_type=%s AND status=%s',
            (call_id, call_type, status),
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )


def _session_call_types(session_id: str) -> frozenset[str]:
    rows: Final = read_rows('SELECT call_type FROM "LiteLLM_SpendLogs" WHERE session_id=%s', (session_id,))
    return frozenset(str(row["call_type"]) for row in rows)


def _message_text(message: Mapping[str, JsonValue]) -> str:
    content: Final = message.get("content")
    if isinstance(content, str):
        return content
    blocks: Final = content if isinstance(content, list) else []
    return "".join(str(block["text"]) for block in blocks if isinstance(block, dict) and block.get("type") == "text")


def _assistant_texts(body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    messages: Final = body.get("messages")
    turns: Final = messages if isinstance(messages, list) else []
    return tuple(_message_text(turn) for turn in turns if isinstance(turn, dict) and turn.get("role") == "assistant")


def test_nonstream_responses_auto_execute_logs_the_answer_and_bills_both_rounds(gateway: Gateway) -> None:
    billing: Final = Billing(FIRST_ROUND, FOLLOW_UP_ROUND)
    with _rig(gateway, "responses", billing=billing, prices=PRICES) as rig:
        response: Final = rig.send(_granted_key(rig), [AUTO])
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert rig.final_text(body) == ANSWER, response.text
        rows: Final = _spend_rows(response.headers["x-litellm-call-id"], "aresponses", "success")
        assert len(rows) == 1, rows
        row: Final = rows[0]
        assert row["request_id"] == body["id"], row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (
            FIRST_ROUND[0] + FOLLOW_UP_ROUND[0],
            FIRST_ROUND[1] + FOLLOW_UP_ROUND[1],
        ), row
        assert object_value(body["usage"])["input_tokens"] == row["prompt_tokens"], body
        assert float(str(row["spend"])) == pytest.approx(_price(FIRST_ROUND, FOLLOW_UP_ROUND)), row
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(float(str(row["spend"])))
        session_id: Final = string_value(row["session_id"])
        eventually(
            lambda: _session_call_types(session_id),
            lambda found: {"list_mcp_tools", "call_mcp_tool"} <= found,
            seconds=70,
        )


def test_nonstream_chat_auto_execute_logs_the_answer_and_bills_both_rounds(gateway: Gateway) -> None:
    billing: Final = Billing(FIRST_ROUND, FOLLOW_UP_ROUND)
    with _rig(gateway, "chat", billing=billing, prices=PRICES) as rig:
        response: Final = rig.send(_granted_key(rig), [AUTO])
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert rig.final_text(body) == ANSWER, response.text
        rows: Final = _spend_rows(response.headers["x-litellm-call-id"], "acompletion", "success")
        assert len(rows) == 1, rows
        row: Final = rows[0]
        assert row["request_id"] == body["id"], row
        assert (row["prompt_tokens"], row["completion_tokens"]) == (
            FIRST_ROUND[0] + FOLLOW_UP_ROUND[0],
            FIRST_ROUND[1] + FOLLOW_UP_ROUND[1],
        ), row
        assert object_value(body["usage"])["prompt_tokens"] == row["prompt_tokens"], body
        assert float(str(row["spend"])) == pytest.approx(_price(FIRST_ROUND, FOLLOW_UP_ROUND)), row
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(float(str(row["spend"])))


def test_nonstream_chat_auto_execute_without_a_tool_call_bills_the_one_call_it_made(gateway: Gateway) -> None:
    billing: Final = Billing(FIRST_ROUND, FOLLOW_UP_ROUND)
    with _rig(gateway, "chat", billing=billing, prices=PRICES) as rig:
        response: Final = rig.send(_granted_key(rig), [AUTO], prompt=f"{DIRECT} {rig.alias}")
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert rig.final_text(body) == ANSWER, response.text
        assert len(rig.upstream_tools()) == 1 and _peer_add_calls(rig.peer) == ()
        rows: Final = _spend_rows(response.headers["x-litellm-call-id"], "acompletion", "success")
        assert len(rows) == 1 and rows[0]["request_id"] == body["id"], rows
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == FIRST_ROUND, rows
        assert float(str(rows[0]["spend"])) == pytest.approx(_price(FIRST_ROUND)), rows
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(float(str(rows[0]["spend"])))


def test_nonstream_chat_auto_execute_charges_the_fixed_margin_once(gateway: Gateway) -> None:
    billing: Final = Billing(FIRST_ROUND, FOLLOW_UP_ROUND)
    with _rig(gateway, "chat", upstream="messages_bridge", billing=billing, prices=PRICES) as rig:
        response: Final = rig.send(_granted_key(rig), [AUTO])
        assert response.status_code == 200, response.text
        assert len(rig.upstream_tools()) == 2
        rows: Final = _spend_rows(response.headers["x-litellm-call-id"], "acompletion", "success")
        assert len(rows) == 1, rows
        assert float(str(rows[0]["spend"])) == pytest.approx(_price(FIRST_ROUND, FOLLOW_UP_ROUND) + FIXED_MARGIN), rows
        assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(float(str(rows[0]["spend"])))


def test_nonstream_chat_auto_execute_whose_calls_both_come_from_the_response_cache_bills_nothing(
    gateway: Gateway,
) -> None:
    billing: Final = Billing(FIRST_ROUND, FOLLOW_UP_ROUND)
    with _rig(gateway, "chat", billing=billing, prices=PRICES) as rig:
        key: Final = _granted_key(rig)
        paid: Final = rig.send(key, [AUTO])
        assert paid.status_code == 200, paid.text
        assert len(rig.upstream_tools()) == 2 and len(_peer_add_calls(rig.peer)) == 1
        cached: Final = rig.send(key, [{**AUTO, "server_label": "relabeled"}])
        assert cached.status_code == 200, cached.text
        assert rig.final_text(JSON_OBJECT.validate_json(cached.content)) == ANSWER, cached.text
        assert rig.upstream_tools() == () and len(_peer_add_calls(rig.peer)) == 1
        rows: Final = _spend_rows(cached.headers["x-litellm-call-id"], "acompletion", "success")
        assert all(float(str(row["spend"])) == 0.0 for row in rows), rows
        assert float(cached.headers["x-litellm-response-cost"]) == 0.0


@pytest.mark.parametrize(("surface", "call_type"), (("responses", "aresponses"), ("chat", "acompletion")))
def test_nonstream_failed_follow_up_bills_round_one_on_the_failure_row(
    gateway: Gateway, surface: Surface, call_type: str
) -> None:
    billing: Final = Billing(FIRST_ROUND, FOLLOW_UP_ROUND, follow_up_fails=True)
    with _rig(gateway, surface, billing=billing, prices=PRICES) as rig:
        response: Final = rig.send(_granted_key(rig), [AUTO])
        assert response.status_code == 500 and FOLLOW_UP_OUTAGE in response.text, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        rows: Final = _spend_rows(call_id, call_type, "failure")
        assert len(rows) == 1, rows
        assert float(str(rows[0]["spend"])) == pytest.approx(_price(FIRST_ROUND)), rows
        assert (
            read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s AND call_type=%s AND status=%s',
                (call_id, call_type, "success"),
            )
            == []
        )


def _tool_rows(call_id: str, expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE call_type=%s AND strpos(request_id, %s) = 1',
            ("call_mcp_tool", call_id),
        ),
        lambda found: len(found) >= expected,
        seconds=70,
    )


@pytest.mark.parametrize(
    ("surface", "stream"),
    (("chat", False), ("chat", True), ("messages", False), ("responses", True), ("responses", False)),
    ids=("chat", "chat-stream", "messages", "responses-stream", "responses"),
)
def test_every_auto_executed_tool_call_lands_its_own_spend_row_under_the_request_call_id(
    gateway: Gateway, surface: Surface, stream: bool
) -> None:
    with _rig(gateway, surface, parallel=2) as rig:
        response: Final = rig.send(_granted_key(rig), [AUTO], stream=stream)
        assert response.status_code == 200, response.text
        assert [_add_call_arguments(call) for call in _peer_add_calls(rig.peer)] == [ADD, ADD]
        call_id: Final = response.headers["x-litellm-call-id"]
        request_ids: Final = frozenset(str(row["request_id"]) for row in _tool_rows(call_id, 2))
        assert len(request_ids) == 2 and all(request_id.startswith(f"{call_id}_") for request_id in request_ids), (
            call_id,
            request_ids,
        )


def test_bridged_responses_auto_execute_answers_every_tool_use_with_its_result(gateway: Gateway) -> None:
    with _rig(gateway, "responses", upstream="messages") as rig:
        response: Final = rig.send(_granted_key(rig), [AUTO])
        assert response.status_code == 200, response.text
        assert rig.final_text(response.json()) == ANSWER, response.text
        assert [call["body"]["params"]["name"] for call in _peer_add_calls(rig.peer)] == ["add"]


def test_bridged_next_turn_replays_the_mcp_turn_answer(gateway: Gateway) -> None:
    with _rig(gateway, "responses", upstream="messages") as rig:
        key: Final = _granted_key(rig)
        first: Final = rig.send(key, [AUTO])
        assert first.status_code == 200, first.text
        first_id: Final = string_value(JSON_OBJECT.validate_json(first.content)["id"])
        eventually(
            lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (first_id,)),
            lambda found: len(found) == 1,
            seconds=70,
        )
        rig.wire.drain()
        second: Final = rig.send(key, [AUTO], previous_response_id=first_id, cache={"no-cache": True})
        assert second.status_code == 200, second.text
        opening: Final = next(
            JSON_OBJECT.validate_json(request.body) for request in rig.wire.drain() if request.method == "POST"
        )
        assert ANSWER in _assistant_texts(opening), opening


Bridge = Literal["chat", "responses", "messages"]
BRIDGES: Final[tuple[Bridge, ...]] = ("chat", "responses")
Client = Literal["sync", "async"]
HOOK_ECHO: Final = "bridge-echo:"
HOOK_PROBE: Final = "bridge-probe"
HOOK_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    texts = list(inputs.get("texts") or [])\n'
    '    function = inputs.get("tools", [{}])[0].get("function", {})\n'
    "    for text in texts:\n"
    f'        if "{HOOK_PROBE}" in text:\n'
    f'            return block("{HOOK_ECHO}" + json_stringify('
    '{"description": function.get("description"), "parameters": function.get("parameters")}))\n'
    "    return allow()\n"
)
LOOKUP: Final[tuple[str, dict[str, JsonValue]]] = (
    "Look up one record",
    {"type": "object", "properties": {"query": {"type": "string"}}},
)
REPORT: Final[tuple[str, dict[str, JsonValue]]] = (
    "Write one report",
    {"type": "object", "properties": {"query": {"type": "string"}, "format": {}}},
)
COLD: Final[tuple[str, dict[str, JsonValue]]] = (
    "",
    {"type": "object", "properties": {}, "additionalProperties": False},
)
RELOAD_FAST: Final = {"PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "5"}
CALL_ID: Final = "x-litellm-call-id"
T = TypeVar("T")
Definition = tuple[str, str, JsonValue]
Echo = tuple[JsonValue, JsonValue]


def _served(listing: tuple[str, Mapping[str, JsonValue]]) -> Echo:
    return listing[0], {**listing[1], "additionalProperties": False}


@dataclass(frozen=True, slots=True)
class Hooked:
    proxy: Gateway
    sink: Wire


@pytest.fixture(scope="module")
def hooked(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Hooked]:
    directory: Final = tmp_path_factory.mktemp("bridge-hooks")
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    with wire_server(lambda _: _json({"flagged": False, "session_id": "scripted"})) as sink:
        echo: Final = {"guardrail": "custom_code", "mode": "pre_mcp_call", "default_on": True, "custom_code": HOOK_CODE}
        pillar: Final = {
            "guardrail": "pillar",
            "mode": ["pre_call", "pre_mcp_call"],
            "default_on": True,
            "api_key": "sk-pillar-" + uuid.uuid4().hex,
            "api_base": sink.url,
            "on_flagged_action": "monitor",
        }
        guardrails: Final = [
            {"guardrail_name": "bridge-echo-" + uuid.uuid4().hex[:8], "litellm_params": echo},
            {"guardrail_name": "bridge-sink-" + uuid.uuid4().hex[:8], "litellm_params": pillar},
        ]
        path: Final = directory / "config.yaml"
        path.write_text(yaml.safe_dump({**base, "guardrails": guardrails}))
        with (
            gateway_from_environment() as gateway,
            owned_proxy(gateway, directory, RELOAD_FAST, config=path, workers=2) as proxy,
        ):
            yield Hooked(proxy, sink)


@dataclass(frozen=True, slots=True)
class BridgeRig:
    hooked: Hooked
    scenario: Scenario
    peer: McpPeer
    wire: Wire
    alias: str
    server_id: str
    model: str
    bridge: Bridge

    def tool(self, name: str) -> str:
        return f"{self.alias}-{name}"

    def names(self) -> frozenset[str]:
        return frozenset(("lookup", "report"))

    def mcp(self, name: str) -> Mcp:
        return {**AUTO_MCP, "allowed_tools": [self.tool(name)]}

    def url(self, path: str) -> str:
        return str(self.hooked.proxy.client.base_url).rstrip("/") + path

    def post(self, key: str, prompt: str, tools: Sequence[Mcp], **extra: object) -> httpx.Response:
        headers: Final = {"Authorization": f"Bearer {key}"}
        if self.bridge == "chat":
            body: Final = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "tools": list(tools)}
            return httpx.post(self.url("/v1/chat/completions"), headers=headers, json={**body, **extra}, timeout=90)
        if self.bridge == "responses":
            body_r: Final = {"model": self.model, "input": prompt, "tools": list(tools), **extra}
            return httpx.post(self.url("/v1/responses"), headers=headers, json=body_r, timeout=90)
        body_m: Final = {
            "model": self.model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": prompt}],
            "tools": list(tools),
            **extra,
        }
        return httpx.post(self.url("/v1/messages"), headers=headers, json=body_m, timeout=90)

    def upstream_by_prompt(self) -> Mapping[str, tuple[tuple[Definition, ...], ...]]:
        bodies: Final = tuple(
            JSON_OBJECT.validate_json(request.body) for request in self.wire.drain() if request.method == "POST"
        )
        prompts: Final = frozenset(_prompt(body) for body in bodies)
        return {prompt: tuple(_definitions(body) for body in bodies if _prompt(body) == prompt) for prompt in prompts}

    def peer_calls(self) -> tuple[tuple[str, JsonValue], ...]:
        return tuple(_peer_call(call) for call in tool_calls(self.peer.drain()))

    def hook_messages(self, marker: str) -> tuple[str, ...]:
        posted: Final = tuple(
            JSON_OBJECT.validate_json(request.body) for request in self.hooked.sink.drain() if request.method == "POST"
        )
        contents: Final = tuple(content for payload in posted for content in _contents(payload))
        return tuple(content for content in contents if _synthetic(content, marker))


AUTO_MCP: Final[Mcp] = {
    "type": "mcp",
    "server_label": "litellm",
    "server_url": "litellm_proxy",
    "require_approval": "never",
}


def _peer_call(call: Mapping[str, object]) -> tuple[str, JsonValue]:
    params: Final = object_value(object_value(JSON_VALUE.validate_python(call["body"]))["params"])
    return str(params["name"]), params["arguments"]


def _contents(payload: Mapping[str, JsonValue]) -> Iterator[str]:
    messages: Final = payload.get("messages")
    for message in messages if isinstance(messages, list) else ():
        if isinstance(message, dict):
            yield str(message.get("content"))


def _function(tool: JsonValue) -> dict[str, JsonValue] | None:
    if not isinstance(tool, dict):
        return None
    function: Final = tool.get("function", tool)
    return function if isinstance(function, dict) else None


def _definitions(body: Mapping[str, JsonValue]) -> tuple[Definition, ...]:
    tools: Final = body.get("tools")
    functions: Final = tuple(_function(tool) for tool in tools) if isinstance(tools, list) else ()
    return tuple(
        (str(function["name"]), str(function.get("description", "")), function.get("parameters"))
        for function in functions
        if function is not None
    )


def _uniform(upstream: Mapping[str, tuple[tuple[Definition, ...], ...]]) -> Mapping[str, tuple[Definition, ...] | None]:
    return {
        prompt: rounds[0] if all(definitions == rounds[0] for definitions in rounds) else None
        for prompt, rounds in upstream.items()
    }


def _strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
        return
    children: Final = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    for child in children:
        yield from _strings(child)


def _echoed(value: JsonValue) -> Echo:
    carrier: Final = next((text for text in _strings(value) if HOOK_ECHO in text), None)
    assert carrier is not None, value
    payload: Final = carrier.split(HOOK_ECHO, 1)[1]
    end: Final = json.JSONDecoder().raw_decode(payload)[1]
    echoed: Final = JSON_OBJECT.validate_json(payload[:end])
    return echoed.get("description"), echoed.get("parameters")


@contextmanager
def _bridge_rig(hooked: Hooked, bridge: Bridge) -> Generator[BridgeRig, None, None]:
    alias: Final = "brg" + uuid.uuid4().hex[:8]
    lookup: Final = ScriptedTool(
        "lookup",
        lambda params: text_result("found:" + json.dumps(JSON_OBJECT.validate_python(params)["arguments"])),
        description=LOOKUP[0],
        input_schema=LOOKUP[1],
    )
    report: Final = ScriptedTool(
        "report", lambda _: text_result("reported"), description=REPORT[0], input_schema=REPORT[1]
    )
    with (
        scripted_peer(lookup, report) as peer,
        wire_server(_model_double(_echoing_turn)) as wire,
        hooked.proxy.scenario() as scenario,
    ):
        server_id: Final = register_mcp(scenario, peer, alias)
        model: Final = scenario.model(model=_upstream_model(bridge), api_base=wire.url + "/v1")
        rig: Final = BridgeRig(hooked, scenario, peer, wire, alias, server_id, model, bridge)
        eventually(
            lambda: tuple(_on_worker(hooked.proxy, lambda client: _master_listing(rig, client)) for _ in range(6)),
            lambda seen: len({pid for pid, _ in seen}) >= 2 and all(names == rig.names() for _, names in seen),
            seconds=45,
        )
        peer.drain()
        hooked.sink.drain()
        yield rig


def _bridge_key(rig: BridgeRig) -> str:
    return rig.scenario.key(object_permission={"mcp_servers": [rig.server_id]})


@dataclass(frozen=True, slots=True)
class Seen:
    response_id: str
    call_id: str
    text: str


def _ask_sync(rig: BridgeRig, key: str, prompt: str, tools: Sequence[Mcp], stream: bool) -> Seen:
    sdk: Final = OpenAI(base_url=rig.url("/v1"), api_key=key, max_retries=0, timeout=90)
    if rig.bridge == "responses":
        if stream:
            raw_events: Final = sdk.responses.with_raw_response.create(
                model=rig.model, input=prompt, tools=tools, stream=True
            )
            completed: Final = next(
                event.response for event in raw_events.parse() if event.type == "response.completed"
            )
            return Seen(completed.id, raw_events.headers[CALL_ID], completed.output_text)
        raw_response: Final = sdk.responses.with_raw_response.create(model=rig.model, input=prompt, tools=tools)
        response: Final = raw_response.parse()
        return Seen(response.id, raw_response.headers[CALL_ID], response.output_text)
    messages: Final[list[ChatCompletionMessageParam]] = [{"role": "user", "content": prompt}]
    extra: Final = {"tools": list(tools)}
    if stream:
        raw_chunks: Final = sdk.chat.completions.with_raw_response.create(
            model=rig.model, messages=messages, stream=True, extra_body=extra
        )
        parts: Final = tuple(
            (chunk.id, chunk.choices[0].delta.content or "") for chunk in raw_chunks.parse() if chunk.choices
        )
        return Seen(parts[0][0], raw_chunks.headers[CALL_ID], "".join(text for _, text in parts))
    raw_completion: Final = sdk.chat.completions.with_raw_response.create(
        model=rig.model, messages=messages, extra_body=extra
    )
    completion: Final = raw_completion.parse()
    return Seen(completion.id, raw_completion.headers[CALL_ID], completion.choices[0].message.content or "")


async def _ask_async(rig: BridgeRig, key: str, prompt: str, tools: Sequence[Mcp], stream: bool) -> Seen:
    sdk: Final = AsyncOpenAI(base_url=rig.url("/v1"), api_key=key, max_retries=0, timeout=90)
    if rig.bridge == "responses":
        if stream:
            raw_events: Final = await sdk.responses.with_raw_response.create(
                model=rig.model, input=prompt, tools=tools, stream=True
            )
            completed: Final = [
                event.response async for event in raw_events.parse() if event.type == "response.completed"
            ]
            return Seen(completed[0].id, raw_events.headers[CALL_ID], completed[0].output_text)
        raw_response: Final = await sdk.responses.with_raw_response.create(model=rig.model, input=prompt, tools=tools)
        response: Final = raw_response.parse()
        return Seen(response.id, raw_response.headers[CALL_ID], response.output_text)
    messages: Final[list[ChatCompletionMessageParam]] = [{"role": "user", "content": prompt}]
    extra: Final = {"tools": list(tools)}
    if stream:
        raw_chunks: Final = await sdk.chat.completions.with_raw_response.create(
            model=rig.model, messages=messages, stream=True, extra_body=extra
        )
        parts: Final = [
            (chunk.id, chunk.choices[0].delta.content or "") async for chunk in raw_chunks.parse() if chunk.choices
        ]
        return Seen(parts[0][0], raw_chunks.headers[CALL_ID], "".join(text for _, text in parts))
    raw_completion: Final = await sdk.chat.completions.with_raw_response.create(
        model=rig.model, messages=messages, extra_body=extra
    )
    completion: Final = raw_completion.parse()
    return Seen(completion.id, raw_completion.headers[CALL_ID], completion.choices[0].message.content or "")


def _ask(rig: BridgeRig, key: str, prompt: str, tools: Sequence[Mcp], stream: bool, client: Client) -> Seen:
    if client == "async":
        return asyncio.run(_ask_async(rig, key, prompt, tools, stream))
    return _ask_sync(rig, key, prompt, tools, stream)


def _synthetic(content: str, marker: str) -> bool:
    return content.startswith("Tool: lookup\n") and marker in content


def _on_worker(gateway: Gateway, act: Callable[[httpx.Client], T]) -> tuple[int, T]:
    with httpx.Client(base_url=str(gateway.client.base_url), timeout=30) as client:
        summary: Final = client.get("/debug/memory/summary", headers={"Authorization": f"Bearer {gateway.key}"})
        assert summary.status_code == 200, summary.text
        pid: Final = JSON_OBJECT.validate_json(summary.content)["worker_pid"]
        assert isinstance(pid, int), summary.text
        return pid, act(client)


def _both_workers(gateway: Gateway) -> frozenset[int]:
    return eventually(
        lambda: frozenset(_on_worker(gateway, lambda _: None)[0] for _ in range(6)), lambda pids: len(pids) >= 2
    )


def _master_listing(rig: BridgeRig, client: httpx.Client) -> frozenset[str]:
    headers: Final = {"x-litellm-api-key": rig.hooked.proxy.key}
    response: Final = client.get("/mcp-rest/tools/list", headers=headers, params={"server_id": rig.server_id})
    tools: Final = JSON_OBJECT.validate_json(response.content).get("tools") if response.status_code == 200 else None
    return frozenset(str(object_value(tool)["name"]) for tool in tools) if isinstance(tools, list) else frozenset()


def _direct_probe(rig: BridgeRig, key: str, name: str, client: httpx.Client) -> Echo:
    body: Final = {"server_id": rig.server_id, "name": rig.tool(name), "arguments": {"query": HOOK_PROBE}}
    response: Final = client.post("/mcp-rest/tools/call", headers={"x-litellm-api-key": key}, json=body)
    return _echoed(JSON_VALUE.validate_json(response.content))


def _spend_row(key: str, call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT api_key, call_type, status, cache_hit FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (call_id,)
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    assert rows[0]["api_key"] == sha256(key.encode()).hexdigest(), rows
    return rows[0]


@pytest.mark.parametrize("client", ("sync", "async"))
@pytest.mark.parametrize("stream", (False, True), ids=("plain", "stream"))
@pytest.mark.parametrize("bridge", BRIDGES)
def test_bridge_hook_sees_the_definition_of_the_tool_filtered_for_that_request(
    hooked: Hooked, bridge: Bridge, stream: bool, client: Client
) -> None:
    with _bridge_rig(hooked, bridge) as rig:
        key: Final = _bridge_key(rig)
        marker: Final = "m" + uuid.uuid4().hex
        probe: Final = f"{marker} {HOOK_PROBE}"
        found: Final = _ask(rig, key, marker, [rig.mcp("lookup")], stream, client)
        assert found.text == "found:" + json.dumps({"query": marker}), found
        blocked: Final = _ask(rig, key, probe, [rig.mcp("lookup")], stream, client)
        assert _echoed(blocked.text) == _served(LOOKUP), blocked
        expected: Final = ((rig.tool("lookup"), *_served(LOOKUP)),)
        upstream: Final = rig.upstream_by_prompt()
        assert set(upstream) == {marker, probe} and all(
            definitions == expected for definitions in upstream[marker] + upstream[probe]
        ), upstream
        assert rig.peer_calls() == (("lookup", {"query": marker}),)
        assert rig.hook_messages(marker) == (f"Tool: lookup\nArguments: {dict(query=marker)}",)
        assert _spend_row(key, found.call_id)["status"] == "success"


@pytest.mark.parametrize("bridge", BRIDGES)
def test_direct_call_after_bridge_only_discovery_stays_cold(hooked: Hooked, bridge: Bridge) -> None:
    with _bridge_rig(hooked, bridge) as rig:
        key: Final = _bridge_key(rig)
        workers: Final = _both_workers(rig.hooked.proxy)
        bridged: Final = _ask(rig, key, HOOK_PROBE, [rig.mcp("lookup")], False, "sync")
        assert _echoed(bridged.text) == _served(LOOKUP), bridged
        direct: Final = eventually(
            lambda: tuple(
                _on_worker(rig.hooked.proxy, lambda client: _direct_probe(rig, key, "lookup", client)) for _ in range(6)
            ),
            lambda seen: frozenset(pid for pid, _ in seen) == workers,
        )
        assert all(echo == COLD for _, echo in direct) and frozenset(pid for pid, _ in direct) == workers, (
            direct,
            workers,
        )
        assert rig.peer_calls() == ()


@pytest.mark.parametrize("bridge", BRIDGES)
def test_concurrent_requests_of_one_key_with_different_allowed_tools_each_see_their_own_definition(
    hooked: Hooked, bridge: Bridge
) -> None:
    with _bridge_rig(hooked, bridge) as rig:
        key: Final = _bridge_key(rig)
        prompts: Final = {name: f"{name} {uuid.uuid4().hex} {HOOK_PROBE}" for name in ("lookup", "report")}

        def ask(name: str) -> Seen:
            return _ask(rig, key, prompts[name], [rig.mcp(name)], False, "sync")

        with ThreadPoolExecutor(2) as pool:
            lookup, report = pool.map(ask, ("lookup", "report"))
        assert (_echoed(lookup.text), _echoed(report.text)) == (_served(LOOKUP), _served(REPORT)), (lookup, report)
        upstream: Final = rig.upstream_by_prompt()
        assert _uniform(upstream) == {
            prompts["lookup"]: ((rig.tool("lookup"), *_served(LOOKUP)),),
            prompts["report"]: ((rig.tool("report"), *_served(REPORT)),),
        }, upstream
        assert rig.peer_calls() == ()


@pytest.mark.parametrize("bridge", BRIDGES)
def test_provider_outage_reaches_the_caller_and_never_the_peer_or_the_hooks(hooked: Hooked, bridge: Bridge) -> None:
    with _bridge_rig(hooked, bridge) as rig:
        key: Final = _bridge_key(rig)
        prompt: Final = f"{OUTAGE} {uuid.uuid4().hex}"
        response: Final = rig.post(key, prompt, [rig.mcp("lookup")])
        assert response.status_code == 500, response.text
        upstream: Final = rig.upstream_by_prompt()
        assert set(upstream) == {prompt} and all(
            definitions == ((rig.tool("lookup"), *_served(LOOKUP)),) for definitions in upstream[prompt]
        ), upstream
        assert rig.peer_calls() == ()
        assert rig.hook_messages(prompt) == ()


@pytest.mark.parametrize("bridge", BRIDGES)
def test_identical_nonstream_repeat_is_a_cache_hit_without_new_model_peer_or_hook_traffic(
    hooked: Hooked, bridge: Bridge
) -> None:
    with _bridge_rig(hooked, bridge) as rig:
        key: Final = _bridge_key(rig)
        marker: Final = "m" + uuid.uuid4().hex
        first: Final = _ask(rig, key, marker, [rig.mcp("lookup")], False, "sync")
        assert first.text == "found:" + json.dumps({"query": marker}), first
        assert set(rig.upstream_by_prompt()) == {marker} and rig.peer_calls() == (("lookup", {"query": marker}),)
        assert rig.hook_messages(marker) == (f"Tool: lookup\nArguments: {dict(query=marker)}",)
        assert _spend_row(key, first.call_id)["cache_hit"] != "True"
        repeat: Final = _ask(rig, key, marker, [rig.mcp("lookup")], False, "sync")
        assert repeat.text == first.text, (first, repeat)
        assert (rig.upstream_by_prompt(), rig.peer_calls(), rig.hook_messages(marker)) == ({}, (), ()), repeat
        assert _spend_row(key, repeat.call_id)["cache_hit"] == "True"


def test_messages_bridge_hook_sees_the_definition_the_request_served(hooked: Hooked) -> None:
    with _bridge_rig(hooked, "messages") as rig:
        key: Final = _bridge_key(rig)
        marker: Final = "m" + uuid.uuid4().hex
        probe: Final = f"{marker} {HOOK_PROBE}"
        found: Final = rig.post(key, marker, [rig.mcp("lookup")])
        assert found.status_code == 200, found.text
        blocked: Final = rig.post(key, probe, [rig.mcp("lookup")])
        assert blocked.status_code == 200, blocked.text
        assert _echoed(JSON_VALUE.validate_json(blocked.content)) == _served(LOOKUP), blocked.text
        assert rig.peer_calls() == (("lookup", {"query": marker}),)
        assert rig.hook_messages(marker) == (f"Tool: lookup\nArguments: {dict(query=marker)}",)
