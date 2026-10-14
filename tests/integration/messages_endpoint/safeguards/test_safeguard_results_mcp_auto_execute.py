"""An MCP tool the proxy runs on the caller's behalf (``require_approval: never``) never reaches the classifier. The
loop's inner turns cost no classifier call, and a final reply that hands no tool use back to the agent is available
with no verdicts."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Final

import pytest
from integration._support.client import object_value
from integration._support.mcp import McpPeer, mcp_peer, tool_calls
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    BASH_TOOL,
    CHAT_MODEL,
    Responder,
    Rig,
    auto_mode_body,
    available,
    chat_backend,
    json_body,
    nonce,
    request_json,
    safeguards_rig,
    stream_results,
    stream_tool_use_ids,
    tool_id,
    tool_use_ids,
)
from integration._support.wire import Reply, Request
from pydantic import JsonValue, TypeAdapter

pytestmark = pytest.mark.timeout(graceful_stop_seconds() + 240)

MCP_ALIAS: Final = "sgmcp"
AUTO_EXECUTED: Final[Mapping[str, JsonValue]] = {
    "type": "mcp",
    "server_url": "litellm_proxy",
    "server_label": "litellm",
    "require_approval": "never",
}
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@pytest.fixture(scope="module")
def peer() -> Iterator[McpPeer]:
    with mcp_peer() as value:
        yield value


@pytest.fixture(scope="module")
def rig(peer: McpPeer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    def mcp_servers(wire_url: str) -> Mapping[str, JsonValue]:
        return {"mcp_servers": {MCP_ALIAS: _JSON.validate_python(peer.registration())}}

    with safeguards_rig(tmp_path_factory.mktemp("safeguards-mcp"), top_level=mcp_servers) as value:
        yield value


def carries_a_tool_result(request: Request) -> bool:
    messages: Final = request_json(request).get("messages")
    return isinstance(messages, list) and any(object_value(message).get("role") == "tool" for message in messages)


def mcp_tool_then_text(request_nonce: str) -> Responder:
    mcp_call: Final = chat_backend(
        f"call_mcp_{request_nonce}", stream=False, name=f"{MCP_ALIAS}-add", arguments={"a": 2, "b": 3}
    )
    final_text: Final = chat_backend(tool_id(request_nonce), stream=False, text="the sum is 5", finish_reason="stop")

    def respond(request: Request) -> Reply:
        return final_text(request) if carries_a_tool_result(request) else mcp_call(request)

    return respond


def executed_tools(peer: McpPeer) -> tuple[JsonValue, ...]:
    return tuple(
        object_value(object_value(_JSON.validate_python(call["body"]))["params"])["name"]
        for call in tool_calls(peer.drain())
    )


def non_stream_turn(rig: Rig, body: Mapping[str, JsonValue]) -> tuple[tuple[str, ...], JsonValue | None]:
    response: Final = rig.post(body)
    assert response.status_code == 200, response.text
    reply: Final = json_body(response)
    return tool_use_ids(reply), reply.get("safeguard_results")


def stream_turn(rig: Rig, body: Mapping[str, JsonValue]) -> tuple[tuple[str, ...], JsonValue | None]:
    status, raw = rig.stream(body)
    assert status == 200, raw
    return stream_tool_use_ids(raw), stream_results(raw)


TURNS: Final[Mapping[str, Callable[[Rig, Mapping[str, JsonValue]], tuple[tuple[str, ...], JsonValue | None]]]] = {
    "non_stream": non_stream_turn,
    "stream": stream_turn,
}


@pytest.mark.parametrize("mode", tuple(TURNS))
def test_auto_executed_mcp_tool_costs_no_classifier_call(rig: Rig, peer: McpPeer, mode: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, mcp_tool_then_text(request_nonce))
    peer.drain()
    ids, results = TURNS[mode](
        rig, auto_mode_body(CHAT_MODEL, request_nonce, tools=[dict(AUTO_EXECUTED), dict(BASH_TOOL)])
    )
    assert executed_tools(peer) == ("add",)
    assert ids == (), ids
    assert results == available({}), results
    received: Final = rig.received(request_nonce)
    assert rig.classifier_calls(received) == (), [request.target for request in received]
    assert len(rig.backend_calls(received)) == 2, [request.target for request in received]
