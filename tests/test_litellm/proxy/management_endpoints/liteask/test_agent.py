import json
from collections import deque
from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.proxy.management_endpoints.liteask.agent import chat
from litellm.proxy.management_endpoints.liteask.catalog import Operation, Tool
from litellm.proxy.management_endpoints.liteask.dispatch import DispatchResult
from litellm.proxy.management_endpoints.liteask.models import LiteAskMessage, LiteAskResponse

_READ: Final = Tool(Operation("teams_list", "List teams", "GET", "/team/list", False), {"type": "object"})
_WRITE: Final = Tool(Operation("team_create", "Create a team", "POST", "/team/new", True), {"type": "object"})


def completion(name: str | None = None, arguments: str = "{}") -> DispatchResult:
    message: Final[JsonValue] = (
        {"content": "There is one team."}
        if name is None
        else {
            "tool_calls": [{"id": "call1", "type": "function", "function": {"name": name, "arguments": arguments}}],
        }
    )
    return DispatchResult(200, {"choices": [{"message": message}]})


class Transport:
    def __init__(self, results: tuple[DispatchResult, ...]) -> None:
        self.results = deque(results)
        self.requests: list[tuple[str, str, JsonValue]] = []

    async def send(
        self,
        method: str,
        path: str,
        body: JsonValue,
        query: tuple[tuple[str, str], ...],
    ) -> DispatchResult:
        self.requests.append((method, path, body))
        return self.results.popleft()


async def propose(tool: Tool, arguments: Mapping[str, JsonValue]) -> LiteAskResponse:
    return LiteAskResponse(message="Review " + tool.operation.name, result=TypeAdapter(JsonValue).validate_python(arguments))


@pytest.mark.asyncio
async def test_reads_use_gateway_and_redact_credentials_before_model_context() -> None:
    transport: Final = Transport(
        (
            completion("teams_list"),
            DispatchResult(200, {"teams": [{
                "team_id": "team", "api_key": "private-value", "metadata": {"provider": {"api_key": "b" * 64}}
            }]}),
            completion(),
        )
    )
    result: Final = await chat(
        model="configured-model",
        messages=(LiteAskMessage(role="user", content="List teams private-value"),),
        tools=(_READ,),
        send=transport.send,
        propose=propose,
        secrets=("private-value",),
    )
    assert result.message == "There is one team."
    assert [(method, path) for method, path, _ in transport.requests] == [
        ("POST", "/chat/completions"),
        ("GET", "/team/list"),
        ("POST", "/chat/completions"),
    ]
    assert "private-value" not in json.dumps(transport.requests)
    assert "b" * 64 not in json.dumps(transport.requests)
    assert "team_id" in json.dumps(transport.requests[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "arguments"), (
    (_WRITE, {"body": {"team_alias": "test"}}),
    (Tool(Operation("key_update", "Update a virtual key", "POST", "/key/update", True), {"type": "object"}),
     {"body": {"key": "a" * 64, "max_budget": 10}}),
))
async def test_mutations_only_propose_and_never_dispatch_writes(tool: Tool, arguments: dict[str, JsonValue]) -> None:
    transport: Final = Transport((completion(tool.operation.name, json.dumps(arguments)),))
    result: Final = await chat(
        model="configured-model",
        messages=(LiteAskMessage(role="user", content="Create team test"),),
        tools=(tool,),
        send=transport.send,
        propose=propose,
        secrets=(),
    )
    assert result.message == "Review " + tool.operation.name
    assert result.result == arguments
    assert [(method, path) for method, path, _ in transport.requests] == [("POST", "/chat/completions")]


@pytest.mark.asyncio
async def test_continued_model_request_preserves_tool_call_shape() -> None:
    first: Final[JsonValue] = {
        "choices": [{"message": {"tool_calls": [{"id": "call1", "function": {"name": "teams_list", "arguments": "{}"}}]}}],
    }
    transport: Final = Transport((DispatchResult(200, first), DispatchResult(200, {"teams": []}), completion()))
    await chat(
        model="configured-model",
        messages=(LiteAskMessage(role="user", content="List teams"),),
        tools=(_READ,),
        send=transport.send,
        propose=propose,
        secrets=(),
    )
    payload: Final = transport.requests[-1][2]
    assert isinstance(payload, dict)
    assert payload["model"] == "configured-model"
    assert payload["stream"] is False
    messages: Final = payload["messages"]
    assert isinstance(messages, list)
    assert messages[1:] == [
        {"role": "user", "content": "List teams"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call1", "type": "function", "function": {"name": "teams_list", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call1", "content": '{"status_code": 200, "data": {"teams": []}}'},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "arguments"), (("arbitrary_url", "{}"), ("team_create", "not json")))
async def test_unknown_or_malformed_operations_cannot_dispatch(tool: str, arguments: str) -> None:
    transport: Final = Transport((completion(tool, arguments),))
    result: Final = await chat(
        model="configured-model",
        messages=(LiteAskMessage(role="user", content="Do something"),),
        tools=(_WRITE,),
        send=transport.send,
        propose=propose,
        secrets=(),
    )
    assert result.proposal is None
    assert len(transport.requests) == 1
