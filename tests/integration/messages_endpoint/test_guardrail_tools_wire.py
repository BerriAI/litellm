import json
import uuid
from collections.abc import Sequence
from typing import Final

import httpx
import pytest
from integration._support.anthropic_sse import message_json
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.upstream import delete_scenario, register_scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.cost_calculation.cost_tracking_case import JsonResponse
from pydantic import JsonValue, TypeAdapter

_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])
_CUSTOM: Final = {
    "name": "Read",
    "description": "Read private data",
    "input_schema": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ReadInput",
        "type": "object",
        "properties": {},
    },
    "defer_loading": True,
    "cache_control": {"type": "ephemeral"},
    "strict": True,
    "input_examples": [{}],
    "allowed_callers": ["direct"],
}
_NATIVE: Final = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 1},
    {"type": "mcp_toolset", "mcp_server_name": "github"},
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    {"type": "code_execution_20250825", "name": "code_execution"},
]


def _guard_reply(request: Request) -> Reply:
    if request.target.startswith("/echo/"):
        return Reply(body=json.dumps({"action": "NONE"}).encode())
    body: Final = object_value(json.loads(request.body))
    tools: Final = _OBJECTS.validate_python(body.get("tools") or [])
    edited: Final = [
        {**tool, "function": {**object_value(tool["function"]), "description": "Read [MASKED] data"}}
        if isinstance(tool.get("function"), dict) and object_value(tool["function"]).get("name") == "Read"
        else tool
        for tool in tools
    ]
    return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "tools": edited}).encode())


def _delete_guardrail(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("DELETE", f"/guardrails/{identity}")
    assert response.status_code == 200, response.text


def _guardrail(gateway: Gateway, scenario: Scenario, guard: Wire, mode: str) -> None:
    created: Final = gateway.post(
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": f"tools-{uuid.uuid4().hex}",
                "litellm_params": {
                    "guardrail": "generic_guardrail_api",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": f"{guard.url}/{mode}",
                    "api_key": "synthetic-guardrail-key",
                },
            }
        },
    )
    scenario.cleanups.callback(_delete_guardrail, gateway, string_value(created["guardrail_id"]))


@pytest.mark.parametrize("mode", ("echo", "edit"))
@pytest.mark.parametrize("native", ([], _NATIVE), ids=("custom", "mixed-native"))
def test_guardrail_preserves_anthropic_tools_except_explicit_edits(
    gateway: Gateway, mode: str, native: Sequence[JsonValue]
) -> None:
    tools: Final = [*native, _CUSTOM, {**_CUSTOM, "name": "Untouched"}]
    expected: Final = [
        *native,
        {**_CUSTOM, "description": "Read [MASKED] data"} if mode == "edit" else _CUSTOM,
        {**_CUSTOM, "name": "Untouched"},
    ]
    with wire_server(_guard_reply) as guard, gateway.scenario() as scenario:
        handle: Final = register_scenario(
            uuid.uuid4().hex,
            JsonResponse(
                content_type="application/json",
                body=object_value(json.loads(message_json("msg_tools", "claude-haiku-4-5", "Checked."))),
            ),
            control_url=gateway.upstream_url,
        )
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            model="anthropic/claude-haiku-4-5", api_base=handle.api_base(), api_key="synthetic"
        )
        _guardrail(gateway, scenario, guard, mode)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 32,
                "tools": tools,
                "messages": [{"role": "user", "content": "Read a file"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"] == [{"type": "text", "text": "Checked."}], response.text
        calls: Final = guard.drain()
        assert len(calls) == 1, calls
        scanned: Final = _OBJECTS.validate_python(object_value(json.loads(calls[0].body))["tools"])
        assert [
            object_value(tool["function"])["name"] for tool in scanned if isinstance(tool.get("function"), dict)
        ] == [
            *(object_value(tool).get("name", object_value(tool).get("mcp_server_name")) for tool in native),
            "Read",
            "Untouched",
        ]
        captured: Final = httpx.get(f"{gateway.upstream_url}/__observations", trust_env=False).json()["requests"]
        assert len(captured) == 1, captured
        assert captured[0]["body"]["tools"] == expected
