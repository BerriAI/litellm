from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.factory import NormalizedToolCall
from litellm.litellm_core_utils.prompt_templates.server_tools import (
    ServerToolRoute,
    append_server_reference,
    continue_server_tools,
    inject_server_tools,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.memory.policy import MemoryIdentity


@pytest.mark.parametrize("route", ["acompletion", "aresponses", "anthropic_messages"])
def test_memory_injection_preserves_client_tools_output_constraints_and_streaming(route: ServerToolRoute) -> None:
    original: Final = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Remember my preference"}],
        "input": "Remember my preference",
        "tools": [{"name": "application_tool"}],
        "tool_choice": {"type": "function", "name": "application_tool"},
        "max_tokens": 1,
        "max_output_tokens": 1,
        "stream": True,
    }
    prepared: Final = inject_server_tools(
        original,
        route,
        ({"name": "memory_search", "description": "Search", "parameters": {"type": "object"}},),
        "Use memory throughout the task",
    )
    output_field: Final = "max_output_tokens" if route == "aresponses" else "max_tokens"
    assert prepared[output_field] == 1
    assert prepared["stream"] is True
    assert prepared["tool_choice"] == original["tool_choice"]
    assert prepared["tools"][0] == original["tools"][0]
    assert len(prepared["tools"]) == 2
    final: Final = append_server_reference(original, route, "Stored preference")
    assert final["tools"] == original["tools"]
    assert final["tool_choice"] == original["tool_choice"]
    assert final[output_field] == 1
    assert final["stream"] is True
    assert len(original["messages"]) == 1


def test_anthropic_continuation_preserves_signed_thinking_and_matches_tool_result() -> None:
    content: Final = [
        {"type": "thinking", "thinking": "Checking a fact", "signature": "provider-signature"},
        {"type": "tool_use", "id": "call-1", "name": "memory_read", "input": {"memory_id": "memory-1"}},
    ]
    calls: Final[list[NormalizedToolCall]] = [
        {"id": "call-1", "name": "memory_read", "arguments": {"memory_id": "memory-1"}}
    ]
    result: Final = continue_server_tools({"messages": []}, "anthropic_messages", {"content": content}, calls, ["fact"])
    assert result["messages"] == [
        {"role": "assistant", "content": content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1", "content": '"fact"'}]},
    ]


def test_responses_continuation_keeps_reasoning_and_function_call_output() -> None:
    output: Final = [
        {"type": "reasoning", "id": "reason-1", "encrypted_content": "opaque-provider-data"},
        {"type": "function_call", "call_id": "call-1", "name": "memory_read", "arguments": "{}"},
    ]
    calls: Final[list[NormalizedToolCall]] = [{"id": "call-1", "name": "memory_read", "arguments": {}}]
    result: Final = continue_server_tools({"input": "question"}, "aresponses", {"output": output}, calls, ["fact"])
    assert result["input"] == [
        {"role": "user", "content": "question"},
        *output,
        {"type": "function_call_output", "call_id": "call-1", "output": '"fact"'},
    ]


def test_missing_tool_result_is_rejected_before_continuation() -> None:
    calls: Final[list[NormalizedToolCall]] = [{"id": "call-1", "name": "memory_read", "arguments": {}}]
    with pytest.raises(ValueError, match="match every tool call"):
        continue_server_tools({}, "acompletion", {}, calls, [])


def test_private_namespaces_follow_authenticated_key_and_organization() -> None:
    owner: Final = MemoryIdentity.from_auth(
        UserAPIKeyAuth(token="a" * 64, user_id="owner", team_id="team", org_id="org")
    )
    sibling: Final = MemoryIdentity.from_auth(
        UserAPIKeyAuth(token="b" * 64, user_id="owner", team_id="team", org_id="org")
    )
    elsewhere: Final = MemoryIdentity.from_auth(
        UserAPIKeyAuth(token="a" * 64, user_id="owner", team_id="team", org_id="other")
    )
    assert owner.namespace("key") != sibling.namespace("key")
    assert owner.namespace("key") != elsewhere.namespace("key")
    assert owner.namespace("user") == sibling.namespace("user")
    assert owner.namespace("user") != elsewhere.namespace("user")
    assert owner.namespace("team") == sibling.namespace("team")
