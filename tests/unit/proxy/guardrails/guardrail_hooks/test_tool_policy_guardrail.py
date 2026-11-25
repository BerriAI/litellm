"""
Unit tests for ToolPolicyGuardrail.
"""

from collections.abc import Mapping
from itertools import chain
from types import SimpleNamespace
from typing import Any, Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.llms.anthropic.pass_through.adapters.transformation import truncate_tool_name
from litellm.proxy.db import tool_registry_writer
from litellm.proxy.guardrails.guardrail_hooks.tool_policy.tool_policy_guardrail import (
    ToolPolicyGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks


@pytest.fixture
def guardrail():
    return ToolPolicyGuardrail()


# --- helpers ---


def _tool_request_inputs(tool_names: list) -> dict:
    return {
        "tools": [
            {"type": "function", "function": {"name": name, "description": ""}}
            for name in tool_names
        ]
    }


def _tool_response_inputs(tool_names: list) -> dict:
    return {
        "tool_calls": [
            {"type": "function", "function": {"name": name}} for name in tool_names
        ]
    }


# --- tests ---


def test_guardrail_supports_pre_and_post_call(guardrail):
    hooks = guardrail.supported_event_hooks
    assert GuardrailEventHooks.pre_call in hooks
    assert GuardrailEventHooks.post_call in hooks


@pytest.mark.asyncio
async def test_no_tools_in_request_passes_through(guardrail):
    inputs: Any = {"tools": []}
    result = await guardrail.apply_guardrail(
        inputs=inputs, request_data={}, input_type="request"
    )
    assert result is inputs


@pytest.mark.asyncio
async def test_no_tool_calls_in_response_passes_through(guardrail):
    inputs: Any = {"tool_calls": []}
    result = await guardrail.apply_guardrail(
        inputs=inputs, request_data={}, input_type="response"
    )
    assert result is inputs


def _registry_mock(policy_map: dict):
    """Return a mock registry with is_initialized=True and get_effective_policies returning policy_map."""
    reg = MagicMock()
    reg.is_initialized.return_value = True
    reg.get_effective_policies.return_value = policy_map
    return reg


@pytest.mark.asyncio
async def test_untrusted_tools_pass_through(guardrail):
    policy_map = {"search": "untrusted", "read_file": "trusted"}
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(policy_map),
    ):
        inputs: Any = _tool_request_inputs(["search", "read_file"])
        result = await guardrail.apply_guardrail(
            inputs=inputs, request_data={}, input_type="request"
        )
    assert result is inputs


@pytest.mark.asyncio
async def test_blocked_tool_in_request_raises_http_exception(guardrail):
    policy_map = {"dangerous_tool": "blocked"}
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(policy_map),
    ):
        inputs: Any = _tool_request_inputs(["dangerous_tool"])
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs, request_data={}, input_type="request"
            )
    assert exc_info.value.status_code == 400
    assert "dangerous_tool" in exc_info.value.detail["blocked_tools"]


@pytest.mark.asyncio
async def test_blocked_tool_in_response_raises_http_exception(guardrail):
    policy_map = {"exfil_tool": "blocked"}
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(policy_map),
    ):
        inputs: Any = _tool_response_inputs(["exfil_tool"])
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs, request_data={}, input_type="response"
            )
    assert exc_info.value.status_code == 400
    assert "exfil_tool" in exc_info.value.detail["blocked_tools"]


@pytest.mark.asyncio
async def test_mixed_blocked_and_allowed_raises_for_blocked(guardrail):
    policy_map = {"safe_tool": "trusted", "bad_tool": "blocked"}
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(policy_map),
    ):
        inputs: Any = _tool_request_inputs(["safe_tool", "bad_tool"])
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs, request_data={}, input_type="request"
            )
    blocked = exc_info.value.detail["blocked_tools"]
    assert "bad_tool" in blocked
    assert "safe_tool" not in blocked


@pytest.mark.asyncio
async def test_tool_not_in_db_passes_through(guardrail):
    """When registry returns no policy (or empty), tools are not blocked."""
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock({}),
    ):
        inputs: Any = _tool_request_inputs(["unknown_tool"])
        result = await guardrail.apply_guardrail(
            inputs=inputs, request_data={}, input_type="request"
        )
    assert result is inputs


@pytest.mark.asyncio
async def test_registry_not_initialized_passes_through(guardrail):
    """When registry is not initialized, no tools are blocked (empty policy map)."""
    reg = MagicMock()
    reg.is_initialized.return_value = False
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=reg,
    ):
        inputs: Any = _tool_request_inputs(["any_tool"])
        result = await guardrail.apply_guardrail(
            inputs=inputs, request_data={}, input_type="request"
        )
    assert result is inputs
    reg.get_effective_policies.assert_not_called()


@pytest.mark.asyncio
async def test_response_tool_calls_as_objects(guardrail):
    """tool_calls that are objects (not dicts) with .function.name should work."""
    policy_map = {"obj_tool": "blocked"}
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock(policy_map),
    ):
        fn = MagicMock()
        fn.name = "obj_tool"
        tc = MagicMock()
        tc.function = fn
        inputs: Any = {"tool_calls": [tc]}
        with pytest.raises(HTTPException):
            await guardrail.apply_guardrail(
                inputs=inputs, request_data={}, input_type="response"
            )


@pytest.mark.asyncio
async def test_blocked_tool_loaded_by_a_replayed_tool_search_raises_next_to_allowed_tools(guardrail):
    request_data = {
        "tools": [{"type": "function", "function": {"name": "get_weather"}}, {"type": "tool_search"}],
        "messages": [
            {"role": "user", "content": "clean up the disk"},
            {
                "role": "assistant",
                "content": None,
                "provider_specific_fields": {
                    "tool_search_items": [
                        {"type": "tool_search_call", "call_id": "ts_1", "execution": "client"},
                        {
                            "type": "tool_search_output",
                            "call_id": "ts_1",
                            "tools": [{"type": "function", "name": "run_shell", "parameters": {"type": "object"}}],
                        },
                    ]
                },
            },
        ],
    }
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock({"get_weather": "untrusted", "run_shell": "blocked"}),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=_tool_request_inputs(["get_weather"]),
                request_data=request_data,
                input_type="request",
            )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["blocked_tools"] == ["run_shell"]


@pytest.mark.asyncio
async def test_policy_on_a_qualified_member_name_over_64_characters_still_blocks(guardrail):
    from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

    namespace: Final = "mcp__codex_apps__github_enterprise_connector_prod"
    tools: Final = [
        {
            "type": "namespace",
            "name": namespace,
            "description": "GitHub",
            "tools": [{"type": "function", "name": "delete_repository", "parameters": {"type": "object"}}],
        }
    ]
    blocked: Final = f"{namespace}__delete_repository"
    chat_forms: Final = LiteLLMCompletionResponsesConfig.responses_tools_to_chat_forms(tools)
    inputs: Final = {"tools": list(chain.from_iterable(form.chat_tools for form in chat_forms))}
    with patch(
        "litellm.proxy.db.tool_registry_writer.get_tool_policy_registry",
        return_value=_registry_mock({blocked: "blocked"}),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs=inputs,
                request_data={"user_api_key_request_route": "/v1/responses", "tools": tools},
                input_type="request",
            )
    assert exc_info.value.detail["blocked_tools"] == [blocked]


async def _install_tool_policies(monkeypatch: pytest.MonkeyPatch, input_policies: Mapping[str, str]) -> None:
    prisma_client = MagicMock()
    prisma_client.db.litellm_tooltable.find_many = AsyncMock(
        return_value=[
            SimpleNamespace(tool_name=name, input_policy=policy, output_policy="untrusted")
            for name, policy in input_policies.items()
        ]
    )
    prisma_client.db.litellm_objectpermissiontable.find_many = AsyncMock(return_value=[])
    registry = tool_registry_writer.ToolPolicyRegistry()
    await registry.sync_tool_policy_from_db(prisma_client)
    monkeypatch.setattr(tool_registry_writer, "_tool_policy_registry", registry)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [["drop_database"], {"name": "drop_database"}], ids=["list", "dict"])
async def test_function_tool_name_that_is_not_a_string_is_left_for_the_provider_to_refuse(
    guardrail: ToolPolicyGuardrail, name: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _install_tool_policies(monkeypatch, {"get_weather": "untrusted"})
    inputs: Final = {"tools": [{"type": "function", "function": {"name": name}}]}
    assert await guardrail.apply_guardrail(inputs=inputs, request_data={}, input_type="request") == inputs


_LONG_MCP_TOOL: Final = "mcp__client-tools__q__automations__open_automation_builder_prefill"


def _long_tool_messages_request() -> dict[str, object]:
    return {
        "user_api_key_request_route": "/v1/messages",
        "tools": [{"name": _LONG_MCP_TOOL, "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "open the builder"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_name",
    [_LONG_MCP_TOOL, truncate_tool_name(_LONG_MCP_TOOL)],
    ids=["full_name", "64_char_alias"],
)
async def test_blocked_long_messages_tool_raises_under_its_full_name_or_its_alias(
    guardrail: ToolPolicyGuardrail, blocked_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _install_tool_policies(monkeypatch, {blocked_name: "blocked"})
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs=_tool_request_inputs([truncate_tool_name(_LONG_MCP_TOOL)]),
            request_data=_long_tool_messages_request(),
            input_type="request",
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["blocked_tools"] == [blocked_name]
