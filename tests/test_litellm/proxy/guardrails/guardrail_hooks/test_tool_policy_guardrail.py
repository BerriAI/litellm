"""
Unit tests for ToolPolicyGuardrail.
"""

import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.proxy.guardrails.guardrail_hooks.tool_policy.tool_policy_guardrail import \
    ToolPolicyGuardrail
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
            {"type": "function", "function": {"name": name}}
            for name in tool_names
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
