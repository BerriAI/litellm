import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import ToolPermissionGuardrail
from litellm.proxy.guardrails.guardrail_initializers import initialize_tool_permission
from litellm.types.guardrails import LitellmParams

_RULES = [
    {"id": "allow_bash", "tool_name": "^Bash$", "decision": "allow"},
    {"id": "deny_read", "tool_name": "^Read$", "decision": "deny"},
]


def _tool(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": {}}}}


def _initialize(**tool_permission_params: object) -> ToolPermissionGuardrail:
    litellm_params = LitellmParams(
        guardrail="tool_permission", mode="pre_call", default_on=True, **tool_permission_params
    )
    return initialize_tool_permission(litellm_params, {"guardrail_name": "tool-guard"})


async def _pre_call(guardrail: ToolPermissionGuardrail, tool_names: list[str]) -> dict:
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(),
        cache=DualCache(),
        data={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "list the files"}],
            "tools": [_tool(name) for name in tool_names],
        },
        call_type="completion",
    )


@pytest.mark.asyncio
async def test_initialize_tool_permission_rewrite_mode_strips_only_the_tool_a_configured_rule_denies() -> None:
    guardrail = _initialize(rules=_RULES, default_action="allow", on_disallowed_action="rewrite")

    data = await _pre_call(guardrail, ["Bash", "Read", "Grep"])

    assert data["tools"] == [_tool("Bash"), _tool("Grep")]
    assert litellm.callbacks == [guardrail]


@pytest.mark.asyncio
async def test_initialize_tool_permission_block_mode_rejects_request_naming_the_matching_rule() -> None:
    guardrail = _initialize(rules=_RULES, default_action="allow")

    with pytest.raises(HTTPException) as blocked:
        await _pre_call(guardrail, ["Bash", "Read"])

    assert blocked.value.status_code == 400
    assert blocked.value.detail == {
        "error": "Violated guardrail policy",
        "detection_message": "Tool 'Read' denied by rule 'deny_read'",
    }


@pytest.mark.asyncio
async def test_initialize_tool_permission_without_rules_denies_every_tool_by_default_action() -> None:
    guardrail = _initialize()

    with pytest.raises(HTTPException) as blocked:
        await _pre_call(guardrail, ["Bash"])

    assert blocked.value.detail == {
        "error": "Violated guardrail policy",
        "detection_message": "Tool 'Bash' denied by default action",
    }


@pytest.mark.asyncio
async def test_initialize_tool_permission_without_rules_and_allow_default_keeps_every_tool() -> None:
    guardrail = _initialize(default_action="allow")

    data = await _pre_call(guardrail, ["Bash", "Read"])

    assert data["tools"] == [_tool("Bash"), _tool("Read")]
