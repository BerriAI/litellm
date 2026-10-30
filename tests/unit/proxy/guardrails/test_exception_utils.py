from typing import Final

import pytest
from fastapi import HTTPException

from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException
from litellm.proxy.guardrails.exception_utils import (
    MCPGuardrailBlock,
    get_mcp_guardrail_block,
    guardrail_provider_error,
    mcp_guardrail_block_context,
    raise_mcp_guardrail_block,
)


@pytest.mark.parametrize(
    "error, is_block",
    [
        (GuardrailRaisedException(message="denied", blocked_content=True), True),
        (BlockedPiiEntityError(entity_type="SSN"), True),
        (HTTPException(400, {"error": "content denied"}), True),
        (HTTPException(403, {"error": "content denied"}), True),
        (HTTPException(422, {"error": "content denied"}), True),
        (GuardrailRaisedException(message="backend unreachable"), False),
        (HTTPException(401, "unauthorized"), False),
        (HTTPException(429, "rate limited"), False),
        (HTTPException(503, "unavailable"), False),
        (TimeoutError("provider timed out"), False),
        (guardrail_provider_error(HTTPException(403, "upstream access denied")), False),
    ],
)
def test_mcp_policy_preserves_original_error_and_distinguishes_provider_failures(
    error: Exception,
    is_block: bool,
) -> None:
    original_text: Final = str(error)
    with pytest.raises(type(error)) as raised:
        with mcp_guardrail_block_context("content-policy", "raise", "call_mcp_tool", "pre_call"):
            raise error
    assert raised.value is error
    assert str(raised.value) == original_text
    assert get_mcp_guardrail_block(error) == (MCPGuardrailBlock("content-policy", "raise") if is_block else None)


def test_mcp_policy_does_not_treat_chained_http_failure_as_content_verdict() -> None:
    with pytest.raises(HTTPException) as raised:
        with mcp_guardrail_block_context("content-policy", "raise", "call_mcp_tool", "during_call"):
            raise HTTPException(400, "provider rejected request") from ValueError("invalid provider response")
    assert get_mcp_guardrail_block(raised.value) is None
    assert isinstance(raised.value.__cause__, ValueError)


@pytest.mark.parametrize(
    "call_type, hook_type",
    [
        ("completion", "pre_call"),
        ("completion", "during_call"),
        ("list_mcp_tools", "pre_call"),
        ("call_mcp_tool", "post_call"),
    ],
)
def test_mcp_policy_does_not_attach_to_other_guardrail_hooks(call_type: str, hook_type: str) -> None:
    with pytest.raises(HTTPException) as raised:
        with mcp_guardrail_block_context("content-policy", "raise", call_type, hook_type):
            raise HTTPException(400, "content denied")
    assert get_mcp_guardrail_block(raised.value) is None


def test_outer_guardrail_context_does_not_overwrite_selected_deployment_policy() -> None:
    with (
        pytest.raises(HTTPException) as raised,
        mcp_guardrail_block_context("outer", "tool_error", "call_mcp_tool", "pre_call"),
        mcp_guardrail_block_context("selected", "raise", "call_mcp_tool", "pre_call"),
    ):
        raise HTTPException(400, "content denied")
    assert get_mcp_guardrail_block(raised.value) == MCPGuardrailBlock("selected", "raise")


@pytest.mark.parametrize("status", [400, 403, 422])
def test_managed_mcp_raise_policy_returns_http_400_with_guardrail_identity(status: int) -> None:
    with pytest.raises(HTTPException) as refusal:
        with mcp_guardrail_block_context("input-filter", "raise", "call_mcp_tool", "pre_call"):
            raise HTTPException(status, {"error": "blocked argument"})
    with pytest.raises(HTTPException) as response:
        raise_mcp_guardrail_block(refusal.value, stream=False)
    assert response.value.status_code == 400
    assert response.value.detail == {"error": {"error": "blocked argument"}, "guardrail_name": "input-filter"}
    assert response.value.__cause__ is refusal.value


@pytest.mark.parametrize("stream, action", [(True, "raise"), (False, "tool_error")])
def test_streams_and_tool_error_policy_keep_the_existing_tool_result_path(stream, action) -> None:
    with pytest.raises(HTTPException) as refusal:
        with mcp_guardrail_block_context("input-filter", action, "call_mcp_tool", "pre_call"):
            raise HTTPException(400, "blocked argument")
    assert raise_mcp_guardrail_block(refusal.value, stream=stream) is None


def test_provider_failure_does_not_become_mcp_http_400() -> None:
    error: Final = GuardrailRaisedException(message="guardrail service unavailable")
    with pytest.raises(GuardrailRaisedException):
        with mcp_guardrail_block_context("input-filter", "raise", "call_mcp_tool", "pre_call"):
            raise error
    assert raise_mcp_guardrail_block(error, stream=False) is None
