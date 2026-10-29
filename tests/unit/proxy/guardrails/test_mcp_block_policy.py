from typing import Final, Literal

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.caching import DualCache
from litellm.exceptions import BlockedPiiEntityError, GuardrailRaisedException, ModifyResponseException
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.mcp_block_policy import (
    MCPGuardrailBlock,
    UnsupportedMCPBlockPolicy,
    mcp_execution_surface,
    run_mcp_guardrail,
    validate_mcp_block_policy,
)
from litellm.proxy.utils import ProxyLogging
from litellm.types.guardrails import GuardrailEventHooks, MCPBlockAction
from litellm.types.utils import GenericGuardrailAPIInputs


def _error(kind: str) -> Exception:
    match kind:
        case "pii":
            return BlockedPiiEntityError("EMAIL_ADDRESS", guardrail_name="tool-policy")
        case "guardrail":
            return GuardrailRaisedException("tool-policy", "policy refusal", blocked_content=True)
        case "http":
            return HTTPException(403, detail={"error": "policy refusal"})
        case "passthrough":
            return ModifyResponseException("policy refusal", "test-model", {}, guardrail_name="tool-policy")
        case "outage":
            return GuardrailRaisedException("tool-policy", "service unavailable")
        case "timeout":
            return HTTPException(504, detail="timeout")
        case _:
            return ValueError("provider failure")


class _RefusingGuardrail(CustomGuardrail):
    def __init__(self, action: MCPBlockAction, hook: GuardrailEventHooks, error: Exception) -> None:
        super().__init__(guardrail_name="tool-policy", event_hook=hook, default_on=True)
        self.apply_mcp_block_policy(action)
        self.error = error

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: Logging | None = None,
    ) -> GenericGuardrailAPIInputs:
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("pii", "guardrail", "http", "passthrough"))
@pytest.mark.parametrize("hook", (GuardrailEventHooks.pre_mcp_call, GuardrailEventHooks.during_mcp_call))
async def test_shared_dispatch_propagates_each_refusal_form_with_the_blocking_callback_policy(
    kind: str, hook: GuardrailEventHooks
) -> None:
    error: Final = _error(kind)
    guardrail: Final = _RefusingGuardrail("raise", hook, error)
    logger: Final = ProxyLogging(user_api_key_cache=DualCache())
    data: Final = {"mcp_tool_name": "send_email", "mcp_arguments": {"body": "refused"}, "metadata": {}}
    user: Final = UserAPIKeyAuth()
    call: Final = (
        logger._execute_guardrail_hook(guardrail, "pre_call", data, user, "call_mcp_tool")
        if hook == GuardrailEventHooks.pre_mcp_call
        else logger._run_during_call_guardrail(guardrail, data, user, user, "call_mcp_tool")
    )
    with pytest.raises(MCPGuardrailBlock) as raised:
        await call
    assert raised.value.status_code == 400
    assert raised.value.guardrail_name == "tool-policy"
    assert raised.value.guardrail_mode == hook.value
    assert raised.value.__cause__ is error


async def _reject(error: Exception) -> None:
    raise error


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("outage", "timeout", "unknown"))
async def test_raise_does_not_reclassify_provider_failures_as_policy_refusals(kind: str) -> None:
    error: Final = _error(kind)
    guardrail: Final = _RefusingGuardrail("raise", GuardrailEventHooks.pre_mcp_call, error)
    with pytest.raises(type(error)) as raised:
        await run_mcp_guardrail(guardrail, _reject(error), GuardrailEventHooks.pre_mcp_call)
    assert raised.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("pii", "guardrail", "http", "passthrough"))
async def test_default_policy_preserves_the_original_refusal_for_existing_tool_error_handlers(kind: str) -> None:
    error: Final = _error(kind)
    guardrail: Final = _RefusingGuardrail("tool_error", GuardrailEventHooks.pre_mcp_call, error)
    with pytest.raises(type(error)) as raised:
        await run_mcp_guardrail(guardrail, _reject(error), GuardrailEventHooks.pre_mcp_call)
    assert raised.value is error


@pytest.mark.asyncio
async def test_ordinary_llm_hook_is_unchanged_even_when_mcp_policy_is_raise() -> None:
    error: Final = _error("pii")
    guardrail: Final = _RefusingGuardrail("raise", GuardrailEventHooks.pre_call, error)
    with pytest.raises(BlockedPiiEntityError) as raised:
        await run_mcp_guardrail(guardrail, _reject(error), GuardrailEventHooks.pre_call)
    assert raised.value is error


def test_unsupported_surfaces_reject_only_guardrails_selected_for_this_tool_call() -> None:
    guardrail: Final = CustomGuardrail(guardrail_name="selected", event_hook=GuardrailEventHooks.during_mcp_call)
    guardrail.apply_mcp_block_policy("raise")
    validate_mcp_block_policy((guardrail,), {"metadata": {}}, surface="streaming")
    for surface in ("streaming", "native"):
        with pytest.raises(UnsupportedMCPBlockPolicy) as raised:
            validate_mcp_block_policy((guardrail,), {"metadata": {"guardrails": ["selected"]}}, surface=surface)
        assert raised.value.status_code == 400
        assert raised.value.detail["guardrail_name"] == "selected"


@pytest.mark.asyncio
async def test_managed_scope_is_reset_after_a_refusal_and_rest_preserves_its_contract() -> None:
    guardrail: Final = _RefusingGuardrail("raise", GuardrailEventHooks.pre_mcp_call, _error("pii"))

    @mcp_execution_surface("managed")
    async def managed() -> None:
        validate_mcp_block_policy((guardrail,), {})
        await run_mcp_guardrail(guardrail, _reject(guardrail.error), GuardrailEventHooks.pre_mcp_call)

    with pytest.raises(MCPGuardrailBlock):
        await managed()
    with pytest.raises(UnsupportedMCPBlockPolicy):
        validate_mcp_block_policy((guardrail,), {})
    validate_mcp_block_policy((guardrail,), {}, surface="rest")


@pytest.mark.asyncio
async def test_concurrent_legacy_refusal_cannot_hide_a_raise_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    legacy: Final = _RefusingGuardrail("tool_error", GuardrailEventHooks.during_mcp_call, _error("http"))
    stopping: Final = _RefusingGuardrail("raise", GuardrailEventHooks.during_mcp_call, _error("pii"))
    monkeypatch.setattr(litellm, "callbacks", [legacy, stopping])
    logger: Final = ProxyLogging(user_api_key_cache=DualCache())
    data: Final = {"mcp_tool_name": "send_email", "mcp_arguments": {"body": "refused"}, "metadata": {}}
    with pytest.raises(MCPGuardrailBlock) as raised:
        await logger.during_call_hook(data, UserAPIKeyAuth(), "call_mcp_tool")
    assert raised.value.__cause__ is stopping.error
