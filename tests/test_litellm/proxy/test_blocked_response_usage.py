"""
Token usage on synthetic guardrail-blocked responses for the OpenAI-format
proxy endpoints (/v1/chat/completions and /v1/completions).

A post-call block replaces the LLM response with the violation message, but the
upstream call already consumed tokens. `_blocked_response_usage` reports that
real usage (carried on `ModifyResponseException.original_response`) rather than
zero; a pre-call block never invoked the LLM, so usage is zero.
"""

import pytest

import litellm
from litellm.proxy.proxy_server import _blocked_response_usage


def test_uses_original_response_usage():
    resp = litellm.ModelResponse()
    resp.usage = litellm.Usage(prompt_tokens=42, completion_tokens=7, total_tokens=49)

    usage = _blocked_response_usage(resp)

    assert usage.prompt_tokens == 42
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 49


def test_zero_usage_when_no_original_response():
    usage = _blocked_response_usage(None)

    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


@pytest.mark.asyncio
async def test_success_hook_attaches_original_response_on_block():
    """The unified guardrail's post-call success hook must attach the blocked
    LLM response to ModifyResponseException so its real usage isn't discarded."""
    from unittest.mock import AsyncMock, MagicMock, patch

    import litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail as ug
    from litellm.integrations.custom_guardrail import ModifyResponseException
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.utils import CallTypes

    response = litellm.ModelResponse()
    response.usage = litellm.Usage(prompt_tokens=15, completion_tokens=3, total_tokens=18)

    guardrail = MagicMock()
    guardrail.should_run_guardrail.return_value = True
    guardrail.guardrail_name = "rubrik"

    # The translation layer raises a block without pre-setting original_response.
    translation = MagicMock()
    translation.process_output_response = AsyncMock(
        side_effect=ModifyResponseException(
            message="blocked",
            model="gpt-4o",
            request_data={},
            guardrail_name="rubrik",
        )
    )

    unified = ug.UnifiedLLMGuardrails()
    user_api_key_dict = UserAPIKeyAuth(api_key="test", request_route="/chat/completions")
    data = {"guardrail_to_apply": guardrail, "model": "gpt-4o"}

    # Inject our translation for the inferred call type (the module global is
    # cached across tests, so patch it directly rather than the loader).
    with patch.object(
        ug,
        "endpoint_guardrail_translation_mappings",
        {
            CallTypes.acompletion: lambda: translation,
            CallTypes.completion: lambda: translation,
        },
    ):
        with pytest.raises(ModifyResponseException) as excinfo:
            await unified.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )

    assert excinfo.value.original_response is response
