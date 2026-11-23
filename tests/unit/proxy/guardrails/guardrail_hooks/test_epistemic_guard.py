"""Unit tests for the Epistemic Guard post-call guardrail."""

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.epistemic_guard import (
    EpistemicGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
    initialize_guardrail,
)
from litellm.types.guardrails import (
    GuardrailEventHooks,
    LitellmParams,
    SupportedGuardrailIntegrations,
)


def test_registry_exports():
    key = SupportedGuardrailIntegrations.EPISTEMIC_GUARD.value
    assert key == "epistemic_guard"
    assert key in guardrail_initializer_registry
    assert key in guardrail_class_registry
    assert guardrail_class_registry[key] is EpistemicGuardrail


def test_initialize_guardrail_defaults():
    params = LitellmParams(
        guardrail="epistemic_guard",
        mode="post_call",
        on_fail="rewrite",
        verify_output_values=True,
    )
    guardrail_cfg = {
        "guardrail_name": "my-epistemic-guard",
        "litellm_params": {
            "guardrail": "epistemic_guard",
            "mode": "post_call",
            "on_fail": "rewrite",
            "verify_output_values": True,
        },
    }
    instance = initialize_guardrail(params, guardrail_cfg)
    assert isinstance(instance, EpistemicGuardrail)
    assert instance.guardrail_name == "my-epistemic-guard"
    assert instance.on_fail == "rewrite"
    assert instance.verify_output_values is True
    assert instance.get_supported_event_hooks() == [GuardrailEventHooks.post_call]


@pytest.mark.asyncio
async def test_pre_call_is_passthrough():
    guardrail = EpistemicGuardrail(guardrail_name="eg-test", on_fail="block")
    inputs = {"texts": ["Please delete report.txt"]}
    result = await guardrail.apply_guardrail(
        inputs=inputs,
        request_data={"messages": [{"role": "user", "content": "Delete report.txt"}]},
        input_type="request",
    )
    assert result["texts"] == ["Please delete report.txt"]


@pytest.mark.asyncio
async def test_ungrounded_claim_blocked():
    pytest.importorskip("epistemic_guard")
    guardrail = EpistemicGuardrail(guardrail_name="eg-block", on_fail="block")
    request_data = {
        "messages": [
            {"role": "user", "content": "Delete report.txt for me."},
        ]
    }
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["I deleted report.txt for you."]},
            request_data=request_data,
            input_type="response",
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["guardrail"] == "eg-block"
    assert exc_info.value.detail["unsupported_claims"]


@pytest.mark.asyncio
async def test_ungrounded_claim_rewritten():
    pytest.importorskip("epistemic_guard")
    guardrail = EpistemicGuardrail(guardrail_name="eg-rewrite", on_fail="rewrite")
    request_data = {
        "messages": [
            {"role": "user", "content": "Send the email to alice@example.com."},
        ]
    }
    result = await guardrail.apply_guardrail(
        inputs={"texts": ["I emailed alice@example.com."]},
        request_data=request_data,
        input_type="response",
    )
    assert result["texts"][0].startswith("[UNGROUNDED CLAIM BLOCKED]")


@pytest.mark.asyncio
async def test_ungrounded_claim_log_only():
    pytest.importorskip("epistemic_guard")
    guardrail = EpistemicGuardrail(guardrail_name="eg-log", on_fail="log_only")
    request_data = {
        "messages": [
            {"role": "user", "content": "Delete report.txt."},
        ]
    }
    result = await guardrail.apply_guardrail(
        inputs={"texts": ["I deleted report.txt."]},
        request_data=request_data,
        input_type="response",
    )
    assert result["texts"] == ["I deleted report.txt."]


@pytest.mark.asyncio
async def test_grounded_claim_passes():
    pytest.importorskip("epistemic_guard")
    guardrail = EpistemicGuardrail(guardrail_name="eg-pass", on_fail="block")
    request_data = {
        "messages": [
            {"role": "user", "content": "Delete report.txt."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "delete_file",
                            "arguments": '{"path": "report.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "delete_file",
                "content": '{"ok": true, "deleted": "report.txt"}',
            },
        ]
    }
    result = await guardrail.apply_guardrail(
        inputs={"texts": ["I deleted report.txt."]},
        request_data=request_data,
        input_type="response",
    )
    assert result["texts"] == ["I deleted report.txt."]


@pytest.mark.asyncio
async def test_failed_tool_claim_blocked():
    pytest.importorskip("epistemic_guard")
    guardrail = EpistemicGuardrail(guardrail_name="eg-failed", on_fail="block")
    request_data = {
        "messages": [
            {"role": "user", "content": "Delete report.txt."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "delete_file",
                            "arguments": '{"path": "report.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "delete_file",
                "content": "Error: PermissionError: access denied",
            },
        ]
    }
    with pytest.raises(HTTPException) as exc_info:
        await guardrail.apply_guardrail(
            inputs={"texts": ["Done, I deleted report.txt."]},
            request_data=request_data,
            input_type="response",
        )
    assert exc_info.value.status_code == 400
