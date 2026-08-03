import asyncio
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, "../../../../..")

import litellm
from litellm.proxy.guardrails.guardrail_hooks.agentguards.agentguards import (
    AgentGuardsGuardrail,
)


def _guardrail(**overrides):
    kwargs = dict(
        guardrail_name="agentguards",
        api_base="http://agentguards.local",
        tenant_id="dev",
        default_on=True,
        event_hook=["pre_call", "post_call"],
    )
    kwargs.update(overrides)
    return AgentGuardsGuardrail(**kwargs)


def _data(text):
    return {"messages": [{"role": "user", "content": text}], "litellm_call_id": "c1"}


def _stub(guard, responses):
    async def fake_post(path, payload):
        return responses.get(path)

    guard._post = fake_post


def test_provider_is_discovered():
    from litellm.types.guardrails import SupportedGuardrailIntegrations
    from litellm.proxy.guardrails.guardrail_registry import guardrail_class_registry

    assert SupportedGuardrailIntegrations.AGENTGUARDS.value == "agentguards"
    assert "agentguards" in guardrail_class_registry


def test_config_model_ui_name():
    cm = AgentGuardsGuardrail.get_config_model()
    assert cm is not None
    assert cm.ui_friendly_name() == "AgentGuards"


@pytest.mark.asyncio
async def test_input_allow_passes_through():
    g = _guardrail()
    _stub(g, {"/v1/guardrails/evaluate-input": {"decision": "allow"}})
    out = await g.async_pre_call_hook(None, None, _data("hi"), "completion")
    assert out["messages"][0]["content"] == "hi"


@pytest.mark.asyncio
async def test_input_block_raises_400():
    g = _guardrail()
    _stub(
        g,
        {"/v1/guardrails/evaluate-input": {"decision": "block", "message": "BLOCKED"}},
    )
    with pytest.raises(HTTPException) as exc:
        await g.async_pre_call_hook(None, None, _data("jailbreak"), "completion")
    assert exc.value.status_code == 400
    assert exc.value.detail["message"] == "BLOCKED"


@pytest.mark.asyncio
async def test_input_redact_substitutes():
    g = _guardrail()
    _stub(
        g,
        {
            "/v1/guardrails/evaluate-input": {
                "decision": "redact",
                "redacted_text": "my ssn is [REDACTED]",
            }
        },
    )
    out = await g.async_pre_call_hook(None, None, _data("my ssn is 1"), "completion")
    assert out["messages"][0]["content"] == "my ssn is [REDACTED]"


@pytest.mark.asyncio
async def test_output_reject_raises_400():
    g = _guardrail()
    _stub(g, {"/v1/outputs/validate": {"decision": "reject", "message": "EXFIL"}})
    resp = litellm.ModelResponse(choices=[litellm.Choices(message=litellm.Message(content="secret sk-abc"))])
    with pytest.raises(HTTPException) as exc:
        await g.async_post_call_success_hook(_data("q"), None, resp)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_output_pass_returns_response():
    g = _guardrail()
    _stub(g, {"/v1/outputs/validate": {"decision": "pass"}})
    resp = litellm.ModelResponse(choices=[litellm.Choices(message=litellm.Message(content="all good"))])
    out = await g.async_post_call_success_hook(_data("q"), None, resp)
    assert out is resp


@pytest.mark.asyncio
async def test_fail_open_on_unreachable():
    g = _guardrail(fail_closed=False)
    _stub(g, {})  # returns None -> simulates fail-open
    out = await g.async_pre_call_hook(None, None, _data("hi"), "completion")
    assert out["messages"][0]["content"] == "hi"


def _capturing_stub(guard, responses):
    """Stub _post that records the payloads it was called with."""
    calls = []

    async def fake_post(path, payload):
        calls.append((path, payload))
        return responses.get(path)

    guard._post = fake_post
    return calls


@pytest.mark.asyncio
async def test_input_inspects_all_messages():
    """Injection hidden in an earlier / system message must still be screened,
    not just the last user turn."""
    g = _guardrail()
    calls = _capturing_stub(g, {"/v1/guardrails/evaluate-input": {"decision": "allow"}})
    data = {
        "messages": [
            {"role": "system", "content": "SECRET_SYSTEM_MARKER instructions"},
            {"role": "user", "content": "benign question"},
        ],
        "litellm_call_id": "c1",
    }
    await g.async_pre_call_hook(None, None, data, "completion")
    sent_text = calls[0][1]["text"]
    assert "SECRET_SYSTEM_MARKER" in sent_text
    assert "benign question" in sent_text


@pytest.mark.asyncio
async def test_output_extracts_tool_call_args():
    """Exfiltration in tool-call arguments (no message content) must be validated."""
    g = _guardrail()
    calls = _capturing_stub(g, {"/v1/outputs/validate": {"decision": "reject", "message": "EXFIL"}})
    tool_calls = [{"id": "1", "type": "function", "function": {"name": "send", "arguments": '{"body":"EXFIL_MARKER"}'}}]
    resp = litellm.ModelResponse(
        choices=[litellm.Choices(message=litellm.Message(content=None, tool_calls=tool_calls))]
    )
    with pytest.raises(HTTPException) as exc:
        await g.async_post_call_success_hook(_data("q"), None, resp)
    assert exc.value.status_code == 400
    # the tool-call arguments were included in the validated output text
    assert "EXFIL_MARKER" in calls[0][1]["output_text"]


@pytest.mark.asyncio
async def test_redaction_applied_to_responses_api_input():
    """redact decision rewrites Responses-API string `input`, not just messages."""
    g = _guardrail()
    _stub(
        g,
        {"/v1/guardrails/evaluate-input": {"decision": "redact", "redacted_text": "my card is [REDACTED]"}},
    )
    data = {"input": "my card is 4111 1111 1111 1111", "litellm_call_id": "c1"}
    out = await g.async_pre_call_hook(None, None, data, "responses")
    assert out["input"] == "my card is [REDACTED]"
