"""Unit tests for the Microsoft Purview DLP guardrail.

Focus areas:
1. _check_content – API/network errors are only re-raised when block_on_violation=True.
2. async_logging_hook – response audit always runs even if prompt audit raises.
3. _convert_content_list_to_str – extracts tool_calls/function_call arguments.
4. _extract_response_text – captures model-generated tool-call arguments.
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
    MicrosoftPurviewDLPGuardrail,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_guardrail(**kwargs: Any) -> MicrosoftPurviewDLPGuardrail:
    """Return a guardrail with dummy credentials so __init__ does not validate."""
    defaults = dict(
        tenant_id="test-tenant",
        client_id="test-client",
        api_key="test-secret",
        guardrail_name="purview-test",
    )
    defaults.update(kwargs)
    return MicrosoftPurviewDLPGuardrail(**defaults)


def _purview_violation_response() -> Dict[str, Any]:
    return {"matches": [{"name": "Credit Card Number", "count": 1}]}


def _purview_clean_response() -> Dict[str, Any]:
    return {"matches": []}


# ---------------------------------------------------------------------------
# Bug 1 – _check_content: API errors respect block_on_violation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_content_api_error_reraises_when_block_on_violation_true():
    """should re-raise a network error when block_on_violation=True."""
    guardrail = _make_guardrail(block_on_violation=True)

    with patch.object(
        guardrail,
        "_call_purview_api",
        new=AsyncMock(side_effect=ConnectionError("network down")),
    ):
        with pytest.raises(ConnectionError, match="network down"):
            await guardrail._check_content("sensitive text", block_on_violation=True)


@pytest.mark.asyncio
async def test_check_content_api_error_swallowed_when_block_on_violation_false():
    """should NOT re-raise a network error when block_on_violation=False."""
    guardrail = _make_guardrail(block_on_violation=False)

    with patch.object(
        guardrail,
        "_call_purview_api",
        new=AsyncMock(side_effect=ConnectionError("network down")),
    ):
        # Must return None without raising
        result = await guardrail._check_content("text", block_on_violation=False)
        assert result is None


@pytest.mark.asyncio
async def test_check_content_violation_raises_http_exception_when_block_true():
    """should raise HTTPException(400) on a violation when block_on_violation=True."""
    guardrail = _make_guardrail(block_on_violation=True)

    with patch.object(
        guardrail,
        "_call_purview_api",
        new=AsyncMock(return_value=_purview_violation_response()),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail._check_content(
                "4111 1111 1111 1111", block_on_violation=True
            )
        assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_check_content_violation_does_not_raise_when_block_false():
    """should NOT raise on a violation when block_on_violation=False (log only)."""
    guardrail = _make_guardrail(block_on_violation=False)

    with patch.object(
        guardrail,
        "_call_purview_api",
        new=AsyncMock(return_value=_purview_violation_response()),
    ):
        result = await guardrail._check_content(
            "4111 1111 1111 1111", block_on_violation=False
        )
        # Returns the purview response, does not raise
        assert result is not None
        assert result["matches"]


@pytest.mark.asyncio
async def test_check_content_clean_content_returns_response():
    """should return the Purview response when no violation is found."""
    guardrail = _make_guardrail()

    with patch.object(
        guardrail,
        "_call_purview_api",
        new=AsyncMock(return_value=_purview_clean_response()),
    ):
        result = await guardrail._check_content("hello world", block_on_violation=True)
        assert result == _purview_clean_response()


# ---------------------------------------------------------------------------
# Bug 2 – async_logging_hook: response audit runs even if prompt audit fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_response_audit_runs_after_prompt_error():
    """should still audit the response when the prompt scan raises an API error."""
    guardrail = _make_guardrail()

    prompt_check = AsyncMock(side_effect=ConnectionError("prompt API error"))
    response_check = AsyncMock(return_value=_purview_clean_response())

    call_log: List[str] = []

    async def fake_check_content(text: str, block_on_violation: bool = True) -> Any:
        if "prompt" in text:
            call_log.append("prompt")
            raise ConnectionError("prompt API error")
        call_log.append("response")
        return _purview_clean_response()

    with patch.object(guardrail, "_check_content", side_effect=fake_check_content):
        mock_response = MagicMock(spec=litellm.ModelResponse)
        mock_response.choices = []

        # _extract_messages_text returns something containing "prompt"
        with patch.object(
            guardrail, "_extract_messages_text", return_value="prompt text"
        ):
            # _extract_response_text returns something NOT containing "prompt"
            with patch.object(
                guardrail, "_extract_response_text", return_value="response text"
            ):
                kwargs, result = await guardrail.async_logging_hook(
                    kwargs={"messages": [{"role": "user", "content": "prompt text"}]},
                    result=mock_response,
                    start_time=None,
                    end_time=None,
                )

    # Both audits must have been attempted
    assert "prompt" in call_log, "prompt audit was not called"
    assert "response" in call_log, "response audit was skipped after prompt error"


@pytest.mark.asyncio
async def test_async_logging_hook_both_audits_run_on_success():
    """should call _check_content twice (prompt and response) when both succeed."""
    guardrail = _make_guardrail()

    call_order: List[str] = []

    async def fake_check(text: str, block_on_violation: bool = True) -> Any:
        call_order.append("called")
        return _purview_clean_response()

    mock_response = MagicMock(spec=litellm.ModelResponse)
    mock_response.choices = []

    with patch.object(guardrail, "_check_content", side_effect=fake_check):
        with patch.object(guardrail, "_extract_messages_text", return_value="prompt"):
            with patch.object(guardrail, "_extract_response_text", return_value="resp"):
                await guardrail.async_logging_hook(
                    kwargs={"messages": [{"role": "user", "content": "prompt"}]},
                    result=mock_response,
                    start_time=None,
                    end_time=None,
                )

    assert len(call_order) == 2


@pytest.mark.asyncio
async def test_async_logging_hook_skips_audit_when_text_empty():
    """should not call _check_content when text extraction returns empty string."""
    guardrail = _make_guardrail()

    check_mock = AsyncMock()

    mock_response = MagicMock(spec=litellm.ModelResponse)
    mock_response.choices = []

    with patch.object(guardrail, "_check_content", check_mock):
        with patch.object(guardrail, "_extract_messages_text", return_value=""):
            with patch.object(guardrail, "_extract_response_text", return_value=""):
                await guardrail.async_logging_hook(
                    kwargs={"messages": []},
                    result=mock_response,
                    start_time=None,
                    end_time=None,
                )

    check_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Bug 3 – _convert_content_list_to_str: tool-call arguments are included
# ---------------------------------------------------------------------------


def test_convert_content_list_to_str_plain_string():
    """should return the content string as-is."""
    guardrail = _make_guardrail()
    msg = {"role": "user", "content": "hello world"}
    assert guardrail._convert_content_list_to_str(msg) == "hello world"


def test_convert_content_list_to_str_content_list():
    """should concatenate text parts from a content list."""
    guardrail = _make_guardrail()
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ],
    }
    result = guardrail._convert_content_list_to_str(msg)
    assert "part one" in result
    assert "part two" in result


def test_convert_content_list_to_str_includes_tool_call_arguments():
    """should include tool_calls[].function.arguments so DLP sees them."""
    guardrail = _make_guardrail()
    sensitive_args = json.dumps({"ssn": "123-45-6789"})
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup_user",
                    "arguments": sensitive_args,
                },
            }
        ],
    }
    result = guardrail._convert_content_list_to_str(msg)
    assert sensitive_args in result


def test_convert_content_list_to_str_includes_function_call_arguments():
    """should include legacy function_call.arguments so DLP sees them."""
    guardrail = _make_guardrail()
    sensitive_args = json.dumps({"credit_card": "4111111111111111"})
    msg = {
        "role": "assistant",
        "content": None,
        "function_call": {
            "name": "charge_card",
            "arguments": sensitive_args,
        },
    }
    result = guardrail._convert_content_list_to_str(msg)
    assert sensitive_args in result


def test_convert_content_list_to_str_combines_content_and_tool_args():
    """should include both content text and tool-call arguments."""
    guardrail = _make_guardrail()
    msg = {
        "role": "user",
        "content": "public text",
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "do_thing", "arguments": '{"secret": "xyz"}'},
            }
        ],
    }
    result = guardrail._convert_content_list_to_str(msg)
    assert "public text" in result
    assert '"secret"' in result


def test_convert_content_list_to_str_empty_message():
    """should return empty string for a message with no extractable text."""
    guardrail = _make_guardrail()
    msg: Dict[str, Any] = {"role": "assistant", "content": None}
    assert guardrail._convert_content_list_to_str(msg) == ""


# ---------------------------------------------------------------------------
# _extract_response_text: model-generated tool calls are included
# ---------------------------------------------------------------------------


def test_extract_response_text_includes_model_tool_call_arguments():
    """should capture tool_calls[].function.arguments from a model response."""
    guardrail = _make_guardrail()

    sensitive_args = json.dumps({"password": "hunter2"})

    tc = MagicMock()
    tc.function.arguments = sensitive_args

    message = MagicMock()
    message.content = None
    message.tool_calls = [tc]
    message.function_call = None

    choice = MagicMock()
    choice.message = message

    response = MagicMock(spec=litellm.ModelResponse)
    response.choices = [choice]

    result = guardrail._extract_response_text(response)
    assert sensitive_args in result


def test_extract_response_text_includes_legacy_function_call():
    """should capture function_call.arguments from a model response."""
    guardrail = _make_guardrail()

    sensitive_args = json.dumps({"api_key": "sk-secret"})

    message = MagicMock()
    message.content = None
    message.tool_calls = None
    message.function_call = MagicMock()
    message.function_call.arguments = sensitive_args

    choice = MagicMock()
    choice.message = message

    response = MagicMock(spec=litellm.ModelResponse)
    response.choices = [choice]

    result = guardrail._extract_response_text(response)
    assert sensitive_args in result


def test_extract_response_text_non_model_response_returns_empty():
    """should return empty string for non-ModelResponse types (TTS, images, etc.)."""
    guardrail = _make_guardrail()
    assert guardrail._extract_response_text("raw string") == ""
    assert guardrail._extract_response_text(None) == ""
    assert guardrail._extract_response_text({"choices": []}) == ""


# ---------------------------------------------------------------------------
# Integration: pre-call hook respects block_on_violation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_call_hook_blocks_when_violation_and_block_on_violation_true():
    """should raise HTTPException from pre_call when block_on_violation=True."""
    from litellm.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    # default_on=True ensures should_run_guardrail() returns True
    guardrail = _make_guardrail(block_on_violation=True, default_on=True)

    with patch.object(
        guardrail,
        "_call_purview_api",
        new=AsyncMock(return_value=_purview_violation_response()),
    ):
        with patch.object(
            guardrail, "_get_access_token", new=AsyncMock(return_value="tok")
        ):
            with pytest.raises(HTTPException) as exc_info:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=MagicMock(spec=DualCache),
                    data={
                        "messages": [{"role": "user", "content": "4111 1111 1111 1111"}]
                    },
                    call_type="completion",
                )
            assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_pre_call_hook_passes_when_violation_and_block_on_violation_false():
    """should NOT raise from pre_call when block_on_violation=False (audit only)."""
    from litellm.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth

    # default_on=True ensures should_run_guardrail() returns True
    guardrail = _make_guardrail(block_on_violation=False, default_on=True)

    with patch.object(
        guardrail,
        "_call_purview_api",
        new=AsyncMock(return_value=_purview_violation_response()),
    ):
        with patch.object(
            guardrail, "_get_access_token", new=AsyncMock(return_value="tok")
        ):
            data = {"messages": [{"role": "user", "content": "4111 1111 1111 1111"}]}
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=MagicMock(spec=DualCache),
                data=data,
                call_type="completion",
            )
            assert result is not None  # data dict returned, no exception


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_guardrail_class_is_registered():
    """should be discoverable via the guardrail_class_registry in __init__."""
    from litellm.proxy.guardrails.guardrail_hooks.microsoft_purview import (
        guardrail_class_registry,
    )
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    assert (
        SupportedGuardrailIntegrations.MICROSOFT_PURVIEW.value
        in guardrail_class_registry
    )
    assert (
        guardrail_class_registry[SupportedGuardrailIntegrations.MICROSOFT_PURVIEW.value]
        is MicrosoftPurviewDLPGuardrail
    )


def test_get_config_model_returns_microsoft_purview_config():
    """should return MicrosoftPurviewDLPConfigModel from get_config_model()."""
    from litellm.types.proxy.guardrails.guardrail_hooks.microsoft_purview import (
        MicrosoftPurviewDLPConfigModel,
    )

    assert (
        MicrosoftPurviewDLPGuardrail.get_config_model()
        is MicrosoftPurviewDLPConfigModel
    )
