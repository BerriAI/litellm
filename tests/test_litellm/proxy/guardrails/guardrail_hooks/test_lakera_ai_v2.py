"""
Tests for Lakera AI v2 guardrail hook (post-call and shared behavior).

PR checklist requires at least one test in tests/test_litellm/.
Additional tests live in tests/guardrails_tests/test_lakera_v2.py.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.llms.base_llm.guardrail_translation.utils import (
    filter_messages_by_skip_flags,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2 import LakeraAIGuardrail
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_lakera_post_call_success_hook_returns_model_response_when_pii_masked():
    """
    Post-call hook must return a ModelResponse (not a dict) when PII is masked,
    so the parent async_post_call_success_deployment_hook accepts it via _is_valid_response_type.
    """
    lakera_guardrail = LakeraAIGuardrail(api_key="test_key")
    mock_response = {
        "payload": [
            {"detector_type": "pii/email", "start": 11, "end": 26, "message_id": 1}
        ],
        "flagged": True,
        "breakdown": [
            {"detector_type": "pii/email", "detected": True, "message_id": 1},
        ],
    }
    llm_response = MagicMock()
    llm_response.model_dump.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Your email is test@example.com",
                }
            },
        ]
    }

    with patch.object(
        lakera_guardrail, "call_v2_guard", new_callable=AsyncMock
    ) as mock_call:
        mock_call.return_value = (mock_response, {})
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-3.5-turbo",
            "metadata": {},
        }
        user_api_key_dict = UserAPIKeyAuth(api_key="test_key")

        result = await lakera_guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=llm_response,
        )

    assert isinstance(
        result, ModelResponse
    ), "Must return ModelResponse so deployment hook does not discard masked response"
    result_dict = result.model_dump()
    assert "[MASKED" in result_dict["choices"][0]["message"]["content"]
    assert "test@example.com" not in result_dict["choices"][0]["message"]["content"]


SYSTEM_MSG = {"role": "system", "content": "be nice"}
USER_MSG = {"role": "user", "content": "hello"}
TOOL_MSG = {"role": "tool", "content": "tool result", "tool_call_id": "1"}


class TestFilterSkippedMessages:
    def test_drops_system_when_flag_true(self):
        guardrail = LakeraAIGuardrail(api_key="test_key", skip_system_message_in_guardrail=True)
        filtered, was_skipped = guardrail._filter_skipped_messages([SYSTEM_MSG, USER_MSG])
        assert filtered == [USER_MSG]
        assert was_skipped is True

    def test_keeps_system_when_flag_false_and_no_global_default(self, monkeypatch):
        monkeypatch.setattr(litellm, "skip_system_message_in_guardrail", False)
        guardrail = LakeraAIGuardrail(api_key="test_key", skip_system_message_in_guardrail=False)
        filtered, was_skipped = guardrail._filter_skipped_messages([SYSTEM_MSG, USER_MSG])
        assert filtered == [SYSTEM_MSG, USER_MSG]
        assert was_skipped is False

    def test_drops_tool_when_flag_true(self):
        guardrail = LakeraAIGuardrail(api_key="test_key", skip_tool_message_in_guardrail=True)
        filtered, was_skipped = guardrail._filter_skipped_messages([TOOL_MSG, USER_MSG])
        assert filtered == [USER_MSG]
        assert was_skipped is True

    def test_combined_flags_drop_both_system_and_tool(self):
        guardrail = LakeraAIGuardrail(
            api_key="test_key",
            skip_system_message_in_guardrail=True,
            skip_tool_message_in_guardrail=True,
        )
        filtered, was_skipped = guardrail._filter_skipped_messages([SYSTEM_MSG, TOOL_MSG, USER_MSG])
        assert filtered == [USER_MSG]
        assert was_skipped is True

    def test_global_default_used_when_per_instance_flag_is_none(self, monkeypatch):
        monkeypatch.setattr(litellm, "skip_system_message_in_guardrail", True)
        guardrail = LakeraAIGuardrail(api_key="test_key")
        assert guardrail.skip_system_message_in_guardrail is None
        filtered, was_skipped = guardrail._filter_skipped_messages([SYSTEM_MSG, USER_MSG])
        assert filtered == [USER_MSG]
        assert was_skipped is True

    def test_no_drop_returns_was_skipped_false_when_nothing_to_drop(self):
        guardrail = LakeraAIGuardrail(api_key="test_key", skip_system_message_in_guardrail=True)
        filtered, was_skipped = guardrail._filter_skipped_messages([USER_MSG])
        assert filtered == [USER_MSG]
        assert was_skipped is False


class TestSharedFilterMessagesBySkipFlagsUtil:
    def test_importable_directly_from_shared_utils_module(self):
        from litellm.llms.base_llm.guardrail_translation import utils as guardrail_utils

        assert guardrail_utils.filter_messages_by_skip_flags is filter_messages_by_skip_flags

    def test_lakera_delegates_to_shared_function(self):
        guardrail = LakeraAIGuardrail(api_key="test_key", skip_system_message_in_guardrail=True)
        sentinel = ([USER_MSG], True)
        with patch(
            "litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2.filter_messages_by_skip_flags",
            return_value=sentinel,
        ) as mock_shared:
            result = guardrail._filter_skipped_messages([SYSTEM_MSG, USER_MSG])
        mock_shared.assert_called_once_with(guardrail, [SYSTEM_MSG, USER_MSG])
        assert result == sentinel

    def test_shared_function_works_against_any_object_exposing_the_two_attributes(self):
        class _FakeGuardrail:
            def __init__(self, skip_system, skip_tool):
                self.skip_system_message_in_guardrail = skip_system
                self.skip_tool_message_in_guardrail = skip_tool

        fake = _FakeGuardrail(skip_system=True, skip_tool=True)
        filtered, was_skipped = filter_messages_by_skip_flags(fake, [SYSTEM_MSG, TOOL_MSG, USER_MSG])
        assert filtered == [USER_MSG]
        assert was_skipped is True


@pytest.mark.asyncio
class TestAsyncPreCallHookWiring:
    async def test_excludes_system_message_from_lakera_request_when_flag_set(self):
        guardrail = LakeraAIGuardrail(api_key="test_key", skip_system_message_in_guardrail=True)
        data = {
            "messages": [SYSTEM_MSG, USER_MSG],
            "model": "gpt-3.5-turbo",
            "metadata": {},
        }
        with patch.object(guardrail, "call_v2_guard", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ({"flagged": False}, {})
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                cache=MagicMock(),
                data=data,
                call_type="completion",
            )
        sent_messages = mock_call.call_args.kwargs["messages"]
        assert all(m.get("role") != "system" for m in sent_messages)
        assert any(m.get("role") == "user" for m in sent_messages)

    async def test_includes_system_message_when_flag_not_set(self):
        guardrail = LakeraAIGuardrail(api_key="test_key")
        data = {
            "messages": [SYSTEM_MSG, USER_MSG],
            "model": "gpt-3.5-turbo",
            "metadata": {},
        }
        with patch.object(guardrail, "call_v2_guard", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ({"flagged": False}, {})
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                cache=MagicMock(),
                data=data,
                call_type="completion",
            )
        sent_messages = mock_call.call_args.kwargs["messages"]
        assert any(m.get("role") == "system" for m in sent_messages)


@pytest.mark.asyncio
class TestAsyncModerationHookWiring:
    async def test_excludes_tool_message_from_lakera_request_when_flag_set(self):
        guardrail = LakeraAIGuardrail(api_key="test_key", skip_tool_message_in_guardrail=True)
        data = {
            "messages": [TOOL_MSG, USER_MSG],
            "model": "gpt-3.5-turbo",
            "metadata": {},
        }
        with patch.object(guardrail, "call_v2_guard", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = ({"flagged": False}, {})
            await guardrail.async_moderation_hook(
                data=data,
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                call_type="completion",
            )
        sent_messages = mock_call.call_args.kwargs["messages"]
        assert all(m.get("role") != "tool" for m in sent_messages)


PII_ONLY_LAKERA_RESPONSE = {
    "flagged": True,
    "breakdown": [{"detector_type": "pii/email", "detected": True, "message_id": 0}],
    "payload": [{"detector_type": "pii/email", "start": 0, "end": 5, "message_id": 0}],
}


@pytest.mark.asyncio
class TestPiiMaskingSafetyGuard:
    async def test_pii_only_violation_masks_in_place_when_nothing_skipped(self):
        guardrail = LakeraAIGuardrail(api_key="test_key", on_flagged="block")
        data = {
            "messages": [USER_MSG.copy()],
            "model": "gpt-3.5-turbo",
            "metadata": {},
        }
        with patch.object(guardrail, "call_v2_guard", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = (PII_ONLY_LAKERA_RESPONSE, {})
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                cache=MagicMock(),
                data=data,
                call_type="completion",
            )
        assert result["messages"][0]["content"] != USER_MSG["content"]
        assert "[MASKED" in result["messages"][0]["content"]

    async def test_pii_only_violation_with_skipped_system_message_blocks_instead_of_masking(self):
        guardrail = LakeraAIGuardrail(
            api_key="test_key", on_flagged="block", skip_system_message_in_guardrail=True
        )
        data = {
            "messages": [SYSTEM_MSG.copy(), USER_MSG.copy()],
            "model": "gpt-3.5-turbo",
            "metadata": {},
        }
        with patch.object(guardrail, "call_v2_guard", new_callable=AsyncMock) as mock_call, patch(
            "litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2.apply_redacted_messages_back"
        ) as mock_apply_redacted:
            mock_call.return_value = (PII_ONLY_LAKERA_RESPONSE, {})
            with pytest.raises(HTTPException):
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                    cache=MagicMock(),
                    data=data,
                    call_type="completion",
                )
        mock_apply_redacted.assert_not_called()

    async def test_pii_only_violation_with_skipped_system_message_monitor_mode_does_not_mask_or_raise(self):
        guardrail = LakeraAIGuardrail(
            api_key="test_key", on_flagged="monitor", skip_system_message_in_guardrail=True
        )
        data = {
            "messages": [SYSTEM_MSG.copy(), USER_MSG.copy()],
            "model": "gpt-3.5-turbo",
            "metadata": {},
        }
        with patch.object(guardrail, "call_v2_guard", new_callable=AsyncMock) as mock_call, patch(
            "litellm.proxy.guardrails.guardrail_hooks.lakera_ai_v2.apply_redacted_messages_back"
        ) as mock_apply_redacted:
            mock_call.return_value = (PII_ONLY_LAKERA_RESPONSE, {})
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                cache=MagicMock(),
                data=data,
                call_type="completion",
            )
        mock_apply_redacted.assert_not_called()
        assert result["messages"][1]["content"] == USER_MSG["content"]
