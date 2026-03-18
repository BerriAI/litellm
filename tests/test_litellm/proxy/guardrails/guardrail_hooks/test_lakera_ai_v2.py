"""
Tests for Lakera AI v2 guardrail hook (post-call and shared behavior).

PR checklist requires at least one test in tests/test_litellm/.
Additional tests live in tests/guardrails_tests/test_lakera_v2.py.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
