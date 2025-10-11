"""
Test suite for PointGuard AI Guardrail Integration
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.pointguardai import (
    PointGuardAIGuardrail,
)
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.mark.asyncio
async def test_pointguard_pre_call_hook_no_violations():
    """Test pre_call hook when no violations detected"""
    guardrail = PointGuardAIGuardrail(
        guardrail_name="pointguardai",
        api_key="test_api_key",
        api_email="test@example.com",
        api_base="https://api.appsoc.com",
        org_code="test-org",
        policy_config_name="test-policy",
    )

    with patch.object(
            guardrail, "make_pointguard_api_request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = None  # No modifications

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
            cache=None,
            data={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            },
            call_type="completion",
        )

        mock_request.assert_called_once()
        # Should return original data
        assert result["messages"][0]["content"] == "Hello, how are you?"


@pytest.mark.asyncio
async def test_pointguard_pre_call_hook_content_blocked():
    """Test pre_call hook when content is blocked"""
    guardrail = PointGuardAIGuardrail(
        guardrail_name="pointguardai",
        api_key="test_api_key",
        api_email="test@example.com",
        api_base="https://api.appsoc.com",
        org_code="test-org",
        policy_config_name="test-policy",
    )

    with patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        # Mock blocked response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "input": {
                    "blocked": True,
                    "content": [
                        {
                            "originalContent": "Hello, how are you?",
                            "violations": [
                                {
                                    "severity": "HIGH",
                                    "categories": ["prohibited_content"],
                                }
                            ],
                        }
                    ],
                }
            },
            raise_for_status=lambda: None,
        )

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                cache=None,
                data={
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ]
                },
                call_type="completion",
            )

        assert exc_info.value.status_code == 400
        assert "Violated PointGuardAI policy" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_pointguard_pre_call_hook_content_modified():
    """Test pre_call hook when content is modified"""
    guardrail = PointGuardAIGuardrail(
        guardrail_name="pointguardai",
        api_key="test_api_key",
        api_email="test@example.com",
        api_base="https://api.appsoc.com",
        org_code="test-org",
        policy_config_name="test-policy",
    )

    with patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        # Mock modified response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "input": {
                    "blocked": False,
                    "modified": True,
                    "content": [
                        {
                            "originalContent": "Hello, how are you?",
                            "modifiedContent": "Hello, [REDACTED]",
                        }
                    ],
                }
            },
            raise_for_status=lambda: None,
        )

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
            cache=None,
            data={
                "messages": [
                    {"role": "user", "content": "Hello, how are you?"}
                ]
            },
            call_type="completion",
        )

        # Content should be modified
        assert result["messages"][0]["content"] == "Hello, [REDACTED]"


@pytest.mark.asyncio
async def test_pointguard_post_call_hook_no_violations():
    """Test post_call hook when response has no violations"""
    guardrail = PointGuardAIGuardrail(
        guardrail_name="pointguardai",
        api_key="test_api_key",
        api_email="test@example.com",
        api_base="https://api.appsoc.com",
        org_code="test-org",
        policy_config_name="test-policy",
    )

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="I'm doing well, thank you!",
                    role="assistant"
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
    )

    with patch.object(
            guardrail, "make_pointguard_api_request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.return_value = None  # No modifications

        result = await guardrail.async_post_call_success_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
            data={"messages": [{"role": "user", "content": "Hello"}]},
            response=response,
        )

        mock_request.assert_called_once()
        # Response should be unchanged
        assert result.choices[0].message.content == "I'm doing well, thank you!"


@pytest.mark.asyncio
async def test_pointguard_post_call_hook_response_blocked():
    """Test post_call hook when response is blocked"""
    guardrail = PointGuardAIGuardrail(
        guardrail_name="pointguardai",
        api_key="test_api_key",
        api_email="test@example.com",
        api_base="https://api.appsoc.com",
        org_code="test-org",
        policy_config_name="test-policy",
    )

    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="I'm doing well, thank you!",
                    role="assistant"
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
    )

    with patch.object(
            guardrail.async_handler, "post", new_callable=AsyncMock
    ) as mock_post:
        # Mock blocked response
        mock_post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "output": {
                    "blocked": True,
                    "content": [
                        {
                            "originalContent": "I'm doing well, thank you!",
                            "violations": [
                                {
                                    "severity": "MEDIUM",
                                    "categories": ["sensitive_info"],
                                }
                            ],
                        }
                    ],
                }
            },
            raise_for_status=lambda: None,
        )

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_post_call_success_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test_key"),
                data={"messages": [{"role": "user", "content": "Hello"}]},
                response=response,
            )

        assert exc_info.value.status_code == 400
        assert "Violated PointGuardAI policy" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_pointguard_initialization_missing_required_params():
    """Test that initialization fails without required parameters"""
    with pytest.raises(HTTPException) as exc_info:
        PointGuardAIGuardrail(
            guardrail_name="pointguardai",
            api_key="",  # Missing required param
            api_email="test@example.com",
            api_base="https://api.appsoc.com",
            org_code="test-org",
            policy_config_name="test-policy",
        )

    assert exc_info.value.status_code == 401