from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.azure.prompt_shield import (
    AzureContentSafetyPromptShieldGuardrail,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.mark.asyncio
async def test_azure_prompt_shield_guardrail_pre_call_hook():

    azure_prompt_shield_guardrail = AzureContentSafetyPromptShieldGuardrail(
        guardrail_name="azure_prompt_shield",
        api_key="azure_prompt_shield_api_key",
        api_base="azure_prompt_shield_api_base",
    )
    with patch.object(
        azure_prompt_shield_guardrail, "async_make_request"
    ) as mock_async_make_request:
        mock_async_make_request.return_value = {
            "userPromptAnalysis": {"attackDetected": False},
            "documentsAnalysis": [],
        }
        await azure_prompt_shield_guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="azure_prompt_shield_api_key"),
            cache=None,
            data={
                "messages": [
                    {
                        "role": "user",
                        "content": "Hello, how are you?",
                    }
                ]
            },
            call_type="acompletion",
        )

        mock_async_make_request.assert_called_once()
        assert (
            mock_async_make_request.call_args.kwargs["user_prompt"]
            == "Hello, how are you?"
        )
