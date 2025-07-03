from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import (
    AzureContentSafetyTextModerationGuardrail,
)
from litellm.proxy.guardrails.init_guardrails import init_guardrails_v2
from litellm.types.utils import Choices, Message, ModelResponse


@pytest.mark.asyncio
async def test_azure_text_moderation_guardrail_pre_call_hook():

    azure_text_moderation_guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure_text_moderation",
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
    )
    with patch.object(
        azure_text_moderation_guardrail, "async_make_request"
    ) as mock_async_make_request:
        mock_async_make_request.return_value = {
            "blocklistsMatch": [],
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 2},
            ],
        }
        with pytest.raises(HTTPException):
            await azure_text_moderation_guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="azure_text_moderation_api_key"
                ),
                cache=None,
                data={
                    "messages": [
                        {
                            "role": "user",
                            "content": "I hate you!",
                        }
                    ]
                },
                call_type="acompletion",
            )

        mock_async_make_request.assert_called_once()
        assert mock_async_make_request.call_args.kwargs["text"] == "I hate you!"


@pytest.mark.asyncio
async def test_azure_text_moderation_guardrail_post_call_success_hook():

    azure_text_moderation_guardrail = AzureContentSafetyTextModerationGuardrail(
        guardrail_name="azure_text_moderation",
        api_key="azure_text_moderation_api_key",
        api_base="azure_text_moderation_api_base",
    )
    with patch.object(
        azure_text_moderation_guardrail, "async_make_request"
    ) as mock_async_make_request:
        mock_async_make_request.return_value = {
            "blocklistsMatch": [],
            "categoriesAnalysis": [
                {"category": "Hate", "severity": 2},
            ],
        }
        with pytest.raises(HTTPException):
            result = await azure_text_moderation_guardrail.async_post_call_success_hook(
                data={},
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="azure_text_moderation_api_key"
                ),
                response=ModelResponse(
                    choices=[
                        Choices(
                            index=0,
                            message=Message(content="I hate you!"),
                        )
                    ]
                ),
            )

        mock_async_make_request.assert_called_once()
        mock_async_make_request.call_args.kwargs["text"] == "I hate you!"
