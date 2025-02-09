# +-------------------------------------------------------------+
#
#           Use Lasso Security Guardrails for your LLM calls
#
# +-------------------------------------------------------------+

import os
from typing import Literal, Optional, Union

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth


class LassoGuardrailMissingSecrets(Exception):
    pass


class LassoGuardrail(CustomGuardrail):
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_key = api_key or os.environ.get("LASSO_API_KEY")
        if not self.api_key:
            msg = (
                "Failed to retrieve the Lasso API key. Please ensure it is either set in the environment variables " 
                "or provided as a parameter within the guardrail configuration file."
            )
            raise LassoGuardrailMissingSecrets(msg)
            # FIXME: I don't think we support a custom base url
        self.api_base = api_base or os.environ.get("LASSO_API_BASE") or "https://server.lasso.security"
        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside Lasso Pre-Call Hook")

        await self.call_lasso_guardrail(data, hook="pre_call")
        return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside Lasso Moderation Hook")

        await self.call_lasso_guardrail(data, hook="moderation")
        return data


    async def call_lasso_guardrail(self, data: dict, hook: str) -> None:
        # Prepare headers for the request
        headers = {
            "lasso-x-api-key": self.api_key,
            "x-lasso-litellm-hook": hook
        }

        # Prepare request payload and endpoint
        prompt = data.get("messages", [])
        endpoint = f"{self.api_base}/gateway/v1"

        # Make the POST request to the Lasso endpoint
        response = await self.async_handler.post(
            endpoint,
            headers=headers,
            json={"prompt": prompt}
        )

        # Check for response errors
        if response.status_code == 401:
            raise HTTPException(status_code=401, detail="Invalid API key.")
        elif response.status_code == 403:
            raise HTTPException(status_code=403, detail="Forbidden resource.")

        # Parse and log the response
        response_data = response.json()
        detected = any(response_data.values())
        verbose_proxy_logger.info(
            "Lasso: detected: {detected}, services response: {response}".format(
                detected=detected, response=response_data
            )
        )

        # Raise an exception if any service detects an issue
        if detected:
            raise HTTPException(status_code=400, detail="Detected policy violation.")
