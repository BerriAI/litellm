# +-------------------------------------------------------------+
#
#           Use Aim Security Guardrails for your LLM calls
#                   https://www.aim.security/
#
# +-------------------------------------------------------------+

import os
from typing import Literal, Optional

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth


class AimGuardrailMissingSecrets(Exception):
    pass


class AimGuardrail(CustomGuardrail):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.api_key = api_key or os.environ.get("AIM_API_KEY")
        if not self.api_key:
            msg = (
                "Couldn't get Aim api key, either set the `AIM_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise AimGuardrailMissingSecrets(msg)
        self.api_base = (
            api_base or os.environ.get("AIM_API_BASE") or "https://api.aim.security"
        )
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
    ) -> Exception | str | dict | None:
        verbose_proxy_logger.debug("Inside AIM Pre-Call Hook")

        user_email = data.get("metadata", {}).get("headers", {}).get("x-aim-user-email")
        headers = {"Authorization": f"Bearer {self.api_key}"} | (
            {"x-aim-user-email": user_email} if user_email else {}
        )
        response = await self.async_handler.post(
            f"{self.api_base}/detect/openai",
            headers=headers,
            json={"messages": data.get("messages", [])},
        )
        response.raise_for_status()
        res = response.json()
        detected = res["detected"]
        verbose_proxy_logger.info(
            "Aim: detected: {detected}, enabled policies: {policies}".format(
                detected=detected, policies=list(res["details"].keys())
            )
        )
        if detected:
            raise HTTPException(status_code=400, detail=res["detection_message"])
        return data
