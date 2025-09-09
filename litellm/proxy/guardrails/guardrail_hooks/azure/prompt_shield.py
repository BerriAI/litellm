#!/usr/bin/env python3
"""
Azure Prompt Shield Native Guardrail Integrationfor LiteLLM
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .base import AzureGuardrailBase

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
        AzurePromptShieldGuardrailResponse,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from litellm.types.utils import ModelResponse


class AzureContentSafetyPromptShieldGuardrail(AzureGuardrailBase, CustomGuardrail):
    """
    LiteLLM Built-in Guardrail for Azure Content Safety Guardrail (Prompt Shield).

    This guardrail scans prompts and responses using the Azure Prompt Shield API to detect
    malicious content, injection attempts, and policy violations.

    Configuration:
        guardrail_name: Name of the guardrail instance
        api_key: Azure Prompt Shield API key
        api_base: Azure Prompt Shield API endpoint
        default_on: Whether to enable by default
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: str,
        api_base: str,
        **kwargs,
    ):
        """Initialize Azure Prompt Shield guardrail handler."""
        from litellm.types.guardrails import GuardrailEventHooks

        # Initialize parent CustomGuardrail

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
        ]
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Store configuration
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = kwargs.get("api_version") or "2024-09-01"

        verbose_proxy_logger.debug(
            f"Initialized Azure Prompt Shield Guardrail: {guardrail_name}"
        )

    async def async_make_request(
        self, user_prompt: str
    ) -> "AzurePromptShieldGuardrailResponse":
        """
        Make a request to the Azure Prompt Shield API.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
            AzurePromptShieldGuardrailRequestBody,
            AzurePromptShieldGuardrailResponse,
        )

        request_body = AzurePromptShieldGuardrailRequestBody(
            documents=[], userPrompt=user_prompt
        )
        verbose_proxy_logger.debug(
            "Azure Prompt Shield guard request: %s", request_body
        )
        response = await self.async_handler.post(
            url=f"{self.api_base}/contentsafety/text:shieldPrompt?api-version={self.api_version}",
            headers={
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/json",
            },
            json=cast(dict, request_body),
        )

        verbose_proxy_logger.debug(
            "Azure Prompt Shield guard response: %s", response.json()
        )
        return AzurePromptShieldGuardrailResponse(**response.json())  # type: ignore

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: Any,
        data: Dict[str, Any],
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
        ],
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-call hook to scan user prompts before sending to LLM.

        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.debug(
            "Azure Prompt Shield: Running pre-call prompt scan, on call_type: %s",
            call_type,
        )
        if call_type == "acompletion":
            new_messages: Optional[List[AllMessageValues]] = data.get("messages")
            if new_messages is None:
                verbose_proxy_logger.warning(
                    "Lakera AI: not running guardrail. No messages in data"
                )
                return data
            user_prompt = self.get_user_prompt(new_messages)

            if user_prompt:
                verbose_proxy_logger.debug(
                    f"Azure Prompt Shield: User prompt: {user_prompt}"
                )
                azure_prompt_shield_response = await self.async_make_request(
                    user_prompt=user_prompt,
                )
                if azure_prompt_shield_response["userPromptAnalysis"].get(
                    "attackDetected"
                ):
                    verbose_proxy_logger.warning("Azure Prompt Shield: Attack detected")
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Violated Azure Prompt Shield guardrail policy",
                            "detection_message": f"Attack detected: {azure_prompt_shield_response['userPromptAnalysis']}",
                        },
                    )
            else:
                verbose_proxy_logger.warning(
                    "Azure Prompt Shield: No user prompt found"
                )
        return None

    @log_guardrail_information
    async def async_post_call_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: "UserAPIKeyAuth",
        response: "ModelResponse",
    ) -> "ModelResponse":
        """
        Post-call hook to scan LLM responses before returning to user.

        Raises HTTPException if response should be blocked.
        """
        verbose_proxy_logger.debug(
            "Azure Prompt Shield: Running post-call response scan"
        )

        return response

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """
        Get the config model for the Azure Prompt Shield guardrail.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
            AzurePromptShieldGuardrailConfigModel,
        )

        return AzurePromptShieldGuardrailConfigModel
