#!/usr/bin/env python3
"""
Azure Prompt Shield Native Guardrail Integrationfor LiteLLM
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.utils import CallTypesLiteral

from .base import AzureGuardrailBase

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
        AzurePromptShieldGuardrailResponse,
    )
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


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

        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
        ]
        # AzureGuardrailBase.__init__ stores api_key, api_base, api_version,
        # async_handler and forwards the rest to CustomGuardrail.
        super().__init__(
            api_key=api_key,
            api_base=api_base,
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

        verbose_proxy_logger.debug(
            f"Initialized Azure Prompt Shield Guardrail: {guardrail_name}"
        )

    async def async_make_request(
        self, user_prompt: str
    ) -> "AzurePromptShieldGuardrailResponse":
        """
        Make a request to the Azure Prompt Shield API.

        Long prompts are automatically split at word boundaries into chunks
        that respect the Azure Content Safety 10 000-character limit.  Each
        chunk is analysed independently; an attack in *any* chunk raises
        an HTTPException immediately.
        """
        from .base import AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
            AzurePromptShieldGuardrailRequestBody,
            AzurePromptShieldGuardrailResponse,
        )

        chunks = self.split_text_by_words(
            user_prompt, AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH
        )

        last_response: Optional[AzurePromptShieldGuardrailResponse] = None

        for chunk in chunks:
            request_body = AzurePromptShieldGuardrailRequestBody(
                documents=[], userPrompt=chunk
            )
            response_json = await self._post_to_content_safety(
                "text:shieldPrompt", cast(dict, request_body)
            )

            last_response = cast(AzurePromptShieldGuardrailResponse, response_json)

            if last_response["userPromptAnalysis"].get("attackDetected"):
                verbose_proxy_logger.warning(
                    "Azure Prompt Shield: Attack detected in chunk of length %d",
                    len(chunk),
                )
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated Azure Prompt Shield guardrail policy",
                        "detection_message": f"Attack detected: {last_response['userPromptAnalysis']}",
                    },
                )

        # chunks is always non-empty (split_text_by_words guarantees ≥1 element)
        assert last_response is not None
        return last_response

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: Any,
        data: Dict[str, Any],
        call_type: CallTypesLiteral,
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-call hook to scan user prompts before sending to LLM.

        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.debug(
            "Azure Prompt Shield: Running pre-call prompt scan, on call_type: %s",
            call_type,
        )
        new_messages: Optional[List[AllMessageValues]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning(
                "Azure Prompt Shield: not running guardrail. No messages in data"
            )
            return data
        user_prompt = self.get_user_prompt(new_messages)

        if user_prompt:
            verbose_proxy_logger.debug(
                f"Azure Prompt Shield: User prompt: {user_prompt}"
            )
            await self.async_make_request(
                user_prompt=user_prompt,
            )
        else:
            verbose_proxy_logger.warning("Azure Prompt Shield: No user prompt found")
        return None

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """
        Get the config model for the Azure Prompt Shield guardrail.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_prompt_shield import (
            AzurePromptShieldGuardrailConfigModel,
        )

        return AzurePromptShieldGuardrailConfigModel
