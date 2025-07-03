#!/usr/bin/env python3
"""
Azure Text Moderation Native Guardrail Integrationfor LiteLLM
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

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
from litellm.proxy._types import UserAPIKeyAuth

from .base import AzureGuardrailBase

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import AllMessageValues
    from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_text_moderation import (
        AzureTextModerationGuardrailResponse,
    )
    from litellm.types.utils import EmbeddingResponse, ImageResponse, ModelResponse


class AzureContentSafetyTextModerationGuardrail(AzureGuardrailBase, CustomGuardrail):
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

    default_severity_threshold: int = 2

    def __init__(
        self,
        guardrail_name: str,
        api_key: str,
        api_base: str,
        severity_threshold: Optional[int] = None,
        severity_threshold_by_category: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        """Initialize Azure Text Moderation guardrail handler."""
        # Initialize parent CustomGuardrail
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_text_moderation import (
            AzureTextModerationRequestBodyOptionalParams,
        )

        super().__init__(
            guardrail_name=guardrail_name,
            **kwargs,
        )
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Store configuration
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = kwargs.get("api_version") or "2024-09-01"
        self.optional_params_request_body: (
            AzureTextModerationRequestBodyOptionalParams
        ) = {
            "categories": kwargs.get("categories")
            or [
                "Hate",
                "Sexual",
                "SelfHarm",
                "Violence",
            ],
            "blocklistNames": cast(
                Optional[List[str]], kwargs.get("blocklistNames") or None
            ),
            "haltOnBlocklistHit": kwargs.get("haltOnBlocklistHit") or False,
            "outputType": kwargs.get("outputType") or "FourSeverityLevels",
        }

        self.severity_threshold = severity_threshold
        self.severity_threshold_by_category = severity_threshold_by_category

        verbose_proxy_logger.info(
            f"Initialized Azure Prompt Shield Guardrail: {guardrail_name}"
        )

    async def async_make_request(
        self, text: str
    ) -> "AzureTextModerationGuardrailResponse":
        """
        Make a request to the Azure Prompt Shield API.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.azure.azure_text_moderation import (
            AzureTextModerationGuardrailRequestBody,
            AzureTextModerationGuardrailResponse,
        )

        request_body = AzureTextModerationGuardrailRequestBody(
            text=text,
            **self.optional_params_request_body,
        )
        verbose_proxy_logger.debug(
            "Azure Text Moderation guard request: %s", request_body
        )
        response = await self.async_handler.post(
            url=f"{self.api_base}/contentsafety/text:analyze?api-version={self.api_version}",
            headers={
                "Ocp-Apim-Subscription-Key": self.api_key,
                "Content-Type": "application/json",
            },
            json=cast(dict, request_body),
        )

        verbose_proxy_logger.debug(
            "Azure Text Moderation guard response: %s", response.json()
        )
        return AzureTextModerationGuardrailResponse(**response.json())  # type: ignore

    def check_severity_threshold(
        self, response: "AzureTextModerationGuardrailResponse"
    ) -> Literal[True]:
        """
        - Check if threshold set by category
        - Check if general severity threshold set
        - If both none, use default_severity_threshold
        """
        if self.severity_threshold_by_category:
            for category in response["categoriesAnalysis"]:
                severity_category_threshold_item = (
                    self.severity_threshold_by_category.get(category["category"])
                )
                if (
                    severity_category_threshold_item is not None
                    and severity_category_threshold_item >= category["severity"]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Azure Content Safety Guardrail: {} crossed severity {}, Got severity: {}".format(
                                category["category"],
                                self.severity_threshold_by_category.get(
                                    category["category"]
                                ),
                                category["severity"],
                            )
                        },
                    )
        if self.severity_threshold:
            for category in response["categoriesAnalysis"]:
                if category["severity"] >= self.severity_threshold:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Azure Content Safety Guardrail: {} crossed severity {}, Got severity: {}".format(
                                category["category"],
                                self.severity_threshold,
                                category["severity"],
                            )
                        },
                    )
        if (
            self.severity_threshold is None
            and self.severity_threshold_by_category is None
        ):
            for category in response["categoriesAnalysis"]:

                if category["severity"] >= self.default_severity_threshold:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Azure Content Safety Guardrail: {} crossed severity {}, Got severity: {}".format(
                                category["category"],
                                self.default_severity_threshold,
                                category["severity"],
                            )
                        },
                    )
        return True

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
        ],
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-call hook to scan user prompts before sending to LLM.

        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.info(
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
                verbose_proxy_logger.info(
                    f"Azure Text Moderation: User prompt: {user_prompt}"
                )
                azure_text_moderation_response = await self.async_make_request(
                    text=user_prompt,
                )
                self.check_severity_threshold(response=azure_text_moderation_response)
            else:
                verbose_proxy_logger.warning("Azure Text Moderation: No text found")
        return None

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Union[Any, "ModelResponse", "EmbeddingResponse", "ImageResponse"],
    ) -> Any:
        from litellm.types.utils import Choices, ModelResponse

        if (
            isinstance(response, ModelResponse)
            and response.choices
            and isinstance(response.choices[0], Choices)
        ):
            content = response.choices[0].message.content or ""
            azure_text_moderation_response = await self.async_make_request(
                text=content,
            )
            self.check_severity_threshold(response=azure_text_moderation_response)
        return response

    async def async_post_call_streaming_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response: str
    ) -> Any:
        try:
            if response is not None and len(response) > 0:
                azure_text_moderation_response = await self.async_make_request(
                    text=response,
                )
                self.check_severity_threshold(response=azure_text_moderation_response)
            return response
        except HTTPException as e:
            import json

            error_returned = json.dumps({"error": e.detail})
            return f"data: {error_returned}\n\n"
