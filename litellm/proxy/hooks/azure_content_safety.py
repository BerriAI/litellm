import sys
import traceback
import uuid
from typing import Optional

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth


class _PROXY_AzureContentSafety(
    CustomLogger
):  # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes

    def __init__(self, endpoint, api_key, thresholds=None):
        try:
            from azure.ai.contentsafety.aio import ContentSafetyClient
            from azure.ai.contentsafety.models import (
                AnalyzeTextOptions,
                AnalyzeTextOutputType,
                TextCategory,
            )
            from azure.core.credentials import AzureKeyCredential
            from azure.core.exceptions import HttpResponseError
        except Exception as e:
            raise Exception(
                f"\033[91mAzure Content-Safety not installed, try running 'pip install azure-ai-contentsafety' to fix this error: {e}\n{traceback.format_exc()}\033[0m"
            )
        self.endpoint = endpoint
        self.api_key = api_key
        self.text_category = TextCategory
        self.analyze_text_options = AnalyzeTextOptions
        self.analyze_text_output_type = AnalyzeTextOutputType
        self.azure_http_error = HttpResponseError

        self.thresholds = self._configure_thresholds(thresholds)

        self.client = ContentSafetyClient(
            self.endpoint, AzureKeyCredential(self.api_key)
        )

    def _configure_thresholds(self, thresholds=None):
        default_thresholds = {
            self.text_category.HATE: 4,
            self.text_category.SELF_HARM: 4,
            self.text_category.SEXUAL: 4,
            self.text_category.VIOLENCE: 4,
        }

        if thresholds is None:
            return default_thresholds

        for key, default in default_thresholds.items():
            if key not in thresholds:
                thresholds[key] = default

        return thresholds

    def _compute_result(self, response):
        result = {}

        category_severity = {
            item.category: item.severity for item in response.categories_analysis
        }
        for category in self.text_category:
            severity = category_severity.get(category)
            if severity is not None:
                result[category] = {
                    "filtered": severity >= self.thresholds[category],
                    "severity": severity,
                }

        return result

    async def test_violation(self, content: str, source: Optional[str] = None):
        verbose_proxy_logger.debug("Testing Azure Content-Safety for: %s", content)

        # Construct a request
        request = self.analyze_text_options(
            text=content,
            output_type=self.analyze_text_output_type.EIGHT_SEVERITY_LEVELS,
        )

        # Analyze text
        try:
            response = await self.client.analyze_text(request)
        except self.azure_http_error:
            verbose_proxy_logger.debug(
                "Error in Azure Content-Safety: %s", traceback.format_exc()
            )
            verbose_proxy_logger.debug(traceback.format_exc())
            raise

        result = self._compute_result(response)
        verbose_proxy_logger.debug("Azure Content-Safety Result: %s", result)

        for key, value in result.items():
            if value["filtered"]:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated content safety policy",
                        "source": source,
                        "category": key,
                        "severity": value["severity"],
                    },
                )

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,  # "completion", "embeddings", "image_generation", "moderation"
    ):
        verbose_proxy_logger.debug("Inside Azure Content-Safety Pre-Call Hook")
        try:
            if call_type == "completion" and "messages" in data:
                for m in data["messages"]:
                    if "content" in m and isinstance(m["content"], str):
                        await self.test_violation(content=m["content"], source="input")

        except HTTPException as e:
            raise e
        except Exception as e:
            verbose_proxy_logger.error(
                "litellm.proxy.hooks.azure_content_safety.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            verbose_proxy_logger.debug(traceback.format_exc())

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        verbose_proxy_logger.debug("Inside Azure Content-Safety Post-Call Hook")
        if isinstance(response, litellm.ModelResponse) and isinstance(
            response.choices[0], litellm.utils.Choices
        ):
            await self.test_violation(
                content=response.choices[0].message.content or "", source="output"
            )

    # async def async_post_call_streaming_hook(
    #    self,
    #    user_api_key_dict: UserAPIKeyAuth,
    #    response: str,
    # ):
    #    verbose_proxy_logger.debug("Inside Azure Content-Safety Call-Stream Hook")
    #    await self.test_violation(content=response, source="output")
