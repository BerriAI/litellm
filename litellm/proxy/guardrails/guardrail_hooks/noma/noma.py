# +-------------------------------------------------------------+
#
#           Noma Security Guardrail Integration for LiteLLM
#                       https://noma.security
#
# +-------------------------------------------------------------+

import copy
import os
from typing import Any, Dict, Literal, Optional, Union
from urllib.parse import urljoin

from fastapi import HTTPException

import litellm
from litellm import DualCache, ModelResponse
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import EmbeddingResponse, ImageResponse


class NomaBlockedMessage(HTTPException):
    """Exception raised when Noma guardrail blocks a message"""

    def __init__(self, classification_response: dict):
        classification = self._filter_triggered_classifications(classification_response)
        super().__init__(
            status_code=400,
            detail={
                "error": "Request blocked by Noma guardrail",
                "details": classification,
            },
        )

    def _filter_triggered_classifications(
        self,
        response_dict: dict,
    ) -> dict:
        """Filter and return only triggered classifications"""
        filtered_response = copy.deepcopy(response_dict)

        # Filter prompt classifications if present
        if filtered_response.get("prompt"):
            filtered_response["prompt"] = self.filter_classification_object(
                filtered_response["prompt"]
            )

        # Filter response classifications if present
        if filtered_response.get("response"):
            filtered_response["response"] = self.filter_classification_object(
                filtered_response["response"]
            )

        return filtered_response

    def filter_classification_object(
        self,
        classification_obj: dict,
    ) -> dict:
        """Filter classification object to only include triggered items"""
        if not classification_obj:
            return {}

        result = {}

        for key, value in classification_obj.items():
            if value is None:
                continue

            if key in [
                "allowedTopics",
                "bannedTopics",
                "topicGuardrails",
            ] and isinstance(value, dict):
                filtered_topics = {}
                for topic, topic_result in value.items():
                    if self._is_result_true(topic_result):
                        filtered_topics[topic] = topic_result

                if filtered_topics:
                    result[key] = filtered_topics

            elif key == "sensitiveData" and isinstance(value, dict):
                filtered_sensitive = {}
                for data_type, data_result in value.items():
                    if self._is_result_true(data_result):
                        filtered_sensitive[data_type] = data_result

                if filtered_sensitive:
                    result[key] = filtered_sensitive

            elif isinstance(value, dict) and "result" in value:
                if self._is_result_true(value):
                    result[key] = value

        return result

    def _is_result_true(self, result_obj: Optional[Dict[str, Any]]) -> bool:
        """
        Check if a result object has a "result" field that is True.

        Args:
            result_obj: A dictionary that may contain a "result" field

        Returns:
            True if the "result" field exists and is True, False otherwise
        """
        if not result_obj or not isinstance(result_obj, dict):
            return False

        return result_obj.get("result") is True


class NomaGuardrail(CustomGuardrail):
    """
    Noma Security Guardrail for LiteLLM

    This guardrail integrates with Noma Security's AI-DR API to provide
    content moderation and safety checks for LLM inputs and outputs.
    """

    _DEFAULT_API_BASE = "https://api.noma.security/"
    _AIDR_ENDPOINT = "/ai-dr/v1/prompt/scan/aggregate"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        application_id: Optional[str] = None,
        monitor_mode: Optional[bool] = None,
        block_failures: Optional[bool] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.api_key = api_key or os.environ.get("NOMA_API_KEY")
        self.api_base = api_base or os.environ.get(
            "NOMA_API_BASE", NomaGuardrail._DEFAULT_API_BASE
        )
        self.application_id = application_id or os.environ.get(
            "NOMA_APPLICATION_ID", "litellm"
        )

        if monitor_mode is None:
            self.monitor_mode = (
                os.environ.get("NOMA_MONITOR_MODE", "false").lower() == "true"
            )
        else:
            self.monitor_mode = monitor_mode

        if block_failures is None:
            self.block_failures = (
                os.environ.get("NOMA_BLOCK_FAILURES", "true").lower() == "true"
            )
        else:
            self.block_failures = block_failures

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
            "mcp_call",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        verbose_proxy_logger.debug("Running Noma pre-call hook")

        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.pre_call
            )
            is False
        ):
            return data

        try:
            return await self._check_user_message(data, user_api_key_dict)
        except NomaBlockedMessage:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Noma pre-call hook failed: {str(e)}")

            if self.block_failures and not self.monitor_mode:
                raise
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
            "responses",
            "mcp_call",
        ],
    ) -> Union[Exception, str, dict, None]:
        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        try:
            return await self._check_user_message(data, user_api_key_dict)
        except NomaBlockedMessage:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Noma moderation hook failed: {str(e)}")

            if self.block_failures and not self.monitor_mode:
                raise
            return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ):
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        try:
            return await self._check_llm_response(data, response, user_api_key_dict)
        except NomaBlockedMessage:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Noma post-call hook failed: {str(e)}")
            if self.block_failures and not self.monitor_mode:
                raise
            return response

    async def _check_user_message(
        self,
        request_data: dict,
        user_auth: UserAPIKeyAuth,
    ) -> Union[Exception, str, dict, None]:
        """Check user message for policy violations"""
        extra_data = self.get_guardrail_dynamic_request_body_params(request_data)

        user_message = await self._extract_user_message(request_data)
        if not user_message:
            return request_data

        payload = {"request": {"text": user_message}}
        response_json = await self._call_noma_api(
            payload=payload,
            llm_request_id=None,
            request_data=request_data,
            user_auth=user_auth,
            extra_data=extra_data,
        )
        await self._check_verdict("user", user_message, response_json)

        return request_data

    async def _check_llm_response(
        self,
        request_data: dict,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
        user_auth: UserAPIKeyAuth,
    ) -> Union[Exception, ModelResponse, Any]:
        """Check LLM response for policy violations"""
        extra_data = self.get_guardrail_dynamic_request_body_params(request_data)

        if not isinstance(response, litellm.ModelResponse):
            return response

        content = None
        for choice in response.choices:
            if isinstance(choice, litellm.Choices) and choice.message.content:
                content = choice.message.content
                break

        if not content or not isinstance(content, str):
            return response

        payload = {"response": {"text": content}}

        response_json = await self._call_noma_api(
            payload=payload,
            llm_request_id=response.id,
            request_data=request_data,
            user_auth=user_auth,
            extra_data=extra_data,
        )
        await self._check_verdict("assistant", content, response_json)

        return response

    async def _extract_user_message(self, data: dict) -> Optional[str]:
        """Extract the last user message from request data"""
        messages = data.get("messages", [])
        if not messages:
            return None

        # Get the last user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            return None

        last_user_message = user_messages[-1].get("content", "")
        if not last_user_message or not isinstance(last_user_message, str):
            return None

        return last_user_message

    async def _call_noma_api(
        self,
        payload: dict,
        llm_request_id: Optional[str],
        request_data: dict,
        user_auth: UserAPIKeyAuth,
        extra_data: dict,
    ) -> dict:
        call_id = request_data.get("litellm_call_id")
        headers = {
            "X-Noma-AIDR-Application-ID": self.application_id,
            **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            **({"X-Noma-Request-ID": call_id} if call_id else {}),
        }
        endpoint = urljoin(
            self.api_base or "https://api.noma.security/", NomaGuardrail._AIDR_ENDPOINT
        )

        response = await self.async_handler.post(
            endpoint,
            headers=headers,
            json={
                **payload,
                "context": {
                    "applicationId": extra_data.get("application_id")
                    or request_data.get("metadata", {})
                    .get("headers", {})
                    .get("x-noma-application-id"),
                    "ipAddress": request_data.get("metadata", {}).get(
                        "requester_ip_address", None
                    ),
                    "userId": user_auth.user_email
                    if user_auth.user_email
                    else user_auth.user_id,
                    "sessionId": call_id,
                    "requestId": llm_request_id,
                },
            },
        )
        response.raise_for_status()

        return response.json()

    async def _check_verdict(
        self,
        type: Literal["user", "assistant"],
        message: str,
        response_json: dict,
    ) -> None:
        """
        Check the verdict from the Noma API and raise an exception if needed
        """
        if not response_json.get("verdict", True):
            msg = str.format(
                "Noma guardrail blocked {type} message: {message}",
                type=type,
                message=message,
            )

            if self.monitor_mode:
                verbose_proxy_logger.warning(msg)
            else:
                verbose_proxy_logger.debug(msg)
                original_response = response_json.get("originalResponse", {})
                raise NomaBlockedMessage(original_response)
        else:
            msg = str.format(
                "Noma guardrail allowed {type} message: {message}",
                type=type,
                message=message,
            )
            if self.monitor_mode:
                verbose_proxy_logger.info(msg)
            else:
                verbose_proxy_logger.debug(msg)
