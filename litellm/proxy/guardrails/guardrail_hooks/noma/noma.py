# +-------------------------------------------------------------+
#
#           Noma Security Guardrail Integration for LiteLLM
#                       https://noma.security
#
# +-------------------------------------------------------------+

import asyncio
import copy
import os
from typing import Any, Dict, Final, Literal, Optional, Union, Type, TYPE_CHECKING
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

# Constants
USER_ROLE: Final[Literal["user"]] = "user"
ASSISTANT_ROLE: Final[Literal["assistant"]] = "assistant"
SENSITIVE_DATA_DETECTOR_KEYS: Final[list[str]] = ["sensitiveData", "dataDetector"]

# Type aliases
MessageRole = Literal["user", "assistant"]
LLMResponse = Union[Any, ModelResponse, EmbeddingResponse, ImageResponse]

if TYPE_CHECKING:
  from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

  
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
                "topicDetector",  # Mock name for tests
            ] and isinstance(value, dict):
                filtered_topics = {}
                for topic, topic_result in value.items():
                    if self._is_result_true(topic_result):
                        filtered_topics[topic] = topic_result

                if filtered_topics:
                    result[key] = filtered_topics

            elif key in SENSITIVE_DATA_DETECTOR_KEYS and isinstance(value, dict):
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
        anonymize_input: Optional[bool] = None,
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

        if anonymize_input is None:
            self.anonymize_input = (
                os.environ.get("NOMA_ANONYMIZE_INPUT", "false").lower() == "true"
            )
        else:
            self.anonymize_input = anonymize_input

        super().__init__(**kwargs)

    def _create_background_noma_check(
        self,
        coro,
    ) -> None:
        """Create a background task for Noma API calls without blocking the main flow"""
        try:
            asyncio.create_task(coro)
        except Exception as e:
            verbose_proxy_logger.error(
                f"Failed to create background Noma task: {str(e)}"
            )

    async def _process_user_message_check(
        self,
        request_data: dict,
        user_auth: UserAPIKeyAuth,
    ) -> Optional[str]:
        """Shared logic for processing user message checks"""
        extra_data = self.get_guardrail_dynamic_request_body_params(request_data)

        user_message = await self._extract_user_message(request_data)
        if not user_message:
            return None

        payload = {"request": {"text": user_message}}
        response_json = await self._call_noma_api(
            payload=payload,
            llm_request_id=None,
            request_data=request_data,
            user_auth=user_auth,
            extra_data=extra_data,
        )

        if self.monitor_mode:
            await self._handle_verdict_background(
                USER_ROLE, user_message, response_json
            )
            return user_message

        # Check if we should anonymize content
        if self._should_anonymize(response_json, USER_ROLE):
            anonymized_content = self._extract_anonymized_content(
                response_json, USER_ROLE
            )
            if anonymized_content:
                # Replace the user message content with anonymized version
                self._replace_user_message_content(request_data, anonymized_content)
                verbose_proxy_logger.debug(
                    f"Noma guardrail anonymized user message: {anonymized_content}"
                )
                return anonymized_content

        await self._check_verdict(USER_ROLE, user_message, response_json)
        return user_message

    async def _process_llm_response_check(
        self,
        request_data: dict,
        response: LLMResponse,
        user_auth: UserAPIKeyAuth,
    ) -> Optional[str]:
        """Shared logic for processing LLM response checks"""
        extra_data = self.get_guardrail_dynamic_request_body_params(request_data)

        if not isinstance(response, litellm.ModelResponse):
            return None

        content = None
        for choice in response.choices:
            if isinstance(choice, litellm.Choices) and choice.message.content:
                content = choice.message.content
                break

        if not content or not isinstance(content, str):
            return None

        payload = {"response": {"text": content}}

        response_json = await self._call_noma_api(
            payload=payload,
            llm_request_id=response.id,
            request_data=request_data,
            user_auth=user_auth,
            extra_data=extra_data,
        )

        if self.monitor_mode:
            await self._handle_verdict_background(
                ASSISTANT_ROLE, content, response_json
            )
            return content

        # Check if we should anonymize content
        if self._should_anonymize(response_json, ASSISTANT_ROLE):
            anonymized_content = self._extract_anonymized_content(
                response_json, ASSISTANT_ROLE
            )
            if anonymized_content:
                # Replace the LLM response content with anonymized version
                self._replace_llm_response_content(response, anonymized_content)
                verbose_proxy_logger.debug(
                    f"Noma guardrail anonymized LLM response: {anonymized_content}"
                )
                return anonymized_content

        await self._check_verdict(ASSISTANT_ROLE, content, response_json)
        return content

    def _should_only_sensitive_data_failed(self, classification_obj: dict) -> bool:
        """
        Check if only sensitive data detectors (PII, PCI, secrets) have result=true in the classification.

        Args:
            classification_obj: The prompt or response classification object from Noma API

        Returns:
            True if only sensitiveData detectors have result=true, False otherwise
        """
        if not classification_obj:
            return False

        # Track which detectors have result=true (detected violations)
        failed_detectors = []
        sensitive_data_detected = False

        for key, value in classification_obj.items():
            if key in SENSITIVE_DATA_DETECTOR_KEYS and isinstance(value, dict):
                # Check if any sensitive data detector has result=true
                for data_type, data_result in value.items():
                    if self._is_result_true(data_result):
                        sensitive_data_detected = True
                        # Don't add to failed_detectors as we want to allow these

            elif isinstance(value, dict) and "result" in value:
                # Check other detectors - these should NOT have result=true
                if self._is_result_true(value):
                    failed_detectors.append(key)

            elif isinstance(value, dict):
                # Handle nested detectors
                for nested_key, nested_value in value.items():
                    if self._is_result_true(nested_value):
                        failed_detectors.append(f"{key}.{nested_key}")

        # Return True only if sensitive data was detected AND no other detectors have result=true
        return sensitive_data_detected and len(failed_detectors) == 0

    def _extract_anonymized_content(
        self, response_json: dict, message_type: MessageRole
    ) -> Optional[str]:
        """
        Extract anonymized content from Noma API response.

        Args:
            response_json: The full response from Noma API
            message_type: Either 'user' or 'assistant' to determine which content to extract

        Returns:
            The anonymized content string if available, None otherwise
        """
        original_response = response_json.get("originalResponse", {})

        if message_type == USER_ROLE:
            prompt_data = original_response.get("prompt", {})
            anonymized_data = prompt_data.get("anonymizedContent", {})
            return anonymized_data.get("anonymized")
        elif message_type == ASSISTANT_ROLE:
            response_data = original_response.get("response", {})
            anonymized_data = response_data.get("anonymizedContent", {})
            return anonymized_data.get("anonymized")

        return None

    def _should_anonymize(self, response_json: dict, message_type: MessageRole) -> bool:
        """
        Determine if content should be anonymized based on Noma API response.

        Logic:
        - If verdict=True: Content is safe, anonymize if anonymized version exists
        - If verdict=False: Check if only sensitiveData detectors have result=True
          - If yes: Anonymize
          - If no: Block (other violations detected)

        Args:
            response_json: The full response from Noma API
            message_type: Either 'user' or 'assistant' to determine which classification to check

        Returns:
            True if content should be anonymized, False if it should be blocked
        """
        # Only anonymize in blocking mode when anonymize_input is enabled
        if self.monitor_mode or not self.anonymize_input:
            return False

        verdict = response_json.get("verdict", True)
        # If verdict is True, anonymize (content is considered safe)
        if verdict:
            return True

        # If verdict is False, check if only sensitive data detectors have result=True
        original_response = response_json.get("originalResponse", {})

        if message_type == USER_ROLE:
            classification_obj = original_response.get("prompt", {})
        elif message_type == ASSISTANT_ROLE:
            classification_obj = original_response.get("response", {})
        else:
            return False

        # Anonymize only if solely sensitive data (PII/PCI/secrets) was detected
        return self._should_only_sensitive_data_failed(classification_obj)

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

    def _replace_user_message_content(
        self, request_data: dict, anonymized_content: str
    ):
        """
        Replace the user message content in request data with anonymized version.

        Args:
            request_data: The original request data
            anonymized_content: The anonymized content to replace with
        """
        messages = request_data.get("messages", [])
        if not messages:
            return

        # Find and replace the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == USER_ROLE:
                messages[i]["content"] = anonymized_content
                break

    def _replace_llm_response_content(
        self, response: LLMResponse, anonymized_content: str
    ):
        """
        Replace the LLM response content with anonymized version.

        Args:
            response: The original LLM response
            anonymized_content: The anonymized content to replace with
        """
        if not isinstance(response, litellm.ModelResponse):
            return

        # Replace content in all choices
        for choice in response.choices:
            if isinstance(choice, litellm.Choices) and choice.message.content:
                choice.message.content = anonymized_content

    async def _check_user_message_background(
        self,
        request_data: dict,
        user_auth: UserAPIKeyAuth,
    ) -> None:
        """Check user message in background for monitor mode - non-blocking"""
        try:
            await self._process_user_message_check(request_data, user_auth)
        except Exception as e:
            verbose_proxy_logger.error(
                f"Noma background user message check failed: {str(e)}"
            )

    async def _check_llm_response_background(
        self,
        request_data: dict,
        response: LLMResponse,
        user_auth: UserAPIKeyAuth,
    ) -> None:
        """Check LLM response in background for monitor mode - non-blocking"""
        try:
            await self._process_llm_response_check(request_data, response, user_auth)
        except Exception as e:
            verbose_proxy_logger.error(
                f"Noma background response check failed: {str(e)}"
            )

    async def _handle_verdict_background(
        self,
        type: MessageRole,
        message: str,
        response_json: dict,
    ) -> None:
        """Handle verdict from Noma API in background - logging only, never blocks"""
        try:
            if not response_json.get("verdict", True):
                msg = f"Noma guardrail blocked {type} message: {message}"
                verbose_proxy_logger.warning(msg)
            else:
                msg = f"Noma guardrail allowed {type} message: {message}"
                verbose_proxy_logger.info(msg)
        except Exception as e:
            verbose_proxy_logger.error(
                f"Noma background verdict handling failed: {str(e)}"
            )

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

        # In monitor mode, run Noma check in background and return immediately
        if self.monitor_mode:
            try:
                self._create_background_noma_check(
                    self._check_user_message_background(data, user_api_key_dict)
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Failed to start background Noma pre-call check: {str(e)}"
                )
            return data

        try:
            return await self._check_user_message(data, user_api_key_dict)
        except NomaBlockedMessage:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Noma pre-call hook failed: {str(e)}")

            if self.block_failures:
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

        # In monitor mode, run Noma check in background and return immediately
        if self.monitor_mode:
            try:
                self._create_background_noma_check(
                    self._check_user_message_background(data, user_api_key_dict)
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Failed to start background Noma moderation check: {str(e)}"
                )
            return data

        try:
            return await self._check_user_message(data, user_api_key_dict)
        except NomaBlockedMessage:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Noma moderation hook failed: {str(e)}")

            if self.block_failures:
                raise
            return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponse,
    ):
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        # In monitor mode, run Noma check in background and return immediately
        if self.monitor_mode:
            try:
                self._create_background_noma_check(
                    self._check_llm_response_background(
                        data, response, user_api_key_dict
                    )
                )
            except Exception as e:
                verbose_proxy_logger.error(
                    f"Failed to start background Noma post-call check: {str(e)}"
                )
            return response

        try:
            return await self._check_llm_response(data, response, user_api_key_dict)
        except NomaBlockedMessage:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Noma post-call hook failed: {str(e)}")
            if self.block_failures:
                raise
            return response

    async def _check_user_message(
        self,
        request_data: dict,
        user_auth: UserAPIKeyAuth,
    ) -> Union[Exception, str, dict, None]:
        """Check user message for policy violations"""
        user_message = await self._process_user_message_check(request_data, user_auth)
        if not user_message:
            return request_data

        return request_data

    async def _check_llm_response(
        self,
        request_data: dict,
        response: LLMResponse,
        user_auth: UserAPIKeyAuth,
    ) -> Union[Exception, ModelResponse, Any]:
        """Check LLM response for policy violations"""
        content = await self._process_llm_response_check(
            request_data, response, user_auth
        )
        if not content:
            return response

        return response

    async def _extract_user_message(self, data: dict) -> Optional[str]:
        """Extract the last user message from request data"""
        messages = data.get("messages", [])
        if not messages:
            return None

        # Get the last user message
        user_messages = [msg for msg in messages if msg.get("role") == USER_ROLE]
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
        type: MessageRole,
        message: str,
        response_json: dict,
    ) -> None:
        """
        Check the verdict from the Noma API and raise an exception if needed
        """
        if not response_json.get("verdict", True):
            msg = f"Noma guardrail blocked {type} message: {message}"

            if self.monitor_mode:
                verbose_proxy_logger.warning(msg)
            else:
                verbose_proxy_logger.debug(msg)
                original_response = response_json.get("originalResponse", {})
                raise NomaBlockedMessage(original_response)
        else:
            msg = f"Noma guardrail allowed {type} message: {message}"
            if self.monitor_mode:
                verbose_proxy_logger.info(msg)
            else:
                verbose_proxy_logger.debug(msg)
    
    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.noma import (
            NomaGuardrailConfigModel,
        )

        return NomaGuardrailConfigModel

