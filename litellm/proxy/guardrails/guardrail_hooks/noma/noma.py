# +-------------------------------------------------------------+
#
#           Noma Security Guardrail Integration for LiteLLM
#                       https://noma.security
#
# +-------------------------------------------------------------+

import asyncio
import json
import os
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Type,
    Union,
)
from urllib.parse import urljoin

from fastapi import HTTPException

import litellm
from litellm import DualCache, ModelResponse
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.main import stream_chunk_builder
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    CallTypesLiteral,
    EmbeddingResponse,
    GuardrailStatus,
    ImageResponse,
    ModelResponseStream,
    TextCompletionResponse,
)

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
        super().__init__(
            status_code=400,
            detail={
                "error": "Request blocked by Noma guardrail",
                "details": classification_response,
            },
        )

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
    _AIDR_ENDPOINT = "/ai-dr/v2/prompt/scan"

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
        start_time = datetime.now()
        extra_data = self.get_guardrail_dynamic_request_body_params(request_data)

        user_message = await self._extract_user_message(request_data)
        if not user_message:
            return None

        payload = {
            "input": [{"type": "message", "role": "user", "content": user_message}]
        }
        response_json = await self._call_noma_api(
            payload=payload,
            llm_request_id=None,
            request_data=request_data,
            user_auth=user_auth,
            extra_data=extra_data,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Determine guardrail status based on response
        guardrail_status = self._determine_guardrail_status(response_json)

        # Always log guardrail information for consistency
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider="noma",
            guardrail_json_response=response_json,
            request_data=request_data,
            guardrail_status=guardrail_status,
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=duration,
        )

        if self.monitor_mode:
            await self._handle_verdict_background(
                USER_ROLE, json.dumps(user_message), response_json
            )
            return json.dumps(user_message)

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

        await self._check_verdict(USER_ROLE, json.dumps(user_message), response_json)
        return json.dumps(user_message)

    async def _process_llm_response_check(
        self,
        request_data: dict,
        response: LLMResponse,
        user_auth: UserAPIKeyAuth,
    ) -> Optional[str]:
        """Shared logic for processing LLM response checks"""

        start_time = datetime.now()
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

        payload = {
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": content}],
                }
            ]
        }

        response_json = await self._call_noma_api(
            payload=payload,
            llm_request_id=response.id,
            request_data=request_data,
            user_auth=user_auth,
            extra_data=extra_data,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Determine guardrail status based on response
        guardrail_status = self._determine_guardrail_status(response_json)

        # Always log guardrail information for consistency
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider="noma",
            guardrail_json_response=response_json,
            request_data=request_data,
            guardrail_status=guardrail_status,
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=duration,
        )

        if self.monitor_mode:
            await self._handle_verdict_background(
                ASSISTANT_ROLE, json.dumps(content), response_json
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

    def _determine_guardrail_status(self, response_json: dict) -> GuardrailStatus:
        """
        Determine the guardrail status based on NOMA API response.

        Args:
            response_json: Response from NOMA API

        Returns:
            "success": Content allowed through with no violations
            "guardrail_intervened": Content blocked due to policy violations
            "guardrail_failed_to_respond": Technical error or API failure
        """
        try:
            # Check if we got a valid response structure
            if not isinstance(response_json, dict):
                return "guardrail_failed_to_respond"

            # Get the aggregatedScanResult from the response
            # aggregatedScanResult=True means unsafe (block), False means safe (allow)
            aggregated_scan_result = response_json.get("aggregatedScanResult", False)

            # If aggregatedScanResult is False, content is safe/allowed
            if aggregated_scan_result is False:
                return "success"

            # If aggregatedScanResult is True, content is blocked/flagged
            if aggregated_scan_result is True:
                return "guardrail_intervened"

            # If aggregatedScanResult is missing or invalid, treat as failure
            return "guardrail_failed_to_respond"

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error determining NOMA guardrail status: {str(e)}"
            )
            return "guardrail_failed_to_respond"

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
        # Extract from new scanResult structure
        scan_result = response_json.get("scanResult", [])
        if not scan_result:
            return None

        # Find the scan result matching the message type (role)
        for result_item in scan_result:
            if result_item.get("role") == message_type:
                return (
                    result_item.get("results", {})
                    .get("anonymizedContent", {})
                    .get("anonymized", "")
                )

        return None

    def _should_anonymize(self, response_json: dict, message_type: MessageRole) -> bool:
        """
        Determine if content should be anonymized based on Noma API response.

        Logic:
        - If aggregatedScanResult=False: Content is safe, anonymize if anonymized version exists
        - If aggregatedScanResult=True: Check if only sensitiveData detectors have result=True
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

        # aggregatedScanResult=False means safe, True means unsafe
        aggregated_scan_result = response_json.get("aggregatedScanResult", False)

        # If aggregatedScanResult is False, content is safe - anonymize if available
        if not aggregated_scan_result:
            return True

        # If aggregatedScanResult is True (unsafe), check if only sensitive data detectors triggered
        scan_result = response_json.get("scanResult", [])
        if not scan_result:
            return False

        if not isinstance(scan_result, list) or len(scan_result) == 0:
            return False

        for result_item in scan_result:
            if result_item.get("role") == message_type:
                return self._should_only_sensitive_data_failed(
                    result_item.get("results", {})
                )

        return False

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
        """Handle aggregatedScanResult from Noma API in background - logging only, never blocks
        aggregatedScanResult=True means unsafe, False means safe
        """
        try:
            # aggregatedScanResult=True means blocked, False means allowed
            aggregated_scan_result = response_json.get("aggregatedScanResult", False)

            if aggregated_scan_result:  # True = unsafe
                msg = f"Noma guardrail blocked {type} message: {message}"
                verbose_proxy_logger.warning(msg)
            else:  # False = safe
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
        call_type: CallTypesLiteral,
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
            # Blocked requests were already logged in _process_user_message_check with "blocked" status
            raise
        except Exception as e:
            # Log technical failures
            from datetime import datetime

            start_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="noma",
                guardrail_json_response=str(e),
                request_data=data,
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=start_time.timestamp(),
                duration=0.0,
            )

            verbose_proxy_logger.error(f"Noma pre-call hook failed: {str(e)}")

            if self.block_failures:
                raise
            return data

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
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
            # Blocked requests were already logged in _process_user_message_check with "blocked" status
            raise
        except Exception as e:
            # Log technical failures
            from datetime import datetime

            start_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="noma",
                guardrail_json_response=str(e),
                request_data=data,
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=start_time.timestamp(),
                duration=0.0,
            )

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
            # Blocked requests were already logged in _process_llm_response_check with "blocked" status
            raise
        except Exception as e:
            # Log technical failures
            from datetime import datetime

            start_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="noma",
                guardrail_json_response=str(e),
                request_data=data,
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=start_time.timestamp(),
                duration=0.0,
            )

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
    ) -> Any:
        """Check LLM response for policy violations"""
        content = await self._process_llm_response_check(
            request_data, response, user_auth
        )
        if not content:
            return response

        return response

    async def _extract_user_message(self, data: dict) -> Optional[List[dict]]:
        """Extract the last user message from request data"""
        messages = data.get("messages", [])
        if not messages:
            return None

        # Get the last user message
        user_messages = [msg for msg in messages if msg.get("role") == USER_ROLE]
        if not user_messages:
            return None

        last_user_message = user_messages[-1].get("content", "")
        if isinstance(last_user_message, str):
            return [{"type": "input_text", "text": last_user_message}]
        elif isinstance(last_user_message, list):
            converted_messages = []
            for message in last_user_message:
                converted_message = self._convert_single_user_message_to_payload(
                    message
                )
                if converted_message is not None:
                    converted_messages.append(converted_message)
            return converted_messages
        else:
            return None

    def _convert_single_user_message_to_payload(
        self, user_message: Any
    ) -> Optional[dict]:
        if isinstance(user_message, str):
            return {"type": "input_text", "text": user_message}
        elif user_message.get("type", "") == "image_url":
            return {
                "type": "input_image",
                "image_url": user_message.get("image_url", {}).get("url", ""),
            }
        elif user_message.get("type", "") == "text":
            return {"type": "input_text", "text": user_message.get("text", "")}
        else:
            return None

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
                "x-noma-context": {
                    "applicationId": extra_data.get("application_id")
                    or request_data.get("metadata", {})
                    .get("headers", {})
                    .get("x-noma-application-id")
                    or self.application_id,
                    "ipAddress": request_data.get("metadata", {}).get(
                        "requester_ip_address", None
                    ),
                    "userId": (
                        user_auth.user_email
                        if user_auth.user_email
                        else user_auth.user_id
                    ),
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
        Check the aggregatedScanResult from the Noma API and raise an exception if needed.
        aggregatedScanResult=True means unsafe (block), False means safe (allow)
        """
        # aggregatedScanResult=True means blocked, False means allowed
        aggregated_scan_result = response_json.get("aggregatedScanResult", False)

        if aggregated_scan_result:  # True = unsafe, block it
            msg = f"Noma guardrail blocked {type} message: {message}"

            if self.monitor_mode:
                verbose_proxy_logger.warning(msg)
            else:
                verbose_proxy_logger.debug(msg)
                original_response = response_json.get("scanResult", {})
                # Use the full response as the original response for error details
                raise NomaBlockedMessage(original_response)
        else:  # False = safe, allow it
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

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Process streaming response chunks with Noma guardrail."""

        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        if not all_chunks:
            return

        assembled_model_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = stream_chunk_builder(chunks=all_chunks)

        if isinstance(assembled_model_response, ModelResponse):
            try:
                processed_response = await self._check_llm_response(
                    request_data, assembled_model_response, user_api_key_dict
                )
            except NomaBlockedMessage:
                raise
            except Exception as e:
                if self.block_failures:
                    raise
                verbose_proxy_logger.error(
                    f"Noma streaming post-call hook failed: {str(e)}"
                )
                for chunk in all_chunks:
                    yield chunk
                return

            mock_response = MockResponseIterator(model_response=processed_response)
            async for chunk in mock_response:
                yield chunk
            return

        for chunk in all_chunks:
            yield chunk
