# +-------------------------------------------------------------+
#
#           Use Lasso Security Guardrails for your LLM calls
#                   https://www.lasso.security/
#
# +-------------------------------------------------------------+

import os
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

try:
    from ulid import ULID

    ULID_AVAILABLE = True
except ImportError:
    ULID_AVAILABLE = False

from fastapi import HTTPException

from litellm import DualCache
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
from litellm.types.guardrails import GuardrailEventHooks
import litellm


class LassoGuardrailMissingSecrets(Exception):
    """Exception raised when Lasso API key is missing."""

    pass


class LassoGuardrailAPIError(Exception):
    """Exception raised when there's an error calling the Lasso API."""

    pass


class LassoGuardrail(CustomGuardrail):
    """
    Lasso Security Guardrail integration for LiteLLM.

    Provides content moderation, PII detection, and policy enforcement
    through the Lasso Security API.
    """

    def __init__(
        self,
        lasso_api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        mask: Optional[bool] = False,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.lasso_api_key = lasso_api_key or os.environ.get("LASSO_API_KEY")
        self.user_id = user_id or os.environ.get("LASSO_USER_ID")
        self.conversation_id = conversation_id or os.environ.get("LASSO_CONVERSATION_ID")
        self.api_base = api_base or "https://server.lasso.security/gateway/v3/classify"
        self.mask = mask or False

        if self.lasso_api_key is None:
            raise LassoGuardrailMissingSecrets(
                "Couldn't get Lasso api key, either set the `LASSO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )

        verbose_proxy_logger.debug(
            f"Lasso guardrail initialized: {kwargs.get('guardrail_name', 'unknown')}, "
            f"event_hook: {kwargs.get('event_hook', 'unknown')}, mask: {self.mask}"
        )

        super().__init__(**kwargs)

    def _generate_ulid(self) -> str:
        """
        Generate a ULID (Universally Unique Lexicographically Sortable Identifier).
        Falls back to UUID if ULID library is not available.
        """
        if ULID_AVAILABLE:
            return str(ULID())
        else:
            verbose_proxy_logger.debug("ULID library not available, using UUID")
            return str(uuid.uuid4())

    @log_guardrail_information
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
        """
        Runs before the LLM API call to validate and potentially modify input.
        Uses 'PROMPT' messageType as this is input to the model.
        """
        # Check if this guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        # Get or generate conversation_id and store it in data for post-call consistency
        conversation_id = self._get_or_generate_conversation_id(data, cache)
        data.setdefault("_lasso_internal", {})["conversation_id"] = conversation_id

        return await self._run_lasso_guardrail(data, cache, message_type="PROMPT")

    @log_guardrail_information
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
        ],
        cache: DualCache,
    ):
        """
        This is used for during_call moderation.
        Uses 'PROMPT' messageType as this runs concurrently with input processing.
        """
        # Check if this guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        return await self._run_lasso_guardrail(data, cache, message_type="PROMPT")

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """
        Runs after the LLM API call to validate the response.
        Uses 'COMPLETION' messageType as this is output from the model.
        """
        # Check if this guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        # Extract messages from the response for validation
        if isinstance(response, litellm.ModelResponse):
            response_messages = []
            for choice in response.choices:
                if hasattr(choice, "message") and choice.message.content:
                    response_messages.append({"role": "assistant", "content": choice.message.content})

            if response_messages:
                # Include litellm_call_id from original data for conversation_id consistency
                response_data = {"messages": response_messages, "litellm_call_id": data.get("litellm_call_id")}

                # Copy stored conversation_id from pre-call hook
                if data.get("_lasso_internal", {}).get("conversation_id"):
                    response_data.setdefault("_lasso_internal", {})["conversation_id"] = data["_lasso_internal"][
                        "conversation_id"
                    ]

                # Handle masking for post-call
                if self.mask:
                    headers = self._prepare_headers(response_data)
                    payload = self._prepare_payload(response_messages, "COMPLETION", response_data)
                    api_url = self.api_base.replace("/gateway/v3/classify", "/gateway/v1/classifix")

                    try:
                        lasso_response = await self._call_lasso_api(headers=headers, payload=payload, api_url=api_url)
                        self._process_lasso_response(lasso_response)

                        # Apply masking to the actual response if masked content is available
                        if lasso_response.get("violations_detected") and lasso_response.get("messages"):
                            self._apply_masking_to_model_response(response, lasso_response["messages"])
                            verbose_proxy_logger.debug("Applied Lasso masking to model response")
                    except Exception as e:
                        if isinstance(e, HTTPException):
                            raise e
                        verbose_proxy_logger.error(f"Error in post-call Lasso masking: {str(e)}")
                        raise LassoGuardrailAPIError(f"Failed to apply post-call masking: {str(e)}")
                else:
                    # Use the same data for conversation_id consistency (no cache access needed)
                    await self._run_lasso_guardrail(response_data, cache=None, message_type="COMPLETION")
                    verbose_proxy_logger.debug("Post-call Lasso validation completed")
            else:
                verbose_proxy_logger.warning("No response messages found to validate")
        else:
            verbose_proxy_logger.warning(f"Unexpected response type for post-call hook: {type(response)}")

        return response

    def _get_or_generate_conversation_id(self, data: dict, cache: DualCache) -> str:
        """
        Get or generate a conversation_id for this request.

        Uses litellm_call_id to ensure the same conversation_id is used for both pre-call
        and post-call hooks within the same request, enabling proper conversation grouping in Lasso UI.

        Args:
            data: The request data
            cache: The cache instance for storing conversation_id

        Returns:
            str: The conversation_id to use for this request
        """
        # Use global conversation_id if set
        if self.conversation_id:
            return self.conversation_id

        # Get the litellm_call_id which is consistent across all hooks for this request
        litellm_call_id = data.get("litellm_call_id")

        if not litellm_call_id:
            # Fallback to generating a new ULID if no litellm_call_id available
            return self._generate_ulid()

        # Use litellm_call_id as cache key for conversation_id
        cache_key = f"lasso_conversation_id:{litellm_call_id}"

        # Try to get existing conversation_id from cache
        try:
            cached_conversation_id = cache.get_cache(cache_key)
            if cached_conversation_id:
                return cached_conversation_id
        except Exception as e:
            verbose_proxy_logger.warning(f"Cache retrieval failed: {e}")

        # Generate new conversation_id and store in cache
        generated_id = self._generate_ulid()

        try:
            cache.set_cache(cache_key, generated_id, ttl=3600)  # Cache for 1 hour
        except Exception as e:
            verbose_proxy_logger.warning(f"Cache storage failed: {e}")

        return generated_id

    async def _run_lasso_guardrail(
        self,
        data: dict,
        cache: Optional[DualCache],
        message_type: Literal["PROMPT", "COMPLETION"] = "PROMPT",
    ):
        """
        Run the Lasso guardrail with the specified message type.

        Args:
            data: The request data containing messages
            cache: The cache instance for storing conversation_id (optional for post-call)
            message_type: Either "PROMPT" for input or "COMPLETION" for output

        Raises:
            LassoGuardrailAPIError: If the Lasso API call fails
        """
        messages: List[Dict[str, str]] = data.get("messages", [])
        if not messages:
            return data

        try:
            headers = self._prepare_headers(data, cache)
            payload = self._prepare_payload(messages, message_type, data, cache)

            # Use classifix endpoint when masking is enabled
            if self.mask:
                api_url = self.api_base.replace("/gateway/v3/classify", "/gateway/v1/classifix")
                response = await self._call_lasso_api(headers=headers, payload=payload, api_url=api_url)
                self._process_lasso_response(response)

                # Apply masking to messages if violations detected and masked messages are available
                if response.get("violations_detected") and response.get("messages"):
                    data["messages"] = response["messages"]
                    verbose_proxy_logger.debug("Applied Lasso masking to messages")
            else:
                response = await self._call_lasso_api(headers=headers, payload=payload)
                self._process_lasso_response(response)

            return data
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            verbose_proxy_logger.error(f"Error calling Lasso API: {str(e)}")
            raise LassoGuardrailAPIError(f"Failed to verify request safety with Lasso API: {str(e)}")

    def _prepare_headers(self, data: dict, cache: Optional[DualCache] = None) -> Dict[str, str]:
        """Prepare headers for the Lasso API request."""
        if not self.lasso_api_key:
            raise LassoGuardrailMissingSecrets(
                "Couldn't get Lasso api key, either set the `LASSO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )

        headers: Dict[str, str] = {
            "lasso-api-key": self.lasso_api_key,
            "Content-Type": "application/json",
        }

        # Add optional headers if provided
        if self.user_id:
            headers["lasso-user-id"] = self.user_id

        # Always include conversation_id (generated or provided)
        if cache is not None:
            conversation_id = self._get_or_generate_conversation_id(data, cache)
        else:
            # For post-call hook, use stored conversation_id or generate a new one
            conversation_id = (
                data.get("_lasso_internal", {}).get("conversation_id") or self.conversation_id or self._generate_ulid()
            )

        headers["lasso-conversation-id"] = conversation_id

        return headers

    def _prepare_payload(
        self,
        messages: List[Dict[str, str]],
        message_type: Literal["PROMPT", "COMPLETION"] = "PROMPT",
        data: dict = None,
        cache: Optional[DualCache] = None,
    ) -> Dict[str, Any]:
        """
        Prepare the payload for the Lasso API request.

        Args:
            messages: List of message objects
            message_type: Type of message - "PROMPT" for input, "COMPLETION" for output
            data: Request data (used for conversation_id generation)
            cache: Cache instance for storing conversation_id (optional for post-call)
        """
        payload: Dict[str, Any] = {"messages": messages, "messageType": message_type}

        # Add optional parameters if available
        if self.user_id:
            payload["userId"] = self.user_id

        # Always include sessionId (conversation_id - generated or provided)
        if data is not None:
            if cache is not None:
                conversation_id = self._get_or_generate_conversation_id(data, cache)
            else:
                # For post-call hook, use stored conversation_id or fallback
                conversation_id = (
                    data.get("_lasso_internal", {}).get("conversation_id")
                    or self.conversation_id
                    or self._generate_ulid()
                )

            payload["sessionId"] = conversation_id
        elif self.conversation_id:
            payload["sessionId"] = self.conversation_id

        return payload

    async def _call_lasso_api(
        self, headers: Dict[str, str], payload: Dict[str, Any], api_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Call the Lasso API and return the response."""
        url = api_url or self.api_base
        verbose_proxy_logger.debug(f"Calling Lasso API with messageType: {payload.get('messageType')}")
        response = await self.async_handler.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()

    def _process_lasso_response(self, response: Dict[str, Any]) -> None:
        """Process the Lasso API response and raise exceptions if violations are detected."""
        if response and response.get("violations_detected") is True:
            violated_deputies = self._parse_violated_deputies(response)
            verbose_proxy_logger.warning(f"Lasso guardrail detected violations: {violated_deputies}")

            # Check if any findings have "BLOCK" action
            blocking_violations = self._check_for_blocking_actions(response)

            if blocking_violations:
                # Block the request/response for findings with "BLOCK" action
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated Lasso guardrail policy",
                        "detection_message": f"Blocking violations detected: {', '.join(blocking_violations)}",
                        "lasso_response": response,
                    },
                )
            else:
                # Continue with warning for non-blocking violations (e.g., AUTO_MASKING)
                verbose_proxy_logger.info(
                    f"Non-blocking Lasso violations detected, continuing with warning: {violated_deputies}"
                )

    def _check_for_blocking_actions(self, response: Dict[str, Any]) -> List[str]:
        """Check findings for actions that should block the request/response."""
        blocking_violations = []
        findings = response.get("findings", {})

        for deputy_name, deputy_findings in findings.items():
            if isinstance(deputy_findings, list):
                for finding in deputy_findings:
                    if isinstance(finding, dict) and finding.get("action") == "BLOCK":
                        if deputy_name not in blocking_violations:
                            blocking_violations.append(deputy_name)
                        break  # No need to check other findings for this deputy

        return blocking_violations

    def _parse_violated_deputies(self, response: Dict[str, Any]) -> List[str]:
        """Parse the response to extract violated deputies."""
        violated_deputies = []
        if "deputies" in response:
            for deputy, is_violated in response["deputies"].items():
                if is_violated:
                    violated_deputies.append(deputy)
        return violated_deputies

    def _apply_masking_to_model_response(
        self, model_response: litellm.ModelResponse, masked_messages: List[Dict[str, str]]
    ) -> None:
        """Apply masking to the actual model response when mask=True and masked content is available."""
        masked_index = 0
        for choice in model_response.choices:
            if hasattr(choice, "message") and choice.message.content and masked_index < len(masked_messages):
                # Replace the content with the masked version from Lasso
                choice.message.content = masked_messages[masked_index]["content"]
                masked_index += 1
                verbose_proxy_logger.debug(f"Applied masked content to choice {masked_index}")
