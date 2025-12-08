# +-------------------------------------------------------------+
#
#           Use Lasso Security Guardrails for your LLM calls
#                   https://www.lasso.security/
#
# +-------------------------------------------------------------+

import os
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, Union, TypedDict

try:
    import ulid

    ULID_AVAILABLE = True
except ImportError:
    ulid = None  # type: ignore
    ULID_AVAILABLE = False

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.integrations.custom_guardrail import dc as global_cache

from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
import litellm


class LassoResponse(TypedDict):
    """Type definition for Lasso API response."""

    violations_detected: bool
    deputies: Dict[str, bool]
    findings: Dict[str, List[Dict[str, Any]]]
    messages: Optional[List[Dict[str, str]]]


if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


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
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        mask: Optional[bool] = False,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.lasso_api_key = lasso_api_key or api_key or os.environ.get("LASSO_API_KEY")
        self.user_id = user_id or os.environ.get("LASSO_USER_ID")
        self.conversation_id = conversation_id or os.environ.get("LASSO_CONVERSATION_ID")
        self.mask = mask or False

        if self.lasso_api_key is None:
            raise LassoGuardrailMissingSecrets(
                "Couldn't get Lasso api key, either set the `LASSO_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )

        self.api_base = (
            api_base or os.getenv("LASSO_API_BASE") or "https://server.lasso.security/gateway/v3"
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
        if ULID_AVAILABLE and ulid is not None:
            return str(ulid.new())  # type: ignore
        else:
            verbose_proxy_logger.debug("ULID library not available, using UUID")
            return str(uuid.uuid4())

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,       # Deprecated, use global_cache instead (kept to align with CustomGuardrail interface)
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
            "anthropic_messages",
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
        # The conversation_id is being stored in the cache so it can be used by the post_call hook
        self._get_or_generate_conversation_id(data, global_cache)

        return await self._run_lasso_guardrail(data, global_cache, message_type="PROMPT")

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
            "mcp_call",
            "anthropic_messages",
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
                response_data = {
                    "messages": response_messages,
                    "litellm_call_id": data.get("litellm_call_id"),
                }


                # Handle masking for post-call
                if self.mask:
                    headers = self._prepare_headers(response_data, global_cache)
                    payload = self._prepare_payload(response_messages, response_data, global_cache, "COMPLETION")
                    api_url = f"{self.api_base}/classifix"

                    try:
                        lasso_response = await self._call_lasso_api(headers=headers, payload=payload, api_url=api_url)
                        self._process_lasso_response(lasso_response)

                        # Apply masking to the actual response if masked content is available
                        masked_messages = lasso_response.get("messages")
                        if lasso_response.get("violations_detected") and masked_messages:
                            self._apply_masking_to_model_response(response, masked_messages)
                            verbose_proxy_logger.debug("Applied Lasso masking to model response")
                    except Exception as e:
                        if isinstance(e, HTTPException):
                            raise e
                        verbose_proxy_logger.error(f"Error in post-call Lasso masking: {str(e)}")
                        raise LassoGuardrailAPIError(f"Failed to apply post-call masking: {str(e)}")
                else:
                    # Use the same data for conversation_id consistency (no cache access needed)
                    await self._run_lasso_guardrail(response_data, cache=global_cache, message_type="COMPLETION")
                    verbose_proxy_logger.debug("Post-call Lasso validation completed")
            else:
                verbose_proxy_logger.warning("No response messages found to validate")
        else:
            verbose_proxy_logger.warning(f"Unexpected response type for post-call hook: {type(response)}")

        return response

    def _get_or_generate_conversation_id(self, data: dict, cache: DualCache) -> str:
        """
        Get or generate a conversation_id for this request.

        This method ensures session consistency by using litellm_call_id as a cache key.
        The same conversation_id is used for both pre-call and post-call hooks within
        the same request, enabling proper conversation grouping in Lasso UI.

        Example:
            >>> guardrail = LassoGuardrail(lasso_api_key="key")
            >>> data = {"litellm_call_id": "call_123"}
            >>> conversation_id = guardrail._get_or_generate_conversation_id(data, cache)
            >>> # Returns consistent ID for same litellm_call_id

        Args:
            data: The request data containing litellm_call_id
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
        cache: DualCache,
        message_type: Literal["PROMPT", "COMPLETION"] = "PROMPT",
    ):
        """
        Run the Lasso guardrail with the specified message type.

        This is the core method that handles both classification and masking workflows.
        It chooses the appropriate API endpoint based on the masking configuration
        and processes the response according to Lasso's action-based system.

        Workflow:
        1. Validate messages are present
        2. Prepare headers and payload
        3. Choose API endpoint (classify vs classifix)
        4. Call Lasso API
        5. Process response and apply masking if needed
        6. Handle blocking vs non-blocking violations

        Args:
            data: The request data containing messages
            cache: The cache instance for storing conversation_id (optional for post-call)
            message_type: Either "PROMPT" for input or "COMPLETION" for output

        Raises:
            LassoGuardrailAPIError: If the Lasso API call fails
            HTTPException: If blocking violations are detected
        """
        messages: List[Dict[str, str]] = data.get("messages", [])
        if not messages:
            return data

        if self.mask:
            return await self._handle_masking(data, cache, message_type, messages)
        else:
            return await self._handle_classification(data, cache, message_type, messages)

    async def _handle_classification(
        self,
        data: dict,
        cache: DualCache,
        message_type: Literal["PROMPT", "COMPLETION"],
        messages: List[Dict[str, str]],
    ) -> dict:
        """Handle classification without masking."""
        try:
            headers = self._prepare_headers(data, cache)
            payload = self._prepare_payload(messages, data, cache, message_type)
            response = await self._call_lasso_api(headers=headers, payload=payload)
            self._process_lasso_response(response)
            return data
        except Exception as e:
            await self._handle_api_error(e, message_type)
            return data  # This line won't be reached due to exception, but satisfies type checker

    async def _handle_masking(
        self,
        data: dict,
        cache: DualCache,
        message_type: Literal["PROMPT", "COMPLETION"],
        messages: List[Dict[str, str]],
    ) -> dict:
        """Handle masking with classifix endpoint."""
        try:
            headers = self._prepare_headers(data, cache)
            payload = self._prepare_payload(messages, data, cache, message_type)
            api_url = f"{self.api_base}/classifix"
            response = await self._call_lasso_api(headers=headers, payload=payload, api_url=api_url)
            self._process_lasso_response(response)

            # Apply masking to messages if violations detected and masked messages are available
            if response.get("violations_detected") and response.get("messages"):
                data["messages"] = response["messages"]
                self._log_masking_applied(message_type, dict(response))

            return data
        except Exception as e:
            await self._handle_api_error(e, message_type)
            return data  # This line won't be reached due to exception, but satisfies type checker

    async def _handle_api_error(
        self,
        error: Exception,
        message_type: Literal["PROMPT", "COMPLETION"],
    ) -> None:
        """Handle API errors with specific error types."""
        if isinstance(error, HTTPException):
            raise error

        # Log error with context
        verbose_proxy_logger.error(
            f"Error calling Lasso API: {str(error)}",
            extra={
                "guardrail_name": getattr(self, "guardrail_name", "unknown"),
                "message_type": message_type,
                "error_type": type(error).__name__,
            },
        )

        # Handle specific error types if httpx is available
        if HTTPX_AVAILABLE:
            if isinstance(error, httpx.TimeoutException):
                raise LassoGuardrailAPIError("Lasso API timeout")
            elif isinstance(error, httpx.HTTPStatusError):
                if error.response.status_code == 401:
                    raise LassoGuardrailMissingSecrets("Invalid API key")
                elif error.response.status_code == 429:
                    raise LassoGuardrailAPIError("Lasso API rate limit exceeded")
                else:
                    raise LassoGuardrailAPIError(f"API error: {error.response.status_code}")

        # Generic error handling
        raise LassoGuardrailAPIError(f"Failed to verify request safety with Lasso API: {str(error)}")

    def _log_masking_applied(
        self,
        message_type: Literal["PROMPT", "COMPLETION"],
        response: Dict[str, Any],
    ) -> None:
        """Log masking application with structured context."""
        conversation_id = getattr(self, "conversation_id", "unknown")
        verbose_proxy_logger.debug(
            "Lasso masking applied",
            extra={
                "guardrail_name": getattr(self, "guardrail_name", "unknown"),
                "message_type": message_type,
                "violations_count": len(response.get("findings", {})),
                "masked_fields": len(response.get("messages", [])),
                "conversation_id": conversation_id,
            },
        )

    def _prepare_headers(self, data: dict, cache: DualCache) -> Dict[str, str]:
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
        conversation_id = self._get_or_generate_conversation_id(data, cache)        

        headers["lasso-conversation-id"] = conversation_id

        return headers

    def _prepare_payload(
        self,
        messages: List[Dict[str, str]],
        data: dict,
        cache: DualCache,
        message_type: Literal["PROMPT", "COMPLETION"] = "PROMPT",
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
        conversation_id = self._get_or_generate_conversation_id(data, cache)

        payload["sessionId"] = conversation_id

        return payload

    async def _call_lasso_api(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        api_url: Optional[str] = None,
    ) -> LassoResponse:
        """Call the Lasso API and return the response."""
        url = api_url or f"{self.api_base}/classify"
        verbose_proxy_logger.debug(f"Calling Lasso API with messageType: {payload.get('messageType')}")
        response = await self.async_handler.post(
            url=url,
            headers=headers,
            json=payload,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()

    def _process_lasso_response(self, response: LassoResponse) -> None:
        """
        Process the Lasso API response and handle violations according to action types.

        This method implements the action-based blocking logic:
        - BLOCK: Raises HTTPException to stop request/response
        - AUTO_MASKING: Logs warning and continues (masking applied elsewhere)
        - WARN: Logs warning and continues

        Example Response:
            {
                "violations_detected": true,
                "findings": {
                    "jailbreak": [{
                        "action": "BLOCK",
                        "severity": "HIGH"
                    }]
                }
            }

        Args:
            response: The response dictionary from Lasso API

        Raises:
            HTTPException: If any finding has "action": "BLOCK"
        """
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

    def _check_for_blocking_actions(self, response: LassoResponse) -> List[str]:
        """
        Check findings for actions that should block the request/response.

        Examines the findings section of the Lasso response to identify which
        deputies have violations with "BLOCK" action. This enables granular
        control where some violations (like PII) can be masked while others
        (like jailbreaks) are blocked entirely.

        Args:
            response: The response dictionary from Lasso API

        Returns:
            List[str]: Names of deputies with blocking violations

        Example:
            >>> response = {
            ...     "findings": {
            ...         "jailbreak": [{"action": "BLOCK"}],
            ...         "pattern-detection": [{"action": "AUTO_MASKING"}]
            ...     }
            ... }
            >>> guardrail._check_for_blocking_actions(response)
            ['jailbreak']
        """
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

    def _parse_violated_deputies(self, response: LassoResponse) -> List[str]:
        """Parse the response to extract violated deputies."""
        violated_deputies = []
        if "deputies" in response:
            for deputy, is_violated in response["deputies"].items():
                if is_violated:
                    violated_deputies.append(deputy)
        return violated_deputies

    def _apply_masking_to_model_response(
        self,
        model_response: litellm.ModelResponse,
        masked_messages: List[Dict[str, str]],
    ) -> None:
        """Apply masking to the actual model response when mask=True and masked content is available."""
        masked_index = 0
        for choice in model_response.choices:
            if hasattr(choice, "message") and choice.message.content and masked_index < len(masked_messages):
                # Replace the content with the masked version from Lasso
                choice.message.content = masked_messages[masked_index]["content"]
                masked_index += 1
                verbose_proxy_logger.debug(f"Applied masked content to choice {masked_index}")

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.lasso import (
            LassoGuardrailConfigModel,
        )

        return LassoGuardrailConfigModel
