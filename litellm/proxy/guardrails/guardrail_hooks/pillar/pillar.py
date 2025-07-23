# +-------------------------------------------------------------+
#
#            Pillar Security Guardrails
#           https://www.pillar.security/
#
# +-------------------------------------------------------------+

# Standard library imports
import os
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Tuple, Type, Union

# Third-party imports
from fastapi import HTTPException

# LiteLLM imports
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import LLMResponseTypes
from litellm._version import version as litellm_version


if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


# Exception classes
class PillarGuardrailMissingSecrets(Exception):
    """Exception raised when Pillar API key is missing."""

    pass


class PillarGuardrailAPIError(Exception):
    """Exception raised when there's an error calling the Pillar API."""

    pass


# Main guardrail class
class PillarGuardrail(CustomGuardrail):
    """
    Pillar Security Guardrail for LiteLLM.

    Provides comprehensive AI security scanning for input prompts and output responses
    using the Pillar Security API.
    """

    SUPPORTED_ON_FLAGGED_ACTIONS = ["block", "monitor"]
    DEFAULT_ON_FLAGGED_ACTION = "monitor"
    BASE_API_URL = "https://api.pillar.security"

    def __init__(
        self,
        guardrail_name: Optional[str] = "pillar-security",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        on_flagged_action: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Pillar guardrail.

        Args:
            guardrail_name: Name of the guardrail instance
            api_key: Pillar API key
            api_base: Pillar API base URL
            on_flagged_action: Action to take when content is flagged ('block' or 'monitor')
            **kwargs: Additional arguments passed to parent class
        """
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.api_key = api_key or os.environ.get("PILLAR_API_KEY")

        if self.api_key is None:
            msg = (
                "Couldn't get Pillar API key, either set the `PILLAR_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise PillarGuardrailMissingSecrets(msg)

        self.api_base = api_base or os.getenv("PILLAR_API_BASE") or self.BASE_API_URL

        # Validate and set on_flagged_action
        action = on_flagged_action or os.environ.get("PILLAR_ON_FLAGGED_ACTION")
        if action and action in self.SUPPORTED_ON_FLAGGED_ACTIONS:
            self.on_flagged_action = action
        else:
            if action:
                verbose_proxy_logger.warning(
                    f"Invalid action '{action}', using default"
                )
            self.on_flagged_action = self.DEFAULT_ON_FLAGGED_ACTION

        verbose_proxy_logger.debug(
            f"Pillar Guardrail: Initialized with on_flagged_action: {self.on_flagged_action}"
        )

        # Define supported event hooks
        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

    # =========================================================================
    # PUBLIC HOOK METHODS (Main Interface)
    # =========================================================================

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
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Pre-call hook to scan the request for security threats before sending to LLM.

        Args:
            user_api_key_dict: User API key authentication info
            cache: LiteLLM cache instance
            data: Request data
            call_type: Type of LLM call

        Returns:
            Original data if safe, raises HTTPException if blocked

        Raises:
            HTTPException: If request should be blocked due to security threats
        """
        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"Pillar Guardrail: Pre-call scanning disabled for {self.guardrail_name}"
            )
            return data

        verbose_proxy_logger.debug("Pillar Guardrail: Pre-call hook")
        result = await self.run_pillar_guardrail(data)

        # Add guardrail name to response headers
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return result

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
    ) -> Optional[Union[Exception, str, dict]]:
        """
        During-call hook to scan the request in parallel with LLM processing.

        Args:
            data: Request data
            user_api_key_dict: User API key authentication info
            call_type: Type of LLM call

        Returns:
            Original data if safe, raises HTTPException if blocked

        Raises:
            HTTPException: If request should be blocked due to security threats
        """
        event_type = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"Pillar Guardrail: During-call scanning disabled for {self.guardrail_name}"
            )
            return data

        verbose_proxy_logger.debug("Pillar Guardrail: During-call moderation hook")
        result = await self.run_pillar_guardrail(data)

        # Add guardrail name to response headers
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return result

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        """
        Post-call hook to scan LLM responses before returning to user.

        Args:
            data: Original request data
            user_api_key_dict: User API key authentication info
            response: LLM response to scan

        Returns:
            Original response if safe, raises HTTPException if blocked

        Raises:
            HTTPException: If response should be blocked due to security threats
        """
        event_type = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            verbose_proxy_logger.debug(
                f"Pillar Guardrail: Post-call scanning disabled for {self.guardrail_name}"
            )
            return response

        verbose_proxy_logger.debug("Pillar Guardrail: Post-call hook")

        # Extract response messages in the format Pillar expects
        response_dict = response.model_dump() if hasattr(response, "model_dump") else {}
        response_messages = [
            choice.get("message")
            for choice in response_dict.get("choices", [])
            if choice.get("message")
        ]

        if not response_messages:
            verbose_proxy_logger.debug(
                "Pillar Guardrail: No response content to scan, skipping post-call analysis"
            )
            return response

        # Create complete conversation: original messages + response messages
        post_call_data = data.copy()
        post_call_data["messages"] = data.get("messages", []) + response_messages

        # Reuse the existing guardrail logic - zero duplication!
        await self.run_pillar_guardrail(post_call_data)

        # Add guardrail name to response headers
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return response

    # =========================================================================
    # CORE LOGIC METHOD
    # =========================================================================

    async def run_pillar_guardrail(self, data: dict) -> dict:
        """
        Core method to run the Pillar guardrail scan.

        Args:
            data: Request data containing messages and metadata

        Returns:
            Original data if safe or in monitor mode

        Raises:
            PillarGuardrailAPIError: If the Pillar API call fails
            HTTPException: If content is flagged and action is 'block'
        """
        # Check if messages are present
        if not data.get("messages"):
            verbose_proxy_logger.debug(
                "Pillar Guardrail: No messages detected, bypassing security scan"
            )
            return data

        try:
            headers = self._prepare_headers()
            payload = self._prepare_payload(data)

            response = await self._call_pillar_api(
                headers=headers,
                payload=payload,
            )

            # Process the response - handles blocking or monitoring
            self._process_pillar_response(response, data)
            return data

        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            verbose_proxy_logger.error(
                f"Pillar Guardrail: API communication failed - {str(e)}"
            )
            raise PillarGuardrailAPIError(
                f"Pillar Guardrail scan failed - unable to verify request safety: {str(e)}"
            )

    # =========================================================================
    # PRIVATE HELPER METHODS (In logical order of usage)
    # =========================================================================

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare headers for the Pillar API request."""
        if not self.api_key:
            msg = (
                "Couldn't get Pillar API key, either set the `PILLAR_API_KEY` in the environment or "
                "pass it as a parameter to the guardrail in the config file"
            )
            raise PillarGuardrailMissingSecrets(msg)

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add Pillar-specific headers for enhanced response data
        headers["plr_evidence"] = "true"
        headers["plr_scanners"] = "true"

        return headers

    def _extract_model_and_provider(self, data: dict) -> Tuple[str, str]:
        """
        Extract the model and provider from the request data.

        Args:
            data: Request data

        Returns:
            Tuple of (model_name, provider_name)
        """
        model = data.get("model")
        if not model:
            return "unknown", "unknown"

        # Use LiteLLM's standard provider detection and model cleaning
        try:
            clean_model, provider, _, _ = get_llm_provider(
                model=model,
                custom_llm_provider=data.get("custom_llm_provider"),
                api_base=data.get("api_base"),
                api_key=data.get("api_key"),
            )
            return clean_model or "unknown", provider or "unknown"
        except Exception:
            # Fallback if get_llm_provider fails
            return (
                model or "unknown",
                data.get("custom_llm_provider") or data.get("provider") or "unknown",
            )

    def _prepare_payload(self, data: dict) -> Dict[str, Any]:
        """
        Prepare the payload for the Pillar API request following the /api/v1/protect contract.

        Args:
            data: Request data

        Returns:
            Formatted payload for Pillar API
        """
        messages = data.get("messages", [])
        tools = data.get("tools", [])
        metadata = {
            "source": "litellm",
            "version": litellm_version,
        }

        # Build payload following Pillar API format
        payload = {
            "messages": messages,
            "tools": tools,
            "metadata": metadata,
        }

        # User ID: use LiteLLM user field
        user_id = data.get("user")
        if user_id:
            payload["user_id"] = user_id

        # Session ID: use metadata.pillar_session_id if provided
        session_id = data.get("metadata", {}).get("pillar_session_id")
        if session_id:
            payload["session_id"] = session_id

        # Extract model and provider from actual request data
        model, provider = self._extract_model_and_provider(data)
        payload["model"] = model
        payload["provider"] = provider

        verbose_proxy_logger.debug(
            f"Pillar Guardrail: Request context - user={user_id}, session={session_id}, "
            f"model={model}, provider={provider}"
        )
        return payload

    async def _call_pillar_api(
        self, headers: Dict[str, str], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call the Pillar API and return the response.

        Args:
            headers: HTTP headers for the request
            payload: Request payload

        Returns:
            Pillar API response as dictionary
        """
        verbose_proxy_logger.debug(
            f"Pillar Guardrail: Scanning {len(payload.get('messages', []))} messages for security threats"
        )
        response = await self.async_handler.post(
            url=f"{self.api_base}/api/v1/protect",
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        res = response.json()

        flagged = res.get("flagged")
        session_id = res.get("session_id")
        verbose_proxy_logger.debug(
            f"Pillar Guardrail: Analysis complete - flagged={flagged}, session={session_id}"
        )
        return res

    def _process_pillar_response(
        self, pillar_response: Dict[str, Any], original_data: dict
    ) -> None:
        """
        Process the Pillar API response and handle detections based on configuration.

        Args:
            pillar_response: Response from Pillar API
            original_data: Original request data (modified in-place with session info)

        Raises:
            HTTPException: If content is flagged and action is 'block'
        """
        if not pillar_response:
            return

        flagged = pillar_response.get("flagged", False)

        # Store session_id from Pillar response for potential reuse
        pillar_session_id = pillar_response.get("session_id")
        if pillar_session_id:
            verbose_proxy_logger.debug(
                f"Pillar Guardrail: Received session_id from server: {pillar_session_id}"
            )
            # Store in request metadata for use in subsequent hooks
            if "metadata" not in original_data:
                original_data["metadata"] = {}
            if "pillar_session_id" not in original_data["metadata"]:
                original_data["metadata"]["pillar_session_id"] = pillar_session_id

        if flagged:
            verbose_proxy_logger.warning("Pillar Guardrail: Threat detected")
            if self.on_flagged_action == "block":
                self._raise_pillar_detection_exception(pillar_response)
            elif self.on_flagged_action == "monitor":
                verbose_proxy_logger.info(
                    "Pillar Guardrail: Monitoring mode - allowing flagged content to proceed"
                )

    def _raise_pillar_detection_exception(
        self, pillar_response: Dict[str, Any]
    ) -> None:
        """
        Raise an HTTPException for Pillar security detections.

        Args:
            pillar_response: Response from Pillar API containing detection details

        Raises:
            HTTPException: Always raises with security detection details
        """
        error_detail = {
            "error": "Blocked by Pillar Security Guardrail",
            "detection_message": "Security threats detected",
            "pillar_response": {
                "session_id": pillar_response.get("session_id"),
                "scanners": pillar_response.get("scanners", {}),
                "evidence": pillar_response.get("evidence", []),
            },
        }

        verbose_proxy_logger.warning(
            "Pillar Guardrail: Request blocked - Security threats detected"
        )

        raise HTTPException(status_code=400, detail=error_detail)

    # =========================================================================
    # STATIC/CLASS METHODS
    # =========================================================================

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """
        Get the configuration model for this guardrail.

        Returns:
            Pydantic model class for guardrail configuration
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.pillar import (
            PillarGuardrailConfigModel,
        )

        return PillarGuardrailConfigModel
