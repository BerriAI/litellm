# +-------------------------------------------------------------+
#
#            Pillar Security Guardrails
#           https://www.pillar.security/
#
# +-------------------------------------------------------------+

# Standard library imports
import json
import os
from urllib.parse import quote
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type, Union

# Third-party imports
from fastapi import HTTPException

# LiteLLM imports
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
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
    get_metadata_variable_name_from_kwargs,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import LLMResponseTypes

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

MAX_PILLAR_HEADER_VALUE_BYTES = 8 * 1024


def _encode_json_for_header(data: Any) -> str:
    """
    JSON-serialize and URL-encode data for safe header transmission.
    """
    json_payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return quote(json_payload, safe="")


def _truncate_evidence_payload(
    evidence: Any, max_bytes: int = MAX_PILLAR_HEADER_VALUE_BYTES
) -> Tuple[Any, str, bool]:
    """
    Truncate evidence payload so the encoded header value stays within max_bytes.

    Returns:
        truncated_evidence: Evidence list/value after truncation
        encoded_value: URL-encoded JSON string for header
        was_truncated: Whether truncation occurred
    """
    if not isinstance(evidence, list):
        encoded = _encode_json_for_header(evidence)
        if len(encoded.encode("utf-8")) <= max_bytes:
            return evidence, encoded, False
        truncated_value = "[truncated]"
        return truncated_value, _encode_json_for_header(truncated_value), True

    truncated: List[Any] = []
    encoded = _encode_json_for_header(truncated)
    truncated_flag = False

    for entry in evidence:
        working_entry: Any
        if isinstance(entry, dict):
            working_entry = dict(entry)
        else:
            working_entry = entry

        truncated.append(working_entry)
        encoded = _encode_json_for_header(truncated)

        if len(encoded.encode("utf-8")) <= max_bytes:
            continue

        truncated_flag = True
        if isinstance(working_entry, dict):
            evidence_text = str(working_entry.get("evidence", ""))
            if evidence_text:
                step = max(1, len(evidence_text) // 2)
                while len(encoded.encode("utf-8")) > max_bytes and evidence_text:
                    evidence_text = (
                        evidence_text[:-step] if len(evidence_text) > step else evidence_text[:-1]
                    )
                    step = max(1, step // 2)
                    truncated_text = (
                        f"{evidence_text}...[truncated]" if evidence_text else "[truncated]"
                    )
                    working_entry["evidence"] = truncated_text
                    working_entry["evidence_truncated"] = True
                    encoded = _encode_json_for_header(truncated)

                if len(encoded.encode("utf-8")) <= max_bytes:
                    continue

        truncated.pop()
        encoded = _encode_json_for_header(truncated)

    return truncated, encoded, truncated_flag


def build_pillar_response_headers(metadata_store: Dict[str, Any]) -> Dict[str, str]:
    """
    Create URL-safe Pillar response headers and apply truncation metadata.
    """
    headers: Dict[str, str] = {}

    if "pillar_flagged" in metadata_store:
        headers["x-pillar-flagged"] = str(metadata_store["pillar_flagged"]).lower()

    if "pillar_scanners" in metadata_store:
        headers["x-pillar-scanners"] = _encode_json_for_header(metadata_store["pillar_scanners"])

    if "pillar_evidence" in metadata_store:
        truncated_evidence, encoded_value, truncated_flag = _truncate_evidence_payload(
            metadata_store["pillar_evidence"]
        )
        metadata_store["pillar_evidence"] = truncated_evidence
        if truncated_flag:
            metadata_store["pillar_evidence_truncated"] = True
        headers["x-pillar-evidence"] = encoded_value

    if "pillar_session_id_response" in metadata_store:
        headers["x-pillar-session-id"] = quote(
            str(metadata_store["pillar_session_id_response"]), safe=""
        )

    if headers:
        metadata_store["pillar_response_headers"] = headers

    return headers


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
    SUPPORTED_FALLBACK_ACTIONS = ["allow", "block"]
    DEFAULT_FALLBACK_ACTION = "allow"
    BASE_API_URL = "https://api.pillar.security"
    DEFAULT_TIMEOUT = 5.0  # 5 seconds - fast failure detection with graceful degradation

    def __init__(
        self,
        guardrail_name: Optional[str] = "pillar-security",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        on_flagged_action: Optional[str] = None,
        async_mode: Optional[bool] = None,
        persist_session: Optional[bool] = None,
        include_scanners: Optional[bool] = None,
        include_evidence: Optional[bool] = None,
        fallback_on_error: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the Pillar guardrail.

        Args:
            guardrail_name: Name of the guardrail instance
            api_key: Pillar API key
            api_base: Pillar API base URL
            on_flagged_action: Action to take when content is flagged ('block' or 'monitor')
            fallback_on_error: Action when API errors occur ('allow' or 'block')
            timeout: Timeout for API calls in seconds
            **kwargs: Additional arguments passed to parent class

        Note:
            LiteLLM virtual key context (user_id, team_id, key_alias, etc.) is always
            automatically passed as X-LiteLLM-* headers to enable application/user tracking.
        """
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
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
                verbose_proxy_logger.warning(f"Invalid action '{action}', using default")
            self.on_flagged_action = self.DEFAULT_ON_FLAGGED_ACTION

        verbose_proxy_logger.debug(f"Pillar Guardrail: Initialized with on_flagged_action: {self.on_flagged_action}")

        self.async_mode = self._resolve_bool_config(
            provided_value=async_mode,
            env_var="PILLAR_ASYNC",
            default=None,
            setting_name="async_mode",
        )
        self.persist_session = self._resolve_bool_config(
            provided_value=persist_session,
            env_var="PILLAR_PERSIST",
            default=None,
            setting_name="persist_session",
        )
        self.include_scanners = self._resolve_bool_config(
            provided_value=include_scanners,
            env_var="PILLAR_INCLUDE_SCANNERS",
            default=True,
            setting_name="include_scanners",
        )
        self.include_evidence = self._resolve_bool_config(
            provided_value=include_evidence,
            env_var="PILLAR_INCLUDE_EVIDENCE",
            default=True,
            setting_name="include_evidence",
        )

        # Validate and set fallback_on_error
        action = fallback_on_error or os.environ.get("PILLAR_FALLBACK_ON_ERROR")
        if action and action in self.SUPPORTED_FALLBACK_ACTIONS:
            self.fallback_on_error = action
        else:
            if action:
                verbose_proxy_logger.warning(
                    f"Invalid fallback action '{action}', using default '{self.DEFAULT_FALLBACK_ACTION}'"
                )
            self.fallback_on_error = self.DEFAULT_FALLBACK_ACTION

        verbose_proxy_logger.debug(f"Pillar Guardrail: Initialized with fallback_on_error: {self.fallback_on_error}")

        # Set timeout with graceful fallback on invalid configuration
        if timeout is not None:
            self.timeout = timeout
        else:
            try:
                self.timeout = float(os.environ.get("PILLAR_TIMEOUT", str(self.DEFAULT_TIMEOUT)))
            except (ValueError, TypeError):
                verbose_proxy_logger.warning(
                    f"Pillar Guardrail: Invalid PILLAR_TIMEOUT value '{os.environ.get('PILLAR_TIMEOUT')}', "
                    f"falling back to default {self.DEFAULT_TIMEOUT}s"
                )
                self.timeout = self.DEFAULT_TIMEOUT

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
            "mcp_call",
            "anthropic_messages",
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
            verbose_proxy_logger.debug(f"Pillar Guardrail: Pre-call scanning disabled for {self.guardrail_name}")
            return data

        verbose_proxy_logger.debug("Pillar Guardrail: Pre-call hook")
        result = await self.run_pillar_guardrail(data, user_api_key_dict)

        # Add guardrail name to response headers
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

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
            "mcp_call",
            "anthropic_messages",
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
            verbose_proxy_logger.debug(f"Pillar Guardrail: During-call scanning disabled for {self.guardrail_name}")
            return data

        verbose_proxy_logger.debug("Pillar Guardrail: During-call moderation hook")
        result = await self.run_pillar_guardrail(data, user_api_key_dict)

        # Add guardrail name to response headers
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

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
            verbose_proxy_logger.debug(f"Pillar Guardrail: Post-call scanning disabled for {self.guardrail_name}")
            return response

        verbose_proxy_logger.debug("Pillar Guardrail: Post-call hook")

        # Extract response messages in the format Pillar expects
        response_dict = response.model_dump() if hasattr(response, "model_dump") else {}  # type: ignore[union-attr]
        response_messages = [
            choice.get("message") for choice in response_dict.get("choices", []) if choice.get("message")
        ]

        if not response_messages:
            verbose_proxy_logger.debug("Pillar Guardrail: No response content to scan, skipping post-call analysis")
            return response

        # Create complete conversation: original messages + response messages
        post_call_data = data.copy()
        post_call_data["messages"] = data.get("messages", []) + response_messages

        # Reuse the existing guardrail logic - zero duplication!
        await self.run_pillar_guardrail(post_call_data, user_api_key_dict)

        # Add guardrail name to response headers
        add_guardrail_to_applied_guardrails_header(request_data=data, guardrail_name=self.guardrail_name)

        return response

    # =========================================================================
    # CORE LOGIC METHOD
    # =========================================================================

    async def run_pillar_guardrail(self, data: dict, user_api_key_dict: UserAPIKeyAuth) -> dict:
        """
        Core method to run the Pillar guardrail scan.

        Args:
            data: Request data containing messages and metadata
            user_api_key_dict: User API key authentication info containing key context

        Returns:
            Original data if safe or in monitor mode

        Raises:
            HTTPException: If content is flagged and action is 'block', or if API fails and fallback_on_error is 'block'
        """
        # Check if messages are present
        if not data.get("messages"):
            verbose_proxy_logger.debug("Pillar Guardrail: No messages detected, bypassing security scan")
            return data

        try:
            headers = self._prepare_headers(user_api_key_dict)
            payload = self._prepare_payload(data)

            response = await self._call_pillar_api(
                headers=headers,
                payload=payload,
            )

            # Process the response - handles blocking or monitoring
            self._process_pillar_response(response, data)
            return data

        except Exception as e:
            # If it's already an HTTPException from content being flagged, re-raise it
            if isinstance(e, HTTPException):
                raise e

            # Handle API communication errors based on fallback_on_error setting
            verbose_proxy_logger.error(f"Pillar Guardrail: API communication failed - {str(e)}")

            return self._handle_api_error(e, data)

    # =========================================================================
    # PRIVATE HELPER METHODS (In logical order of usage)
    # =========================================================================

    def _handle_api_error(self, error: Exception, data: dict) -> dict:
        """
        Handle API errors based on fallback_on_error configuration.

        Args:
            error: The exception that occurred during API communication
            data: Original request data

        Returns:
            Original data if fallback_on_error is 'allow'

        Raises:
            HTTPException: If fallback_on_error is 'block'
        """
        if self.fallback_on_error == "allow":
            verbose_proxy_logger.warning(
                "Pillar Guardrail: API unavailable, proceeding without scanning (fallback_on_error=allow)"
            )
            return data
        else:  # fallback_on_error == "block"
            verbose_proxy_logger.warning(
                "Pillar Guardrail: API unavailable, blocking request (fallback_on_error=block)"
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Pillar Security Guardrail Unavailable",
                    "message": "Security scanning service is temporarily unavailable and fallback is set to block",
                    "original_error": str(error),
                },
            )

    def _prepare_headers(self, user_api_key_dict: UserAPIKeyAuth) -> Dict[str, str]:
        """
        Prepare headers for the Pillar API request.

        Args:
            user_api_key_dict: User API key authentication info containing key context

        Returns:
            Dictionary of headers to send to Pillar API
        """
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

        # Add Pillar-specific headers based on configuration
        self._set_bool_header(headers, "plr_scanners", self.include_scanners)
        self._set_bool_header(headers, "plr_evidence", self.include_evidence)
        self._set_bool_header(headers, "plr_async", self.async_mode)
        self._set_bool_header(headers, "plr_persist", self.persist_session)

        # Always add LiteLLM virtual key context headers (metadata excluded for security)
        context_mapping = {
            "X-LiteLLM-Key-Name": user_api_key_dict.key_name,
            "X-LiteLLM-Key-Alias": user_api_key_dict.key_alias,
            "X-LiteLLM-User-Id": user_api_key_dict.user_id,
            "X-LiteLLM-User-Email": user_api_key_dict.user_email,
            "X-LiteLLM-Team-Id": user_api_key_dict.team_id,
            "X-LiteLLM-Team-Name": user_api_key_dict.team_alias,
            "X-LiteLLM-Org-Id": user_api_key_dict.org_id,
        }
        for header_name, value in context_mapping.items():
            if value:
                headers[header_name] = str(value)

        return headers

    def _set_bool_header(self, headers: Dict[str, str], header_name: str, value: Optional[bool]) -> None:
        """Apply a boolean value as a lowercase string HTTP header when provided."""

        if value is None:
            return
        headers[header_name] = "true" if value else "false"

    def _resolve_bool_config(
        self,
        provided_value: Optional[Union[bool, str, int]],
        env_var: Optional[str],
        default: Optional[bool],
        setting_name: str,
    ) -> Optional[bool]:
        """Resolve configuration precedence: explicit value -> environment -> default."""

        if provided_value is not None:
            try:
                return self._parse_bool_value(provided_value)
            except ValueError:
                verbose_proxy_logger.warning(
                    "Pillar Guardrail: Invalid boolean value '%s' for %s, falling back to default.",
                    provided_value,
                    setting_name,
                )
                return default

        if env_var:
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    return self._parse_bool_value(env_value)
                except ValueError:
                    verbose_proxy_logger.warning(
                        "Pillar Guardrail: Invalid boolean env value '%s' for %s, falling back to default.",
                        env_value,
                        env_var,
                    )
                    return default

        return default

    @staticmethod
    def _parse_bool_value(value: Union[bool, str, int]) -> bool:
        """Normalise various truthy/falsey inputs to a strict boolean."""

        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return bool(value)

        value_str = str(value).strip().lower()
        if value_str in {"true", "1", "yes", "y", "on"}:
            return True
        if value_str in {"false", "0", "no", "n", "off"}:
            return False
        raise ValueError(f"Unrecognised boolean value: {value}")

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

        This method supports multi-modal content (images, files, audio, video, etc.) as messages
        are passed through without modification. The messages array can contain any OpenAI-compatible
        message structure including:
        - Text content (string)
        - Multi-modal content blocks (image_url, image_file, audio, video, document, file)
        - Attachments
        - Tool calls

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

    async def _call_pillar_api(self, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
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
            timeout=self.timeout,
        )
        response.raise_for_status()
        res = response.json()

        flagged = res.get("flagged")
        session_id = res.get("session_id")
        verbose_proxy_logger.debug(f"Pillar Guardrail: Analysis complete - flagged={flagged}, session={session_id}")
        return res

    def _process_pillar_response(self, pillar_response: Dict[str, Any], original_data: dict) -> None:
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

        metadata_field = get_metadata_variable_name_from_kwargs(original_data)
        if metadata_field not in original_data or not isinstance(original_data.get(metadata_field), dict):
            original_data[metadata_field] = {}
        metadata_store = original_data[metadata_field]

        # Backwards compatibility - ensure metadata alias exists when different key used
        if metadata_field != "metadata":
            if "metadata" not in original_data or not isinstance(original_data.get("metadata"), dict):
                original_data["metadata"] = metadata_store

        # Store session_id from Pillar response for potential reuse
        pillar_session_id = pillar_response.get("session_id")
        if pillar_session_id:
            verbose_proxy_logger.debug(f"Pillar Guardrail: Received session_id from server: {pillar_session_id}")
            # Store in request metadata for use in subsequent hooks
            if "pillar_session_id" not in metadata_store:
                metadata_store["pillar_session_id"] = pillar_session_id
            metadata_store["pillar_session_id_response"] = pillar_session_id

        # Always set flagged status and scanner/evidence data for monitor mode
        metadata_store["pillar_flagged"] = flagged
        if self.include_scanners:
            metadata_store["pillar_scanners"] = pillar_response.get("scanners", {})
        if self.include_evidence:
            metadata_store["pillar_evidence"] = pillar_response.get("evidence", [])

        if flagged:
            verbose_proxy_logger.warning("Pillar Guardrail: Threat detected")
            if self.on_flagged_action == "block":
                self._raise_pillar_detection_exception(pillar_response)
            elif self.on_flagged_action == "monitor":
                verbose_proxy_logger.info("Pillar Guardrail: Monitoring mode - allowing flagged content to proceed")

        build_pillar_response_headers(metadata_store)

    def _raise_pillar_detection_exception(self, pillar_response: Dict[str, Any]) -> None:
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

        verbose_proxy_logger.warning("Pillar Guardrail: Request blocked - Security threats detected")

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
