#!/usr/bin/env python3
"""
Palo Alto Networks Prisma AI Runtime Security (AIRS) Guardrail Integration for LiteLLM

Provides real-time threat detection, DLP, URL filtering, content masking, and policy enforcement for AI applications.
"""

import os
import httpx
from datetime import datetime
from litellm._uuid import uuid
from litellm.caching import DualCache
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type

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
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypesLiteral, ModelResponse

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class PanwPrismaAirsHandler(CustomGuardrail):
    """
    LiteLLM Built-in Guardrail for Palo Alto Networks Prisma AI Runtime Security (AIRS).

    Scans prompts and responses using PANW Prisma AIRS API to detect malicious content,
    injection attempts, and policy violations. Supports content masking and fail-closed error handling.

    Configuration:
        guardrail_name: Name of the guardrail instance
        api_key: PANW Prisma AIRS API key
        api_base: PANW Prisma AIRS API endpoint (default: https://service.api.aisecurity.paloaltonetworks.com)
        profile_name: PANW security profile name (optional if API key has linked profile)
        app_name: Application name for tracking in Prisma AIRS analytics (default: "LiteLLM")
        mask_request_content: Apply masking to prompts (default: False)
        mask_response_content: Apply masking to responses (default: False)
        mask_on_block: Backwards compatible flag that enables both request and response masking
    """

    def __init__(
        self,
        guardrail_name: str,
        profile_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        default_on: bool = True,
        mask_on_block: bool = False,
        mask_request_content: bool = False,
        mask_response_content: bool = False,
        app_name: Optional[str] = None,
        fallback_on_error: Literal["block", "allow"] = "block",
        timeout: float = 10.0,
        violation_message_template: Optional[str] = None,
        **kwargs,
    ):
        """Initialize PANW Prisma AIRS guardrail handler."""

        # Masking configuration - mask_on_block enables both for backwards compatibility
        self.mask_on_block = mask_on_block
        _mask_request_content = mask_request_content or mask_on_block
        _mask_response_content = mask_response_content or mask_on_block

        # Initialize parent CustomGuardrail with masking flags
        super().__init__(
            guardrail_name=guardrail_name,
            default_on=default_on,
            mask_request_content=_mask_request_content,
            mask_response_content=_mask_response_content,
            violation_message_template=violation_message_template,
            **kwargs,
        )

        # Store configuration with env var fallbacks
        self.api_key = api_key or os.getenv("PANW_PRISMA_AIRS_API_KEY")
        self.api_base = (
            api_base
            or os.getenv("PANW_PRISMA_AIRS_API_BASE")
            or "https://service.api.aisecurity.paloaltonetworks.com"
        )
        self.profile_name = profile_name

        # Handle app_name: Default to "LiteLLM", or prefix user's app_name with "LiteLLM-"
        if app_name:
            self.app_name = f"LiteLLM-{app_name}"
        else:
            self.app_name = "LiteLLM"

        # Validate required configuration
        if not self.api_key:
            raise ValueError(
                "PANW Prisma AIRS: api_key is required. "
                "Set it via config or PANW_PRISMA_AIRS_API_KEY environment variable."
            )

        # Warn if no profile is configured (user must have API key with linked profile)
        if not self.profile_name:
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS Guardrail '{guardrail_name}': No profile_name configured. "
                f"Ensure your API key has a linked profile in Strata Cloud Manager, "
                f"or provide 'profile_name'/'profile_id' via config or per-request metadata. "
                f"Requests will fail if the API key is not linked to a profile."
            )

        self.fallback_on_error = fallback_on_error
        self.timeout = timeout

        if self.fallback_on_error == "allow":
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS Guardrail '{guardrail_name}': fallback_on_error='allow' - "
                f"requests will proceed without scanning when API is unavailable."
            )

        verbose_proxy_logger.info(
            f"Initialized PANW Prisma AIRS Guardrail: {guardrail_name} "
            f"(profile={self.profile_name or 'API-key-linked'}, "
            f"mask_request={self.mask_request_content}, mask_response={self.mask_response_content}, "
            f"fallback_on_error={self.fallback_on_error}, timeout={self.timeout})"
        )

    def _extract_text_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text content from messages array."""
        if not isinstance(messages, list) or not messages:
            return ""

        # Find the last user message
        for message in reversed(messages):
            if message.get("role") != "user":
                continue

            content = message.get("content")
            if not content:
                continue

            if isinstance(content, str):
                return content

            if isinstance(content, list):
                return self._extract_text_from_content_list(content)

        return ""

    def _extract_text_from_content_list(
        self, content_list: List[Dict[str, Any]]
    ) -> str:
        """Extract text from content list format."""
        text_parts = [
            part.get("text", "")
            for part in content_list
            if isinstance(part, dict)
            and part.get("type") == "text"
            and part.get("text")
        ]
        return " ".join(text_parts) if text_parts else ""

    def _extract_response_text(self, response: ModelResponse) -> str:
        """
        Extract all text content from LLM response.
        Handles multiple choices, tool calls, and function calls.
        Returns concatenated text for scanning.
        """
        try:
            from litellm.types.utils import Choices

            text_parts = []

            if hasattr(response, "choices") and response.choices:
                for choice in response.choices:
                    if isinstance(choice, Choices):
                        # Extract message content
                        if choice.message.content:
                            text_parts.append(str(choice.message.content))

                        # Extract tool call arguments
                        if (
                            hasattr(choice.message, "tool_calls")
                            and choice.message.tool_calls
                        ):
                            for tool_call in choice.message.tool_calls:
                                if hasattr(tool_call, "function") and hasattr(
                                    tool_call.function, "arguments"
                                ):
                                    text_parts.append(str(tool_call.function.arguments))

                        # Extract function call arguments (legacy)
                        if (
                            hasattr(choice.message, "function_call")
                            and choice.message.function_call
                        ):
                            if hasattr(choice.message.function_call, "arguments"):
                                text_parts.append(
                                    str(choice.message.function_call.arguments)
                                )

            return " ".join(text_parts) if text_parts else ""
        except (AttributeError, IndexError) as e:
            verbose_proxy_logger.error(
                f"PANW Prisma AIRS: Error extracting response text: {str(e)}"
            )
        return ""

    async def _call_panw_api(  # noqa: PLR0915
        self,
        content: str,
        is_response: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call PANW Prisma AIRS API to scan content."""

        if not content.strip():
            return {"action": "allow", "category": "empty"}

        # Use litellm_trace_id as Prisma AIRS AI Session ID for session grouping
        transaction_id = metadata.get("litellm_trace_id") if metadata else None
        if not transaction_id:
            transaction_id = call_id or str(uuid.uuid4())

        # Build Prisma AIRS API metadata
        # Handle app_name: LiteLLM by default, or LiteLLM-{user_app_name} if user provides one
        user_app_name = metadata.get("app_name") if metadata else None
        if user_app_name:
            app_name_value = f"LiteLLM-{user_app_name}"
        else:
            app_name_value = self.app_name  # Defaults to "LiteLLM"

        panw_metadata = {
            "app_user": (
                metadata.get("app_user") or metadata.get("user") or "litellm_user"
            )
            if metadata
            else "litellm_user",
            "ai_model": metadata.get("model", "unknown") if metadata else "unknown",
            "app_name": app_name_value,
            "source": "litellm_builtin_guardrail",
        }

        # Include user_ip if available (from LiteLLM metadata or request)
        if metadata and metadata.get("user_ip"):
            panw_metadata["user_ip"] = metadata["user_ip"]
        elif metadata and metadata.get("requester_ip_address"):
            panw_metadata["user_ip"] = metadata["requester_ip_address"]

        payload = {
            "tr_id": transaction_id,
            "metadata": panw_metadata,
            "contents": [{"response" if is_response else "prompt": content}],
        }

        # Build ai_profile object per PANW API schema
        # Priority: per-request profile_id > per-request profile_name > config profile_name
        # Note: If both are provided, PANW API uses profile_id (profile_id takes precedence)
        profile_name = None
        profile_id = None

        if metadata:
            profile_id = metadata.get("profile_id")
            profile_name = metadata.get("profile_name", self.profile_name)
        else:
            profile_name = self.profile_name

        # Add ai_profile to payload if profile is specified
        # If neither profile_name nor profile_id is provided, PANW API will use the
        # profile linked to the API key (if configured in Strata Cloud Manager)
        if profile_name or profile_id:
            ai_profile = {}
            if profile_id:
                ai_profile["profile_id"] = profile_id
            if profile_name:
                ai_profile["profile_name"] = profile_name
            payload["ai_profile"] = ai_profile

        if is_response:
            payload["metadata"]["is_response"] = True  # type: ignore[call-overload, index]

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-pan-token": self.api_key
            or "",  # api_key validated in __init__, never None
        }

        try:
            # Use LiteLLM's async HTTP client
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.GuardrailCallback
            )

            # Bypass wrapper to access follow_redirects parameter
            response = await async_client.client.post(  # type: ignore[attr-defined]
                f"{self.api_base}/v1/scan/sync/request",
                headers=headers,
                json=payload,
                timeout=self.timeout,
                follow_redirects=False,  # Prevent redirect attacks
            )
            response.raise_for_status()

            result = response.json()

            # Validate response format
            if "action" not in result:
                verbose_proxy_logger.error(
                    f"PANW Prisma AIRS: Invalid API response format: {result}"
                )
                return {"action": "block", "category": "api_error"}

            # Check for profile-related errors from PANW API
            if result.get("action") == "block" and "error" in result:
                error_msg = str(result.get("error", "")).lower()
                if "profile" in error_msg and (
                    "not found" in error_msg
                    or "required" in error_msg
                    or "invalid" in error_msg
                ):
                    verbose_proxy_logger.error(
                        f"PANW Prisma AIRS: Profile configuration error. "
                        f"Ensure your API key has a linked profile in Strata Cloud Manager, "
                        f"or provide 'profile_name' or 'profile_id' in config/metadata. "
                        f"PANW API response: {result}"
                    )

            verbose_proxy_logger.debug(
                f"PANW Prisma AIRS: Scan result - Action: {result.get('action')}, Category: {result.get('category', 'unknown')}"
            )
            return result

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_body = ""
            try:
                error_body = e.response.text[:200]
            except Exception:
                pass

            is_profile_error = any(
                phrase in error_body.lower()
                for phrase in [
                    "profile not found",
                    "profile required",
                    "invalid profile",
                ]
            )

            if status in (401, 403) or is_profile_error:
                verbose_proxy_logger.error(
                    f"PANW Prisma AIRS: Authentication/config error (HTTP {status}). "
                    f"Check API key and profile configuration."
                )
                return {
                    "action": "block",
                    "category": "config_error",
                    "_always_block": True,
                }
            else:
                verbose_proxy_logger.error(
                    f"PANW Prisma AIRS: API error (HTTP {status}): {error_body}"
                )
                return {
                    "action": "block",
                    "category": f"http_{status}_error",
                    "_is_transient": True,
                }

        except httpx.TimeoutException as e:
            verbose_proxy_logger.error(f"PANW Prisma AIRS: Timeout error: {str(e)}")
            return {
                "action": "block",
                "category": "timeout_error",
                "_is_transient": True,
            }

        except httpx.RequestError as e:
            verbose_proxy_logger.error(
                f"PANW Prisma AIRS: Network/request error: {str(e)}"
            )
            return {
                "action": "block",
                "category": "network_error",
                "_is_transient": True,
            }

        except Exception as e:
            verbose_proxy_logger.error(f"PANW Prisma AIRS: Unexpected error: {str(e)}")
            return {"action": "block", "category": "api_error", "_is_transient": True}

    def _get_masked_text(
        self, scan_result: Dict[str, Any], is_response: bool = False
    ) -> Optional[str]:
        """Extract masked text from PANW scan result."""
        masked_key = "response_masked_data" if is_response else "prompt_masked_data"
        masked_data = scan_result.get(masked_key)
        if masked_data and isinstance(masked_data, dict):
            return masked_data.get("data")
        return None

    def _apply_masking_to_messages(
        self, messages: List[Dict[str, Any]], masked_text: str
    ) -> List[Dict[str, Any]]:
        """Apply masked text to the last user message."""
        if not messages:
            return messages

        for i, message in enumerate(reversed(messages)):
            if message.get("role") == "user":
                new_message = message.copy()
                content = message.get("content")

                if isinstance(content, str):
                    new_message["content"] = masked_text
                elif isinstance(content, list):
                    new_content = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            new_content.append({"type": "text", "text": masked_text})
                        else:
                            new_content.append(part)
                    new_message["content"] = new_content

                idx = len(messages) - i - 1
                return messages[:idx] + [new_message] + messages[idx + 1 :]

        return messages

    def _apply_masking_to_response(
        self, response: ModelResponse, masked_text: str
    ) -> None:
        """
        Apply masked text to all content in response in-place.
        Handles message content, tool calls, and function calls across all choices.
        Preserves list-based content structure (e.g., multimodal messages).
        """
        from litellm.types.utils import Choices

        if not hasattr(response, "choices") or not response.choices:
            return

        for choice in response.choices:
            if isinstance(choice, Choices):
                # Mask message content - handle both string and list formats
                content = choice.message.content
                if content:
                    if isinstance(content, str):
                        choice.message.content = masked_text
                    elif isinstance(content, list):
                        # Preserve list structure, only replace text parts
                        new_content = []
                        for part in content:  # type: ignore
                            if isinstance(part, dict) and part.get("type") == "text":
                                new_content.append(
                                    {"type": "text", "text": masked_text}
                                )
                            else:
                                # Preserve non-text parts (images, etc.)
                                new_content.append(part)
                        choice.message.content = new_content  # type: ignore

                # Mask tool call arguments
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        if hasattr(tool_call, "function") and hasattr(
                            tool_call.function, "arguments"
                        ):
                            tool_call.function.arguments = masked_text

                # Mask function call arguments (legacy)
                if (
                    hasattr(choice.message, "function_call")
                    and choice.message.function_call
                ):
                    if hasattr(choice.message.function_call, "arguments"):
                        choice.message.function_call.arguments = masked_text

    def _build_error_detail(
        self, scan_result: Dict[str, Any], is_response: bool = False
    ) -> Dict[str, Any]:
        """Build enhanced error detail with scan information."""
        action_type = "Response" if is_response else "Prompt"
        code_suffix = "_response_blocked" if is_response else "_blocked"
        detection_key = "response_detected" if is_response else "prompt_detected"

        category = scan_result.get("category", "unknown")
        default_msg = f"{action_type} blocked by PANW Prisma AI Security policy (Category: {category})"

        # Use custom violation message template if configured
        error_msg = self.render_violation_message(
            default=default_msg,
            context={
                "guardrail_name": self.guardrail_name,
                "category": category,
                "action_type": action_type,
                "default_message": default_msg,
            },
        )

        error_detail = {
            "error": {
                "message": error_msg,
                "type": "guardrail_violation",
                "code": f"panw_prisma_airs{code_suffix}",
                "guardrail": self.guardrail_name,
                "category": category,
            }
        }

        # Add optional fields if present
        optional_fields = [
            "scan_id",
            "report_id",
            "profile_name",
            "profile_id",
            "tr_id",
        ]
        for field in optional_fields:
            if scan_result.get(field):
                error_detail["error"][field] = scan_result[field]

        # Add detection details
        if scan_result.get(detection_key):
            error_detail["error"][detection_key] = scan_result[detection_key]

        return error_detail

    def _handle_api_error_with_logging(
        self,
        scan_result: Dict[str, Any],
        data: Dict[str, Any],
        start_time: datetime,
        event_type: GuardrailEventHooks,
        is_response: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Handle API errors with fail-open/fail-closed logic."""
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        category = scan_result.get("category", "api_error")

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider="panw_prisma_airs",
            guardrail_json_response=scan_result,
            request_data=data,
            guardrail_status="guardrail_failed_to_respond",
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=duration,
            event_type=event_type,
        )

        if scan_result.get("_always_block"):
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "Security scan failed - configuration error",
                        "type": "guardrail_config_error",
                        "code": "panw_prisma_airs_config_error",
                        "guardrail": self.guardrail_name,
                        "category": category,
                    }
                },
            )

        if scan_result.get("_is_transient") and self.fallback_on_error == "allow":
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS: Allowing {'response' if is_response else 'request'} "
                f"without scanning (fallback_on_error='allow', error: {category})"
            )
            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=f"{self.guardrail_name}:unscanned"
            )
            return None

        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Security scan failed - request blocked for safety",
                    "type": "guardrail_scan_error",
                    "code": "panw_prisma_airs_scan_failed",
                    "guardrail": self.guardrail_name,
                    "category": category,
                }
            },
        )

    def _prepare_metadata_from_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and prepare metadata from request data for PANW API call.

        Supported metadata fields (from request.metadata):
        - profile_name: AI security profile name (PANW API field)
        - profile_id: AI security profile ID (PANW API field, takes precedence)
        - user_ip: User IP address for tracking
        - app_name: Application identifier (will be prefixed with "LiteLLM-")

        Note: If neither profile_name nor profile_id is provided, PANW API will use
        the profile linked to the API key (configured in Strata Cloud Manager).
        If both are provided, PANW API uses profile_id (profile_id takes precedence).
        """
        user_metadata = data.get("metadata", {}) or {}
        metadata = {
            "user": data.get("user") or "litellm_user",
            "model": data.get("model") or "unknown",
        }

        # Pass through PANW API fields
        if "profile_name" in user_metadata:
            metadata["profile_name"] = user_metadata["profile_name"]

        if "profile_id" in user_metadata:
            metadata["profile_id"] = user_metadata["profile_id"]

        if "user_ip" in user_metadata:
            metadata["user_ip"] = user_metadata["user_ip"]

        if "app_name" in user_metadata:
            metadata["app_name"] = user_metadata["app_name"]

        if "app_user" in user_metadata:
            metadata["app_user"] = user_metadata["app_user"]

        # Include litellm_trace_id for session tracking
        if data.get("litellm_trace_id"):
            metadata["litellm_trace_id"] = data["litellm_trace_id"]

        return metadata

    def _check_and_mark_scanned(self, data: dict, scan_type: str) -> bool:
        """
        Check if request has already been scanned and mark it as scanned.

        Args:
            data: Request data dictionary
            scan_type: Type of scan ('pre', 'post', 'streaming')

        Returns:
            True if already scanned (should skip), False if needs scanning
        """
        call_id = data.get("litellm_call_id")
        if not call_id:
            call_id = str(uuid.uuid4())
            data["litellm_call_id"] = call_id

        scan_key = f"_panw_{scan_type}_scanned_{call_id}"
        litellm_metadata = data.setdefault("litellm_metadata", {})

        if litellm_metadata.get(scan_key):
            verbose_proxy_logger.debug(
                f"PANW Prisma AIRS: Skipping duplicate {scan_type}-call scan"
            )
            return True  # Already scanned

        litellm_metadata[scan_key] = True
        return False  # Needs scanning

    def _extract_prompt_from_request(self, data: dict) -> str:
        """
        Extract prompt text from request data.

        Handles both chat completion (messages) and text completion (prompt) formats.

        Args:
            data: Request data dictionary

        Returns:
            Extracted prompt text, or empty string if not found
        """
        # Extract from messages (chat completion)
        messages = data.get("messages", [])
        prompt_text = self._extract_text_from_messages(messages)

        # Fallback to prompt field for text completion requests
        if not prompt_text:
            prompt_value = data.get("prompt")
            if isinstance(prompt_value, str):
                prompt_text = prompt_value
            elif isinstance(prompt_value, list):
                # Handle list of prompts (batch text completion)
                prompt_text = " ".join(str(p) for p in prompt_value if p)
            else:
                prompt_text = ""

        return prompt_text

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict[str, Any],
        call_type: CallTypesLiteral,
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-call hook to scan user prompts before sending to LLM.

        Raises HTTPException if content should be blocked.
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        verbose_proxy_logger.info("PANW Prisma AIRS: Running pre-call prompt scan")

        # Check if guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        # Prevent duplicate scans by checking if already processed
        if self._check_and_mark_scanned(data, "pre"):
            return data

        try:
            start_time = datetime.now()

            # Extract prompt text from request
            prompt_text = self._extract_prompt_from_request(data)
            messages = data.get("messages", [])  # Keep for masking operations

            if not prompt_text:
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: No user prompt found in request (checked 'messages' and 'prompt' fields)"
                )
                return None

            # Prepare metadata - include user's metadata for profile override
            metadata = self._prepare_metadata_from_request(data)

            # Scan prompt with PANW Prisma AIRS
            scan_result = await self._call_panw_api(
                content=prompt_text,
                is_response=False,
                metadata=metadata,
                call_id=data.get("litellm_call_id"),
            )

            if scan_result.get("_is_transient") or scan_result.get("_always_block"):
                return self._handle_api_error_with_logging(
                    scan_result,
                    data,
                    start_time,
                    is_response=False,
                    event_type=GuardrailEventHooks.pre_call,
                )

            end_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="panw_prisma_airs",
                guardrail_json_response=scan_result,
                request_data=data,
                guardrail_status="success"
                if scan_result.get("action") == "allow"
                else "guardrail_intervened",
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
                event_type=GuardrailEventHooks.pre_call,
            )

            action = scan_result.get("action", "block")
            category = scan_result.get("category", "unknown")
            masked_text = self._get_masked_text(scan_result, is_response=False)

            # If action is "allow", apply masking if available and allow through
            if action == "allow":
                if masked_text:
                    if messages:
                        data["messages"] = self._apply_masking_to_messages(
                            messages, masked_text
                        )
                    elif "prompt" in data:
                        data["prompt"] = masked_text
                    verbose_proxy_logger.info(
                        f"PANW Prisma AIRS: Prompt allowed with masking (Category: {category})"
                    )
                else:
                    verbose_proxy_logger.info(
                        f"PANW Prisma AIRS: Prompt allowed (Category: {category})"
                    )
                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )
                return None

            # Action is "block" - check if we should mask instead of blocking
            if masked_text and self.mask_request_content:
                if messages:
                    data["messages"] = self._apply_masking_to_messages(
                        messages, masked_text
                    )
                elif "prompt" in data:
                    data["prompt"] = masked_text
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: Prompt blocked but masked instead (mask_request_content=True)"
                )
                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )
                return None

            # Block the request
            error_detail = self._build_error_detail(scan_result, is_response=False)
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS: {error_detail['error']['message']}"
            )
            raise HTTPException(status_code=400, detail=error_detail)

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"PANW Prisma AIRS scan failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "Security scan failed - request blocked for safety",
                        "type": "guardrail_scan_error",
                        "code": "panw_prisma_airs_scan_failed",
                        "guardrail": self.guardrail_name,
                    }
                },
            )

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Post-call hook to scan LLM responses before returning to user.

        Raises HTTPException if response should be blocked.
        """
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        # Only process ModelResponse objects
        if not isinstance(response, ModelResponse):
            return response

        verbose_proxy_logger.info("PANW Prisma AIRS: Running post-call response scan")

        # Check if guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        # Prevent duplicate scans by checking if already processed
        if self._check_and_mark_scanned(data, "post"):
            return response

        try:
            start_time = datetime.now()

            # Extract response text
            response_text = self._extract_response_text(response)

            if not response_text:
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: No response content found to scan"
                )
                return response

            # Prepare metadata - include user's metadata for profile override
            metadata = self._prepare_metadata_from_request(data)

            # Scan response with PANW Prisma AIRS
            scan_result = await self._call_panw_api(
                content=response_text,
                is_response=True,
                metadata=metadata,
                call_id=data.get("litellm_call_id"),
            )

            if scan_result.get("_is_transient") or scan_result.get("_always_block"):
                self._handle_api_error_with_logging(
                    scan_result,
                    data,
                    start_time,
                    is_response=True,
                    event_type=GuardrailEventHooks.post_call,
                )
                return response

            end_time = datetime.now()
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="panw_prisma_airs",
                guardrail_json_response=scan_result,
                request_data=data,
                guardrail_status="success"
                if scan_result.get("action") == "allow"
                else "guardrail_intervened",
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=(end_time - start_time).total_seconds(),
                event_type=GuardrailEventHooks.post_call,
            )

            action = scan_result.get("action", "block")
            category = scan_result.get("category", "unknown")
            masked_text = self._get_masked_text(scan_result, is_response=True)

            # If action is "allow", apply masking if available and allow through
            if action == "allow":
                if masked_text:
                    self._apply_masking_to_response(response, masked_text)
                    verbose_proxy_logger.info(
                        f"PANW Prisma AIRS: Response allowed with masking (Category: {category})"
                    )
                else:
                    verbose_proxy_logger.info(
                        f"PANW Prisma AIRS: Response allowed (Category: {category})"
                    )
                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )
                return response

            # Action is "block" - check if we should mask instead of blocking
            if masked_text and self.mask_response_content:
                self._apply_masking_to_response(response, masked_text)
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: Response blocked but masked instead (mask_response_content=True)"
                )
                add_guardrail_to_applied_guardrails_header(
                    request_data=data, guardrail_name=self.guardrail_name
                )
                return response

            # Block the response
            error_detail = self._build_error_detail(scan_result, is_response=True)
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS: {error_detail['error']['message']}"
            )
            raise HTTPException(status_code=400, detail=error_detail)

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"PANW Prisma AIRS scan failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "Security scan failed - response blocked for safety",
                        "type": "guardrail_scan_error",
                        "code": "panw_prisma_airs_scan_failed",
                        "guardrail": self.guardrail_name,
                    }
                },
            )

    async def _scan_and_process_streaming_response(
        self,
        assembled_model_response: ModelResponse,
        request_data: dict,
        start_time: datetime,
    ) -> Tuple[bool, ModelResponse, Dict[str, Any]]:
        """
        Scan assembled streaming response and apply masking if needed.
        Returns (content_was_modified, response, scan_result).
        """
        content_was_modified = False
        response_text = self._extract_response_text(assembled_model_response)

        if not response_text or not response_text.strip():
            verbose_proxy_logger.info(
                "PANW Prisma AIRS: No content to scan in streaming response"
            )
            return (
                content_was_modified,
                assembled_model_response,
                {"action": "allow", "category": "no_content"},
            )

        # Prepare metadata - include user's metadata for profile override
        metadata = self._prepare_metadata_from_request(request_data)

        scan_result = await self._call_panw_api(
            content=response_text,
            is_response=True,
            metadata=metadata,
            call_id=request_data.get("litellm_call_id"),
        )

        action = scan_result.get("action", "block")
        category = scan_result.get("category", "unknown")
        masked_text = self._get_masked_text(scan_result, is_response=True)

        # Handle scan results
        if action == "allow":
            if masked_text:
                self._apply_masking_to_response(assembled_model_response, masked_text)
                content_was_modified = True
                verbose_proxy_logger.info(
                    f"PANW Prisma AIRS: Streaming response allowed with masking (Category: {category})"
                )
            else:
                verbose_proxy_logger.info(
                    f"PANW Prisma AIRS: Streaming response allowed (Category: {category})"
                )
        elif masked_text and self.mask_response_content:
            self._apply_masking_to_response(assembled_model_response, masked_text)
            content_was_modified = True
            verbose_proxy_logger.warning(
                "PANW Prisma AIRS: Streaming response blocked but masked instead (mask_response_content=True)"
            )
        else:
            error_detail = self._build_error_detail(scan_result, is_response=True)
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS: {error_detail['error']['message']}"
            )
            raise HTTPException(status_code=400, detail=error_detail)

        return content_was_modified, assembled_model_response, scan_result

    @log_guardrail_information
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ):
        """
        Process streaming response chunks and scan the assembled response.
        """
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        # Check if guardrail should run for this request
        from litellm.types.guardrails import GuardrailEventHooks as EventHooks

        if not self.should_run_guardrail(
            data=request_data, event_type=EventHooks.post_call
        ):
            async for chunk in response:
                yield chunk
            return

        # Prevent duplicate scans by checking if already processed
        if self._check_and_mark_scanned(request_data, "streaming"):
            async for chunk in response:
                yield chunk
            return

        verbose_proxy_logger.info("PANW Prisma AIRS: Running post-call streaming scan")

        all_chunks = []
        content_was_modified = False

        try:
            start_time = datetime.now()

            # Collect all chunks
            async for chunk in response:
                all_chunks.append(chunk)

            # Assemble complete response from chunks
            assembled_model_response = stream_chunk_builder(chunks=all_chunks)

            if isinstance(assembled_model_response, ModelResponse):
                # Scan and process the assembled response
                (
                    content_was_modified,
                    assembled_model_response,
                    scan_result,
                ) = await self._scan_and_process_streaming_response(
                    assembled_model_response, request_data, start_time
                )

                if scan_result.get("_is_transient") or scan_result.get("_always_block"):
                    self._handle_api_error_with_logging(
                        scan_result,
                        request_data,
                        start_time,
                        is_response=True,
                        event_type=EventHooks.post_call,
                    )
                    for chunk in all_chunks:
                        yield chunk
                    return

                end_time = datetime.now()
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider="panw_prisma_airs",
                    guardrail_json_response=scan_result,
                    request_data=request_data,
                    guardrail_status="success"
                    if scan_result.get("action") == "allow"
                    else "guardrail_intervened",
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    duration=(end_time - start_time).total_seconds(),
                    event_type=EventHooks.post_call,
                )

                # Add guardrail to applied guardrails header for observability
                add_guardrail_to_applied_guardrails_header(
                    request_data=request_data, guardrail_name=self.guardrail_name
                )

                # Only use MockResponseIterator if content was modified
                # Otherwise, yield original chunks to preserve streaming behavior
                if content_was_modified:
                    mock_response = MockResponseIterator(
                        model_response=assembled_model_response
                    )
                    async for chunk in mock_response:
                        yield chunk
                else:
                    for chunk in all_chunks:
                        yield chunk
            else:
                # If not a ModelResponse, just yield original chunks
                for chunk in all_chunks:
                    yield chunk

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"PANW Prisma AIRS streaming error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": {
                        "message": "Security scan failed - streaming response blocked for safety",
                        "type": "guardrail_scan_error",
                        "code": "panw_prisma_airs_scan_failed",
                        "guardrail": self.guardrail_name,
                    }
                },
            )

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.panw_prisma_airs import (
            PanwPrismaAirsGuardrailConfigModel,
        )

        return PanwPrismaAirsGuardrailConfigModel
