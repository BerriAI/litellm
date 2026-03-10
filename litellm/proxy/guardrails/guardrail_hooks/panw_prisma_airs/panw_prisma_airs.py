#!/usr/bin/env python3
"""
Palo Alto Networks Prisma AI Runtime Security (AIRS) Guardrail Integration for LiteLLM

Provides real-time threat detection, DLP, URL filtering, content masking, and policy enforcement for AI applications.
"""

import json
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Type
from urllib.parse import urlparse

import httpx

from litellm._uuid import uuid
from litellm.caching import DualCache

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
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.utils import (
    CallTypes,
    CallTypesLiteral,
    Choices,
    GenericGuardrailAPIInputs,
    ModelResponse,
    ModelResponseStream,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
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

    _PROVIDER_NAME = "panw_prisma_airs"

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
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.logging_only,
                GuardrailEventHooks.pre_mcp_call,
                GuardrailEventHooks.during_mcp_call,
            ],
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

        # Tri-state: None = not set (default-on for Anthropic), True = explicit on, False = explicit off
        self.experimental_use_latest_role_message_only: Optional[bool] = kwargs.get(
            "experimental_use_latest_role_message_only"
        )

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

    # MCP event → base-call compatibility map.
    # Allows guardrails configured with mode: pre_call / during_call to
    # automatically run on MCP tool invocations (pre_mcp_call / during_mcp_call).
    _MCP_COMPAT_MAP = {
        GuardrailEventHooks.pre_mcp_call: GuardrailEventHooks.pre_call,
        GuardrailEventHooks.during_mcp_call: GuardrailEventHooks.during_call,
    }

    def should_run_guardrail(self, data: Any, event_type: GuardrailEventHooks) -> bool:
        if super().should_run_guardrail(data, event_type):
            return True
        compat = self._MCP_COMPAT_MAP.get(event_type)
        if compat is not None:
            if super().should_run_guardrail(data, compat):
                return True
        return False

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
        content: str = "",
        is_response: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        tool_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call PANW Prisma AIRS API to scan content or a tool_event."""

        if tool_event is None and not content.strip():
            return {"action": "allow", "category": "empty"}

        # tr_id is optional in the AIRS API. Allow call_id=None only for
        # MCP tool_events (ecosystem == "mcp"). All other paths (content
        # scans, non-MCP tool_events) remain fail-closed.
        if not call_id:
            _is_mcp_tool_event = (
                tool_event is not None
                and isinstance(tool_event.get("metadata"), dict)
                and tool_event["metadata"].get("ecosystem") == "mcp"
            )
            if not _is_mcp_tool_event:
                return {
                    "action": "block",
                    "category": "missing_call_id",
                    "_always_block": True,
                }

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

        # Forward litellm_trace_id in AIRS metadata for session correlation
        if metadata and metadata.get("litellm_trace_id"):
            panw_metadata["litellm_trace_id"] = metadata["litellm_trace_id"]

        # Build contents: tool_event takes priority, else prompt/response text
        if tool_event is not None:
            contents = [{"tool_event": tool_event}]
        else:
            contents = [{"response" if is_response else "prompt": content}]

        payload = {
            "metadata": panw_metadata,
            "contents": contents,
        }
        # Use per-request litellm_call_id as AIRS tr_id; keep litellm_trace_id in metadata.
        if call_id:
            payload["tr_id"] = call_id

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

        if is_response and tool_event is None:
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
                error_body = e.response.text
            except Exception:
                pass

            # Enhanced 400 diagnostics for tool_event schema debugging
            if status == 400:
                diag_parts = ["PANW Prisma AIRS: HTTP 400 from AIRS API."]
                if tool_event is not None:
                    diag_parts.append(
                        f"tool_event.metadata={tool_event.get('metadata')}"
                    )
                    has_input = "input" in tool_event
                    input_len = len(tool_event["input"]) if has_input else 0
                    diag_parts.append(f"input present={has_input}, len={input_len}")
                diag_parts.append(f"response body: {error_body[:500]}")
                verbose_proxy_logger.error(" | ".join(diag_parts))

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
                if status != 400:  # Already logged above with diagnostics
                    verbose_proxy_logger.error(
                        f"PANW Prisma AIRS: API error (HTTP {status}): {error_body[:500]}"
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

    @staticmethod
    def _get_mcp_server_name(request_data: dict, mcp_tool_name: str) -> str:
        """Resolve MCP server name from request data or MCP registry."""
        if request_data.get("mcp_server_name"):
            return request_data["mcp_server_name"]
        if request_data.get("server_name"):
            return request_data["server_name"]
        try:
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )

            server_id = request_data.get("server_id")
            if server_id:
                server = global_mcp_server_manager.get_mcp_server_by_id(server_id)
                if server:
                    return (
                        getattr(server, "alias", None)
                        or getattr(server, "server_name", None)
                        or getattr(server, "name", None)
                        or getattr(server, "server_id", None)
                        or "unknown"
                    )
            return global_mcp_server_manager.tool_name_to_mcp_server_name_mapping.get(
                mcp_tool_name, "unknown"
            )
        except ImportError:
            return "unknown"
        except Exception:
            verbose_proxy_logger.debug(
                "PANW Prisma AIRS: unexpected error resolving MCP server name",
                exc_info=True,
            )
            return "unknown"

    def _get_masked_text(
        self, scan_result: Dict[str, Any], is_response: bool = False
    ) -> Optional[str]:
        """Extract masked text from PANW scan result."""
        masked_key = "response_masked_data" if is_response else "prompt_masked_data"
        masked_data = scan_result.get(masked_key)
        if masked_data and isinstance(masked_data, dict):
            return masked_data.get("data")
        return None

    @staticmethod
    def _mask_content_list(content_list: List, masked_text: str) -> List:
        """Replace text parts in a content list, preserving non-text parts (images, etc.)."""
        new_content = []
        for part in content_list:
            if isinstance(part, dict) and part.get("type") == "text":
                new_content.append({"type": "text", "text": masked_text})
            else:
                new_content.append(part)
        return new_content

    @staticmethod
    def _apply_mcp_masking(
        request_data: dict,
        original_args: Any,
        masked_text: str,
        *,
        is_blocked: bool = True,
    ) -> None:
        """Write masked arguments back to MCP request_data fields.

        - ``arguments`` is the authoritative field that ``call_mcp_tool``
          reads, so it must be updated first.
        - ``mcp_arguments`` is mirrored for consistency / test observability.
        - If the original args were structured (dict/list), attempt
          ``json.loads`` to preserve the type; block if the masked text
          is not valid JSON (to avoid corrupting structured args).
        - If neither ``arguments`` nor ``mcp_arguments`` is present in
          request_data, block — do not silently invent a new field.
        """
        has_arguments = "arguments" in request_data
        has_mcp_arguments = "mcp_arguments" in request_data
        if not has_arguments and not has_mcp_arguments:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "MCP request blocked: no rewritable argument field present",
                        "type": "guardrail_violation",
                        "code": "panw_prisma_airs_blocked",
                    }
                },
            )

        # If the original args were structured, preserve the type.
        if isinstance(original_args, (dict, list)):
            try:
                parsed = json.loads(masked_text)
            except (json.JSONDecodeError, TypeError):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": "MCP request blocked: masked data is not valid JSON for structured arguments",
                            "type": "guardrail_violation",
                            "code": "panw_prisma_airs_blocked",
                        }
                    },
                )
            masked_value: Any = parsed
        else:
            masked_value = masked_text

        if has_arguments:
            request_data["arguments"] = masked_value
        if has_mcp_arguments:
            request_data["mcp_arguments"] = masked_value

        if is_blocked:
            verbose_proxy_logger.warning(
                "PANW Prisma AIRS: MCP request blocked but masked instead (mask_request_content=True)"
            )
        else:
            verbose_proxy_logger.info(
                "PANW Prisma AIRS: MCP request allowed with PII masking applied"
            )

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
                    new_message["content"] = self._mask_content_list(
                        content, masked_text
                    )

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
                        choice.message.content = self._mask_content_list(  # type: ignore
                            content, masked_text
                        )

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
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        category = scan_result.get("category", "api_error")

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider=self._PROVIDER_NAME,
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
        requester_meta = user_metadata.get("requester_metadata", {}) or {}
        metadata = {
            "user": data.get("user") or "litellm_user",
            "model": data.get("model") or "unknown",
        }

        # Pass through PANW API fields (check requester_metadata fallback for /v1/messages routes)
        for key in ("profile_name", "profile_id", "user_ip", "app_name", "app_user"):
            val = user_metadata.get(key) or requester_meta.get(key)
            if val:
                metadata[key] = val

        # Include litellm_trace_id for session tracking.
        # Sources (checked in priority order):
        #   1. data["litellm_trace_id"]             — top-level body field
        #   2. metadata["litellm_trace_id"]         — user passes in request metadata
        #   3. metadata["trace_id"]                 — x-litellm-trace-id header
        #      (litellm_pre_call_utils stores it as "trace_id", not "litellm_trace_id")
        #   4. requester_metadata["litellm_trace_id"] — deep copy for /v1/messages routes
        trace_id = (
            data.get("litellm_trace_id")
            or user_metadata.get("litellm_trace_id")
            or user_metadata.get("trace_id")
            or requester_meta.get("litellm_trace_id")
        )
        if trace_id:
            metadata["litellm_trace_id"] = trace_id

        return metadata

    @staticmethod
    def _extract_text_from_sse_bytes(chunks: List[bytes]) -> str:
        """Extract text from Anthropic SSE byte chunks (content_block_delta → text_delta)."""
        texts: List[str] = []
        raw = b"".join(chunks).decode("utf-8", errors="replace")
        for line in raw.split("\n"):
            line = line.strip()
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get("type") == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    texts.append(delta.get("text", ""))
        return "".join(texts)

    @staticmethod
    def _extract_text_from_streaming_events(chunks: list) -> str:
        """Extract text from /v1/responses streaming events (object or dict)."""

        def _attr(c, key):
            return getattr(c, key, None) or (
                c.get(key) if isinstance(c, dict) else None
            )

        parts: List[str] = []
        for chunk in chunks:
            if _attr(chunk, "type") == "response.output_text.delta":
                delta = _attr(chunk, "delta")
                if isinstance(delta, str):
                    parts.append(delta)
            # Defense-in-depth: handle dict chat.completion.chunk format
            elif (
                isinstance(chunk, dict)
                and chunk.get("object") == "chat.completion.chunk"
            ):
                for choice in chunk.get("choices") or []:
                    if isinstance(choice, dict):
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str):
                            parts.append(content)
        # Fallback: response.output_text.done carries full text if no deltas captured
        if not parts:
            for chunk in chunks:
                if _attr(chunk, "type") == "response.output_text.done":
                    text = _attr(chunk, "text")
                    if isinstance(text, str):
                        parts.append(text)
        return "".join(parts)

    async def _scan_raw_streaming_text(
        self, text: str, request_data: dict, start_time: datetime
    ) -> None:
        """Scan text from non-ModelResponse streaming chunks. Raises HTTPException(400) on block.

        Note: response masking is not supported on raw streaming paths
        (/v1/messages, /v1/responses) because the response is raw SSE
        bytes/events that cannot be reliably reconstructed. If
        mask_response_content is configured, a warning is logged and the
        response is blocked instead. Request-side masking
        (mask_request_content) is unaffected — it runs in async_pre_call_hook
        before streaming begins.
        """
        if not text or not text.strip():
            return

        metadata = self._prepare_metadata_from_request(request_data)
        scan_result = await self._call_panw_api(
            content=text,
            is_response=True,
            metadata=metadata,
            call_id=request_data.get("litellm_call_id"),
        )
        if scan_result.get("_is_transient") or scan_result.get("_always_block"):
            self._handle_api_error_with_logging(
                scan_result,
                request_data,
                start_time,
                is_response=True,
                event_type=GuardrailEventHooks.post_call,
            )
            return  # _always_block raises inside; transient errors fail-open here
        action = scan_result.get("action", "block")
        if action != "allow":
            masked_text = self._get_masked_text(scan_result, is_response=True)
            if masked_text and self.mask_response_content:
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: mask_response_content is configured but "
                    "cannot be applied to raw streaming responses (/v1/messages "
                    "or /v1/responses). Blocking response instead."
                )
            raise HTTPException(
                status_code=400,
                detail=self._build_error_detail(scan_result, is_response=True),
            )
        # Success logging + observability header
        end_time = datetime.now()
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider=self._PROVIDER_NAME,
            guardrail_json_response=scan_result,
            request_data=request_data,
            guardrail_status="success",
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=(end_time - start_time).total_seconds(),
            event_type=GuardrailEventHooks.post_call,
        )
        add_guardrail_to_applied_guardrails_header(
            request_data=request_data, guardrail_name=self.guardrail_name
        )

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
            verbose_proxy_logger.warning(
                "PANW Prisma AIRS: litellm_call_id missing from request data, "
                "synthesized %s for %s scan deduplication",
                call_id,
                scan_type,
            )

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
                guardrail_provider=self._PROVIDER_NAME,
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
                guardrail_provider=self._PROVIDER_NAME,
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

        # Early return for transient/always-block results — let the
        # streaming iterator hook handle fallback_on_error semantics.
        if scan_result.get("_is_transient") or scan_result.get("_always_block"):
            return (content_was_modified, assembled_model_response, scan_result)

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

        # Check if guardrail should run for this request

        if not self.should_run_guardrail(
            data=request_data, event_type=GuardrailEventHooks.post_call
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

            # Handle /v1/messages streaming: chunks are raw bytes (Anthropic SSE)
            if all_chunks and isinstance(all_chunks[0], bytes):
                text = self._extract_text_from_sse_bytes(all_chunks)
                await self._scan_raw_streaming_text(text, request_data, start_time)
                for chunk in all_chunks:
                    yield chunk
                return

            # Handle /v1/responses streaming: chunks are Pydantic events (not ModelResponse/ModelResponseStream)
            if all_chunks and not isinstance(
                all_chunks[0], (ModelResponse, ModelResponseStream)
            ):
                text = self._extract_text_from_streaming_events(all_chunks)
                await self._scan_raw_streaming_text(text, request_data, start_time)
                for chunk in all_chunks:
                    yield chunk
                return

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
                        event_type=GuardrailEventHooks.post_call,
                    )
                    # Control only reaches here for _is_transient errors with
                    # fallback_on_error="allow"; _always_block and fail-closed
                    # paths raise inside _handle_api_error_with_logging above.
                    for chunk in all_chunks:
                        yield chunk
                    return

                end_time = datetime.now()
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider=self._PROVIDER_NAME,
                    guardrail_json_response=scan_result,
                    request_data=request_data,
                    guardrail_status="success"
                    if scan_result.get("action") == "allow"
                    else "guardrail_intervened",
                    start_time=start_time.timestamp(),
                    end_time=end_time.timestamp(),
                    duration=(end_time - start_time).total_seconds(),
                    event_type=GuardrailEventHooks.post_call,
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
                # stream_chunk_builder returned None; yield original chunks unmodified
                for chunk in all_chunks:
                    yield chunk

        except HTTPException as e:
            # Yield error as SSE event so create_response() detects it and
            # returns a proper JSON error response with the correct status code.
            # (Raising from a generator hits create_response's generic except → 500.)
            detail = (
                e.detail if isinstance(e.detail, dict) else {"message": str(e.detail)}
            )
            error_obj = dict(detail.get("error", detail))
            error_obj["code"] = e.status_code
            yield f"data: {json.dumps({'error': error_obj})}\n\n"
        except Exception as e:
            verbose_proxy_logger.error(f"PANW Prisma AIRS streaming error: {str(e)}")
            yield f'data: {json.dumps({"error": {"message": "Security scan failed - streaming response blocked for safety", "type": "guardrail_scan_error", "code": 500, "guardrail": self.guardrail_name}})}\n\n'

    async def _scan_tool_calls_for_guardrail(
        self,
        tool_calls: list,
        is_response: bool,
        metadata: Dict[str, Any],
        call_id: str,
        request_data: dict,
        start_time: datetime,
    ) -> None:
        """Scan tool call arguments with allow/block/mask treatment (in-place modification).

        Each tool call is sent as a ``tool_event`` using the canonical PANW
        AIRS schema::

            {
                "metadata": {
                    "ecosystem": "openai",
                    "method": "tools/call",
                    "server_name": "litellm",
                    "tool_invoked": "<function_name>",
                },
                "input": "<args_json>",   # optional, omitted for empty args
            }

        Empty-arg invocations are still reported (without ``input``) so AIRS
        can enforce tool-name-based policies.
        """
        for tool_call in tool_calls:
            # --- extract tool_name and args_text --------------------------
            tool_name: Optional[str] = None
            args_text: Optional[str] = None

            if hasattr(tool_call, "function") and hasattr(
                tool_call.function, "arguments"
            ):
                args_text = tool_call.function.arguments
                tool_name = getattr(tool_call.function, "name", None)
            elif isinstance(tool_call, dict):
                func = tool_call.get("function", {})
                if isinstance(func, dict):
                    args_text = func.get("arguments")
                    tool_name = func.get("name")

            # --- build tool_event payload (canonical PANW schema) -----------
            tool_event: Dict[str, Any] = {
                "metadata": {
                    "ecosystem": "openai",
                    "method": "tools/call",
                    "server_name": "litellm",
                    "tool_invoked": tool_name or "unknown",
                },
            }
            if args_text and args_text.strip():
                tool_event["input"] = args_text

            scan_result = await self._call_panw_api(
                is_response=False,  # tool_event is always request-side in AIRS schema
                metadata=metadata,
                call_id=call_id,
                tool_event=tool_event,
            )

            if scan_result.get("_is_transient") or scan_result.get("_always_block"):
                event_type = (
                    GuardrailEventHooks.post_call
                    if is_response
                    else GuardrailEventHooks.pre_call
                )
                self._handle_api_error_with_logging(
                    scan_result=scan_result,
                    data=request_data,
                    start_time=start_time,
                    event_type=event_type,
                    is_response=is_response,
                )
                continue  # fallback_on_error="allow" — leave args unchanged

            action = scan_result.get("action", "block")
            # Always is_response=False for masked data lookup because
            # tool_event scans are request-side in AIRS schema and
            # AIRS returns prompt_masked_data for them.
            masked_text = self._get_masked_text(scan_result, is_response=False)

            if action == "allow":
                if masked_text:
                    self._set_tool_call_arguments(tool_call, masked_text)
            elif masked_text and (
                (is_response and self.mask_response_content)
                or (not is_response and self.mask_request_content)
            ):
                self._set_tool_call_arguments(tool_call, masked_text)
            else:
                error_detail = self._build_error_detail(
                    scan_result, is_response=is_response
                )
                raise HTTPException(status_code=400, detail=error_detail)

    @staticmethod
    def _set_tool_call_arguments(tool_call, masked_text: str) -> None:
        """Set masked text on a tool call's function arguments, handling both object and dict forms."""
        if hasattr(tool_call, "function"):
            tool_call.function.arguments = masked_text
        elif isinstance(tool_call, dict) and isinstance(
            tool_call.get("function"), dict
        ):
            tool_call["function"]["arguments"] = masked_text

    @staticmethod
    def _is_anthropic_request(
        request_data: dict,
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> bool:
        """Detect if the current request is an Anthropic /v1/messages call."""
        if logging_obj:
            call_type = getattr(logging_obj, "call_type", None)
            if call_type in (
                CallTypes.anthropic_messages.value,
                CallTypes.anthropic_messages,
            ):
                return True
        psr = request_data.get("proxy_server_request") or {}
        if not isinstance(psr, dict):
            return False
        url = psr.get("url") or ""
        if not isinstance(url, str):
            return False
        # Match exact path segments, not substring (avoid matching e.g. /v1/messages_batch)
        path = urlparse(url).path.rstrip("/")
        if path.endswith("/v1/messages"):
            return True
        return False

    def _use_latest_user_only(
        self,
        request_data: dict,
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> bool:
        """Resolve whether to scan only the latest user message.

        - Non-Anthropic requests: always False (existing behavior)
        - Anthropic requests:
          - Flag explicitly True/False: respect it
          - Flag None (not set): default to True
        """
        if not self._is_anthropic_request(request_data, logging_obj):
            return False
        if self.experimental_use_latest_role_message_only is None:
            return True  # Default-on for Anthropic
        return self.experimental_use_latest_role_message_only

    @staticmethod
    def _get_latest_user_text_indices(
        texts: List[str],
        messages: list,
    ) -> Optional[set]:
        """Return text indices belonging to only the latest scannable human-authored (user or developer) message.

        Args:
            texts: Flattened text entries from the framework.
            messages: Original request messages (request_data["messages"]),
                      NOT structured_messages (which may have injected system content).

        Returns a set of scannable indices, or None on count mismatch or no user/developer
        message (safety fallback to existing role-filter behavior).
        """
        last_human_msg_idx: Optional[int] = None
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if isinstance(msg, dict) and msg.get("role") in ("user", "developer"):
                last_human_msg_idx = idx
                break

        if last_human_msg_idx is None:
            return None  # No user/developer message → fallback to existing role-filter scan

        scannable: set = set()
        text_idx = 0
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            is_latest_human = msg_idx == last_human_msg_idx

            if content is None:
                pass
            elif isinstance(content, str):
                if is_latest_human:
                    scannable.add(text_idx)
                text_idx += 1
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("text") is not None:
                        if is_latest_human:
                            scannable.add(text_idx)
                        text_idx += 1

        if text_idx != len(texts):
            return None  # Count mismatch → safety fallback

        return scannable

    @staticmethod
    def _get_scannable_text_indices(
        texts: List[str],
        structured_messages: list,
    ) -> Optional[set]:
        """Derive which ``texts`` indices originate from user/system messages.

        The unified guardrail framework flattens message content into ``texts``
        without preserving role info.  This helper re-walks
        ``structured_messages`` using the **same** extraction logic the
        framework uses (string content → 1 entry, list content → 1 per text
        item, None → 0) and records the running text index for each entry
        whose source role is ``"user"``, ``"system"``, or ``"developer"``.

        Returns a set of scannable indices, or ``None`` if the count doesn't
        match ``len(texts)`` (safety fallback → scan everything).
        """
        scannable: set = set()
        text_idx = 0
        for msg in structured_messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            content = msg.get("content")
            is_scannable = role in ("user", "system", "developer")

            if content is None:
                # No content → 0 text entries
                pass
            elif isinstance(content, str):
                if is_scannable:
                    scannable.add(text_idx)
                text_idx += 1
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("text") is not None:
                        if is_scannable:
                            scannable.add(text_idx)
                        text_idx += 1
            # Ignore other content types (shouldn't happen)

        if text_idx != len(texts):
            # Count mismatch → safety fallback: scan all
            return None

        return scannable

    @log_guardrail_information
    async def apply_guardrail(  # noqa: PLR0915
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Unified guardrail method for the apply_guardrail framework.

        Called by the UI "Test Guardrail" endpoint, UnifiedLLMGuardrails orchestrator,
        and MCP tool input scanning.
        """
        texts = inputs.get("texts", [])
        is_response = input_type == "response"

        # Resolve litellm_call_id: request_data first, then logging_obj fallback.
        # Post-call path reconstructs request_data as {"response": ...} without
        # litellm_call_id, but logging_obj.litellm_call_id is available.
        call_id = request_data.get("litellm_call_id")
        if not call_id and logging_obj:
            call_id = getattr(logging_obj, "litellm_call_id", None)
        if not call_id:
            # Use MCP name fallback: mcp_tool_name (canonical) or name (/mcp-rest path)
            _mcp_tool = str(
                request_data.get("mcp_tool_name") or request_data.get("name") or ""
            ).strip()
            if input_type == "request" and logging_obj is None and _mcp_tool:
                # Synthesize a tool-prefixed call_id for AIRS grouping.
                # Slug: lowercase, non-alphanum → "-", truncate to 40 chars.
                slug = re.sub(r"[^a-z0-9]+", "-", _mcp_tool.lower()).strip("-")[:40]
                if not slug:
                    slug = "mcp-tool"
                call_id = f"{slug}-{uuid.uuid4()}"
                request_data["litellm_call_id"] = call_id
                verbose_proxy_logger.debug(
                    "PANW Prisma AIRS: synthesized MCP tr_id=%s for tool=%s",
                    call_id,
                    _mcp_tool,
                )
            elif not request_data and logging_obj is None and input_type == "request":
                # Direct /apply_guardrail endpoint — empty request_data, no
                # logging_obj. Existing behavior: synthesize UUID.
                call_id = str(uuid.uuid4())
                request_data["litellm_call_id"] = call_id
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: litellm_call_id missing from empty "
                    "request_data, synthesized %s (direct /apply_guardrail?)",
                    call_id,
                )
            else:
                call_id = str(uuid.uuid4())
                request_data["litellm_call_id"] = call_id
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: litellm_call_id missing, synthesized %s "
                    "(input_type=%s)",
                    call_id,
                    input_type,
                )

        # Enrich request_data with model if missing (post-call metadata loss)
        if not request_data.get("model"):
            if inputs.get("model"):
                request_data["model"] = inputs["model"]
            elif logging_obj:
                request_data["model"] = getattr(logging_obj, "model", None)

        # Enrich request_data with metadata from logging_obj (post-call metadata loss).
        # Merge: logging_obj provides the base, request_data keys win on conflict.
        if logging_obj:
            _lp = (getattr(logging_obj, "model_call_details", {}) or {}).get(
                "litellm_params", {}
            ) or {}
            _orig_meta = _lp.get("metadata") or {}
            if _orig_meta:
                existing_meta = request_data.get("metadata")
                if not isinstance(existing_meta, dict):
                    existing_meta = {}
                request_data["metadata"] = {**_orig_meta, **existing_meta}

        metadata = self._prepare_metadata_from_request(request_data)
        start_time = datetime.now()
        new_texts: List[str] = []

        # On request side, determine which text indices correspond to scannable
        # messages so we can skip scanning assistant/tool history text.
        scannable_indices: Optional[set] = None
        if input_type == "request":
            structured_messages = inputs.get("structured_messages")
            if structured_messages:
                # For Anthropic /v1/messages: default to latest-user-only scanning.
                # Uses request_data["messages"] (original format), NOT structured_messages
                # (which has injected system content from adapter translation).
                if self._use_latest_user_only(request_data, logging_obj):
                    original_messages = request_data.get("messages")
                    if original_messages:
                        scannable_indices = self._get_latest_user_text_indices(
                            texts, original_messages
                        )
                # Fall through to existing role filtering if:
                # - not Anthropic, OR flag explicitly False, OR
                # - no original messages, OR
                # - latest-user extraction returned None (no user / count mismatch)
                if scannable_indices is None:
                    scannable_indices = self._get_scannable_text_indices(
                        texts, structured_messages
                    )

        for i, text in enumerate(texts):
            if not text or not text.strip():
                new_texts.append(text)
                continue

            # Skip non-user/system texts on request side
            if scannable_indices is not None and i not in scannable_indices:
                new_texts.append(text)
                continue

            scan_result = await self._call_panw_api(
                content=text,
                is_response=is_response,
                metadata=metadata,
                call_id=call_id,
            )

            # Handle API errors (transient/config)
            if scan_result.get("_is_transient") or scan_result.get("_always_block"):
                event_type = (
                    GuardrailEventHooks.post_call
                    if is_response
                    else GuardrailEventHooks.pre_call
                )
                self._handle_api_error_with_logging(
                    scan_result=scan_result,
                    data=request_data,
                    start_time=start_time,
                    event_type=event_type,
                    is_response=is_response,
                )
                # If we reach here, fallback_on_error="allow"
                new_texts.append(text)
                continue

            action = scan_result.get("action", "block")
            masked_text = self._get_masked_text(scan_result, is_response=is_response)

            if action == "allow":
                new_texts.append(masked_text if masked_text else text)
            elif masked_text and (
                (is_response and self.mask_response_content)
                or (not is_response and self.mask_request_content)
            ):
                new_texts.append(masked_text)
            else:
                error_detail = self._build_error_detail(
                    scan_result, is_response=is_response
                )
                raise HTTPException(status_code=400, detail=error_detail)

        # Scan tool call arguments — same masking policy as texts.
        # In-place modifications propagate for pre-call and OpenAI post-call.
        # Anthropic post-call drops tool_call modifications (framework limitation).
        tool_calls = inputs.get("tool_calls", [])
        if tool_calls:
            await self._scan_tool_calls_for_guardrail(
                tool_calls=tool_calls,
                is_response=is_response,
                metadata=metadata,
                call_id=call_id,
                request_data=request_data,
                start_time=start_time,
            )

        # MCP REST tool invocation scan (request-side only).
        # When an MCP tool is being invoked via /mcp-rest/tools/call, the
        # proxy sets mcp_tool_name (and optional mcp_arguments) on request_data.
        # We send a tool_event so AIRS can apply tool-aware policies.
        # REST MCP path sets "name"/"arguments"; canonical keys are
        # "mcp_tool_name"/"mcp_arguments". Check canonical first, then fallback.
        mcp_tool_name = request_data.get("mcp_tool_name") or request_data.get("name")
        if mcp_tool_name and input_type == "request":
            mcp_tool_event: Dict[str, Any] = {
                "metadata": {
                    "ecosystem": "mcp",
                    "method": "tools/call",
                    "server_name": self._get_mcp_server_name(
                        request_data, mcp_tool_name
                    ),
                    "tool_invoked": mcp_tool_name,
                },
            }
            mcp_arguments = request_data.get("mcp_arguments")
            if mcp_arguments is None:
                mcp_arguments = request_data.get("arguments")
            if mcp_arguments is not None and mcp_arguments != "":
                if isinstance(mcp_arguments, (dict, list)):
                    serialized_args = json.dumps(mcp_arguments)
                else:
                    serialized_args = str(mcp_arguments)
                if serialized_args.strip():
                    mcp_tool_event["input"] = serialized_args

            mcp_scan_result = await self._call_panw_api(
                tool_event=mcp_tool_event,
                metadata=metadata,
                call_id=call_id,
            )

            if mcp_scan_result.get("_is_transient") or mcp_scan_result.get(
                "_always_block"
            ):
                self._handle_api_error_with_logging(
                    scan_result=mcp_scan_result,
                    data=request_data,
                    start_time=start_time,
                    event_type=GuardrailEventHooks.pre_call,
                    is_response=False,
                )
                # If we reach here, fallback_on_error="allow"
            else:
                action = mcp_scan_result.get("action", "block")
                masked_text = self._get_masked_text(mcp_scan_result, is_response=False)
                if action == "allow":
                    # PANW says OK — apply PII scrubbing if present (unconditional,
                    # matching _scan_tool_calls_for_guardrail behavior).
                    if masked_text:
                        self._apply_mcp_masking(
                            request_data,
                            mcp_arguments,
                            masked_text,
                            is_blocked=False,
                        )
                elif masked_text and self.mask_request_content:
                    self._apply_mcp_masking(request_data, mcp_arguments, masked_text)
                else:
                    error_detail = self._build_error_detail(
                        mcp_scan_result, is_response=False
                    )
                    raise HTTPException(status_code=400, detail=error_detail)

        inputs["texts"] = new_texts
        add_guardrail_to_applied_guardrails_header(
            request_data=request_data, guardrail_name=self.guardrail_name
        )
        return inputs

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.panw_prisma_airs import (
            PanwPrismaAirsGuardrailConfigModel,
        )

        return PanwPrismaAirsGuardrailConfigModel
