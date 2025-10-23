#!/usr/bin/env python3
"""
PANW Prisma AIRS Built-in Guardrail for LiteLLM

"""

import os
from litellm._uuid import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Type, cast

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
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


class PanwPrismaAirsHandler(CustomGuardrail):
    """
    LiteLLM Built-in Guardrail for Palo Alto Networks Prisma AI Runtime Security (AIRS).

    This guardrail scans prompts and responses using the PANW Prisma AIRS API to detect
    malicious content, injection attempts, and policy violations.

    Configuration:
        guardrail_name: Name of the guardrail instance
        api_key: PANW Prisma AIRS API key
        api_base: PANW Prisma AIRS API endpoint
        profile_name: PANW Prisma AIRS security profile name
        default_on: Whether to enable by default
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: str,
        api_base: str,
        profile_name: str,
        default_on: bool = True,
        **kwargs,
    ):
        """Initialize PANW Prisma AIRS guardrail handler."""

        # Initialize parent CustomGuardrail
        super().__init__(guardrail_name=guardrail_name, default_on=default_on, **kwargs)

        # Store configuration
        self.api_key = api_key or os.getenv("PANW_PRISMA_AIRS_API_KEY")
        self.api_base = (
            api_base
            or os.getenv("PANW_PRISMA_AIRS_API_BASE")
            or "https://service.api.aisecurity.paloaltonetworks.com"
        )
        self.profile_name = profile_name

        verbose_proxy_logger.debug(
            f"Initialized PANW Prisma AIRS Guardrail: {guardrail_name}"
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
        """Extract text from LLM response."""
        try:
            from litellm.types.utils import Choices

            if (
                hasattr(response, "choices")
                and response.choices
                and len(response.choices) > 0
                and hasattr(response.choices[0], "message")
            ):
                return cast(Choices, response.choices[0]).message.content or ""
        except (AttributeError, IndexError):
            verbose_proxy_logger.error(
                "PANW Prisma AIRS: Error extracting response text"
            )
        return ""

    async def _call_panw_api(
        self,
        content: str,
        is_response: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call PANW Prisma AIRS API to scan content."""

        if not content.strip():
            return {"action": "allow", "category": "empty"}

        # Build request payload
        transaction_id = (
            f"litellm-{'resp' if is_response else 'req'}-{uuid.uuid4().hex[:8]}"
        )

        payload = {
            "tr_id": transaction_id,
            "ai_profile": {"profile_name": self.profile_name},
            "metadata": {
                "app_user": (
                    metadata.get("user", "litellm_user") if metadata else "litellm_user"
                ),
                "ai_model": metadata.get("model", "unknown") if metadata else "unknown",
                "source": "litellm_builtin_guardrail",
            },
            "contents": [{"response" if is_response else "prompt": content}],
        }

        if is_response:
            payload["metadata"]["is_response"] = True  # type: ignore[index]

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-pan-token": self.api_key,
        }

        try:
            # Use LiteLLM's async HTTP client
            async_client = get_async_httpx_client(
                llm_provider=httpxSpecialProvider.LoggingCallback
            )

            response = await async_client.post(
                f"{self.api_base}/v1/scan/sync/request",
                headers=headers,
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()

            result = response.json()

            # Validate response format
            if "action" not in result:
                verbose_proxy_logger.error(
                    f"PANW Prisma AIRS: Invalid API response format: {result}"
                )
                return {"action": "block", "category": "api_error"}

            verbose_proxy_logger.debug(
                f"PANW Prisma AIRS: Scan result - Action: {result.get('action')}, Category: {result.get('category', 'unknown')}"
            )
            return result

        except Exception as e:
            verbose_proxy_logger.error(f"PANW Prisma AIRS: API call failed: {str(e)}")
            return {"action": "block", "category": "api_error"}

    def _build_error_detail(
        self, scan_result: Dict[str, Any], is_response: bool = False
    ) -> Dict[str, Any]:
        """Build enhanced error detail with scan information."""
        action_type = "Response" if is_response else "Prompt"
        code_suffix = "_response_blocked" if is_response else "_blocked"
        detection_key = "response_detected" if is_response else "prompt_detected"

        category = scan_result.get("category", "unknown")
        error_msg = f"{action_type} blocked by PANW Prisma AI Security policy (Category: {category})"

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

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: Dict[str, Any],
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
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-call hook to scan user prompts before sending to LLM.

        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.debug("PANW Prisma AIRS: Running pre-call prompt scan")

        # Extract prompt text from messages
        messages = data.get("messages", [])
        prompt_text = self._extract_text_from_messages(messages)

        if not prompt_text:
            verbose_proxy_logger.warning(
                "PANW Prisma AIRS: No user prompt found in request"
            )
            return None

        # Prepare metadata
        metadata = {
            "user": data.get("user", "litellm_user"),
            "model": data.get("model", "unknown"),
        }

        # Scan prompt with PANW Prisma AIRS
        scan_result = await self._call_panw_api(
            content=prompt_text, is_response=False, metadata=metadata
        )

        action = scan_result.get("action", "block")
        category = scan_result.get("category", "unknown")

        if action == "allow":
            verbose_proxy_logger.debug(
                f"PANW Prisma AIRS: Response allowed (Category: {category})"
            )

        else:
            error_detail = self._build_error_detail(scan_result, is_response=True)
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS: {error_detail['error']['message']}"
            )
            raise HTTPException(status_code=400, detail=error_detail)

        return None

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: ModelResponse,
    ) -> ModelResponse:
        """
        Post-call hook to scan LLM responses before returning to user.

        Raises HTTPException if response should be blocked.
        """
        verbose_proxy_logger.debug("PANW Prisma AIRS: Running post-call response scan")

        # Extract response text
        response_text = self._extract_response_text(response)

        if not response_text:
            verbose_proxy_logger.warning(
                "PANW Prisma AIRS: No response content found to scan"
            )
            return response

        # Prepare metadata
        metadata = {
            "user": data.get("user", "litellm_user"),
            "model": data.get("model", "unknown"),
        }

        # Scan response with PANW Prisma AIRS
        scan_result = await self._call_panw_api(
            content=response_text, is_response=True, metadata=metadata
        )

        action = scan_result.get("action", "block")
        category = scan_result.get("category", "unknown")

        if action == "allow":
            verbose_proxy_logger.debug(
                f"PANW Prisma AIRS: Response allowed (Category: {category})"
            )

        else:
            error_detail = self._build_error_detail(scan_result, is_response=True)
            verbose_proxy_logger.warning(
                f"PANW Prisma AIRS: {error_detail['error']['message']}"
            )
            raise HTTPException(status_code=400, detail=error_detail)

        return response

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.panw_prisma_airs import (
            PanwPrismaAirsGuardrailConfigModel,
        )

        return PanwPrismaAirsGuardrailConfigModel
