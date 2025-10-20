#!/usr/bin/env python3
"""
Palo Alto Networks Prisma AI Runtime Security (AIRS) Guardrail Integration for LiteLLM

Provides real-time threat detection, DLP, URL filtering, content masking, and policy enforcement for AI applications.
"""

import os
from litellm._uuid import uuid
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
from litellm.types.utils import ModelResponse

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
        profile_name: PANW security profile name
        mask_request_content: Apply masking to prompts (default: False)
        mask_response_content: Apply masking to responses (default: False)
        mask_on_block: Backwards compatible flag that enables both request and response masking
    """

    def __init__(
        self,
        guardrail_name: str,
        profile_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        default_on: bool = True,
        mask_on_block: bool = False,
        mask_request_content: bool = False,
        mask_response_content: bool = False,
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
            **kwargs
        )

        # Store configuration with env var fallbacks
        self.api_key = api_key or os.getenv("PANW_PRISMA_AIRS_API_KEY")
        self.api_base = (
            api_base
            or os.getenv("PANW_PRISMA_AIRS_API_BASE")
            or "https://service.api.aisecurity.paloaltonetworks.com"
        )
        self.profile_name = profile_name
        
        # Validate required configuration
        if not self.api_key:
            raise ValueError(
                "PANW Prisma AIRS: api_key is required. "
                "Set it via config or PANW_PRISMA_AIRS_API_KEY environment variable."
            )

        verbose_proxy_logger.info(
            f"Initialized PANW Prisma AIRS Guardrail: {guardrail_name} "
            f"(mask_request={self.mask_request_content}, mask_response={self.mask_response_content})"
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
                        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                            for tool_call in choice.message.tool_calls:
                                if hasattr(tool_call, "function") and hasattr(tool_call.function, "arguments"):
                                    text_parts.append(str(tool_call.function.arguments))
                        
                        # Extract function call arguments (legacy)
                        if hasattr(choice.message, "function_call") and choice.message.function_call:
                            if hasattr(choice.message.function_call, "arguments"):
                                text_parts.append(str(choice.message.function_call.arguments))
            
            return " ".join(text_parts) if text_parts else ""
        except (AttributeError, IndexError) as e:
            verbose_proxy_logger.error(
                f"PANW Prisma AIRS: Error extracting response text: {str(e)}"
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

    def _get_masked_text(self, scan_result: Dict[str, Any], is_response: bool = False) -> Optional[str]:
        """Extract masked text from PANW scan result."""
        masked_key = "response_masked_data" if is_response else "prompt_masked_data"
        masked_data = scan_result.get(masked_key)
        if masked_data and isinstance(masked_data, dict):
            return masked_data.get("data")
        return None

    def _apply_masking_to_messages(
        self,
        messages: List[Dict[str, Any]],
        masked_text: str
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
                return messages[:idx] + [new_message] + messages[idx+1:]
        
        return messages

    def _apply_masking_to_response(
        self, 
        response: ModelResponse,
        masked_text: str
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
                                new_content.append({"type": "text", "text": masked_text})
                            else:
                                # Preserve non-text parts (images, etc.)
                                new_content.append(part)
                        choice.message.content = new_content  # type: ignore
                
                # Mask tool call arguments
                if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        if hasattr(tool_call, "function") and hasattr(tool_call.function, "arguments"):
                            tool_call.function.arguments = masked_text
                
                # Mask function call arguments (legacy)
                if hasattr(choice.message, "function_call") and choice.message.function_call:
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
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        verbose_proxy_logger.info("PANW Prisma AIRS: Running pre-call prompt scan")

        # Check if guardrail should run for this request
        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        try:
            # Extract prompt text from messages (chat completion) or prompt (text completion)
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

            if not prompt_text:
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: No user prompt found in request (checked 'messages' and 'prompt' fields)"
                )
                return None

            # Prepare metadata
            metadata = {
                "user": data.get("user") or "litellm_user",
                "model": data.get("model") or "unknown",
            }

            # Scan prompt with PANW Prisma AIRS
            scan_result = await self._call_panw_api(
                content=prompt_text, is_response=False, metadata=metadata
            )

            action = scan_result.get("action", "block")
            category = scan_result.get("category", "unknown")
            masked_text = self._get_masked_text(scan_result, is_response=False)

            # If action is "allow", apply masking if available and allow through
            if action == "allow":
                if masked_text:
                    if messages:
                        data["messages"] = self._apply_masking_to_messages(messages, masked_text)
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
                    data["messages"] = self._apply_masking_to_messages(messages, masked_text)
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
                    }
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

        try:
            # Extract response text
            response_text = self._extract_response_text(response)

            if not response_text:
                verbose_proxy_logger.warning(
                    "PANW Prisma AIRS: No response content found to scan"
                )
                return response

            # Prepare metadata
            metadata = {
                "user": data.get("user") or "litellm_user",
                "model": data.get("model") or "unknown",
            }

            # Scan response with PANW Prisma AIRS
            scan_result = await self._call_panw_api(
                content=response_text, is_response=True, metadata=metadata
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
                }
            )

    async def _scan_and_process_streaming_response(
        self,
        assembled_model_response: ModelResponse,
        request_data: dict,
    ) -> Tuple[bool, ModelResponse]:
        """
        Scan assembled streaming response and apply masking if needed.
        Returns (content_was_modified, response).
        """
        content_was_modified = False
        response_text = self._extract_response_text(assembled_model_response)
        
        if not response_text or not response_text.strip():
            verbose_proxy_logger.info("PANW Prisma AIRS: No content to scan in streaming response")
            return content_was_modified, assembled_model_response
        
        # Prepare metadata and scan
        metadata = {
            "user": request_data.get("user") or "litellm_user",
            "model": request_data.get("model") or "unknown",
        }
        
        scan_result = await self._call_panw_api(
            content=response_text, is_response=True, metadata=metadata
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
        
        return content_was_modified, assembled_model_response

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
        
        verbose_proxy_logger.info("PANW Prisma AIRS: Running post-call streaming scan")
        
        all_chunks = []
        content_was_modified = False
        
        try:
            # Collect all chunks
            async for chunk in response:
                all_chunks.append(chunk)
            
            # Assemble complete response from chunks
            assembled_model_response = stream_chunk_builder(chunks=all_chunks)
            
            if isinstance(assembled_model_response, ModelResponse):
                # Scan and process the assembled response
                content_was_modified, assembled_model_response = await self._scan_and_process_streaming_response(
                    assembled_model_response, request_data
                )
                
                # Add guardrail to applied guardrails header for observability
                add_guardrail_to_applied_guardrails_header(
                    request_data=request_data, guardrail_name=self.guardrail_name
                )
                
                # Only use MockResponseIterator if content was modified
                # Otherwise, yield original chunks to preserve streaming behavior
                if content_was_modified:
                    mock_response = MockResponseIterator(model_response=assembled_model_response)
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
                }
            )

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.panw_prisma_airs import (
            PanwPrismaAirsGuardrailConfigModel,
        )

        return PanwPrismaAirsGuardrailConfigModel
