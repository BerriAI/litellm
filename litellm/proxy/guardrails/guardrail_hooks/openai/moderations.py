#!/usr/bin/env python3
"""
OpenAI Moderation Guardrail Integration for LiteLLM
"""

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

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

from .base import OpenAIGuardrailBase

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import AllMessageValues, OpenAIModerationResponse
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from litellm.types.utils import ModelResponse, ModelResponseStream


class OpenAIModerationGuardrail(OpenAIGuardrailBase, CustomGuardrail):
    """
    LiteLLM Built-in Guardrail for OpenAI Content Moderation.

    This guardrail scans prompts and responses using the OpenAI Moderation API to detect
    harmful content, including violence, hate, harassment, self-harm, sexual content, etc.

    Configuration:
        guardrail_name: Name of the guardrail instance
        api_key: OpenAI API key
        api_base: OpenAI API endpoint
        model: OpenAI moderation model to use
        default_on: Whether to enable by default
    """

    def __init__(
        self,
        guardrail_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[Literal["omni-moderation-latest", "text-moderation-latest"]] = None,
        **kwargs,
    ):
        """Initialize OpenAI Moderation guardrail handler."""
        from litellm.types.guardrails import GuardrailEventHooks

        # Initialize parent CustomGuardrail
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
        
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Store configuration
        self.api_key = api_key or self._get_api_key()
        self.api_base = api_base or "https://api.openai.com/v1"
        self.model: Literal["omni-moderation-latest", "text-moderation-latest"] = model or "omni-moderation-latest"

        if not self.api_key:
            raise ValueError("OpenAI Moderation: api_key is required. Set OPENAI_API_KEY environment variable or pass it in configuration.")

        verbose_proxy_logger.info(
            f"Initialized OpenAI Moderation Guardrail: {guardrail_name} with model: {self.model}"
        )

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables or litellm configuration"""
        import os

        import litellm
        from litellm.secret_managers.main import get_secret_str
        
        return (
            os.environ.get("OPENAI_API_KEY")
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )

    async def async_make_request(
        self, input_text: str
    ) -> "OpenAIModerationResponse":
        """
        Make a request to the OpenAI Moderation API.
        """
        request_body = {
            "model": self.model,
            "input": input_text
        }
        
        verbose_proxy_logger.debug(
            "OpenAI Moderation guard request: %s", request_body
        )
        
        response = await self.async_handler.post(
            url=f"{self.api_base}/moderations",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_body,
        )

        verbose_proxy_logger.debug(
            "OpenAI Moderation guard response: %s", response.json()
        )
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail={
                    "error": "OpenAI Moderation API request failed",
                    "details": response.text,
                },
            )

        from litellm.types.llms.openai import OpenAIModerationResponse
        return OpenAIModerationResponse(**response.json())

    def _check_moderation_result(self, moderation_response: "OpenAIModerationResponse") -> None:
        """
        Check if the moderation response indicates harmful content and raise exception if needed.
        """
        if not moderation_response.results:
            return

        result = moderation_response.results[0]
        if result.flagged:
            # Build detailed violation information
            violated_categories = []
            if result.categories:
                for category, is_violated in result.categories.items():
                    if is_violated:
                        violated_categories.append(category)

            violation_details = {
                "violated_categories": violated_categories,
                "category_scores": result.category_scores or {},
            }

            verbose_proxy_logger.warning(
                "OpenAI Moderation: Content flagged for violations: %s", 
                violation_details
            )
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Violated OpenAI moderation policy",
                    "moderation_result": violation_details,
                },
            )

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
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
        ],
    ) -> Optional[Dict[str, Any]]:
        """
        Pre-call hook to scan user prompts before sending to LLM.

        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.info(
            "OpenAI Moderation: Running pre-call prompt scan, on call_type: %s",
            call_type,
        )
        
        # Skip moderation calls to avoid infinite recursion
        if call_type == "moderation":
            return data

        new_messages: Optional[List["AllMessageValues"]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning(
                "OpenAI Moderation: not running guardrail. No messages in data"
            )
            return data

        user_prompt = self.get_user_prompt(new_messages)
        if user_prompt:
            verbose_proxy_logger.info(
                f"OpenAI Moderation: User prompt: {user_prompt[:100]}..."  # Log first 100 chars for debugging
            )
            
            moderation_response = await self.async_make_request(
                input_text=user_prompt,
            )
            
            # Check if content is flagged and raise exception if needed
            self._check_moderation_result(moderation_response)
        else:
            verbose_proxy_logger.warning(
                "OpenAI Moderation: No user prompt found"
            )

        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: "UserAPIKeyAuth",
        call_type: Literal[
            "completion",
            "embeddings", 
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
        ],
    ) -> Optional[Dict[str, Any]]:
        """
        Moderation hook to scan user prompts during call processing.

        Raises HTTPException if content should be blocked.
        """
        verbose_proxy_logger.info(
            "OpenAI Moderation: Running moderation hook, on call_type: %s",
            call_type,
        )
        
        # Skip moderation calls to avoid infinite recursion
        if call_type == "moderation":
            return data

        new_messages: Optional[List["AllMessageValues"]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning(
                "OpenAI Moderation: not running guardrail. No messages in data"
            )
            return data

        user_prompt = self.get_user_prompt(new_messages)
        if user_prompt:
            moderation_response = await self.async_make_request(
                input_text=user_prompt,
            )
            
            # Check if content is flagged and raise exception if needed
            self._check_moderation_result(moderation_response)

        return data

    @log_guardrail_information
    async def async_post_call_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: "UserAPIKeyAuth",
        response: "ModelResponse",
    ) -> "ModelResponse":
        """
        Post-call hook to scan LLM responses before returning to user.

        Raises HTTPException if response should be blocked.
        """
        verbose_proxy_logger.info(
            "OpenAI Moderation: Running post-call response scan"
        )

        # Extract response text for moderation
        response_text = self._extract_response_text(response)
        if response_text:
            verbose_proxy_logger.info(
                f"OpenAI Moderation: Response text: {response_text[:100]}..."  # Log first 100 chars
            )
            
            moderation_response = await self.async_make_request(
                input_text=response_text,
            )
            
            # Check if content is flagged and raise exception if needed
            self._check_moderation_result(moderation_response)

        return response

    @log_guardrail_information
    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        response: Any,
        request_data: Dict[str, Any],
    ) -> AsyncGenerator["ModelResponseStream", None]:
        """
        Process streaming response chunks for OpenAI moderation.

        Collects all chunks from the stream, assembles them into a complete response,
        and applies moderation check. If content violates moderation policy, raises HTTPException.
        """
        # Import here to avoid circular imports
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import TextCompletionResponse

        verbose_proxy_logger.info(
            "OpenAI Moderation: Running streaming response scan"
        )

        # Collect all chunks to process them together
        all_chunks: List["ModelResponseStream"] = []
        async for chunk in response:
            all_chunks.append(chunk)

        # Assemble the complete response from chunks
        assembled_model_response: Optional[
            Union["ModelResponse", TextCompletionResponse]
        ] = stream_chunk_builder(
            chunks=all_chunks,
        )

        if isinstance(assembled_model_response, (type(None), TextCompletionResponse)):
            # If we can't assemble a ModelResponse or it's a text completion, 
            # just yield the original chunks without moderation
            verbose_proxy_logger.warning(
                "OpenAI Moderation: Could not assemble ModelResponse from chunks, skipping moderation"
            )
            for chunk in all_chunks:
                yield chunk
            return

        # Extract response text for moderation
        response_text = self._extract_response_text(assembled_model_response)
        if response_text:
            verbose_proxy_logger.info(
                f"OpenAI Moderation: Streaming response text: {response_text[:100]}..."  # Log first 100 chars
            )
            
            # Make moderation request - this will raise HTTPException if content is flagged
            moderation_response = await self.async_make_request(
                input_text=response_text,
            )
            
            # Check if content is flagged and raise exception if needed
            self._check_moderation_result(moderation_response)

        # If we reach here, content passed moderation - yield the original chunks
        mock_response = MockResponseIterator(
            model_response=assembled_model_response
        )

        # Return the reconstructed stream
        async for chunk in mock_response:
            yield chunk

    def _extract_response_text(self, response: "ModelResponse") -> Optional[str]:
        """
        Extract text content from the model response for moderation.
        """
        if not hasattr(response, 'choices') or not response.choices:
            return None

        response_texts = []
        for choice in response.choices:
            try:
                # Try to get content from message (chat completion)
                message = getattr(choice, 'message', None)
                if message:
                    content = getattr(message, 'content', None)
                    if content and isinstance(content, str):
                        response_texts.append(content)
                        continue
                
                # Try to get text (text completion)
                text = getattr(choice, 'text', None)
                if text and isinstance(text, str):
                    response_texts.append(text)
                    continue
                
                # Try to get content from delta (streaming)
                delta = getattr(choice, 'delta', None)
                if delta:
                    content = getattr(delta, 'content', None)
                    if content and isinstance(content, str):
                        response_texts.append(content)
                        continue
                        
            except (AttributeError, TypeError):
                # Skip choices that don't have expected attributes
                continue

        return "\n".join(response_texts) if response_texts else None

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """
        Get the config model for the OpenAI Moderation guardrail.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.openai.openai_moderation import (
            OpenAIModerationGuardrailConfigModel,
        )

        return OpenAIModerationGuardrailConfigModel
