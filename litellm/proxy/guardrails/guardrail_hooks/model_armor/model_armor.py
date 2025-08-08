from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from fastapi import HTTPException

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    Choices,
    ModelResponse,
    ModelResponseStream,
)

GUARDRAIL_NAME = "model_armor"


class ModelArmorGuardrail(CustomGuardrail, VertexBase):
    """
    Google Cloud Model Armor Guardrail integration for LiteLLM.

    Supports:
    - Pre-call sanitization (sanitizeUserPrompt)
    - Post-call sanitization (sanitizeModelResponse)
    """

    def __init__(
        self,
        template_id: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        credentials: Optional[Any] = None,
        api_endpoint: Optional[str] = None,
        **kwargs,
    ):
        # Initialize parent classes first
        super().__init__(**kwargs)
        VertexBase.__init__(self)
        
        # Then set our attributes (this ensures project_id is not overwritten)
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.template_id = template_id
        self.project_id = project_id
        self.location = location or "us-central1"
        self.credentials = credentials
        self.api_endpoint = api_endpoint

        # Store optional params
        self.optional_params = kwargs

        verbose_proxy_logger.debug(
            "Model Armor Guardrail initialized with template_id: %s, project_id: %s, location: %s",
            self.template_id,
            self.project_id,
            self.location,
        )

    def _get_api_endpoint(self) -> str:
        """Get the API endpoint for Model Armor."""
        if self.api_endpoint:
            return self.api_endpoint
        return f"https://modelarmor.{self.location}.rep.googleapis.com"

    def _create_sanitize_request(
        self, content: str, source: Literal["user_prompt", "model_response"]
    ) -> dict:
        """Create request body for Model Armor API."""
        if source == "user_prompt":
            return {"user_prompt_data": {"text": content}}
        else:
            return {"model_response_data": {"text": content}}



    def _extract_content_from_response(
        self, response: Union[Any, ModelResponse]
    ) -> str:
        """
        Extract text content from model response.
        
        Returns empty string for non-text responses (TTS, images, etc.) to skip guardrail processing.
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_content_from_model_response,
        )

        # Handle ModelResponse objects
        if isinstance(response, litellm.ModelResponse):
            return get_content_from_model_response(response)

        # For non-ModelResponse types (e.g., TTS, images), return empty string
        # These response types are not text-based and shouldn't be processed by text guardrails
        verbose_proxy_logger.debug(
            "Model Armor: Skipping non-ModelResponse type: %s", type(response).__name__
        )
        return ""

    async def make_model_armor_request(
        self,
        content: str,
        source: Literal["user_prompt", "model_response"],
        request_data: Optional[dict] = None,
    ) -> dict:
        """Make request to Model Armor API."""
        # Get access token using VertexBase auth
        access_token, resolved_project_id = await self._ensure_access_token_async(
            credentials=self.credentials,
            project_id=self.project_id,
            custom_llm_provider="vertex_ai",
        )

        # Use resolved project ID if not explicitly set
        if not self.project_id and resolved_project_id:
            self.project_id = resolved_project_id

        # Construct URL
        endpoint = self._get_api_endpoint()
        if source == "user_prompt":
            url = f"{endpoint}/v1/projects/{self.project_id}/locations/{self.location}/templates/{self.template_id}:sanitizeUserPrompt"
        else:
            url = f"{endpoint}/v1/projects/{self.project_id}/locations/{self.location}/templates/{self.template_id}:sanitizeModelResponse"

        # Create request body
        body = self._create_sanitize_request(content, source)

        # Set headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        verbose_proxy_logger.debug(
            "Model Armor request - URL: %s, Body: %s",
            url,
            body,
        )

        # Make request
        if self.async_handler is None:
            raise ValueError("Async handler not initialized")

        response = await self.async_handler.post(
            url=url,
            json=body,
            headers=headers,
        )

        verbose_proxy_logger.debug(
            "Model Armor response - Status: %s, Body: %s",
            response.status_code,
            response.text,
        )

        if response.status_code != 200:
            verbose_proxy_logger.error(
                "Model Armor API error - Status: %s, Response: %s",
                response.status_code,
                response.text,
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Model Armor API error: {response.text}",
            )

        json_response = response.json()
        if hasattr(json_response, "__await__"):
            return await json_response
        return json_response

    def _should_block_content(self, armor_response: dict) -> bool:
        """Check if Model Armor response indicates content should be blocked."""
        # Model Armor may return different response structures
        # This is a basic implementation - adjust based on actual API response
        if armor_response.get("blocked", False):
            return True

        # Check for sanitization actions
        if armor_response.get("action") == "BLOCK":
            return True

        return False

    def _get_sanitized_content(self, armor_response: dict) -> Optional[str]:
        """Extract sanitized content from Model Armor response."""
        # This depends on the actual Model Armor API response structure
        # Adjust based on documentation
        return armor_response.get("sanitized_text") or armor_response.get("text")

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
        ],
    ) -> Union[Exception, str, dict, None]:
        """Pre-call hook to sanitize user prompts."""
        verbose_proxy_logger.debug("Inside Model Armor Pre-Call Hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        messages = data.get("messages")
        if not messages:
            verbose_proxy_logger.warning(
                "Model Armor: not running guardrail. No messages in data"
            )
            return data

        # Extract content from messages using helper from common_utils
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            get_last_user_message,
        )

        content = get_last_user_message(messages)
        if not content:
            return data

        # Make Model Armor request
        try:
            armor_response = await self.make_model_armor_request(
                content=content,
                source="user_prompt",
                request_data=data,
            )

            # Check if content should be blocked
            if self._should_block_content(armor_response):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Content blocked by Model Armor",
                        "model_armor_response": armor_response,
                    },
                )

            # If mask_request_content is enabled, update messages with sanitized content
            if self.mask_request_content:
                sanitized_content = self._get_sanitized_content(armor_response)
                if sanitized_content and sanitized_content != content:
                    # Use the helper to set the last user message with sanitized content
                    from litellm.litellm_core_utils.prompt_templates.common_utils import (
                        set_last_user_message,
                    )

                    data["messages"] = set_last_user_message(
                        messages, sanitized_content
                    )

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Model Armor pre-call error: %s", str(e), exc_info=True
            )
            # Depending on configuration, either fail or continue
            if self.optional_params.get("fail_on_error", True):
                raise

        # Add guardrail to headers
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        """Post-call hook to sanitize model responses."""
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return

        # Extract content from response
        content = self._extract_content_from_response(response)
        if not content:
            verbose_proxy_logger.debug(
                "Model Armor: No text content to process in response, skipping guardrail"
            )
            return

        # Make Model Armor request
        try:
            armor_response = await self.make_model_armor_request(
                content=content,
                source="model_response",
                request_data=data,
            )

            # Check if content should be blocked
            if self._should_block_content(armor_response):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Response blocked by Model Armor",
                        "model_armor_response": armor_response,
                    },
                )

            # If mask_response_content is enabled, update response with sanitized content
            if self.mask_response_content:
                sanitized_content = self._get_sanitized_content(armor_response)
                if sanitized_content and sanitized_content != content:
                    # Update response content
                    if isinstance(response, litellm.ModelResponse):
                        for choice in response.choices:
                            if isinstance(choice, Choices):
                                if choice.message.content:
                                    choice.message.content = sanitized_content

        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                "Model Armor post-call error: %s", str(e), exc_info=True
            )
            if self.optional_params.get("fail_on_error", True):
                raise

        # Add guardrail to headers
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """Process streaming response chunks."""

        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder

        # Collect all chunks
        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        # Build complete response
        assembled_response = stream_chunk_builder(chunks=all_chunks)

        if isinstance(assembled_response, ModelResponse):
            # Extract content
            content = self._extract_content_from_response(assembled_response)

            if content:
                try:
                    # Check with Model Armor
                    armor_response = await self.make_model_armor_request(
                        content=content,
                        source="model_response",
                        request_data=request_data,
                    )

                    # Check if blocked
                    if self._should_block_content(armor_response):
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "Streaming response blocked by Model Armor",
                                "model_armor_response": armor_response,
                            },
                        )

                    # Apply sanitization if enabled
                    if self.mask_response_content:
                        sanitized_content = self._get_sanitized_content(armor_response)
                        if sanitized_content and sanitized_content != content:
                            # Update assembled response
                            for choice in assembled_response.choices:
                                if isinstance(choice, Choices):
                                    if choice.message.content:
                                        choice.message.content = sanitized_content

                            # Return sanitized stream
                            mock_response = MockResponseIterator(
                                model_response=assembled_response
                            )
                            async for chunk in mock_response:
                                yield chunk
                            return

                except HTTPException:
                    raise
                except Exception as e:
                    verbose_proxy_logger.error(
                        "Model Armor streaming error: %s", str(e), exc_info=True
                    )
                    if self.optional_params.get("fail_on_error", True):
                        raise
            else:
                verbose_proxy_logger.debug(
                    "Model Armor: No text content in streaming response, skipping guardrail"
                )

        # Return original chunks if no sanitization needed
        for chunk in all_chunks:
            yield chunk

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        """
        Get the config model for the Model Armor guardrail.
        """
        from litellm.types.proxy.guardrails.guardrail_hooks.model_armor import (
            ModelArmorGuardrailConfigModel,
        )

        return ModelArmorGuardrailConfigModel
