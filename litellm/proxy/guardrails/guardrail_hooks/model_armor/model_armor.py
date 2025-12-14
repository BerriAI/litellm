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
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    CallTypesLiteral,
    Choices,
    GuardrailStatus,
    ModelResponse,
    ModelResponseStream,
    StandardLoggingGuardrailInformation,
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
        # Set supported event hooks if not already provided
        if "event_hook" not in kwargs:
            kwargs["event_hook"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.post_call,
            ]

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
        """Create request body for Model Armor API with correct camelCase field names."""
        if source == "user_prompt":
            return {"userPromptData": {"text": content}}
        else:
            return {"modelResponseData": {"text": content}}

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
        content: Optional[str] = None,
        source: Literal["user_prompt", "model_response"] = "user_prompt",
        request_data: Optional[dict] = None,
        file_bytes: Optional[bytes] = None,
        file_type: Optional[str] = None,
    ) -> dict:
        """
        Make request to Model Armor API. Supports both text and file prompt sanitization.
        If file_bytes and file_type are provided, file prompt sanitization is performed.
        """
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
        if file_bytes is not None and file_type is not None:
            body = self.sanitize_file_prompt(file_bytes, file_type, source)
        elif content is not None:
            body = self._create_sanitize_request(content, source)
        else:
            raise ValueError(
                "Either content or file_bytes and file_type must be provided."
            )

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

    def sanitize_file_prompt(
        self, file_bytes: bytes, file_type: str, source: str = "user_prompt"
    ) -> dict:
        """
        Helper to build the request body for file prompt sanitization for Model Armor.
        file_type should be one of: PLAINTEXT_UTF8, PDF, WORD_DOCUMENT, EXCEL_DOCUMENT, POWERPOINT_DOCUMENT, TXT, CSV
        Returns the request body dict.
        """
        import base64

        base64_data = base64.b64encode(file_bytes).decode("utf-8")
        if source == "user_prompt":
            return {
                "userPromptData": {
                    "byteItem": {"byteDataType": file_type, "byteData": base64_data}
                }
            }
        else:
            return {
                "modelResponseData": {
                    "byteItem": {"byteDataType": file_type, "byteData": base64_data}
                }
            }

    def _should_block_content(
        self, armor_response: dict, allow_sanitization: bool = False
    ) -> bool:
        """Check if Model Armor response indicates content should be blocked, including both inspectResult and deidentifyResult."""
        sanitization_result = armor_response.get("sanitizationResult", {})
        filter_results = sanitization_result.get("filterResults", {})

        # filterResults can be a dict (named keys) or a list (array of filter result dicts)
        filter_result_items = []
        if isinstance(filter_results, dict):
            filter_result_items = list(filter_results.values())
        elif isinstance(filter_results, list):
            filter_result_items = filter_results

        for filt in filter_result_items:
            # Check RAI, PI/Jailbreak, Malicious URI, CSAM, Virus scan as before
            if filt.get("raiFilterResult", {}).get("matchState") == "MATCH_FOUND":
                return True
            if (
                filt.get("piAndJailbreakFilterResult", {}).get("matchState")
                == "MATCH_FOUND"
            ):
                return True
            if (
                filt.get("maliciousUriFilterResult", {}).get("matchState")
                == "MATCH_FOUND"
            ):
                return True
            if (
                filt.get("csamFilterFilterResult", {}).get("matchState")
                == "MATCH_FOUND"
            ):
                return True
            if filt.get("virusScanFilterResult", {}).get("matchState") == "MATCH_FOUND":
                return True
            # Check sdpFilterResult for both inspectResult and deidentifyResult
            sdp = filt.get("sdpFilterResult")
            if sdp:
                if sdp.get("inspectResult", {}).get("matchState") == "MATCH_FOUND":
                    return True
                # Only block on deidentifyResult if sanitization is not allowed
                if sdp.get("deidentifyResult", {}).get("matchState") == "MATCH_FOUND":
                    if not allow_sanitization:
                        return True
        # Fallback dict code removed; all cases handled above
        return False

    def _get_sanitized_content(self, armor_response: dict) -> Optional[str]:
        """
        Get the sanitized content from a Model Armor response, if available.
        Looks for sanitized text in deidentifyResult, and falls back to root-level fields if not found.
        """
        result = armor_response.get("sanitizationResult", {})
        filter_results = result.get("filterResults", {})

        # filterResults can be a dict (single filter) or a list (multiple filters)
        filters = (
            list(filter_results.values())
            if isinstance(filter_results, dict)
            else filter_results if isinstance(filter_results, list) else []
        )

        # Prefer sanitized text from deidentifyResult if present
        for filter_entry in filters:
            sdp = filter_entry.get("sdpFilterResult")
            if sdp:
                deid = sdp.get("deidentifyResult", {})
                sanitized = deid.get("data", {}).get("text", "")
                # If Model Armor found something and returned a sanitized version, use it
                if deid.get("matchState") == "MATCH_FOUND" and sanitized:
                    return sanitized

        # If no deidentifyResult, optionally check for inspectResult (rare, but could have findings)
        for filter_entry in filters:
            sdp = filter_entry.get("sdpFilterResult")
            if sdp:
                inspect = sdp.get("inspectResult", {})
                # If Model Armor flagged something but didn't sanitize, return None
                if inspect.get("matchState") == "MATCH_FOUND":
                    return None

        # Fallback: if Model Armor put sanitized text at the root, use it
        return armor_response.get("sanitizedText") or armor_response.get("text")

    def _process_response(
        self,
        response: Optional[dict],
        request_data: dict,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
    ):
        """
        Override to store only the Model Armor API response, not the entire data dict.
        This prevents circular references in logging.
        """
        # Retrieve the Model Armor response & status stored on the per-request `metadata` object.
        metadata = (
            request_data.get("metadata", {}) if isinstance(request_data, dict) else {}
        )

        guardrail_response = metadata.get("_model_armor_response", {})

        # Determine status – default to "success" but prefer the explicit value if present.
        guardrail_status: GuardrailStatus = metadata.get(
            "_model_armor_status", "success"
        )  # type: ignore

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=guardrail_response,
            request_data=request_data,
            guardrail_status=guardrail_status,
            duration=duration,
            start_time=start_time,
            end_time=end_time,
        )
        return response

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
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

            # Store the armor response for logging
            # Attach Model Armor response + evaluation status directly to the per-request metadata to avoid
            #   race-conditions between concurrent requests which share the same guardrail instance.
            #   This ensures each request logs its own Model Armor response instead of a potentially stale value
            #   overwritten by another coroutine.
            if isinstance(data, dict):
                metadata = data.setdefault(
                    "metadata", {}
                )  # ensures metadata exists and is unique per request
                metadata["_model_armor_response"] = armor_response
                # Pre-compute guardrail status for downstream logging. A blocked response will eventually raise
                #   an HTTPException, however in scenarios where the caller decides to ignore the exception (e.g.
                #   fail_on_error=False) we still want the correct status reflected.
                metadata["_model_armor_status"] = (
                    "blocked"
                    if self._should_block_content(
                        armor_response, allow_sanitization=self.mask_request_content
                    )
                    else "success"
                )
            # Check if content should be blocked
            if self._should_block_content(
                armor_response, allow_sanitization=self.mask_request_content
            ):
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
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, dict, None]:
        """During-call hook to sanitize user prompts in parallel with LLM call."""
        verbose_proxy_logger.debug("Inside Model Armor Moderation Hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        messages = data.get("messages")
        if not messages:
            verbose_proxy_logger.warning(
                "Model Armor: not running guardrail. No messages in data"
            )
            return data

        # Extract content from messages
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

            # Store the armor response for logging
            if isinstance(data, dict):
                metadata = data.setdefault("metadata", {})
                metadata["_model_armor_response"] = armor_response
                metadata["_model_armor_status"] = (
                    "blocked"
                    if self._should_block_content(
                        armor_response, allow_sanitization=self.mask_request_content
                    )
                    else "success"
                )

            # Check if content should be blocked
            if self._should_block_content(
                armor_response, allow_sanitization=self.mask_request_content
            ):
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
                "Model Armor moderation error: %s", str(e), exc_info=True
            )
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
            add_guardrail_response_to_standard_logging_object,
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

            # Attach Model Armor response & status to this request's metadata to prevent race conditions
            if isinstance(armor_response, dict):
                model_armor_logged_object = {
                    "model_armor_response": armor_response,
                    "model_armor_status": (
                        "blocked"
                        if self._should_block_content(
                            armor_response,
                            allow_sanitization=self.mask_response_content,
                        )
                        else "success"
                    ),
                }
                standard_logging_guardrail_information = (
                    StandardLoggingGuardrailInformation(
                        guardrail_name=self.guardrail_name,
                        guardrail_provider="model_armor",
                        guardrail_mode=GuardrailEventHooks.post_call,
                        guardrail_response=model_armor_logged_object,
                        guardrail_status="success",
                        start_time=data.get("start_time"),
                    )
                )
                add_guardrail_response_to_standard_logging_object(
                    litellm_logging_obj=data.get("litellm_logging_obj"),
                    guardrail_response=standard_logging_guardrail_information,
                )

            # Check if content should be blocked
            if self._should_block_content(
                armor_response, allow_sanitization=self.mask_response_content
            ):
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

        return response

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

                    # Attach Model Armor response & status to this request's metadata to avoid race conditions
                    if isinstance(request_data, dict):
                        metadata = request_data.setdefault("metadata", {})
                        metadata["_model_armor_response"] = armor_response
                        metadata["_model_armor_status"] = (
                            "blocked"
                            if self._should_block_content(armor_response)
                            else "success"
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
