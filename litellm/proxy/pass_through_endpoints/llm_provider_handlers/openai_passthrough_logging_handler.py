"""
OpenAI Passthrough Logging Handler

Handles cost tracking and logging for OpenAI passthrough endpoints, specifically /chat/completions.
"""

import base64
from datetime import datetime
from typing import List, Optional, Union, cast
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.openai.openai import OpenAIConfig
from litellm.llms.openai.openai import OpenAIConfig as OpenAIConfigType
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict, UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    BasePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import (
    ImageResponse,
    LiteLLMBatch,
    LlmProviders,
    PassthroughCallTypes,
    SpecialEnums,
)
from litellm.utils import ModelResponse, TextCompletionResponse


class OpenAIPassthroughLoggingHandler(BasePassthroughLoggingHandler):
    @staticmethod
    def _get_unified_batch_id(batch_id: str, model_id: str) -> str:
        """
        Create a properly formatted and encoded unified batch ID for passthrough batches.

        Args:
            model_id: The model ID (e.g., 'gpt-4o')
            batch_id: The OpenAI batch ID (e.g., 'batch_123')

        Returns:
            Base64-encoded unified batch ID that passes _is_base64_encoded_unified_file_id validation
        """
        unified_batch_id = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(
            model_id, batch_id
        )
        return base64.urlsafe_b64encode(unified_batch_id.encode()).decode().rstrip("=")

    @staticmethod
    def _extract_model_from_batch_output(output_file_id: str) -> Optional[str]:
        """
        Extract the model ID from a completed batch's output file content.
        This is used because in passthrough batches, the user doesn't specify the model, so we need to extract it from the file.

        Args:
            output_file_id: The ID of the batch output file

        Returns:
            The model ID extracted from the first request in the batch, or None if extraction fails
        """
        try:
            from litellm.batches.batch_utils import _get_file_content_as_dictionary
            from litellm.files.main import file_content

            # Retrieve the file content
            file_response = file_content(
                file_id=output_file_id,
                custom_llm_provider="openai",  # Passthrough batches are always OpenAI
            )

            # Handle both sync and async responses
            if hasattr(file_response, 'content'):
                content = file_response.content
            else:
                # If it's a coroutine, we can't handle it in sync context
                return None

            # Parse the file content as JSON Lines
            file_content_dict = _get_file_content_as_dictionary(content)

            # Extract model from the first request
            if file_content_dict and len(file_content_dict) > 0:
                first_request = file_content_dict[0]
                if isinstance(first_request, dict) and "body" in first_request:
                    body = first_request["body"]
                    if isinstance(body, dict) and "model" in body:
                        model = body["model"]
                        return model

            return None

        except Exception:
            return None

    @staticmethod
    def _create_user_api_key_dict_from_kwargs(kwargs: dict) -> Optional[UserAPIKeyAuth]:
        """
        Extract user information from kwargs metadata and create UserAPIKeyAuth object.

        Args:
            kwargs: The kwargs dictionary containing litellm_params and metadata

        Returns:
            UserAPIKeyAuth object if user_id is found, None otherwise
        """
        try:
            # Extract user information from kwargs metadata
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {})

            # Extract user information from metadata
            user_id = metadata.get("user_api_key_user_id")
            api_key = metadata.get("user_api_key")
            team_id = metadata.get("user_api_key_team_id")
            team_alias = metadata.get("user_api_key_team_alias")
            user_email = metadata.get("user_api_key_user_email")

            if not user_id:
                return None

            # Create user_api_key_dict for managed files hook
            return UserAPIKeyAuth(
                user_id=user_id,
                api_key=api_key,
                team_id=team_id,
                team_alias=team_alias,
                user_role=None,  # Not available in metadata
                user_email=user_email,
                max_budget=None,  # Not available in metadata
                models=[],  # Not available in metadata
                metadata=metadata,  # Pass the full metadata
            )
        except Exception:
            return None

    """
    OpenAI-specific passthrough logging handler that provides cost tracking for /chat/completions endpoints.
    """

    @property
    def llm_provider_name(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def get_provider_config(self, model: str) -> OpenAIConfigType:
        """Get OpenAI provider configuration for the given model."""
        return OpenAIConfig()

    @staticmethod
    def is_openai_chat_completions_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI chat completions endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
            )
            and "/v1/chat/completions" in parsed_url.path
        )

    @staticmethod
    def is_openai_image_generation_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image generation endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
            )
            and "/v1/images/generations" in parsed_url.path
        )

    @staticmethod
    def is_openai_batch_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI batch endpoint (only creation)."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
            )
            and parsed_url.path
            in ["/v1/batches", "/batches"]  # Only creation endpoints
        )

    @staticmethod
    def is_openai_batch_create_route(url_route: str, request_method: str) -> bool:
        """Check if this is specifically a batch creation request (POST)."""
        return (
            OpenAIPassthroughLoggingHandler.is_openai_batch_route(url_route)
            and request_method.upper() == "POST"
        )

    @staticmethod
    def is_openai_image_editing_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image editing endpoint."""
        if not url_route:
            return False
        parsed_url = urlparse(url_route)
        return bool(
            parsed_url.hostname
            and (
                "api.openai.com" in parsed_url.hostname
                or "openai.azure.com" in parsed_url.hostname
            )
            and "/v1/images/edits" in parsed_url.path
        )

    def _get_user_from_metadata(
        self,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> Optional[str]:
        """Extract user information from passthrough logging payload."""
        request_body = passthrough_logging_payload.get("request_body")
        if request_body:
            return request_body.get("user")
        return None

    @staticmethod
    def _calculate_image_generation_cost(
        model: str,
        response_body: dict,
        request_body: dict,
    ) -> float:
        """Calculate cost for OpenAI image generation."""
        try:
            # Extract parameters from request
            n = request_body.get("n", 1)
            try:
                n = int(n)
            except Exception:
                n = 1
            size = request_body.get("size", "1024x1024")
            quality = request_body.get("quality", None)

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=quality,
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Error calculating image generation cost: {str(e)}"
            )
            return 0.0

    @staticmethod
    def _calculate_image_editing_cost(
        model: str,
        response_body: dict,
        request_body: dict,
    ) -> float:
        """Calculate cost for OpenAI image editing."""
        try:
            # Extract parameters from request
            n = request_body.get("n", 1)
            # Image edit typically uses multipart/form-data (because of files), so all fields arrive as strings (e.g., n = "1").
            try:
                n = int(n)
            except Exception:
                n = 1
            size = request_body.get("size", "1024x1024")

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=None,  # Image editing doesn't have quality parameter
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning(
                f"Error calculating image editing cost: {str(e)}"
            )
            return 0.0

    @staticmethod
    def openai_passthrough_handler(  # noqa: PLR0915
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        request_body: dict,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle OpenAI passthrough logging with cost tracking for chat completions, image generation, and image editing.
        """

        # Check if this is a supported endpoint for cost tracking
        is_chat_completions = (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route)
        )
        is_image_generation = (
            OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url_route)
        )
        is_image_editing = (
            OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route)
        )
        is_batch_create = OpenAIPassthroughLoggingHandler.is_openai_batch_create_route(
            url_route,
            kwargs.get("litellm_params", {})
            .get("proxy_server_request", {})
            .get("method", request_body.get("method", "GET")),
        )

        if not (
            is_chat_completions
            or is_image_generation
            or is_image_editing
            or is_batch_create
        ):
            # For unsupported endpoints, return None to let the system fall back to generic behavior
            return {
                "result": None,
                "kwargs": kwargs,
            }

        # Extract model from request or response
        model = request_body.get("model", response_body.get("model", ""))
        if not model and not is_batch_create:
            base_handler = OpenAIPassthroughLoggingHandler()
            return base_handler.passthrough_chat_handler(
                httpx_response=httpx_response,
                response_body=response_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )

        try:
            response_cost = 0.0
            litellm_model_response: Optional[
                Union[
                    ModelResponse, TextCompletionResponse, ImageResponse, LiteLLMBatch
                ]
            ] = None
            handler_instance = OpenAIPassthroughLoggingHandler()

            if is_chat_completions:
                # Handle chat completions with existing logic
                provider_config = handler_instance.get_provider_config(model=model)
                litellm_model_response = provider_config.transform_response(
                    raw_response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    model=model,
                    messages=request_body.get("messages", []),
                    logging_obj=logging_obj,
                    optional_params=request_body.get("optional_params", {}),
                    api_key="",
                    request_data=request_body,
                    encoding=litellm.encoding,
                    json_mode=request_body.get("response_format", {}).get("type")
                    == "json_object",
                    litellm_params={},
                )

                # Calculate cost using LiteLLM's cost calculator
                response_cost = litellm.completion_cost(
                    completion_response=litellm_model_response,
                    model=model,
                    custom_llm_provider="openai",
                )
            elif is_image_generation:
                # Handle image generation cost calculation
                response_cost = (
                    OpenAIPassthroughLoggingHandler._calculate_image_generation_cost(
                        model=model,
                        response_body=response_body,
                        request_body=request_body,
                    )
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = (
                        PassthroughCallTypes.passthrough_image_generation.value
                    )
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if litellm_model_response is not None:
                    if not hasattr(litellm_model_response, "_hidden_params"):
                        litellm_model_response._hidden_params = {}
                    litellm_model_response._hidden_params[
                        "response_cost"
                    ] = response_cost
            elif is_image_editing:
                # Handle image editing cost calculation
                response_cost = (
                    OpenAIPassthroughLoggingHandler._calculate_image_editing_cost(
                        model=model,
                        response_body=response_body,
                        request_body=request_body,
                    )
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = (
                        PassthroughCallTypes.passthrough_image_generation.value
                    )
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if litellm_model_response is not None:
                    if not hasattr(litellm_model_response, "_hidden_params"):
                        litellm_model_response._hidden_params = {}
                    litellm_model_response._hidden_params[
                        "response_cost"
                    ] = response_cost
            elif is_batch_create:
                # Handle batch creation cost calculation
                response_cost = 0.0

                # add it in base model first
                litellm_model_response = LiteLLMBatch(**response_body)

                if litellm_model_response.input_file_id:
                    extracted_model = OpenAIPassthroughLoggingHandler._extract_model_from_batch_output(
                        litellm_model_response.input_file_id
                    )
                    if extracted_model:
                        model = extracted_model
                    else:
                        model = "gpt-4o"  # Default fallback
                else:
                    model = "gpt-4o"  # Default fallback

                # Replace the batch ID with encoded unified batch ID for proper polling compatibility
                original_batch_id = litellm_model_response.id
                unified_batch_id = (
                    OpenAIPassthroughLoggingHandler._get_unified_batch_id(
                        model_id=model, batch_id=original_batch_id
                    )
                )
                litellm_model_response.id = unified_batch_id

                # Set the calculated cost in _hidden_params to prevent recalculation
                if litellm_model_response is not None:
                    if not hasattr(litellm_model_response, "_hidden_params"):
                        litellm_model_response._hidden_params = {}
                    litellm_model_response._hidden_params[
                        "response_cost"
                    ] = response_cost
                    # Add model_id to enable proper unified batch ID encoding in managed files hook
                    litellm_model_response._hidden_params["model_id"] = model

                # Trigger managed files hook for batch tracking (same as non-passthrough)
                try:
                    import asyncio

                    from litellm.proxy.proxy_server import proxy_logging_obj

                    # Create user_api_key_dict from kwargs
                    user_api_key_dict = OpenAIPassthroughLoggingHandler._create_user_api_key_dict_from_kwargs(
                        kwargs
                    )
                    if user_api_key_dict:
                        # Trigger the managed files hook asynchronously
                        asyncio.create_task(
                            proxy_logging_obj.post_call_success_hook(
                                data=request_body,
                                user_api_key_dict=user_api_key_dict,
                                response=litellm_model_response,
                            )
                        )
                except Exception:
                    pass
            # Update kwargs with cost information
            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            # Extract user information for tracking
            passthrough_logging_payload: Optional[
                PassthroughStandardLoggingPayload
            ] = kwargs.get("passthrough_logging_payload")
            if passthrough_logging_payload:
                user = handler_instance._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs.setdefault("litellm_params", {})
                    kwargs["litellm_params"].update(
                        {"proxy_server_request": {"body": {"user": user}}}
                    )

            # Create standard logging object
            if litellm_model_response is not None:
                get_standard_logging_object_payload(
                    kwargs=kwargs,
                    init_response_obj=litellm_model_response,
                    start_time=start_time,
                    end_time=end_time,
                    logging_obj=logging_obj,
                    status="success",
                )

            # Update logging object with cost information
            logging_obj.model_call_details["model"] = model
            logging_obj.model_call_details["custom_llm_provider"] = "openai"
            logging_obj.model_call_details["response_cost"] = response_cost

            endpoint_type = (
                "chat_completions"
                if is_chat_completions
                else "image_generation"
                if is_image_generation
                else "image_editing"
                if is_image_editing
                else "batch_create"
            )
            verbose_proxy_logger.debug(
                f"OpenAI passthrough cost tracking - Endpoint: {endpoint_type}, Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": cast(
                    Optional[
                        Union[ModelResponse, TextCompletionResponse, ImageResponse]
                    ],
                    litellm_model_response,
                ),
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in OpenAI passthrough cost tracking: {str(e)}"
            )
            # Fall back to base handler without cost tracking
            base_handler = OpenAIPassthroughLoggingHandler()
            return base_handler.passthrough_chat_handler(
                httpx_response=httpx_response,
                response_body=response_body,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                request_body=request_body,
                **kwargs,
            )

    def _build_complete_streaming_response(
        self,
        all_chunks: list,
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        """
        Builds complete response from raw chunks for OpenAI streaming responses.

        - Converts str chunks to generic chunks
        - Converts generic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        try:
            # OpenAI's response iterator to parse chunks
            from litellm.llms.openai.openai import OpenAIChatCompletionResponseIterator

            openai_iterator = OpenAIChatCompletionResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )

            all_openai_chunks = []
            for chunk_str in all_chunks:
                try:
                    # Parse the string chunk using the base iterator's string parser
                    from litellm.llms.base_llm.base_model_iterator import (
                        BaseModelResponseIterator,
                    )

                    # Convert string chunk to dict
                    stripped_json_chunk = (
                        BaseModelResponseIterator._string_to_dict_parser(
                            str_line=chunk_str
                        )
                    )

                    if stripped_json_chunk:
                        # Parse the chunk using OpenAI's chunk parser
                        transformed_chunk = openai_iterator.chunk_parser(
                            chunk=stripped_json_chunk
                        )
                        if transformed_chunk is not None:
                            all_openai_chunks.append(transformed_chunk)

                except (StopIteration, StopAsyncIteration, Exception) as e:
                    verbose_proxy_logger.debug(f"Error parsing streaming chunk: {e}")
                    continue

            if not all_openai_chunks:
                verbose_proxy_logger.warning(
                    "No valid chunks found in streaming response"
                )
                return None

            # Build complete response from chunks
            complete_streaming_response = litellm.stream_chunk_builder(
                chunks=all_openai_chunks
            )

            return complete_streaming_response

        except Exception:
            return None

    @staticmethod
    def _handle_logging_openai_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle logging for collected OpenAI streaming chunks with cost tracking.
        """
        try:
            # Extract model from request body
            model = request_body.get("model", "gpt-4o")

            # Build complete response from chunks using our streaming handler
            handler = OpenAIPassthroughLoggingHandler()
            handler_instance = handler
            complete_response = handler._build_complete_streaming_response(
                all_chunks=all_chunks,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
            )

            if complete_response is None:
                return {
                    "result": None,
                    "kwargs": {},
                }

            # Calculate cost using LiteLLM's cost calculator
            response_cost = litellm.completion_cost(
                completion_response=complete_response,
                model=model,
                custom_llm_provider="openai",
            )

            # Prepare kwargs for logging
            kwargs = {
                "response_cost": response_cost,
                "model": model,
                "custom_llm_provider": "openai",
            }

            # Extract user information for tracking
            passthrough_logging_payload: Optional[
                PassthroughStandardLoggingPayload
            ] = litellm_logging_obj.model_call_details.get(
                "passthrough_logging_payload"
            )
            if passthrough_logging_payload:
                user = handler_instance._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs.setdefault("litellm_params", {})
                    kwargs["litellm_params"].update(
                        {"proxy_server_request": {"body": {"user": user}}}
                    )

            # Create standard logging object
            get_standard_logging_object_payload(
                kwargs=kwargs,
                init_response_obj=complete_response,
                start_time=start_time,
                end_time=end_time,
                logging_obj=litellm_logging_obj,
                status="success",
            )

            # Update logging object with cost information
            litellm_logging_obj.model_call_details["model"] = model
            litellm_logging_obj.model_call_details["custom_llm_provider"] = "openai"
            litellm_logging_obj.model_call_details["response_cost"] = response_cost

            verbose_proxy_logger.debug(
                f"OpenAI streaming passthrough cost tracking - Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": complete_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error(
                f"Error in OpenAI streaming passthrough cost tracking: {str(e)}"
            )
            return {
                "result": None,
                "kwargs": {},
            }
