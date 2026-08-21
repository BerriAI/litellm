"""
OpenAI Passthrough Logging Handler

Handles cost tracking and logging for OpenAI passthrough endpoints, specifically /chat/completions.
"""

from datetime import datetime
from typing import Final
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
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.base_passthrough_logging_handler import (
    BasePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)
from litellm.types.utils import EmbeddingResponse, ImageResponse, LlmProviders, PassthroughCallTypes
from litellm.utils import ModelResponse, TextCompletionResponse, convert_to_model_response_object

# Hostnames that route to OpenAI-compatible APIs.
#
# `api.openai.com` is OpenAI proper. The two Azure domains below are *shared by
# every Azure Cognitive Service* (Speech, Vision, Language, ...), not just Azure
# OpenAI: `openai.azure.com` is the classic Azure OpenAI domain, while
# `cognitiveservices.azure.com` is used by newer "Azure AI Foundry" /
# Cognitive Services-hosted Azure OpenAI deployments. Because the hostname alone
# cannot tell Azure OpenAI apart from the other Cognitive Services on those
# domains, requests there must additionally carry an OpenAI-style path segment.
_OPENAI_HOSTNAMES: Final = ("api.openai.com",)
_AZURE_OPENAI_HOSTNAMES: Final = ("openai.azure.com", "cognitiveservices.azure.com")
# Path markers that identify an Azure request as Azure OpenAI rather than Speech
# / Vision / Language / ... `/openai/` is the native Azure OpenAI path prefix;
# `/v1/` is the OpenAI-v1 surface used by LiteLLM's pass-through routing. Other
# Cognitive Services use service-named prefixes and versions like `/v3.1/`,
# `/v1.0/`, so they do not collide with these markers.
_AZURE_OPENAI_PATH_MARKERS: Final = ("/openai/", "/v1/")


def _hostname_matches(hostname: str, suffixes: tuple) -> bool:
    """True if hostname equals one of `suffixes` or is a subdomain of it.

    Uses suffix matching (not a bare substring test) so look-alikes such as
    `cognitiveservices.azure.com.attacker.example` are not accepted.
    """
    return any(hostname == suffix or hostname.endswith("." + suffix) for suffix in suffixes)


def _is_openai_compatible_host(hostname: str | None) -> bool:
    """True if the hostname is OpenAI proper or one of the Azure OpenAI domains.

    Hostname-only check, kept for the route-level helpers that additionally
    require a specific OpenAI path (e.g. `/v1/chat/completions`). When only the
    hostname would otherwise gate dispatch, use `_is_openai_compatible_url` so
    non-OpenAI Azure Cognitive Services on the shared domains are excluded.
    """
    if not hostname:
        return False
    return _hostname_matches(hostname, _OPENAI_HOSTNAMES) or _hostname_matches(hostname, _AZURE_OPENAI_HOSTNAMES)


def _is_openai_compatible_url(url_route: str | None) -> bool:
    """True if the URL targets an OpenAI-compatible API surface.

    For the shared Azure Cognitive Services domains we additionally require an
    OpenAI-style path segment (`/openai/` or `/v1/`) so non-OpenAI Azure services
    (Speech, Vision, Language, ...) on the same domain are not misclassified as
    OpenAI routes.
    """
    if not url_route:
        return False
    parsed_url: Final = urlparse(url_route)
    hostname: Final = parsed_url.hostname
    if not hostname:
        return False
    if _hostname_matches(hostname, _OPENAI_HOSTNAMES):
        return True
    if _hostname_matches(hostname, _AZURE_OPENAI_HOSTNAMES):
        return any(marker in parsed_url.path for marker in _AZURE_OPENAI_PATH_MARKERS)
    return False


class OpenAIPassthroughLoggingHandler(BasePassthroughLoggingHandler):
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
        parsed_url: Final = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and "/v1/chat/completions" in parsed_url.path

    @staticmethod
    def is_openai_image_generation_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image generation endpoint."""
        if not url_route:
            return False
        parsed_url: Final = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and "/v1/images/generations" in parsed_url.path

    @staticmethod
    def is_openai_image_editing_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI image editing endpoint."""
        if not url_route:
            return False
        parsed_url: Final = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and "/v1/images/edits" in parsed_url.path

    @staticmethod
    def is_openai_responses_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI responses API endpoint."""
        if not url_route:
            return False
        parsed_url: Final = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and (
            "/v1/responses" in parsed_url.path or "/responses" in parsed_url.path
        )

    @staticmethod
    def is_openai_embeddings_route(url_route: str) -> bool:
        """Check if the URL route is an OpenAI embeddings endpoint."""
        if not url_route:
            return False
        parsed_url: Final = urlparse(url_route)
        return _is_openai_compatible_host(parsed_url.hostname) and "/v1/embeddings" in parsed_url.path

    def _get_user_from_metadata(
        self,
        passthrough_logging_payload: PassthroughStandardLoggingPayload,
    ) -> str | None:
        """Extract user information from passthrough logging payload."""
        request_body: Final = passthrough_logging_payload.get("request_body")
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
            size: Final = request_body.get("size", "1024x1024")
            quality: Final = request_body.get("quality", None)

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost: Final = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=quality,
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning("Error calculating image generation cost: %s", e)
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
            size: Final = request_body.get("size", "1024x1024")

            # Use LiteLLM's default image cost calculator
            from litellm.cost_calculator import default_image_cost_calculator

            cost: Final = default_image_cost_calculator(
                model=model,
                custom_llm_provider="openai",
                quality=None,  # Image editing doesn't have quality parameter
                n=n,
                size=size,
                optional_params=request_body,
            )

            return cost
        except Exception as e:
            verbose_proxy_logger.warning("Error calculating image editing cost: %s", e)
            return 0.0

    @staticmethod
    def _calculate_embeddings_cost(
        litellm_model_response: EmbeddingResponse,
        model: str,
        custom_llm_provider: str,
    ) -> float:
        try:
            return litellm.completion_cost(
                completion_response=litellm_model_response,
                model=model,
                custom_llm_provider=custom_llm_provider,
                call_type="aembedding",
            )
        except Exception as e:  # noqa: BLE001  # completion_cost raises bare Exception for unmapped models; cost failure must never drop the spend log
            verbose_proxy_logger.warning(
                "Error calculating embeddings cost for model %s, logging spend with cost 0: %s", model, e
            )
            return 0.0

    @staticmethod
    def _build_responses_api_response_and_cost(
        model: str,
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ) -> tuple[ResponsesAPIResponse, float]:
        """Transform a Responses API raw response into a ResponsesAPIResponse
        and compute its cost.

        The Responses API has a different on-the-wire shape from chat
        completions (`output: [...]` instead of `choices: [...]`), so the
        chat-completions `transform_response` raises KeyError 'choices' on
        a Responses payload. Use the dedicated Responses-API transformer
        (`OpenAIResponsesAPIConfig.transform_response_api_response`) here.

        Returns (litellm_model_response, response_cost) — symmetric with the
        chat-completions branch which produces the same two values inline,
        and analogous to the image branches' `_calculate_image_*_cost` helpers
        (which return cost only because the image-response object is trivial
        to build inline; the Responses payload needs a real transformer).
        """
        responses_config: Final = OpenAIResponsesAPIConfig()
        litellm_model_response: Final = responses_config.transform_response_api_response(
            model=model,
            raw_response=httpx_response,
            logging_obj=logging_obj,
        )
        response_cost: Final = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider=custom_llm_provider,
            call_type="responses",
        )
        return litellm_model_response, response_cost

    @staticmethod
    def openai_passthrough_handler(
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
        Handle OpenAI passthrough logging with cost tracking for chat completions,
        embeddings, image generation, image editing, and responses API.
        """
        is_chat_completions: Final = OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(url_route)
        is_embeddings: Final = OpenAIPassthroughLoggingHandler.is_openai_embeddings_route(url_route)
        is_image_generation: Final = OpenAIPassthroughLoggingHandler.is_openai_image_generation_route(url_route)
        is_image_editing: Final = OpenAIPassthroughLoggingHandler.is_openai_image_editing_route(url_route)
        is_responses: Final = OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)

        if not (is_chat_completions or is_embeddings or is_image_generation or is_image_editing or is_responses):
            return {
                "result": None,
                "kwargs": kwargs,
            }

        model: Final = request_body.get("model", response_body.get("model", ""))
        if not model:
            verbose_proxy_logger.warning("No model found in request or response for OpenAI passthrough cost tracking")
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
            litellm_model_response: (
                ModelResponse | TextCompletionResponse | EmbeddingResponse | ImageResponse | ResponsesAPIResponse | None
            ) = None
            handler_instance: Final = OpenAIPassthroughLoggingHandler()

            custom_llm_provider: Final = kwargs.get("custom_llm_provider", "openai")

            if is_chat_completions:
                # Handle chat completions with existing logic
                provider_config: Final = handler_instance.get_provider_config(model=model)
                # Preserve existing litellm_params to maintain metadata tags
                existing_litellm_params: Final = kwargs.get("litellm_params", {}) or {}
                litellm_model_response = provider_config.transform_response(
                    raw_response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    model=model,
                    messages=request_body.get("messages", []),
                    logging_obj=logging_obj,
                    optional_params=request_body.get("optional_params", {}),
                    api_key="",
                    request_data=request_body,
                    encoding=getattr(litellm, "encoding", None),
                    json_mode=request_body.get("response_format", {}).get("type") == "json_object",
                    litellm_params=existing_litellm_params,
                )

                # Calculate cost using LiteLLM's cost calculator
                response_cost = litellm.completion_cost(
                    completion_response=litellm_model_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
            elif is_embeddings:
                litellm_model_response = convert_to_model_response_object(
                    response_object=response_body,
                    model_response_object=EmbeddingResponse(),
                    response_type="embedding",
                )
                response_cost = OpenAIPassthroughLoggingHandler._calculate_embeddings_cost(
                    litellm_model_response=litellm_model_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_image_generation:
                # Handle image generation cost calculation
                response_cost = OpenAIPassthroughLoggingHandler._calculate_image_generation_cost(
                    model=model,
                    response_body=response_body,
                    request_body=request_body,
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = PassthroughCallTypes.passthrough_image_generation.value
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_image_editing:
                # Handle image editing cost calculation
                response_cost = OpenAIPassthroughLoggingHandler._calculate_image_editing_cost(
                    model=model,
                    response_body=response_body,
                    request_body=request_body,
                )
                # Mark call type for downstream image-aware logic/metrics
                try:
                    logging_obj.call_type = PassthroughCallTypes.passthrough_image_generation.value
                except Exception:
                    pass
                # Create a simple response object for logging
                litellm_model_response = ImageResponse(
                    data=response_body.get("data", []),
                    model=model,
                )
                # Set the calculated cost in _hidden_params to prevent recalculation
                if not hasattr(litellm_model_response, "_hidden_params"):
                    litellm_model_response._hidden_params = {}
                litellm_model_response._hidden_params["response_cost"] = response_cost
            elif is_responses:
                # Responses-API cost tracking — see
                # `_build_responses_api_response_and_cost` for why this needs
                # a dedicated transformer (the chat-completions transform
                # crashes on the Responses payload shape).
                (
                    litellm_model_response,
                    response_cost,
                ) = OpenAIPassthroughLoggingHandler._build_responses_api_response_and_cost(
                    model=model,
                    httpx_response=httpx_response,
                    logging_obj=logging_obj,
                    custom_llm_provider=custom_llm_provider,
                )

            # Update kwargs with cost information
            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            kwargs["custom_llm_provider"] = custom_llm_provider

            # Extract user information for tracking
            passthrough_logging_payload: Final[PassthroughStandardLoggingPayload | None] = kwargs.get(
                "passthrough_logging_payload"
            )
            if passthrough_logging_payload:
                user: Final = handler_instance._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs["litellm_params"].setdefault("proxy_server_request", {}).setdefault("body", {})["user"] = (
                        user
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
            logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
            logging_obj.model_call_details["response_cost"] = response_cost

            endpoint_type: Final = (
                "chat_completions"
                if is_chat_completions
                else "embeddings"
                if is_embeddings
                else "image_generation"
                if is_image_generation
                else "image_editing"
                if is_image_editing
                else "responses"
            )
            verbose_proxy_logger.debug(
                f"OpenAI passthrough cost tracking - Endpoint: {endpoint_type}, Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error("Error in OpenAI passthrough cost tracking: %s", e)
            if not is_chat_completions:
                unbilled_result: Final[PassThroughEndpointLoggingTypedDict] = {
                    "result": None,
                    "kwargs": kwargs,
                }
                return unbilled_result
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
        all_chunks: list[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
    ) -> ModelResponse | TextCompletionResponse | None:
        """
        Builds complete response from raw chunks for OpenAI streaming responses.

        - Converts str chunks to generic chunks
        - Converts generic chunks to litellm chunks (OpenAI format)
        - Builds complete response from litellm chunks
        """
        try:
            # OpenAI's response iterator to parse chunks
            from litellm.llms.openai.openai import OpenAIChatCompletionResponseIterator

            openai_iterator: Final = OpenAIChatCompletionResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )

            all_openai_chunks: Final = []
            for chunk_str in all_chunks:
                try:
                    # Parse the string chunk using the base iterator's string parser
                    from litellm.llms.base_llm.base_model_iterator import (
                        BaseModelResponseIterator,
                    )

                    # Convert string chunk to dict
                    stripped_json_chunk = BaseModelResponseIterator._string_to_dict_parser(str_line=chunk_str)

                    if stripped_json_chunk:
                        # Parse the chunk using OpenAI's chunk parser
                        transformed_chunk = openai_iterator.chunk_parser(chunk=stripped_json_chunk)
                        if transformed_chunk is not None:
                            all_openai_chunks.append(transformed_chunk)

                except (StopIteration, StopAsyncIteration, Exception) as e:
                    verbose_proxy_logger.debug("Error parsing streaming chunk: %s", e)
                    continue

            if not all_openai_chunks:
                verbose_proxy_logger.warning("No valid chunks found in streaming response")
                return None

            # Build complete response from chunks
            complete_streaming_response: Final = litellm.stream_chunk_builder(chunks=all_openai_chunks)

            return complete_streaming_response

        except Exception as e:
            verbose_proxy_logger.error("Error building complete streaming response: %s", e)
            return None

    @staticmethod
    def _handle_logging_openai_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: list[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Handle logging for collected OpenAI streaming chunks with cost tracking.
        """
        try:
            # Extract model from request body
            model: Final = request_body.get("model", "gpt-4o")

            is_responses: Final = OpenAIPassthroughLoggingHandler.is_openai_responses_route(url_route)

            # Build complete response from chunks using our streaming handler
            handler: Final = OpenAIPassthroughLoggingHandler()
            handler_instance: Final = handler
            complete_response: Final = (
                OpenAIResponsesAPIConfig.parse_terminal_response_from_stream_chunks(all_chunks=all_chunks)
                if is_responses
                else handler._build_complete_streaming_response(
                    all_chunks=all_chunks,
                    litellm_logging_obj=litellm_logging_obj,
                    model=model,
                )
            )

            if complete_response is None:
                verbose_proxy_logger.warning("Failed to build complete response from OpenAI streaming chunks")
                return {
                    "result": None,
                    "kwargs": {},
                }

            custom_llm_provider: Final = litellm_logging_obj.model_call_details.get("custom_llm_provider", "openai")
            # Calculate cost using LiteLLM's cost calculator
            response_cost: Final = (
                litellm.completion_cost(
                    completion_response=complete_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                    call_type="responses",
                )
                if is_responses
                else litellm.completion_cost(
                    completion_response=complete_response,
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
            )

            # Preserve existing litellm_params to maintain metadata tags
            existing_litellm_params: Final = litellm_logging_obj.model_call_details.get("litellm_params", {}) or {}

            # Prepare kwargs for logging
            kwargs: Final = {
                "response_cost": response_cost,
                "model": model,
                "custom_llm_provider": custom_llm_provider,
                "call_type": litellm_logging_obj.call_type,
                "messages": litellm_logging_obj.model_call_details.get("messages"),
                "litellm_params": existing_litellm_params.copy(),
            }

            # Extract user information for tracking
            passthrough_logging_payload: Final[PassthroughStandardLoggingPayload | None] = (
                litellm_logging_obj.model_call_details.get("passthrough_logging_payload")
            )
            if passthrough_logging_payload:
                user: Final = handler_instance._get_user_from_metadata(
                    passthrough_logging_payload=passthrough_logging_payload,
                )
                if user:
                    kwargs["litellm_params"].setdefault("proxy_server_request", {}).setdefault("body", {})["user"] = (
                        user
                    )

            # Attach the payload to kwargs so the success handler adopts it;
            # its later rebuild runs on a copy whose Responses usage was
            # coerced to chat shape and serializes as total_tokens only,
            # zeroing the prompt/completion split in spend logs.
            standard_logging_object: Final = get_standard_logging_object_payload(
                kwargs=kwargs,
                init_response_obj=complete_response,
                start_time=start_time,
                end_time=end_time,
                logging_obj=litellm_logging_obj,
                status="success",
            )
            if standard_logging_object is not None:
                kwargs["standard_logging_object"] = standard_logging_object

            # Update logging object with cost information
            litellm_logging_obj.model_call_details["model"] = model
            litellm_logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
            litellm_logging_obj.model_call_details["response_cost"] = response_cost

            verbose_proxy_logger.debug(
                f"OpenAI streaming passthrough cost tracking - Model: {model}, Cost: ${response_cost:.6f}"
            )

            return {
                "result": complete_response,
                "kwargs": kwargs,
            }

        except Exception as e:
            verbose_proxy_logger.error("Error in OpenAI streaming passthrough cost tracking: %s", e)
            return {
                "result": None,
                "kwargs": {},
            }
