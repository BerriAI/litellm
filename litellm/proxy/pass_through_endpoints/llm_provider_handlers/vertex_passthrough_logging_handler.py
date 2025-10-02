import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as VertexModelResponseIterator,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from ..types import EndpointType
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class VertexPassthroughLoggingHandler:
    @staticmethod
    def vertex_passthrough_handler(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        if "generateContent" in url_route:
            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)

            instance_of_vertex_llm = litellm.VertexGeminiConfig()
            litellm_model_response: ModelResponse = (
                instance_of_vertex_llm.transform_response(
                    model=model,
                    messages=[
                        {"role": "user", "content": "no-message-pass-through-endpoint"}
                    ],
                    raw_response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    logging_obj=logging_obj,
                    optional_params={},
                    litellm_params={},
                    api_key="",
                    request_data={},
                    encoding=litellm.encoding,
                )
            )
            kwargs = VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider=VertexPassthroughLoggingHandler._get_custom_llm_provider_from_url(
                    url_route
                ),
            )

            return {
                "result": litellm_model_response,
                "kwargs": kwargs,
            }

        elif "predict" in url_route:
            from litellm.llms.vertex_ai.image_generation.image_generation_handler import (
                VertexImageGeneration,
            )
            from litellm.llms.vertex_ai.multimodal_embeddings.transformation import (
                VertexAIMultimodalEmbeddingConfig,
            )
            from litellm.types.utils import PassthroughCallTypes

            vertex_image_generation_class = VertexImageGeneration()

            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)

            _json_response = httpx_response.json()

            litellm_prediction_response: Union[
                ModelResponse, EmbeddingResponse, ImageResponse
            ] = ModelResponse()
            if vertex_image_generation_class.is_image_generation_response(
                _json_response
            ):
                litellm_prediction_response = (
                    vertex_image_generation_class.process_image_generation_response(
                        _json_response,
                        model_response=litellm.ImageResponse(),
                        model=model,
                    )
                )

                logging_obj.call_type = (
                    PassthroughCallTypes.passthrough_image_generation.value
                )
            elif VertexPassthroughLoggingHandler._is_multimodal_embedding_response(
                json_response=_json_response,
            ):
                # Use multimodal embedding transformation
                vertex_multimodal_config = VertexAIMultimodalEmbeddingConfig()
                litellm_prediction_response = (
                    vertex_multimodal_config.transform_embedding_response(
                        model=model,
                        raw_response=httpx_response,
                        model_response=litellm.EmbeddingResponse(),
                        logging_obj=logging_obj,
                        api_key="",
                        request_data={},
                        optional_params={},
                        litellm_params={},
                    )
                )
            else:
                litellm_prediction_response = litellm.vertexAITextEmbeddingConfig.transform_vertex_response_to_openai(
                    response=_json_response,
                    model=model,
                    model_response=litellm.EmbeddingResponse(),
                )
            if isinstance(litellm_prediction_response, litellm.EmbeddingResponse):
                litellm_prediction_response.model = model

            logging_obj.model = model
            logging_obj.model_call_details["model"] = logging_obj.model
            response_cost = litellm.completion_cost(
                completion_response=litellm_prediction_response,
                model=model,
                custom_llm_provider="vertex_ai",
            )

            kwargs["response_cost"] = response_cost
            kwargs["model"] = model
            logging_obj.model_call_details["response_cost"] = response_cost

            return {
                "result": litellm_prediction_response,
                "kwargs": kwargs,
            }
        elif "rawPredict" in url_route or "streamRawPredict" in url_route:
            from litellm.llms.vertex_ai.vertex_ai_partner_models import (
                get_vertex_ai_partner_model_config,
            )

            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)
            vertex_publisher_or_api_spec = VertexPassthroughLoggingHandler._get_vertex_publisher_or_api_spec_from_url(
                url_route
            )

            _json_response = httpx_response.json()

            litellm_prediction_response = ModelResponse()

            if vertex_publisher_or_api_spec is not None:
                vertex_ai_partner_model_config = get_vertex_ai_partner_model_config(
                    model=model,
                    vertex_publisher_or_api_spec=vertex_publisher_or_api_spec,
                )
                litellm_prediction_response = (
                    vertex_ai_partner_model_config.transform_response(
                        model=model,
                        raw_response=httpx_response,
                        model_response=litellm_prediction_response,
                        logging_obj=logging_obj,
                        request_data={},
                        encoding=litellm.encoding,
                        optional_params={},
                        litellm_params={},
                        api_key="",
                        messages=[
                            {
                                "role": "user",
                                "content": "no-message-pass-through-endpoint",
                            }
                        ],
                    )
                )

            kwargs = VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
                litellm_model_response=litellm_prediction_response,
                model="vertex_ai/" + model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="vertex_ai",
            )

            return {
                "result": litellm_prediction_response,
                "kwargs": kwargs,
            }
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    @staticmethod
    def _handle_logging_vertex_collected_chunks(
        litellm_logging_obj: LiteLLMLoggingObj,
        passthrough_success_handler_obj: PassThroughEndpointLogging,
        url_route: str,
        request_body: dict,
        endpoint_type: EndpointType,
        start_time: datetime,
        all_chunks: List[str],
        model: Optional[str],
        end_time: datetime,
    ) -> PassThroughEndpointLoggingTypedDict:
        """
        Takes raw chunks from Vertex passthrough endpoint and logs them in litellm callbacks

        - Builds complete response from chunks
        - Creates standard logging object
        - Logs in litellm callbacks
        """
        kwargs: Dict[str, Any] = {}
        model = model or VertexPassthroughLoggingHandler.extract_model_from_url(
            url_route
        )
        complete_streaming_response = (
            VertexPassthroughLoggingHandler._build_complete_streaming_response(
                all_chunks=all_chunks,
                litellm_logging_obj=litellm_logging_obj,
                model=model,
                url_route=url_route,
            )
        )

        if complete_streaming_response is None:
            verbose_proxy_logger.error(
                "Unable to build complete streaming response for Vertex passthrough endpoint, not logging..."
            )
            return {
                "result": None,
                "kwargs": kwargs,
            }

        kwargs = VertexPassthroughLoggingHandler._create_vertex_response_logging_payload_for_generate_content(
            litellm_model_response=complete_streaming_response,
            model=model,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=litellm_logging_obj,
            custom_llm_provider=VertexPassthroughLoggingHandler._get_custom_llm_provider_from_url(
                url_route
            ),
        )

        return {
            "result": complete_streaming_response,
            "kwargs": kwargs,
        }

    @staticmethod
    def _build_complete_streaming_response(
        all_chunks: List[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        url_route: str,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        parsed_chunks = []
        if "generateContent" in url_route or "streamGenerateContent" in url_route:
            vertex_iterator: Any = VertexModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
                logging_obj=litellm_logging_obj,
            )
            chunk_parsing_logic: Any = vertex_iterator._common_chunk_parsing_logic
            parsed_chunks = [chunk_parsing_logic(chunk) for chunk in all_chunks]
        elif "rawPredict" in url_route or "streamRawPredict" in url_route:
            from litellm.llms.anthropic.chat.handler import ModelResponseIterator
            from litellm.llms.base_llm.base_model_iterator import (
                BaseModelResponseIterator,
            )

            vertex_iterator = ModelResponseIterator(
                streaming_response=None,
                sync_stream=False,
            )
            chunk_parsing_logic = vertex_iterator.chunk_parser
            for chunk in all_chunks:
                dict_chunk = BaseModelResponseIterator._string_to_dict_parser(chunk)
                if dict_chunk is None:
                    continue
                parsed_chunks.append(chunk_parsing_logic(dict_chunk))
        else:
            return None
        if len(parsed_chunks) == 0:
            return None
        all_openai_chunks = []
        for parsed_chunk in parsed_chunks:
            if parsed_chunk is None:
                continue
            all_openai_chunks.append(parsed_chunk)

        complete_streaming_response = litellm.stream_chunk_builder(
            chunks=all_openai_chunks
        )

        return complete_streaming_response

    @staticmethod
    def extract_model_from_url(url: str) -> str:
        pattern = r"/models/([^:]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return "unknown"

    @staticmethod
    def _get_vertex_publisher_or_api_spec_from_url(url: str) -> Optional[str]:
        # Check for specific Vertex AI partner publishers
        if "/publishers/mistralai/" in url:
            return "mistralai"
        elif "/publishers/anthropic/" in url:
            return "anthropic"
        elif "/publishers/ai21/" in url:
            return "ai21"
        elif "/endpoints/openapi/" in url:
            return "openapi"
        return None

    @staticmethod
    def _get_custom_llm_provider_from_url(url: str) -> str:
        parsed_url = urlparse(url)
        if parsed_url.hostname and parsed_url.hostname.endswith(
            "generativelanguage.googleapis.com"
        ):
            return litellm.LlmProviders.GEMINI.value
        return litellm.LlmProviders.VERTEX_AI.value

    @staticmethod
    def _is_multimodal_embedding_response(json_response: dict) -> bool:
        """
        Detect if the response is from a multimodal embedding request.

        Check if the response contains multimodal embedding fields:
            - Docs: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/multimodal-embeddings-api#response-body


        Args:
            json_response: The JSON response from Vertex AI

        Returns:
            bool: True if this is a multimodal embedding response
        """
        # Check if response contains multimodal embedding fields
        if "predictions" in json_response:
            predictions = json_response["predictions"]
            for prediction in predictions:
                if isinstance(prediction, dict):
                    # Check for multimodal embedding response fields
                    if any(
                        key in prediction
                        for key in [
                            "textEmbedding",
                            "imageEmbedding",
                            "videoEmbeddings",
                        ]
                    ):
                        return True

        return False

    @staticmethod
    def _create_vertex_response_logging_payload_for_generate_content(
        litellm_model_response: Union[ModelResponse, TextCompletionResponse],
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ):
        """
        Create the standard logging object for Vertex passthrough generateContent (streaming and non-streaming)

        """

        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider="vertex_ai",
        )

        kwargs["response_cost"] = response_cost
        kwargs["model"] = model

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", json.dumps(kwargs, indent=4))

        # set litellm_call_id to logging response object
        litellm_model_response.id = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response.model or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        return kwargs
