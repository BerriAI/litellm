import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from ..types import EndpointType
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class VertexPassthroughLoggingHandler:
    @staticmethod
    async def vertex_passthrough_handler(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        if "generateContent" in url_route:
            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)

            instance_of_vertex_llm = litellm.VertexGeminiConfig()
            litellm_model_response: litellm.ModelResponse = (
                instance_of_vertex_llm._transform_response(
                    model=model,
                    messages=[
                        {"role": "user", "content": "no-message-pass-through-endpoint"}
                    ],
                    response=httpx_response,
                    model_response=litellm.ModelResponse(),
                    logging_obj=logging_obj,
                    optional_params={},
                    litellm_params={},
                    api_key="",
                    data={},
                    print_verbose=litellm.print_verbose,
                    encoding=None,
                )
            )
            logging_obj.model = litellm_model_response.model or model
            logging_obj.model_call_details["model"] = logging_obj.model

            await logging_obj.async_success_handler(
                result=litellm_model_response,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
        elif "predict" in url_route:
            from litellm.llms.vertex_ai_and_google_ai_studio.image_generation.image_generation_handler import (
                VertexImageGeneration,
            )
            from litellm.types.utils import PassthroughCallTypes

            vertex_image_generation_class = VertexImageGeneration()

            model = VertexPassthroughLoggingHandler.extract_model_from_url(url_route)
            _json_response = httpx_response.json()

            litellm_prediction_response: Union[
                litellm.ModelResponse, litellm.EmbeddingResponse, litellm.ImageResponse
            ] = litellm.ModelResponse()
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

            await logging_obj.async_success_handler(
                result=litellm_prediction_response,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )

    @staticmethod
    def extract_model_from_url(url: str) -> str:
        pattern = r"/models/([^:]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return "unknown"
