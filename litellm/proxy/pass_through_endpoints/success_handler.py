import json
import re
import threading
from datetime import datetime
from typing import Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
    VertexLLM,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.utils import StandardPassThroughResponseObject


class PassThroughEndpointLogging:
    def __init__(self):
        self.TRACKED_VERTEX_ROUTES = [
            "generateContent",
            "streamGenerateContent",
            "predict",
        ]

    async def pass_through_async_success_handler(
        self,
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        if self.is_vertex_route(url_route):
            await self.vertex_passthrough_handler(
                httpx_response=httpx_response,
                logging_obj=logging_obj,
                url_route=url_route,
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
        else:
            standard_logging_response_object = StandardPassThroughResponseObject(
                response=httpx_response.text
            )
            threading.Thread(
                target=logging_obj.success_handler,
                args=(
                    standard_logging_response_object,
                    start_time,
                    end_time,
                    cache_hit,
                ),
            ).start()
            await logging_obj.async_success_handler(
                result=(
                    json.dumps(result)
                    if isinstance(result, dict)
                    else standard_logging_response_object
                ),
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
                **kwargs,
            )

    def is_vertex_route(self, url_route: str):
        for route in self.TRACKED_VERTEX_ROUTES:
            if route in url_route:
                return True
        return False

    def extract_model_from_url(self, url: str) -> str:
        pattern = r"/models/([^:]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return "unknown"

    async def vertex_passthrough_handler(
        self,
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
            model = self.extract_model_from_url(url_route)

            instance_of_vertex_llm = VertexLLM()
            litellm_model_response: litellm.ModelResponse = (
                instance_of_vertex_llm._process_response(
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

            model = self.extract_model_from_url(url_route)
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
