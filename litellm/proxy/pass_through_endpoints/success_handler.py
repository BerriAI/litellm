import json
import re
import threading
from datetime import datetime
from typing import Optional, Union

import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
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

        # Anthropic
        self.TRACKED_ANTHROPIC_ROUTES = ["/messages"]

    async def pass_through_async_success_handler(
        self,
        httpx_response: httpx.Response,
        response_body: Optional[dict],
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
        elif self.is_anthropic_route(url_route):
            await self.anthropic_passthrough_handler(
                httpx_response=httpx_response,
                response_body=response_body or {},
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

    def is_anthropic_route(self, url_route: str):
        for route in self.TRACKED_ANTHROPIC_ROUTES:
            if route in url_route:
                return True
        return False

    def extract_model_from_url(self, url: str) -> str:
        pattern = r"/models/([^:]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return "unknown"

    async def anthropic_passthrough_handler(
        self,
        httpx_response: httpx.Response,
        response_body: dict,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        """
        Transforms Anthropic response to OpenAI response, generates a standard logging object so downstream logging can be handled
        """
        model = response_body.get("model", "")
        litellm_model_response: litellm.ModelResponse = (
            AnthropicConfig._process_response(
                response=httpx_response,
                model_response=litellm.ModelResponse(),
                model=model,
                stream=False,
                messages=[],
                logging_obj=logging_obj,
                optional_params={},
                api_key="",
                data={},
                print_verbose=litellm.print_verbose,
                encoding=None,
                json_mode=False,
            )
        )

        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            model=model,
        )
        kwargs["response_cost"] = response_cost
        kwargs["model"] = model

        # Make standard logging object for Vertex AI
        standard_logging_object = get_standard_logging_object_payload(
            kwargs=kwargs,
            init_response_obj=litellm_model_response,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
            status="success",
        )

        # pretty print standard logging object
        verbose_proxy_logger.debug(
            "standard_logging_object= %s", json.dumps(standard_logging_object, indent=4)
        )
        kwargs["standard_logging_object"] = standard_logging_object

        await logging_obj.async_success_handler(
            result=litellm_model_response,
            start_time=start_time,
            end_time=end_time,
            cache_hit=cache_hit,
            **kwargs,
        )

        pass

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
