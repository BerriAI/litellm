import re
from datetime import datetime

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.vertex_ai_and_google_ai_studio.gemini.vertex_and_google_ai_studio_gemini import (
    VertexLLM,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


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
            await logging_obj.async_success_handler(
                result="",
                start_time=start_time,
                end_time=end_time,
                cache_hit=False,
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
            logging_obj.model = litellm_model_response.model
            logging_obj.model_call_details["model"] = logging_obj.model

            await logging_obj.async_success_handler(
                result=litellm_model_response,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
            )
