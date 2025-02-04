import json
from datetime import datetime
from typing import Optional

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.types.utils import StandardPassThroughResponseObject
from litellm.utils import executor as thread_pool_executor

from .llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from .llm_provider_handlers.cohere_passthrough_logging_handler import (
    CoherePassthroughLoggingHandler,
)
from .llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)

cohere_passthrough_logging_handler = CoherePassthroughLoggingHandler()


class PassThroughEndpointLogging:
    def __init__(self):
        self.TRACKED_VERTEX_ROUTES = [
            "generateContent",
            "streamGenerateContent",
            "predict",
        ]

        # Anthropic
        self.TRACKED_ANTHROPIC_ROUTES = ["/messages"]

        # Cohere
        self.TRACKED_COHERE_ROUTES = ["/v1/chat", "/v2/chat"]

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
        request_body: dict,
        **kwargs,
    ):
        standard_logging_response_object: Optional[
            PassThroughEndpointLoggingResultValues
        ] = None
        if self.is_vertex_route(url_route):
            vertex_passthrough_logging_handler_result = (
                VertexPassthroughLoggingHandler.vertex_passthrough_handler(
                    httpx_response=httpx_response,
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                vertex_passthrough_logging_handler_result["result"]
            )
            kwargs = vertex_passthrough_logging_handler_result["kwargs"]
        elif self.is_anthropic_route(url_route):
            anthropic_passthrough_logging_handler_result = (
                AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
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
            )

            standard_logging_response_object = (
                anthropic_passthrough_logging_handler_result["result"]
            )
            kwargs = anthropic_passthrough_logging_handler_result["kwargs"]
        elif self.is_cohere_route(url_route):
            cohere_passthrough_logging_handler_result = (
                cohere_passthrough_logging_handler.passthrough_chat_handler(
                    httpx_response=httpx_response,
                    response_body=response_body or {},
                    logging_obj=logging_obj,
                    url_route=url_route,
                    result=result,
                    start_time=start_time,
                    end_time=end_time,
                    cache_hit=cache_hit,
                    request_body=request_body,
                    **kwargs,
                )
            )
            standard_logging_response_object = (
                cohere_passthrough_logging_handler_result["result"]
            )
            kwargs = cohere_passthrough_logging_handler_result["kwargs"]
        if standard_logging_response_object is None:
            standard_logging_response_object = StandardPassThroughResponseObject(
                response=httpx_response.text
            )
        thread_pool_executor.submit(
            logging_obj.success_handler,
            standard_logging_response_object,  # Positional argument 1
            start_time,  # Positional argument 2
            end_time,  # Positional argument 3
            cache_hit,  # Positional argument 4
            **kwargs,  # Unpacked keyword arguments
        )

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

    def is_cohere_route(self, url_route: str):
        for route in self.TRACKED_COHERE_ROUTES:
            if route in url_route:
                return True
        return False
