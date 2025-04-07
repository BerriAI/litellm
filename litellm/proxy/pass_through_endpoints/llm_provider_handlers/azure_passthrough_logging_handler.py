import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from urllib.parse import urlparse, urlunparse
import asyncio
import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    ModelResponseIterator as VertexModelResponseIterator,
)
from litellm.proxy._types import PassThroughEndpointLoggingTypedDict
from litellm.types.utils import (
    ModelResponse, EmbeddingResponse, ImageResponse,
    TextCompletionResponse,
)

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from ..types import EndpointType
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class AzurePassthroughLoggingHandler:
    @staticmethod
    async def azure_passthrough_handler(
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        extra_query_params: Optional[str],
        **kwargs,
    ) -> PassThroughEndpointLoggingTypedDict:
        if extra_query_params:
            if "?" in str(url_route):
                url_route = str(url_route) + "&" + extra_query_params
            else:
                url_route = str(url_route) + "?" + extra_query_params
        _, _, path, _, _, _ = urlparse(url_route)
        if re.search(r"/threads/thread_.+/runs", path): # handle various requests here
            litellm_run_response = httpx_response.json()
            model = litellm_run_response['model']
            usage_litellm_model_response = await AzurePassthroughLoggingHandler._poll_unfinished_run(
                httpx_response, url_route
            )
            kwargs = AzurePassthroughLoggingHandler._create_azure_response_logging_payload_for_assistants(
                litellm_model_response=usage_litellm_model_response,
                model=model,
                kwargs=kwargs,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
                custom_llm_provider="azure",
            )
            verbose_proxy_logger.info("The cost is %.20f", kwargs['response_cost'])
            return {
                "result": litellm_run_response,
                "kwargs": kwargs,
            }
        
        else:
            return {
                "result": None,
                "kwargs": kwargs,
            }

    @staticmethod
    def extract_model_from_url(url: str) -> str:
        pattern = r"/models/([^:]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return "unknown"


# Called only when a run is created and not on other routes.
    @staticmethod
    async def _poll_unfinished_run(
        litellm_run_response: httpx.Response,
        url_route: str,
        interval: int = 5,
        max_retries: int = 5,
        max_delay: int = 60,
    ):  
        response = litellm_run_response.json()
        async_client_obj = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.PassThroughEndpoint,
            params={"timeout": 600},
        )
        async_client = async_client_obj.client
        thread_id = response['thread_id']
        run_id = response['id']

        request = litellm_run_response.request
        scheme, netloc, _, params, query_params, fragment = urlparse(url_route)
        retrieve_thread_url = urlunparse((scheme, netloc, f"/openai/threads/{thread_id}/runs/{run_id}", params, query_params, fragment))
        # get proxy URL to hit using AzureOpenAI client
        # Now, can use pass_through_request fnc logic here, but how should you build the URL to retrieve the Run, 
        # Just add all the query_params here and get the thread_id and run_id from response to?
        # After this just wait for this polling function to finish.
        verbose_proxy_logger.info("The Retrieve URL is: %s", retrieve_thread_url)
        request.headers['content-length'] = "0" # convert from the POST to GET request
        new_request = async_client.build_request(
            method='GET',
            url=retrieve_thread_url,
            headers=request.headers,
            params=query_params,
        )
        for attempt in range(1, max_retries+1):
            response = await async_client.send(new_request)
            try:
                data = response.json()

                if data.get("status") in ["completed", "failed", "cancelled"]:
                    return data

                if attempt == max_retries:
                    raise Exception("Max Retries reached for Run to finish")
                
                interval = min(interval * (2 ** (attempt - 1)), max_delay)
                await asyncio.sleep(interval)
            except Exception:
                return response

    @staticmethod
    def _create_azure_response_logging_payload_for_assistants(
        litellm_model_response: dict,
        model: str,
        kwargs: dict,
        start_time: datetime,
        end_time: datetime,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str,
    ):
        """
        Create the standard logging object for Azure passthrough Assistants

        """
        # Here I should wait for Run to be completed before calculating cost - but this wants to calculate cost and send it forward to be logged asynchronously.

        response_cost = litellm.completion_cost(
            completion_response=litellm_model_response,
            custom_llm_provider="azure",
            model=model,
        )
        kwargs["response_cost"] = response_cost
        kwargs["model"] = model

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", json.dumps(kwargs, indent=4))

        # set litellm_call_id to logging response object
        litellm_model_response["id"] = logging_obj.litellm_call_id
        logging_obj.model = litellm_model_response["model"] or model
        logging_obj.model_call_details["model"] = logging_obj.model
        logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
        return kwargs