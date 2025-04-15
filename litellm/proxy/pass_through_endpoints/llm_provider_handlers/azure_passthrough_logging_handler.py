import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional
from urllib.parse import urlparse, urlunparse
import asyncio
import httpx

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.thread_pool_executor import executor
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.llms.openai import Run
import time

if TYPE_CHECKING:
    from ..success_handler import PassThroughEndpointLogging
    from ..types import EndpointType
else:
    PassThroughEndpointLogging = Any
    EndpointType = Any


class AzurePassthroughLoggingHandler:
    def __init__(self):

        self.polling_interval = 5
        self.max_polling_attempts = 5

    def azure_passthrough_handler(
        self,
        httpx_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        extra_query_params: Optional[str],
        **kwargs,
    ):
        
        if extra_query_params:
            if "?" in str(url_route):
                url_route = str(url_route) + "&" + extra_query_params
            else:
                url_route = str(url_route) + "?" + extra_query_params
        
        _, _, path, _, _, _ = urlparse(url_route)

        if re.search(r"/threads/thread_.+/runs", path): # handle various requests here
            executor.submit(
                self._handle_azure_passthrough_logging,
                    httpx_response,
                    logging_obj,
                    url_route,
                    result,
                    start_time,
                    end_time,
                    cache_hit,
                    **kwargs,
            )

    @staticmethod
    def _should_log_request(request_method: str) -> bool:
        return request_method == "POST"
    
    def _handle_azure_passthrough_logging(
        self,
        run_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        url_route: str,
        result: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
        **kwargs,
    ):
        from ..pass_through_endpoints import pass_through_endpoint_logging
        populated_run = self._poll_unfinished_run(
            run_response, url_route
        )
        verbose_proxy_logger.info("The populated run is %s", populated_run)
        kwargs = AzurePassthroughLoggingHandler._create_azure_response_logging_payload_for_assistants(
            litellm_run=populated_run,
            kwargs=kwargs,
            start_time=start_time,
            end_time=end_time,
            logging_obj=logging_obj,
            custom_llm_provider="azure",
        )
        verbose_proxy_logger.info("The cost is %.20f", kwargs['response_cost'])
        for k,v in kwargs.items():
            verbose_proxy_logger.info("The k is %s, v is %s", k, v)

        asyncio.run(
            pass_through_endpoint_logging._handle_logging(
                logging_obj=logging_obj,
                standard_logging_response_object=self._get_response_to_log(
                    populated_run
                ),
                result=result,
                start_time=start_time,
                end_time=end_time,
                cache_hit=cache_hit,
                **kwargs,
            )
        )

    def _get_response_to_log( 
            self,
            run_response: Optional[Run] 
    ) -> dict:
        if run_response is None:
            return {}
        return dict(run_response)

    @staticmethod
    def _get_retrieve_url(
        litellm_run: Run,
        url_route: str,
    ) -> str: 
        thread_id = litellm_run.thread_id
        if thread_id is None:
            raise ValueError(
                "Thread ID is required to log the cost of the Run"
            )
        run_id = litellm_run.id
        if run_id is None:
            raise ValueError(
                "Run ID is required to log the cost of the Run"
            )
        scheme, netloc, _, params, query_params, fragment = urlparse(url_route)
        retrieve_thread_url = urlunparse((scheme, netloc, f"/openai/threads/{thread_id}/runs/{run_id}", params, query_params, fragment))
        return retrieve_thread_url
    
    def _get_populated_run(
        self,
        litellm_run: Run,
        url_route: str,
    ) -> dict: 
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            passthrough_endpoint_router,
        )
        retrieve_url = AzurePassthroughLoggingHandler._get_retrieve_url(litellm_run, url_route)
        azure_api_key = passthrough_endpoint_router.get_credentials(
            custom_llm_provider=litellm.LlmProviders.AZURE.value,
            region_name=None,
        )
        try:
            if azure_api_key is None:
                raise Exception(
                    "Required 'AZURE_API_KEY' in environment to make pass-through calls to Azure."
                )
            headers = {
                "Authorization": f"Bearer {azure_api_key}",
                "Content-Type": "application/json",
            }

            response = httpx.get(retrieve_url, headers=headers)
            response.raise_for_status()

            return response.json()
        except Exception as e:
            verbose_proxy_logger.exception(
                f"[Non blocking logging error] Error getting Azure OpenAI Run: {str(e)}"
            )
            return litellm_run.model_dump()
            
# Called only when a run is created and not on other routes.
    def _poll_unfinished_run(
        self,
        litellm_run_response: httpx.Response,
        url_route: str,
    ) -> Optional[Run]:  
        response = litellm_run_response.json()
        run_obj = Run(**response)
        for attempt in range(1, self.max_polling_attempts+1):
            try:
                response = self._get_populated_run(run_obj, url_route)

                if response is not None and response.get("status") in ["completed", "failed", "cancelled"]:
                    return Run(**response)

                if attempt == self.max_polling_attempts:
                    raise Exception("Max Retries reached for Run to finish")
                
                time.sleep(self.polling_interval)
            except Exception:
                return Run(**litellm_run_response.json())
        return None

    @staticmethod
    def _create_azure_response_logging_payload_for_assistants(
        litellm_run: Optional[Run],
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
        if litellm_run is None:
            model = ''
        else: 
            model = litellm_run.model

        response_cost = litellm.completion_cost(
            completion_response=litellm_run,
            custom_llm_provider="azure",
            model=model,
        )
        kwargs["response_cost"] = response_cost
        kwargs["model"] = model

        # pretty print standard logging object
        verbose_proxy_logger.debug("kwargs= %s", json.dumps(kwargs, indent=4))

        # set litellm_call_id to logging response object
        if litellm_run is not None:
            litellm_run.id = logging_obj.litellm_call_id
            logging_obj.model = model
            logging_obj.model_call_details["model"] = logging_obj.model
            logging_obj.model_call_details["custom_llm_provider"] = custom_llm_provider
            logging_obj.model_call_details["response_cost"] = response_cost

        return kwargs