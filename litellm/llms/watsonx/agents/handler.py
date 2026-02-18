"""
Handler for IBM watsonx.ai Orchestrate Agent API.

Model format: watsonx_agent/<agent_id>

API Reference: https://developer.watson-orchestrate.ibm.com/apis/orchestrate-agent/chat-with-agents
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.types.utils import ModelResponse

from .transformation import IBMWatsonXAgentConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any
    HTTPHandler = Any
    AsyncHTTPHandler = Any


class WatsonXAgentHandler:
    """
    Handler for watsonx agent completions.

    Executes agent API calls to watsonx Orchestrate.
    """

    def __init__(self):
        self.config = IBMWatsonXAgentConfig()

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        headers: Optional[dict] = None,
    ) -> ModelResponse:
        """
        Execute synchronous completion using watsonx agent API.

        Args:
            model: Model identifier (format: watsonx_agent/AGENT_ID)
            messages: List of messages
            api_base: API base URL
            api_key: API key
            model_response: ModelResponse object to populate
            logging_obj: Logging object
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            timeout: Request timeout
            headers: Request headers

        Returns:
            ModelResponse object
        """
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client = _get_httpx_client(params={"ssl_verify": litellm_params.get("ssl_verify", None)})

        # Validate environment and update headers
        headers = self.config.validate_environment(
            headers=headers or {},
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        # Get complete URL
        url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=optional_params.get("stream", False),
        )

        # Transform request
        request_data = self.config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Add thread_id header if provided
        thread_id = optional_params.get("thread_id")
        if thread_id:
            headers["X-IBM-THREAD-ID"] = str(thread_id)

        verbose_logger.debug(
            f"Watsonx Agent API request - URL: {url}, Headers: {headers}, Data: {request_data}"
        )

        # Make synchronous request
        response = client.post(
            url=url,
            json=request_data,
            headers=headers,
            timeout=timeout,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_message = e.response.text
            status_code = e.response.status_code
            raise self.config.get_error_class(
                error_message=error_message,
                status_code=status_code,
                headers=e.response.headers,
            )

        # Transform response
        return self.config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=None,
            api_key=api_key,
        )

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        headers: Optional[dict] = None,
    ) -> ModelResponse:
        """
        Execute asynchronous completion using watsonx agent API.

        Args:
            model: Model identifier (format: watsonx_agent/AGENT_ID)
            messages: List of messages
            api_base: API base URL
            api_key: API key
            model_response: ModelResponse object to populate
            logging_obj: Logging object
            optional_params: Optional parameters
            litellm_params: LiteLLM parameters
            timeout: Request timeout
            headers: Request headers

        Returns:
            ModelResponse object
        """
        import litellm
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.WATSONX,
            params={"ssl_verify": litellm_params.get("ssl_verify", None)},
        )

        # Validate environment and update headers
        headers = self.config.validate_environment(
            headers=headers or {},
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        # Get complete URL
        url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=optional_params.get("stream", False),
        )

        # Transform request
        request_data = self.config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Add thread_id header if provided
        thread_id = optional_params.get("thread_id")
        if thread_id:
            headers["X-IBM-THREAD-ID"] = str(thread_id)

        verbose_logger.debug(
            f"Watsonx Agent API request - URL: {url}, Headers: {headers}, Data: {request_data}"
        )

        # Make asynchronous request
        response = await client.post(
            url=url,
            json=request_data,
            headers=headers,
            timeout=timeout,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_message = e.response.text
            status_code = e.response.status_code
            raise self.config.get_error_class(
                error_message=error_message,
                status_code=status_code,
                headers=e.response.headers,
            )

        # Transform response
        return self.config.transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=None,
            api_key=api_key,
        )


# Singleton instance
watsonx_agent_handler = WatsonXAgentHandler()
