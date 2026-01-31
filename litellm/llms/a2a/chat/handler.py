"""
A2A Agent HTTP Handler

Handles making HTTP requests to A2A agents and processing responses.
"""

import json
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import ModelResponse, ModelResponseStream

from .streaming import A2AStreamingIterator, create_streaming_response
from .transformation import A2AAgentConfig, A2AAgentError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


async def make_a2a_call_async(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    timeout: Optional[Union[float, httpx.Timeout]],
    stream: bool = False,
) -> Tuple[Any, httpx.Headers]:
    """
    Make an async HTTP call to an A2A agent.
    
    Args:
        client: AsyncHTTPHandler instance
        api_base: The A2A agent endpoint URL
        headers: Request headers
        data: JSON-encoded request body
        timeout: Request timeout
        stream: Whether to stream the response
    
    Returns:
        Tuple of (response content or iterator, response headers)
    """
    if client is None:
        client = litellm.module_level_aclient

    try:
        response = await client.post(
            api_base,
            headers=headers,
            data=data,
            stream=stream,
            timeout=timeout,
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        
        error_body = ""
        if error_response:
            try:
                error_body = await error_response.aread()
                error_body = error_body.decode("utf-8") if isinstance(error_body, bytes) else error_body
            except Exception:
                pass
        
        raise A2AAgentError(
            status_code=e.response.status_code,
            message=f"A2A agent request failed: {error_body or str(e)}",
            headers=error_headers,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise A2AAgentError(status_code=500, message=str(e))

    if stream:
        return response.aiter_lines(), response.headers
    else:
        return response, response.headers


def make_a2a_call_sync(
    client: Optional[HTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    timeout: Optional[Union[float, httpx.Timeout]],
    stream: bool = False,
) -> Tuple[Any, httpx.Headers]:
    """
    Make a sync HTTP call to an A2A agent.
    
    Args:
        client: HTTPHandler instance
        api_base: The A2A agent endpoint URL
        headers: Request headers
        data: JSON-encoded request body
        timeout: Request timeout
        stream: Whether to stream the response
    
    Returns:
        Tuple of (response content or iterator, response headers)
    """
    if client is None:
        client = litellm.module_level_client

    try:
        response = client.post(
            api_base,
            headers=headers,
            data=data,
            stream=stream,
            timeout=timeout,
        )
    except httpx.HTTPStatusError as e:
        error_headers = getattr(e, "headers", None)
        error_response = getattr(e, "response", None)
        if error_headers is None and error_response:
            error_headers = getattr(error_response, "headers", None)
        
        error_body = ""
        if error_response:
            try:
                error_body = error_response.read()
                error_body = error_body.decode("utf-8") if isinstance(error_body, bytes) else error_body
            except Exception:
                pass
        
        raise A2AAgentError(
            status_code=e.response.status_code,
            message=f"A2A agent request failed: {error_body or str(e)}",
            headers=error_headers,
        )
    except Exception as e:
        for exception in litellm.LITELLM_EXCEPTION_TYPES:
            if isinstance(e, exception):
                raise e
        raise A2AAgentError(status_code=500, message=str(e))

    if stream:
        return response.iter_lines(), response.headers
    else:
        return response, response.headers


class A2AAgentChatCompletion(BaseLLM):
    """
    Handler for A2A Agent chat completions.
    
    This class handles making requests to A2A agents and transforming
    responses to OpenAI format.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = A2AAgentConfig()

    async def async_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Union[float, httpx.Timeout],
        client: Optional[AsyncHTTPHandler],
        encoding: Any,
        api_key: Optional[str],
        logging_obj: "LiteLLMLoggingObj",
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        custom_llm_provider: str,
        stream: bool = False,
    ):
        """
        Async completion call to A2A agent.
        """
        # Validate and get headers
        headers = self.config.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        # Get complete URL
        complete_url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

        # Transform request to A2A format
        request_data = self.config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        verbose_logger.info(f"A2A async completion: url={complete_url}, stream={stream}")

        if stream:
            return await self._async_streaming_completion(
                client=client,
                api_base=complete_url,
                headers=headers,
                data=json.dumps(request_data),
                timeout=timeout,
                model=model,
                logging_obj=logging_obj,
            )
        else:
            response, response_headers = await make_a2a_call_async(
                client=client,
                api_base=complete_url,
                headers=headers,
                data=json.dumps(request_data),
                timeout=timeout,
                stream=False,
            )

            # Transform response
            return self.config.transform_response(
                model=model,
                raw_response=response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
            )

    async def _async_streaming_completion(
        self,
        client: Optional[AsyncHTTPHandler],
        api_base: str,
        headers: dict,
        data: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        model: str,
        logging_obj: "LiteLLMLoggingObj",
    ) -> AsyncIterator[ModelResponseStream]:
        """
        Handle async streaming completion.
        """
        response_iterator, response_headers = await make_a2a_call_async(
            client=client,
            api_base=api_base,
            headers=headers,
            data=data,
            timeout=timeout,
            stream=True,
        )

        # Create streaming iterator
        streaming_iterator = A2AStreamingIterator(
            streaming_response=response_iterator,
            sync_stream=False,
        )

        # Yield transformed chunks
        async for chunk in streaming_iterator:
            yield create_streaming_response(chunk=chunk, model=model)

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        model_response: ModelResponse,
        print_verbose: Callable,
        timeout: Union[float, httpx.Timeout],
        client: Optional[HTTPHandler],
        encoding: Any,
        api_key: Optional[str],
        logging_obj: "LiteLLMLoggingObj",
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        custom_llm_provider: str,
        stream: bool = False,
    ):
        """
        Sync completion call to A2A agent.
        """
        # Validate and get headers
        headers = self.config.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        # Get complete URL
        complete_url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

        # Transform request to A2A format
        request_data = self.config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        verbose_logger.info(f"A2A sync completion: url={complete_url}, stream={stream}")

        if stream:
            return self._sync_streaming_completion(
                client=client,
                api_base=complete_url,
                headers=headers,
                data=json.dumps(request_data),
                timeout=timeout,
                model=model,
                logging_obj=logging_obj,
            )
        else:
            response, response_headers = make_a2a_call_sync(
                client=client,
                api_base=complete_url,
                headers=headers,
                data=json.dumps(request_data),
                timeout=timeout,
                stream=False,
            )

            # Transform response
            return self.config.transform_response(
                model=model,
                raw_response=response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=request_data,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=encoding,
            )

    def _sync_streaming_completion(
        self,
        client: Optional[HTTPHandler],
        api_base: str,
        headers: dict,
        data: str,
        timeout: Optional[Union[float, httpx.Timeout]],
        model: str,
        logging_obj: "LiteLLMLoggingObj",
    ) -> Iterator[ModelResponseStream]:
        """
        Handle sync streaming completion.
        """
        response_iterator, response_headers = make_a2a_call_sync(
            client=client,
            api_base=api_base,
            headers=headers,
            data=data,
            timeout=timeout,
            stream=True,
        )

        # Create streaming iterator
        streaming_iterator = A2AStreamingIterator(
            streaming_response=response_iterator,
            sync_stream=True,
        )

        # Yield transformed chunks
        for chunk in streaming_iterator:
            yield create_streaming_response(chunk=chunk, model=model)


# Create singleton instance
a2a_agent_chat_completion = A2AAgentChatCompletion()
