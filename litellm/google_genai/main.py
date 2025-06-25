import asyncio
import contextvars
from functools import partial
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import httpx
from pydantic import BaseModel

import litellm
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.google_genai.transformation import (
    BaseGoogleGenAIGenerateContentConfig,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

####### ENVIRONMENT VARIABLES ###################
# Initialize any necessary instances or variables here
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


class GenerateContentSetupResult(BaseModel):
    """Internal Type - Result of setting up a generate content call"""
    model: str
    custom_llm_provider: str
    generate_content_provider_config: BaseGoogleGenAIGenerateContentConfig
    generate_content_request_params: Dict[str, Any]
    litellm_params: GenericLiteLLMParams
    litellm_logging_obj: Optional[LiteLLMLoggingObj]
    litellm_call_id: Optional[str]

    class Config:
        arbitrary_types_allowed = True


class GenerateContentHelper:
    """Helper class for Google GenAI generate content operations"""
    
    @staticmethod
    def mock_generate_content_response(
        mock_response: str = "This is a mock response from Google GenAI generate_content.",
    ) -> Dict[str, Any]:
        """Mock response for generate_content for testing purposes"""
        return {
            "text": mock_response,
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": mock_response}],
                        "role": "model"
                    },
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": []
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "totalTokenCount": 30
            }
        }

    @staticmethod
    def setup_generate_content_call(
        model: str,
        contents: Union[str, List[Dict[str, Any]]],
        config: Optional[Dict[str, Any]] = None,
        custom_llm_provider: Optional[str] = None,
        stream: bool = False,
        local_vars: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> GenerateContentSetupResult:
        """
        Common setup logic for generate_content calls
        
        Args:
            model: The model name
            contents: The content to generate from
            config: Optional configuration
            custom_llm_provider: Optional custom LLM provider
            stream: Whether this is a streaming call
            local_vars: Local variables from the calling function
            **kwargs: Additional keyword arguments
            
        Returns:
            GenerateContentSetupResult containing all setup information
        """
        litellm_logging_obj: Optional[LiteLLMLoggingObj] = kwargs.get("litellm_logging_obj")
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        
        # get llm provider logic
        litellm_params = GenericLiteLLMParams(**kwargs)

        ## MOCK RESPONSE LOGIC (only for non-streaming)
        if not stream and litellm_params.mock_response and isinstance(litellm_params.mock_response, str):
            raise ValueError("Mock response should be handled by caller")

        (
            model,
            custom_llm_provider,
            dynamic_api_key,
            dynamic_api_base,
        ) = litellm.get_llm_provider(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=litellm_params.api_base,
            api_key=litellm_params.api_key,
        )

        # get provider config
        generate_content_provider_config: Optional[BaseGoogleGenAIGenerateContentConfig] = (
            ProviderConfigManager.get_provider_google_genai_generate_content_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if generate_content_provider_config is None:
            operation = "streaming" if stream else ""
            raise ValueError(
                f"Generate content {operation} is not supported for {custom_llm_provider}".strip()
            )

        if local_vars:
            local_vars.update(kwargs)
        
        # Get optional parameters for generate content
        generate_content_request_params: Dict[str, Any] = {}
        # Pre Call logging
        if litellm_logging_obj:
            litellm_logging_obj.update_environment_variables(
                model=model,
                optional_params=dict(generate_content_request_params),
                litellm_params={
                    "litellm_call_id": litellm_call_id,
                    **generate_content_request_params,
                },
                custom_llm_provider=custom_llm_provider,
            )

        return GenerateContentSetupResult(
            model=model,
            custom_llm_provider=custom_llm_provider,
            generate_content_provider_config=generate_content_provider_config,
            generate_content_request_params=generate_content_request_params,
            litellm_params=litellm_params,
            litellm_logging_obj=litellm_logging_obj,
            litellm_call_id=litellm_call_id
        )


@client
async def agenerate_content(
    model: str,
    contents: Union[str, List[Dict[str, Any]]],
    config: Optional[Dict[str, Any]] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Async: Generate content using Google GenAI
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["agenerate_content"] = True

        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
            )

        func = partial(
            generate_content,
            model=model,
            contents=contents,
            config=config,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def generate_content(
    model: str,
    contents: Union[str, List[Dict[str, Any]]],
    config: Optional[Dict[str, Any]] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Any:
    """
    Generate content using Google GenAI
    """
    local_vars = locals()
    try:
        _is_async = kwargs.pop("agenerate_content", False) is True
        
        # Check for mock response first
        litellm_params = GenericLiteLLMParams(**kwargs)
        if litellm_params.mock_response and isinstance(litellm_params.mock_response, str):
            return GenerateContentHelper.mock_generate_content_response(
                mock_response=litellm_params.mock_response
            )

        # Setup the call
        setup_result = GenerateContentHelper.setup_generate_content_call(
            model=model,
            contents=contents,
            config=config,
            custom_llm_provider=custom_llm_provider,
            stream=False,
            local_vars=local_vars,
            **kwargs
        )

        # Call the handler
        response = base_llm_http_handler.generate_content_handler(
            model=setup_result.model,
            contents=contents,
            generate_content_provider_config=setup_result.generate_content_provider_config,
            generate_content_request_params=setup_result.generate_content_request_params,
            custom_llm_provider=setup_result.custom_llm_provider,
            litellm_params=setup_result.litellm_params,
            logging_obj=setup_result.litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            stream=False,
            litellm_metadata=kwargs.get("litellm_metadata", {}),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
async def agenerate_content_stream(
    model: str,
    contents: Union[str, List[Dict[str, Any]]],
    config: Optional[Dict[str, Any]] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs
) -> AsyncIterator[Any]:
    """
    Async: Generate content using Google GenAI with streaming response
    """
    local_vars = locals()
    try:
        # Get the streaming generator from the sync version
        kwargs["agenerate_content_stream"] = True
        
        # get custom llm provider so we can use this for mapping exceptions
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=local_vars.get("base_url", None)
            )

        # Call the sync streaming version to get the generator
        stream_response = generate_content_stream(
            model=model,
            contents=contents,
            config=config,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            **kwargs,
        )

        # Convert the sync generator to async
        for chunk in stream_response:
            yield chunk
            
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def generate_content_stream(
    model: str,
    contents: Union[str, List[Dict[str, Any]]],
    config: Optional[Dict[str, Any]] = None,
    # Use the following arguments if you need to pass additional parameters to the API that aren't available via kwargs.
    # The extra values given here take precedence over values defined on the client or passed to this method.
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_query: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM specific params,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Iterator[Any]:
    """
    Generate content using Google GenAI with streaming response
    """
    local_vars = locals()
    try:
        _is_async = kwargs.pop("agenerate_content_stream", False) is True

        # Setup the call
        setup_result = GenerateContentHelper.setup_generate_content_call(
            model=model,
            contents=contents,
            config=config,
            custom_llm_provider=custom_llm_provider,
            stream=True,
            local_vars=local_vars,
            **kwargs
        )

        # Call the handler with streaming enabled
        return base_llm_http_handler.generate_content_handler(
            model=setup_result.model,
            contents=contents,
            generate_content_provider_config=setup_result.generate_content_provider_config,
            generate_content_request_params=setup_result.generate_content_request_params,
            custom_llm_provider=setup_result.custom_llm_provider,
            litellm_params=setup_result.litellm_params,
            logging_obj=setup_result.litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout or request_timeout,
            _is_async=_is_async,
            client=kwargs.get("client"),
            stream=True,
            litellm_metadata=kwargs.get("litellm_metadata", {}),
        )

    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
    
