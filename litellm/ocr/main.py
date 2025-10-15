"""
Main OCR function for LiteLLM.
"""
import asyncio
import contextvars
from functools import partial
from typing import Any, Dict, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig, OCRResponse
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.utils import FileTypes
from litellm.utils import ProviderConfigManager, client

####### ENVIRONMENT VARIABLES ###################
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


@client
async def aocr(
    model: str,
    image: FileTypes,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> OCRResponse:
    """
    Async OCR function.
    
    Args:
        model: Model name (e.g., "mistral/pixtral-12b-2409")
        image: Image file to process (file path, bytes, or file object)
        api_key: Optional API key
        api_base: Optional API base URL
        timeout: Optional timeout
        custom_llm_provider: Optional custom LLM provider
        extra_headers: Optional extra headers
        **kwargs: Additional parameters
        
    Returns:
        OCRResponse in Mistral OCR format with pages, model, usage_info, etc.
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aocr"] = True

        # Get custom llm provider
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=api_base
            )

        func = partial(
            ocr,
            model=model,
            image=image,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

        if response is None:
            raise ValueError(
                f"Got an unexpected None response from the OCR API: {response}"
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
def ocr(
    model: str,
    image: FileTypes,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> OCRResponse:
    """
    Synchronous OCR function.
    
    Args:
        model: Model name (e.g., "mistral/pixtral-12b-2409")
        image: Image file to process (file path, bytes, or file object)
        api_key: Optional API key
        api_base: Optional API base URL
        timeout: Optional timeout
        custom_llm_provider: Optional custom LLM provider
        extra_headers: Optional extra headers
        **kwargs: Additional parameters
        
    Returns:
        OCRResponse in Mistral OCR format with pages, model, usage_info, etc.
        
    Example:
        ```python
        import litellm
        from litellm import ocr
        
        # OCR with Mistral
        response = ocr(
            model="mistral/pixtral-12b-2409",
            image="path/to/image.jpg",
            api_key="your-mistral-api-key"
        )
        
        # Access pages
        for page in response.pages:
            print(f"Page {page.index}: {page.markdown}")
        
        # Get all text
        all_text = "\\n\\n".join(page.markdown for page in response.pages)
        ```
    """
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("aocr", False) is True

        model, custom_llm_provider, dynamic_api_key, dynamic_api_base = (
            litellm.get_llm_provider(
                model=model,
                custom_llm_provider=custom_llm_provider,
                api_base=api_base,
                api_key=api_key,
            )
        )
        
        # Update with dynamic values if available
        if dynamic_api_key:
            api_key = dynamic_api_key
        if dynamic_api_base:
            api_base = dynamic_api_base

        # Get provider config
        ocr_provider_config: Optional[BaseOCRConfig] = (
            ProviderConfigManager.get_provider_ocr_config(
                model=model,
                provider=litellm.LlmProviders(custom_llm_provider),
            )
        )

        if ocr_provider_config is None:
            raise ValueError(
                f"OCR is not supported for provider: {custom_llm_provider}"
            )

        verbose_logger.debug(
            f"OCR call - model: {model}, provider: {custom_llm_provider}"
        )

        # Extract OCR-specific parameters from kwargs
        supported_params = ocr_provider_config.get_supported_ocr_params(model=model)
        non_default_params = {}
        for param in supported_params:
            if param in kwargs:
                non_default_params[param] = kwargs.pop(param)
        
        # Map parameters to provider-specific format
        optional_params = ocr_provider_config.map_ocr_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
        )
        
        verbose_logger.debug(f"OCR optional_params after mapping: {optional_params}")

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=model,
            optional_params=optional_params,
            litellm_params={
                "litellm_call_id": litellm_call_id,
                "api_base": api_base,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler with _is_async flag
        response = base_llm_http_handler.ocr(
            model=model,
            image=image,
            optional_params=optional_params,
            timeout=timeout or request_timeout,
            logging_obj=litellm_logging_obj,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            aocr=_is_async,
            headers=extra_headers,
            provider_config=ocr_provider_config,
            litellm_params={
                "api_base": api_base,
                "api_key": api_key,
            },
        )

        # Since _is_async is False for sync ocr(), response should be OCRResponse
        if not isinstance(response, OCRResponse):
            raise ValueError(f"Expected OCRResponse, got {type(response)}")
        
        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )

