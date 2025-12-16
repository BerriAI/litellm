"""
LiteLLM Interactions API - Main Module

This module provides the SDK-level methods for Google's Interactions API.
Following the same pattern as litellm/responses/main.py.

Endpoints supported:
- Create interaction: POST /v1beta/models/{model}:generateContent
- Get interaction: GET /v1beta/interactions/{interaction_id}
- Delete interaction: DELETE /v1beta/interactions/{interaction_id}
- Cancel interaction: POST /v1beta/interactions/{interaction_id}:cancel
"""

import asyncio
import contextvars
from functools import partial
from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.interactions.main import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionGenerationConfig,
    InteractionInput,
    InteractionInputContent,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
    InteractionSafetySettings,
    InteractionTool,
    InteractionToolConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import client



# ============================================================
# HTTP Handler for Interactions API
# ============================================================


class InteractionsHTTPHandler:
    """
    HTTP handler for Interactions API requests.
    """

    def _handle_error(
        self,
        e: Exception,
        provider_config: BaseInteractionsAPIConfig,
    ) -> Exception:
        """Handle errors from HTTP requests."""
        if isinstance(e, httpx.HTTPStatusError):
            error_message = e.response.text
            status_code = e.response.status_code
            headers = dict(e.response.headers)
            return provider_config.get_error_class(
                error_message=error_message,
                status_code=status_code,
                headers=headers,
            )
        return e

    # =========================================================
    # CREATE INTERACTION
    # =========================================================

    def create_interaction(
        self,
        model: str,
        input: InteractionInput,
        interactions_api_config: BaseInteractionsAPIConfig,
        optional_params: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
    ) -> Union[
        InteractionsAPIResponse,
        Iterator[InteractionsAPIStreamingResponse],
        Coroutine[Any, Any, Union[InteractionsAPIResponse, AsyncIterator[InteractionsAPIStreamingResponse]]],
    ]:
        """
        Create a new interaction (synchronous or async based on _is_async flag).
        """
        if _is_async:
            return self.async_create_interaction(
                model=model,
                input=input,
                interactions_api_config=interactions_api_config,
                optional_params=optional_params,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                extra_body=extra_body,
                timeout=timeout,
                stream=stream,
            )

        if client is None:
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = interactions_api_config.validate_environment(
            headers=extra_headers or {},
            model=model,
            litellm_params=litellm_params,
        )

        api_base = interactions_api_config.get_complete_url(
            api_base=litellm_params.api_base,
            model=model,
            litellm_params=dict(litellm_params),
            stream=stream,
        )

        data = interactions_api_config.transform_request(
            model=model,
            input=input,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        if extra_body:
            data.update(extra_body)

        # Logging
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            if stream:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout or request_timeout,
                    stream=True,
                )
                return self._create_sync_streaming_iterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    interactions_api_config=interactions_api_config,
                )
            else:
                response = sync_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout or request_timeout,
                )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=interactions_api_config)

        return interactions_api_config.transform_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_create_interaction(
        self,
        model: str,
        input: InteractionInput,
        interactions_api_config: BaseInteractionsAPIConfig,
        optional_params: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
        stream: Optional[bool] = None,
    ) -> Union[InteractionsAPIResponse, AsyncIterator[InteractionsAPIStreamingResponse]]:
        """
        Create a new interaction (async version).
        """
        if client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = interactions_api_config.validate_environment(
            headers=extra_headers or {},
            model=model,
            litellm_params=litellm_params,
        )

        api_base = interactions_api_config.get_complete_url(
            api_base=litellm_params.api_base,
            model=model,
            litellm_params=dict(litellm_params),
            stream=stream,
        )

        data = interactions_api_config.transform_request(
            model=model,
            input=input,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        if extra_body:
            data.update(extra_body)

        # Logging
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            if stream:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout or request_timeout,
                    stream=True,
                )
                return self._create_async_streaming_iterator(
                    response=response,
                    model=model,
                    logging_obj=logging_obj,
                    interactions_api_config=interactions_api_config,
                )
            else:
                response = await async_httpx_client.post(
                    url=api_base,
                    headers=headers,
                    json=data,
                    timeout=timeout or request_timeout,
                )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=interactions_api_config)

        return interactions_api_config.transform_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    def _create_sync_streaming_iterator(
        self,
        response: httpx.Response,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        interactions_api_config: BaseInteractionsAPIConfig,
    ) -> Iterator[InteractionsAPIStreamingResponse]:
        """Create a synchronous streaming iterator.
        
        Google AI's streaming format is a JSON array: [{...},{...},...]
        We read the full response and parse it as a JSON array, then yield each element.
        """
        import json

        # Read the full response and parse as JSON array
        full_content = response.read().decode("utf-8")
        try:
            chunks = json.loads(full_content)
            if isinstance(chunks, list):
                for parsed_chunk in chunks:
                    yield interactions_api_config.transform_streaming_response(
                        model=model,
                        parsed_chunk=parsed_chunk,
                        logging_obj=logging_obj,
                    )
            else:
                # Single object response
                yield interactions_api_config.transform_streaming_response(
                    model=model,
                    parsed_chunk=chunks,
                    logging_obj=logging_obj,
                )
        except json.JSONDecodeError as e:
            verbose_logger.debug(f"Failed to parse streaming response: {full_content[:500]}..., error: {e}")

    async def _create_async_streaming_iterator(
        self,
        response: httpx.Response,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        interactions_api_config: BaseInteractionsAPIConfig,
    ) -> AsyncIterator[InteractionsAPIStreamingResponse]:
        """Create an asynchronous streaming iterator.
        
        Google AI's streaming format is a JSON array: [{...},{...},...]
        We read the full response and parse it as a JSON array, then yield each element.
        """
        import json

        # Read the full response and parse as JSON array
        full_content = (await response.aread()).decode("utf-8")
        try:
            chunks = json.loads(full_content)
            if isinstance(chunks, list):
                for parsed_chunk in chunks:
                    yield interactions_api_config.transform_streaming_response(
                        model=model,
                        parsed_chunk=parsed_chunk,
                        logging_obj=logging_obj,
                    )
            else:
                # Single object response
                yield interactions_api_config.transform_streaming_response(
                    model=model,
                    parsed_chunk=chunks,
                    logging_obj=logging_obj,
                )
        except json.JSONDecodeError as e:
            verbose_logger.debug(f"Failed to parse async streaming response: {full_content[:500]}..., error: {e}")

    # =========================================================
    # GET INTERACTION
    # =========================================================

    def get_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
    ) -> Union[InteractionsAPIResponse, Coroutine[Any, Any, InteractionsAPIResponse]]:
        """Get an interaction by ID."""
        if _is_async:
            return self.async_get_interaction(
                interaction_id=interaction_id,
                interactions_api_config=interactions_api_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )

        if client is None:
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = interactions_api_config.validate_environment(
            headers=extra_headers or {},
            model="",
            litellm_params=litellm_params,
        )

        url, params = interactions_api_config.transform_get_interaction_request(
            interaction_id=interaction_id,
            api_base=litellm_params.api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        logging_obj.pre_call(
            input=interaction_id,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )

        try:
            response = sync_httpx_client.get(
                url=url,
                headers=headers,
                params=params,
                timeout=timeout or request_timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=interactions_api_config)

        return interactions_api_config.transform_get_interaction_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_get_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> InteractionsAPIResponse:
        """Get an interaction by ID (async version)."""
        if client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = interactions_api_config.validate_environment(
            headers=extra_headers or {},
            model="",
            litellm_params=litellm_params,
        )

        url, params = interactions_api_config.transform_get_interaction_request(
            interaction_id=interaction_id,
            api_base=litellm_params.api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        logging_obj.pre_call(
            input=interaction_id,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )

        try:
            response = await async_httpx_client.get(
                url=url,
                headers=headers,
                params=params,
                timeout=timeout or request_timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=interactions_api_config)

        return interactions_api_config.transform_get_interaction_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    # =========================================================
    # DELETE INTERACTION
    # =========================================================

    def delete_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
    ) -> Union[DeleteInteractionResult, Coroutine[Any, Any, DeleteInteractionResult]]:
        """Delete an interaction by ID."""
        if _is_async:
            return self.async_delete_interaction(
                interaction_id=interaction_id,
                interactions_api_config=interactions_api_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )

        if client is None:
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = interactions_api_config.validate_environment(
            headers=extra_headers or {},
            model="",
            litellm_params=litellm_params,
        )

        url, data = interactions_api_config.transform_delete_interaction_request(
            interaction_id=interaction_id,
            api_base=litellm_params.api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        logging_obj.pre_call(
            input=interaction_id,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )

        try:
            response = sync_httpx_client.delete(
                url=url,
                headers=headers,
                timeout=timeout or request_timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=interactions_api_config)

        return interactions_api_config.transform_delete_interaction_response(
            raw_response=response,
            logging_obj=logging_obj,
            interaction_id=interaction_id,
        )

    async def async_delete_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> DeleteInteractionResult:
        """Delete an interaction by ID (async version)."""
        if client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = interactions_api_config.validate_environment(
            headers=extra_headers or {},
            model="",
            litellm_params=litellm_params,
        )

        url, data = interactions_api_config.transform_delete_interaction_request(
            interaction_id=interaction_id,
            api_base=litellm_params.api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        logging_obj.pre_call(
            input=interaction_id,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )

        try:
            response = await async_httpx_client.delete(
                url=url,
                headers=headers,
                timeout=timeout or request_timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=interactions_api_config)

        return interactions_api_config.transform_delete_interaction_response(
            raw_response=response,
            logging_obj=logging_obj,
            interaction_id=interaction_id,
        )

    # =========================================================
    # CANCEL INTERACTION
    # =========================================================

    def cancel_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[HTTPHandler] = None,
        _is_async: bool = False,
    ) -> Union[CancelInteractionResult, Coroutine[Any, Any, CancelInteractionResult]]:
        """Cancel an interaction by ID."""
        if _is_async:
            return self.async_cancel_interaction(
                interaction_id=interaction_id,
                interactions_api_config=interactions_api_config,
                custom_llm_provider=custom_llm_provider,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                extra_headers=extra_headers,
                timeout=timeout,
            )

        if client is None:
            sync_httpx_client = _get_httpx_client(
                params={"ssl_verify": litellm_params.get("ssl_verify", None)}
            )
        else:
            sync_httpx_client = client

        headers = interactions_api_config.validate_environment(
            headers=extra_headers or {},
            model="",
            litellm_params=litellm_params,
        )

        url, data = interactions_api_config.transform_cancel_interaction_request(
            interaction_id=interaction_id,
            api_base=litellm_params.api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        logging_obj.pre_call(
            input=interaction_id,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )

        try:
            response = sync_httpx_client.post(
                url=url,
                headers=headers,
                json=data,
                timeout=timeout or request_timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=interactions_api_config)

        return interactions_api_config.transform_cancel_interaction_response(
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_cancel_interaction(
        self,
        interaction_id: str,
        interactions_api_config: BaseInteractionsAPIConfig,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        extra_headers: Optional[Dict[str, Any]] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> CancelInteractionResult:
        """Cancel an interaction by ID (async version)."""
        if client is None:
            async_httpx_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders(custom_llm_provider),
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )
        else:
            async_httpx_client = client

        headers = interactions_api_config.validate_environment(
            headers=extra_headers or {},
            model="",
            litellm_params=litellm_params,
        )

        url, data = interactions_api_config.transform_cancel_interaction_request(
            interaction_id=interaction_id,
            api_base=litellm_params.api_base,
            litellm_params=litellm_params,
            headers=headers,
        )

        logging_obj.pre_call(
            input=interaction_id,
            api_key="",
            additional_args={"api_base": url, "headers": headers},
        )

        try:
            response = await async_httpx_client.post(
                url=url,
                headers=headers,
                json=data,
                timeout=timeout or request_timeout,
            )
        except Exception as e:
            raise self._handle_error(e=e, provider_config=interactions_api_config)

        return interactions_api_config.transform_cancel_interaction_response(
            raw_response=response,
            logging_obj=logging_obj,
        )


# Initialize the HTTP handler
interactions_http_handler = InteractionsHTTPHandler()


# ============================================================
# Provider Config Manager Extension
# ============================================================


def get_provider_interactions_api_config(
    provider: str,
    model: Optional[str] = None,
) -> Optional[BaseInteractionsAPIConfig]:
    """
    Get the interactions API config for the given provider.
    
    Args:
        provider: The LLM provider name
        model: Optional model name
        
    Returns:
        The provider-specific interactions API config, or None if not supported
    """
    from litellm.types.utils import LlmProviders
    
    if provider == LlmProviders.GEMINI.value or provider == "gemini":
        from litellm.llms.gemini.interactions.transformation import (
            GoogleAIStudioInteractionsConfig,
        )
        return GoogleAIStudioInteractionsConfig()
    
    return None


# ============================================================
# SDK Methods - CREATE INTERACTION
# ============================================================


@client
async def ainteractions(
    model: str,
    contents: InteractionInput,
    # Generation config
    generation_config: Optional[InteractionGenerationConfig] = None,
    # Safety settings
    safety_settings: Optional[List[InteractionSafetySettings]] = None,
    # Tools
    tools: Optional[List[InteractionTool]] = None,
    tool_config: Optional[InteractionToolConfig] = None,
    # System instruction
    system_instruction: Optional[InteractionInputContent] = None,
    # Caching
    cached_content: Optional[str] = None,
    # Streaming
    stream: Optional[bool] = None,
    # Extra params
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM params
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[InteractionsAPIResponse, AsyncIterator[InteractionsAPIStreamingResponse]]:
    """
    Async: Create a new interaction using Google's Interactions API.
    
    Args:
        model: The model to use (e.g., "gemini/gemini-2.0-flash")
        contents: The input content (string or list of content objects)
        generation_config: Generation configuration (temperature, max_tokens, etc.)
        safety_settings: Safety filter settings
        tools: Tools available for the model to use
        tool_config: Configuration for tool usage
        system_instruction: System instruction for the model
        cached_content: Reference to cached content
        stream: Whether to stream the response
        extra_headers: Additional headers to send
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Override the LLM provider
        **kwargs: Additional parameters
        
    Returns:
        InteractionsAPIResponse for non-streaming, or async iterator for streaming
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["ainteractions"] = True
        
        # Get provider from model
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=kwargs.get("api_base", None)
            )
        
        func = partial(
            interactions,
            model=model,
            contents=contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            tools=tools,
            tool_config=tool_config,
            system_instruction=system_instruction,
            cached_content=cached_content,
            stream=stream,
            extra_headers=extra_headers,
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
def interactions(
    model: str,
    contents: InteractionInput,
    # Generation config
    generation_config: Optional[InteractionGenerationConfig] = None,
    # Safety settings
    safety_settings: Optional[List[InteractionSafetySettings]] = None,
    # Tools
    tools: Optional[List[InteractionTool]] = None,
    tool_config: Optional[InteractionToolConfig] = None,
    # System instruction
    system_instruction: Optional[InteractionInputContent] = None,
    # Caching
    cached_content: Optional[str] = None,
    # Streaming
    stream: Optional[bool] = None,
    # Extra params
    extra_headers: Optional[Dict[str, Any]] = None,
    extra_body: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    # LiteLLM params
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[
    InteractionsAPIResponse,
    Iterator[InteractionsAPIStreamingResponse],
    Coroutine[Any, Any, Union[InteractionsAPIResponse, AsyncIterator[InteractionsAPIStreamingResponse]]],
]:
    """
    Sync: Create a new interaction using Google's Interactions API.
    
    Args:
        model: The model to use (e.g., "gemini/gemini-2.0-flash")
        contents: The input content (string or list of content objects)
        generation_config: Generation configuration (temperature, max_tokens, etc.)
        safety_settings: Safety filter settings
        tools: Tools available for the model to use
        tool_config: Configuration for tool usage
        system_instruction: System instruction for the model
        cached_content: Reference to cached content
        stream: Whether to stream the response
        extra_headers: Additional headers to send
        extra_body: Additional body parameters
        timeout: Request timeout
        custom_llm_provider: Override the LLM provider
        **kwargs: Additional parameters
        
    Returns:
        InteractionsAPIResponse for non-streaming, or iterator for streaming
    """
    local_vars = locals()
    
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("ainteractions", False) is True
        
        # Get LiteLLM params
        litellm_params = GenericLiteLLMParams(**kwargs)
        
        # Get provider from model
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
        
        # Get provider config
        interactions_api_config = get_provider_interactions_api_config(
            provider=custom_llm_provider,
            model=model,
        )
        
        if interactions_api_config is None:
            raise ValueError(
                f"Interactions API is not supported for provider: {custom_llm_provider}. "
                "Currently only 'gemini' (Google AI Studio) is supported."
            )
        
        # Build optional params
        optional_params: InteractionsAPIOptionalRequestParams = {}
        if generation_config:
            optional_params["generation_config"] = generation_config
        if safety_settings:
            optional_params["safety_settings"] = safety_settings
        if tools:
            optional_params["tools"] = tools
        if tool_config:
            optional_params["tool_config"] = tool_config
        if system_instruction:
            optional_params["system_instruction"] = system_instruction
        if cached_content:
            optional_params["cached_content"] = cached_content
        if stream is not None:
            optional_params["stream"] = stream
        
        # Pre-call logging
        litellm_logging_obj.update_environment_variables(
            model=model,
            optional_params=dict(optional_params),
            litellm_params={
                "litellm_call_id": litellm_call_id,
            },
            custom_llm_provider=custom_llm_provider,
        )
        
        response = interactions_http_handler.create_interaction(
            model=model,
            input=contents,
            interactions_api_config=interactions_api_config,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
            _is_async=_is_async,
            stream=stream,
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


# ============================================================
# SDK Methods - GET INTERACTION
# ============================================================


@client
async def aget_interaction(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> InteractionsAPIResponse:
    """
    Async: Get an interaction by its ID.
    
    Args:
        interaction_id: The interaction ID to fetch
        extra_headers: Additional headers
        timeout: Request timeout
        custom_llm_provider: The LLM provider (defaults to "gemini")
        **kwargs: Additional parameters
        
    Returns:
        The interaction response
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aget_interaction"] = True
        
        func = partial(
            get_interaction,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
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
            model=None,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def get_interaction(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[InteractionsAPIResponse, Coroutine[Any, Any, InteractionsAPIResponse]]:
    """
    Sync: Get an interaction by its ID.
    
    Args:
        interaction_id: The interaction ID to fetch
        extra_headers: Additional headers
        timeout: Request timeout
        custom_llm_provider: The LLM provider (defaults to "gemini")
        **kwargs: Additional parameters
        
    Returns:
        The interaction response
    """
    local_vars = locals()
    custom_llm_provider = custom_llm_provider or "gemini"
    
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("aget_interaction", False) is True
        
        litellm_params = GenericLiteLLMParams(**kwargs)
        
        interactions_api_config = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )
        
        if interactions_api_config is None:
            raise ValueError(
                f"Interactions API is not supported for provider: {custom_llm_provider}"
            )
        
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={"interaction_id": interaction_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )
        
        return interactions_http_handler.get_interaction(
            interaction_id=interaction_id,
            interactions_api_config=interactions_api_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ============================================================
# SDK Methods - DELETE INTERACTION
# ============================================================


@client
async def adelete_interaction(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> DeleteInteractionResult:
    """
    Async: Delete an interaction by its ID.
    
    Args:
        interaction_id: The interaction ID to delete
        extra_headers: Additional headers
        timeout: Request timeout
        custom_llm_provider: The LLM provider (defaults to "gemini")
        **kwargs: Additional parameters
        
    Returns:
        The delete result
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["adelete_interaction"] = True
        
        func = partial(
            delete_interaction,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
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
            model=None,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def delete_interaction(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[DeleteInteractionResult, Coroutine[Any, Any, DeleteInteractionResult]]:
    """
    Sync: Delete an interaction by its ID.
    
    Args:
        interaction_id: The interaction ID to delete
        extra_headers: Additional headers
        timeout: Request timeout
        custom_llm_provider: The LLM provider (defaults to "gemini")
        **kwargs: Additional parameters
        
    Returns:
        The delete result
    """
    local_vars = locals()
    custom_llm_provider = custom_llm_provider or "gemini"
    
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("adelete_interaction", False) is True
        
        litellm_params = GenericLiteLLMParams(**kwargs)
        
        interactions_api_config = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )
        
        if interactions_api_config is None:
            raise ValueError(
                f"Interactions API is not supported for provider: {custom_llm_provider}"
            )
        
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={"interaction_id": interaction_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )
        
        return interactions_http_handler.delete_interaction(
            interaction_id=interaction_id,
            interactions_api_config=interactions_api_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


# ============================================================
# SDK Methods - CANCEL INTERACTION
# ============================================================


@client
async def acancel_interaction(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> CancelInteractionResult:
    """
    Async: Cancel an interaction by its ID.
    
    Args:
        interaction_id: The interaction ID to cancel
        extra_headers: Additional headers
        timeout: Request timeout
        custom_llm_provider: The LLM provider (defaults to "gemini")
        **kwargs: Additional parameters
        
    Returns:
        The cancel result
    """
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["acancel_interaction"] = True
        
        func = partial(
            cancel_interaction,
            interaction_id=interaction_id,
            extra_headers=extra_headers,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider or "gemini",
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
            model=None,
            custom_llm_provider=custom_llm_provider or "gemini",
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def cancel_interaction(
    interaction_id: str,
    extra_headers: Optional[Dict[str, Any]] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    **kwargs,
) -> Union[CancelInteractionResult, Coroutine[Any, Any, CancelInteractionResult]]:
    """
    Sync: Cancel an interaction by its ID.
    
    Args:
        interaction_id: The interaction ID to cancel
        extra_headers: Additional headers
        timeout: Request timeout
        custom_llm_provider: The LLM provider (defaults to "gemini")
        **kwargs: Additional parameters
        
    Returns:
        The cancel result
    """
    local_vars = locals()
    custom_llm_provider = custom_llm_provider or "gemini"
    
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.get("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("acancel_interaction", False) is True
        
        litellm_params = GenericLiteLLMParams(**kwargs)
        
        interactions_api_config = get_provider_interactions_api_config(
            provider=custom_llm_provider,
        )
        
        if interactions_api_config is None:
            raise ValueError(
                f"Interactions API is not supported for provider: {custom_llm_provider}"
            )
        
        litellm_logging_obj.update_environment_variables(
            model=None,
            optional_params={"interaction_id": interaction_id},
            litellm_params={"litellm_call_id": litellm_call_id},
            custom_llm_provider=custom_llm_provider,
        )
        
        return interactions_http_handler.cancel_interaction(
            interaction_id=interaction_id,
            interactions_api_config=interactions_api_config,
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            logging_obj=litellm_logging_obj,
            extra_headers=extra_headers,
            timeout=timeout,
            _is_async=_is_async,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=None,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )
