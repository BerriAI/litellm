"""
HTTP Handler for Interactions API requests.

This module handles the HTTP communication for the Google Interactions API.
"""

import json
from typing import (
    Any,
    AsyncIterator,
    Coroutine,
    Dict,
    Iterator,
    Optional,
    Union,
)

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.interactions.streaming_iterator import (
    InteractionsAPIStreamingIterator,
    SyncInteractionsAPIStreamingIterator,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.interactions import (
    CancelInteractionResult,
    DeleteInteractionResult,
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIResponse,
    InteractionsAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams


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
        interactions_api_config: BaseInteractionsAPIConfig,
        optional_params: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        model: Optional[str] = None,
        agent: Optional[str] = None,
        input: Optional[InteractionInput] = None,
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
        
        Per Google's OpenAPI spec, the endpoint is POST /{api_version}/interactions
        """
        if _is_async:
            return self.async_create_interaction(
                model=model,
                agent=agent,
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
            model=model or "",
            litellm_params=litellm_params,
        )

        api_base = interactions_api_config.get_complete_url(
            api_base=litellm_params.api_base or "",
            model=model,
            agent=agent,
            litellm_params=dict(litellm_params),
            stream=stream,
        )

        data = interactions_api_config.transform_request(
            model=model,
            agent=agent,
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
        interactions_api_config: BaseInteractionsAPIConfig,
        optional_params: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: str,
        litellm_params: GenericLiteLLMParams,
        logging_obj: LiteLLMLoggingObj,
        model: Optional[str] = None,
        agent: Optional[str] = None,
        input: Optional[InteractionInput] = None,
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
            model=model or "",
            litellm_params=litellm_params,
        )

        api_base = interactions_api_config.get_complete_url(
            api_base=litellm_params.api_base or "",
            model=model,
            agent=agent,
            litellm_params=dict(litellm_params),
            stream=stream,
        )

        data = interactions_api_config.transform_request(
            model=model,
            agent=agent,
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
        model: Optional[str],
        logging_obj: LiteLLMLoggingObj,
        interactions_api_config: BaseInteractionsAPIConfig,
    ) -> SyncInteractionsAPIStreamingIterator:
        """Create a synchronous streaming iterator.
        
        Google AI's streaming format uses SSE (Server-Sent Events).
        Returns a proper streaming iterator that yields chunks as they arrive.
        """
        return SyncInteractionsAPIStreamingIterator(
            response=response,
            model=model,
            interactions_api_config=interactions_api_config,
            logging_obj=logging_obj,
        )

    def _create_async_streaming_iterator(
        self,
        response: httpx.Response,
        model: Optional[str],
        logging_obj: LiteLLMLoggingObj,
        interactions_api_config: BaseInteractionsAPIConfig,
    ) -> InteractionsAPIStreamingIterator:
        """Create an asynchronous streaming iterator.
        
        Google AI's streaming format uses SSE (Server-Sent Events).
        Returns a proper streaming iterator that yields chunks as they arrive.
        """
        return InteractionsAPIStreamingIterator(
            response=response,
            model=model,
            interactions_api_config=interactions_api_config,
            logging_obj=logging_obj,
        )

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
            api_base=litellm_params.api_base or "",
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
            api_base=litellm_params.api_base or "",
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
            api_base=litellm_params.api_base or "",
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
            api_base=litellm_params.api_base or "",
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
            api_base=litellm_params.api_base or "",
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
            api_base=litellm_params.api_base or "",
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


# Initialize the HTTP handler singleton
interactions_http_handler = InteractionsHTTPHandler()

