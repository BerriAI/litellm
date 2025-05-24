"""
Common helpers / utils across al OpenAI endpoints
"""

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    _DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    AsyncHTTPHandler,
)


class OpenAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        request: Optional[httpx.Request] = None,
        response: Optional[httpx.Response] = None,
        headers: Optional[Union[dict, httpx.Headers]] = None,
        body: Optional[dict] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.headers = headers
        if request:
            self.request = request
        else:
            self.request = httpx.Request(method="POST", url="https://api.openai.com/v1")
        if response:
            self.response = response
        else:
            self.response = httpx.Response(
                status_code=status_code, request=self.request
            )
        super().__init__(
            status_code=status_code,
            message=self.message,
            headers=self.headers,
            request=self.request,
            response=self.response,
            body=body,
        )


####### Error Handling Utils for OpenAI API #######################
###################################################################
def drop_params_from_unprocessable_entity_error(
    e: Union[openai.UnprocessableEntityError, httpx.HTTPStatusError],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Helper function to read OpenAI UnprocessableEntityError and drop the params that raised an error from the error message.

    Args:
    e (UnprocessableEntityError): The UnprocessableEntityError exception
    data (Dict[str, Any]): The original data dictionary containing all parameters

    Returns:
    Dict[str, Any]: A new dictionary with invalid parameters removed
    """
    invalid_params: List[str] = []
    if isinstance(e, httpx.HTTPStatusError):
        error_json = e.response.json()
        error_message = error_json.get("error", {})
        error_body = error_message
    else:
        error_body = e.body
    if (
        error_body is not None
        and isinstance(error_body, dict)
        and error_body.get("message")
    ):
        message = error_body.get("message", {})
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                message = {"detail": message}
        detail = message.get("detail")

        if isinstance(detail, List) and len(detail) > 0 and isinstance(detail[0], dict):
            for error_dict in detail:
                if (
                    error_dict.get("loc")
                    and isinstance(error_dict.get("loc"), list)
                    and len(error_dict.get("loc")) == 2
                ):
                    invalid_params.append(error_dict["loc"][1])

    new_data = {k: v for k, v in data.items() if k not in invalid_params}

    return new_data


class BaseOpenAILLM:
    """
    Base class for OpenAI LLMs for getting their httpx clients and SSL verification settings
    """

    @staticmethod
    def get_cached_openai_client(
        client_initialization_params: dict, client_type: Literal["openai", "azure"]
    ) -> Optional[Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI]]:
        """Retrieves the OpenAI client from the in-memory cache based on the client initialization parameters"""
        _cache_key = BaseOpenAILLM.get_openai_client_cache_key(
            client_initialization_params=client_initialization_params,
            client_type=client_type,
        )
        _cached_client = litellm.in_memory_llm_clients_cache.get_cache(_cache_key)
        return _cached_client

    @staticmethod
    def set_cached_openai_client(
        openai_client: Union[OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI],
        client_type: Literal["openai", "azure"],
        client_initialization_params: dict,
    ):
        """Stores the OpenAI client in the in-memory cache for _DEFAULT_TTL_FOR_HTTPX_CLIENTS SECONDS"""
        _cache_key = BaseOpenAILLM.get_openai_client_cache_key(
            client_initialization_params=client_initialization_params,
            client_type=client_type,
        )
        litellm.in_memory_llm_clients_cache.set_cache(
            key=_cache_key,
            value=openai_client,
            ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
        )

    @staticmethod
    def get_openai_client_cache_key(
        client_initialization_params: dict, client_type: Literal["openai", "azure"]
    ) -> str:
        """Creates a cache key for the OpenAI client based on the client initialization parameters"""
        hashed_api_key = None
        if client_initialization_params.get("api_key") is not None:
            hash_object = hashlib.sha256(
                client_initialization_params.get("api_key", "").encode()
            )
            # Hexadecimal representation of the hash
            hashed_api_key = hash_object.hexdigest()

        # Create a more readable cache key using a list of key-value pairs
        key_parts = [
            f"hashed_api_key={hashed_api_key}",
            f"is_async={client_initialization_params.get('is_async')}",
        ]

        LITELLM_CLIENT_SPECIFIC_PARAMS = [
            "timeout",
            "max_retries",
            "organization",
            "api_base",
        ]
        openai_client_fields = (
            BaseOpenAILLM.get_openai_client_initialization_param_fields(
                client_type=client_type
            )
            + LITELLM_CLIENT_SPECIFIC_PARAMS
        )

        for param in openai_client_fields:
            key_parts.append(f"{param}={client_initialization_params.get(param)}")

        _cache_key = ",".join(key_parts)
        return _cache_key

    @staticmethod
    def get_openai_client_initialization_param_fields(
        client_type: Literal["openai", "azure"]
    ) -> List[str]:
        """Returns a list of fields that are used to initialize the OpenAI client"""
        import inspect

        from openai import AzureOpenAI, OpenAI

        if client_type == "openai":
            signature = inspect.signature(OpenAI.__init__)
        else:
            signature = inspect.signature(AzureOpenAI.__init__)

        # Extract parameter names, excluding 'self'
        param_names = [param for param in signature.parameters if param != "self"]
        return param_names

    @staticmethod
    def _get_async_http_client() -> Optional[httpx.AsyncClient]:
        if litellm.aclient_session is not None:
            return litellm.aclient_session

        return httpx.AsyncClient(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
            verify=litellm.ssl_verify,
            transport=AsyncHTTPHandler._create_async_transport(),
        )

    @staticmethod
    def _get_sync_http_client() -> Optional[httpx.Client]:
        if litellm.client_session is not None:
            return litellm.client_session
        return httpx.Client(
            limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100),
            verify=litellm.ssl_verify,
        )
