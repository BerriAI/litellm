"""
Common helpers / utils across al OpenAI endpoints
"""

import hashlib
import json
import ssl
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Union

import httpx
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

if TYPE_CHECKING:
    from aiohttp import ClientSession

import inspect

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    _DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    AsyncHTTPHandler,
    get_ssl_configuration,
)


def _get_client_init_params(cls: type) -> List[str]:
    """Extract __init__ parameter names (excluding 'self') from a class."""
    return [p for p in inspect.signature(cls.__init__).parameters if p != "self"]


_OPENAI_INIT_PARAMS: List[str] = _get_client_init_params(OpenAI)
_AZURE_OPENAI_INIT_PARAMS: List[str] = _get_client_init_params(AzureOpenAI)

# Ordered params included in the cache key (excluding api_key which is hashed separately).
# get_openai_client_cache_key iterates this tuple; safety-net tests use it to detect drift.
_CACHE_KEY_IDENTITY_PARAMS = (
    "is_async",
    "api_base",
    "api_version",
    "timeout",
    "max_retries",
    "organization",
)

# Full set of params that affect client identity (includes api_key).
_CACHE_KEY_PARAMS = frozenset(_CACHE_KEY_IDENTITY_PARAMS) | {"api_key"}


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
        api_key = client_initialization_params.get("api_key")
        hashed_api_key = (
            hashlib.sha256(api_key.encode()).hexdigest() if api_key else None
        )
        return (
            f"{client_type},{hashed_api_key},"
            + ",".join(str(client_initialization_params.get(p)) for p in _CACHE_KEY_IDENTITY_PARAMS)
        )

    @staticmethod
    def get_openai_client_initialization_param_fields(
        client_type: Literal["openai", "azure"]
    ) -> List[str]:
        """Returns a list of fields that are used to initialize the OpenAI client"""
        if client_type == "openai":
            return _OPENAI_INIT_PARAMS
        else:
            return _AZURE_OPENAI_INIT_PARAMS

    @staticmethod
    def _get_async_http_client(
        shared_session: Optional["ClientSession"] = None,
    ) -> Optional[httpx.AsyncClient]:
        if litellm.aclient_session is not None:
            return litellm.aclient_session

        # Get unified SSL configuration
        ssl_config = get_ssl_configuration()

        return httpx.AsyncClient(
            verify=ssl_config,
            transport=AsyncHTTPHandler._create_async_transport(
                ssl_context=ssl_config
                if isinstance(ssl_config, ssl.SSLContext)
                else None,
                ssl_verify=ssl_config if isinstance(ssl_config, bool) else None,
                shared_session=shared_session,
            ),
            follow_redirects=True,
        )

    @staticmethod
    def _get_sync_http_client() -> Optional[httpx.Client]:
        if litellm.client_session is not None:
            return litellm.client_session

        # Get unified SSL configuration
        ssl_config = get_ssl_configuration()

        return httpx.Client(
            verify=ssl_config,
            follow_redirects=True,
        )
