"""
Common helpers / utils across al OpenAI endpoints
"""

import hashlib
import inspect
import json
import os
import ssl
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import httpx
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI

if TYPE_CHECKING:
    from aiohttp import ClientSession

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    _DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    AsyncHTTPHandler,
    _get_http2_limits,
    _get_httpx_client,
    _should_enable_http2,
    _verify_http2_available,
    get_ssl_configuration,
)


def _get_client_init_params(cls: type) -> Tuple[str, ...]:
    """Extract __init__ parameter names (excluding 'self') from a class."""
    return tuple(p for p in inspect.signature(cls.__init__).parameters if p != "self")  # type: ignore[misc]


_OPENAI_INIT_PARAMS: Tuple[str, ...] = _get_client_init_params(OpenAI)
_AZURE_OPENAI_INIT_PARAMS: Tuple[str, ...] = _get_client_init_params(AzureOpenAI)


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

        LITELLM_CLIENT_SPECIFIC_PARAMS = (
            "timeout",
            "max_retries",
            "organization",
            "api_base",
        )
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
        client_type: Literal["openai", "azure"],
    ) -> Tuple[str, ...]:
        """Returns a tuple of fields that are used to initialize the OpenAI client"""
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

        if getattr(litellm, "network_mock", False):
            from litellm.llms.custom_httpx.mock_transport import MockOpenAITransport

            return httpx.AsyncClient(transport=MockOpenAITransport())

        # Get unified SSL configuration
        ssl_config = get_ssl_configuration()

        # Respect the opt-in outbound HTTP/2 setting. A shared aiohttp session
        # cannot speak HTTP/2, so it takes priority — fall back to HTTP/1.1 and
        # warn (this builder calls the static _create_async_transport directly,
        # so it must emit the warning itself rather than relying on
        # AsyncHTTPHandler.create_client).
        http2_enabled = _should_enable_http2()
        if http2_enabled and shared_session is not None:
            verbose_logger.warning(
                "litellm: HTTP/2 is enabled but a shared aiohttp session was provided "
                "for the OpenAI/Azure client. aiohttp cannot speak HTTP/2 — using the "
                "shared session over HTTP/1.1 for this client."
            )
            http2_enabled = False
        http2_limits = _get_http2_limits() if http2_enabled else None
        if http2_enabled:
            _verify_http2_available()

        client_kwargs: dict = dict(
            verify=ssl_config,
            transport=AsyncHTTPHandler._create_async_transport(
                ssl_context=(
                    ssl_config if isinstance(ssl_config, ssl.SSLContext) else None
                ),
                ssl_verify=ssl_config if isinstance(ssl_config, bool) else None,
                shared_session=shared_session,
                http2=http2_enabled,
                limits=http2_limits,
            ),
            follow_redirects=True,
        )
        if http2_enabled:
            # Honored only when httpx builds its own transport (transport=None,
            # i.e. no force_ipv4); ignored on the explicit-transport path which
            # already carries http2/limits.
            client_kwargs["http2"] = True
            if http2_limits is not None:
                client_kwargs["limits"] = http2_limits

        return httpx.AsyncClient(**client_kwargs)

    @staticmethod
    def _get_sync_http_client() -> Optional[httpx.Client]:
        if litellm.client_session is not None:
            return litellm.client_session

        if getattr(litellm, "network_mock", False):
            from litellm.llms.custom_httpx.mock_transport import MockOpenAITransport

            return httpx.Client(transport=MockOpenAITransport())

        # Respect the opt-in outbound HTTP/2 setting. Reuse the cached HTTPHandler
        # client (via _get_httpx_client) so sync OpenAI calls share a single
        # connection pool across requests; it centralizes the transport/limits/SSL
        # wiring (incl. force_ipv4) and resolves SSL config internally.
        if _should_enable_http2():
            return _get_httpx_client().client

        # Get unified SSL configuration
        ssl_config = get_ssl_configuration()
        return httpx.Client(
            verify=ssl_config,
            follow_redirects=True,
        )


class OpenAICredentials(NamedTuple):
    api_base: str
    api_key: Optional[str]
    organization: Optional[str]


def get_openai_credentials(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    organization: Optional[str] = None,
) -> OpenAICredentials:
    """Resolve OpenAI credentials from params, litellm globals, and env vars."""
    resolved_api_base = (
        api_base
        or litellm.api_base
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    resolved_organization = (
        organization
        or litellm.organization
        or os.getenv("OPENAI_ORGANIZATION", None)
        or None
    )
    resolved_api_key = (
        api_key or litellm.api_key or litellm.openai_key or os.getenv("OPENAI_API_KEY")
    )
    return OpenAICredentials(
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        organization=resolved_organization,
    )
