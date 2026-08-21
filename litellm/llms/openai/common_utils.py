"""
Common helpers / utils across al OpenAI endpoints
"""

import hashlib
import inspect
import json
import os
import ssl
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Final, Literal, NamedTuple, Optional

import httpx
import openai
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage

if TYPE_CHECKING:
    from aiohttp import ClientSession

import litellm
from litellm.litellm_core_utils.token_counter import token_counter
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    _DEFAULT_TTL_FOR_HTTPX_CLIENTS,
    AsyncHTTPHandler,
    get_ssl_configuration,
)


def _get_client_init_params(cls: type) -> tuple[str, ...]:
    """Extract __init__ parameter names (excluding 'self') from a class."""
    return tuple(p for p in inspect.signature(cls.__init__).parameters if p != "self")


_OPENAI_INIT_PARAMS: Final[tuple[str, ...]] = _get_client_init_params(OpenAI)
_AZURE_OPENAI_INIT_PARAMS: Final[tuple[str, ...]] = _get_client_init_params(AzureOpenAI)


class OpenAIError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        request: httpx.Request | None = None,
        response: httpx.Response | None = None,
        headers: dict | httpx.Headers | None = None,
        body: dict | None = None,
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
            self.response = httpx.Response(status_code=status_code, request=self.request)
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
    e: openai.UnprocessableEntityError | httpx.HTTPStatusError,
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Helper function to read OpenAI UnprocessableEntityError and drop the params that raised an error from the error message.

    Args:
    e (UnprocessableEntityError): The UnprocessableEntityError exception
    data (Dict[str, Any]): The original data dictionary containing all parameters

    Returns:
    Dict[str, Any]: A new dictionary with invalid parameters removed
    """
    invalid_params: Final[list[str]] = []
    if isinstance(e, httpx.HTTPStatusError):
        error_json: Final = e.response.json()
        error_message: Final = error_json.get("error", {})
        error_body = error_message
    else:
        error_body = e.body
    if error_body is not None and isinstance(error_body, dict) and error_body.get("message"):
        message = error_body.get("message", {})
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError:
                message = {"detail": message}
        detail: Final = message.get("detail")

        if isinstance(detail, list) and len(detail) > 0 and isinstance(detail[0], dict):
            for error_dict in detail:
                if (
                    error_dict.get("loc")
                    and isinstance(error_dict.get("loc"), list)
                    and len(error_dict.get("loc")) == 2
                ):
                    invalid_params.append(error_dict["loc"][1])

    new_data: Final = {k: v for k, v in data.items() if k not in invalid_params}

    return new_data


_OUTPUT_TOKEN_LIMIT_ERROR_MARKER: Final[str] = (
    "could not finish the message because max_tokens or model output limit was reached"
)


def is_output_token_limit_error(e: openai.BadRequestError) -> bool:
    """
    True when OpenAI/Azure rejected a chat request because the output budget could not fit a single visible token.

    GPT-5.x turns that case into a 400 while returning a length-truncated 200 for marginally larger budgets, so the
    match has to stay pinned to the full provider sentence to avoid swallowing genuine bad requests.
    """
    return _OUTPUT_TOKEN_LIMIT_ERROR_MARKER in e.message.lower()


def _output_token_limit_completion(model: str, prompt_tokens: int) -> ChatCompletion:
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4()}",
        choices=(
            Choice(
                index=0,
                finish_reason="length",
                message=ChatCompletionMessage(role="assistant", content=""),
            ),
        ),
        created=int(time.time()),
        model=model,
        object="chat.completion",
        usage=CompletionUsage(completion_tokens=0, prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


def _output_token_limit_chunk(model: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=f"chatcmpl-{uuid.uuid4()}",
        choices=(
            ChunkChoice(
                index=0,
                finish_reason="length",
                delta=ChoiceDelta(role="assistant", content=""),
            ),
        ),
        created=int(time.time()),
        model=model,
        object="chat.completion.chunk",
    )


def _iter_once(chunk: ChatCompletionChunk) -> Iterator[ChatCompletionChunk]:
    yield chunk


async def _aiter_once(chunk: ChatCompletionChunk) -> AsyncIterator[ChatCompletionChunk]:
    yield chunk


def build_output_token_limit_response(
    e: openai.BadRequestError, data: Mapping[str, object], is_async: bool
) -> tuple[httpx.Headers, ChatCompletion | Iterator[ChatCompletionChunk] | AsyncIterator[ChatCompletionChunk]]:
    """Synthesize the length-truncated response the provider itself returns for slightly larger output budgets.

    The provider billed the prompt it processed but sends no usage object with the 400, so the prompt is estimated
    the way every other usage-less path estimates it: reporting zero would spend input tokens against no budget.
    """
    model: Final[str] = str(data.get("model", ""))
    messages: Final = data.get("messages")
    prompt_tokens: Final = token_counter(model=model, messages=messages) if isinstance(messages, list) else 0
    if not data.get("stream"):
        return e.response.headers, _output_token_limit_completion(model, prompt_tokens)
    chunk: Final = _output_token_limit_chunk(model)
    return e.response.headers, (_aiter_once(chunk) if is_async else _iter_once(chunk))


class BaseOpenAILLM:
    """
    Base class for OpenAI LLMs for getting their httpx clients and SSL verification settings
    """

    @staticmethod
    def get_cached_openai_client(
        client_initialization_params: dict, client_type: Literal["openai", "azure"]
    ) -> OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI | None:
        """Retrieves the OpenAI client from the in-memory cache based on the client initialization parameters"""
        _cache_key: Final = BaseOpenAILLM.get_openai_client_cache_key(
            client_initialization_params=client_initialization_params,
            client_type=client_type,
        )
        _cached_client: Final = litellm.in_memory_llm_clients_cache.get_cache(_cache_key)
        return _cached_client

    @staticmethod
    def owns_wrapped_http_client(http_client: httpx.Client | httpx.AsyncClient | None) -> bool:
        """Whether litellm may close an SDK client built around ``http_client``.

        ``_get_async_http_client`` / ``_get_sync_http_client`` hand back
        ``litellm.aclient_session`` / ``litellm.client_session`` when the caller
        configured one. The SDK's ``close()`` closes whatever http client it was
        given, so an SDK client wrapping one of those shared sessions must never be
        closed on eviction; the caller goes on using the session. ``None`` means the
        SDK built its own http client, which litellm does own.
        """
        if http_client is None:
            return True
        return http_client is not litellm.aclient_session and http_client is not litellm.client_session

    @staticmethod
    def set_cached_openai_client(
        openai_client: OpenAI | AsyncOpenAI | AzureOpenAI | AsyncAzureOpenAI,
        client_type: Literal["openai", "azure"],
        client_initialization_params: dict,
        litellm_owned_client: bool = False,
    ):
        """Stores the OpenAI client in the in-memory cache for _DEFAULT_TTL_FOR_HTTPX_CLIENTS SECONDS

        ``litellm_owned_client`` says litellm built this client, so the cache may close it once it
        is evicted. A client the caller supplied stays open, since litellm does not own it.
        """
        _cache_key: Final = BaseOpenAILLM.get_openai_client_cache_key(
            client_initialization_params=client_initialization_params,
            client_type=client_type,
        )
        litellm.in_memory_llm_clients_cache.set_cache(
            key=_cache_key,
            value=openai_client,
            ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS,
            litellm_owned_client=litellm_owned_client,
        )

    @staticmethod
    def get_openai_client_cache_key(client_initialization_params: dict, client_type: Literal["openai", "azure"]) -> str:
        """Creates a cache key for the OpenAI client based on the client initialization parameters"""
        hashed_api_key = None
        if client_initialization_params.get("api_key") is not None:
            hash_object: Final = hashlib.sha256(client_initialization_params.get("api_key", "").encode())
            # Hexadecimal representation of the hash
            hashed_api_key = hash_object.hexdigest()

        # Create a more readable cache key using a list of key-value pairs
        key_parts: Final = [
            f"hashed_api_key={hashed_api_key}",
            f"is_async={client_initialization_params.get('is_async')}",
        ]

        LITELLM_CLIENT_SPECIFIC_PARAMS: Final = (
            "timeout",
            "max_retries",
            "organization",
            "api_base",
        )
        openai_client_fields: Final = (
            BaseOpenAILLM.get_openai_client_initialization_param_fields(client_type=client_type)
            + LITELLM_CLIENT_SPECIFIC_PARAMS
        )

        for param in openai_client_fields:
            key_parts.append(f"{param}={client_initialization_params.get(param)}")

        _cache_key: Final = ",".join(key_parts)
        return _cache_key

    @staticmethod
    def get_openai_client_initialization_param_fields(
        client_type: Literal["openai", "azure"],
    ) -> tuple[str, ...]:
        """Returns a tuple of fields that are used to initialize the OpenAI client"""
        if client_type == "openai":
            return _OPENAI_INIT_PARAMS
        else:
            return _AZURE_OPENAI_INIT_PARAMS

    @staticmethod
    def _get_async_http_client(
        shared_session: Optional["ClientSession"] = None,
    ) -> httpx.AsyncClient | None:
        if litellm.aclient_session is not None:
            return litellm.aclient_session

        if getattr(litellm, "network_mock", False):
            from litellm.llms.custom_httpx.mock_transport import MockOpenAITransport

            return httpx.AsyncClient(transport=MockOpenAITransport())

        # Get unified SSL configuration
        ssl_config: Final = get_ssl_configuration()

        return httpx.AsyncClient(
            verify=ssl_config,
            transport=AsyncHTTPHandler._create_async_transport(
                ssl_context=(ssl_config if isinstance(ssl_config, ssl.SSLContext) else None),
                ssl_verify=ssl_config if isinstance(ssl_config, bool) else None,
                shared_session=shared_session,
            ),
            follow_redirects=True,
        )

    @staticmethod
    def _get_sync_http_client() -> httpx.Client | None:
        if litellm.client_session is not None:
            return litellm.client_session

        if getattr(litellm, "network_mock", False):
            from litellm.llms.custom_httpx.mock_transport import MockOpenAITransport

            return httpx.Client(transport=MockOpenAITransport())

        # Get unified SSL configuration
        ssl_config: Final = get_ssl_configuration()

        return httpx.Client(
            verify=ssl_config,
            follow_redirects=True,
        )


class OpenAICredentials(NamedTuple):
    api_base: str
    api_key: str | None
    organization: str | None


def get_openai_credentials(
    api_base: str | None = None,
    api_key: str | None = None,
    organization: str | None = None,
) -> OpenAICredentials:
    """Resolve OpenAI credentials from params, litellm globals, and env vars."""
    resolved_api_base: Final = (
        api_base
        or litellm.api_base
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    resolved_organization = organization or litellm.organization or os.getenv("OPENAI_ORGANIZATION", None) or None
    resolved_api_key: Final = api_key or litellm.api_key or litellm.openai_key or os.getenv("OPENAI_API_KEY")
    return OpenAICredentials(
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        organization=resolved_organization,
    )
