from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Final,
    Protocol,
    cast,  # noqa: TID251  # adapter protocols cover pluggable logging and untyped HTTP methods
)
from urllib.parse import quote

import httpx
from aiohttp import ClientSession
from pydantic import TypeAdapter, ValidationError

from litellm import LlmProviders
from litellm.constants import (
    MICROSOFT_365_COPILOT_DEFAULT_TOKEN_EXCHANGE_PROFILE,
    MICROSOFT_365_COPILOT_DEFAULT_TOKEN_EXCHANGE_SCOPE,
    MICROSOFT_GRAPH_BETA_BASE,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.oauth_token_exchange import (
    OAuthTokenExchangeConfig,
    OAuthTokenExchangeError,
    TokenExchangeProfile,
    aexchange_token,
    exchange_token,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.custom_httpx import http_handler
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
)
from litellm.llms.custom_httpx.llm_http_handler import MockResponseIterator
from litellm.llms.microsoft_365_copilot.chat.transformation import (
    GraphChatRequest,
    build_chat_request,
    extract_graph_error_message,
    map_graph_response,
    parse_graph_conversation_id,
)
from litellm.llms.microsoft_365_copilot.common_utils import (
    Microsoft365CopilotError,
    extract_caller_assertion,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.litellm_pre_call_utils import SecretFields
from litellm.types.utils import ModelResponse


@dataclass(frozen=True, slots=True, repr=False)
class _TokenExchangeCredentials:
    config: OAuthTokenExchangeConfig
    subject_token: str


class _CopilotLogging(Protocol):
    def pre_call(
        self,
        *,
        input: Sequence[AllMessageValues],
        api_key: str,
        model: str,
        additional_args: Mapping[str, object],
    ) -> object: ...

    def post_call(
        self,
        *,
        original_response: object,
        input: Sequence[AllMessageValues],
        api_key: str,
        additional_args: Mapping[str, object],
    ) -> object: ...


class _SyncGraphClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object],
        timeout: float | httpx.Timeout | None,
    ) -> httpx.Response: ...


class _AsyncGraphClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object],
        timeout: float | httpx.Timeout | None,
    ) -> httpx.Response: ...


class _HTTPClientFactory(Protocol):
    def get_httpx_client(self, params: Mapping[str, object] | None = None) -> HTTPHandler: ...

    def get_async_httpx_client(
        self,
        llm_provider: LlmProviders | str,
        params: Mapping[str, object] | None = None,
        shared_session: ClientSession | None = None,
    ) -> AsyncHTTPHandler: ...


_JSON_VALUE_ADAPTER: Final = TypeAdapter(object)
_HTTP_CLIENT_FACTORY: Final = cast(  # cast-ok: shared cached-client methods have partially typed signatures
    _HTTPClientFactory, http_handler
)


def _get_sync_http_client(params: Mapping[str, object] | None) -> HTTPHandler:
    return _HTTP_CLIENT_FACTORY.get_httpx_client(dict(params) if params is not None else None)


def _get_async_http_client(
    provider: LlmProviders | str,
    params: Mapping[str, object] | None,
    shared_session: ClientSession | None,
) -> AsyncHTTPHandler:
    return _HTTP_CLIENT_FACTORY.get_async_httpx_client(
        provider,
        dict(params) if params is not None else None,
        shared_session,
    )


def _token_exchange_profile(value: object) -> TokenExchangeProfile:
    if value is None:
        return MICROSOFT_365_COPILOT_DEFAULT_TOKEN_EXCHANGE_PROFILE
    if value == "rfc8693":
        return "rfc8693"
    if value == "jwt_bearer_obo":
        return "jwt_bearer_obo"
    raise Microsoft365CopilotError(
        status_code=400,
        message="microsoft_365_copilot token_exchange_profile must be 'rfc8693' or 'jwt_bearer_obo'",
    )


def _get_token_exchange_credentials(
    litellm_params: Mapping[str, object],
    secret_fields: SecretFields | None,
) -> _TokenExchangeCredentials | None:
    token_endpoint: Final[object] = litellm_params.get("token_exchange_endpoint")
    client_id: Final[object] = litellm_params.get("client_id")
    client_secret: Final[object] = litellm_params.get("client_secret")
    credential_values: Final = (token_endpoint, client_id, client_secret)
    configured_values: Final = tuple(value is not None and value != "" for value in credential_values)
    if not any(configured_values):
        return None
    if not all(configured_values):
        raise Microsoft365CopilotError(
            status_code=400,
            message="microsoft_365_copilot requires token_exchange_endpoint, client_id, and client_secret together",
        )
    if not isinstance(token_endpoint, str) or not isinstance(client_id, str) or not isinstance(client_secret, str):
        raise Microsoft365CopilotError(
            status_code=400,
            message=(
                "microsoft_365_copilot requires token_exchange_endpoint, client_id, and client_secret "
                "as non-empty strings"
            ),
        )
    if not token_endpoint.strip() or not client_id.strip() or not client_secret.strip():
        raise Microsoft365CopilotError(
            status_code=400,
            message="microsoft_365_copilot requires token_exchange_endpoint, client_id, and client_secret together",
        )
    profile: Final = _token_exchange_profile(litellm_params.get("token_exchange_profile"))
    scope_value: Final[object] = litellm_params.get("token_exchange_scope")
    if scope_value is not None and not isinstance(scope_value, str):
        raise Microsoft365CopilotError(
            status_code=400,
            message="microsoft_365_copilot token_exchange_scope must be a string",
        )
    audience_value: Final[object] = litellm_params.get("token_exchange_audience")
    if audience_value is not None and not isinstance(audience_value, str):
        raise Microsoft365CopilotError(
            status_code=400,
            message="microsoft_365_copilot token_exchange_audience must be a string",
        )
    scope: Final = MICROSOFT_365_COPILOT_DEFAULT_TOKEN_EXCHANGE_SCOPE if scope_value is None else scope_value
    try:
        config: Final = OAuthTokenExchangeConfig(
            token_endpoint=token_endpoint,
            client_id=client_id,
            client_secret=client_secret,
            profile=profile,
            scope=scope,
            audience=audience_value,
        )
    except OAuthTokenExchangeError as error:
        raise Microsoft365CopilotError(status_code=error.status_code, message=error.message) from error
    subject_token: Final = extract_caller_assertion(secret_fields)
    if subject_token is None:
        raise Microsoft365CopilotError(
            status_code=401,
            message=(
                "microsoft_365_copilot with OAuth token exchange requires the caller's "
                "IdP-issued access token in the Authorization header"
            ),
        )
    return _TokenExchangeCredentials(config=config, subject_token=subject_token)


def _resolve_access_token(
    client: HTTPHandler,
    api_key: str | None,
    litellm_params: Mapping[str, object],
    secret_fields: SecretFields | None,
    timeout: float | httpx.Timeout | None,
) -> str:
    exchange_credentials: Final = _get_token_exchange_credentials(
        litellm_params=litellm_params,
        secret_fields=secret_fields,
    )
    if exchange_credentials is not None:
        try:
            return exchange_token(
                client=client,
                config=exchange_credentials.config,
                subject_token=exchange_credentials.subject_token,
                timeout=timeout,
            )
        except OAuthTokenExchangeError as error:
            raise Microsoft365CopilotError(status_code=error.status_code, message=error.message) from error
    if isinstance(api_key, str) and api_key:
        return api_key
    raise Microsoft365CopilotError(
        status_code=401,
        message="microsoft_365_copilot requires api_key or a complete token exchange configuration",
    )


async def _aresolve_access_token(
    client: AsyncHTTPHandler,
    api_key: str | None,
    litellm_params: Mapping[str, object],
    secret_fields: SecretFields | None,
    timeout: float | httpx.Timeout | None,
) -> str:
    exchange_credentials: Final = _get_token_exchange_credentials(
        litellm_params=litellm_params,
        secret_fields=secret_fields,
    )
    if exchange_credentials is not None:
        try:
            return await aexchange_token(
                client=client,
                config=exchange_credentials.config,
                subject_token=exchange_credentials.subject_token,
                timeout=timeout,
            )
        except OAuthTokenExchangeError as error:
            raise Microsoft365CopilotError(status_code=error.status_code, message=error.message) from error
    if isinstance(api_key, str) and api_key:
        return api_key
    raise Microsoft365CopilotError(
        status_code=401,
        message="microsoft_365_copilot requires api_key or a complete token exchange configuration",
    )


def _json_body(response: httpx.Response) -> object:
    try:
        return _JSON_VALUE_ADAPTER.validate_json(response.content)
    except ValidationError:
        return {}


def _graph_error(
    response: httpx.Response,
    sensitive_values: tuple[str, ...],
) -> Microsoft365CopilotError:
    return Microsoft365CopilotError(
        status_code=response.status_code,
        message=extract_graph_error_message(
            response_body=_json_body(response),
            sensitive_values=sensitive_values,
        ),
    )


def _pre_call(
    logging_obj: Logging | None,
    messages: Sequence[AllMessageValues],
    model: str,
    request_data: GraphChatRequest,
) -> None:
    if logging_obj is None:
        return
    typed_logging: Final = cast(  # cast-ok: Logging callback arguments are provider-pluggable
        _CopilotLogging,
        logging_obj,
    )
    typed_logging.pre_call(
        input=messages,
        api_key="",
        model=model,
        additional_args={
            "api_base": MICROSOFT_GRAPH_BETA_BASE,
            "complete_input_dict": request_data,
        },
    )


def _post_call(
    logging_obj: Logging | None,
    messages: Sequence[AllMessageValues],
    request_data: GraphChatRequest,
    status_code: int,
) -> None:
    if logging_obj is None:
        return
    typed_logging: Final = cast(  # cast-ok: Logging callback arguments are provider-pluggable
        _CopilotLogging,
        logging_obj,
    )
    typed_logging.post_call(
        original_response={"status_code": status_code},
        input=messages,
        api_key="",
        additional_args={
            "api_base": MICROSOFT_GRAPH_BETA_BASE,
            "complete_input_dict": request_data,
        },
    )


def _post_graph_request(
    client: HTTPHandler,
    url: str,
    access_token: str,
    request_data: GraphChatRequest | Mapping[str, object],
    timeout: float | httpx.Timeout | None,
) -> httpx.Response:
    try:
        http_client: Final = cast(  # cast-ok: HTTPHandler.post has an untyped response contract
            _SyncGraphClient,
            client,
        )
        request_mapping: Final[Mapping[str, object]] = request_data
        return http_client.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=dict(request_mapping),
            timeout=timeout,
        )
    except httpx.HTTPStatusError as error:
        return error.response
    except httpx.HTTPError:
        raise Microsoft365CopilotError(status_code=502, message="Microsoft Graph request failed") from None


async def _apost_graph_request(
    client: AsyncHTTPHandler,
    url: str,
    access_token: str,
    request_data: GraphChatRequest | Mapping[str, object],
    timeout: float | httpx.Timeout | None,
) -> httpx.Response:
    try:
        http_client: Final = cast(  # cast-ok: AsyncHTTPHandler.post has an untyped response contract
            _AsyncGraphClient,
            client,
        )
        request_mapping: Final[Mapping[str, object]] = request_data
        return await http_client.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json=dict(request_mapping),
            timeout=timeout,
        )
    except httpx.HTTPStatusError as error:
        return error.response
    except httpx.HTTPError:
        raise Microsoft365CopilotError(status_code=502, message="Microsoft Graph request failed") from None


def _stream_response(
    response: ModelResponse,
    model: str,
    logging_obj: Logging | None,
) -> CustomStreamWrapper:
    if logging_obj is None:
        raise Microsoft365CopilotError(
            status_code=500,
            message="Microsoft 365 Copilot streaming requires a logging context",
        )
    return CustomStreamWrapper(
        completion_stream=MockResponseIterator(model_response=response),
        model=model,
        custom_llm_provider=LlmProviders.MICROSOFT_365_COPILOT.value,
        logging_obj=logging_obj,
    )


def _raise_graph_error_if_needed(
    response: httpx.Response,
    sensitive_values: tuple[str, ...],
    logging_obj: Logging | None,
    messages: Sequence[AllMessageValues],
    request_data: GraphChatRequest,
) -> None:
    if response.is_success:
        return
    _post_call(
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
        status_code=response.status_code,
    )
    raise _graph_error(response=response, sensitive_values=sensitive_values)


def _timeout_value(timeout: float | str | httpx.Timeout | None) -> float | httpx.Timeout | None:
    if not isinstance(timeout, str):
        return timeout
    try:
        return float(timeout)
    except ValueError:
        raise Microsoft365CopilotError(status_code=400, message="timeout must be a number") from None


def completion(
    model: str,
    messages: Sequence[AllMessageValues],
    api_key: str | None,
    litellm_params: Mapping[str, object],
    optional_params: Mapping[str, object],
    secret_fields: SecretFields | None = None,
    stream: bool = False,
    timeout: float | str | httpx.Timeout | None = None,
    logging_obj: Logging | None = None,
    client: HTTPHandler | None = None,
) -> ModelResponse | CustomStreamWrapper:
    request_data: Final = build_chat_request(
        messages=messages,
        optional_params=optional_params,
    )
    http_client: Final = client if client is not None else _get_sync_http_client({})
    timeout_value: Final = _timeout_value(timeout)
    _pre_call(logging_obj=logging_obj, messages=messages, model=model, request_data=request_data)
    access_token: Final = _resolve_access_token(
        client=http_client,
        api_key=api_key,
        litellm_params=litellm_params,
        secret_fields=secret_fields,
        timeout=timeout_value,
    )
    assertion: Final = extract_caller_assertion(secret_fields)
    configured_client_secret: Final[object] = litellm_params.get("client_secret")
    sensitive_client_secret: Final = configured_client_secret if isinstance(configured_client_secret, str) else ""
    sensitive_values: Final = (access_token, api_key or "", assertion or "", sensitive_client_secret)
    conversation_url: Final = f"{MICROSOFT_GRAPH_BETA_BASE}/copilot/conversations"
    conversation_response: Final = _post_graph_request(
        client=http_client,
        url=conversation_url,
        access_token=access_token,
        request_data={},
        timeout=timeout_value,
    )
    _raise_graph_error_if_needed(
        response=conversation_response,
        sensitive_values=sensitive_values,
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
    )
    conversation_id: Final = parse_graph_conversation_id(_json_body(conversation_response))
    chat_url: Final = f"{conversation_url}/{quote(conversation_id, safe='')}/chat"
    chat_response: Final = _post_graph_request(
        client=http_client,
        url=chat_url,
        access_token=access_token,
        request_data=request_data,
        timeout=timeout_value,
    )
    _raise_graph_error_if_needed(
        response=chat_response,
        sensitive_values=sensitive_values,
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
    )
    _post_call(
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
        status_code=chat_response.status_code,
    )
    response: Final = map_graph_response(
        graph_response=_json_body(chat_response),
        model=model,
        messages=messages,
    )
    return _stream_response(response, model, logging_obj) if stream else response


async def acompletion(
    model: str,
    messages: Sequence[AllMessageValues],
    api_key: str | None,
    litellm_params: Mapping[str, object],
    optional_params: Mapping[str, object],
    secret_fields: SecretFields | None = None,
    stream: bool = False,
    timeout: float | str | httpx.Timeout | None = None,
    logging_obj: Logging | None = None,
    client: AsyncHTTPHandler | None = None,
    shared_session: ClientSession | None = None,
) -> ModelResponse | CustomStreamWrapper:
    request_data: Final = build_chat_request(
        messages=messages,
        optional_params=optional_params,
    )
    http_client: Final = (
        client if client is not None else _get_async_http_client(LlmProviders.MICROSOFT_365_COPILOT, {}, shared_session)
    )
    timeout_value: Final = _timeout_value(timeout)
    _pre_call(logging_obj=logging_obj, messages=messages, model=model, request_data=request_data)
    access_token: Final = await _aresolve_access_token(
        client=http_client,
        api_key=api_key,
        litellm_params=litellm_params,
        secret_fields=secret_fields,
        timeout=timeout_value,
    )
    assertion: Final = extract_caller_assertion(secret_fields)
    configured_client_secret: Final[object] = litellm_params.get("client_secret")
    sensitive_client_secret: Final = configured_client_secret if isinstance(configured_client_secret, str) else ""
    sensitive_values: Final = (access_token, api_key or "", assertion or "", sensitive_client_secret)
    conversation_url: Final = f"{MICROSOFT_GRAPH_BETA_BASE}/copilot/conversations"
    conversation_response: Final = await _apost_graph_request(
        client=http_client,
        url=conversation_url,
        access_token=access_token,
        request_data={},
        timeout=timeout_value,
    )
    _raise_graph_error_if_needed(
        response=conversation_response,
        sensitive_values=sensitive_values,
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
    )
    conversation_id: Final = parse_graph_conversation_id(_json_body(conversation_response))
    chat_url: Final = f"{conversation_url}/{quote(conversation_id, safe='')}/chat"
    chat_response: Final = await _apost_graph_request(
        client=http_client,
        url=chat_url,
        access_token=access_token,
        request_data=request_data,
        timeout=timeout_value,
    )
    _raise_graph_error_if_needed(
        response=chat_response,
        sensitive_values=sensitive_values,
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
    )
    _post_call(
        logging_obj=logging_obj,
        messages=messages,
        request_data=request_data,
        status_code=chat_response.status_code,
    )
    response: Final = map_graph_response(
        graph_response=_json_body(chat_response),
        model=model,
        messages=messages,
    )
    return _stream_response(response, model, logging_obj) if stream else response
