from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from types import MappingProxyType
from typing import Final, NoReturn, Protocol, TypeAlias, runtime_checkable

import httpx
from openai import OpenAIError
from pydantic import JsonValue, TypeAdapter

from litellm.constants import request_timeout
from litellm.exceptions import APIConnectionError, PermissionDeniedError
from litellm.interactions.utils import get_provider_interactions_api_config
from litellm.litellm_core_utils.core_helpers import (
    process_response_headers,  # pyright: ignore[reportUnknownVariableType]  # upstream helper uses bare dict
)
from litellm.litellm_core_utils.exception_mapping_utils import (
    exception_type,  # pyright: ignore[reportUnknownVariableType]  # upstream helper lacks parameter annotations
)
from litellm.litellm_core_utils.hidden_params import get_or_create_hidden_params
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.interactions.transformation import BaseInteractionsAPIConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # upstream helper uses bare dict
    get_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # upstream helper uses bare dict
)
from litellm.llms.gemini.common_utils import GeminiError
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.llms.vertex_ai.gemini.long_decode import with_initial_prompt_tokens
from litellm.llms.vertex_ai.gemini.transformation import text_chat_model_max_output_tokens
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import Choices, LlmProviders, Message, ModelResponse, Usage

from .continuation import (
    JsonMapping,
    RecitationBlockedHop,
    initial_input_tokens_of,
    is_recitation_block,
    log_failed_hop,
    merge_hops,
    next_hop_body,
    output_tokens_of,
    resends_input_on_resume,
    resumable_body,
    with_default_output_budget,
)
from .request_transformation import MediaFetcher, transform_chat_to_interactions_request
from .response_transformation import transform_interactions_response_to_chat_response
from .streaming_iterator import (
    RECITATION_ERROR_CODE,
    InteractionsChatStreamingIterator,
    build_stream_exception,
    extract_stream_error_code_and_message,
)

CompletionResult: TypeAlias = ModelResponse | CustomStreamWrapper
DispatchResult: TypeAlias = CompletionResult | Coroutine[None, None, CompletionResult]

_JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_OBJECT_MAP: Final[TypeAdapter[Mapping[object, object]]] = TypeAdapter(Mapping[object, object])
_EMPTY_OBJ_MAP: Final[Mapping[str, object]] = MappingProxyType({})
_STRIPPED_CHAT_PARAM_KEYS: Final = frozenset({"stream", "extra_headers", "api_key", "api_base"})


class _CallLogger(Protocol):
    def pre_call(
        self,
        input: object,
        api_key: str,
        model: str | None = None,
        additional_args: Mapping[str, object] = _EMPTY_OBJ_MAP,
    ) -> object: ...

    def post_call(
        self,
        original_response: object,
        input: object = None,
        api_key: str | None = None,
        additional_args: Mapping[str, object] = _EMPTY_OBJ_MAP,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class _PreparedWireCall:
    url: str
    headers: dict[str, str]  # mutable-ok: httpx headers dict contract
    body: dict[str, object]  # mutable-ok: JSON request body passed to HTTPHandler.post
    is_stream: bool
    stream_options: dict[str, object] | None  # mutable-ok: CustomStreamWrapper stream_options contract
    renewed_headers: Callable[[], dict[str, str]]  # mutable-ok: httpx headers dict contract


class _CredentialRenewalError(ValueError):
    pass


def _hop_wire(wire: _PreparedWireCall, hop_body: dict[str, object]) -> _PreparedWireCall:  # mutable-ok: JSON body
    try:
        return replace(wire, body=hop_body, headers=wire.renewed_headers())
    except Exception as error:  # noqa: BLE001  # any credential failure ends the chain with the answer so far
        raise _CredentialRenewalError(str(error)) from error


def _resolve_stream_flag(
    optional_params: Mapping[str, object],
    litellm_params: Mapping[str, object],
    explicit: bool | None,
) -> bool:
    if explicit is not None:
        return explicit
    opt_val: Final = optional_params.get("stream")
    if isinstance(opt_val, bool):
        return opt_val
    return bool(litellm_params.get("stream", False))


def _resolve_provider_config(
    custom_llm_provider: str,
    model: str,
    override: BaseInteractionsAPIConfig | None,
) -> BaseInteractionsAPIConfig:
    if override is not None:
        return override
    resolved: Final = get_provider_interactions_api_config(custom_llm_provider, model)
    if resolved is None:
        raise ValueError(f"Interactions API is not supported for provider: {custom_llm_provider!r}")
    return resolved


def _first_non_none(candidates: Sequence[object]) -> object:
    return next((val for val in candidates if val is not None), None)


def _build_generic_litellm_params(
    litellm_params: Mapping[str, object],
    optional_params: Mapping[str, object],
    *,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str,
) -> GenericLiteLLMParams:
    explicit_project: Final = _first_non_none(
        (
            optional_params.get("vertex_project"),
            optional_params.get("vertex_ai_project"),
            litellm_params.get("vertex_project"),
            litellm_params.get("vertex_ai_project"),
        )
    )
    explicit_location: Final = _first_non_none(
        (
            optional_params.get("vertex_location"),
            optional_params.get("vertex_ai_location"),
            litellm_params.get("vertex_location"),
            litellm_params.get("vertex_ai_location"),
        )
    )
    explicit_credentials: Final = _first_non_none(
        (
            optional_params.get("vertex_credentials"),
            optional_params.get("vertex_ai_credentials"),
            litellm_params.get("vertex_credentials"),
            litellm_params.get("vertex_ai_credentials"),
        )
    )
    merged: Final = {
        **litellm_params,
        "custom_llm_provider": custom_llm_provider,
        **({"api_key": api_key} if api_key is not None else {}),
        **({"api_base": api_base} if api_base is not None else {}),
        **({"vertex_project": explicit_project} if explicit_project is not None else {}),
        **({"vertex_location": explicit_location} if explicit_location is not None else {}),
        **({"vertex_credentials": explicit_credentials} if explicit_credentials is not None else {}),
    }
    return GenericLiteLLMParams.model_validate(merged)


def _str_headers(raw_headers: object) -> dict[str, str]:  # mutable-ok: BaseInteractionsAPIConfig contract
    if not isinstance(raw_headers, Mapping):
        return {}
    validated: Final = _OBJECT_MAP.validate_python(raw_headers)
    return {str(k): str(v) for k, v in validated.items() if v is not None}


def _extract_stream_options(
    optional_params: Mapping[str, object],
) -> dict[str, object] | None:  # mutable-ok: CustomStreamWrapper contract
    raw: Final = optional_params.get("stream_options")
    if isinstance(raw, Mapping):
        validated: Final = _OBJECT_MAP.validate_python(raw)
        return {str(k): v for k, v in validated.items()}
    return None


def _attach_response_metadata(
    target: ModelResponse | CustomStreamWrapper,
    raw_response: httpx.Response,
    url: str,
) -> None:
    response_headers: Final = dict(raw_response.headers)
    hidden: Final = get_or_create_hidden_params(target)
    hidden["api_base"] = url
    hidden["headers"] = response_headers
    hidden["additional_headers"] = process_response_headers(response_headers)


def _as_call_logger(raw_logger: _CallLogger) -> _CallLogger:
    return raw_logger


@runtime_checkable
class _SyncPoster(Protocol):
    def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,  # mutable-ok: HTTPHandler.post signature
        json: dict[str, object] | None = None,  # mutable-ok: HTTPHandler.post signature
        timeout: float | httpx.Timeout | None = None,
        stream: bool = False,
    ) -> httpx.Response: ...


@runtime_checkable
class _AsyncPoster(Protocol):
    async def post(
        self,
        url: str,
        headers: dict[str, str] | None = None,  # mutable-ok: AsyncHTTPHandler.post signature
        json: dict[str, object] | None = None,  # mutable-ok: AsyncHTTPHandler.post signature
        timeout: float | httpx.Timeout | None = None,
        stream: bool = False,
    ) -> httpx.Response: ...


def _build_provider_http_error(
    custom_llm_provider: str,
    status_code: int,
    error_text: str,
    response_headers: Mapping[str, str],
) -> BaseLLMException:
    hdrs: Final = dict(response_headers)
    if custom_llm_provider in ("vertex_ai", "vertex_ai_beta"):
        return VertexAIError(status_code=status_code, message=error_text, headers=hdrs)
    return GeminiError(status_code=status_code, message=error_text, headers=hdrs)


def _safe_response_text(response: httpx.Response, fallback: str = "") -> str:
    try:
        return response.text
    except httpx.ResponseNotRead:
        return fallback


def _raise_mapped_http_error(
    *,
    model: str,
    custom_llm_provider: str,
    original_exception: Exception,
    llm_map: Mapping[str, object],
    cause: Exception | None,
    response: httpx.Response | None = None,
) -> NoReturn:
    if isinstance(original_exception, BaseLLMException) and original_exception.status_code == 403:
        stub_resp: Final = response or httpx.Response(
            403,
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/interactions"),
        )
        perm_err: Final = PermissionDeniedError(
            message=f"{custom_llm_provider.capitalize()}Exception - {original_exception.message}",
            llm_provider=custom_llm_provider,
            model=model,
            response=stub_resp,
        )
        if cause is not None:
            raise perm_err from cause
        raise perm_err
    try:
        mapped_val: Final[object] = exception_type(  # pyright: ignore[reportUnknownVariableType]  # upstream helper lacks return annotation
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=original_exception,
            extra_kwargs=dict(llm_map),
        )
    except Exception as mapped_exc:
        if cause is not None:
            raise mapped_exc from cause
        raise
    fallback_exc: Final = mapped_val if isinstance(mapped_val, Exception) else original_exception
    if cause is not None:
        raise fallback_exc from cause
    raise fallback_exc


def _handle_httpx_error(
    *,
    exc: httpx.HTTPError,
    model: str,
    custom_llm_provider: str,
    wire: _PreparedWireCall,
    messages: Sequence[object],
    api_key: str | None,
    call_logger: _CallLogger,
    llm_map: Mapping[str, object],
) -> NoReturn:
    if isinstance(exc, httpx.HTTPStatusError):
        attr_text: Final = str(getattr(exc, "text", "") or "")
        error_text: Final = attr_text or _safe_response_text(exc.response, str(exc))
        call_logger.post_call(
            input=list(messages),
            api_key=api_key or "",
            original_response=error_text,
            additional_args={"complete_input_dict": wire.body},
        )
        provider_err: Final = _build_provider_http_error(
            custom_llm_provider,
            exc.response.status_code,
            error_text,
            dict(exc.response.headers),
        )
        _raise_mapped_http_error(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=provider_err,
            llm_map=llm_map,
            cause=exc,
            response=exc.response,
        )

    call_logger.post_call(
        input=list(messages),
        api_key=api_key or "",
        original_response=str(exc),
        additional_args={"complete_input_dict": wire.body},
    )
    if isinstance(exc, httpx.TimeoutException):
        _raise_mapped_http_error(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=exc,
            llm_map=llm_map,
            cause=exc,
        )
    raw_req: Final[object] = getattr(exc, "_request", None)
    effective_req: Final = raw_req if isinstance(raw_req, httpx.Request) else httpx.Request("POST", wire.url)
    raise APIConnectionError(
        message=f"{custom_llm_provider}Exception - {exc}",
        llm_provider=custom_llm_provider,
        model=model,
        request=effective_req,
    ) from exc


def _send_sync_request(
    client: object,
    *,
    wire: _PreparedWireCall,
    timeout: float | httpx.Timeout | None,
    ssl_verify: object,
) -> httpx.Response:
    effective_timeout: Final = timeout or request_timeout
    if isinstance(client, HTTPHandler):
        resp: Final = client.post(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # HTTPHandler.post lacks return annotation
            url=wire.url,
            headers=wire.headers,
            json=wire.body,
            timeout=effective_timeout,
            stream=wire.is_stream,
        )
        assert isinstance(resp, httpx.Response)
        return resp
    if isinstance(client, httpx.Client):
        req: Final = client.build_request(
            "POST",
            wire.url,
            headers=wire.headers,
            json=wire.body,
            timeout=effective_timeout,
        )
        return client.send(req, stream=wire.is_stream)
    if isinstance(client, _SyncPoster):
        return client.post(
            url=wire.url,
            headers=wire.headers,
            json=wire.body,
            timeout=effective_timeout,
            stream=wire.is_stream,
        )
    pooled_client: Final = get_httpx_client(params={"ssl_verify": ssl_verify})
    pooled_resp: Final = pooled_client.post(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # HTTPHandler.post lacks return annotation
        url=wire.url,
        headers=wire.headers,
        json=wire.body,
        timeout=effective_timeout,
        stream=wire.is_stream,
    )
    assert isinstance(pooled_resp, httpx.Response)
    return pooled_resp


async def _send_async_request(
    client: object,
    *,
    wire: _PreparedWireCall,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str,
    ssl_verify: object,
) -> httpx.Response:
    effective_timeout: Final = timeout or request_timeout
    if isinstance(client, AsyncHTTPHandler):
        resp: Final = await client.post(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # AsyncHTTPHandler.post lacks return annotation
            url=wire.url,
            headers=wire.headers,
            json=wire.body,
            timeout=effective_timeout,
            stream=wire.is_stream,
        )
        assert isinstance(resp, httpx.Response)
        return resp
    if isinstance(client, httpx.AsyncClient):
        req: Final = client.build_request(
            "POST",
            wire.url,
            headers=wire.headers,
            json=wire.body,
            timeout=effective_timeout,
        )
        return await client.send(req, stream=wire.is_stream)
    if isinstance(client, _AsyncPoster):
        return await client.post(
            url=wire.url,
            headers=wire.headers,
            json=wire.body,
            timeout=effective_timeout,
            stream=wire.is_stream,
        )
    pooled_async: Final = get_async_httpx_client(
        llm_provider=LlmProviders(custom_llm_provider),
        params={"ssl_verify": ssl_verify},
    )
    pooled_resp: Final = await pooled_async.post(  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # AsyncHTTPHandler.post lacks return annotation
        url=wire.url,
        headers=wire.headers,
        json=wire.body,
        timeout=effective_timeout,
        stream=wire.is_stream,
    )
    assert isinstance(pooled_resp, httpx.Response)
    return pooled_resp


_HOP_FAILURES: Final = (BaseLLMException, httpx.HTTPError, ValueError, OpenAIError)


def _mark_initial_prompt_tokens(result: ModelResponse, hops: Sequence[JsonMapping]) -> None:
    usage: Final = getattr(result, "usage", None)
    if isinstance(usage, Usage):
        setattr(result, "usage", with_initial_prompt_tokens(usage, initial_input_tokens_of(hops)))  # noqa: B010  # ModelResponse sets usage dynamically


_VERTEX_MODEL_PATH: Final = re.compile(
    r"/v1(?:beta1)?/projects/[^/]+/locations/[^/]+/(?:publishers/[^/]+/)?models/[^/?]+/?(?:\?.*)?$"
)


def _without_vertex_model_path(api_base: str | None) -> str | None:
    return _VERTEX_MODEL_PATH.sub("", api_base) if api_base is not None else None


def _interaction_of(response: httpx.Response) -> JsonMapping:
    parsed: Final = _JSON_VALUE.validate_json(response.content)
    return parsed if isinstance(parsed, dict) else MappingProxyType({})


def _is_recitation_block(error_body: str) -> bool:
    try:
        parsed: Final = _JSON_VALUE.validate_json(error_body)
    except ValueError:
        return False
    error: Final = parsed.get("error") if isinstance(parsed, dict) else None
    return isinstance(error, dict) and error.get("code") == RECITATION_ERROR_CODE


def _recitation_block_response(
    model: str, model_response: ModelResponse, blocked_response: httpx.Response, url: str
) -> ModelResponse:
    blocked: Final = transform_interactions_response_to_chat_response(
        raw_response={"steps": []},
        model=model,
        model_response=model_response,
    )
    blocked.choices = [
        Choices(finish_reason="content_filter", index=0, message=Message(role="assistant", content=None))
    ]
    _attach_response_metadata(blocked, blocked_response, url)
    return blocked


_FAILED_STATUSES: Final = frozenset({"failed", "cancelled"})


def _raise_if_interaction_failed(interaction: JsonMapping, *, model: str, custom_llm_provider: str, url: str) -> None:
    status: Final = interaction.get("status")
    if status not in _FAILED_STATUSES and interaction.get("error") is None:
        return
    code, message = extract_stream_error_code_and_message(
        MappingProxyType({}), interaction, default_code=status if isinstance(status, str) else None
    )
    raise build_stream_exception(
        code_str=code,
        message=message,
        model=model,
        llm_provider=custom_llm_provider,
        request=httpx.Request("POST", url),
    )


def _completed_hop(response: httpx.Response) -> JsonMapping:
    interaction: Final = _interaction_of(response)
    if interaction.get("status") in _FAILED_STATUSES or interaction.get("error") is not None:
        raise ValueError(f"continuation hop ended with status {interaction.get('status')!r}")
    return interaction


def _checked_hop_response(response: httpx.Response, custom_llm_provider: str) -> httpx.Response:
    if response.status_code >= 400:
        with contextlib.suppress(httpx.HTTPError):
            _ = response.read()
        error_text: Final = _safe_response_text(response, f"HTTP {response.status_code}")
        if _is_recitation_block(error_text):
            raise RecitationBlockedHop(error_text)
        raise _build_provider_http_error(custom_llm_provider, response.status_code, error_text, dict(response.headers))
    return response


def _blocked_for_recitation(answer_so_far: ModelResponse) -> ModelResponse:
    choice: Final = answer_so_far.choices[0]
    choice.finish_reason = "content_filter"
    kept: Final = _OBJECT_MAP.validate_python(choice.message.provider_specific_fields or {})
    choice.message.provider_specific_fields = {
        str(key): kept[key] for key in kept if key != "continuation_token"
    } or None
    return answer_so_far


async def _checked_async_hop_response(response: httpx.Response, custom_llm_provider: str) -> httpx.Response:
    if response.status_code >= 400:
        with contextlib.suppress(httpx.HTTPError):
            _ = await response.aread()
    return _checked_hop_response(response, custom_llm_provider)


class InteractionsToCompletionBridgeHandler:
    def __init__(
        self,
        *,
        provider_config: BaseInteractionsAPIConfig | None = None,
        media_fetcher: MediaFetcher | None = None,
    ) -> None:
        self._provider_config: Final = provider_config
        self._media_fetcher: Final = media_fetcher

    def _prepare_wire_call(
        self,
        *,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        stream: bool | None,
        interactions_config: BaseInteractionsAPIConfig,
    ) -> _PreparedWireCall:
        is_stream: Final = _resolve_stream_flag(optional_params, litellm_params, stream)
        generic_params: Final = _build_generic_litellm_params(
            litellm_params,
            optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
        )
        merged_headers: Final = {
            **_str_headers(optional_params.get("extra_headers")),
            **_str_headers(headers),
        }

        def validated() -> dict[str, str]:  # mutable-ok: httpx headers dict contract
            return _str_headers(
                interactions_config.validate_environment(  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]  # BaseInteractionsAPIConfig uses bare dict
                    headers=dict(merged_headers),
                    model=model,
                    litellm_params=generic_params,
                )
            )

        validated_headers: Final = validated()
        interactions_base: Final = _without_vertex_model_path(generic_params.api_base or api_base)
        complete_url: Final = interactions_config.get_complete_url(  # pyright: ignore[reportUnknownMemberType]  # BaseInteractionsAPIConfig uses bare dict
            api_base=interactions_base,
            model=model,
            agent=None,
            litellm_params={
                **generic_params.model_dump(exclude_none=True),
                **({"api_base": interactions_base} if interactions_base is not None else {}),
            },
            stream=is_stream,
        )
        cleaned_optional: Final = {k: v for k, v in optional_params.items() if k not in _STRIPPED_CHAT_PARAM_KEYS}
        prepared: Final = transform_chat_to_interactions_request(
            messages=messages,
            model=model,
            optional_params=cleaned_optional,
            litellm_params=litellm_params,
            custom_llm_provider=custom_llm_provider,
            stream=is_stream,
            media_fetcher=self._media_fetcher,
        )
        raw_body: Final = _OBJECT_MAP.validate_python(
            interactions_config.transform_request(  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]  # BaseInteractionsAPIConfig uses bare dict
                model=model,
                agent=None,
                input=list(prepared.input_steps),  # pyright: ignore[reportArgumentType]  # steps schema JSON list
                optional_params=dict(prepared.optional_params),
                litellm_params=generic_params,
                headers=validated_headers,
            )
        )
        return _PreparedWireCall(
            url=complete_url,
            headers=validated_headers,
            body=resumable_body(
                with_default_output_budget(
                    {
                        **{str(k): v for k, v in raw_body.items()},
                        **{
                            key: prepared.optional_params[key]
                            for key in ("safety_settings",)
                            if key in prepared.optional_params
                        },
                    },
                    partial(
                        text_chat_model_max_output_tokens,
                        model,
                        custom_llm_provider,
                        (litellm_params or {}).get("model_info"),
                    ),
                ),
                resends_input=resends_input_on_resume(custom_llm_provider),
            ),
            is_stream=is_stream,
            stream_options=_extract_stream_options(optional_params),
            renewed_headers=validated,
        )

    def completion(
        self,
        *,
        model: str,
        messages: Sequence[object],
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
        headers: Mapping[str, object] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        acompletion: bool = False,
        timeout: float | httpx.Timeout | None = None,
        client: object = None,
        custom_llm_provider: str = "gemini",
        stream: bool | None = None,
        provider_config_override: BaseInteractionsAPIConfig | None = None,
    ) -> DispatchResult:
        opt_map: Final = optional_params if optional_params is not None else _EMPTY_OBJ_MAP
        llm_map: Final = litellm_params if litellm_params is not None else _EMPTY_OBJ_MAP
        if acompletion:
            return self.acompletion(
                model=model,
                messages=messages,
                model_response=model_response,
                logging_obj=logging_obj,
                optional_params=opt_map,
                litellm_params=llm_map,
                headers=headers,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                client=client,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                provider_config_override=provider_config_override,
            )

        interactions_config: Final = _resolve_provider_config(
            custom_llm_provider, model, provider_config_override or self._provider_config
        )
        wire: Final = self._prepare_wire_call(
            model=model,
            messages=messages,
            optional_params=opt_map,
            litellm_params=llm_map,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            stream=stream,
            interactions_config=interactions_config,
        )
        call_logger: Final = _as_call_logger(logging_obj)
        call_logger.pre_call(
            input=list(messages),
            api_key=api_key or "",
            additional_args={
                "complete_input_dict": wire.body,
                "api_base": wire.url,
                "headers": wire.headers,
            },
        )
        try:
            raw_response: Final = _send_sync_request(
                client,
                wire=wire,
                timeout=timeout,
                ssl_verify=llm_map.get("ssl_verify"),
            )
        except httpx.HTTPError as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                with contextlib.suppress(httpx.HTTPError):
                    exc.response.read()
                if not wire.is_stream and _is_recitation_block(blocked_text := _safe_response_text(exc.response)):
                    call_logger.post_call(
                        input=list(messages),
                        api_key=api_key or "",
                        original_response=blocked_text,
                        additional_args={"complete_input_dict": wire.body},
                    )
                    return _recitation_block_response(model, model_response, exc.response, wire.url)
            _handle_httpx_error(
                exc=exc,
                model=model,
                custom_llm_provider=custom_llm_provider,
                wire=wire,
                messages=messages,
                api_key=api_key,
                call_logger=call_logger,
                llm_map=llm_map,
            )

        if raw_response.status_code >= 400:
            with contextlib.suppress(httpx.HTTPError):
                raw_response.read()
            error_body: Final = _safe_response_text(raw_response, f"HTTP {raw_response.status_code}")
            call_logger.post_call(
                input=list(messages),
                api_key=api_key or "",
                original_response=error_body,
                additional_args={"complete_input_dict": wire.body},
            )
            if not wire.is_stream and _is_recitation_block(error_body):
                return _recitation_block_response(model, model_response, raw_response, wire.url)
            provider_err: Final = _build_provider_http_error(
                custom_llm_provider,
                raw_response.status_code,
                error_body,
                dict(raw_response.headers),
            )
            _raise_mapped_http_error(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=provider_err,
                llm_map=llm_map,
                cause=None,
                response=raw_response,
            )

        if wire.is_stream:
            iterator: Final = InteractionsChatStreamingIterator(
                model=model,
                custom_llm_provider=custom_llm_provider,
                sync_lines=raw_response.iter_lines(),
                http_response=raw_response,
                request_body=wire.body,
                open_sync_hop=lambda hop_body: _checked_hop_response(
                    _send_sync_request(
                        client,
                        wire=_hop_wire(wire, hop_body),
                        timeout=timeout,
                        ssl_verify=llm_map.get("ssl_verify"),
                    ),
                    custom_llm_provider,
                ),
            )
            wrapper: Final = CustomStreamWrapper(
                completion_stream=iterator,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
                stream_options=wire.stream_options,
            )
            call_logger.post_call(
                input=list(messages),
                api_key=api_key or "",
                original_response="first stream response received",
                additional_args={"complete_input_dict": wire.body},
            )
            _attach_response_metadata(wrapper, raw_response, wire.url)
            return wrapper

        call_logger.post_call(
            input=list(messages),
            api_key=api_key or "",
            original_response=raw_response.text,
            additional_args={"complete_input_dict": wire.body},
        )
        first_hop: Final = _interaction_of(raw_response)
        _raise_if_interaction_failed(first_hop, model=model, custom_llm_provider=custom_llm_provider, url=wire.url)
        hops: tuple[JsonMapping, ...] = (first_hop,)  # rebind-ok: grows by one hop per loop pass
        is_blocked: bool = False  # rebind-ok: set when a continuation hop is blocked for recitation
        while (
            hop_body := next_hop_body(
                wire.body,
                hops[-1],
                sum(map(output_tokens_of, hops[:-1])),
                resends_input=resends_input_on_resume(custom_llm_provider),
            )
        ) is not None:
            try:
                hop_response = _checked_hop_response(
                    _send_sync_request(
                        client,
                        wire=_hop_wire(wire, hop_body),
                        timeout=timeout,
                        ssl_verify=llm_map.get("ssl_verify"),
                    ),
                    custom_llm_provider,
                )
                hops = (*hops, _completed_hop(hop_response))
            except _HOP_FAILURES as hop_error:
                is_blocked = is_recitation_block(hop_error)
                log_failed_hop(hop_error)
                break
        result: Final = transform_interactions_response_to_chat_response(
            raw_response=merge_hops(hops),
            model=model,
            model_response=model_response,
        )
        _mark_initial_prompt_tokens(result, hops)
        _attach_response_metadata(result, raw_response, wire.url)
        return _blocked_for_recitation(result) if is_blocked else result

    async def acompletion(
        self,
        *,
        model: str,
        messages: Sequence[object],
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
        headers: Mapping[str, object] | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        client: object = None,
        custom_llm_provider: str = "gemini",
        stream: bool | None = None,
        provider_config_override: BaseInteractionsAPIConfig | None = None,
    ) -> CompletionResult:
        opt_map: Final = optional_params if optional_params is not None else _EMPTY_OBJ_MAP
        llm_map: Final = litellm_params if litellm_params is not None else _EMPTY_OBJ_MAP
        interactions_config: Final = _resolve_provider_config(
            custom_llm_provider, model, provider_config_override or self._provider_config
        )
        wire: Final = await asyncio.to_thread(
            self._prepare_wire_call,
            model=model,
            messages=messages,
            optional_params=opt_map,
            litellm_params=llm_map,
            headers=headers,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            stream=stream,
            interactions_config=interactions_config,
        )
        call_logger: Final = _as_call_logger(logging_obj)
        call_logger.pre_call(
            input=list(messages),
            api_key=api_key or "",
            additional_args={
                "complete_input_dict": wire.body,
                "api_base": wire.url,
                "headers": wire.headers,
            },
        )
        try:
            raw_response: Final = await _send_async_request(
                client,
                wire=wire,
                timeout=timeout,
                custom_llm_provider=custom_llm_provider,
                ssl_verify=llm_map.get("ssl_verify"),
            )
        except httpx.HTTPError as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                with contextlib.suppress(httpx.HTTPError):
                    await exc.response.aread()
                if not wire.is_stream and _is_recitation_block(blocked_text := _safe_response_text(exc.response)):
                    call_logger.post_call(
                        input=list(messages),
                        api_key=api_key or "",
                        original_response=blocked_text,
                        additional_args={"complete_input_dict": wire.body},
                    )
                    return _recitation_block_response(model, model_response, exc.response, wire.url)
            _handle_httpx_error(
                exc=exc,
                model=model,
                custom_llm_provider=custom_llm_provider,
                wire=wire,
                messages=messages,
                api_key=api_key,
                call_logger=call_logger,
                llm_map=llm_map,
            )

        if raw_response.status_code >= 400:
            with contextlib.suppress(httpx.HTTPError):
                await raw_response.aread()
            error_body: Final = _safe_response_text(raw_response, f"HTTP {raw_response.status_code}")
            call_logger.post_call(
                input=list(messages),
                api_key=api_key or "",
                original_response=error_body,
                additional_args={"complete_input_dict": wire.body},
            )
            if not wire.is_stream and _is_recitation_block(error_body):
                return _recitation_block_response(model, model_response, raw_response, wire.url)
            provider_err: Final = _build_provider_http_error(
                custom_llm_provider,
                raw_response.status_code,
                error_body,
                dict(raw_response.headers),
            )
            _raise_mapped_http_error(
                model=model,
                custom_llm_provider=custom_llm_provider,
                original_exception=provider_err,
                llm_map=llm_map,
                cause=None,
                response=raw_response,
            )

        if wire.is_stream:

            async def open_async_hop(hop_body: dict[str, object]) -> httpx.Response:  # mutable-ok: JSON request body
                return await _checked_async_hop_response(
                    await _send_async_request(
                        client,
                        wire=_hop_wire(wire, hop_body),
                        timeout=timeout,
                        custom_llm_provider=custom_llm_provider,
                        ssl_verify=llm_map.get("ssl_verify"),
                    ),
                    custom_llm_provider,
                )

            iterator: Final = InteractionsChatStreamingIterator(
                model=model,
                custom_llm_provider=custom_llm_provider,
                async_lines=raw_response.aiter_lines(),
                http_response=raw_response,
                request_body=wire.body,
                open_async_hop=open_async_hop,
            )
            wrapper: Final = CustomStreamWrapper(
                completion_stream=iterator,
                model=model,
                custom_llm_provider=custom_llm_provider,
                logging_obj=logging_obj,
                stream_options=wire.stream_options,
            )
            call_logger.post_call(
                input=list(messages),
                api_key=api_key or "",
                original_response="first stream response received",
                additional_args={"complete_input_dict": wire.body},
            )
            _attach_response_metadata(wrapper, raw_response, wire.url)
            return wrapper

        call_logger.post_call(
            input=list(messages),
            api_key=api_key or "",
            original_response=raw_response.text,
            additional_args={"complete_input_dict": wire.body},
        )
        first_hop: Final = _interaction_of(raw_response)
        _raise_if_interaction_failed(first_hop, model=model, custom_llm_provider=custom_llm_provider, url=wire.url)
        hops: tuple[JsonMapping, ...] = (first_hop,)  # rebind-ok: grows by one hop per loop pass
        is_blocked: bool = False  # rebind-ok: set when a continuation hop is blocked for recitation
        while (
            hop_body := next_hop_body(
                wire.body,
                hops[-1],
                sum(map(output_tokens_of, hops[:-1])),
                resends_input=resends_input_on_resume(custom_llm_provider),
            )
        ) is not None:
            try:
                hop_response = await _checked_async_hop_response(
                    await _send_async_request(
                        client,
                        wire=_hop_wire(wire, hop_body),
                        timeout=timeout,
                        custom_llm_provider=custom_llm_provider,
                        ssl_verify=llm_map.get("ssl_verify"),
                    ),
                    custom_llm_provider,
                )
                hops = (*hops, _completed_hop(hop_response))
            except _HOP_FAILURES as hop_error:
                is_blocked = is_recitation_block(hop_error)
                log_failed_hop(hop_error)
                break
        result: Final = transform_interactions_response_to_chat_response(
            raw_response=merge_hops(hops),
            model=model,
            model_response=model_response,
        )
        _mark_initial_prompt_tokens(result, hops)
        _attach_response_metadata(result, raw_response, wire.url)
        return _blocked_for_recitation(result) if is_blocked else result


interactions_api_bridge: Final = InteractionsToCompletionBridgeHandler()
