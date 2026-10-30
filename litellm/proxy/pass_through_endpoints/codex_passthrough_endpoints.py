import hashlib
import hmac
import json
import re
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter, ValidationError
from starlette.types import Message

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params, normalize_drop_params
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client factory has untyped params
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import check_model_access_helper
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.codex_model_catalog import (
    CodexCatalogRow,
    bundled_codex_models,
    codex_catalog_etag,
    codex_entry_names_default_reasoning_level,
    merged_codex_catalog,
    upstream_codex_slugs,
)
from litellm.proxy.common_utils.http_parsing_utils import (
    get_client_requested_model,
    read_request_body,  # pyright: ignore[reportUnknownVariableType]  # returns a bare dict, validated at the call
)
from litellm.proxy.litellm_pre_call_utils import should_auto_drop_params_for_agentic_cli
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import pass_through_request
from litellm.proxy.response_api_endpoints.endpoints import responses_api
from litellm.responses.additional_tools import with_additional_tools_hoisted
from litellm.responses.litellm_completion_transformation.reasoning_items import is_other_provider_reasoning_item
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.router import Deployment
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, supports_none_reasoning_effort

if TYPE_CHECKING:
    from litellm.router import Router

router: Final = APIRouter()
_CHATGPT_PROVIDER: Final = "chatgpt"
_CHATGPT_CREDENTIAL_HEADERS: Final = frozenset({b"authorization", b"chatgpt-account-id"})
_MERGED_CATALOG_DROPPED_HEADERS: Final = frozenset({"etag", "content-length", "content-type"})
_ETAG_HEADER: Final = "etag"
_MODELS_ETAG_HEADER: Final = "x-models-etag"
_ENCRYPTED_CONTENT_AFFINITY_FLAG: Final = "encrypted_content_affinity_enabled"
_LITELLM_RESPONSES_PATH: Final = "/v1/responses"
_CODEX_NO_DEFAULT_EFFORT: Final = "none"
_NONE_EFFORT_SUPPORT_FLAG: Final = "supports_none_reasoning_effort"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_INPUT_ITEMS: Final = TypeAdapter(tuple[object, ...])
_CACHED_SLUGS: Final = TypeAdapter(frozenset[str])
_CACHED_MODELS_ETAG: Final = TypeAdapter(dict[Literal["etag"], str])
_MODEL_NAMES: Final = TypeAdapter(list[str])
_HEADER_DERIVED_STATE: Final = frozenset({"_cached_headers"})
_ACCOUNT_CATALOG_HEADERS: Final = frozenset({"authorization", "chatgpt-account-id", "originator", "user-agent"})
_CODEX_USER_AGENT_PRODUCT: Final = re.compile(r"([^/]+)/(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?)")
_ACCOUNT_CATALOG_TIMEOUT_SECONDS: Final = 5.0
_SHARED_CACHE_TTL_SECONDS: Final = 3600
_ACCOUNT_SLUGS_UNAVAILABLE: Final = "unavailable"
_ACCOUNT_SLUGS_UNAVAILABLE_TTL_SECONDS: Final = 60
_ACCOUNT_SLUGS_CACHE_PREFIX: Final = "codex_account_slugs:"
_MODELS_ETAG_CACHE_PREFIX: Final = "codex_models_etag:"
_MODELS_ETAG_TTL_SECONDS: Final = 300
_OPENAI_API_PROVIDERS: Final = frozenset({LlmProviders.OPENAI.value, LlmProviders.AZURE.value})
_CODEX_TOOL_PROVIDERS: Final = frozenset(
    {*_OPENAI_API_PROVIDERS, LlmProviders.FIREWORKS_AI.value, LlmProviders.BEDROCK_MANTLE.value}
)
_CHAT_COMPLETIONS_BRIDGE: Final = "use_chat_completions_api"
_HEADERS: Final = frozenset(
    {
        "authorization",
        "chatgpt-account-id",
        "content-type",
        "accept",
        "user-agent",
        "originator",
        "openai-beta",
        "session-id",
        "thread-id",
        "session_id",
        "conversation_id",
        "x-codex-turn-metadata",
        "x-codex-turn-state",
    }
)


def _is_chatgpt(deployment: Deployment) -> bool:
    return (
        deployment.litellm_params.custom_llm_provider == _CHATGPT_PROVIDER
        or deployment.litellm_params.model.partition("/")[0] == _CHATGPT_PROVIDER
    )


def _serves_by_name(deployment: Deployment, names: frozenset[str]) -> bool:
    return not _is_chatgpt(deployment) and (
        deployment.model_name in names or deployment.model_info.team_public_model_name in names
    )


def _candidate_deployments(model: str, llm_router: "Router", team_id: str | None) -> tuple[Deployment, ...]:
    registry: Final = llm_router.registry()
    candidates: Final = tuple(
        registry.deployment(model_id) for model_id in registry.names(model, kind="route_candidates", team_id=team_id)
    )
    return tuple(deployment for deployment in candidates if deployment is not None)


def _routes_to_litellm(model: object, llm_router: "Router | None", team_id: str | None) -> bool:
    if (
        not isinstance(model, str)
        or llm_router is None
        or model in bundled_codex_models()
        or f"{_CHATGPT_PROVIDER}/{model}" in litellm.model_cost  # pyright: ignore[reportUnknownMemberType]  # registry is dynamically typed
    ):
        return False
    deployments: Final = _candidate_deployments(model, llm_router, team_id)
    names: Final = frozenset({model, llm_router.registry().resolve(model, kind="alias") or model})
    return bool(deployments) and all(_serves_by_name(deployment, names) for deployment in deployments)


def _key_and_team_allow(model: str, llm_router: "Router", user_api_key_dict: UserAPIKeyAuth) -> bool:
    allowed_lists: Final = (
        _MODEL_NAMES.validate_python(user_api_key_dict.models),
        _MODEL_NAMES.validate_python(user_api_key_dict.team_models),
    )
    return all(
        check_model_access_helper(model, llm_router, allowed, team_id=user_api_key_dict.team_id)
        for allowed in allowed_lists
    )


def native_chatgpt_model(model: str, llm_router: "Router", user_api_key_dict: UserAPIKeyAuth) -> str | None:
    if model.startswith(f"{_CHATGPT_PROVIDER}/") or _key_and_team_allow(model, llm_router, user_api_key_dict):
        return None
    chatgpt_model: Final = f"{_CHATGPT_PROVIDER}/{model}"
    deployments: Final = _candidate_deployments(chatgpt_model, llm_router, user_api_key_dict.team_id)
    if not deployments or not all(_is_chatgpt(deployment) for deployment in deployments):
        return None
    return chatgpt_model if _key_and_team_allow(chatgpt_model, llm_router, user_api_key_dict) else None


class _SharedCache(Protocol):
    async def async_get_cache(self, key: str) -> object: ...

    async def async_set_cache(self, key: str, value: object, *, ttl: int) -> None: ...


def codex_shared_cache() -> _SharedCache:
    from litellm.proxy.proxy_server import proxy_logging_obj

    return proxy_logging_obj.internal_usage_cache.dual_cache


def _may_call_litellm_responses(user_api_key_dict: UserAPIKeyAuth) -> bool:
    try:
        _ = RouteChecks.should_call_route(route=_LITELLM_RESPONSES_PATH, valid_token=user_api_key_dict)
    except HTTPException:
        return False
    return True


def _account_slugs_cache_key(headers: Mapping[str, str]) -> str | None:
    account_id: Final = headers.get("chatgpt-account-id")
    if not account_id:
        return None
    caller: Final = json.dumps([account_id, headers.get("authorization", "")])
    return f"{_ACCOUNT_SLUGS_CACHE_PREFIX}{hashlib.sha256(caller.encode()).hexdigest()}"


def _cached_slugs(cached: object) -> frozenset[str] | None:
    if cached is None:
        return None
    try:
        return _CACHED_SLUGS.validate_python(cached)
    except ValidationError:
        return None


async def _remember_account_slugs(cache_key: str, slugs: frozenset[str] | None, cache: _SharedCache) -> None:
    if slugs is None:
        await cache.async_set_cache(cache_key, _ACCOUNT_SLUGS_UNAVAILABLE, ttl=_ACCOUNT_SLUGS_UNAVAILABLE_TTL_SECONDS)
        return
    await cache.async_set_cache(cache_key, tuple(sorted(slugs)), ttl=_SHARED_CACHE_TTL_SECONDS)


def _codex_client_version(headers: Mapping[str, str]) -> str | None:
    product: Final = _CODEX_USER_AGENT_PRODUCT.fullmatch(headers.get("user-agent", "").partition(" ")[0])
    if product is None or product.group(1) != headers.get("originator"):
        return None
    return product.group(2)


async def _fetched_account_slugs(headers: Mapping[str, str], client_version: str) -> frozenset[str] | None:
    client: Final = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PassThroughEndpoint,
        params={"timeout": _ACCOUNT_CATALOG_TIMEOUT_SECONDS},
    )
    try:
        response: Final = await client.client.get(
            f"{CHATGPT_API_BASE}/models",
            params={"client_version": client_version},
            headers={name: value for name, value in headers.items() if name in _ACCOUNT_CATALOG_HEADERS},
        )
    except httpx.HTTPError:
        return None
    return upstream_codex_slugs(response.content) if response.status_code == 200 else None


async def _account_slugs(headers: Mapping[str, str], cache: _SharedCache) -> frozenset[str] | None:
    cache_key: Final = _account_slugs_cache_key(headers)
    if cache_key is None:
        return None
    cached_value: Final = await cache.async_get_cache(cache_key)
    if cached_value == _ACCOUNT_SLUGS_UNAVAILABLE:
        return None
    cached: Final = _cached_slugs(cached_value)
    if cached is not None:
        return cached
    client_version: Final = _codex_client_version(headers)
    if client_version is None:
        return None
    fetched: Final = await _fetched_account_slugs(headers, client_version)
    await _remember_account_slugs(cache_key, fetched, cache)
    return fetched


async def _serves_through_litellm(
    model: object,
    llm_router: "Router | None",
    user_api_key_dict: UserAPIKeyAuth,
    headers: Mapping[str, str],
    cache: _SharedCache,
) -> bool:
    if not _routes_to_litellm(model, llm_router, user_api_key_dict.team_id) or not _may_call_litellm_responses(
        user_api_key_dict
    ):
        return False
    account_slugs: Final = await _account_slugs(headers, cache)
    return account_slugs is not None and model not in account_slugs


def _json_object(value: object) -> Mapping[str, object]:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return MappingProxyType({})


def _scope_state(request: Request) -> Mapping[str, object]:
    return _json_object(request.scope.get("state"))


def _as_litellm_responses_request(request: Request, body: bytes, parsed_body: Mapping[str, object]) -> Request:
    replayed: Final[Iterator[Message]] = iter(({"type": "http.request", "body": body, "more_body": False},))

    async def receive() -> Message:
        return next(replayed, None) or await request.receive()

    return Request(
        {
            **request.scope,
            "path": _LITELLM_RESPONSES_PATH,
            "raw_path": _LITELLM_RESPONSES_PATH.encode(),
            "headers": [
                (name, value) for name, value in request.headers.raw if name not in _CHATGPT_CREDENTIAL_HEADERS
            ],
            "parsed_body": (tuple(parsed_body), parsed_body),
            "state": {
                name: value for name, value in _scope_state(request).items() if name not in _HEADER_DERIVED_STATE
            },
        },
        receive,
    )


def _drops_params(body: Mapping[str, object], model: str, llm_router: "Router", team_id: str | None) -> bool:
    deployments: Final = _candidate_deployments(model, llm_router, team_id)
    return drops_unsupported_params(body.get("drop_params")) or (
        bool(deployments)
        and all(normalize_drop_params(deployment.litellm_params.drop_params) is True for deployment in deployments)
    )


def _deployment_supports_no_effort(deployment: Deployment) -> bool:
    configured: Final = _json_object(deployment.model_info.model_extra).get(_NONE_EFFORT_SUPPORT_FLAG)
    if isinstance(configured, bool):
        return configured
    return supports_none_reasoning_effort(
        model=deployment.litellm_params.model, custom_llm_provider=deployment.litellm_params.custom_llm_provider
    )


def _supports_no_effort(model: str, llm_router: "Router", team_id: str | None) -> bool:
    deployments: Final = _candidate_deployments(model, llm_router, team_id)
    return bool(deployments) and all(_deployment_supports_no_effort(deployment) for deployment in deployments)


def _with_provider_default_reasoning(
    body: Mapping[str, object], llm_router: "Router | None", team_id: str | None, client_version: str | None
) -> Mapping[str, object]:
    model: Final = body.get("model")
    try:
        reasoning: Final = _JSON_OBJECT.validate_python(body.get("reasoning"))
    except ValidationError:
        return body
    if (
        reasoning.get("effort") != _CODEX_NO_DEFAULT_EFFORT
        or not isinstance(model, str)
        or llm_router is None
        or not _drops_params(body, model, llm_router, team_id)
        or codex_entry_names_default_reasoning_level(model, llm_router, team_id, client_version)
        or _supports_no_effort(model, llm_router, team_id)
    ):
        return body
    return {**body, "reasoning": {key: value for key, value in reasoning.items() if key != "effort"}}


def _with_wrapped_reasoning(body: Mapping[str, object]) -> Mapping[str, object]:
    return {
        **body,
        "litellm_metadata": {**_json_object(body.get("litellm_metadata")), _ENCRYPTED_CONTENT_AFFINITY_FLAG: True},
    }


def _with_reasoning_the_targets_can_decrypt(
    body: Mapping[str, object], model: str, llm_router: "Router", deployments: tuple[Deployment, ...]
) -> Mapping[str, object]:
    try:
        items: Final = _INPUT_ITEMS.validate_python(body.get("input"))
    except ValidationError:
        return body
    kept: Final = llm_router.registry().decryptable_reasoning(model, items, deployments)
    if body.get("store") is True:
        return {**body, "input": list(kept)}
    resolvable: Final = (_without_unpersisted_reasoning_reference(item) for item in kept)
    return {**body, "input": [item for item in resolvable if item is not None]}


def _without_unpersisted_reasoning_reference(item: object) -> object | None:
    if not _is_unpersisted_reasoning_reference(item):
        return item
    return ResponsesAPIRequestUtils.readable_reasoning(_JSON_OBJECT.validate_python(item))


def _model_and_provider(deployment: Deployment) -> tuple[str, str] | None:
    try:
        model, provider, _, _ = get_llm_provider(
            model=deployment.litellm_params.model, custom_llm_provider=deployment.litellm_params.custom_llm_provider
        )
    except BadRequestError:
        return None
    return model, provider


def _is_openai_api(deployment: Deployment) -> bool:
    resolved: Final = _model_and_provider(deployment)
    return resolved is not None and resolved[1] in _OPENAI_API_PROVIDERS


def _takes_codex_tools(deployment: Deployment) -> bool:
    resolved: Final = _model_and_provider(deployment)
    if resolved is None:
        return False
    model, provider = resolved
    return (
        provider in _CODEX_TOOL_PROVIDERS
        or ProviderConfigManager.get_provider_responses_api_config(
            provider=provider, model=model, api_base=deployment.litellm_params.api_base
        )
        is None
    )


def _with_tools_the_targets_accept(
    body: Mapping[str, object], deployments: tuple[Deployment, ...], drop_params: bool
) -> Mapping[str, object]:
    if all(_is_openai_api(deployment) for deployment in deployments):
        return body
    if all(_takes_codex_tools(deployment) for deployment in deployments):
        return with_additional_tools_hoisted(body, drop_params=drop_params)
    return {**body, _CHAT_COMPLETIONS_BRIDGE: True}


def _drops_params_for_client(
    body: Mapping[str, object], model: str, llm_router: "Router", team_id: str | None, user_agent: str
) -> bool:
    from litellm.proxy.proxy_server import proxy_config

    return _drops_params(body, model, llm_router, team_id) or should_auto_drop_params_for_agentic_cli(
        user_agent, "drop_params" in body, proxy_config
    )


def _litellm_body(
    body: Mapping[str, object], llm_router: "Router | None", team_id: str | None, headers: Mapping[str, str]
) -> Mapping[str, object]:
    wrapped: Final = _with_wrapped_reasoning(
        _with_provider_default_reasoning(body, llm_router, team_id, _codex_client_version(headers))
    )
    model: Final = body.get("model")
    if not isinstance(model, str) or llm_router is None:
        return wrapped
    deployments: Final = _candidate_deployments(model, llm_router, team_id)
    return _with_tools_the_targets_accept(
        _with_reasoning_the_targets_can_decrypt(wrapped, model, llm_router, deployments),
        deployments,
        _drops_params_for_client(body, model, llm_router, team_id, headers.get("user-agent", "")),
    )


def _is_unpersisted_reasoning_reference(item: object) -> bool:
    try:
        fields: Final = _JSON_OBJECT.validate_python(item)
    except ValidationError:
        return False
    item_id: Final = fields.get("id")
    encrypted_content: Final = fields.get("encrypted_content")
    return (
        fields.get("type") == "reasoning"
        and isinstance(item_id, str)
        and bool(item_id)
        and not (isinstance(encrypted_content, str) and bool(encrypted_content))
    )


def _rejected_by_chatgpt(item: object, stored: bool) -> bool:
    return is_other_provider_reasoning_item(item) or (not stored and _is_unpersisted_reasoning_reference(item))


def _input_chatgpt_accepts(body: Mapping[str, object]) -> tuple[object, ...] | None:
    try:
        items: Final = _INPUT_ITEMS.validate_python(body.get("input"))
    except ValidationError:
        return None
    stored: Final = body.get("store") is True
    kept: Final = tuple(item for item in items if not _rejected_by_chatgpt(item, stored))
    return kept if len(kept) < len(items) else None


def _chatgpt_body(body: Mapping[str, object], client_model: str | None) -> Mapping[str, object] | None:
    kept_input: Final = _input_chatgpt_accepts(body)
    restored_model: Final = client_model is not None and body.get("model") == f"{_CHATGPT_PROVIDER}/{client_model}"
    if kept_input is None and not restored_model:
        return None
    return {
        **body,
        **({"input": list(kept_input)} if kept_input is not None else {}),
        **({"model": client_model} if restored_model else {}),
    }


async def _litellm_catalog_rows(
    user_api_key_dict: UserAPIKeyAuth, llm_router: "Router | None"
) -> tuple[CodexCatalogRow, ...]:
    from litellm.proxy.proxy_server import codex_catalog_rows_for_key

    if llm_router is None or not _may_call_litellm_responses(user_api_key_dict):
        return ()
    try:
        rows: Final = await codex_catalog_rows_for_key(user_api_key_dict)
    except Exception as error:  # noqa: BLE001  # a failed listing must never fail the ChatGPT answer it extends
        verbose_proxy_logger.warning(
            "Codex catalog keeps only ChatGPT's models: listing the key's LiteLLM models raised %s",
            type(error).__name__,
        )
        return ()
    return tuple(row for row in rows if _routes_to_litellm(row.id, llm_router, user_api_key_dict.team_id))


def _models_etag_caller(user_api_key_dict: UserAPIKeyAuth, gateway_token: str) -> str:
    return user_api_key_dict.token or hashlib.sha256(gateway_token.encode()).hexdigest()


def _models_etag_cache_key(caller: str, upstream_etag: str) -> str:
    return f"{_MODELS_ETAG_CACHE_PREFIX}{hashlib.sha256(json.dumps([caller, upstream_etag]).encode()).hexdigest()}"


async def _remembered_models_etag(
    caller: str, upstream_etag: str, rows: tuple[CodexCatalogRow, ...], cache: _SharedCache
) -> str:
    etag: Final = codex_catalog_etag(upstream_etag, rows) if rows else upstream_etag
    await cache.async_set_cache(
        _models_etag_cache_key(caller, upstream_etag), {"etag": etag}, ttl=_MODELS_ETAG_TTL_SECONDS
    )
    return etag


async def _cached_models_etag(caller: str, upstream_etag: str, cache: _SharedCache) -> str | None:
    try:
        return _CACHED_MODELS_ETAG.validate_python(
            await cache.async_get_cache(_models_etag_cache_key(caller, upstream_etag))
        )["etag"]
    except ValidationError:
        return None


async def _key_models_etag(
    upstream_etag: str,
    user_api_key_dict: UserAPIKeyAuth,
    caller: str,
    llm_router: "Router | None",
    cache: _SharedCache,
) -> str:
    cached: Final = await _cached_models_etag(caller, upstream_etag, cache)
    if cached is not None:
        return cached
    rows: Final = await _litellm_catalog_rows(user_api_key_dict, llm_router)
    return await _remembered_models_etag(caller, upstream_etag, rows, cache)


def _with_etag(response: Response, content: bytes | str, etag: str | None) -> Response:
    kept_headers: Final = {
        name: value for name, value in response.headers.items() if name not in _MERGED_CATALOG_DROPPED_HEADERS
    }
    return Response(
        content=content,
        status_code=response.status_code,
        headers={**kept_headers, **({_ETAG_HEADER: etag} if etag is not None else {})},
        media_type="application/json",
    )


async def _catalog(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth,
    caller: str,
    headers: Mapping[str, str],
    cache: _SharedCache,
) -> Response:
    from litellm.proxy.proxy_server import llm_router

    upstream: Final = await pass_through_request(
        request=request,
        target=f"{CHATGPT_API_BASE}/models",
        custom_headers=dict(headers),
        user_api_key_dict=user_api_key_dict,
        custom_llm_provider=_CHATGPT_PROVIDER,
        upstream_headers_filter=lambda _: headers,
    )
    if upstream.status_code != 200 or isinstance(upstream, StreamingResponse):
        return upstream
    upstream_body: Final = bytes(upstream.body)
    slugs: Final = upstream_codex_slugs(upstream_body)
    if slugs is None:
        return upstream
    account_slugs_key: Final = _account_slugs_cache_key(headers)
    if account_slugs_key is not None:
        await _remember_account_slugs(account_slugs_key, slugs, cache)
    rows: Final = await _litellm_catalog_rows(user_api_key_dict, llm_router)
    upstream_etag: Final = upstream.headers.get(_ETAG_HEADER)
    etag: Final = None if upstream_etag is None else await _remembered_models_etag(caller, upstream_etag, rows, cache)
    if not rows:
        return upstream
    merged: Final = merged_codex_catalog(upstream_body, rows, client_version=request.query_params.get("client_version"))
    return _with_etag(upstream, merged if merged is not None else upstream_body, etag)


async def _responses(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    caller: str,
    headers: Mapping[str, str],
    cache: _SharedCache,
) -> object:
    from litellm.proxy.proxy_server import llm_router

    body: Final = _JSON_OBJECT.validate_python(await read_request_body(request))
    if await _serves_through_litellm(body.get("model"), llm_router, user_api_key_dict, headers, cache):
        litellm_body: Final = _litellm_body(body, llm_router, user_api_key_dict.team_id, headers)
        return await responses_api(
            _as_litellm_responses_request(request, json.dumps(litellm_body).encode(), litellm_body),
            fastapi_response,
            user_api_key_dict.model_copy(update={"request_route": _LITELLM_RESPONSES_PATH}),
        )
    chatgpt_body: Final = _chatgpt_body(body, get_client_requested_model(request))
    upstream: Final = await pass_through_request(
        request=request,
        target=f"{CHATGPT_API_BASE}/responses",
        custom_headers=dict(headers),
        user_api_key_dict=user_api_key_dict,
        custom_body=dict(chatgpt_body) if chatgpt_body is not None else None,
        custom_llm_provider=_CHATGPT_PROVIDER,
        upstream_headers_filter=lambda _: headers,
    )
    upstream_models_etag: Final = upstream.headers.get(_MODELS_ETAG_HEADER)
    if upstream_models_etag is not None:
        upstream.headers[_MODELS_ETAG_HEADER] = await _key_models_etag(
            upstream_models_etag, user_api_key_dict, caller, llm_router, cache
        )
    return upstream


@router.get("/backend-api/codex/{endpoint}", tags=["ChatGPT Pass-through", "pass-through"], response_model=None)
@router.post(
    "/backend-api/codex/{endpoint}",
    tags=["ChatGPT Pass-through", "pass-through"],
    response_model=None,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"type": "object", "additionalProperties": True}}},
        }
    },
    responses={
        200: {
            "description": "Codex response, the model catalog with the key's LiteLLM models, or server-sent events",
            "content": {
                "application/json": {"schema": {"type": "object", "additionalProperties": True}},
                "text/event-stream": {"schema": {"type": "string"}},
            },
        }
    },
)
async def codex_passthrough_route(
    endpoint: Literal["models", "responses"],
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    cache: Annotated[_SharedCache, Depends(codex_shared_cache)],
) -> object:
    """Forward a client's ChatGPT subscription with separate LiteLLM authentication, listing and serving the
    key's other chat models through LiteLLM."""
    expected_method: Final = "GET" if endpoint == "models" else "POST"
    if request.method != expected_method:
        raise HTTPException(status_code=405, detail="Method not allowed", headers={"Allow": expected_method})

    authorization: Final = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    gateway_key: Final = request.headers.get("x-litellm-api-key", "")
    gateway_parts: Final = gateway_key.split()
    gateway_token: Final = (
        gateway_parts[-1]
        if len(gateway_parts) == 1 or (len(gateway_parts) == 2 and gateway_parts[0].lower() == "bearer")
        else ""
    )
    if scheme.lower() != "bearer" or not token.strip() or not gateway_token:
        raise HTTPException(
            status_code=401,
            detail="Send the ChatGPT bearer in Authorization and the LiteLLM key in x-litellm-api-key",
        )
    if hmac.compare_digest(token.strip().encode(), gateway_token.encode()):
        raise HTTPException(status_code=401, detail="ChatGPT and LiteLLM credentials must be separate")

    headers: Final = MappingProxyType({name: value for name, value in request.headers.items() if name in _HEADERS})
    caller: Final = _models_etag_caller(user_api_key_dict, gateway_token)
    if endpoint == "models":
        return await _catalog(request, user_api_key_dict, caller, headers, cache)
    return await _responses(request, fastapi_response, user_api_key_dict, caller, headers, cache)
