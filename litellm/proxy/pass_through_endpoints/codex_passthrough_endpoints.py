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
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client factory has untyped params
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.codex_model_catalog import (
    bundled_codex_models,
    codex_entry_names_default_reasoning_level,
    merged_codex_catalog,
    upstream_codex_slugs,
)
from litellm.proxy.common_utils.http_parsing_utils import (
    read_request_body,  # pyright: ignore[reportUnknownVariableType]  # returns a bare dict, validated at the call
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import pass_through_request
from litellm.proxy.response_api_endpoints.endpoints import responses_api
from litellm.responses.litellm_completion_transformation.reasoning_items import is_other_provider_reasoning_item
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.router import Deployment

if TYPE_CHECKING:
    from litellm.router import Router

router: Final = APIRouter()
_CHATGPT_PROVIDER: Final = "chatgpt"
_CHATGPT_CREDENTIAL_HEADERS: Final = frozenset({b"authorization", b"chatgpt-account-id"})
_MERGED_CATALOG_DROPPED_HEADERS: Final = frozenset({"etag", "content-length", "content-type"})
_MODELS_ETAG_HEADER: Final = "x-models-etag"
_LITELLM_RESPONSES_PATH: Final = "/v1/responses"
_CODEX_NO_DEFAULT_EFFORT: Final = "none"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_INPUT_ITEMS: Final = TypeAdapter(tuple[object, ...])
_CACHED_SLUGS: Final = TypeAdapter(frozenset[str])
_HEADER_DERIVED_STATE: Final = frozenset({"_cached_headers"})
_ACCOUNT_CATALOG_HEADERS: Final = frozenset({"authorization", "chatgpt-account-id", "originator", "user-agent"})
_CODEX_USER_AGENT_PRODUCT: Final = re.compile(r"[^/]+/(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?)")
_ACCOUNT_CATALOG_TIMEOUT_SECONDS: Final = 5.0
_SHARED_CACHE_TTL_SECONDS: Final = 3600
_ACCOUNT_SLUGS_CACHE_PREFIX: Final = "codex_account_slugs:"
_MERGED_CATALOG_CACHE_PREFIX: Final = "codex_catalog_added_litellm_rows:"
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


def _routes_to_litellm(model: object, llm_router: "Router | None", team_id: str | None) -> bool:
    if (
        not isinstance(model, str)
        or llm_router is None
        or model in bundled_codex_models()
        or f"{_CHATGPT_PROVIDER}/{model}" in litellm.model_cost  # pyright: ignore[reportUnknownMemberType]  # registry is dynamically typed
    ):
        return False
    candidates: Final = tuple(
        llm_router.get_deployment(model_id) for model_id in llm_router.get_candidate_model_ids_for_route(model, team_id)
    )
    deployments: Final = tuple(deployment for deployment in candidates if deployment is not None)
    names: Final = frozenset({model, llm_router.get_model_from_alias(model=model) or model})
    return bool(deployments) and all(_serves_by_name(deployment, names) for deployment in deployments)


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


def _account_slugs_cache_key(account_id: str) -> str:
    return f"{_ACCOUNT_SLUGS_CACHE_PREFIX}{hashlib.sha256(account_id.encode()).hexdigest()}"


def _cached_slugs(cached: object) -> frozenset[str] | None:
    if cached is None:
        return None
    try:
        return _CACHED_SLUGS.validate_python(cached)
    except ValidationError:
        return None


async def _remember_account_slugs(account_id: str, slugs: frozenset[str], cache: _SharedCache) -> None:
    await cache.async_set_cache(
        _account_slugs_cache_key(account_id), tuple(sorted(slugs)), ttl=_SHARED_CACHE_TTL_SECONDS
    )


def _codex_client_version(user_agent: str) -> str | None:
    product: Final = _CODEX_USER_AGENT_PRODUCT.fullmatch(user_agent.partition(" ")[0])
    return product.group(1) if product is not None else None


async def _fetched_account_slugs(headers: Mapping[str, str]) -> frozenset[str] | None:
    client_version: Final = _codex_client_version(headers.get("user-agent", ""))
    if client_version is None:
        return None
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
    account_id: Final = headers.get("chatgpt-account-id")
    if not account_id:
        return None
    cached: Final = _cached_slugs(await cache.async_get_cache(_account_slugs_cache_key(account_id)))
    if cached is not None:
        return cached
    fetched: Final = await _fetched_account_slugs(headers)
    if fetched is not None:
        await _remember_account_slugs(account_id, fetched, cache)
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
    return account_slugs is None or model not in account_slugs


def _merged_catalog_cache_key(token: str) -> str:
    return f"{_MERGED_CATALOG_CACHE_PREFIX}{token}"


async def _remember_catalog(
    headers: Mapping[str, str],
    slugs: frozenset[str],
    user_api_key_dict: UserAPIKeyAuth,
    added_litellm_rows: bool,
    cache: _SharedCache,
) -> None:
    account_id: Final = headers.get("chatgpt-account-id")
    if account_id:
        await _remember_account_slugs(account_id, slugs, cache)
    if user_api_key_dict.token is not None:
        await cache.async_set_cache(
            _merged_catalog_cache_key(user_api_key_dict.token), added_litellm_rows, ttl=_SHARED_CACHE_TTL_SECONDS
        )


async def _catalog_may_have_litellm_rows(user_api_key_dict: UserAPIKeyAuth, cache: _SharedCache) -> bool:
    if user_api_key_dict.token is None:
        return True
    return await cache.async_get_cache(_merged_catalog_cache_key(user_api_key_dict.token)) is not False


def _scope_state(request: Request) -> Mapping[str, object]:
    try:
        return _JSON_OBJECT.validate_python(request.scope.get("state"))
    except ValidationError:
        return MappingProxyType({})


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


def _with_provider_default_reasoning(
    body: Mapping[str, object], llm_router: "Router | None", team_id: str | None
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
        or codex_entry_names_default_reasoning_level(model, llm_router, team_id)
    ):
        return body
    return {**body, "reasoning": {key: value for key, value in reasoning.items() if key != "effort"}}


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


async def _merged_catalog(
    upstream_body: bytes, user_api_key_dict: UserAPIKeyAuth, llm_router: "Router | None"
) -> str | None:
    from litellm.proxy.proxy_server import codex_catalog_rows_for_key

    if llm_router is None or not _may_call_litellm_responses(user_api_key_dict):
        return None
    rows: Final = await codex_catalog_rows_for_key(user_api_key_dict)
    return merged_codex_catalog(
        upstream_body,
        tuple(row for row in rows if _routes_to_litellm(row.id, llm_router, user_api_key_dict.team_id)),
    )


async def _catalog(
    request: Request, user_api_key_dict: UserAPIKeyAuth, headers: Mapping[str, str], cache: _SharedCache
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
    merged: Final = await _merged_catalog(upstream_body, user_api_key_dict, llm_router)
    await _remember_catalog(headers, slugs, user_api_key_dict, merged is not None, cache)
    if merged is None:
        return upstream
    return Response(
        content=merged,
        status_code=upstream.status_code,
        headers={
            name: value for name, value in upstream.headers.items() if name not in _MERGED_CATALOG_DROPPED_HEADERS
        },
        media_type="application/json",
    )


async def _responses(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth,
    headers: Mapping[str, str],
    cache: _SharedCache,
) -> object:
    from litellm.proxy.proxy_server import llm_router

    body: Final = _JSON_OBJECT.validate_python(await read_request_body(request))
    if await _serves_through_litellm(body.get("model"), llm_router, user_api_key_dict, headers, cache):
        litellm_body: Final = _with_provider_default_reasoning(body, llm_router, user_api_key_dict.team_id)
        raw_body: Final = await request.body() if litellm_body is body else json.dumps(litellm_body).encode()
        return await responses_api(
            _as_litellm_responses_request(request, raw_body, litellm_body),
            fastapi_response,
            user_api_key_dict.model_copy(update={"request_route": _LITELLM_RESPONSES_PATH}),
        )
    kept_input: Final = _input_chatgpt_accepts(body)
    upstream: Final = await pass_through_request(
        request=request,
        target=f"{CHATGPT_API_BASE}/responses",
        custom_headers=dict(headers),
        user_api_key_dict=user_api_key_dict,
        custom_body={**body, "input": list(kept_input)} if kept_input is not None else None,
        custom_llm_provider=_CHATGPT_PROVIDER,
        upstream_headers_filter=lambda _: headers,
    )
    if await _catalog_may_have_litellm_rows(user_api_key_dict, cache):
        del upstream.headers[_MODELS_ETAG_HEADER]
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
    if endpoint == "models":
        return await _catalog(request, user_api_key_dict, headers, cache)
    return await _responses(request, fastapi_response, user_api_key_dict, headers, cache)
