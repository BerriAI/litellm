import hmac
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Final, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter, ValidationError
from starlette.types import Message

import litellm
from litellm.llms.chatgpt.common_utils import CHATGPT_API_BASE
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.codex_model_catalog import bundled_codex_models, merged_codex_catalog
from litellm.proxy.common_utils.http_parsing_utils import (
    read_request_body,  # pyright: ignore[reportUnknownVariableType]  # returns a bare dict, validated at the call
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import pass_through_request
from litellm.proxy.response_api_endpoints.endpoints import responses_api
from litellm.responses.litellm_completion_transformation.reasoning_items import is_other_provider_reasoning_item
from litellm.types.router import Deployment

if TYPE_CHECKING:
    from litellm.router import Router

router: Final = APIRouter()
_CHATGPT_PROVIDER: Final = "chatgpt"
_CHATGPT_CREDENTIAL_HEADERS: Final = frozenset({b"authorization", b"chatgpt-account-id"})
_MERGED_CATALOG_DROPPED_HEADERS: Final = frozenset({"etag", "content-length", "content-type"})
_MODELS_ETAG_HEADER: Final = "x-models-etag"
_LITELLM_RESPONSES_PATH: Final = "/v1/responses"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_INPUT_ITEMS: Final = TypeAdapter(tuple[object, ...])
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


def _as_litellm_responses_request(request: Request, body: bytes) -> Request:
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
        },
        receive,
    )


def _input_without_other_provider_reasoning(body: Mapping[str, object]) -> tuple[object, ...] | None:
    try:
        items: Final = _INPUT_ITEMS.validate_python(body.get("input"))
    except ValidationError:
        return None
    kept: Final = tuple(item for item in items if not is_other_provider_reasoning_item(item))
    return kept if len(kept) < len(items) else None


async def _catalog(request: Request, user_api_key_dict: UserAPIKeyAuth, headers: Mapping[str, str]) -> Response:
    from litellm.proxy.proxy_server import codex_catalog_rows_for_key, llm_router

    upstream: Final = await pass_through_request(
        request=request,
        target=f"{CHATGPT_API_BASE}/models",
        custom_headers=dict(headers),
        user_api_key_dict=user_api_key_dict,
        custom_llm_provider=_CHATGPT_PROVIDER,
        upstream_headers_filter=lambda _: headers,
    )
    if llm_router is None or upstream.status_code != 200 or isinstance(upstream, StreamingResponse):
        return upstream
    rows: Final = await codex_catalog_rows_for_key(user_api_key_dict)
    merged: Final = merged_codex_catalog(
        bytes(upstream.body),
        tuple(row for row in rows if _routes_to_litellm(row.id, llm_router, user_api_key_dict.team_id)),
    )
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
    request: Request, fastapi_response: Response, user_api_key_dict: UserAPIKeyAuth, headers: Mapping[str, str]
) -> object:
    from litellm.proxy.proxy_server import llm_router

    body: Final = _JSON_OBJECT.validate_python(await read_request_body(request))
    if _routes_to_litellm(body.get("model"), llm_router, user_api_key_dict.team_id):
        return await responses_api(
            _as_litellm_responses_request(request, await request.body()),
            fastapi_response,
            user_api_key_dict.model_copy(update={"request_route": _LITELLM_RESPONSES_PATH}),
        )
    kept_input: Final = _input_without_other_provider_reasoning(body)
    upstream: Final = await pass_through_request(
        request=request,
        target=f"{CHATGPT_API_BASE}/responses",
        custom_headers=dict(headers),
        user_api_key_dict=user_api_key_dict,
        custom_body={**body, "input": list(kept_input)} if kept_input is not None else None,
        custom_llm_provider=_CHATGPT_PROVIDER,
        upstream_headers_filter=lambda _: headers,
    )
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
        return await _catalog(request, user_api_key_dict, headers)
    return await _responses(request, fastapi_response, user_api_key_dict, headers)
