from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, NamedTuple, Protocol

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from litellm.llms.chatgpt.search.handler import ChatGPTSearchHandler
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.types.utils import LlmProviders

router: Final = APIRouter()
_FORWARDED_REQUEST_HEADERS: Final = ("originator", "x-codex-turn-metadata")
_FORWARDED_RESPONSE_HEADERS: Final = frozenset({"content-type", "openai-processing-ms", "retry-after", "x-request-id"})
_EMPTY_OBJECT_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})


class ChatGPTSearchTarget(NamedTuple):
    model: str
    api_base: str | None
    timeout: float | None


class ChatGPTSearchLitellmParams(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    model: str = Field(min_length=1)
    custom_llm_provider: str | None = None
    api_base: str | None = None
    timeout: float | int | None = None
    request_timeout: float | int | None = None


class ChatGPTSearchDeployment(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    litellm_params: ChatGPTSearchLitellmParams


class ChatGPTSearchRequest(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)

    id: str | None = None
    model: str = Field(min_length=1)


class DeploymentRouter(Protocol):
    async def async_get_available_deployment(
        self,
        model: str,
        request_kwargs: dict[str, object],  # mutable-ok: matches the Router's mutating request context contract
    ) -> object: ...


def _positive_timeout(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def target_from_litellm_params(
    requested_model: str,
    litellm_params: ChatGPTSearchLitellmParams,
    default_api_base: str | None,
    default_timeout: float | None,
) -> ChatGPTSearchTarget:
    configured_model: Final = litellm_params.model
    configured_provider: Final = litellm_params.custom_llm_provider
    provider_prefix, separator, unprefixed_model = configured_model.partition("/")
    provider: Final = configured_provider or (provider_prefix if separator else "unknown")
    is_chatgpt_target: Final = configured_provider in (None, LlmProviders.CHATGPT.value) and (
        (separator and provider_prefix == LlmProviders.CHATGPT.value)
        or (not separator and configured_provider == LlmProviders.CHATGPT.value)
    )
    if not is_chatgpt_target:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model `{requested_model}` resolves to provider `{provider}`; "
                "the alpha search endpoint requires a ChatGPT subscription model"
            ),
        )
    model: Final = unprefixed_model if separator else configured_model
    if not model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model `{requested_model}` resolves to an empty ChatGPT provider model",
        )

    timeout: Final = (
        _positive_timeout(litellm_params.timeout)
        or _positive_timeout(litellm_params.request_timeout)
        or default_timeout
    )
    return ChatGPTSearchTarget(
        model=model,
        api_base=litellm_params.api_base or default_api_base,
        timeout=timeout,
    )


async def resolve_chatgpt_search_target(
    requested_model: str,
    llm_router: DeploymentRouter | None,
    user_api_key_dict: UserAPIKeyAuth,
    user_model: str | None,
    user_api_base: str | None,
    user_request_timeout: float | None,
) -> ChatGPTSearchTarget:
    default_timeout: Final = _positive_timeout(user_request_timeout)
    if llm_router is None:
        if not user_model:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "The alpha search endpoint requires a configured ChatGPT model; "
                    "start the proxy with --model or configure model_list"
                ),
            )
        return target_from_litellm_params(
            requested_model=requested_model,
            litellm_params=ChatGPTSearchLitellmParams(model=user_model, api_base=user_api_base),
            default_api_base=user_api_base,
            default_timeout=default_timeout,
        )

    team_metadata: Final[Mapping[str, object]] = (
        MappingProxyType({"user_api_key_team_id": user_api_key_dict.team_id})
        if user_api_key_dict.team_id is not None
        else _EMPTY_OBJECT_MAPPING
    )
    metadata: Final[Mapping[str, object]] = MappingProxyType({"user_api_key_auth": user_api_key_dict}) | team_metadata
    region_params: Final[Mapping[str, object]] = (
        MappingProxyType({"allowed_model_region": user_api_key_dict.allowed_model_region})
        if user_api_key_dict.allowed_model_region is not None
        else _EMPTY_OBJECT_MAPPING
    )
    request_kwargs: Final = MappingProxyType({"metadata": metadata}) | region_params
    deployment_value: Final = await llm_router.async_get_available_deployment(
        model=requested_model,
        request_kwargs=request_kwargs,
    )
    try:
        deployment: Final = ChatGPTSearchDeployment.model_validate(deployment_value)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Router returned an invalid deployment for model `{requested_model}`",
        ) from exc
    return target_from_litellm_params(
        requested_model=requested_model,
        litellm_params=deployment.litellm_params,
        default_api_base=user_api_base,
        default_timeout=default_timeout,
    )


def _request_headers(request: Request) -> Mapping[str, str]:
    return MappingProxyType(
        {name: value for name in _FORWARDED_REQUEST_HEADERS if (value := request.headers.get(name)) is not None}
    )


def _response_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            name: value
            for name, value in headers.items()
            if name.lower() in _FORWARDED_RESPONSE_HEADERS or name.lower().startswith("x-codex-")
        }
    )


@router.post("/v1/alpha/search")
@router.post("/alpha/search")
async def chatgpt_alpha_search(
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    from litellm.proxy import proxy_server

    try:
        data: Final = ChatGPTSearchRequest.model_validate(await request.json())
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The alpha search request requires a JSON object with a non-empty string `model`",
        ) from exc

    target: Final = await resolve_chatgpt_search_target(
        requested_model=data.model,
        llm_router=proxy_server.llm_router,
        user_api_key_dict=user_api_key_dict,
        user_model=proxy_server.user_model,
        user_api_base=proxy_server.user_api_base,
        user_request_timeout=proxy_server.user_request_timeout,
    )
    routed_data: Final = data.model_copy(update=MappingProxyType({"model": target.model}))
    payload: Final = orjson.dumps(routed_data.model_dump(exclude_unset=True))
    upstream_response: Final = await ChatGPTSearchHandler().search(
        payload=payload,
        model=target.model,
        session_id=data.id,
        api_base=target.api_base,
        extra_headers=_request_headers(request),
        timeout=target.timeout,
    )
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_response_headers(upstream_response.headers),
    )
