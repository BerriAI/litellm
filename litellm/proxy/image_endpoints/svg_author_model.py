import json
from collections.abc import Iterator, Mapping
from itertools import chain
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.constants import SVG_AUTHOR_DEPLOYMENT_METADATA_KEY
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _is_dashboard_session,
    can_agent_call_model,
    can_key_call_resolved_model,
    can_user_call_model,
    get_user_object,
    key_model_aliases_for_auth_check,
    personal_key_team_model_access_applies,
)
from litellm.proxy.litellm_pre_call_utils import customer_request_auth, pre_call_served_model
from litellm.types.proxy.auth.auth_checks import UserNotFoundError
from litellm.types.utils import ImageSVGAuthorDeployment

if TYPE_CHECKING:
    from litellm.router import Router

_SVG_FIELD: Final = "svg"
_SVG_FORM_MODEL_FIELD: Final = "svg[model]"
_EXTRA_BODY_FIELD: Final = "extra_body"
_ROUTER_SETTINGS_FIELD: Final = "router_settings_override"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class _AuthorParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str


class _AuthorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | None = None


class _AuthorDeployment(BaseModel):
    model_config = ConfigDict(frozen=True)

    litellm_params: _AuthorParams
    model_info: _AuthorInfo = _AuthorInfo()


def _as_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_json(value) if isinstance(value, str) else _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _author_of(svg: object) -> str | None:
    model: Final = (_as_object(svg) or {}).get("model")
    return model if isinstance(model, str) and model else None


def _containers(body: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    yield body
    yield _as_object(body.get(_EXTRA_BODY_FIELD)) or {}


def _authors_in(container: Mapping[str, object]) -> Iterator[str | None]:
    yield _author_of(container.get(_SVG_FIELD))
    form_model: Final = container.get(_SVG_FORM_MODEL_FIELD)
    yield form_model if isinstance(form_model, str) else None


def _svg_author_models(body: Mapping[str, object]) -> tuple[str, ...]:
    named: Final = chain.from_iterable(_authors_in(container) for container in _containers(body))
    return tuple(dict.fromkeys(name for name in named if name))


def _encoded_like(original: object, value: Mapping[str, object]) -> object:
    return json.dumps(value) if isinstance(original, str) else dict(value)


def _with_author(svg: object, upstream: str) -> object:
    parsed: Final = _as_object(svg)
    if parsed is None or _author_of(parsed) is None:
        return svg
    return _encoded_like(svg, {**parsed, "model": upstream})


def _rewritten_svg_fields(container: Mapping[str, object], upstream: str) -> Mapping[str, object]:
    return {
        **({_SVG_FIELD: _with_author(container[_SVG_FIELD], upstream)} if _SVG_FIELD in container else {}),
        **({_SVG_FORM_MODEL_FIELD: upstream} if isinstance(container.get(_SVG_FORM_MODEL_FIELD), str) else {}),
    }


def _rewritten_body_fields(body: Mapping[str, object], upstream: str) -> Mapping[str, object]:
    extra_body: Final = body.get(_EXTRA_BODY_FIELD)
    parsed_extra_body: Final = _as_object(extra_body)
    extra_body_fields: Final = {} if parsed_extra_body is None else _rewritten_svg_fields(parsed_extra_body, upstream)
    return {
        **_rewritten_svg_fields(body, upstream),
        **(
            {_EXTRA_BODY_FIELD: _encoded_like(extra_body, {**(parsed_extra_body or {}), **extra_body_fields})}
            if extra_body_fields
            else {}
        ),
    }


def _author_deployment(author: str, team_id: str | None, llm_router: "Router | None") -> _AuthorDeployment | None:
    deployments: Final = (llm_router.registry().deployments(author, team_id=team_id) or ()) if llm_router else ()
    return _AuthorDeployment.model_validate(deployments[0]) if deployments else None


def _provider_model(deployment_model: str) -> str:
    try:
        return litellm.get_llm_provider(model=deployment_model)[0]
    except litellm.BadRequestError:
        return deployment_model


async def _check_key_owner_access(model: str, user_api_key_dict: UserAPIKeyAuth, llm_router: "Router | None") -> None:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    await can_agent_call_model(
        model=model,
        valid_token=user_api_key_dict,
        llm_router=llm_router,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
    )
    if user_api_key_dict.team_id is not None or user_api_key_dict.user_id is None:
        return
    if prisma_client is None or (
        _is_dashboard_session(user_api_key_dict) and personal_key_team_model_access_applies(user_api_key_dict)
    ):
        return
    try:
        user_object: Final = await get_user_object(
            user_id=user_api_key_dict.user_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            user_id_upsert=False,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
    except UserNotFoundError:
        return
    await can_user_call_model(
        model=model,
        llm_router=llm_router,
        user_object=user_object,
        key_model_aliases=key_model_aliases_for_auth_check(user_api_key_dict),
    )


async def svg_author_model_updates(
    body: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router | None",
    metadata_variable_name: str,
) -> Mapping[str, object]:
    """The request fields to replace so ``svg.model`` is authorized, sent by its provider name, and billed by its deployment."""
    authors: Final = _svg_author_models(body)
    if not authors:
        return {}
    if len(authors) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"svg.model is set more than once with different values: {', '.join(authors)}",
        )
    author: Final = authors[0]
    router_settings: Final = _as_object(body.get(_ROUTER_SETTINGS_FIELD)) or {}
    served_author: Final = pre_call_served_model(author, user_api_key_dict, router_settings.get("model_group_alias"))
    auth: Final = customer_request_auth(body) or user_api_key_dict
    deployment: Final = _author_deployment(served_author, user_api_key_dict.team_id, llm_router)
    sent_names: Final = (served_author,) if deployment is not None else tuple(dict.fromkeys((author, served_author)))
    for name in sent_names:
        await can_key_call_resolved_model(
            model=name,
            llm_model_list=llm_router.registry().all() if llm_router else None,
            valid_token=auth,
            llm_router=llm_router,
        )
        await _check_key_owner_access(name, auth, llm_router)
    if deployment is None:
        return {}
    provider_model: Final = _provider_model(deployment.litellm_params.model)
    metadata: Final = _as_object(body.get(metadata_variable_name)) or {}
    return {
        **_rewritten_body_fields(body, provider_model),
        **(
            {
                metadata_variable_name: {
                    **metadata,
                    SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: ImageSVGAuthorDeployment(
                        deployment_id=deployment.model_info.id, provider_model=provider_model
                    ).model_dump(),
                }
            }
            if deployment.model_info.id is not None
            else {}
        ),
    }
