import json
from collections.abc import Iterator, Mapping, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.constants import SVG_AUTHOR_DEPLOYMENT_METADATA_KEY
from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    can_agent_call_model,
    can_key_call_resolved_model,
    can_user_call_model,
    dashboard_uses_personal_key_team_access,
    get_user_object,
    key_model_aliases_for_auth_check,
)
from litellm.proxy.hooks.model_max_budget_limiter import delegated_model_over_budget
from litellm.proxy.litellm_pre_call_utils import customer_request_auth, request_served_model
from litellm.types.images.main import ImageSVGAuthorDeployment
from litellm.types.proxy.auth.auth_checks import UserNotFoundError

if TYPE_CHECKING:
    from litellm.router import Router

_SVG_FIELD: Final = "svg"
_SVG_FORM_MODEL_FIELD: Final = "svg[model]"
_EXTRA_BODY_FIELD: Final = "extra_body"
_ROUTER_SETTINGS_FIELD: Final = "router_settings_override"
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_KEY_VALUE_PAIRS: Final = TypeAdapter(list[tuple[str, object]])


class _DeploymentParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    custom_llm_provider: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    organization: str | None = None
    litellm_credential_name: str | None = None


class _DeploymentInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | None = None


class _Deployment(BaseModel):
    model_config = ConfigDict(frozen=True)

    litellm_params: _DeploymentParams
    model_info: _DeploymentInfo = _DeploymentInfo()


def _as_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_json(value) if isinstance(value, str) else _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _author_of(svg: object) -> str | None:
    model: Final = (_as_object(svg) or {}).get("model")
    return model if isinstance(model, str) and model else None


def _as_extra_body(value: object) -> Mapping[str, object] | None:
    parsed: Final = _as_object(value)
    if parsed is not None or isinstance(value, str):
        return parsed
    try:
        return dict(_KEY_VALUE_PAIRS.validate_python(value))
    except ValidationError:
        return None


def _containers(body: Mapping[str, object]) -> Iterator[Mapping[str, object]]:
    yield body
    yield _as_extra_body(body.get(_EXTRA_BODY_FIELD)) or {}


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
    parsed_extra_body: Final = _as_extra_body(extra_body)
    extra_body_fields: Final[Mapping[str, object]] = (
        {} if parsed_extra_body is None else _rewritten_svg_fields(parsed_extra_body, upstream)
    )
    return {
        **_rewritten_svg_fields(body, upstream),
        **(
            {_EXTRA_BODY_FIELD: _encoded_like(extra_body, {**(parsed_extra_body or {}), **extra_body_fields})}
            if extra_body_fields
            else {}
        ),
    }


def _deployments(model_group: str, team_id: str | None, llm_router: "Router | None") -> tuple[_Deployment, ...]:
    deployments: Final[Sequence[object]] = (
        (llm_router.registry().deployments(model_group, team_id=team_id) or ()) if llm_router else ()
    )
    return tuple(_Deployment.model_validate(deployment) for deployment in deployments)


def _provider_and_model(params: _DeploymentParams) -> tuple[str | None, str]:
    try:
        model, provider, _, _ = litellm.get_llm_provider(
            model=params.model, custom_llm_provider=params.custom_llm_provider
        )
    except litellm.BadRequestError:
        return params.custom_llm_provider, params.model
    return provider, model


def _upstream_account(params: _DeploymentParams) -> tuple[object, ...]:
    return (
        _provider_and_model(params)[0],
        (params.api_base or "").rstrip("/") or None,
        params.api_key,
        params.organization,
        params.litellm_credential_name,
    )


def _image_deployments_billed_like(
    author: _Deployment, body: Mapping[str, object], team_id: str | None, llm_router: "Router | None"
) -> tuple[str, ...]:
    image_model: Final = body.get("model")
    if not isinstance(image_model, str):
        return ()
    model_group_alias: Final = (_as_object(body.get(_ROUTER_SETTINGS_FIELD)) or {}).get("model_group_alias")
    image_group: Final = resolve_model_group_alias(model_group_alias, image_model) or image_model
    author_account: Final = _upstream_account(author.litellm_params)
    return tuple(
        deployment.model_info.id
        for deployment in _deployments(image_group, team_id, llm_router)
        if deployment.model_info.id is not None and _upstream_account(deployment.litellm_params) == author_account
    )


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
    if prisma_client is None or dashboard_uses_personal_key_team_access(user_api_key_dict):
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


async def _refuse_an_author_over_its_budget(authors: tuple[str, ...], caller: UserAPIKeyAuth) -> None:
    for author in authors:
        if (budgeted := await delegated_model_over_budget(author, caller)) is not None:
            raise ProxyException(
                message=f"Budget exceeded for model={budgeted}, so svg.model cannot use {author}",
                type=ProxyErrorTypes.budget_exceeded,
                param="svg.model",
                code=400,
            )


def _metadata_with_author_guardrails(
    body: Mapping[str, object], served_author: str, llm_router: "Router | None"
) -> Mapping[str, object] | None:
    """The request metadata with the author deployment's guardrails added, or None when it has none to add."""
    from litellm.proxy.utils import (
        check_and_merge_model_level_guardrails,  # pyright: ignore[reportUnknownVariableType]  # takes and returns the untyped request dict
    )

    metadata: Final = _as_object(body.get("metadata")) or {}
    merged: Final = _as_object(
        check_and_merge_model_level_guardrails(  # pyright: ignore[reportUnknownArgumentType]  # returns the untyped request dict
            data={**body, "metadata": dict(metadata)},
            llm_router=llm_router,
            trust_client_model_info=False,
            model_alias=served_author,
        )
    )
    merged_metadata: Final = _as_object((merged or {}).get("metadata")) or {}
    return None if merged_metadata.get("guardrails") == metadata.get("guardrails") else merged_metadata


async def svg_author_model_updates(
    body: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router | None",
    metadata_variable_name: str,
) -> Mapping[str, object]:
    """The request fields to replace so ``svg.model`` is authorized, sent by its provider name, and billed by its deployment."""
    match _svg_author_models(body):
        case ():
            return {}
        case (author,):
            return await _authorized_author_updates(author, body, user_api_key_dict, llm_router, metadata_variable_name)
        case conflicting:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"svg.model is set more than once with different values: {', '.join(conflicting)}",
            )


async def _authorized_author_updates(
    author: str,
    body: Mapping[str, object],
    user_api_key_dict: UserAPIKeyAuth,
    llm_router: "Router | None",
    metadata_variable_name: str,
) -> Mapping[str, object]:
    model_group_alias: Final = (_as_object(body.get(_ROUTER_SETTINGS_FIELD)) or {}).get("model_group_alias")
    grouped_author: Final = resolve_model_group_alias(model_group_alias, author) or author
    served_author: Final = request_served_model(grouped_author, user_api_key_dict, None, llm_router)
    served_without_key_aliases: Final = request_served_model(
        grouped_author, user_api_key_dict.model_copy(update={"aliases": {}}), None, llm_router
    )
    auth: Final[UserAPIKeyAuth] = customer_request_auth(body) or user_api_key_dict
    without_key_aliases: Final = auth.model_copy(update={"aliases": {}})
    deployment: Final = next(iter(_deployments(served_author, user_api_key_dict.team_id, llm_router)), None)
    key_aliased_author: Final = (
        ((served_author, without_key_aliases),) if served_author != served_without_key_aliases else ()
    )
    sent_unchanged_author: Final = () if deployment is not None else ((author, without_key_aliases),)
    checks: Final = ((grouped_author, auth), *key_aliased_author, *sent_unchanged_author)
    for name, valid_token in checks:
        await can_key_call_resolved_model(
            model=name,
            llm_model_list=llm_router.registry().all() if llm_router else None,
            valid_token=valid_token,
            llm_router=llm_router,
        )
        await _check_key_owner_access(name, valid_token, llm_router)
    await _refuse_an_author_over_its_budget(tuple(dict.fromkeys(name for name, _ in checks)), user_api_key_dict)
    if deployment is None:
        return {}
    provider_model: Final = _provider_and_model(deployment.litellm_params)[1]
    billed_like_author: Final = _image_deployments_billed_like(deployment, body, user_api_key_dict.team_id, llm_router)
    guarded_metadata: Final = _metadata_with_author_guardrails(body, served_author, llm_router)
    metadata: Final = (
        guarded_metadata
        if guarded_metadata is not None and metadata_variable_name == "metadata"
        else _as_object(body.get(metadata_variable_name)) or {}
    )
    return {
        **_rewritten_body_fields(body, provider_model),
        **({"metadata": guarded_metadata} if guarded_metadata is not None else {}),
        **(
            {
                metadata_variable_name: {
                    **metadata,
                    SVG_AUTHOR_DEPLOYMENT_METADATA_KEY: ImageSVGAuthorDeployment(
                        deployment_id=deployment.model_info.id,
                        provider_model=provider_model,
                        image_deployment_ids=billed_like_author,
                    ).model_dump(),
                }
            }
            if deployment.model_info.id is not None and billed_like_author
            else {}
        ),
    }
