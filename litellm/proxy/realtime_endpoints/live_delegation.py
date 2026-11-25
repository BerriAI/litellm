"""Access and budget checks for what a GPT-Live session delegates to, and the provider names of its models."""

import functools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain, takewhile
from typing import TYPE_CHECKING, Final

from fastapi import HTTPException
from pydantic import JsonValue, TypeAdapter, ValidationError

import litellm
from litellm.constants import AZURE_OPENAI_AUDIO_PROVIDERS
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    can_key_call_resolved_model,
    check_tools_allowlist,  # pyright: ignore[reportUnknownVariableType]  # its request_body is declared as a bare dict
    vector_store_access_check_for_caller,
)

if TYPE_CHECKING:
    from litellm.router import Router

_DELEGATION_KEY: Final = "delegation"
_TOOL_SCHEMA_KEY: Final = "parameters"
_JSON_ESCAPE: Final = "\\u"
_DELEGATED_TOOLS_ROUTE: Final = "/v1/responses"
_MAX_DELEGATION_DEPTH: Final = 64
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class LiveClientFrame:
    """A client frame as the provider should get it, or why the proxy will not relay it."""

    frame: str
    denial: str | None = None


@dataclass(frozen=True, slots=True)
class LiveDelegation:
    models: tuple[str, ...]
    vector_store_ids: tuple[str, ...]


def _strings(value: JsonValue) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        yield from filter(lambda item: isinstance(item, str), value)  # pyright: ignore[reportReturnType]  # filter keeps only str items


def _children(value: JsonValue) -> Iterator[JsonValue]:
    if isinstance(value, dict):
        yield from value.values()
    elif isinstance(value, list):
        yield from value


def _next_level(level: tuple[JsonValue, ...], _depth: int) -> tuple[JsonValue, ...]:
    return tuple(chain.from_iterable(map(_children, level)))


def _mappings(value: JsonValue) -> Iterator[Mapping[str, JsonValue]]:
    levels: Final = takewhile(bool, accumulate(range(_MAX_DELEGATION_DEPTH), _next_level, initial=(value,)))
    return (node for node in chain.from_iterable(levels) if isinstance(node, dict))


def _named(value: JsonValue, key: str) -> Iterator[str]:
    for mapping in _mappings(value):
        if key in mapping:
            yield from _strings(mapping[key])


def _tool_requests(value: JsonValue) -> Iterator[Mapping[str, JsonValue]]:
    return (mapping for mapping in _mappings(value) if "tools" in mapping)


def _inspected_delegation(session: object) -> JsonValue:
    """The session's delegation, refused when the proxy cannot read all of it and so cannot check all of it."""
    if not isinstance(session, Mapping) or _DELEGATION_KEY not in session:
        return None
    try:
        delegation: Final = _JSON.validate_python(session[_DELEGATION_KEY])
    except ValidationError as error:
        raise _uninspectable_delegation() from error
    below_the_inspected_depth: Final = functools.reduce(_next_level, range(_MAX_DELEGATION_DEPTH), (delegation,))
    if below_the_inspected_depth:
        raise _uninspectable_delegation()
    return delegation


def _uninspectable_delegation() -> ProxyException:
    return ProxyException(
        message=f"session.delegation is not JSON nested at most {_MAX_DELEGATION_DEPTH} levels deep, so it is refused",
        type=ProxyErrorTypes.bad_request_error,
        param="session.delegation",
        code=400,
    )


def _delegated_tool_requests(session: object) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(_tool_requests(_inspected_delegation(session)))


def live_delegation(session: object) -> LiveDelegation:
    delegation: Final = _inspected_delegation(session)
    return LiveDelegation(
        models=tuple(dict.fromkeys(_named(delegation, "model"))),
        vector_store_ids=tuple(dict.fromkeys(_named(delegation, "vector_store_ids"))),
    )


def _proxy_model_list() -> Sequence[object] | None:
    from litellm.proxy import proxy_server

    return proxy_server.llm_model_list  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # proxy_server declares a bare list


def _budgeted_names(model: str, team_id: str | None) -> tuple[str, ...]:
    """`model` and every other name its delegated spend is charged under, so admission reads the budgets it draws down."""
    from litellm.proxy import proxy_server
    from litellm.proxy.hooks.model_max_budget_limiter import budget_names_of

    router: Final = proxy_server.llm_router
    deployments: Final = () if router is None else router.registry().deployments(model, team_id=team_id) or ()
    provider_models: Final = sorted(
        {
            provider_model.rsplit("/", 1)[-1]
            for deployment in deployments
            if isinstance(provider_model := (as_str_mapping(deployment["litellm_params"]) or {}).get("model"), str)
        }
    )
    return tuple(dict.fromkeys((model, *chain.from_iterable(map(budget_names_of, provider_models or (model,))))))


async def _refuse_a_model_over_its_budget(model: str, caller: UserAPIKeyAuth) -> None:
    from litellm.proxy.auth.user_api_key_auth import (
        _is_within_every_model_budget,  # pyright: ignore[reportPrivateUsage]  # the same per-model budget walk a request's own model gets
    )
    from litellm.proxy.proxy_server import model_max_budget_limiter

    for budgeted in _budgeted_names(model, caller.team_id):
        if not await _is_within_every_model_budget(
            valid_token=caller, model_max_budget_limiter=model_max_budget_limiter, model=budgeted
        ):
            raise ProxyException(
                message=f"Budget exceeded for model={budgeted}, so this session cannot delegate to {model}",
                type=ProxyErrorTypes.budget_exceeded,
                param="model",
                code=400,
            )


async def authorize_live_delegation(session: object, caller: UserAPIKeyAuth) -> None:
    from litellm.proxy import proxy_server

    delegation: Final = live_delegation(session)
    for model in delegation.models:
        await can_key_call_resolved_model(
            model=model, valid_token=caller, llm_model_list=_proxy_model_list(), llm_router=proxy_server.llm_router
        )
        await _refuse_a_model_over_its_budget(model, caller)
    if delegation.vector_store_ids:
        await vector_store_access_check_for_caller(delegation.vector_store_ids, caller)
    for tool_request in _delegated_tool_requests(session):
        await check_tools_allowlist(
            request_body=dict(tool_request), valid_token=caller, team_object=None, route=_DELEGATED_TOOLS_ROUTE
        )


def _same_provider(provider: str, other: str | None) -> bool:
    return other == provider or {provider, other} <= AZURE_OPENAI_AUDIO_PROVIDERS


def _model_and_provider(litellm_params: Mapping[str, object] | None) -> tuple[str, str] | None:
    model: Final = None if litellm_params is None else litellm_params.get("model")
    if litellm_params is None or not isinstance(model, str) or not model:
        return None
    custom_llm_provider: Final = litellm_params.get("custom_llm_provider")
    try:
        provider_model, provider, _, _ = litellm.get_llm_provider(
            model=model, custom_llm_provider=custom_llm_provider if isinstance(custom_llm_provider, str) else None
        )
    except litellm.BadRequestError:
        return None
    return provider_model, provider


def _provider_model(litellm_params: Mapping[str, object] | None, provider: str) -> str | None:
    """The model a deployment calls at `provider`, or None when it calls another provider."""
    resolved: Final = _model_and_provider(litellm_params)
    return resolved[0] if resolved is not None and _same_provider(provider, resolved[1]) else None


def live_delegated_model_name(model: str, provider: str, router: "Router | None", team_id: str | None) -> str:
    """The name `provider` knows a delegated proxy model by. A name the proxy does not route is already one."""
    if router is None:
        return model
    deployments: Final = router.registry().deployments(model, team_id=team_id) or ()
    return next(
        (
            provider_model
            for deployment in deployments
            if (provider_model := _provider_model(as_str_mapping(deployment["litellm_params"]), provider)) is not None
        ),
        model,
    )


def _renamed(model: JsonValue, name: Callable[[str], str]) -> JsonValue:
    if isinstance(model, str):
        return name(model)
    if isinstance(model, list):
        return [name(item) if isinstance(item, str) else item for item in model]
    return model


def _model_selecting_children(value: JsonValue) -> Iterator[JsonValue]:
    if isinstance(value, dict):
        yield from (child for key, child in value.items() if key != _TOOL_SCHEMA_KEY)
    elif isinstance(value, list):
        yield from value


def _next_model_selecting_level(level: tuple[JsonValue, ...], _depth: int) -> tuple[JsonValue, ...]:
    return tuple(chain.from_iterable(map(_model_selecting_children, level)))


def with_provider_model_names(session: Mapping[str, object], name: Callable[[str], str]) -> Mapping[str, object]:
    """`session` with every model its delegation selects renamed by `name`, leaving tool schemas as the caller wrote them."""
    if _DELEGATION_KEY not in session:
        return session
    try:
        delegation: Final = _JSON.validate_json(json.dumps(_JSON.validate_python(session[_DELEGATION_KEY])))
    except ValidationError:
        return session
    levels: Final = takewhile(
        bool, accumulate(range(_MAX_DELEGATION_DEPTH), _next_model_selecting_level, initial=(delegation,))
    )
    for node in tuple(chain.from_iterable(levels)):
        if isinstance(node, dict) and "model" in node:
            node["model"] = _renamed(
                node["model"], name
            )  # the copy made above, renamed in place to stay linear in its size
    return {**session, _DELEGATION_KEY: delegation}


def deployment_delegation(
    session: Mapping[str, object], litellm_params: Mapping[str, object], caller: UserAPIKeyAuth
) -> Mapping[str, object]:
    """`session` with its delegated models named as the provider behind `litellm_params` knows them."""
    resolved: Final = _model_and_provider(litellm_params)
    return session if resolved is None else provider_delegation(session, resolved[1], caller)


def provider_delegation(session: Mapping[str, object], provider: str, caller: UserAPIKeyAuth) -> Mapping[str, object]:
    from litellm.proxy import proxy_server

    return with_provider_model_names(
        session, lambda model: live_delegated_model_name(model, provider, proxy_server.llm_router, caller.team_id)
    )


async def live_client_frame(client_frame: str, provider: str, caller: UserAPIKeyAuth) -> LiveClientFrame:
    if _DELEGATION_KEY not in client_frame and _JSON_ESCAPE not in client_frame:
        return LiveClientFrame(frame=client_frame)
    try:
        frame: Final = _JSON.validate_json(client_frame)
    except ValidationError:
        return LiveClientFrame(frame=client_frame, denial=str(_uninspectable_delegation().message))
    session: Final = as_str_mapping(frame.get("session")) if isinstance(frame, dict) else None
    if session is None or not isinstance(frame, dict):
        return LiveClientFrame(frame=client_frame)
    try:
        await authorize_live_delegation(session, caller)
    except ProxyException as error:
        return LiveClientFrame(frame=client_frame, denial=str(error.message))
    except HTTPException as error:
        return LiveClientFrame(frame=client_frame, denial=str(error.detail))
    if _DELEGATION_KEY not in session:
        return LiveClientFrame(frame=client_frame)
    return LiveClientFrame(frame=json.dumps({**frame, "session": provider_delegation(session, provider, caller)}))
