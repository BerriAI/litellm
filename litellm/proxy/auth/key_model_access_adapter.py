from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from fastapi import Request
from pydantic import BaseModel, Field

from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.key_model_access import KeyAndFallbackModelAccessFacts, key_and_fallback_model_access
from litellm._auth.model_access import key_model_grants
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.model_access_adapter import object_model_grants, project_model_facts
from litellm.proxy.auth.model_projection_adapter import read_grouped_scope
from litellm.router import Router


class _Config(BaseModel, frozen=True):
    config: Mapping[str, object] = Field(default_factory=dict[str, object])


@dataclass(frozen=True, slots=True)
class _Sources:
    token: UserAPIKeyAuth
    request_data: dict[str, object]  # mutable-ok: alias resolution mutates the caller's request
    route: str
    request: Request | None
    router: Router | None


async def read_key_and_fallback_model_access_facts(
    *,
    valid_token: UserAPIKeyAuth,
    request_data: dict[str, object],  # mutable-ok: preserve request-local alias mutations
    route: str,
    request: Request | None,
    llm_model_list: Sequence[object] | None,
    llm_router: Router | None,
) -> KeyAndFallbackModelAccessFacts:
    from litellm.proxy.auth.user_api_key_auth import (
        _normalize_claude_model,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # legacy helper takes the mutable request body
        _resolve_router_settings_model_group_alias,  # pyright: ignore[reportPrivateUsage]  # preserve request-local router alias normalization
    )

    await _normalize_claude_model(request_data, valid_token, request, route)
    await _resolve_router_settings_model_group_alias(request_data, valid_token, request, route)
    facts: Final = KeyAndFallbackModelAccessFacts(
        config_nonempty=bool(_Config.model_validate(valid_token, from_attributes=True).config),
        key_models=object_model_grants(valid_token),
    )
    return await _read(facts, _Sources(valid_token, request_data, route, request, llm_router), ())


def _scope(model: str | Sequence[str], source: _Sources) -> GroupedModelScope:
    from litellm.proxy.auth.auth_checks import key_model_aliases_for_auth_check

    grants: Final = key_model_grants(
        object_model_grants(source.token), source.token.team_id, tuple(source.token.team_models or ())
    )
    return GroupedModelScope(
        project_model_facts(
            model,
            source.router,
            grants,
            source.token.team_model_aliases,
            source.token.team_id,
            key_model_aliases_for_auth_check(source.token),
            "key",
        ),
        tuple(source.token.access_group_ids or ()),
    )


async def _groups(scope: GroupedModelScope, source: _Sources) -> GroupedModelScope:
    return await read_grouped_scope(scope, source.router, merge=False)


async def _route(name: str, source: _Sources) -> Exception | None:
    from litellm.proxy.route_llm_request import (
        route_request,  # pyright: ignore[reportUnknownVariableType]  # preserve the existing routing probe and exception identity
    )

    try:
        await route_request(
            data={"model": name, "messages": [{"role": "user", "content": "Who was Alexander?"}]},
            llm_router=source.router,
            user_model=None,
            route_type="acompletion",
        )
    except NativeAuthFailure:
        raise
    except Exception as error:  # noqa: BLE001  # native policy sequences a legacy routing IO failure before rethrowing
        return error
    return None


async def _read(
    facts: KeyAndFallbackModelAccessFacts, source: _Sources, route_errors: tuple[Exception | None, ...]
) -> KeyAndFallbackModelAccessFacts:
    from litellm.proxy.auth.auth_utils import fallback_target_model_name, iter_request_fallback_targets
    from litellm.proxy.auth.user_api_key_auth import (
        _get_model_from_request_context,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # read the existing mutable request model
    )

    decision: Final = key_and_fallback_model_access(facts)
    match decision.kind:
        case "Allow" | "Deny":
            return facts
        case "ReadPrimary":
            model: Final = _get_model_from_request_context(
                request_data=source.request_data,
                route=source.route,
                request=source.request,
                llm_router=source.router,
                team_id=source.token.team_id,
            )
            return await _read(
                replace(facts, primary_loaded=True, primary=_scope(model, source) if model is not None else None),
                source,
                route_errors,
            )
        case "ReadPrimaryGroups":
            if facts.primary is None:
                raise NativeAuthRuntimeError("missing primary model scope")
            return await _read(replace(facts, primary=await _groups(facts.primary, source)), source, route_errors)
        case "ReadFallbackNames":
            names: Final = tuple(
                dict.fromkeys(
                    name
                    for target in iter_request_fallback_targets(source.request_data)
                    if (name := fallback_target_model_name(target)) is not None
                )
            )
            return await _read(replace(facts, fallback_names=names), source, route_errors)
        case "ReadFallbackScope":
            if facts.fallback_names is None:
                raise NativeAuthRuntimeError("missing fallback names")
            scope: Final = _scope(facts.fallback_names[decision.index], source)
            return await _read(replace(facts, fallback_scopes=(*facts.fallback_scopes, scope)), source, route_errors)
        case "ReadFallbackGroups":
            grouped: Final = await _groups(facts.fallback_scopes[decision.index], source)
            scopes: Final = tuple(
                grouped if index == decision.index else scope for index, scope in enumerate(facts.fallback_scopes)
            )
            return await _read(replace(facts, fallback_scopes=scopes), source, route_errors)
        case "RouteFallback":
            if facts.fallback_names is None:
                raise NativeAuthRuntimeError("missing fallback names")
            error: Final = await _route(facts.fallback_names[decision.index], source)
            return await _read(
                replace(facts, routing=(*facts.routing, "Routed" if error is None else "Failed")),
                source,
                (*route_errors, error),
            )
        case "RoutingFailure":
            failure: Final = route_errors[decision.index]
            if failure is None:
                raise NativeAuthRuntimeError("missing fallback routing failure")
            raise failure
