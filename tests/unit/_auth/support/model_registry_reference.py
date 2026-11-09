"""Model authorization and registry ordering frozen from Phase 1, 8bd7f5089a75."""

import re
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from pydantic import BaseModel, TypeAdapter

import litellm
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.proxy._types import LiteLLM_UserTable, ModelAccessDeniedProxyException, ProxyErrorTypes
from litellm.proxy.auth.model_access_denied import (
    customer_model_access_denied_client_message,
    model_access_denied_client_message,
)
from litellm.router_api import RegistryReader
from litellm.router_strategy.complexity_router.context_compaction import native_compaction_parent


def _pattern(model: str, pattern: str) -> bool:
    return "*" in pattern and re.match("^" + pattern.replace("*", ".*") + "$", model) is not None


def _provider_pattern(model: str, pattern: str) -> bool:
    try:
        stripped, provider, _, _ = litellm.get_llm_provider(model=model)
    except Exception:  # noqa: BLE001  # the Phase 1 provider inference helper swallowed lookup failures
        return False
    if stripped == model and "/" in model:
        return False
    return _pattern(f"{provider}/{stripped}", pattern)


def _helper(
    model: str,
    router: RegistryReader | None,
    grants: Sequence[str],
    team: str | None,
    aliases: Mapping[str, str] | None,
) -> bool:
    groups: Final = router.registry().access_groups(model_name=model, team_id=team) if router else {}
    names: Final = (
        router.registry().names() if "all-team-models" in grants and team is not None and router is not None else None
    )
    ordinary: Final = tuple(grant for grant in grants if grant != "all-team-models")
    expanded: Final = (
        (tuple(dict.fromkeys((*ordinary, *names))) or tuple(grants)) if names is not None else tuple(grants)
    )
    if groups and any(grant in groups for grant in expanded):
        return True
    filtered: Final = tuple(grant for grant in expanded if grant not in groups)
    if aliases and model in aliases:
        return True
    if any(_pattern(model, grant) for grant in filtered):
        return True
    if any("*" in grant and _provider_pattern(model, grant) for grant in filtered):
        return True
    return (not filtered and not expanded) or "*" in filtered or "all-proxy-models" in filtered or model in filtered


def _owning(
    model: str, router: RegistryReader | None, team: str | None, aliases: Mapping[str, str] | None
) -> tuple[str, ...]:
    if router is None or team is None or not model.startswith(f"model_name_{team}_"):
        return ()
    deployment: Final = router.registry().deployments(model, source="first_in_group")
    public: Final = (
        deployment.model_info.team_public_model_name
        if deployment is not None and deployment.model_info.team_id == team
        else None
    )
    return (*((public,) if public else ()), *(alias for alias, target in (aliases or {}).items() if target == model))


def can_object_call_model(
    model: str | Sequence[str],
    llm_router: RegistryReader | None,
    models: Sequence[str],
    team_model_aliases: Mapping[str, str] | None = None,
    team_id: str | None = None,
    key_model_aliases: Mapping[str, str] | None = None,
    object_type: Literal["user", "customer", "team", "key", "org", "project", "agent"] = "user",
    fallback_depth: int = 0,
) -> Literal[True]:
    if fallback_depth >= DEFAULT_MAX_RECURSE_DEPTH:
        raise Exception(f"Unable to parse model, max fallback depth exceeded - received model: {model}")  # noqa: TRY002  # frozen Phase 1 exception contract
    if not isinstance(model, str):
        for item in model:
            can_object_call_model(
                item,
                llm_router,
                models,
                team_model_aliases,
                team_id,
                key_model_aliases,
                object_type,
                fallback_depth + 1,
            )
        return True
    return _single(model, llm_router, models, team_model_aliases, team_id, key_model_aliases, object_type)


def _single(
    model: str,
    router: RegistryReader | None,
    grants: Sequence[str],
    team_aliases: Mapping[str, str] | None,
    team: str | None,
    key_aliases: Mapping[str, str] | None,
    object_type: Literal["user", "customer", "team", "key", "org", "project", "agent"],
) -> Literal[True]:
    parent: Final = native_compaction_parent(model)
    target: Final = (
        litellm.model_alias_map[model]
        if model in litellm.model_alias_map
        else router.registry().resolve(model, kind="alias")
        if router is not None and model in router.registry().aliases()
        else None
    )
    after_team: Final = (team_aliases or {}).get(model, model)
    after_key: Final = (key_aliases or {}).get(after_team, after_team)
    after_global: Final = litellm.model_alias_map.get(after_key, after_key)
    dispatched: Final = (key_aliases or {}).get(after_global, after_global)
    key_applied: Final = after_key != after_team or dispatched != after_global
    candidates: Final = (
        (dispatched,)
        if key_applied
        else (model, *((parent,) if parent is not None else ()), *((target,) if target else ()))
    )
    owning: Final = () if key_applied else _owning(model, router, team, team_aliases)
    if any(_helper(candidate, router, grants, team, team_aliases) for candidate in candidates):
        return True
    if any(_helper(candidate, router, grants, team, None) for candidate in owning):
        return True
    raise ModelAccessDeniedProxyException(
        message=customer_model_access_denied_client_message(model)
        if object_type == "customer"
        else model_access_denied_client_message(model),
        internal_message=f"{object_type} not allowed to access model. This {object_type} can only access models={list(grants)}. Tried to access {model}",
        type=ProxyErrorTypes.get_model_access_error_type_for_object(object_type),
        param="model",
        code=403,
    )


class _Grants(BaseModel, frozen=True):
    models: tuple[str, ...] = ()


async def can_user_call_model(
    model: str, llm_router: RegistryReader | None, user_object: LiteLLM_UserTable | None
) -> Literal[True]:
    if user_object is None:
        return True
    grants: Final = _Grants.model_validate(user_object, from_attributes=True).models
    if "no-default-models" in grants:
        raise ModelAccessDeniedProxyException(
            message=model_access_denied_client_message(model),
            internal_message=f"User not allowed to access model. No default model access, only team models allowed. Tried to access {model}",
            type=ProxyErrorTypes.key_model_access_denied,
            param="model",
            code=403,
        )
    if not grants:
        return True
    return can_object_call_model(model, llm_router, grants)


def entry_has_priced_metric(entry: Mapping[str, object]) -> bool:
    if "tiered_pricing" in entry and entry["tiered_pricing"] is not None:
        return True
    return any(_priced(key, value) for key, value in entry.items())


def _priced(key: str, value: object) -> bool:
    if "cost_per" not in key:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, dict):
        return any(
            isinstance(number, (int, float)) and not isinstance(number, bool) and number > 0
            for number in TypeAdapter(Mapping[object, object]).validate_python(value).values()
        )
    return False
