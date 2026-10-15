from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.get_llm_provider_logic import declared_authenticating_provider, get_llm_provider
from litellm.proxy._types import UserAPIKeyAuth, user_api_key_has_admin_view

if TYPE_CHECKING:
    from litellm.router import Router
    from litellm.types.router import RouterModelGroupAliasItem

_PATTERN_DEPLOYMENTS: Final = TypeAdapter(Mapping[str, tuple[Mapping[str, object], ...]])


def is_undiscoverable_deployment(deployment: Mapping[str, object]) -> bool:
    model_info: Final = deployment.get("model_info")
    if not isinstance(model_info, Mapping):
        return False
    return "discoverable" in model_info and model_info["discoverable"] is False


def is_undiscoverable_model_name(model_name: str, llm_router: Router | None, team_id: str | None) -> bool:
    if llm_router is None:
        return False
    deployments: Final = llm_router.get_model_list(model_name=model_name, team_id=team_id)
    if not deployments:
        return False
    return all(is_undiscoverable_deployment(deployment) for deployment in deployments)


def _team_public_model_name(deployment: Mapping[str, object]) -> object:
    model_info: Final = deployment.get("model_info")
    return model_info.get("team_public_model_name") if isinstance(model_info, Mapping) else None


def _alias_target(alias: str | RouterModelGroupAliasItem) -> str:
    return alias if isinstance(alias, str) else alias["model"]


def _undiscoverable_served_names(
    undiscoverable_rows: Iterable[Mapping[str, object]],
    model_group_alias: Mapping[str, str | RouterModelGroupAliasItem],
) -> frozenset[str]:
    served: Final = frozenset(
        name
        for row in undiscoverable_rows
        for name in (row.get("model_name"), _team_public_model_name(row))
        if isinstance(name, str)
    )
    aliases: Final = frozenset(alias for alias, target in model_group_alias.items() if _alias_target(target) in served)
    return served | aliases


def _undiscoverable_patterns(llm_router: Router, team_id: str | None) -> tuple[re.Pattern[str], ...]:
    team_pattern_router: Final = llm_router.team_pattern_routers.get(team_id) if team_id is not None else None
    pattern_routers: Final = (
        (llm_router.pattern_router,)
        if team_pattern_router is None
        else (llm_router.pattern_router, team_pattern_router)
    )
    return tuple(
        re.compile(regex)
        for pattern_router in pattern_routers
        for regex, deployments in _PATTERN_DEPLOYMENTS.validate_python(pattern_router.patterns).items()
        if any(is_undiscoverable_deployment(deployment) for deployment in deployments)
    )


def _resolved_provider(model_name: str) -> str | None:
    try:
        return get_llm_provider(model=model_name)[1]
    except Exception:  # noqa: BLE001  # get_llm_provider raises when the provider is unknown; the name then routes as-is
        return None


def _matches_undiscoverable_pattern(model_name: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    if not patterns:
        return False
    if any(pattern.match(model_name) for pattern in patterns):
        return True
    provider: Final = declared_authenticating_provider(model_name) or _resolved_provider(model_name)
    return any(pattern.match(f"{provider}/{model_name}") for pattern in patterns)


def undiscoverable_model_names(
    model_names: Iterable[str],
    llm_router: Router | None,
    user_api_key_dict: UserAPIKeyAuth,
    team_id: str | None,
) -> frozenset[str]:
    if llm_router is None or user_api_key_has_admin_view(user_api_key_dict):
        return frozenset()
    undiscoverable_rows: Final = tuple(
        row for row in llm_router.get_model_list() or () if is_undiscoverable_deployment(row)
    )
    if not undiscoverable_rows:
        return frozenset()
    served_names: Final = _undiscoverable_served_names(undiscoverable_rows, llm_router.model_group_alias)
    patterns: Final = _undiscoverable_patterns(llm_router, team_id)
    return frozenset(
        name
        for name in model_names
        if (name in served_names or _matches_undiscoverable_pattern(name, patterns))
        and is_undiscoverable_model_name(name, llm_router, team_id)
    )


def discoverable_rows(
    rows: Iterable[Mapping[str, object]],
    user_api_key_dict: UserAPIKeyAuth,
) -> tuple[Mapping[str, object], ...]:
    if user_api_key_has_admin_view(user_api_key_dict):
        return tuple(rows)
    return tuple(row for row in rows if not is_undiscoverable_deployment(row))
