"""
Helpers for GET /key/{key_id}/models: resolve key model lists and build sectioned UI payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from litellm.proxy._types import SpecialModelNames
from litellm.proxy.auth.auth_checks import (
    _is_wildcard_pattern,
    _model_matches_any_wildcard_pattern_in_list,
    is_model_allowed_by_pattern,
)
from litellm.proxy.utils import PrismaClient
from litellm.router import Router

KEY_RESOLVED_MODELS_DISPLAY_LIMIT = 500

SECTION_ALL_PROXY = "all_proxy_models"
SECTION_ALL_TEAM = "all_team_models"
SECTION_ACCESS_GROUP = "access_group"
SECTION_UNGROUPED = "ungrouped"

TITLE_ALL_PROXY = "All proxy models"
TITLE_ALL_TEAM = "All team models"
TITLE_UNGROUPED = "Other models"


class ModelDisplaySection(TypedDict):
    title: str
    section_kind: str
    models: List[str]


def _filter_models_by_search(models: List[str], search: Optional[str]) -> List[str]:
    if not search:
        return list(models)
    needle = search.strip().lower()
    if not needle:
        return list(models)
    return [m for m in models if needle in m.lower()]


async def resolve_key_models_for_display(
    *,
    key_models: List[str],
    team_id: Optional[str],
    prisma_client: PrismaClient,
    llm_router: Optional[Router],
) -> Tuple[List[str], str, bool]:
    """
    Returns (resolved_model_names, source, all_team_models_without_team).
    """
    all_models: List[str] = []
    if llm_router is not None:
        all_models = list(llm_router.get_model_names())

    source: str = SpecialModelNames.no_default_models.value
    resolved: List[str] = list(key_models)
    all_team_models_without_team = False

    if SpecialModelNames.all_team_models.value in key_models:
        if team_id is not None:
            source = SpecialModelNames.all_team_models.value
            team_row = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": team_id},
            )
            if team_row is not None and team_row.models is not None:
                resolved = list(team_row.models)
        else:
            source = SpecialModelNames.all_team_models.value
            all_team_models_without_team = True
            resolved = list(all_models)

    if (
        SpecialModelNames.all_proxy_models.value in key_models
        or SpecialModelNames.all_proxy_models.value in resolved
    ):
        source = SpecialModelNames.all_proxy_models.value
        resolved = list(all_models)

    return resolved, source, all_team_models_without_team


def _concrete_models_allowed_by_resolved(
    resolved: List[str], router_model_names: List[str]
) -> List[str]:
    """Concrete router model names the key may call, given resolved patterns (incl. wildcards)."""
    resolved_list = list(resolved)
    resolved_set = set(resolved)
    out: List[str] = []
    seen: set[str] = set()
    for m in router_model_names:
        if m in seen:
            continue
        if m in resolved_set:
            out.append(m)
            seen.add(m)
        elif _model_matches_any_wildcard_pattern_in_list(m, resolved_list):
            out.append(m)
            seen.add(m)
    return out


def _expand_group_models_for_display(
    group_models: List[str],
    concrete_pool: List[str],
) -> List[str]:
    """
    Expand wildcard entries in a group's model list to concrete names from the pool.
    Non-wildcard entries are included when present in the pool.
    """
    pool_set = set(concrete_pool)
    out: List[str] = []
    seen: set[str] = set()
    for g in group_models:
        if _is_wildcard_pattern(g):
            for m in concrete_pool:
                if m in seen:
                    continue
                if is_model_allowed_by_pattern(m, g):
                    out.append(m)
                    seen.add(m)
        else:
            if g in seen:
                continue
            if g in pool_set:
                out.append(g)
                seen.add(g)
    return out


def _models_in_any_access_group(
    model_access_groups: Dict[str, List[str]], candidate_models: List[str]
) -> set:
    """Set of candidate_models that appear in at least one access group list (concrete names)."""
    cset = set(candidate_models)
    in_group: set = set()
    for models_in_group in model_access_groups.values():
        for m in models_in_group:
            if m in cset:
                in_group.add(m)
    return in_group


def build_model_display_sections(
    *,
    display_models: List[str],
    source: str,
    model_access_groups: Dict[str, List[str]],
    compact: bool,
) -> List[ModelDisplaySection]:
    """
    Build ordered sections for the admin UI.

    When source is all-proxy or all-team, a scope section lists all display_models first,
    then every intersecting **model_access_groups** section is still included (models may
    repeat across the sentinel section and each group). Ungrouped is omitted for sentinels
    only to avoid repeating the full flat list as "Other models".
    """
    sections: List[ModelDisplaySection] = []
    empty_models: List[str] = [] if compact else []

    display_set = set(display_models)

    if source == SpecialModelNames.all_proxy_models.value:
        sections.append(
            ModelDisplaySection(
                title=TITLE_ALL_PROXY,
                section_kind=SECTION_ALL_PROXY,
                models=empty_models if compact else list(display_models),
            )
        )
    elif source == SpecialModelNames.all_team_models.value:
        sections.append(
            ModelDisplaySection(
                title=TITLE_ALL_TEAM,
                section_kind=SECTION_ALL_TEAM,
                models=empty_models if compact else list(display_models),
            )
        )

    for group_name, group_models in model_access_groups.items():
        # group_models are models in this group that appear in resolved; further restrict to display slice
        intersected = [m for m in group_models if m in display_set]
        if not intersected:
            continue
        sections.append(
            ModelDisplaySection(
                title=group_name,
                section_kind=SECTION_ACCESS_GROUP,
                models=empty_models if compact else intersected,
            )
        )

    # Sentinel scope sections already list the full display set; skip ungrouped to avoid duplicating it.
    skip_ungrouped = source in (
        SpecialModelNames.all_proxy_models.value,
        SpecialModelNames.all_team_models.value,
    )
    if not skip_ungrouped:
        in_any = _models_in_any_access_group(model_access_groups, display_models)
        ungrouped = [m for m in display_models if m not in in_any]
        if ungrouped:
            sections.append(
                ModelDisplaySection(
                    title=TITLE_UNGROUPED,
                    section_kind=SECTION_UNGROUPED,
                    models=empty_models if compact else ungrouped,
                )
            )

    return sections


def prepare_key_models_response_payload(
    *,
    resolved: List[str],
    source: str,
    all_team_models_without_team: bool,
    model_access_groups: Dict[str, List[str]],
    search: Optional[str],
    compact: bool,
    all_router_model_names: List[str],
) -> Dict[str, Any]:
    """
    Apply search, truncation, and build the JSON-serializable response dict.

    Access-group sections list concrete model names: wildcards in router group metadata
    are expanded against models allowed for this key (resolved ∩ router names).
    """
    base_concrete = _concrete_models_allowed_by_resolved(resolved, all_router_model_names)

    expanded_groups: Dict[str, List[str]] = {}
    for group_name, group_models in model_access_groups.items():
        expanded = _expand_group_models_for_display(group_models, base_concrete)
        if expanded:
            expanded_groups[group_name] = expanded

    filtered_concrete = _filter_models_by_search(base_concrete, search)
    matched_count = len(filtered_concrete)
    models_truncated = matched_count > KEY_RESOLVED_MODELS_DISPLAY_LIMIT
    display_models = (
        filtered_concrete[:KEY_RESOLVED_MODELS_DISPLAY_LIMIT]
        if models_truncated
        else filtered_concrete
    )

    display_set = set(display_models)
    intersected_groups: Dict[str, List[str]] = {}
    for group_name, models in expanded_groups.items():
        in_slice = [m for m in models if m in display_set]
        if in_slice:
            intersected_groups[group_name] = in_slice

    model_display_sections = build_model_display_sections(
        display_models=display_models,
        source=source,
        model_access_groups=intersected_groups,
        compact=compact,
    )

    return {
        "model_display_sections": model_display_sections,
        "source": source,
        "resolved_total_count": len(resolved),
        "matched_count": matched_count,
        "models_truncated": models_truncated,
        "all_team_models_without_team": all_team_models_without_team,
    }
