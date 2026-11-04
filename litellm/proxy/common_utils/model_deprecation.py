from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry, GetModelCostMap
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_WARN_DAYS,
    DeprecationStatus,
    ModelDeprecationInfo,
    ModelDeprecationResponse,
)

if TYPE_CHECKING:
    from litellm.router import Router

_NO_MODEL_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_SHARED_SCHEDULE_PROVIDER: Final[Mapping[str, str]] = MappingProxyType({"azure_ai": "azure"})

LoadedCatalog: TypeAlias = Callable[[], Mapping[str, Mapping[str, object]]]


@dataclass(frozen=True, slots=True)
class _ResolvedDeprecation:
    deprecation_date: date
    litellm_model: str | None
    litellm_provider: str | None
    copied_from_another_row: bool = False


def _parse_deprecation_date(raw_value: object) -> date | None:
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, date):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    try:
        return date.fromisoformat(raw_value.strip())
    except ValueError:
        return None


def _successor_model(raw_value: object) -> str | None:
    """A non-empty successor name, or None when the field is missing, blank, or not a string"""
    return raw_value.strip() or None if isinstance(raw_value, str) else None


def _cost_map_lookup(model_key: str | None, catalog: Mapping[str, object]) -> _ResolvedDeprecation | None:
    if model_key is None:
        return None
    entry: Final = litellm.model_cost.get(model_key)
    if not isinstance(entry, Mapping):
        return None
    parsed: Final = _parse_deprecation_date(entry.get("deprecation_date"))
    if parsed is None:
        return None
    provider: Final = entry.get("litellm_provider")
    return _ResolvedDeprecation(
        deprecation_date=parsed,
        litellm_model=model_key,
        litellm_provider=provider if isinstance(provider, str) else None,
        copied_from_another_row=isinstance(entry, AliasedCostEntry)
        and "deprecation_date" not in entry.registered_fields
        and model_key not in catalog,
    )


def _first_own_row(model_keys: Sequence[str | None], catalog: Mapping[str, object]) -> _ResolvedDeprecation | None:
    found: Final = tuple(
        resolved for resolved in (_cost_map_lookup(key, catalog) for key in model_keys) if resolved is not None
    )
    return next((resolved for resolved in found if not resolved.copied_from_another_row), next(iter(found), None))


def _deployment_provider(litellm_params: Mapping[str, object]) -> str | None:
    declared: Final = litellm_params.get("custom_llm_provider")
    if isinstance(declared, str) and declared:
        return declared
    model: Final = litellm_params.get("model")
    prefix: Final = model.split("/", 1)[0] if isinstance(model, str) and "/" in model else None
    return prefix if prefix in litellm.provider_list else None


def _provider_qualified(provider: str | None, model_key: str | None) -> str | None:
    if provider is None or model_key is None or model_key.startswith(f"{provider}/"):
        return None
    return f"{provider}/{model_key}"


def _as_model_key(raw_value: object) -> str | None:
    return raw_value if isinstance(raw_value, str) and raw_value else None


def _without_provider(provider: str | None, model_name: str | None) -> str | None:
    if provider is None or model_name is None:
        return model_name
    return model_name.removeprefix(f"{provider}/")


def _schedule_keys(provider: str | None, model_name: str | None) -> tuple[str | None, ...]:
    shared_provider: Final = _SHARED_SCHEDULE_PROVIDER.get(provider) if provider is not None else None
    unprefixed: Final = _without_provider(provider, model_name)
    return (
        _provider_qualified(provider, unprefixed),
        _provider_qualified(shared_provider, unprefixed),
        model_name,
    )


def _mapping_field(deployment: Mapping[str, object], key: str) -> Mapping[str, object]:
    value: Final = deployment.get(key)
    return value if isinstance(value, Mapping) else _NO_MODEL_METADATA


def _resolve_deployment_deprecation(
    deployment: Mapping[str, object], catalog: Mapping[str, object]
) -> _ResolvedDeprecation | None:
    """Resolve a deployment's deprecation date, preferring its explicit override"""
    model_info: Final = _mapping_field(deployment, "model_info")
    litellm_params: Final = _mapping_field(deployment, "litellm_params")
    model_key: Final = _as_model_key(litellm_params.get("model"))

    override: Final = _parse_deprecation_date(model_info.get("deprecation_date"))
    if override is not None:
        override_provider: Final = model_info.get("litellm_provider")
        return _ResolvedDeprecation(
            deprecation_date=override,
            litellm_model=model_key,
            litellm_provider=override_provider if isinstance(override_provider, str) else None,
        )

    provider: Final = _deployment_provider(litellm_params)
    base_model: Final = _as_model_key(model_info.get("base_model"))
    model_name: Final = model_key.split("/", 1)[1] if model_key is not None and "/" in model_key else model_key
    picks: Final = tuple(
        pick
        for pick in (
            _first_own_row(_schedule_keys(provider, base_model), catalog),
            _first_own_row(
                (_provider_qualified(provider, model_key), model_key, *_schedule_keys(provider, model_name)), catalog
            ),
        )
        if pick is not None
    )
    own_rows: Final = tuple(pick for pick in picks if not pick.copied_from_another_row)
    return min(own_rows or picks, key=lambda pick: pick.deprecation_date, default=None)


def _classify(days_until: int, warn_within_days: int) -> DeprecationStatus:
    if days_until < 0:
        return "deprecated"
    if days_until <= warn_within_days:
        return "imminent"
    return "upcoming"


def _checked_successor(model_name: str, model_info: Mapping[str, object], llm_router: Router) -> str | None:
    """The deployment's successor_model as set, with a warning when no model in its team scope matches it"""
    successor: Final = _successor_model(model_info.get("successor_model"))
    if successor is None:
        return None
    team_id: Final = model_info.get("team_id")
    if not llm_router.registry().deployments(successor, team_id=team_id if isinstance(team_id, str) else None):
        verbose_logger.warning(
            "model_deprecation: %s names successor %s, which matches no model group, alias, or wildcard route here",
            model_name,
            successor,
        )
    return successor


def _build_info(
    deployment: Mapping[str, object],
    llm_router: Router,
    today: date,
    warn_within_days: int,
    catalog: Mapping[str, object],
) -> ModelDeprecationInfo | None:
    model_name: Final = deployment.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        return None

    resolved: Final = _resolve_deployment_deprecation(deployment, catalog)
    if resolved is None:
        return None

    days_until: Final = (resolved.deprecation_date - today).days
    return ModelDeprecationInfo(
        model_name=model_name,
        litellm_model=resolved.litellm_model,
        deprecation_date=resolved.deprecation_date,
        days_until_deprecation=days_until,
        status=_classify(days_until, warn_within_days),
        litellm_provider=resolved.litellm_provider,
        successor_model=_checked_successor(model_name, _mapping_field(deployment, "model_info"), llm_router),
    )


def _merge_group(group: Sequence[ModelDeprecationInfo]) -> ModelDeprecationInfo:
    """Keep the group's first deployment in config order, naming the first successor any of its deployments names"""
    first: Final = group[0]
    successors: Final = tuple(model.successor_model for model in group if model.successor_model is not None)
    if not successors:
        return first
    if len(frozenset(successors)) > 1:
        verbose_logger.warning(
            "model_deprecation: %s names several successors %s, reporting %s",
            first.model_name,
            sorted(frozenset(successors)),
            successors[0],
        )
    return first.model_copy(update={"successor_model": successors[0]})


def _dedupe(
    models: Sequence[ModelDeprecationInfo],
) -> tuple[ModelDeprecationInfo, ...]:
    """Report a model group carrying the same date on several deployments once"""
    ordered: Final = sorted(models, key=lambda model: (model.model_name, model.deprecation_date))
    return tuple(
        _merge_group(tuple(group))
        for _, group in groupby(ordered, key=lambda model: (model.model_name, model.deprecation_date))
    )


def _bucket(models: Sequence[ModelDeprecationInfo], status: DeprecationStatus) -> tuple[ModelDeprecationInfo, ...]:
    return tuple(
        sorted(
            (model for model in models if model.status == status),
            key=lambda model: model.deprecation_date,
        )
    )


def _collect_from_router(
    llm_router: Router, today: date, warn_within_days: int, catalog: Mapping[str, object]
) -> tuple[ModelDeprecationInfo, ...]:
    deployments: Final = llm_router.registry().deployments() or ()
    infos: Final = (_build_info(deployment, llm_router, today, warn_within_days, catalog) for deployment in deployments)
    deduped: Final = _dedupe(tuple(info for info in infos if info is not None))

    verbose_logger.debug(
        "model_deprecation: %d/%d deployments carry a deprecation date",
        len(deduped),
        len(deployments),
    )
    return deduped


def collect_model_deprecations(
    llm_router: Router | None,
    warn_within_days: int = DEFAULT_DEPRECATION_WARN_DAYS,
    today: date | None = None,
    loaded_catalog: LoadedCatalog = GetModelCostMap.loaded_model_cost_map,
) -> ModelDeprecationResponse:
    """Bucket every deployment carrying a deprecation date by how urgent it is"""
    snapshot_time: Final = datetime.now(timezone.utc)
    effective_today: Final = today or snapshot_time.date()
    deduped: Final = (
        _collect_from_router(llm_router, effective_today, warn_within_days, loaded_catalog())
        if llm_router is not None
        else ()
    )

    return ModelDeprecationResponse(
        deprecated=_bucket(deduped, "deprecated"),
        imminent=_bucket(deduped, "imminent"),
        upcoming=_bucket(deduped, "upcoming"),
        warn_within_days=warn_within_days,
        checked_at=snapshot_time,
    )


def _escape_slack_mrkdwn(value: str) -> str:
    """Neutralize Slack control characters so a model name cannot forge a mention or link"""
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _format_entry(info: ModelDeprecationInfo) -> str:
    suffix: Final = (
        f"already deprecated {abs(info.days_until_deprecation)}d ago"
        if info.days_until_deprecation < 0
        else f"in {info.days_until_deprecation}d"
    )
    migration: Final = f", migrate to `{_escape_slack_mrkdwn(info.successor_model)}`" if info.successor_model else ""
    return (
        f"• `{_escape_slack_mrkdwn(info.model_name)}` "
        f"(provider: {_escape_slack_mrkdwn(info.litellm_provider) if info.litellm_provider else 'unknown'}, "
        f"deprecates {info.deprecation_date.isoformat()}, {suffix}{migration})"
    )


def format_deprecation_alert_message(
    snapshot: ModelDeprecationResponse,
) -> str | None:
    """Render the alert for the deprecated and imminent buckets, None when both are empty

    Upcoming models are left out of the alert to keep it actionable.
    """
    if not snapshot.deprecated and not snapshot.imminent:
        return None

    deprecated_section: Final = (
        ("\n*Already deprecated:*", *(_format_entry(i) for i in snapshot.deprecated)) if snapshot.deprecated else ()
    )
    imminent_section: Final = (
        (
            f"\n*Deprecating within {snapshot.warn_within_days} days:*",
            *(_format_entry(i) for i in snapshot.imminent),
        )
        if snapshot.imminent
        else ()
    )

    return "\n".join(
        (
            "*⚠️ Model Deprecation Warning*",
            *deprecated_section,
            *imminent_section,
            "\nPlan migrations to a supported model. See "
            "https://docs.litellm.ai/docs/proxy/model_management for guidance.",
        )
    )
