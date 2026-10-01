from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import litellm
from litellm._logging import verbose_logger
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_WARN_DAYS,
    DeprecationStatus,
    ModelDeprecationInfo,
    ModelDeprecationResponse,
)

if TYPE_CHECKING:
    from litellm.router import Router

_NO_MODEL_METADATA: Final[Mapping[str, object]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _ResolvedDeprecation:
    deprecation_date: date
    litellm_model: str | None
    litellm_provider: str | None
    successor_model: str | None = None


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


def _cost_map_lookup(model_key: object) -> _ResolvedDeprecation | None:
    if not isinstance(model_key, str) or not model_key:
        return None
    entry: Final = litellm.model_cost.get(model_key)
    if not isinstance(entry, Mapping):
        return None
    fields: Final[Mapping[str, object]] = entry
    parsed: Final = _parse_deprecation_date(fields.get("deprecation_date"))
    if parsed is None:
        return None
    provider: Final = fields.get("litellm_provider")
    return _ResolvedDeprecation(
        deprecation_date=parsed,
        litellm_model=model_key,
        litellm_provider=provider if isinstance(provider, str) else None,
        successor_model=_successor_model(fields.get("successor_model")),
    )


def _mapping_field(deployment: Mapping[str, object], key: str) -> Mapping[str, object]:
    value: Final = deployment.get(key)
    return value if isinstance(value, Mapping) else _NO_MODEL_METADATA


def _resolve_deployment_deprecation(
    deployment: Mapping[str, object],
) -> _ResolvedDeprecation | None:
    """Resolve a deployment's deprecation date and successor, preferring its explicit overrides field by field"""
    model_info: Final = _mapping_field(deployment, "model_info")
    raw_model: Final = _mapping_field(deployment, "litellm_params").get("model")
    unprefixed: Final = raw_model.split("/", 1)[1] if isinstance(raw_model, str) and "/" in raw_model else None
    from_cost_map: Final = next(
        (
            candidate
            for candidate in (
                _cost_map_lookup(model_info.get("base_model")),
                _cost_map_lookup(raw_model),
                _cost_map_lookup(unprefixed),
            )
            if candidate is not None
        ),
        None,
    )
    successor: Final = _successor_model(model_info.get("successor_model")) or (
        from_cost_map.successor_model if from_cost_map is not None else None
    )

    override: Final = _parse_deprecation_date(model_info.get("deprecation_date"))
    if override is not None:
        provider: Final = model_info.get("litellm_provider")
        return _ResolvedDeprecation(
            deprecation_date=override,
            litellm_model=raw_model if isinstance(raw_model, str) else None,
            litellm_provider=provider if isinstance(provider, str) else None,
            successor_model=successor,
        )
    return None if from_cost_map is None else replace(from_cost_map, successor_model=successor)


def _classify(days_until: int, warn_within_days: int) -> DeprecationStatus:
    if days_until < 0:
        return "deprecated"
    if days_until <= warn_within_days:
        return "imminent"
    return "upcoming"


def _build_info(deployment: Mapping[str, object], today: date, warn_within_days: int) -> ModelDeprecationInfo | None:
    model_name: Final = deployment.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        return None

    resolved: Final = _resolve_deployment_deprecation(deployment)
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
        successor_model=resolved.successor_model,
    )


def _dedupe(
    models: Sequence[ModelDeprecationInfo],
) -> tuple[ModelDeprecationInfo, ...]:
    """Report a model group carrying the same date on several deployments once, keeping one that names a successor"""
    ordered: Final = sorted(
        models, key=lambda model: (model.model_name, model.deprecation_date, model.successor_model is None)
    )
    return tuple(
        next(group) for _, group in groupby(ordered, key=lambda model: (model.model_name, model.deprecation_date))
    )


def _bucket(models: Sequence[ModelDeprecationInfo], status: DeprecationStatus) -> tuple[ModelDeprecationInfo, ...]:
    return tuple(
        sorted(
            (model for model in models if model.status == status),
            key=lambda model: model.deprecation_date,
        )
    )


def collect_model_deprecations(
    llm_router: Router | None,
    warn_within_days: int = DEFAULT_DEPRECATION_WARN_DAYS,
    today: date | None = None,
) -> ModelDeprecationResponse:
    """Bucket every deployment carrying a deprecation date by how urgent it is"""
    snapshot_time: Final = datetime.now(timezone.utc)
    effective_today: Final = today or snapshot_time.date()
    deployments: Final = (llm_router.get_model_list() or ()) if llm_router is not None else ()

    deduped: Final = _dedupe(
        tuple(
            info
            for info in (_build_info(deployment, effective_today, warn_within_days) for deployment in deployments)
            if info is not None
        )
    )

    verbose_logger.debug(
        "model_deprecation: %d/%d deployments carry a deprecation date",
        len(deduped),
        len(deployments),
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
