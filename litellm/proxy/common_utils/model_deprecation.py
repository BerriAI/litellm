from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def _cost_map_lookup(model_key: object) -> _ResolvedDeprecation | None:
    if not isinstance(model_key, str) or not model_key:
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
    )


def _mapping_field(deployment: Mapping[str, object], key: str) -> Mapping[str, object]:
    value: Final = deployment.get(key)
    return value if isinstance(value, Mapping) else _NO_MODEL_METADATA


def _resolve_deployment_deprecation(
    deployment: Mapping[str, object],
) -> _ResolvedDeprecation | None:
    """Resolve a deployment's deprecation date, preferring its explicit override"""
    model_info: Final = _mapping_field(deployment, "model_info")
    raw_model: Final = _mapping_field(deployment, "litellm_params").get("model")

    override: Final = _parse_deprecation_date(model_info.get("deprecation_date"))
    if override is not None:
        provider: Final = model_info.get("litellm_provider")
        return _ResolvedDeprecation(
            deprecation_date=override,
            litellm_model=raw_model if isinstance(raw_model, str) else None,
            litellm_provider=provider if isinstance(provider, str) else None,
        )

    unprefixed: Final = raw_model.split("/", 1)[1] if isinstance(raw_model, str) and "/" in raw_model else None
    return next(
        (
            resolved
            for resolved in (
                _cost_map_lookup(model_info.get("base_model")),
                _cost_map_lookup(raw_model),
                _cost_map_lookup(unprefixed),
            )
            if resolved is not None
        ),
        None,
    )


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
    )


def _dedupe(
    models: Sequence[ModelDeprecationInfo],
) -> tuple[ModelDeprecationInfo, ...]:
    """Report a model group carrying the same date on several deployments once"""
    ordered: Final = sorted(models, key=lambda model: (model.model_name, model.deprecation_date))
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
    return (
        f"• `{_escape_slack_mrkdwn(info.model_name)}` "
        f"(provider: {_escape_slack_mrkdwn(info.litellm_provider) if info.litellm_provider else 'unknown'}, "
        f"deprecates {info.deprecation_date.isoformat()}, {suffix})"
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
