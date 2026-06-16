"""Helpers for surfacing model deprecation/sunset information.

This module reads ``deprecation_date`` metadata that is bundled in
``model_prices_and_context_window.json`` (exposed at runtime via
``litellm.model_cost``) and classifies the proxy's configured models into
``upcoming``, ``imminent`` and ``deprecated`` buckets. It is the single
source of truth used by both the ``/model/deprecations`` endpoint and the
proactive Slack alert.

Resolution order for a deployment's deprecation date:

1. ``model_info.deprecation_date`` – an explicit override on the deployment.
2. ``model_info.base_model`` looked up in ``litellm.model_cost``.
3. The ``litellm_params.model`` string looked up in ``litellm.model_cost``.

Models without any deprecation metadata are skipped silently (most models
are not deprecated, and we don't want to pollute the response).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_logger
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_WARN_DAYS,
    ModelDeprecationInfo,
    ModelDeprecationResponse,
)

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    Router = _Router
else:
    Router = Any


def _parse_deprecation_date(raw_value: Any) -> Optional[date]:
    """Parse a ``deprecation_date`` string in YYYY-MM-DD form.

    Returns ``None`` for missing, malformed, or sentinel placeholder values
    (the JSON map ships a documentation sentinel of the form ``"date when..."``).
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, datetime):
        return raw_value.date()
    if isinstance(raw_value, date):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    try:
        return datetime.strptime(raw_value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _lookup_deprecation_date_from_cost_map(
    model_key: Optional[str],
) -> Tuple[Optional[date], Optional[str]]:
    """Look up a deprecation date in ``litellm.model_cost`` for ``model_key``.

    Returns a tuple of (deprecation_date, litellm_provider).
    """
    if not model_key:
        return None, None
    entry = litellm.model_cost.get(model_key)
    if not isinstance(entry, dict):
        return None, None
    return (
        _parse_deprecation_date(entry.get("deprecation_date")),
        entry.get("litellm_provider"),
    )


def _resolve_deployment_deprecation(
    deployment: Dict[str, Any],
) -> Tuple[Optional[date], Optional[str], Optional[str]]:
    """Resolve a deployment's deprecation metadata.

    Returns a tuple of (deprecation_date, litellm_model, litellm_provider).
    """
    model_info = deployment.get("model_info") or {}
    explicit = _parse_deprecation_date(model_info.get("deprecation_date"))
    if explicit is not None:
        litellm_params = deployment.get("litellm_params") or {}
        return (
            explicit,
            litellm_params.get("model"),
            model_info.get("litellm_provider"),
        )

    base_model = model_info.get("base_model")
    dep_date, provider = _lookup_deprecation_date_from_cost_map(base_model)
    if dep_date is not None:
        return dep_date, base_model, provider

    litellm_params = deployment.get("litellm_params") or {}
    raw_model = litellm_params.get("model")
    dep_date, provider = _lookup_deprecation_date_from_cost_map(raw_model)
    if dep_date is not None:
        return dep_date, raw_model, provider

    if isinstance(raw_model, str) and "/" in raw_model:
        # Try the un-prefixed lookup (e.g. "openai/gpt-4o" → "gpt-4o").
        bare = raw_model.split("/", 1)[1]
        dep_date, provider = _lookup_deprecation_date_from_cost_map(bare)
        if dep_date is not None:
            return dep_date, bare, provider

    return None, raw_model, model_info.get("litellm_provider")


def _classify(days_until: int, warn_within_days: int) -> str:
    if days_until < 0:
        return "deprecated"
    if days_until <= warn_within_days:
        return "imminent"
    return "upcoming"


def _model_dump_compat(deployment: Any) -> Dict[str, Any]:
    """Return a plain dict for both pydantic models and dicts."""
    if isinstance(deployment, dict):
        return deployment
    if hasattr(deployment, "model_dump"):
        return deployment.model_dump(exclude_none=True)
    if hasattr(deployment, "dict"):
        return deployment.dict()
    return dict(deployment)


def collect_model_deprecations(
    llm_router: Optional[Router],
    warn_within_days: int = DEFAULT_DEPRECATION_WARN_DAYS,
    today: Optional[date] = None,
) -> ModelDeprecationResponse:
    """Aggregate deprecation info for all deployments configured on the router.

    De-duplicates by ``(model_name, deprecation_date)`` so multi-deployment
    model groups (load-balanced across regions) only surface once per
    deprecation date.
    """
    snapshot_time = datetime.now(timezone.utc)
    today = today or snapshot_time.date()

    response = ModelDeprecationResponse(
        warn_within_days=warn_within_days,
        checked_at=snapshot_time,
    )

    if llm_router is None:
        return response

    seen: set = set()
    deployments = llm_router.get_model_list() or []
    for deployment in deployments:
        deployment_dict = _model_dump_compat(deployment)
        model_name = deployment_dict.get("model_name")
        if not model_name:
            continue

        dep_date, litellm_model, provider = _resolve_deployment_deprecation(
            deployment_dict
        )
        if dep_date is None:
            continue

        dedup_key = (model_name, dep_date.isoformat())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        days_until = (dep_date - today).days
        status = _classify(days_until, warn_within_days)

        info = ModelDeprecationInfo(
            model_name=model_name,
            litellm_model=litellm_model,
            deprecation_date=dep_date,
            days_until_deprecation=days_until,
            status=status,
            litellm_provider=provider,
        )

        if status == "deprecated":
            response.deprecated.append(info)
        elif status == "imminent":
            response.imminent.append(info)
        else:
            response.upcoming.append(info)

    response.deprecated.sort(key=lambda m: m.deprecation_date)
    response.imminent.sort(key=lambda m: m.deprecation_date)
    response.upcoming.sort(key=lambda m: m.deprecation_date)

    verbose_logger.debug(
        "model_deprecation: deprecated=%d imminent=%d upcoming=%d",
        len(response.deprecated),
        len(response.imminent),
        len(response.upcoming),
    )

    return response


def format_deprecation_alert_message(
    snapshot: ModelDeprecationResponse,
) -> Optional[str]:
    """Format a Slack-friendly alert message for the warning buckets.

    Only ``deprecated`` and ``imminent`` models are included; ``upcoming``
    models are intentionally omitted to avoid alert fatigue. Returns
    ``None`` when there is nothing to alert on.
    """
    if not snapshot.deprecated and not snapshot.imminent:
        return None

    lines: List[str] = ["*⚠️ Model Deprecation Warning*"]

    def _format_entry(info: ModelDeprecationInfo) -> str:
        suffix = (
            f"already deprecated {abs(info.days_until_deprecation)}d ago"
            if info.days_until_deprecation < 0
            else f"in {info.days_until_deprecation}d"
        )
        return (
            f"• `{info.model_name}` "
            f"(provider: {info.litellm_provider or 'unknown'}, "
            f"deprecates {info.deprecation_date.isoformat()} – {suffix})"
        )

    if snapshot.deprecated:
        lines.append("\n*Already deprecated:*")
        lines.extend(_format_entry(i) for i in snapshot.deprecated)

    if snapshot.imminent:
        lines.append(f"\n*Deprecating within {snapshot.warn_within_days} days:*")
        lines.extend(_format_entry(i) for i in snapshot.imminent)

    lines.append(
        "\nPlan migrations to a supported model. See "
        "https://docs.litellm.ai/docs/proxy/model_management for guidance."
    )

    return "\n".join(lines)
