"""Static production-debt analysis for a LiteLLM proxy config.yaml.

Nothing in the proxy validates a config for these patterns today:
`router_settings`/`litellm_settings` are only checked for unknown keys
against `Router.__init__`'s parameter list (see `proxy_server.py`'s
`router_settings` handling), never for whether the *combination* of
values is safe. Each of these compiles and runs fine, then turns into a
real incident once a deployment starts failing or a spend cap was never
set:

  - a deployment with no fallback and no default_fallbacks: one failing
    deployment means requests for that model_name just error out
  - a high num_retries with no allowed_fails/allowed_fails_policy: a
    persistently-failing deployment gets retried on every single
    request instead of cooling down, multiplying cost and latency
  - a deployment with no max_budget and no global litellm_settings
    max_budget: spend on it has no automatic cutoff
  - a deployment pointing at a model litellm's own pricing data already
    marks as deprecated
"""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict


class Severity(str, Enum):
    """Risk tier of a `Finding`.

    CRITICAL: can directly cause a cost or availability incident on its
        own (e.g. retries multiplying spend against a dead deployment).
    WARNING: a real gap, but one a team may have intentionally covered
        another way (provider-side billing alerts, accepting a hard
        failure instead of a fallback).
    """

    CRITICAL = "critical"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Finding:
    rule: str
    severity: Severity
    model_name: str | None
    message: str


_SEVERITY_WEIGHT: Final[Mapping[Severity, int]] = MappingProxyType(
    {Severity.CRITICAL: 10, Severity.WARNING: 3}
)
_SEVERITY_ORDER: Final[Mapping[Severity, int]] = MappingProxyType(
    {Severity.CRITICAL: 0, Severity.WARNING: 1}
)

# Below this, we don't flag num_retries: a couple of retries on a
# transient error is normal and not, by itself, a cost-multiplication
# risk worth surfacing.
_HIGH_RETRY_THRESHOLD: Final = 3


class LiteLLMParamsShape(TypedDict, total=False):
    model: ReadOnly[str]
    num_retries: ReadOnly[int]
    max_budget: ReadOnly[float]


class DeploymentShape(TypedDict):
    model_name: ReadOnly[str]
    litellm_params: ReadOnly[LiteLLMParamsShape]


FallbackEntry: TypeAlias = Mapping[str, Sequence[str]]


class SettingsShape(TypedDict, total=False):
    fallbacks: ReadOnly[Sequence[FallbackEntry]]
    context_window_fallbacks: ReadOnly[Sequence[FallbackEntry]]
    content_policy_fallbacks: ReadOnly[Sequence[FallbackEntry]]
    default_fallbacks: ReadOnly[Sequence[str]]
    num_retries: ReadOnly[int]
    allowed_fails: ReadOnly[int]
    allowed_fails_policy: ReadOnly[Mapping[str, int]]
    max_budget: ReadOnly[float]


class ProxyConfigShape(TypedDict, total=False):
    model_list: ReadOnly[Sequence[DeploymentShape]]
    litellm_settings: ReadOnly[SettingsShape]
    router_settings: ReadOnly[SettingsShape]


ModelCost: TypeAlias = Mapping[str, Mapping[str, object]]

with warnings.catch_warnings():
    # pydantic warns that ReadOnly (PEP 705) isn't enforced against mutation at
    # runtime, which is expected here: we use it only for the static contract.
    warnings.simplefilter("ignore", UserWarning)
    _CONFIG_ADAPTER: Final[TypeAdapter[ProxyConfigShape]] = TypeAdapter(ProxyConfigShape)


def validate_config(raw: object) -> ProxyConfigShape:
    """Validate an arbitrary loaded config.yaml value into `ProxyConfigShape`.

    `ProxyConfig.get_config` returns a loosely-typed `dict`; this is the
    boundary where that untyped value is validated into the shape every
    detector below assumes. Raises `pydantic.ValidationError` on a
    malformed config.
    """
    return _CONFIG_ADAPTER.validate_python(raw)


def production_debt_score(findings: Sequence[Finding]) -> int:
    """Severity-weighted sum of findings (critical=10, warning=3).

    A simple, transparent, reproducible heuristic for trending a
    config's structural health over time -- not a standardized metric.
    """
    return sum(_SEVERITY_WEIGHT[f.severity] for f in findings)


def _model_names(config: ProxyConfigShape) -> frozenset[str]:
    """Real (non-wildcard) model_name values declared in model_list."""
    return frozenset(
        entry["model_name"]
        for entry in config.get("model_list") or ()
        if "*" not in entry["model_name"]
    )


def _fallback_source_keys(fallback_list: Sequence[FallbackEntry] | None) -> frozenset[str]:
    """Source model_names covered by a fallbacks-shaped list.

    `fallbacks`/`context_window_fallbacks`/`content_policy_fallbacks`
    all share the same shape: [{"model_name": ["fallback_1", ...]}, ...]
    (confirmed against Router.__init__ and the example proxy config).
    """
    return frozenset(key for entry in fallback_list or () for key in entry)


def _fallback_target_names(fallback_list: Sequence[FallbackEntry] | None) -> frozenset[str]:
    """Model names appearing as a fallback *target* anywhere in the list.

    A model whose sole purpose is to serve as another model's fallback
    destination is not itself missing coverage just because nothing
    falls back to it in turn.
    """
    return frozenset(
        target for entry in fallback_list or () for targets in entry.values() for target in targets
    )


def _global_settings(config: ProxyConfigShape) -> SettingsShape:
    """Merge litellm_settings and router_settings.

    Both accept fallbacks/num_retries/allowed_fails/max_budget (the
    proxy validates router_settings keys against Router.__init__'s own
    parameter list), so a value set in either place is equally real.
    router_settings wins on overlap since it is the more specific,
    router-scoped block.
    """
    litellm_settings: Final = config.get("litellm_settings")
    router_settings: Final = config.get("router_settings")
    merged: Final[SettingsShape] = {
        **(litellm_settings or {}),
        **(router_settings or {}),
    }
    return merged


def detect_missing_fallback_coverage(config: ProxyConfigShape) -> tuple[Finding, ...]:
    settings: Final = _global_settings(config)
    if settings.get("default_fallbacks"):
        # A catch-all default_fallbacks covers every model_group.
        return ()

    covered: Final = (
        _fallback_source_keys(settings.get("fallbacks"))
        | _fallback_source_keys(settings.get("context_window_fallbacks"))
        | _fallback_source_keys(settings.get("content_policy_fallbacks"))
        | _fallback_target_names(settings.get("fallbacks"))
        | _fallback_target_names(settings.get("context_window_fallbacks"))
        | _fallback_target_names(settings.get("content_policy_fallbacks"))
    )

    return tuple(
        Finding(
            rule="no-fallback-coverage",
            severity=Severity.WARNING,
            model_name=name,
            message=(
                f"model_name '{name}' has no fallbacks, context_window_fallbacks, "
                "content_policy_fallbacks, or default_fallbacks entry -- if every "
                "deployment behind it fails, requests for it error out with nothing "
                "to fall back to"
            ),
        )
        for name in sorted(_model_names(config))
        if name not in covered
    )


def detect_retry_without_cooldown(config: ProxyConfigShape) -> tuple[Finding, ...]:
    settings: Final = _global_settings(config)
    has_cooldown: Final = (
        settings.get("allowed_fails") is not None or settings.get("allowed_fails_policy") is not None
    )
    if has_cooldown:
        return ()

    global_retries: Final = settings.get("num_retries")
    global_findings: Final = (
        (
            Finding(
                rule="retry-without-cooldown",
                severity=Severity.CRITICAL,
                model_name=None,
                message=(
                    f"global num_retries={global_retries} is set with no "
                    "allowed_fails/allowed_fails_policy configured -- a "
                    "persistently-failing deployment is retried on every request "
                    "instead of being cooled down, multiplying cost and latency "
                    "on every call that hits it"
                ),
            ),
        )
        if isinstance(global_retries, int) and global_retries >= _HIGH_RETRY_THRESHOLD
        else ()
    )

    deployment_findings: Final = tuple(
        Finding(
            rule="retry-without-cooldown",
            severity=Severity.CRITICAL,
            model_name=entry["model_name"],
            message=(
                f"deployment '{entry['model_name']}' sets num_retries={retries} with "
                "no allowed_fails/allowed_fails_policy configured anywhere -- a "
                "persistently-failing deployment is retried on every request instead "
                "of being cooled down, multiplying cost and latency on every call "
                "that hits it"
            ),
        )
        for entry in config.get("model_list") or ()
        for retries in (entry["litellm_params"].get("num_retries"),)
        if isinstance(retries, int) and retries >= _HIGH_RETRY_THRESHOLD
    )

    return global_findings + deployment_findings


def detect_missing_budget_cap(config: ProxyConfigShape) -> tuple[Finding, ...]:
    settings: Final = _global_settings(config)
    if settings.get("max_budget") is not None:
        return ()

    return tuple(
        Finding(
            rule="missing-deployment-budget-cap",
            severity=Severity.WARNING,
            model_name=entry["model_name"],
            message=(
                f"deployment '{entry['model_name']}' has no max_budget, and no global "
                "litellm_settings.max_budget is set either -- spend on this "
                "deployment has no automatic cutoff"
            ),
        )
        for entry in config.get("model_list") or ()
        if "*" not in entry["model_name"] and entry["litellm_params"].get("max_budget") is None
    )


def _parse_date(value: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def detect_deprecated_model_references(
    config: ProxyConfigShape,
    model_cost: ModelCost,
    today: datetime.date | None = None,
) -> tuple[Finding, ...]:
    """Flag deployments whose underlying model litellm's own pricing data
    already marks as deprecated (`litellm.model_cost[...]["deprecation_date"]`,
    a real field litellm ships and uses for cost/capability data)."""
    resolved_today: Final = today if today is not None else datetime.date.today()

    return tuple(
        Finding(
            rule="deprecated-model-reference",
            severity=Severity.WARNING,
            model_name=entry["model_name"],
            message=(
                f"deployment '{entry['model_name']}' points at '{model}', deprecated "
                f"since {dep_date_str} per litellm's own model pricing data"
            ),
        )
        for entry in config.get("model_list") or ()
        for model in (entry["litellm_params"].get("model"),)
        if isinstance(model, str)
        for info in (model_cost.get(model),)
        if isinstance(info, Mapping)
        for dep_date_str in (info.get("deprecation_date"),)
        if isinstance(dep_date_str, str)
        for dep_date in (_parse_date(dep_date_str),)
        if dep_date is not None and dep_date <= resolved_today
    )


def analyze(config: ProxyConfigShape, model_cost: ModelCost | None = None) -> tuple[Finding, ...]:
    """Run all detectors and return every finding, most-critical first.

    `model_cost` is optional (pass `litellm.model_cost` for the
    deprecated-model-reference check); the other three detectors need
    only the config itself.
    """
    deprecated_findings: Final = (
        detect_deprecated_model_references(config, model_cost) if model_cost is not None else ()
    )
    findings: Final = (
        *detect_retry_without_cooldown(config),
        *detect_missing_budget_cap(config),
        *detect_missing_fallback_coverage(config),
        *deprecated_findings,
    )
    return tuple(sorted(findings, key=lambda f: (_SEVERITY_ORDER[f.severity], f.rule, f.model_name or "")))
