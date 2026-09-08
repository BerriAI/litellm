"""Baseline-relative license gate for heuristic-v1 complexity-router tuning."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Final

from pydantic import ValidationError

from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig

TUNING_BASELINE_PARAM_NAME: Final = "auto_router_tuning_baseline_v2"

HEURISTIC_V1_TUNING_FIELDS: Final = (
    "tiers",
    "tier_model_configs",
    "classifier_type",
    "tier_boundaries",
    "reasoning_override_min_score",
    "token_thresholds",
    "dimension_weights",
    "code_keywords",
    "reasoning_keywords",
    "technical_keywords",
    "custom_technical_keywords",
    "simple_keywords",
    "escalation_keywords",
    "keyword_tier_rules",
)

_TUNING_FIELD_SET: Final = frozenset(HEURISTIC_V1_TUNING_FIELDS)

_V1_SCORING_CLASSIFIER_TYPES: Final = frozenset({"heuristic", "heuristic_first", "hybrid"})
_AUTO_ROUTER_COMPLEXITY_PREFIX: Final = "auto_router/complexity_router"
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_EMPTY_TAGS: Final[tuple[str, ...]] = ()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else _EMPTY


def tuning_fingerprint(complexity_router_config: object) -> str | None:
    """Digest of explicitly supplied heuristic-v1 tuning fields, normalized, or None when invalid."""
    raw: Final = _mapping(complexity_router_config)
    try:
        validated: Final = ComplexityRouterConfig.model_validate(raw)
    except ValidationError:
        return None
    supplied: Final = ((_TUNING_FIELD_SET - frozenset(("tier_model_configs",))) & frozenset(raw)) | (
        frozenset(("tier_model_configs",)) if validated.tier_model_configs else frozenset()
    )
    payload: Final = validated.model_dump(mode="json", include=supplied)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


DEFAULT_TUNING_FINGERPRINT: Final = tuning_fingerprint(_EMPTY)


def uses_heuristic_v1(complexity_router_config: object) -> bool:
    """Whether a config's primary classifier path is the heuristic-v1 scorer."""
    return _mapping(complexity_router_config).get("classifier_type", "heuristic") in _V1_SCORING_CLASSIFIER_TYPES


def router_identity(deployment: Mapping[str, object]) -> str | None:
    """Stable identity for a complexity-router deployment, across tuning edits."""
    model_info: Final = _mapping(deployment.get("model_info"))
    model_id: Final = model_info.get("id")
    if model_info.get("db_model") is True and isinstance(model_id, str) and model_id:
        return f"db:{model_id}"
    model_name: Final = deployment.get("model_name")
    if not isinstance(model_name, str) or not model_name:
        return None
    litellm_params: Final = _mapping(deployment.get("litellm_params"))
    tags: Final = litellm_params.get("tags")
    normalized_tags: Final = (
        tuple(sorted(str(tag) for tag in tags))
        if isinstance(tags, Iterable) and not isinstance(tags, str)
        else _EMPTY_TAGS
    )
    return f"yaml:{json.dumps((model_name, normalized_tags), separators=(',', ':'))}"


def heuristic_v1_router_fingerprint(deployment: Mapping[str, object]) -> tuple[str, str] | None:
    """The identity/fingerprint pair for a heuristic-v1 complexity router, else None."""
    litellm_params: Final = _mapping(deployment.get("litellm_params"))
    model: Final = litellm_params.get("model")
    config: Final = litellm_params.get("complexity_router_config")
    if (
        not isinstance(model, str)
        or not model.startswith(_AUTO_ROUTER_COMPLEXITY_PREFIX)
        or not uses_heuristic_v1(config)
    ):
        return None
    identity: Final = router_identity(deployment)
    fingerprint: Final = tuning_fingerprint(config)
    if identity is None or fingerprint is None:
        return None
    return identity, fingerprint


def snapshot_tuning_baselines(deployments: Iterable[Mapping[str, object]]) -> Mapping[str, str]:
    """One immutable first-observation baseline for every heuristic-v1 complexity router."""
    return MappingProxyType(
        {
            identity: fingerprint
            for deployment in deployments
            if (pair := heuristic_v1_router_fingerprint(deployment)) is not None
            for identity, fingerprint in (pair,)
        }
    )  # mutable-ok: MappingProxyType owns the completed immutable snapshot


def is_mutable_tuned_candidate(candidate: Mapping[str, object], baselines: Mapping[str, str]) -> bool:
    pair: Final = heuristic_v1_router_fingerprint(candidate)
    if pair is None:
        return False
    identity, fingerprint = pair
    return fingerprint != baselines.get(identity, DEFAULT_TUNING_FINGERPRINT)


def mutable_tuned_identities(
    deployments: Iterable[Mapping[str, object]], baselines: Mapping[str, str]
) -> frozenset[str]:
    """Heuristic-v1 routers whose current tuning differs from their baseline or shipped default."""
    return frozenset(
        identity
        for deployment in deployments
        if (pair := heuristic_v1_router_fingerprint(deployment)) is not None
        for identity, _ in (pair,)
        if is_mutable_tuned_candidate(deployment, baselines)
    )


def tuning_limit_violation(*, held: int, limit: int | None) -> str | None:
    if limit is None or held <= limit:
        return None
    return (
        f"At most {limit} auto-router(s) with changed heuristic scorer settings or tier models can be modified "
        "without an auto-router license. Keep this router on its recorded settings, or revert the other changed "
        "router to its baseline, or remove one of them."
    )


def tuning_quota_violation(
    *,
    candidate: Mapping[str, object],
    others: Iterable[Mapping[str, object]],
    baselines: Mapping[str, str],
    limit: int | None,
) -> str | None:
    """Why a change to candidate tuning exceeds the baseline-relative free quota."""
    if limit is None:
        return None
    pair: Final = heuristic_v1_router_fingerprint(candidate)
    if pair is None:
        return None
    identity, _ = pair
    if not is_mutable_tuned_candidate(candidate, baselines):
        return None
    held: Final = mutable_tuned_identities(others, baselines) - frozenset((identity,))
    return tuning_limit_violation(held=len(held) + 1, limit=limit)
