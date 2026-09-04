from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from .config import (
    AutoSetupCandidate,
    AutoSetupConfig,
    AutoSetupObjective,
    AutoSetupQualityLevel,
    AutoSetupTierPolicy,
    ComplexityRouterConfig,
)

SnapshotComplexity = Literal["trivial", "simple", "standard", "complex"]

_SNAPSHOT_FILENAME: Final = "auto_router_snapshot_v0.json"
_TIER_COMPLEXITIES: Final[Mapping[str, SnapshotComplexity]] = MappingProxyType(
    {
        "SIMPLE": "trivial",
        "MEDIUM": "simple",
        "COMPLEX": "standard",
        "REASONING": "complex",
    }
)
_EASY_COMPLEXITIES: Final = frozenset(("trivial", "simple"))
_VERSION_SEPARATOR: Final = re.compile(r"(?<=\d)\.(?=\d)")
_SNAPSHOT_PAYLOAD_ADAPTER: Final = TypeAdapter(dict[str, object])


class SnapshotQualityTier(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    maximum_quality_regret: float = Field(ge=0, le=1)


class SnapshotIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_model_id: str
    family_id: str
    is_routable: bool
    litellm_model_keys: tuple[str, ...]
    reasoning_effort: str | None
    required_parameters: Mapping[str, object]


class SnapshotQualityProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_model_id: str
    complexity: SnapshotComplexity
    completion_probability: float = Field(ge=0, le=1)
    cost_per_completed_task_usd: float | None = Field(default=None, ge=0)
    mean_input_tokens: float = Field(ge=0)
    mean_output_tokens: float = Field(ge=0)
    mean_request_cost_usd: float | None = Field(default=None, ge=0)
    quality_lower_bound: float = Field(ge=0, le=1)
    quality_score: float = Field(ge=0, le=1)
    question_count: int = Field(gt=0)
    source_input_cost_per_token: float = Field(ge=0)
    source_output_cost_per_token: float = Field(ge=0)
    source_id: str
    subtask_count: int = Field(gt=0)


class SnapshotSpeedProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_count: int = Field(gt=0)
    benchmark_model_id: str
    cohort_id: str
    complexity: SnapshotComplexity
    cost_per_completed_task_usd: float | None = Field(default=None, ge=0)
    excluded_infrastructure_trials: int = Field(ge=0)
    mean_attempt_cost_usd: float | None = Field(default=None, ge=0)
    mean_cache_read_input_tokens: float | None = Field(default=None, ge=0)
    mean_output_tokens: float | None = Field(default=None, ge=0)
    mean_uncached_input_tokens: float | None = Field(default=None, ge=0)
    observed_duration_p50_ms: float = Field(ge=0)
    observed_duration_p95_ms: float = Field(ge=0)
    quality_lower_bound: float = Field(ge=0, le=1)
    retry_adjusted_completion_ms: float | None = Field(default=None, ge=0)
    source_id: str
    success_probability: float = Field(ge=0, le=1)
    task_count: int = Field(gt=0)
    terminal_bench_model_id: str


class SnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_model_id: str
    identity: SnapshotIdentity
    quality_by_complexity: Mapping[SnapshotComplexity, SnapshotQualityProfile]
    task_completion_speed_by_complexity: Mapping[SnapshotComplexity, tuple[SnapshotSpeedProfile, ...]]


class AutoRouterSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    activation_scope: Mapping[str, object]
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    complexity_policy: Mapping[str, object]
    definitions: Mapping[str, str]
    generated_at: str
    limitations: tuple[str, ...]
    models: Mapping[str, SnapshotModel]
    provenance: Mapping[str, object]
    quality_tiers: Mapping[AutoSetupQualityLevel, SnapshotQualityTier]
    reference_routes: Mapping[str, object]
    schema_version: Literal["2.1"]
    snapshot_id: str
    task_complexities: Mapping[str, Mapping[str, SnapshotComplexity]]

    @model_validator(mode="after")
    def _validate_policy(self) -> AutoRouterSnapshot:
        if frozenset(self.quality_tiers) != frozenset(("economy", "balanced", "high", "max")):
            raise ValueError("Auto Router snapshot must define economy, balanced, high, and max quality levels")
        if self.activation_scope.get("setup_mode") != "auto":
            raise ValueError("Auto Router snapshot is not scoped to Auto setup")
        expected_complexities: Final = frozenset(_TIER_COMPLEXITIES.values())
        seen_aliases: dict[str, str] = {}
        for model_id, model in self.models.items():
            if model_id != model.benchmark_model_id or model.identity.benchmark_model_id != model_id:
                raise ValueError(f"Auto Router snapshot model identity mismatch for {model_id}")
            if frozenset(model.quality_by_complexity) != expected_complexities:
                raise ValueError(f"Auto Router snapshot model {model_id} lacks complete quality evidence")
            if frozenset(model.task_completion_speed_by_complexity) != expected_complexities:
                raise ValueError(f"Auto Router snapshot model {model_id} lacks complete speed buckets")
            if model.identity.is_routable != bool(model.identity.litellm_model_keys):
                raise ValueError(f"Auto Router snapshot routability mismatch for {model_id}")
            for complexity in expected_complexities:
                quality = model.quality_by_complexity[complexity]
                if quality.benchmark_model_id != model_id or quality.complexity != complexity:
                    raise ValueError(f"Auto Router snapshot quality identity mismatch for {model_id}/{complexity}")
                for speed in model.task_completion_speed_by_complexity[complexity]:
                    token_values = (
                        speed.mean_uncached_input_tokens,
                        speed.mean_cache_read_input_tokens,
                        speed.mean_output_tokens,
                    )
                    if any(value is None for value in token_values) != all(value is None for value in token_values):
                        raise ValueError(f"Auto Router snapshot has partial token evidence for {model_id}/{complexity}")
                    if speed.benchmark_model_id != model_id or speed.complexity != complexity:
                        raise ValueError(f"Auto Router snapshot speed identity mismatch for {model_id}/{complexity}")
            for alias in model.identity.litellm_model_keys:
                normalized = _normalized_model_key(alias)
                owner = seen_aliases.setdefault(normalized, model.identity.family_id)
                if owner != model.identity.family_id:
                    raise ValueError(f"Auto Router snapshot alias {alias!r} maps to multiple model families")
        regrets = tuple(
            self.quality_tiers[name].maximum_quality_regret for name in ("economy", "balanced", "high", "max")
        )
        if regrets != tuple(sorted(regrets, reverse=True)):
            raise ValueError("Auto Router snapshot quality regret must tighten from economy through max")
        return self


class AutoSetupDeploymentPricing(BaseModel):
    """The token rates LiteLLM will actually bill for one configured deployment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_cost_per_token: float = Field(ge=0)
    output_cost_per_token: float = Field(ge=0)
    cache_read_input_token_cost: float | None = Field(default=None, ge=0)
    input_cost_per_token_above_128k_tokens: float | None = Field(default=None, ge=0)
    output_cost_per_token_above_128k_tokens: float | None = Field(default=None, ge=0)
    cache_read_input_token_cost_above_128k_tokens: float | None = Field(default=None, ge=0)
    input_cost_per_token_above_200k_tokens: float | None = Field(default=None, ge=0)
    output_cost_per_token_above_200k_tokens: float | None = Field(default=None, ge=0)
    cache_read_input_token_cost_above_200k_tokens: float | None = Field(default=None, ge=0)


class AutoSetupDeployment(BaseModel):
    """One deployment inside a model group, kept separate so mixed groups fail closed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_refs: tuple[str, ...] = Field(min_length=1)
    pricing: AutoSetupDeploymentPricing | None = None


AutoSetupExclusionReason = Literal["no_benchmark_match", "mixed_model_group", "pricing_unavailable"]


@dataclass(frozen=True, slots=True)
class AutoSetupInventoryExclusion:
    model_group: str
    reason: AutoSetupExclusionReason


@dataclass(frozen=True, slots=True)
class _AvailableProfile:
    model_name: str
    benchmark_model_id: str
    cost_per_completed_task_usd: float
    quality: SnapshotQualityProfile
    speed: tuple[SnapshotSpeedProfile, ...]
    required_parameters: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ResolvedGroup:
    model_name: str
    matches: tuple[SnapshotModel, ...]
    deployments: tuple[AutoSetupDeployment, ...]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@lru_cache(maxsize=1)
def load_auto_router_snapshot() -> AutoRouterSnapshot:
    raw: Final = _SNAPSHOT_PAYLOAD_ADAPTER.validate_json(
        files("litellm.router_strategy.complexity_router")
        .joinpath("artifacts", _SNAPSHOT_FILENAME)
        .read_text(encoding="utf-8")
    )
    expected = raw.get("artifact_sha256")
    if not isinstance(expected, str):
        raise TypeError("Auto Router snapshot must carry an artifact_sha256")
    unsigned: Final = {  # mutable-ok: json.dumps requires a transient concrete JSON object
        key: value for key, value in raw.items() if key != "artifact_sha256"
    }
    actual: Final = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if expected != actual:
        raise ValueError(f"Auto Router snapshot hash mismatch: expected {expected}, calculated {actual}")
    return AutoRouterSnapshot.model_validate(raw)


def _normalized_model_key(value: str) -> str:
    return _VERSION_SEPARATOR.sub("-", value.strip().casefold())


def _matching_snapshot_models(snapshot: AutoRouterSnapshot, model_refs: Sequence[str]) -> tuple[SnapshotModel, ...]:
    normalized_refs: Final = frozenset(_normalized_model_key(ref) for ref in model_refs)
    return tuple(
        model
        for model in snapshot.models.values()
        if model.identity.is_routable
        and normalized_refs.intersection(_normalized_model_key(key) for key in model.identity.litellm_model_keys)
    )


def _resolve_groups(
    snapshot: AutoRouterSnapshot,
    available_model_deployments: Mapping[str, Sequence[AutoSetupDeployment]],
) -> tuple[tuple[_ResolvedGroup, ...], tuple[AutoSetupInventoryExclusion, ...]]:
    resolved: list[_ResolvedGroup] = []
    excluded: list[AutoSetupInventoryExclusion] = []
    for model_name, deployments_value in available_model_deployments.items():
        deployments = tuple(deployments_value)
        if not deployments:
            excluded.append(AutoSetupInventoryExclusion(model_name, "no_benchmark_match"))
            continue
        match_sets = [
            {model.benchmark_model_id: model for model in _matching_snapshot_models(snapshot, deployment.model_refs)}
            for deployment in deployments
        ]
        if any(len({model.identity.family_id for model in matches.values()}) > 1 for matches in match_sets):
            excluded.append(AutoSetupInventoryExclusion(model_name, "mixed_model_group"))
            continue
        if any(not matches for matches in match_sets):
            excluded.append(AutoSetupInventoryExclusion(model_name, "no_benchmark_match"))
            continue
        common_ids = set(match_sets[0]).intersection(*(set(matches) for matches in match_sets[1:]))
        if not common_ids:
            excluded.append(AutoSetupInventoryExclusion(model_name, "mixed_model_group"))
            continue
        resolved.append(
            _ResolvedGroup(
                model_name=model_name,
                matches=tuple(match_sets[0][model_id] for model_id in sorted(common_ids)),
                deployments=deployments,
            )
        )
    return tuple(resolved), tuple(excluded)


def analyze_auto_setup_inventory(
    snapshot: AutoRouterSnapshot,
    available_model_deployments: Mapping[str, Sequence[AutoSetupDeployment]],
) -> tuple[tuple[str, ...], tuple[AutoSetupInventoryExclusion, ...]]:
    """Return identity-safe groups and explicit reasons for every excluded group."""

    groups, exclusions = _resolve_groups(snapshot, available_model_deployments)
    eligible: list[str] = []
    mutable_exclusions = list(exclusions)
    for group in groups:
        if any(
            _quality_completion_cost(model.quality_by_complexity[complexity], group.deployments) is not None
            for model in group.matches
            for complexity in _TIER_COMPLEXITIES.values()
        ):
            eligible.append(group.model_name)
        else:
            mutable_exclusions.append(AutoSetupInventoryExclusion(group.model_name, "pricing_unavailable"))
    return tuple(eligible), tuple(mutable_exclusions)


def _rate_for_input_size(
    pricing: AutoSetupDeploymentPricing,
    field: Literal["input", "output", "cache"],
    input_tokens: float,
) -> float:
    base = {
        "input": pricing.input_cost_per_token,
        "output": pricing.output_cost_per_token,
        "cache": pricing.cache_read_input_token_cost,
    }[field]
    if field == "cache" and base is None:
        base = pricing.input_cost_per_token
    if input_tokens > 200_000:
        above_200k = {
            "input": pricing.input_cost_per_token_above_200k_tokens,
            "output": pricing.output_cost_per_token_above_200k_tokens,
            "cache": pricing.cache_read_input_token_cost_above_200k_tokens,
        }[field]
        if above_200k is not None:
            return above_200k
    if input_tokens > 128_000:
        above_128k = {
            "input": pricing.input_cost_per_token_above_128k_tokens,
            "output": pricing.output_cost_per_token_above_128k_tokens,
            "cache": pricing.cache_read_input_token_cost_above_128k_tokens,
        }[field]
        if above_128k is not None:
            return above_128k
    if base is None:
        raise ValueError(f"Auto setup pricing lacks a base {field} rate")
    return base


def _base_rate(
    pricing: AutoSetupDeploymentPricing,
    field: Literal["input", "output", "cache"],
) -> float:
    if field == "input":
        return pricing.input_cost_per_token
    if field == "output":
        return pricing.output_cost_per_token
    if pricing.cache_read_input_token_cost is not None:
        return pricing.cache_read_input_token_cost
    return pricing.input_cost_per_token


def _quality_completion_cost(
    profile: SnapshotQualityProfile,
    deployments: Sequence[AutoSetupDeployment],
) -> float | None:
    source_bill = (
        profile.mean_input_tokens * profile.source_input_cost_per_token
        + profile.mean_output_tokens * profile.source_output_cost_per_token
    )
    if source_bill <= 0 or profile.completion_probability <= 0 or profile.mean_request_cost_usd is None:
        return None
    costs: list[float] = []
    for deployment in deployments:
        pricing = deployment.pricing
        if pricing is None:
            return None
        deployed_bill = profile.mean_input_tokens * _rate_for_input_size(
            pricing, "input", profile.mean_input_tokens
        ) + profile.mean_output_tokens * _rate_for_input_size(pricing, "output", profile.mean_input_tokens)
        adjusted_request_cost = profile.mean_request_cost_usd * deployed_bill / source_bill
        costs.append(adjusted_request_cost / profile.completion_probability)
    return max(costs, default=None)


def _speed_completion_cost(
    speed: SnapshotSpeedProfile,
    deployments: Sequence[AutoSetupDeployment],
) -> float | None:
    if (
        speed.success_probability <= 0
        or speed.mean_uncached_input_tokens is None
        or speed.mean_cache_read_input_tokens is None
        or speed.mean_output_tokens is None
    ):
        return None
    costs: list[float] = []
    for deployment in deployments:
        pricing = deployment.pricing
        if pricing is None:
            return None
        attempt_cost = (
            speed.mean_uncached_input_tokens * _base_rate(pricing, "input")
            + speed.mean_cache_read_input_tokens * _base_rate(pricing, "cache")
            + speed.mean_output_tokens * _base_rate(pricing, "output")
        )
        costs.append(attempt_cost / speed.success_probability)
    return max(costs, default=None)


def _available_profiles(
    snapshot: AutoRouterSnapshot,
    available_model_deployments: Mapping[str, Sequence[AutoSetupDeployment]],
    complexity: SnapshotComplexity,
) -> tuple[_AvailableProfile, ...]:
    profiles: Final[list[_AvailableProfile]] = []  # mutable-ok: local accumulator is frozen into the returned tuple
    groups, _ = _resolve_groups(snapshot, available_model_deployments)
    for group in groups:
        best = min(
            group.matches,
            key=lambda model: (
                -model.quality_by_complexity[complexity].quality_lower_bound,
                (
                    model.quality_by_complexity[complexity].cost_per_completed_task_usd
                    if model.quality_by_complexity[complexity].cost_per_completed_task_usd is not None
                    else math.inf
                ),
                model.benchmark_model_id,
            ),
        )
        cost = _quality_completion_cost(best.quality_by_complexity[complexity], group.deployments)
        if cost is None:
            continue
        priced_speed = tuple(
            speed.model_copy(update={"cost_per_completed_task_usd": _speed_completion_cost(speed, group.deployments)})
            for speed in best.task_completion_speed_by_complexity[complexity]
        )
        profiles.append(
            _AvailableProfile(
                model_name=group.model_name,
                benchmark_model_id=best.benchmark_model_id,
                cost_per_completed_task_usd=cost,
                quality=best.quality_by_complexity[complexity],
                speed=priced_speed,
                required_parameters=best.identity.required_parameters,
            )
        )
    return tuple(profiles)


def _cost_rank(profiles: Sequence[_AvailableProfile]) -> tuple[_AvailableProfile, ...]:
    return tuple(
        sorted(
            profiles,
            key=lambda profile: (
                profile.cost_per_completed_task_usd,
                -profile.quality.quality_lower_bound,
                profile.model_name,
            ),
        )
    )


def _normalize_log(value: float, values: Sequence[float]) -> float:
    logged: Final = tuple(math.log(max(item, 1e-12)) for item in values)
    low: Final = min(logged)
    high: Final = max(logged)
    if low == high:
        return 0.0
    return (math.log(max(value, 1e-12)) - low) / (high - low)


def _hard_task_rank(
    profiles: Sequence[_AvailableProfile], objective: AutoSetupObjective
) -> tuple[tuple[_AvailableProfile, ...], str]:
    cost_ranked: Final = _cost_rank(profiles)
    speed_profiles: Final = tuple(
        (profile, speed)
        for profile in profiles
        for speed in profile.speed
        if speed.retry_adjusted_completion_ms is not None
    )
    if not speed_profiles:
        return cost_ranked, "fallback_no_speed_evidence"
    cohorts: dict[  # mutable-ok: local cohort index never escapes
        str, dict[str, tuple[_AvailableProfile, SnapshotSpeedProfile]]
    ] = {}  # mutable-ok: local cohort index is populated before selection
    for profile, speed in speed_profiles:
        by_model = cohorts.setdefault(
            speed.cohort_id,
            {},  # mutable-ok: local cohort index needs an empty per-cohort bucket
        )
        current = by_model.get(profile.model_name)
        if current is None or speed.attempt_count > current[1].attempt_count:
            by_model[profile.model_name] = (profile, speed)
    measured_profiles: Final = tuple(
        {  # mutable-ok: transient dict provides deterministic model-name deduplication
            profile.model_name: profile for profile, _ in speed_profiles
        }.values()
    )
    anchor: Final = max(
        measured_profiles,
        key=lambda profile: (profile.quality.quality_lower_bound, -profile.cost_per_completed_task_usd),
    )
    anchor_cohorts: Final = tuple(cohort_id for cohort_id, by_model in cohorts.items() if anchor.model_name in by_model)
    selected_cohort = max(
        anchor_cohorts,
        key=lambda cohort_id: (
            len(cohorts[cohort_id]),
            sum(speed.attempt_count for _, speed in cohorts[cohort_id].values()),
            cohort_id,
        ),
    )
    comparable = tuple(cohorts[selected_cohort].values())
    if objective == "task_completion_speed":
        return tuple(
            profile
            for profile, _ in sorted(
                comparable,
                key=lambda item: (
                    item[1].retry_adjusted_completion_ms or math.inf,
                    -item[0].quality.quality_lower_bound,
                    item[0].model_name,
                ),
            )
        ), "cohort_scoped"
    balanced_comparable: list[  # mutable-ok: local typed accumulator is consumed immediately
        tuple[_AvailableProfile, float, float]
    ] = []  # mutable-ok: local typed accumulator is frozen by the return expression
    for profile, speed in comparable:
        cost = speed.cost_per_completed_task_usd
        completion_ms = speed.retry_adjusted_completion_ms
        if cost is not None and completion_ms is not None:
            balanced_comparable.append((profile, cost, completion_ms))
    if not balanced_comparable:
        return cost_ranked, "fallback_no_comparable_cost_evidence"
    costs: Final = tuple(cost for _, cost, _ in balanced_comparable)
    times: Final = tuple(completion_ms for _, _, completion_ms in balanced_comparable)
    return tuple(
        profile
        for profile, _, _ in sorted(
            balanced_comparable,
            key=lambda item: (
                math.hypot(
                    _normalize_log(item[1], costs),
                    _normalize_log(item[2], times),
                ),
                -item[0].quality.quality_lower_bound,
                item[0].model_name,
            ),
        )
    ), "cohort_scoped"


def _policy_candidate(profile: _AvailableProfile) -> AutoSetupCandidate:
    return AutoSetupCandidate(
        model_name=profile.model_name,
        benchmark_model_id=profile.benchmark_model_id,
        quality_lower_bound=profile.quality.quality_lower_bound,
        cost_per_completed_task_usd=profile.cost_per_completed_task_usd,
    )


def _tier_policy(
    profiles: Sequence[_AvailableProfile],
    complexity: SnapshotComplexity,
    objective: AutoSetupObjective,
) -> AutoSetupTierPolicy:
    cost_ranked: Final = _cost_rank(profiles)
    if objective == "cost":
        return AutoSetupTierPolicy(
            selection_mode="snapshot_ranked",
            candidates=tuple(_policy_candidate(profile) for profile in cost_ranked),
            evidence_status="direct",
        )
    if complexity in _EASY_COMPLEXITIES:
        return AutoSetupTierPolicy(
            selection_mode="runtime_response_latency",
            candidates=tuple(_policy_candidate(profile) for profile in cost_ranked),
            cold_start_model=cost_ranked[0].model_name,
            evidence_status="runtime_response_speed_required",
        )
    ranked, evidence_status = _hard_task_rank(profiles, objective)
    return AutoSetupTierPolicy(
        selection_mode="snapshot_ranked",
        candidates=tuple(_policy_candidate(profile) for profile in ranked),
        evidence_status=evidence_status,
    )


def build_auto_setup_config(
    *,
    snapshot: AutoRouterSnapshot,
    available_model_deployments: Mapping[str, Sequence[AutoSetupDeployment]],
    quality_level: AutoSetupQualityLevel,
    optimize_for: AutoSetupObjective,
) -> ComplexityRouterConfig:
    """Recompute quality gates and rankings over only the caller's available groups."""

    maximum_regret: Final = snapshot.quality_tiers[quality_level].maximum_quality_regret
    policies: Final[  # mutable-ok: Pydantic config assembly is local to this builder
        dict[str, AutoSetupTierPolicy]
    ] = {}  # mutable-ok: Pydantic config assembly is local to this builder
    tiers: Final[dict[str, list[str]]] = {}  # mutable-ok: Pydantic config assembly is local to this builder
    tier_model_configs: Final[  # mutable-ok: Pydantic config assembly is local to this builder
        dict[str, list[dict[str, object]]]
    ] = {}  # mutable-ok: Pydantic config assembly is local to this builder
    for tier_name, complexity in _TIER_COMPLEXITIES.items():
        profiles = _available_profiles(snapshot, available_model_deployments, complexity)
        if not profiles:
            raise ValueError(f"None of the available model groups has {complexity} evidence in this snapshot")
        best_quality = max(profile.quality.quality_lower_bound for profile in profiles)
        floor = max(0.0, best_quality - maximum_regret)
        admitted = tuple(profile for profile in profiles if profile.quality.quality_lower_bound >= floor)
        policy = _tier_policy(admitted, complexity, optimize_for)
        policies[tier_name] = policy
        tiers[tier_name] = [  # mutable-ok: ComplexityRouterConfig owns this JSON-shaped tier list
            candidate.model_name for candidate in policy.candidates
        ]
        parameterized: list[dict[str, object]] = [  # mutable-ok: Pydantic validates and owns these JSON-shaped rows
            {  # mutable-ok: each row is a Pydantic input payload
                "model_name": profile.model_name,
                "litellm_params": dict(  # mutable-ok: Pydantic needs a concrete JSON-shaped params object
                    profile.required_parameters
                ),
            }
            for profile in admitted
            if profile.model_name in frozenset(tiers[tier_name]) and profile.required_parameters
        ]
        if parameterized:
            tier_model_configs[tier_name] = parameterized

    return ComplexityRouterConfig.model_validate(
        {  # mutable-ok: top-level Pydantic input must be a JSON-shaped mapping
            "tiers": tiers,
            "tier_model_configs": tier_model_configs,
            "classifier_type": "heuristic_v2",
            "auto_setup": AutoSetupConfig(
                snapshot_id=snapshot.snapshot_id,
                snapshot_sha256=snapshot.artifact_sha256,
                quality_level=quality_level,
                optimize_for=optimize_for,
                tier_policies=policies,
            ).model_dump(mode="json"),
        }
    )
