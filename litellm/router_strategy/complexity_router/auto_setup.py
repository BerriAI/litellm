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

from .config import ComplexityRouterConfig

SnapshotComplexity = Literal["trivial", "simple", "standard", "complex"]
AutoSetupQualityLevel = Literal["economy", "balanced", "high", "max"]

_SNAPSHOT_FILENAME: Final = "auto_router_snapshot_v0.json"
_TIER_COMPLEXITIES: Final[Mapping[str, SnapshotComplexity]] = MappingProxyType(
    {
        "SIMPLE": "trivial",
        "MEDIUM": "simple",
        "COMPLEX": "standard",
        "REASONING": "complex",
    }
)
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


class SnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark_model_id: str
    identity: SnapshotIdentity
    quality_by_complexity: Mapping[SnapshotComplexity, SnapshotQualityProfile]


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
    schema_version: Literal["3.0"]
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
            if model.identity.is_routable != bool(model.identity.litellm_model_keys):
                raise ValueError(f"Auto Router snapshot routability mismatch for {model_id}")
            for complexity in expected_complexities:
                quality = model.quality_by_complexity[complexity]
                if quality.benchmark_model_id != model_id or quality.complexity != complexity:
                    raise ValueError(f"Auto Router snapshot quality identity mismatch for {model_id}/{complexity}")
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


AutoSetupExclusionReason = Literal["no_benchmark_match", "mixed_model_group"]


@dataclass(frozen=True, slots=True)
class AutoSetupInventoryExclusion:
    model_group: str
    reason: AutoSetupExclusionReason


@dataclass(frozen=True, slots=True)
class _AvailableProfile:
    model_name: str
    benchmark_model_id: str
    cost_per_completed_task_usd: float | None
    quality: SnapshotQualityProfile
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
    return tuple(group.model_name for group in groups), exclusions


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
        profiles.append(
            _AvailableProfile(
                model_name=group.model_name,
                benchmark_model_id=best.benchmark_model_id,
                cost_per_completed_task_usd=cost,
                quality=best.quality_by_complexity[complexity],
                required_parameters=best.identity.required_parameters,
            )
        )
    return tuple(profiles)


def _cost_rank(profiles: Sequence[_AvailableProfile]) -> tuple[_AvailableProfile, ...]:
    return tuple(
        sorted(
            profiles,
            key=lambda profile: (
                profile.cost_per_completed_task_usd is None,
                profile.cost_per_completed_task_usd if profile.cost_per_completed_task_usd is not None else math.inf,
                -profile.quality.quality_lower_bound,
                profile.model_name,
            ),
        )
    )


def build_auto_setup_config(
    *,
    snapshot: AutoRouterSnapshot,
    available_model_deployments: Mapping[str, Sequence[AutoSetupDeployment]],
    quality_level: AutoSetupQualityLevel,
) -> ComplexityRouterConfig:
    """Recompute quality gates and rankings over only the caller's available groups."""

    maximum_regret: Final = snapshot.quality_tiers[quality_level].maximum_quality_regret
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
        selected = _cost_rank(admitted)[0]
        tiers[tier_name] = [selected.model_name]
        parameterized: list[dict[str, object]] = (
            [  # mutable-ok: Pydantic validates and owns these JSON-shaped rows
                {  # mutable-ok: each row is a Pydantic input payload
                    "model_name": selected.model_name,
                    "litellm_params": dict(  # mutable-ok: Pydantic needs a concrete JSON-shaped params object
                        selected.required_parameters
                    ),
                }
            ]
            if selected.required_parameters
            else []
        )
        if parameterized:
            tier_model_configs[tier_name] = parameterized

    return ComplexityRouterConfig.model_validate(
        {  # mutable-ok: top-level Pydantic input must be a JSON-shaped mapping
            "tiers": tiers,
            "tier_model_configs": tier_model_configs,
            "classifier_type": "heuristic_v2",
        }
    )
