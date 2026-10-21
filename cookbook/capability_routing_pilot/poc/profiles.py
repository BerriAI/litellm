import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from itertools import groupby
from pathlib import Path
from typing import Final

from poc.policy import beta_cell

SUPPORTED_CATEGORIES: Final = (
    "repository_debugging",
    "feature_implementation",
)
PUBLIC_DEV_TENANT: Final = "public_dev_poc"
ALIAS_NOTE: Final = (
    "Model aliases can resolve to changing deployments; returned model versions and deployment IDs are isolated"
)
BETA_NOTE: Final = (
    "Beta(1,1) credible bound describes the exchangeable public-dev task distribution for this category, "
    "not guaranteed per-task success"
)
MIN_EVIDENCE_NOTE: Final = (
    "necessary not sufficient: 13/13 is the minimum perfect cell clearing 0.80; not a general adequacy claim"
)


class SupportStatus(str, Enum):
    NO_EVIDENCE = "no_evidence"
    UNSUPPORTED_ZERO_SUCCESS = "unsupported_zero_success"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EVIDENCED_BELOW_TARGET = "evidenced_below_target"
    QUALIFIED = "qualified"


class DecisionStatus(str, Enum):
    SELECTED = "selected"
    NO_QUALIFIED_MODEL = "no_qualified_model"
    UNSUPPORTED_CATEGORY = "unsupported_category"


@dataclass(frozen=True, slots=True, order=True)
class ProfileKey:
    tenant_id: str
    category: str
    model_alias: str
    returned_model_version: str
    deployment_id: str
    harness_version: str
    budget_version: str
    harness_config_hash: str


@dataclass(frozen=True, slots=True)
class ProfileVersions:
    harness_version: str
    budget_version: str
    harness_config_hash: str
    model_versions: tuple[tuple[str, str], ...]
    model_deployments: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    harness_version: str
    budget_version: str
    per_run_budget_usd: float
    step_limit: int
    wall_timeout_s: int
    mini_swe_agent_version: str
    guard_sha256: str


@dataclass(frozen=True, slots=True)
class TaskAttempt:
    task_id: str
    key: ProfileKey
    is_infra: bool
    success: bool | None
    cost_usd: float
    latency_s: float
    primary_label: str
    secondary_label: str | None
    manifest_sha256: str
    harness_sha256: str

    def __post_init__(self) -> None:
        if not self.task_id or not self.primary_label:
            raise ValueError("task_id and primary_label must be non-empty")
        if not math.isfinite(self.cost_usd) or self.cost_usd < 0:
            raise ValueError("cost_usd must be finite and non-negative")
        if not math.isfinite(self.latency_s) or self.latency_s < 0:
            raise ValueError("latency_s must be finite and non-negative")
        if self.is_infra and self.success is not None:
            raise ValueError("infrastructure attempts cannot have a graded success value")
        if not self.is_infra and not isinstance(self.success, bool):
            raise ValueError("graded attempts require a boolean success value")


@dataclass(frozen=True, slots=True)
class ProfileProvenance:
    manifest_sha256: str
    harness_sha256: str
    model_alias: str
    deployment_id: str | None
    alias_note: str


@dataclass(frozen=True, slots=True)
class Profile:
    key: ProfileKey
    attempt_count: int
    graded_count: int
    infra_count: int
    successes: int
    raw_rate: float | None
    beta_mean: float
    lower_5pct: float
    total_spend_usd: float
    cost_per_graded_task_usd: float | None
    mean_latency_s: float
    primary_labels: tuple[str, ...]
    secondary_labels: tuple[str, ...]
    status: SupportStatus
    tau: float
    min_evidence: int
    provenance: ProfileProvenance


@dataclass(frozen=True, slots=True)
class Selection:
    decision: DecisionStatus
    selected_profile: Profile | None
    threshold_unmet: bool
    reason: str


def harness_config_sha256(config: HarnessConfig) -> str:
    canonical_json: Final = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode()).hexdigest()


def _record_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("each task attempt must be a JSON object")
    return value


def _required_string(row: Mapping[str, object], field: str) -> str:
    value: Final = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_number(row: Mapping[str, object], field: str) -> float:
    value: Final = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    return float(value)


def attempt_from_mapping(value: object) -> TaskAttempt:
    row: Final = _record_mapping(value)
    secondary: Final = row.get("secondary_label")
    success: Final = row.get("success")
    infra: Final = row.get("is_infra")
    deployment_id_value: Final = row.get("deployment_id")
    if not isinstance(infra, bool):
        raise ValueError("is_infra must be a boolean")
    if secondary is not None and not isinstance(secondary, str):
        raise ValueError("secondary_label must be a string or null")
    if success is not None and not isinstance(success, bool):
        raise ValueError("success must be a boolean or null")
    if deployment_id_value is not None and not isinstance(deployment_id_value, str):
        raise ValueError("deployment_id must be a string or null")
    key: Final = ProfileKey(
        tenant_id=_required_string(row, "tenant_id"),
        category=_required_string(row, "category"),
        model_alias=_required_string(row, "model_alias"),
        returned_model_version=_required_string(row, "returned_model_version"),
        deployment_id=deployment_id_value if isinstance(deployment_id_value, str) else "",
        harness_version=_required_string(row, "harness_version"),
        budget_version=_required_string(row, "budget_version"),
        harness_config_hash=_required_string(row, "harness_config_hash"),
    )
    return TaskAttempt(
        task_id=_required_string(row, "task_id"),
        key=key,
        is_infra=infra,
        success=success,
        cost_usd=_required_number(row, "cost_usd"),
        latency_s=_required_number(row, "latency_s"),
        primary_label=_required_string(row, "primary_label"),
        secondary_label=secondary,
        manifest_sha256=_required_string(row, "manifest_sha256"),
        harness_sha256=_required_string(row, "harness_sha256"),
    )


def read_attempts_jsonl(path: Path) -> tuple[TaskAttempt, ...]:
    return tuple(
        attempt_from_mapping(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    )


def _status(
    graded_count: int,
    successes: int,
    lower_5pct: float,
    tau: float,
    min_evidence: int,
) -> SupportStatus:
    if graded_count == 0:
        return SupportStatus.NO_EVIDENCE
    if successes == 0:
        return SupportStatus.UNSUPPORTED_ZERO_SUCCESS
    if graded_count < min_evidence:
        return SupportStatus.INSUFFICIENT_EVIDENCE
    if lower_5pct < tau:
        return SupportStatus.EVIDENCED_BELOW_TARGET
    return SupportStatus.QUALIFIED


def _aggregate(key: ProfileKey, attempts: tuple[TaskAttempt, ...], tau: float, min_evidence: int) -> Profile:
    graded: Final = tuple(attempt for attempt in attempts if not attempt.is_infra)
    graded_task_ids: Final = tuple(attempt.task_id for attempt in graded)
    if len(graded_task_ids) != len(set(graded_task_ids)):
        raise ValueError(f"duplicate graded attempts for {(key, graded_task_ids)}")
    manifest_shas: Final = frozenset(attempt.manifest_sha256 for attempt in attempts)
    harness_shas: Final = frozenset(attempt.harness_sha256 for attempt in attempts)
    if len(manifest_shas) != 1 or len(harness_shas) != 1:
        raise ValueError(f"profile provenance changed for {key}")
    successes: Final = sum(attempt.success is True for attempt in graded)
    evidence: Final = beta_cell(successes, len(graded), 0.90, None)
    total_spend: Final = sum(attempt.cost_usd for attempt in attempts)
    return Profile(
        key=key,
        attempt_count=len(attempts),
        graded_count=len(graded),
        infra_count=sum(attempt.is_infra for attempt in attempts),
        successes=successes,
        raw_rate=successes / len(graded) if graded else None,
        beta_mean=evidence.posterior_mean,
        lower_5pct=evidence.ci_low,
        total_spend_usd=total_spend,
        cost_per_graded_task_usd=total_spend / len(graded) if graded else None,
        mean_latency_s=sum(attempt.latency_s for attempt in attempts) / len(attempts),
        primary_labels=tuple(sorted({attempt.primary_label for attempt in attempts})),
        secondary_labels=tuple(
            sorted({attempt.secondary_label for attempt in attempts if attempt.secondary_label is not None})
        ),
        status=_status(len(graded), successes, evidence.ci_low, tau, min_evidence),
        tau=tau,
        min_evidence=min_evidence,
        provenance=ProfileProvenance(
            manifest_sha256=next(iter(manifest_shas)),
            harness_sha256=next(iter(harness_shas)),
            model_alias=key.model_alias,
            deployment_id=key.deployment_id or None,
            alias_note=ALIAS_NOTE,
        ),
    )


def aggregate_profiles(
    attempts: Sequence[TaskAttempt],
    tau: float = 0.80,
    min_evidence: int = 13,
) -> tuple[Profile, ...]:
    if not 0 <= tau <= 1 or min_evidence < 1:
        raise ValueError("tau must be within [0, 1] and min_evidence must be positive")
    ordered: Final = tuple(sorted(attempts, key=lambda attempt: attempt.key))
    return tuple(
        _aggregate(key, tuple(group), tau, min_evidence)
        for key, group in groupby(ordered, key=lambda attempt: attempt.key)
    )


def _versions_match(profile: Profile, versions: ProfileVersions) -> bool:
    expected_model_version: Final = dict(versions.model_versions).get(profile.key.model_alias)
    expected_deployment_id: Final = dict(versions.model_deployments).get(profile.key.model_alias)
    return (
        profile.key.harness_version == versions.harness_version
        and profile.key.budget_version == versions.budget_version
        and profile.key.harness_config_hash == versions.harness_config_hash
        and profile.key.returned_model_version == expected_model_version
        and (expected_deployment_id is None or profile.key.deployment_id == expected_deployment_id)
    )


def _cheapest(profiles: Sequence[Profile]) -> Profile:
    return min(
        profiles,
        key=lambda profile: (
            profile.cost_per_graded_task_usd
            if profile.cost_per_graded_task_usd is not None
            else math.inf,
            profile.key.model_alias,
            profile.key.returned_model_version,
        ),
    )


def select(
    tenant_id: str,
    category: str,
    versions: ProfileVersions,
    profiles: Sequence[Profile],
    tau: float = 0.80,
    quality_first: bool = False,
    min_evidence: int = 13,
) -> Selection:
    if category not in SUPPORTED_CATEGORIES:
        return Selection(DecisionStatus.UNSUPPORTED_CATEGORY, None, False, "unsupported_category")
    matching: Final = tuple(
        profile
        for profile in profiles
        if profile.key.tenant_id == tenant_id
        and profile.key.category == category
        and _versions_match(profile, versions)
        and profile.tau == tau
        and profile.min_evidence == min_evidence
    )
    qualified: Final = tuple(profile for profile in matching if profile.status is SupportStatus.QUALIFIED)
    if qualified:
        return Selection(DecisionStatus.SELECTED, _cheapest(qualified), False, "cheapest_qualified")
    if quality_first:
        evidenced: Final = tuple(
            profile
            for profile in matching
            if profile.graded_count >= min_evidence and profile.successes > 0
        )
        if evidenced:
            best_lower_bound: Final = max(profile.lower_5pct for profile in evidenced)
            best: Final = _cheapest(
                tuple(profile for profile in evidenced if profile.lower_5pct == best_lower_bound)
            )
            return Selection(DecisionStatus.SELECTED, best, True, "quality_first_threshold_unmet")
    return Selection(DecisionStatus.NO_QUALIFIED_MODEL, None, False, "no_qualified_model")


def compile_map(
    tenant_id: str,
    versions: ProfileVersions,
    profiles: Sequence[Profile],
    tau: float = 0.80,
    min_evidence: int = 13,
    final: bool = False,
) -> dict[str, object]:
    return {
        "final": final,
        "tenant_id": tenant_id,
        "thresholds": {"tau": tau, "min_evidence": min_evidence},
        "min_evidence_note": MIN_EVIDENCE_NOTE,
        "interpretation": BETA_NOTE,
        "category_decisions": {
            "repository_debugging": "profiled",
            "feature_implementation": "profiled",
            "shell_build": DecisionStatus.UNSUPPORTED_CATEGORY.value,
            "algorithms_data": DecisionStatus.UNSUPPORTED_CATEGORY.value,
            "unknown": DecisionStatus.UNSUPPORTED_CATEGORY.value,
        },
        "versions": {
            "harness_version": versions.harness_version,
            "budget_version": versions.budget_version,
            "harness_config_hash": versions.harness_config_hash,
            "model_versions": dict(versions.model_versions),
            "model_deployments": dict(versions.model_deployments),
        },
        "profiles": [
            asdict(profile)
            for profile in sorted(profiles, key=lambda item: item.key)
            if profile.key.tenant_id == tenant_id
        ],
    }


def compile_map_from_profiling_runs(
    path: Path,
    tenant_id: str,
    versions: ProfileVersions,
    tau: float = 0.80,
    min_evidence: int = 13,
) -> dict[str, object]:
    if path.name != "profiling_runs.jsonl" or path.is_symlink():
        raise ValueError("capability maps may only read profiling_runs.jsonl")
    profiles: Final = aggregate_profiles(read_attempts_jsonl(path), tau, min_evidence)
    return compile_map(tenant_id, versions, profiles, tau, min_evidence, final=False)
