import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from poc.profiles import (
    PUBLIC_DEV_TENANT,
    BETA_NOTE,
    MIN_EVIDENCE_NOTE,
    DecisionStatus,
    HarnessConfig,
    Profile,
    ProfileKey,
    ProfileVersions,
    SupportStatus,
    TaskAttempt,
    aggregate_profiles,
    compile_map,
    compile_map_from_profiling_runs,
    harness_config_sha256,
    select,
)

MODEL_VERSION = "anthropic/claude-haiku-4-5-20251001"
HARNESS_VERSION = "profile_harness_v1"
BUDGET_VERSION = "budget_1usd_50_600_v1"
CONFIG_HASH = "config-hash"
MANIFEST_SHA = "manifest-sha"
HARNESS_SHA = "harness-sha"


def _key(
    model_alias: str = "haiku",
    returned_model_version: str = MODEL_VERSION,
    tenant_id: str = PUBLIC_DEV_TENANT,
    category: str = "repository_debugging",
    harness_version: str = HARNESS_VERSION,
    budget_version: str = BUDGET_VERSION,
    harness_config_hash: str = CONFIG_HASH,
    deployment_id: str = "deployment-haiku",
) -> ProfileKey:
    return ProfileKey(
        tenant_id=tenant_id,
        category=category,
        model_alias=model_alias,
        returned_model_version=returned_model_version,
        deployment_id=deployment_id,
        harness_version=harness_version,
        budget_version=budget_version,
        harness_config_hash=harness_config_hash,
    )


def _attempts(
    model_alias: str,
    successes: int,
    graded_count: int,
    cost_per_attempt: float,
    **key_values: str,
) -> tuple[TaskAttempt, ...]:
    key = _key(model_alias=model_alias, **key_values)
    return tuple(
        TaskAttempt(
            task_id=f"task-{index}",
            key=key,
            is_infra=False,
            success=index < successes,
            cost_usd=cost_per_attempt,
            latency_s=float(index + 1),
            primary_label=key.category,
            secondary_label=None,
            manifest_sha256=MANIFEST_SHA,
            harness_sha256=HARNESS_SHA,
        )
        for index in range(graded_count)
    )


def _versions(
    models: tuple[tuple[str, str], ...] = (("haiku", MODEL_VERSION),),
    deployments: tuple[tuple[str, str], ...] = (("haiku", "deployment-haiku"),),
    harness_version: str = HARNESS_VERSION,
    budget_version: str = BUDGET_VERSION,
    harness_config_hash: str = CONFIG_HASH,
) -> ProfileVersions:
    return ProfileVersions(
        harness_version=harness_version,
        budget_version=budget_version,
        harness_config_hash=harness_config_hash,
        model_versions=models,
        model_deployments=deployments,
    )


def _profile(attempts: tuple[TaskAttempt, ...]) -> Profile:
    return aggregate_profiles(attempts)[0]


def test_cheapest_qualified_model_wins() -> None:
    profiles = aggregate_profiles(
        _attempts("haiku", 13, 13, 1.0)
        + _attempts(
            "sonnet",
            13,
            13,
            2.0,
            returned_model_version="anthropic/claude-sonnet-5",
            deployment_id="deployment-sonnet",
        )
    )
    decision = select(
        PUBLIC_DEV_TENANT,
        "repository_debugging",
        _versions(
            models=(("haiku", MODEL_VERSION), ("sonnet", "anthropic/claude-sonnet-5")),
            deployments=(("haiku", "deployment-haiku"), ("sonnet", "deployment-sonnet")),
        ),
        profiles,
    )
    assert decision.decision is DecisionStatus.SELECTED
    assert decision.selected_profile is not None
    assert decision.selected_profile.key.model_alias == "haiku"


def test_qualified_but_ineligible_cheaper_model_is_skipped() -> None:
    profiles = aggregate_profiles(
        _attempts("haiku", 13, 13, 1.0)
        + _attempts(
            "sonnet",
            13,
            13,
            2.0,
            returned_model_version="anthropic/claude-sonnet-5",
            deployment_id="deployment-sonnet",
        )
    )
    decision = select(
        PUBLIC_DEV_TENANT,
        "repository_debugging",
        _versions(
            models=(("haiku", MODEL_VERSION), ("sonnet", "anthropic/claude-sonnet-5")),
            deployments=(("haiku", "deployment-haiku"), ("sonnet", "deployment-sonnet")),
        ),
        profiles,
        eligible=frozenset({("sonnet", "deployment-sonnet")}),
    )
    assert decision.selected_profile is not None
    assert decision.selected_profile.key.model_alias == "sonnet"


def test_more_expensive_model_wins_when_cheaper_one_is_below_threshold() -> None:
    profiles = aggregate_profiles(
        _attempts("haiku", 12, 13, 0.1)
        + _attempts(
            "sonnet",
            13,
            13,
            2.0,
            returned_model_version="anthropic/claude-sonnet-5",
            deployment_id="deployment-sonnet",
        )
    )
    decision = select(
        PUBLIC_DEV_TENANT,
        "repository_debugging",
        _versions(
            models=(("haiku", MODEL_VERSION), ("sonnet", "anthropic/claude-sonnet-5")),
            deployments=(("haiku", "deployment-haiku"), ("sonnet", "deployment-sonnet")),
        ),
        profiles,
    )
    assert decision.decision is DecisionStatus.SELECTED
    assert decision.selected_profile is not None
    assert decision.selected_profile.key.model_alias == "sonnet"


def test_missing_zero_and_insufficient_evidence_are_not_selected() -> None:
    infra_only = TaskAttempt(
        task_id="infra-task",
        key=_key(),
        is_infra=True,
        success=None,
        cost_usd=0.5,
        latency_s=8.0,
        primary_label="repository_debugging",
        secondary_label=None,
        manifest_sha256=MANIFEST_SHA,
        harness_sha256=HARNESS_SHA,
    )
    no_evidence = _profile((infra_only,))
    zero_success = _profile(_attempts("haiku", 0, 13, 0.1))
    insufficient = _profile(_attempts("haiku", 12, 12, 0.1))
    assert no_evidence.status is SupportStatus.NO_EVIDENCE
    assert zero_success.status is SupportStatus.UNSUPPORTED_ZERO_SUCCESS
    assert insufficient.status is SupportStatus.INSUFFICIENT_EVIDENCE
    assert (
        select(
            PUBLIC_DEV_TENANT,
            "repository_debugging",
            _versions(),
            (),
        ).decision
        is DecisionStatus.NO_QUALIFIED_MODEL
    )
    assert (
        select(
            PUBLIC_DEV_TENANT,
            "repository_debugging",
            _versions(),
            (no_evidence, zero_success, insufficient),
        ).decision
        is DecisionStatus.NO_QUALIFIED_MODEL
    )


def test_infra_attempts_add_cost_and_latency_but_not_graded_count() -> None:
    key = _key()
    infra = TaskAttempt(
        task_id="infra-task",
        key=key,
        is_infra=True,
        success=None,
        cost_usd=0.5,
        latency_s=3.0,
        primary_label="repository_debugging",
        secondary_label=None,
        manifest_sha256=MANIFEST_SHA,
        harness_sha256=HARNESS_SHA,
    )
    failed = TaskAttempt(
        task_id="graded-task",
        key=key,
        is_infra=False,
        success=False,
        cost_usd=1.5,
        latency_s=5.0,
        primary_label="repository_debugging",
        secondary_label="shell_build",
        manifest_sha256=MANIFEST_SHA,
        harness_sha256=HARNESS_SHA,
    )
    profile = _profile((infra, failed))
    assert profile.attempt_count == 2
    assert profile.graded_count == 1
    assert profile.infra_count == 1
    assert profile.successes == 0
    assert profile.raw_rate == 0.0
    assert profile.total_spend_usd == 2.0
    assert profile.cost_per_graded_task_usd == 2.0
    assert profile.mean_latency_s == 4.0
    assert profile.status is SupportStatus.UNSUPPORTED_ZERO_SUCCESS


def test_duplicate_graded_attempt_for_task_and_key_is_rejected() -> None:
    attempt = _attempts("haiku", 1, 1, 1.0)[0]
    duplicate = TaskAttempt(
        task_id=attempt.task_id,
        key=attempt.key,
        is_infra=False,
        success=False,
        cost_usd=0.2,
        latency_s=1.0,
        primary_label=attempt.primary_label,
        secondary_label=None,
        manifest_sha256=MANIFEST_SHA,
        harness_sha256=HARNESS_SHA,
    )
    with pytest.raises(ValueError, match="duplicate graded attempts"):
        aggregate_profiles((attempt, duplicate))


def test_beta_bound_uses_closed_form_for_perfect_cells_and_evidence_floor() -> None:
    profile_12 = _profile(_attempts("haiku", 12, 12, 1.0))
    profile_13 = _profile(_attempts("haiku", 13, 13, 1.0))
    profile_12_of_13 = _profile(_attempts("haiku", 12, 13, 1.0))
    assert profile_12.lower_5pct == pytest.approx(0.05 ** (1 / 13))
    assert profile_12.status is SupportStatus.INSUFFICIENT_EVIDENCE
    assert profile_13.lower_5pct == pytest.approx(0.05 ** (1 / 14))
    assert profile_13.status is SupportStatus.QUALIFIED
    assert profile_12_of_13.lower_5pct < 0.80
    assert profile_12_of_13.status is SupportStatus.EVIDENCED_BELOW_TARGET


def test_every_support_status_is_emitted() -> None:
    infra_only = TaskAttempt(
        task_id="infra-only",
        key=_key(),
        is_infra=True,
        success=None,
        cost_usd=0.2,
        latency_s=1.0,
        primary_label="repository_debugging",
        secondary_label=None,
        manifest_sha256=MANIFEST_SHA,
        harness_sha256=HARNESS_SHA,
    )
    profiles = (
        _profile((infra_only,)),
        _profile(_attempts("haiku", 0, 13, 1.0)),
        _profile(_attempts("haiku", 1, 12, 1.0)),
        _profile(_attempts("haiku", 12, 13, 1.0)),
        _profile(_attempts("haiku", 13, 13, 1.0)),
    )
    assert {profile.status for profile in profiles} == set(SupportStatus)


def test_unsupported_categories_return_unsupported_category() -> None:
    for category in ("unknown", "shell_build", "algorithms_data"):
        decision = select(PUBLIC_DEV_TENANT, category, _versions(), ())
        assert decision.decision is DecisionStatus.UNSUPPORTED_CATEGORY
        assert decision.selected_profile is None


def test_tenant_model_harness_budget_and_deployment_versions_are_isolated() -> None:
    base = _attempts("haiku", 13, 13, 2.0)
    competitors = (
        _attempts("haiku", 13, 13, 0.1, tenant_id="other"),
        _attempts("haiku", 13, 13, 0.1, returned_model_version="other-model"),
        _attempts("haiku", 13, 13, 0.1, harness_version="other-harness"),
        _attempts("haiku", 13, 13, 0.1, budget_version="other-budget"),
        _attempts("haiku", 13, 13, 0.1, harness_config_hash="other-config"),
        _attempts("haiku", 13, 13, 0.1, deployment_id="other-deployment"),
    )
    profiles = aggregate_profiles(base + tuple(item for group in competitors for item in group))
    decision = select(PUBLIC_DEV_TENANT, "repository_debugging", _versions(), profiles)
    assert decision.decision is DecisionStatus.SELECTED
    assert decision.selected_profile is not None
    assert decision.selected_profile.cost_per_graded_task_usd == 2.0


def test_alias_and_returned_model_versions_split_profiles() -> None:
    attempts = (
        _attempts("haiku", 1, 1, 1.0)
        + _attempts(
            "haiku",
            1,
            1,
            1.0,
            returned_model_version="anthropic/claude-haiku-4-5-20251002",
        )
        + _attempts("haiku-canary", 1, 1, 1.0)
    )
    profiles = aggregate_profiles(attempts)
    assert len(profiles) == 3
    assert len({profile.key.returned_model_version for profile in profiles}) == 2
    assert len({profile.key.model_alias for profile in profiles}) == 2


def test_quality_first_is_disabled_by_default_and_only_uses_evidenced_nonzero_profiles() -> None:
    twelve_successes = _attempts("haiku", 12, 13, 1.0)
    zero_successes = _attempts(
        "sonnet",
        0,
        13,
        0.01,
        returned_model_version="anthropic/claude-sonnet-5",
        deployment_id="deployment-sonnet",
    )
    insufficient = _attempts(
        "opus",
        11,
        12,
        0.02,
        returned_model_version="anthropic/claude-opus-5",
        deployment_id="deployment-opus",
    )
    profiles = aggregate_profiles(twelve_successes + zero_successes + insufficient)
    versions = _versions(
        models=(
            ("haiku", MODEL_VERSION),
            ("sonnet", "anthropic/claude-sonnet-5"),
            ("opus", "anthropic/claude-opus-5"),
        ),
        deployments=(
            ("haiku", "deployment-haiku"),
            ("sonnet", "deployment-sonnet"),
            ("opus", "deployment-opus"),
        ),
    )
    default_decision = select(PUBLIC_DEV_TENANT, "repository_debugging", versions, profiles)
    assert default_decision.decision is DecisionStatus.NO_QUALIFIED_MODEL
    quality_decision = select(
        PUBLIC_DEV_TENANT,
        "repository_debugging",
        versions,
        profiles,
        quality_first=True,
    )
    assert quality_decision.decision is DecisionStatus.SELECTED
    assert quality_decision.selected_profile is not None
    assert quality_decision.selected_profile.key.model_alias == "haiku"
    assert quality_decision.threshold_unmet
    assert quality_decision.reason == "quality_first_threshold_unmet"


def test_capability_map_is_intermediate_and_contains_bound_and_version_notes() -> None:
    profiles = aggregate_profiles(_attempts("haiku", 13, 13, 1.0))
    capability_map = compile_map(PUBLIC_DEV_TENANT, _versions(), profiles)
    assert capability_map["final"] is False
    assert capability_map["min_evidence_note"] == MIN_EVIDENCE_NOTE
    assert capability_map["interpretation"] == BETA_NOTE
    assert capability_map["category_decisions"]["shell_build"]["decision"] == "unsupported_category"
    assert capability_map["category_decisions"]["algorithms_data"]["decision"] == "unsupported_category"
    assert capability_map["category_coverage"]["repository_debugging"]["status_counts"]["qualified"] == 1
    assert capability_map["versions"]["harness_config_hash"] == CONFIG_HASH


def test_map_builder_checks_manifest_membership_and_digest(tmp_path: Path) -> None:
    versions = _versions()
    manifest = tmp_path / "manifest.json"
    manifest_bytes = json.dumps({"categories": {"repository_debugging": {"selected_ids": ["task-0"]}}}).encode()
    manifest.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    attempt = _attempts("haiku", 1, 1, 0.1)[0]
    attempt_row = {
        **asdict(attempt),
        **asdict(attempt.key),
        "manifest_sha256": manifest_sha256,
    }
    profiling_runs = tmp_path / "profiling_runs.jsonl"
    profiling_runs.write_text(json.dumps(attempt_row) + "\n")

    capability_map = compile_map_from_profiling_runs(
        profiling_runs,
        manifest,
        manifest_sha256,
        PUBLIC_DEV_TENANT,
        versions,
    )

    assert capability_map["profiles"][0]["graded_count"] == 1
    assert capability_map["final"] is False
    with pytest.raises(ValueError, match="manifest sha256 mismatch"):
        compile_map_from_profiling_runs(
            profiling_runs,
            manifest,
            "wrong-digest",
            PUBLIC_DEV_TENANT,
            versions,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("task_id", "outside-manifest", "not selected"),
        ("manifest_sha256", "wrong-digest", "attempt manifest sha256 mismatch"),
    ),
)
def test_map_builder_rejects_attempt_membership_or_digest_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest_bytes = json.dumps({"categories": {"repository_debugging": {"selected_ids": ["task-0"]}}}).encode()
    manifest.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    attempt = _attempts("haiku", 1, 1, 0.1)[0]
    attempt_row = {
        **asdict(attempt),
        **asdict(attempt.key),
        "manifest_sha256": manifest_sha256,
        field: value,
    }
    profiling_runs = tmp_path / "profiling_runs.jsonl"
    profiling_runs.write_text(json.dumps(attempt_row) + "\n")

    with pytest.raises(ValueError, match=message):
        compile_map_from_profiling_runs(
            profiling_runs,
            manifest,
            manifest_sha256,
            PUBLIC_DEV_TENANT,
            _versions(),
        )


def test_harness_hash_is_canonical_json_over_the_full_budget_config() -> None:
    config = HarnessConfig(
        harness_version="run_pilot_attempt_v1",
        budget_version="budget_1usd_50_600_v1",
        per_run_budget_usd=1.0,
        step_limit=50,
        wall_timeout_s=600,
        mini_swe_agent_version="2.4.6",
        guard_sha256="guard-sha",
    )
    assert harness_config_sha256(config) == harness_config_sha256(config)
    assert harness_config_sha256(config) != harness_config_sha256(
        HarnessConfig(
            harness_version="run_pilot_attempt_v1",
            budget_version="budget_1usd_50_600_v1",
            per_run_budget_usd=1.0,
            step_limit=50,
            wall_timeout_s=601,
            mini_swe_agent_version="2.4.6",
            guard_sha256="guard-sha",
        )
    )
