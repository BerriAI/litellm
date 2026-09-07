"""Behavior pins for the baseline-relative heuristic-v1 tuning gate."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from litellm.router_utils.auto_router_tuning_baseline import (
    DEFAULT_TUNING_FINGERPRINT,
    HEURISTIC_V1_TUNING_FIELDS,
    heuristic_v1_router_fingerprint,
    mutable_tuned_identities,
    router_identity,
    snapshot_tuning_baselines,
    tuning_fingerprint,
    tuning_limit_violation,
    tuning_quota_violation,
)

_TIERS = {"SIMPLE": "cheap", "MEDIUM": "mid", "COMPLEX": "strong", "REASONING": "top"}
_ALT_TIERS = {**_TIERS, "COMPLEX": "other-strong"}


def _router(
    name: str,
    config: Mapping[str, object] | None,
    *,
    tags: list[str] | None = None,
    db_id: str | None = None,
    model: str = "auto_router/complexity_router",
) -> dict[str, object]:
    litellm_params: dict[str, object] = {"model": model}
    if config is not None:
        litellm_params["complexity_router_config"] = dict(config)
    if tags is not None:
        litellm_params["tags"] = tags
    row: dict[str, object] = {"model_name": name, "litellm_params": litellm_params}
    if db_id is not None:
        row["model_info"] = {"id": db_id, "db_model": True}
    return row


class TestTuningFingerprint:
    def test_normalized_spellings_share_one_fingerprint(self) -> None:
        canonical = tuning_fingerprint({"tiers": _TIERS, "dimension_weights": {"codePresence": 0.3}})
        assert canonical == tuning_fingerprint({"dimension_weights": {"codePresence": 0.3}, "tiers": _TIERS})
        assert tuning_fingerprint({"tiers": {"SIMPLE": {"model_name": "x"}}}) == tuning_fingerprint(
            {"tiers": {"SIMPLE": "x"}}
        )

    @pytest.mark.parametrize("field", sorted(set(HEURISTIC_V1_TUNING_FIELDS) - {"tier_model_configs"}))
    def test_every_tuning_field_changes_the_fingerprint(self, field: str) -> None:
        samples: dict[str, object] = {
            "tiers": _ALT_TIERS,
            "classifier_type": "heuristic_first",
            "tier_boundaries": {"simple_medium": 0.2, "medium_complex": 0.4, "complex_reasoning": 0.7},
            "reasoning_override_min_score": 0.05,
            "token_thresholds": {"simple": 20, "complex": 500},
            "dimension_weights": {"codePresence": 0.9},
            "code_keywords": ["orionflow"],
            "reasoning_keywords": ["deduce"],
            "technical_keywords": ["ledgerkit"],
            "custom_technical_keywords": ["acmeflow"],
            "simple_keywords": ["hey"],
            "escalation_keywords": ["ESCALATE"],
            "keyword_tier_rules": [{"keywords": ["urgent"], "tier": "COMPLEX"}],
        }
        config: dict[str, object] = {field: samples[field]}
        if field == "classifier_type":
            config["heuristic_first_max_tier"] = "MEDIUM"
            config["classifier_llm_config"] = {"model": "judge"}
        assert tuning_fingerprint(config) != DEFAULT_TUNING_FINGERPRINT

    def test_explicit_empty_tier_model_configs_follow_omission(self) -> None:
        assert tuning_fingerprint({"tier_model_configs": {}}) == DEFAULT_TUNING_FINGERPRINT

    def test_tier_model_overrides_change_the_fingerprint(self) -> None:
        plain = tuning_fingerprint({"tiers": {"SIMPLE": "x"}})
        with_override = tuning_fingerprint(
            {"tiers": {"SIMPLE": {"model_name": "x", "litellm_params": {"temperature": 0.1}}}}
        )
        assert plain != with_override

    def test_non_tuning_fields_do_not_change_the_fingerprint(self) -> None:
        assert (
            tuning_fingerprint({"return_raw_model_name": True, "session_affinity": True}) == DEFAULT_TUNING_FINGERPRINT
        )

    def test_invalid_config_has_no_fingerprint(self) -> None:
        assert tuning_fingerprint({"tier_boundaries": "not-a-mapping"}) is None

    def test_a_release_changing_a_shipped_default_is_not_an_operator_edit(self, monkeypatch) -> None:
        """An omitted setting follows the shipped default and stays off the quota when that default moves;
        only what the operator wrote is fingerprinted, so an explicit value equal to the old default still counts."""
        import litellm.router_strategy.complexity_router.config as config_module

        untouched = _router("a", {})
        tiers_only = _router("b", {"tiers": _TIERS})
        pinned = _router("c", {"dimension_weights": dict(config_module.DEFAULT_DIMENSION_WEIGHTS)})
        baselines = snapshot_tuning_baselines([untouched, tiers_only, pinned])
        assert tuning_fingerprint(pinned["litellm_params"]["complexity_router_config"]) != DEFAULT_TUNING_FINGERPRINT

        monkeypatch.setattr(
            config_module,
            "DEFAULT_DIMENSION_WEIGHTS",
            {**config_module.DEFAULT_DIMENSION_WEIGHTS, "codePresence": 0.99},
        )
        monkeypatch.setattr(
            config_module,
            "DEFAULT_TIER_BOUNDARIES",
            {**config_module.DEFAULT_TIER_BOUNDARIES, "simple_medium": 0.42},
        )

        assert tuning_fingerprint({}) == DEFAULT_TUNING_FINGERPRINT
        assert mutable_tuned_identities([untouched, tiers_only, pinned], baselines) == frozenset()
        assert mutable_tuned_identities([untouched], snapshot_tuning_baselines([])) == frozenset()


class TestRouterIdentity:
    def test_db_rows_key_on_model_id_and_yaml_rows_on_name_and_tags(self) -> None:
        db_row = _router("renamed", {"tiers": _TIERS}, db_id="row-1")
        assert router_identity(db_row) == router_identity(_router("other-name", {"tiers": _TIERS}, db_id="row-1"))
        assert router_identity(_router("a", {"tiers": _TIERS}, tags=["x", "y"])) == router_identity(
            _router("a", {"tiers": _TIERS}, tags=["y", "x"])
        )
        assert router_identity(_router("a", {"tiers": _TIERS}, tags=["x"])) != router_identity(
            _router("a", {"tiers": _TIERS})
        )
        assert router_identity({"litellm_params": {"model": "auto_router/complexity_router"}}) is None


class TestHeuristicV1Scope:
    @pytest.mark.parametrize(
        "config,in_scope",
        [
            ({"tiers": _TIERS}, True),
            ({"classifier_type": "heuristic", "tiers": _TIERS}, True),
            (
                {
                    "classifier_type": "heuristic_first",
                    "heuristic_first_max_tier": "MEDIUM",
                    "classifier_llm_config": {"model": "judge"},
                    "tiers": _TIERS,
                },
                True,
            ),
            (
                {
                    "classifier_type": "hybrid",
                    "hybrid_boundary_margin": 0.05,
                    "classifier_llm_config": {"model": "judge"},
                    "tiers": _TIERS,
                },
                True,
            ),
            ({"classifier_type": "heuristic_v2", "tiers": _TIERS}, False),
            ({"classifier_type": "llm", "classifier_llm_config": {"model": "judge"}, "tiers": _TIERS}, False),
        ],
    )
    def test_only_v1_scoring_classifiers_are_fingerprinted(self, config: Mapping[str, object], in_scope: bool) -> None:
        assert (heuristic_v1_router_fingerprint(_router("r", config)) is not None) is in_scope

    def test_plain_deployments_are_ignored(self) -> None:
        assert heuristic_v1_router_fingerprint(_router("gpt", None, model="openai/gpt-4o")) is None


class TestQuota:
    def test_snapshot_records_every_v1_router_even_at_defaults(self) -> None:
        baselines = snapshot_tuning_baselines([_router("a", {"tiers": _TIERS}), _router("b", {})])
        assert set(baselines) == {router_identity(_router("a", {})), router_identity(_router("b", {}))}
        assert baselines[router_identity(_router("b", {}))] == DEFAULT_TUNING_FINGERPRINT

    def test_unchanged_snapshot_is_never_mutable(self) -> None:
        rows = [_router("a", {"tiers": _TIERS}), _router("b", {"tiers": _ALT_TIERS})]
        baselines = snapshot_tuning_baselines(rows)
        assert mutable_tuned_identities(rows, baselines) == frozenset()

    def test_router_added_after_snapshot_is_mutable_only_when_tuned(self) -> None:
        baselines = snapshot_tuning_baselines([_router("a", {"tiers": _TIERS})])
        assert mutable_tuned_identities([_router("new", {})], baselines) == frozenset()
        assert mutable_tuned_identities([_router("new", {"tiers": _TIERS})], baselines) == {
            router_identity(_router("new", {}))
        }

    def test_quota_matrix(self) -> None:
        legacy_a = _router("a", {"tiers": _TIERS})
        legacy_b = _router("b", {"tiers": _ALT_TIERS})
        baselines = snapshot_tuning_baselines([legacy_a, legacy_b])
        edited_a = _router("a", {"tiers": _TIERS, "dimension_weights": {"codePresence": 0.9}})
        edited_b = _router("b", {"tiers": _TIERS})
        new_c = _router("c", {"tiers": _TIERS})

        assert tuning_quota_violation(candidate=edited_a, others=[legacy_b], baselines=baselines, limit=1) is None
        assert (
            tuning_quota_violation(candidate=edited_a, others=[edited_a, legacy_b], baselines=baselines, limit=1)
            is None
        )
        assert tuning_quota_violation(candidate=legacy_a, others=[edited_b], baselines=baselines, limit=1) is None
        assert tuning_quota_violation(candidate=edited_b, others=[edited_a], baselines=baselines, limit=1) is not None
        assert tuning_quota_violation(candidate=new_c, others=[edited_a], baselines=baselines, limit=1) is not None
        assert tuning_quota_violation(candidate=new_c, others=[edited_a], baselines=baselines, limit=None) is None
        assert (
            tuning_quota_violation(candidate=legacy_a, others=[edited_a, edited_b], baselines=baselines, limit=1)
            is None
        )

    def test_reverting_to_baseline_frees_the_quota(self) -> None:
        legacy_a = _router("a", {"tiers": _TIERS})
        legacy_b = _router("b", {"tiers": _ALT_TIERS})
        baselines = snapshot_tuning_baselines([legacy_a, legacy_b])
        edited_b = _router("b", {"tiers": _TIERS})
        assert tuning_quota_violation(candidate=edited_b, others=[legacy_a], baselines=baselines, limit=1) is None
        assert (
            tuning_quota_violation(candidate=edited_b, others=[legacy_a, edited_b], baselines=baselines, limit=1)
            is None
        )

    def test_violation_message_names_the_limit_and_remedy(self) -> None:
        message = tuning_limit_violation(held=2, limit=1)
        assert message is not None
        assert "At most 1 auto-router(s)" in message
        assert "revert the other changed router to its baseline" in message
        assert tuning_limit_violation(held=1, limit=1) is None
        assert tuning_limit_violation(held=5, limit=None) is None
