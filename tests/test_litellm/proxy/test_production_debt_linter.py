"""Tests + labelled benchmark corpus for litellm.proxy.production_debt_linter.

Each corpus config is labelled with exactly the rule names it should (for
adversarial configs) or should not (for clean configs) trigger. The
parametrized tests below turn that corpus into a real, reproducible
recall/false-positive measurement rather than an asserted claim:

    - test_adversarial_config_detected: every injected defect must be
      caught (recall).
    - test_clean_config_has_no_findings: no clean, realistic config may
      produce a finding (false positives).
"""

from __future__ import annotations

import datetime

import pytest

from litellm.proxy.production_debt_linter import (
    Finding,
    Severity,
    analyze,
    detect_deprecated_model_references,
    detect_missing_budget_cap,
    detect_missing_fallback_coverage,
    detect_retry_without_cooldown,
    production_debt_score,
)

# A small, fixed model_cost stand-in so the deprecated-model tests don't
# drift with the real dataset's dates. The real litellm.model_cost
# integration is exercised separately below.
FAKE_MODEL_COST = {
    "openai/gpt-4o": {"deprecation_date": "2099-01-01"},
    "openai/old-model": {"deprecation_date": "2020-01-01"},
}
FIXED_TODAY = datetime.date(2026, 1, 1)


def _deployment(model_name: str, model: str, **extra_params) -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, **extra_params},
    }


# ---------------------------------------------------------------------------
# Clean corpus: realistic configs that must produce zero findings.
# ---------------------------------------------------------------------------


def clean_single_model_with_fallback_and_budget() -> dict:
    return {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o", max_budget=100.0),
            _deployment("gpt-4o-mini", "openai/gpt-4o-mini", max_budget=50.0),
        ],
        "litellm_settings": {
            "fallbacks": [{"gpt-4o": ["gpt-4o-mini"]}],
        },
    }


def clean_global_budget_and_default_fallbacks() -> dict:
    return {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o"),
            _deployment("claude", "anthropic/claude-3-5-sonnet"),
        ],
        "litellm_settings": {
            "max_budget": 500.0,
            "default_fallbacks": ["gpt-4o"],
        },
    }


def clean_bounded_retries_with_allowed_fails() -> dict:
    return {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o", max_budget=100.0, num_retries=5),
        ],
        "router_settings": {
            "allowed_fails": 3,
            "fallbacks": [{"gpt-4o": ["gpt-4o"]}],
        },
    }


def clean_low_retry_count_needs_no_cooldown() -> dict:
    return {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o", max_budget=100.0, num_retries=1),
        ],
        "litellm_settings": {
            "default_fallbacks": ["gpt-4o"],
        },
    }


def clean_context_window_fallback_counts_as_coverage() -> dict:
    return {
        "model_list": [
            _deployment("gpt-3.5-turbo", "openai/gpt-3.5-turbo", max_budget=10.0),
            _deployment("gpt-3.5-turbo-large", "openai/gpt-4.1", max_budget=10.0),
        ],
        "litellm_settings": {
            "context_window_fallbacks": [
                {"gpt-3.5-turbo": ["gpt-3.5-turbo-large"]}
            ],
        },
    }


CLEAN_CORPUS = {
    "single_model_with_fallback_and_budget": clean_single_model_with_fallback_and_budget,
    "global_budget_and_default_fallbacks": clean_global_budget_and_default_fallbacks,
    "bounded_retries_with_allowed_fails": clean_bounded_retries_with_allowed_fails,
    "low_retry_count_needs_no_cooldown": clean_low_retry_count_needs_no_cooldown,
    "context_window_fallback_counts_as_coverage": clean_context_window_fallback_counts_as_coverage,
}


# ---------------------------------------------------------------------------
# Adversarial corpus: one deliberately injected defect each, labelled with
# the exact rule name that must fire.
# ---------------------------------------------------------------------------


def bad_no_fallback_coverage() -> dict:
    return {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o", max_budget=100.0),
        ],
        "litellm_settings": {},
    }


def bad_high_retries_no_cooldown_global() -> dict:
    return {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o", max_budget=100.0),
        ],
        "litellm_settings": {
            "num_retries": 5,
            "default_fallbacks": ["gpt-4o"],
        },
    }


def bad_high_retries_no_cooldown_deployment() -> dict:
    return {
        "model_list": [
            _deployment(
                "gpt-4o", "openai/gpt-4o", max_budget=100.0, num_retries=10
            ),
        ],
        "litellm_settings": {
            "default_fallbacks": ["gpt-4o"],
        },
    }


def bad_missing_budget_cap() -> dict:
    return {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o"),
        ],
        "litellm_settings": {
            "default_fallbacks": ["gpt-4o"],
        },
    }


def bad_deprecated_model_reference() -> dict:
    return {
        "model_list": [
            _deployment(
                "legacy", "openai/old-model", max_budget=10.0
            ),
        ],
        "litellm_settings": {
            "default_fallbacks": ["legacy"],
        },
    }


ADVERSARIAL_CORPUS = {
    "no_fallback_coverage": (bad_no_fallback_coverage, "no-fallback-coverage"),
    "high_retries_no_cooldown_global": (
        bad_high_retries_no_cooldown_global,
        "retry-without-cooldown",
    ),
    "high_retries_no_cooldown_deployment": (
        bad_high_retries_no_cooldown_deployment,
        "retry-without-cooldown",
    ),
    "missing_budget_cap": (bad_missing_budget_cap, "missing-deployment-budget-cap"),
    "deprecated_model_reference": (
        bad_deprecated_model_reference,
        "deprecated-model-reference",
    ),
}


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CLEAN_CORPUS)
def test_clean_config_has_no_findings(name: str) -> None:
    config = CLEAN_CORPUS[name]()
    findings = analyze(config, model_cost=FAKE_MODEL_COST)
    assert findings == (), f"{name} produced unexpected findings: {findings}"


@pytest.mark.parametrize("name", ADVERSARIAL_CORPUS)
def test_adversarial_config_detected(name: str) -> None:
    factory, expected_rule = ADVERSARIAL_CORPUS[name]
    config = factory()
    findings = analyze(config, model_cost=FAKE_MODEL_COST)
    rules = {f.rule for f in findings}
    assert expected_rule in rules, f"{name}: expected '{expected_rule}' not in {rules}"


def test_benchmark_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """Not an assertion -- prints the real recall/false-positive numbers.

    Run with `-s` to see the summary; the two tests above are what
    actually enforce recall == 100% and false positives == 0 per-case.
    """
    total_clean = len(CLEAN_CORPUS)
    fp_count = sum(
        1
        for factory in CLEAN_CORPUS.values()
        if analyze(factory(), model_cost=FAKE_MODEL_COST)
    )

    total_adversarial = len(ADVERSARIAL_CORPUS)
    detected = 0
    for factory, expected_rule in ADVERSARIAL_CORPUS.values():
        rules = {f.rule for f in analyze(factory(), model_cost=FAKE_MODEL_COST)}
        if expected_rule in rules:
            detected += 1

    print(
        f"\nbenchmark: recall={detected}/{total_adversarial} "
        f"false_positives={fp_count}/{total_clean}"
    )


# ---------------------------------------------------------------------------
# Unit-level tests for individual detectors and the score function
# ---------------------------------------------------------------------------


def test_production_debt_score_weights() -> None:
    findings = [
        Finding("r1", Severity.CRITICAL, None, "x"),
        Finding("r2", Severity.WARNING, None, "y"),
    ]
    assert production_debt_score(findings) == 10 + 3
    assert production_debt_score([]) == 0


def test_no_fallback_coverage_ignores_wildcard_models() -> None:
    config = {
        "model_list": [_deployment("*", "openai/*", max_budget=100.0)],
        "litellm_settings": {"max_budget": 100.0},
    }
    assert detect_missing_fallback_coverage(config) == ()


def test_retry_without_cooldown_respects_allowed_fails_policy() -> None:
    config = {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o", num_retries=10),
        ],
        "litellm_settings": {
            "allowed_fails_policy": {"AuthenticationErrorRetries": 0},
        },
    }
    assert detect_retry_without_cooldown(config) == ()


def test_missing_budget_cap_respects_global_budget() -> None:
    config = {
        "model_list": [_deployment("gpt-4o", "openai/gpt-4o")],
        "router_settings": {"max_budget": 1000.0},
    }
    assert detect_missing_budget_cap(config) == ()


def test_deprecated_model_reference_only_flags_past_dates() -> None:
    config = {
        "model_list": [
            _deployment("current", "openai/gpt-4o"),
            _deployment("legacy", "openai/old-model"),
        ],
    }
    findings = detect_deprecated_model_references(
        config, FAKE_MODEL_COST, today=FIXED_TODAY
    )
    assert len(findings) == 1
    assert findings[0].model_name == "legacy"


def test_analyze_sorts_critical_first() -> None:
    config = {
        "model_list": [
            _deployment("gpt-4o", "openai/gpt-4o", num_retries=10),
        ],
        "litellm_settings": {"default_fallbacks": ["gpt-4o"]},
    }
    findings = analyze(config)
    assert findings[0].severity == Severity.CRITICAL


def test_analyze_works_without_model_cost() -> None:
    # model_cost is optional; the other three detectors don't need it.
    config = {"model_list": [_deployment("gpt-4o", "openai/gpt-4o")]}
    findings = analyze(config)
    assert all(f.rule != "deprecated-model-reference" for f in findings)


def test_real_litellm_model_cost_integration() -> None:
    """Exercises the real litellm.model_cost data, not the fake stand-in,
    to prove the integration point (not just the detector logic) works."""
    import litellm

    deprecated_model = next(
        (
            model
            for model, info in litellm.model_cost.items()
            if isinstance(info, dict)
            and isinstance(info.get("deprecation_date"), str)
            and info["deprecation_date"] not in ("", "date when the model becomes deprecated in the format YYYY-MM-DD")
        ),
        None,
    )
    assert deprecated_model is not None, "expected at least one model with a real deprecation_date in litellm.model_cost"

    config = {
        "model_list": [_deployment("legacy", deprecated_model, max_budget=1.0)],
        "litellm_settings": {"default_fallbacks": ["legacy"]},
    }
    findings = detect_deprecated_model_references(
        config, litellm.model_cost, today=datetime.date(2099, 1, 1)
    )
    assert len(findings) == 1
    assert findings[0].model_name == "legacy"
