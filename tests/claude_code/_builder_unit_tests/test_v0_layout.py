"""Structural tests for the full v0 6x5 matrix layout.

These tests don't run the `claude` CLI — they only verify that the
shape of the test suite on disk matches what the PRD declares: six
features in the prescribed order, and for each feature a directory
with one test file per provider column.

Catching layout drift here means the daily-cron VM and the PR gate
both see the same row set the docs page declares.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "manifest.yaml"

# The PRD's "Features in v0" section, in row order.
EXPECTED_FEATURE_IDS = [
    "basic_messaging_non_streaming",
    "basic_messaging_streaming",
    "tool_use",
    "prompt_caching_5m",
    "vision",
    "extended_thinking",
]

# The PRD's column order. Every feature directory must have one
# `test_<provider>.py` for each of these.
EXPECTED_PROVIDERS = [
    "anthropic",
    "bedrock_invoke",
    "bedrock_converse",
    "vertex_ai",
    "azure",
]


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def test_manifest_lists_all_six_v0_features_in_order(manifest):
    ids = [feature["id"] for feature in manifest["features"]]
    assert ids == EXPECTED_FEATURE_IDS


def test_manifest_lists_all_five_v0_providers_in_order(manifest):
    assert manifest["providers"] == EXPECTED_PROVIDERS


def test_manifest_every_feature_has_human_readable_name(manifest):
    for feature in manifest["features"]:
        assert isinstance(feature["name"], str) and feature["name"].strip()


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
def test_feature_directory_exists(feature_id):
    feature_dir = REPO_ROOT / feature_id
    assert feature_dir.is_dir(), f"missing feature directory: {feature_dir}"


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
@pytest.mark.parametrize("provider", EXPECTED_PROVIDERS)
def test_per_provider_test_file_exists(feature_id, provider):
    test_file = REPO_ROOT / feature_id / f"test_{provider}.py"
    assert test_file.is_file(), f"missing per-provider test file: {test_file}"


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
def test_feature_directory_has_init_file(feature_id):
    """Each feature directory needs an __init__.py so pytest collects
    the per-provider test files as a package — matches the layout
    established by `basic_messaging_non_streaming/`."""
    init_file = REPO_ROOT / feature_id / "__init__.py"
    assert init_file.is_file(), f"missing __init__.py: {init_file}"


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
@pytest.mark.parametrize("provider", EXPECTED_PROVIDERS)
def test_per_provider_test_file_imports_and_parametrizes_three_models(
    feature_id, provider
):
    """Every test file must reference the three Claude tiers required
    by the PRD: Haiku 4.5, Sonnet 4.6, Opus 4.7. Implementations may
    use plain aliases or per-provider-suffixed aliases (e.g.
    `claude-opus-4-7-bedrock-invoke`), so we check for the tier
    substrings rather than exact alias names."""
    text = (REPO_ROOT / feature_id / f"test_{provider}.py").read_text()
    for tier in ("haiku-4-5", "sonnet-4-6", "opus-4-7"):
        assert (
            tier in text
        ), f"{feature_id}/test_{provider}.py does not reference {tier}"


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
def test_azure_test_file_reports_not_applicable(feature_id):
    """Azure OpenAI Service does not host Claude on any v0 feature, so
    every Azure cell in the v0 matrix is `not_applicable`. Pin that
    here so a future "let's just call the proxy and see what happens"
    edit doesn't silently turn the gray cells red."""
    text = (REPO_ROOT / feature_id / "test_azure.py").read_text()
    assert '"status": "not_applicable"' in text
