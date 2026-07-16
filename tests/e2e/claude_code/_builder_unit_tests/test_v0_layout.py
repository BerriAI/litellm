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

SUITE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = SUITE_ROOT / "manifest.yaml"

# The PRD's "Features in v0" section, in row order.
EXPECTED_FEATURE_IDS = [
    "basic_messaging_non_streaming",
    "basic_messaging_streaming",
    "tool_use",
    "prompt_caching_5m",
    "vision",
    # v0 originally shipped this row as `extended_thinking`. It was
    # renamed in-place to `thinking` because Anthropic's docs reserve
    # "extended thinking" for the deprecated manual API mode only; the
    # single row exercises both manual and adaptive shapes since Claude
    # Code picks per model. The PRD's "v0" identity is the *position*
    # (row 6, 0-indexed 5), not the id string.
    "thinking",
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


def _all_manifest_feature_ids() -> list[str]:
    """Every feature_id currently declared in `manifest.yaml`.

    Evaluated at import time so the result can drive parametrized
    structural tests below. Used to catch layout drift on post-v0
    feature rows added after the matrix shipped — the v0 anchor
    constants above only validate the original six rows by design.
    """
    return [
        feature["id"]
        for feature in yaml.safe_load(MANIFEST_PATH.read_text())["features"]
    ]


ALL_FEATURE_IDS = _all_manifest_feature_ids()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def test_manifest_lists_all_six_v0_features_in_order(manifest):
    """The PRD's v0 row set must appear at the top of the manifest in
    order. Features beyond v0 (extensions added after the matrix
    shipped) are allowed but must not reorder or displace the v0
    rows — the docs page anchors row links by index, so v0 stays
    pinned at positions [0:6] for the lifetime of the schema.
    """
    ids = [feature["id"] for feature in manifest["features"]]
    assert ids[: len(EXPECTED_FEATURE_IDS)] == EXPECTED_FEATURE_IDS


def test_manifest_lists_all_five_v0_providers_in_order(manifest):
    assert manifest["providers"] == EXPECTED_PROVIDERS


def test_manifest_every_feature_has_human_readable_name(manifest):
    for feature in manifest["features"]:
        assert isinstance(feature["name"], str) and feature["name"].strip()


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
def test_feature_directory_exists(feature_id):
    feature_dir = SUITE_ROOT / feature_id
    assert feature_dir.is_dir(), f"missing feature directory: {feature_dir}"


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
@pytest.mark.parametrize("provider", EXPECTED_PROVIDERS)
def test_per_provider_test_file_exists(feature_id, provider):
    test_file = SUITE_ROOT / feature_id / f"test_{provider}.py"
    assert test_file.is_file(), f"missing per-provider test file: {test_file}"


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
def test_feature_directory_has_init_file(feature_id):
    """Each feature directory needs an __init__.py so pytest collects
    the per-provider test files as a package — matches the layout
    established by `basic_messaging_non_streaming/`."""
    init_file = SUITE_ROOT / feature_id / "__init__.py"
    assert init_file.is_file(), f"missing __init__.py: {init_file}"


# Manifest-driven structural tests: every feature in `manifest.yaml`
# (v0 and post-v0 alike) must have the expected on-disk layout. The
# v0-only tests above pin the position of the original six rows; these
# extend the same structural guarantees to any row added afterward so
# a broken post-v0 directory still fails CI.
@pytest.mark.parametrize("feature_id", ALL_FEATURE_IDS)
def test_every_manifest_feature_has_directory(feature_id):
    feature_dir = SUITE_ROOT / feature_id
    assert feature_dir.is_dir(), (
        f"manifest declares {feature_id!r} but {feature_dir} is missing — "
        "feature_id MUST match its on-disk directory (see manifest.yaml header)."
    )


@pytest.mark.parametrize("feature_id", ALL_FEATURE_IDS)
def test_every_manifest_feature_has_init_file(feature_id):
    init_file = SUITE_ROOT / feature_id / "__init__.py"
    assert init_file.is_file(), f"missing __init__.py: {init_file}"


@pytest.mark.parametrize("feature_id", ALL_FEATURE_IDS)
@pytest.mark.parametrize("provider", EXPECTED_PROVIDERS)
def test_every_manifest_feature_has_per_provider_test_file(feature_id, provider):
    """Every (feature, provider) cell in the rendered matrix must be
    backed by a per-provider test file. Without this check, a missing
    file silently becomes a `not_tested` cell in the published matrix
    rather than a CI failure surfacing the layout drift."""
    test_file = SUITE_ROOT / feature_id / f"test_{provider}.py"
    assert test_file.is_file(), f"missing per-provider test file: {test_file}"


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
    text = (SUITE_ROOT / feature_id / f"test_{provider}.py").read_text()
    for tier in ("haiku-4-5", "sonnet-4-5", "opus-4-7"):
        assert (
            tier in text
        ), f"{feature_id}/test_{provider}.py does not reference {tier}"


@pytest.mark.parametrize("feature_id", EXPECTED_FEATURE_IDS)
def test_azure_test_file_drives_the_proxy(feature_id):
    """Azure (Microsoft Foundry) hosts Anthropic Claude as of 2025-11-18,
    so every Azure cell in the v0 matrix exercises a real route through
    the LiteLLM proxy — same shape as the other provider columns. Pin
    that here so a future regression doesn't silently revert these
    cells to the old `not_applicable` boilerplate.

    We accept either the direct `run_claude(...)` family of entrypoints
    or a per-feature shared helper (e.g. `run_basic_messaging_cell`)
    that wraps them — both shapes drive the proxy, and we don't want
    this layout pin to block legitimate de-duplication of test bodies.
    """
    text = (SUITE_ROOT / feature_id / "test_azure.py").read_text()
    assert "run_claude" in text or "run_basic_messaging_cell" in text, (
        f"{feature_id}/test_azure.py must drive the claude CLI via run_claude() "
        "or a shared helper that wraps it; the not_applicable stub was removed "
        "when Foundry started hosting Claude."
    )
    assert '"status": "not_applicable"' not in text, (
        f"{feature_id}/test_azure.py still reports not_applicable; Microsoft Foundry "
        "now hosts Claude (Haiku 4.5, Sonnet 4.6, Opus 4.7), so this row must run."
    )
