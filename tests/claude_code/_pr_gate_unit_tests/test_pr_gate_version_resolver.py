"""Unit tests for the Claude Code PR-Gate Version Resolver.

The resolver picks the newest `@anthropic-ai/claude-code` version whose
publish timestamp is at least 3 days old. The 3-day window is a security
review buffer: a malicious or broken Claude Code release that slipped
through the npm publish process gets at least 72 hours to be detected
before it can land in the LiteLLM PR gate.

The unit tests inject npm metadata directly (no network) and a fixed
`as_of` clock (no real time), so they run anywhere and never flake on
the wall clock or registry availability.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.claude_code.pr_gate_version_resolver import (
    NoEligibleVersionError,
    resolve_pr_gate_version,
)


def _t(iso: str) -> str:
    """Helper for readable ISO-8601 publish timestamps in fixtures."""
    return iso


# A clock fixed at a moment well after every fixture publish time below.
NOW = datetime(2026, 4, 25, 12, 0, 0, tzinfo=timezone.utc)


def _metadata_with_times(times: dict) -> dict:
    """Shape an npm `packument`-like dict with the `time` field populated.

    The npm registry response includes `time.created` / `time.modified`
    keys alongside per-version timestamps; the resolver must skip those.
    """
    return {
        "name": "@anthropic-ai/claude-code",
        "time": {
            "created": _t("2024-01-01T00:00:00.000Z"),
            "modified": _t("2026-04-25T00:00:00.000Z"),
            **times,
        },
    }


def test_picks_newest_version_at_least_three_days_old():
    metadata = _metadata_with_times(
        {
            "2.1.118": _t("2026-04-15T10:00:00.000Z"),
            "2.1.119": _t("2026-04-21T10:00:00.000Z"),  # 4d 2h old
            "2.1.120": _t("2026-04-23T10:00:00.000Z"),  # 2d 2h old — too new
            "2.1.121": _t("2026-04-25T11:00:00.000Z"),  # 1h old — too new
        }
    )
    assert resolve_pr_gate_version(metadata=metadata, as_of=NOW) == "2.1.119"


def test_skips_created_and_modified_meta_keys():
    """`time` contains `created` / `modified` non-version entries — must be ignored."""
    metadata = {
        "name": "@anthropic-ai/claude-code",
        "time": {
            "created": _t("2024-01-01T00:00:00.000Z"),
            "modified": _t("2026-04-25T00:00:00.000Z"),
            "2.0.0": _t("2026-04-10T00:00:00.000Z"),
        },
    }
    assert resolve_pr_gate_version(metadata=metadata, as_of=NOW) == "2.0.0"


def test_min_age_boundary_is_inclusive():
    """A version published exactly 3 days ago is eligible (>= cutoff)."""
    three_days_ago = NOW - timedelta(days=3)
    metadata = _metadata_with_times(
        {
            "2.1.0": three_days_ago.isoformat().replace("+00:00", "Z"),
        }
    )
    assert resolve_pr_gate_version(metadata=metadata, as_of=NOW) == "2.1.0"


def test_raises_when_every_version_is_too_new():
    metadata = _metadata_with_times(
        {
            "2.1.121": _t("2026-04-25T08:00:00.000Z"),  # 4h old
            "2.1.120": _t("2026-04-24T10:00:00.000Z"),  # ~26h old
        }
    )
    with pytest.raises(NoEligibleVersionError):
        resolve_pr_gate_version(metadata=metadata, as_of=NOW)


def test_raises_when_metadata_has_no_versions():
    metadata = {"name": "@anthropic-ai/claude-code", "time": {}}
    with pytest.raises(NoEligibleVersionError):
        resolve_pr_gate_version(metadata=metadata, as_of=NOW)


def test_picks_latest_publish_time_not_largest_semver():
    """If a patch is published to an old major after a newer release,
    "newest" is by publish time, not semver string ordering."""
    metadata = _metadata_with_times(
        {
            "1.9.99": _t("2026-04-22T10:00:00.000Z"),  # patched recently — wins
            "2.0.0": _t("2026-03-01T10:00:00.000Z"),  # older publish
        }
    )
    assert resolve_pr_gate_version(metadata=metadata, as_of=NOW) == "1.9.99"


def test_uses_custom_min_age():
    metadata = _metadata_with_times(
        {
            "1.0.0": _t("2026-04-23T10:00:00.000Z"),  # 2d 2h old
            "0.9.0": _t("2026-04-10T10:00:00.000Z"),  # 15d old
        }
    )
    # min_age = 5 days disqualifies 1.0.0
    out = resolve_pr_gate_version(
        metadata=metadata, as_of=NOW, min_age=timedelta(days=5)
    )
    assert out == "0.9.0"


def test_resolver_uses_fetcher_when_metadata_not_provided():
    captured = {}

    def fake_fetch(package_name: str) -> dict:
        captured["package"] = package_name
        return _metadata_with_times({"3.0.0": _t("2026-04-10T10:00:00.000Z")})

    out = resolve_pr_gate_version(as_of=NOW, fetcher=fake_fetch)
    assert out == "3.0.0"
    assert captured["package"] == "@anthropic-ai/claude-code"
