"""Unit tests for the Latest Stable LiteLLM Resolver.

The resolver is the smallest piece of cron-pipeline glue: it asks the GitHub
Releases API for `BerriAI/litellm` and returns the newest tag matching the
`v*-stable` pattern. Tests inject a fake HTTP getter so they run offline.

Per the PRD's "Testing Decisions" section, version resolvers were officially
deferred from v0 — but the resolver is also small enough that a few cheap
unit tests are worth more than the daily-cron failing loudly.
"""

from __future__ import annotations

import json
from typing import List

import pytest

from tests.claude_code.resolver import (
    GITHUB_RELEASES_URL,
    ResolverError,
    latest_stable_litellm_tag,
)


def _fake_getter(payload):
    """Return a callable that records calls and returns the given JSON payload.

    Mirrors the `runner=` injection seam used by `cli_driver` tests so the
    unit tests don't depend on `urllib`.
    """

    captured: List[dict] = []

    def getter(url, *, token=None):
        captured.append({"url": url, "token": token})
        return json.dumps(payload)

    getter.captured = captured  # type: ignore[attr-defined]
    return getter


def test_resolver_returns_newest_stable_tag():
    """Given a list of releases, the newest `v*-stable` wins."""
    getter = _fake_getter(
        [
            {"tag_name": "v1.81.0-stable"},
            {"tag_name": "v1.83.0-stable"},
            {"tag_name": "v1.82.5-stable"},
        ]
    )
    assert latest_stable_litellm_tag(http_get=getter) == "v1.83.0-stable"


def test_resolver_ignores_non_stable_tags():
    """RC, alpha, beta, plain `v1.83.0`, and noise tags are filtered out."""
    getter = _fake_getter(
        [
            {"tag_name": "v1.84.0-rc1"},
            {"tag_name": "v1.84.0-stable.draft"},
            {"tag_name": "v1.84.0"},
            {"tag_name": "v1.83.0-stable"},
            {"tag_name": "stable-2026-04-25"},
            {"tag_name": "v1.85.0-alpha"},
        ]
    )
    assert latest_stable_litellm_tag(http_get=getter) == "v1.83.0-stable"


def test_resolver_uses_numeric_version_sort_not_lexicographic():
    """v1.10.0-stable must be newer than v1.9.0-stable (numeric, not string)."""
    getter = _fake_getter(
        [
            {"tag_name": "v1.9.0-stable"},
            {"tag_name": "v1.10.0-stable"},
            {"tag_name": "v1.9.5-stable"},
        ]
    )
    assert latest_stable_litellm_tag(http_get=getter) == "v1.10.0-stable"


def test_resolver_raises_when_no_stable_tags():
    getter = _fake_getter([{"tag_name": "v1.84.0-rc1"}, {"tag_name": "v1.83.0"}])
    with pytest.raises(ResolverError, match="no v\\*-stable tags found"):
        latest_stable_litellm_tag(http_get=getter)


def test_resolver_raises_when_response_is_not_a_list():
    getter = _fake_getter({"message": "API rate limit exceeded"})
    with pytest.raises(ResolverError, match="not a list"):
        latest_stable_litellm_tag(http_get=getter)


def test_resolver_calls_github_releases_endpoint_with_optional_token():
    """The resolver must call the BerriAI/litellm releases API and forward
    the auth token (if provided) to the http getter so callers can lift
    the unauthenticated rate limit when running on the cron VM."""
    getter = _fake_getter([{"tag_name": "v1.83.0-stable"}])
    latest_stable_litellm_tag(http_get=getter, token="ghs_xxx")
    assert getter.captured == [{"url": GITHUB_RELEASES_URL, "token": "ghs_xxx"}]


def test_resolver_skips_releases_with_missing_tag_name():
    """Defensive: GitHub draft releases can omit `tag_name`; don't crash."""
    getter = _fake_getter(
        [
            {"name": "untagged draft"},
            {"tag_name": None},
            {"tag_name": "v1.83.0-stable"},
        ]
    )
    assert latest_stable_litellm_tag(http_get=getter) == "v1.83.0-stable"
