from __future__ import annotations

import pytest

from litellm.proxy.common_utils.codex_version import codex_version_at_least


@pytest.mark.parametrize(
    ("client_version", "first", "at_least"),
    [
        ("0.166.0", "0.166.0", True),
        ("0.166.1", "0.166.0", True),
        ("0.167.0", "0.166.9", True),
        ("1.0.0", "0.999.0", True),
        ("0.165.9", "0.166.0", False),
        ("0.166.0-alpha.1", "0.166.0", False),
        ("0.166.0", "0.166.0-alpha.9", True),
        ("0.166.0-alpha.2", "0.166.0-alpha.1", True),
        ("0.166.0-alpha.10", "0.166.0-alpha.9", True),
        ("0.166.0-alpha.5.1", "0.166.0-alpha.5", True),
        ("0.166.0-alpha.5.1", "0.166.0-alpha.6", False),
        ("0.166.0-alpha.5.2", "0.166.0-alpha.5.1", True),
        ("0.165.0", "0.165.0-alpha.2", True),
        ("0.165.0-alpha.1.2", "0.166.0-alpha.1", False),
    ],
)
def test_codex_versions_order_as_semver(client_version: str, first: str, at_least: bool) -> None:
    assert codex_version_at_least(client_version, first) is at_least


@pytest.mark.parametrize(
    "client_version", [None, "", "dev", "0.166", "v0.166.0", "0.166.0-beta.1", "9" * 5000 + ".0.0"]
)
def test_a_missing_or_unreadable_version_counts_as_older(client_version: str | None) -> None:
    assert codex_version_at_least(client_version, "0.0.0") is False


def test_an_unreadable_threshold_admits_no_version() -> None:
    assert codex_version_at_least("0.166.0", "latest") is False
