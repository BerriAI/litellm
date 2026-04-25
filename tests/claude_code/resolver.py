"""Latest Stable LiteLLM Resolver.

Queries the GitHub Releases API for `BerriAI/litellm` and returns the
newest tag matching `v*-stable`. Used by the daily-cron publisher to
decide which Docker image to pull for the matrix run.

The resolver is intentionally tiny — its only state is the GitHub API
URL constant — and exposes a single public function so unit tests can
inject a fake HTTP getter and run offline.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Callable, List, Optional

GITHUB_RELEASES_URL = "https://api.github.com/repos/BerriAI/litellm/releases"
USER_AGENT = "litellm-compat-matrix-resolver"
REQUEST_TIMEOUT_SECONDS = 30

STABLE_TAG_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)-stable$")


class ResolverError(RuntimeError):
    """Raised when the resolver cannot determine a latest-stable tag."""


def _default_http_get(url: str, *, token: Optional[str] = None) -> str:
    """Minimal urllib-based GET that the cron VM can call without extra deps.

    Forwards `token` as a Bearer header when set so the cron job can lift
    the unauthenticated GitHub rate limit by passing the GitHub App
    installation token.
    """
    request = urllib.request.Request(url)
    request.add_header("User-Agent", USER_AGENT)
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(  # noqa: S310 - URL is hardcoded above
        request, timeout=REQUEST_TIMEOUT_SECONDS
    ) as response:
        return response.read().decode("utf-8")


def latest_stable_litellm_tag(
    *,
    http_get: Optional[Callable[..., str]] = None,
    token: Optional[str] = None,
) -> str:
    """Return the newest `v*-stable` tag published on BerriAI/litellm.

    Sort is numeric on the (major, minor, patch) triple so v1.10.0-stable
    correctly outranks v1.9.5-stable. Releases with no `tag_name` (drafts)
    or that don't match the `v*-stable` shape are skipped.
    """
    fetch = http_get or _default_http_get
    payload = fetch(GITHUB_RELEASES_URL, token=token)
    releases = json.loads(payload)
    if not isinstance(releases, list):
        raise ResolverError(
            "github releases response is not a list; got " f"{type(releases).__name__}"
        )

    matched: List[tuple] = []
    for release in releases:
        if not isinstance(release, dict):
            continue
        tag = release.get("tag_name")
        if not isinstance(tag, str):
            continue
        m = STABLE_TAG_RE.match(tag)
        if not m:
            continue
        version_key = tuple(int(x) for x in m.groups())
        matched.append((version_key, tag))

    if not matched:
        raise ResolverError("no v*-stable tags found in github releases response")

    matched.sort(key=lambda pair: pair[0], reverse=True)
    return matched[0][1]


__all__ = [
    "GITHUB_RELEASES_URL",
    "ResolverError",
    "latest_stable_litellm_tag",
]
