"""Claude Code PR-Gate Version Resolver.

Resolves the `@anthropic-ai/claude-code` npm version that the PR-gate CI
job installs. Selects the newest version (by publish timestamp) whose
publish timestamp is at least 3 days old. The 3-day window is a security
review buffer — see PRD #26476, "Version resolvers".

Two surfaces:

- ``resolve_pr_gate_version(...)`` — the importable function. Accepts
  pre-fetched npm metadata (for unit tests) or a custom ``fetcher``
  callable. The default fetcher hits the public npm registry.
- ``python -m claude_code.pr_gate_version_resolver`` — prints the
  resolved version string to stdout, suitable for piping into a shell
  ``$(...)`` substitution inside the CircleCI job.

The CLI form is what CircleCI runs at job start; engineers reading the
job log can see the selected version on a single line above the
``npm install -g`` step (acceptance criterion: "the selected Claude
Code version is logged in the CI output").
"""

from __future__ import annotations

import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from http.client import HTTPResponse
from typing import Callable, Mapping, Optional

from claude_code.json_types import JSON_OBJECT_ADAPTER, JSONValue

PACKAGE_NAME = "@anthropic-ai/claude-code"
NPM_REGISTRY_URL = "https://registry.npmjs.org/{package}"
DEFAULT_MIN_AGE = timedelta(days=3)
DEFAULT_FETCH_TIMEOUT_SECONDS = 30

# npm's `time` map mixes per-version timestamps with these meta keys.
_TIME_META_KEYS = frozenset({"created", "modified"})


class NoEligibleVersionError(RuntimeError):
    """Raised when no version in the npm metadata satisfies the min-age cutoff."""


def _parse_npm_timestamp(value: str) -> datetime:
    """Parse the ISO-8601 timestamps npm emits (always UTC, may use ``Z``)."""
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _urlopen(request: urllib.request.Request) -> HTTPResponse:
    return urllib.request.urlopen(  # noqa: S310 — registry URL is constant  # pyright: ignore[reportAny]  # urlopen is typed as Any in typeshed; https URLs yield HTTPResponse
        request, timeout=DEFAULT_FETCH_TIMEOUT_SECONDS
    )


def _default_fetcher(package_name: str) -> dict[str, JSONValue]:
    """Fetch the npm packument for ``package_name`` over HTTPS.

    Uses urllib (stdlib) so this module has no extra dependencies in the
    CI environment. Returns the raw JSON dict.
    """
    # urllib.parse.quote would encode the leading '@' / '/' which the
    # npm registry expects literally; do a minimal hand-roll instead.
    url = NPM_REGISTRY_URL.format(package=package_name.replace("/", "%2F"))
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with _urlopen(req) as response:
        body = response.read().decode("utf-8")
    return JSON_OBJECT_ADAPTER.validate_json(body)


def resolve_pr_gate_version(
    *,
    metadata: Optional[Mapping[str, JSONValue]] = None,
    fetcher: Optional[Callable[[str], Mapping[str, JSONValue]]] = None,
    as_of: Optional[datetime] = None,
    min_age: timedelta = DEFAULT_MIN_AGE,
    package_name: str = PACKAGE_NAME,
) -> str:
    """Return the newest npm version of ``package_name`` published >= ``min_age`` ago.

    "Newest" means newest by **publish time**, not semver string order —
    if a patch lands on an older major after a newer release, the
    patched line is the eligible one.

    Args:
        metadata: Pre-fetched npm packument (skips the HTTP call). Useful
            for unit tests.
        fetcher: Callable taking a package name and returning the
            packument. Defaults to a stdlib HTTPS fetcher.
        as_of: The clock used to decide whether a version is "old
            enough". Defaults to ``datetime.now(timezone.utc)``.
        min_age: Minimum publish age. Defaults to 3 days.
        package_name: Defaults to ``@anthropic-ai/claude-code``.

    Raises:
        NoEligibleVersionError: when no version in the registry meets
            the age cutoff.
    """
    if metadata is None:
        fetch = fetcher or _default_fetcher
        metadata = fetch(package_name)

    times_value = metadata.get("time")
    times: dict[str, JSONValue] = times_value if isinstance(times_value, dict) else {}
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    cutoff = as_of - min_age

    eligible: list[tuple[datetime, str]] = []
    for version, raw_ts in times.items():
        if version in _TIME_META_KEYS:
            continue
        if not isinstance(raw_ts, str):
            continue
        if "-" in version:
            continue
        published = _parse_npm_timestamp(raw_ts)
        if published <= cutoff:
            eligible.append((published, version))

    if not eligible:
        raise NoEligibleVersionError(
            f"no version of {package_name} is at least {min_age} old "
            f"as of {as_of.isoformat()}"
        )

    eligible.sort(key=lambda pair: pair[0], reverse=True)
    return eligible[0][1]


def _main(argv: list[str]) -> int:
    """Print the resolved version to stdout. Exit code 0 on success.

    Stderr carries the human-readable announcement so the version can be
    captured cleanly with ``$(python -m ...)`` in shell.
    """
    try:
        version = resolve_pr_gate_version()
    except Exception as exc:  # noqa: BLE001 — CLI surface, want everything
        print(f"pr_gate_version_resolver: {exc}", file=sys.stderr)  # noqa: T201
        return 1
    print(  # noqa: T201
        f"pr_gate_version_resolver: selected {PACKAGE_NAME}@{version}",
        file=sys.stderr,
    )
    print(version)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
