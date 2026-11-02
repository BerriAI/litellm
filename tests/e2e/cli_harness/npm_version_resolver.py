"""The npm release of a coding-agent CLI that CI installs: the newest one published at least three days ago.

The three days are a security review buffer (PRD #26476, "Version resolvers"), so a release that gets pulled or
patched within days never drives a run. Newest means by publish time, not semver order.

`python tests/e2e/cli_harness/npm_version_resolver.py <npm package>` prints the version on stdout for a shell
`$(...)`. It is stdlib only because the cron image runs it without the project's dependencies.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import BinaryIO, Final, cast

MIN_AGE: Final = timedelta(days=3)
_REGISTRY_URL: Final = "https://registry.npmjs.org/{package}"
_FETCH_TIMEOUT_SECONDS: Final = 30


@dataclass(frozen=True, slots=True)
class Packument:
    published_at: Mapping[str, datetime]
    versions: frozenset[str]

    @classmethod
    def from_json(cls, raw: object) -> Packument:
        document: Final = _as_mapping(raw)
        times: Final = _as_mapping(document.get("time"))
        return cls(
            published_at=MappingProxyType({version: _parse_utc(stamp) for version, stamp in _string_values(times)}),
            versions=frozenset(_as_mapping(document.get("versions"))),
        )


def newest_version_at_least(packument: Packument, *, min_age: timedelta, as_of: datetime) -> str | None:
    cutoff: Final = as_of - min_age
    eligible: Final = (
        (published, version)
        for version, published in packument.published_at.items()
        if version in packument.versions and "-" not in version and published <= cutoff
    )
    newest: Final = max(eligible, default=None)
    return None if newest is None else newest[1]


def fetch_packument(package: str) -> Packument:
    url: Final = _REGISTRY_URL.format(package=urllib.parse.quote(package, safe="@"))
    request: Final = urllib.request.Request(url, headers={"Accept": "application/json"})
    with cast(BinaryIO, urllib.request.urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS)) as response:
        return Packument.from_json(cast(object, json.loads(response.read())))


def _as_mapping(raw: object) -> Mapping[str, object]:
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in cast(dict[object, object], raw).items()}


def _string_values(raw: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple((key, value) for key, value in raw.items() if isinstance(value, str))


def _parse_utc(stamp: str) -> datetime:
    parsed: Final = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: npm_version_resolver.py <npm package>", file=sys.stderr)  # noqa: T201  # CLI surface
        return 2
    package: Final = argv[0]
    version: Final = newest_version_at_least(
        fetch_packument(package), min_age=MIN_AGE, as_of=datetime.now(timezone.utc)
    )
    if version is None:
        print(f"npm_version_resolver: no release of {package} is at least {MIN_AGE} old", file=sys.stderr)  # noqa: T201  # CLI surface
        return 1
    print(f"npm_version_resolver: selected {package}@{version}", file=sys.stderr)  # noqa: T201  # CLI surface
    print(version)  # noqa: T201  # CLI surface
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
