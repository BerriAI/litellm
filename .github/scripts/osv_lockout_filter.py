#!/usr/bin/env python3
"""Turn osv-scanner JSON results into a CI verdict that respects the dependency lockout window.

A finding is only actionable if the fix it points at is old enough for our resolvers to
install it (``[tool.uv] exclude-newer`` for PyPI, ``min-release-age`` in ``.npmrc`` for npm).
Findings whose fix is still inside that window are reported and deferred until it expires.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from packaging.version import InvalidVersion, Version

PYPI = "PyPI"
NPM = "npm"

PublishTimes = Callable[[str, str], Mapping[str, datetime] | None]

_SEMVER = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_PYPI_NAME = re.compile(r"[-_.]+")
_DAYS = re.compile(r"\s*(\d+)\s*days?\s*")
_MIN_RELEASE_AGE = re.compile(r"^\s*min-release-age\s*=\s*(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Finding:
    ecosystem: str
    package: str
    installed: str
    vuln_id: str
    severity: str
    fixed_versions: tuple[str, ...]
    source: str


@dataclass(frozen=True, slots=True)
class Deferred:
    finding: Finding
    target: str
    published: datetime
    unlocks_at: datetime


@dataclass(frozen=True, slots=True)
class Actionable:
    finding: Finding
    target: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class WindowError:
    message: str


Verdict = Deferred | Actionable


def semver_key(version: str) -> tuple[tuple[int, int, int], tuple[int, ...], tuple[tuple[int, int, str], ...]] | None:
    matched = _SEMVER.match(version.strip())
    if matched is None:
        return None
    core = (int(matched.group(1)), int(matched.group(2)), int(matched.group(3)))
    prerelease = matched.group(4)
    if prerelease is None:
        return (core, (1,), ())
    return (
        core,
        (0,),
        tuple((0, int(part), "") if part.isdigit() else (1, 0, part) for part in prerelease.split(".")),
    )


def pypi_key(version: str) -> Version | None:
    try:
        return Version(version)
    except InvalidVersion:
        return None


def version_key(ecosystem: str, version: str) -> object | None:
    if ecosystem == PYPI:
        return pypi_key(version)
    if ecosystem == NPM:
        return semver_key(version)
    return None


def _base_ecosystem(raw: str) -> str:
    return raw.split(":", 1)[0]


def _canonical_name(ecosystem: str, name: str) -> str:
    return _PYPI_NAME.sub("-", name).lower() if ecosystem == PYPI else name


def _fixed_versions(vulnerability: dict, ecosystem: str, package: str) -> tuple[str, ...]:
    wanted = _canonical_name(ecosystem, package)
    return tuple(
        dict.fromkeys(
            event["fixed"]
            for affected in vulnerability.get("affected") or []
            for affected_package in ((affected.get("package") or {}),)
            if _base_ecosystem(str(affected_package.get("ecosystem") or "")) == ecosystem
            and _canonical_name(ecosystem, str(affected_package.get("name") or "")) == wanted
            for version_range in affected.get("ranges") or []
            if version_range.get("type") in ("ECOSYSTEM", "SEMVER")
            for event in version_range.get("events") or []
            if "fixed" in event
        )
    )


def _severity(groups: Sequence[dict], vuln_id: str) -> str:
    matches = tuple(
        str(group.get("max_severity") or "") for group in groups or [] if vuln_id in (group.get("ids") or [])
    )
    return next((severity for severity in matches if severity), "-")


def _findings_for_package(scanned: dict, source: str) -> tuple[Finding, ...]:
    package = scanned["package"]
    ecosystem = _base_ecosystem(str(package["ecosystem"]))
    name = str(package["name"])
    groups = scanned.get("groups") or []
    return tuple(
        Finding(
            ecosystem=ecosystem,
            package=name,
            installed=str(package["version"]),
            vuln_id=str(vulnerability["id"]),
            severity=_severity(groups, str(vulnerability["id"])),
            fixed_versions=_fixed_versions(vulnerability, ecosystem, name),
            source=source,
        )
        for vulnerability in scanned.get("vulnerabilities") or []
    )


def _relative_source(path: str, repo_root: Path) -> str:
    try:
        return str(Path(path).relative_to(repo_root))
    except ValueError:
        return path


def parse_findings(payload: dict, repo_root: Path) -> tuple[Finding, ...]:
    return tuple(
        finding
        for result in payload.get("results") or []
        for source in (_relative_source(str((result.get("source") or {}).get("path") or "?"), repo_root),)
        for scanned in result.get("packages") or []
        for finding in _findings_for_package(scanned, source)
    )


def _fetch_json(url: str, attempts: int = 3) -> dict | None:
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "litellm-osv-lockout"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            print(f"osv-lockout: {url} lookup failed ({exc})", file=sys.stderr)
            if attempt + 1 < attempts:
                time.sleep(2 * (attempt + 1))
    return None


def _parse_timestamp(raw: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _earliest_upload(files: Sequence[dict]) -> datetime | None:
    stamps = tuple(
        stamp
        for entry in files or []
        for stamp in (_parse_timestamp(str(entry.get("upload_time_iso_8601") or "")),)
        if stamp is not None
    )
    return min(stamps) if stamps else None


def _pypi_publish_times(name: str) -> Mapping[str, datetime] | None:
    payload = _fetch_json(f"https://pypi.org/pypi/{urllib.parse.quote(name, safe='')}/json")
    if payload is None:
        return None
    return {
        version: stamp
        for version, files in (payload.get("releases") or {}).items()
        for stamp in (_earliest_upload(files),)
        if stamp is not None
    }


def _npm_publish_times(name: str) -> Mapping[str, datetime] | None:
    payload = _fetch_json(f"https://registry.npmjs.org/{urllib.parse.quote(name, safe='@')}")
    if payload is None:
        return None
    return {
        version: stamp
        for version, raw in (payload.get("time") or {}).items()
        if version not in ("created", "modified")
        for stamp in (_parse_timestamp(str(raw)),)
        if stamp is not None
    }


@functools.lru_cache(maxsize=None)
def registry_publish_times(ecosystem: str, name: str) -> Mapping[str, datetime] | None:
    if ecosystem == PYPI:
        return _pypi_publish_times(name)
    if ecosystem == NPM:
        return _npm_publish_times(name)
    return None


def _uv_window(pyproject: Path) -> timedelta | WindowError:
    if not pyproject.is_file():
        return WindowError(f"{pyproject} not found; cannot determine the PyPI lockout window")
    raw = ((tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool") or {}).get("uv") or {}).get(
        "exclude-newer"
    )
    if raw is None:
        return timedelta(0)
    matched = _DAYS.fullmatch(str(raw))
    if matched is None:
        return WindowError(
            f"[tool.uv] exclude-newer = {raw!r} is not an 'N days' window; teach osv_lockout_filter.py how to read it"
        )
    return timedelta(days=int(matched.group(1)))


def _npm_window(npmrc: Path) -> timedelta:
    if not npmrc.is_file():
        return timedelta(0)
    matched = _MIN_RELEASE_AGE.search(npmrc.read_text(encoding="utf-8"))
    return timedelta(days=int(matched.group(1))) if matched else timedelta(0)


def lockout_windows(repo_root: Path) -> Mapping[str, timedelta] | WindowError:
    pypi = _uv_window(repo_root / "pyproject.toml")
    if isinstance(pypi, WindowError):
        return pypi
    return {PYPI: pypi, NPM: _npm_window(repo_root / ".npmrc")}


def _publish_date(times: Mapping[str, datetime], ecosystem: str, version: str) -> datetime | None:
    if version in times:
        return times[version]
    wanted = version_key(ecosystem, version)
    if wanted is None:
        return None
    return next((stamp for other, stamp in times.items() if version_key(ecosystem, other) == wanted), None)


def evaluate(
    finding: Finding,
    windows: Mapping[str, timedelta],
    publish_times: PublishTimes,
    now: datetime,
) -> Verdict:
    window = windows.get(finding.ecosystem)
    if window is None:
        return Actionable(finding, None, f"no lockout window is configured for the {finding.ecosystem} ecosystem")
    installed = version_key(finding.ecosystem, finding.installed)
    if installed is None:
        return Actionable(finding, None, f"cannot parse the installed version {finding.installed!r}")
    upgrades = tuple(
        (key, version)
        for version in finding.fixed_versions
        for key in (version_key(finding.ecosystem, version),)
        if key is not None and key > installed
    )
    if not upgrades:
        return Actionable(finding, None, "no published fix is newer than the installed version")
    target = sorted(upgrades, key=lambda pair: pair[0])[0][1]
    times = publish_times(finding.ecosystem, finding.package)
    if times is None:
        return Actionable(finding, target, "could not read publish dates from the package registry")
    published = _publish_date(times, finding.ecosystem, target)
    if published is None:
        return Actionable(finding, target, f"the registry lists no publish date for {target}")
    unlocks_at = published + window
    if unlocks_at > now:
        return Deferred(finding, target, published, unlocks_at)
    return Actionable(finding, target, f"{target} has been installable since {_stamp(unlocks_at)}")


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _describe(finding: Finding, target: str | None) -> str:
    upgrade = f"{finding.installed} -> {target}" if target else f"{finding.installed} (no upgrade)"
    return f"{finding.ecosystem} {finding.package} {upgrade}  {finding.vuln_id} (severity {finding.severity})  [{finding.source}]"


def render(
    windows: Mapping[str, timedelta],
    deferred: Sequence[Deferred],
    actionable: Sequence[Actionable],
) -> str:
    header = "Dependency lockout window: " + ", ".join(
        f"{ecosystem} {window.days}d" for ecosystem, window in sorted(windows.items())
    )
    deferred_lines = tuple(
        f"  {_describe(item.finding, item.target)}\n"
        f"    published {_stamp(item.published)}; installable from {_stamp(item.unlocks_at)}"
        for item in deferred
    )
    actionable_lines = tuple(f"  {_describe(item.finding, item.target)}\n    {item.reason}" for item in actionable)
    sections = (
        (
            f"Deferred ({len(deferred)}) - the fix is still inside the lockout window, so it cannot be pulled in yet:",
            deferred_lines or ("  none",),
        ),
        (
            f"Blocking ({len(actionable)}) - the fix can be pulled in now:",
            actionable_lines or ("  none",),
        ),
    )
    return "\n".join((header, "", *(line for title, lines in sections for line in (title, *lines, ""))))


def main(argv: Sequence[str] | None = None, publish_times: PublishTimes = registry_publish_times) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="osv-scanner --format json output file")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)

    windows = lockout_windows(args.repo_root)
    if isinstance(windows, WindowError):
        print(f"osv-lockout: {windows.message}", file=sys.stderr)
        return 1
    if not args.results.is_file():
        print(f"osv-lockout: {args.results} not found", file=sys.stderr)
        return 1

    findings = parse_findings(json.loads(args.results.read_text(encoding="utf-8")), args.repo_root)
    now = datetime.now(timezone.utc)
    verdicts = tuple(evaluate(finding, windows, publish_times, now) for finding in findings)
    deferred = tuple(verdict for verdict in verdicts if isinstance(verdict, Deferred))
    actionable = tuple(verdict for verdict in verdicts if isinstance(verdict, Actionable))

    report = render(windows, deferred, actionable)
    print(report)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"### OSV scan\n\n```\n{report}\n```\n")
    return 1 if actionable else 0


if __name__ == "__main__":
    raise SystemExit(main())
