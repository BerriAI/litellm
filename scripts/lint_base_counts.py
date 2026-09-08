#!/usr/bin/env python3
"""Merge-base counts for the delta-vs-base lint gates.

Each gate (scripts/ruff_strict_gate.py, scripts/type_discipline_gate.py,
scripts/type_check_gate.py, scripts/test_quality_gate.py) counts its rules
across the whole tree at HEAD and at the merge-base with the branch the change
merges into, and fails only when a rule grew past the merge-base count plus
that rule's fixed headroom. There is no committed budget: the merge-base count
is the ceiling, so it moves only when the base branch does.

The merge-base counts come from, in order, the disk cache under the git common
dir, the CI artifact publish-lint-base-counts.yml uploads for every push to the
default branch, and a scan of the base tree in a temporary worktree. Every
entry is keyed by the merge-base commit plus the checker's fingerprints (its
config, its rule logic, its tool version), so counts measured under a different
rule set are never matched, only recomputed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NamedTuple, TypeAlias

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
CACHE_DIR_NAME: Final = "litellm-lint-cache"
CACHE_KEEP_ENTRIES: Final = 8
GH_TIMEOUT_SECONDS: Final = 10

_ORIGIN_SLUG: Final = re.compile(r"(?:git@github\.com:|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?/?")

Counts: TypeAlias = Mapping[str, int]
GhOutput: TypeAlias = Callable[[Sequence[str]], bytes | None]


class Breach(NamedTuple):
    rule: str
    total: int
    cap: int
    added: int


@dataclass(frozen=True, slots=True)
class Checker:
    name: str
    fingerprints: tuple[str, ...]

    def key(self, base_point: str) -> str:
        return cache_key(base_point, self.fingerprints)

    def artifact_name(self, base_point: str) -> str:
        return f"{self.name}-counts-{self.key(base_point)}"

    def cache_file_name(self, base_point: str) -> str:
        return f"{self.name}-base-{self.key(base_point)}.json"

    def cache_glob(self) -> str:
        return f"{self.name}-base-*.json"


Fetch: TypeAlias = Callable[[Checker, str], Counts | None]


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_key(base_point: str, fingerprints: Sequence[str]) -> str:
    return hashlib.sha256("|".join((base_point, *fingerprints)).encode()).hexdigest()[:16]


def _git(args: Sequence[str], cwd: Path) -> str:
    proc: Final = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"git exited {proc.returncode}")
    return proc.stdout


def head_sha(cwd: Path = REPO_ROOT) -> str:
    return _git(["rev-parse", "HEAD"], cwd).strip()


def resolve_base_point(base_ref: str, cwd: Path = REPO_ROOT) -> str:
    """The snapshot commit base counts are measured at: merge-base(base_ref, HEAD),
    made aware of an in-progress merge. Mid-merge, HEAD is still the pre-merge tip,
    so its merge-base is the old branch point and every violation the base gained
    since then would be blamed on this change. While MERGE_HEAD exists, prefer
    merge-base(base_ref, MERGE_HEAD) whenever it is the newer of the two."""
    head_point: Final = _git(["merge-base", base_ref, "HEAD"], cwd).strip()
    if not head_point:
        return base_ref
    merge_head: Final = _git(["rev-parse", "--verify", "--quiet", "MERGE_HEAD"], cwd).strip()
    if not merge_head:
        return head_point
    merge_point: Final = _git(["merge-base", base_ref, merge_head], cwd).strip()
    if not merge_point:
        return head_point
    older: Final = _git(["merge-base", head_point, merge_point], cwd).strip()
    return merge_point if older == head_point else head_point


def default_cache_dir(cwd: Path = REPO_ROOT) -> Path:
    common: Final = Path(_git(["rev-parse", "--git-common-dir"], cwd).strip())
    resolved: Final = common if common.is_absolute() else cwd / common
    return resolved / CACHE_DIR_NAME


def validated_counts(data: object) -> Counts | None:
    counts: Final = data.get("counts") if isinstance(data, dict) else None
    if not isinstance(counts, dict):
        return None
    if not all(
        isinstance(code, str) and isinstance(total, int) and not isinstance(total, bool)
        for code, total in counts.items()
    ):
        return None
    return counts


def load_cached_counts(path: Path) -> Counts | None:
    try:
        data: Final = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return validated_counts(data)


def scratch_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def counts_payload(base_point: str, counts: Counts) -> str:
    return json.dumps({"base_point": base_point, "counts": dict(sorted(counts.items()))}, indent=2) + "\n"


def entry_recency(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def evicted_beyond_cap(entries: Sequence[Path], keep: int) -> tuple[Path, ...]:
    newest_first: Final = sorted(entries, key=entry_recency, reverse=True)
    return tuple(newest_first[keep:])


def store_counts(directory: Path, checker: Checker, base_point: str, counts: Counts) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / checker.cache_file_name(base_point)
    scratch: Final = scratch_path(path)
    scratch.write_text(counts_payload(base_point, counts))
    scratch.replace(path)
    siblings: Final = tuple(entry for entry in directory.glob(checker.cache_glob()) if entry != path)
    for stale in evicted_beyond_cap(siblings, CACHE_KEEP_ENTRIES - 1):
        stale.unlink(missing_ok=True)
    return path


def parse_origin_slug(url: str) -> str | None:
    match: Final = _ORIGIN_SLUG.fullmatch(url.strip())
    return match.group(1) if match else None


def origin_slug(cwd: Path = REPO_ROOT) -> str | None:
    proc: Final = subprocess.run(["git", "remote", "get-url", "origin"], cwd=cwd, capture_output=True, text=True)
    return parse_origin_slug(proc.stdout) if proc.returncode == 0 else None


def gh_output(args: Sequence[str]) -> bytes | None:
    try:
        proc: Final = subprocess.run(["gh", *args], capture_output=True, timeout=GH_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


def _parsed_json(raw: bytes) -> object | None:
    try:
        return json.loads(raw)
    except ValueError:
        return None


def _artifact_download_url(listing: object) -> str | None:
    artifacts: Final = listing.get("artifacts") if isinstance(listing, dict) else None
    if not isinstance(artifacts, list) or not artifacts:
        return None
    newest: Final = artifacts[0]
    if not isinstance(newest, dict) or newest.get("expired"):
        return None
    url: Final = newest.get("archive_download_url")
    return url if isinstance(url, str) else None


def _counts_json_from_zip(zip_bytes: bytes) -> object | None:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            members: Final = tuple(name for name in archive.namelist() if name.endswith(".json"))
            if len(members) != 1:
                return None
            return json.loads(archive.read(members[0]))
    except (zipfile.BadZipFile, ValueError, OSError):
        return None


def counts_for_base(payload: object, base_point: str) -> Counts | None:
    if not isinstance(payload, dict) or payload.get("base_point") != base_point:
        return None
    counts: Final = validated_counts(payload)
    return counts if counts else None


def _fetch_fallback(reason: str) -> None:
    sys.stderr.write(f"{reason}; computing base counts locally\n")


def fetch_ci_base_counts(
    checker: Checker,
    base_point: str,
    gh: GhOutput = gh_output,
    cwd: Path = REPO_ROOT,
) -> Counts | None:
    """Base counts from the CI artifact published for `base_point`, or None.

    Every failure mode (no gh, no auth, offline, expired or missing artifact,
    malformed payload, counts for a different commit) returns None so the
    caller falls back to the local base scan; the fetch is an optimization and
    must never make the gate less available than local compute alone."""
    slug: Final = origin_slug(cwd)
    if slug is None:
        return _fetch_fallback("origin remote is not a github.com URL")
    name: Final = checker.artifact_name(base_point)
    listing: Final = gh(["api", f"repos/{slug}/actions/artifacts?name={name}&per_page=1"])
    if listing is None:
        return _fetch_fallback(f"could not list CI artifacts named {name}")
    url: Final = _artifact_download_url(_parsed_json(listing))
    if url is None:
        return _fetch_fallback(f"no usable CI artifact named {name}")
    zip_bytes: Final = gh(["api", url])
    if zip_bytes is None:
        return _fetch_fallback(f"download failed for CI artifact {name}")
    counts: Final = counts_for_base(_counts_json_from_zip(zip_bytes), base_point)
    if counts is None:
        return _fetch_fallback(f"CI artifact {name} is not valid base counts for {base_point[:12]}")
    sys.stderr.write(f"base counts fetched from CI artifact {name}\n")
    return counts


def base_counts_cached(
    checker: Checker,
    base_point: str,
    compute: Callable[[str], Counts],
    cache_dir: Path | None = None,
    fetch: Fetch = fetch_ci_base_counts,
) -> Counts:
    """`compute` memoized on disk. The base tree at a given commit is immutable,
    so its counts are a pure function of the merge-base plus the checker's
    fingerprints in the cache key; an empty result is never stored because it is
    the signature of a crashed pass, not a clean tree. On a disk miss the counts
    CI already published for the merge-base are fetched before the expensive
    local base scan; a fetch miss of any kind computes locally."""
    directory: Final = default_cache_dir() if cache_dir is None else cache_dir
    cached: Final = load_cached_counts(directory / checker.cache_file_name(base_point))
    if cached is not None:
        return cached
    fetched: Final = fetch(checker, base_point)
    if fetched:
        store_counts(directory, checker, base_point, fetched)
        return fetched
    counts: Final = compute(base_point)
    if counts:
        store_counts(directory, checker, base_point, counts)
    return counts


def emit_counts(checker: Checker, counts: Counts, directory: Path, head_point: str) -> Path:
    """Write HEAD's per-rule counts as the file the publisher workflow uploads.

    The filename stem is exactly the artifact name `fetch_ci_base_counts` will
    later look up for this commit, so emit and fetch cannot drift apart. Empty
    counts are refused: a pass that produced nothing almost certainly crashed,
    and publishing it would poison every branch that fetches it."""
    if not counts:
        print(
            f"FAIL: {checker.name} produced no violations; refusing to publish empty base "
            "counts because the pass almost certainly crashed or emitted nothing."
        )
        raise SystemExit(1)
    name: Final = checker.artifact_name(head_point)
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / f"{name}.json"
    path.write_text(counts_payload(head_point, counts))
    print(f"Emitted base counts for {head_point} as {name}.json ({sum(counts.values())} violations total)")
    return path


def evaluate(head: Counts, base: Counts, headroom: Counts) -> tuple[Breach, ...]:
    return tuple(
        Breach(rule, total, base.get(rule, 0) + headroom.get(rule, 0), total - base.get(rule, 0))
        for rule, total in sorted(head.items())
        if total > base.get(rule, 0) + headroom.get(rule, 0)
    )
