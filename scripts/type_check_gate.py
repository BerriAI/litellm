#!/usr/bin/env python3
"""Delta-vs-base per-rule gate for basedpyright.

basedpyright's ``--outputjson`` is reduced to a count of errors per *rule*
(``reportAny``, ``reportArgumentType``, ...) and checked against a committed
budget of the form ``{rule: {limit}}``, the same shape as
``ruff-strict-budget.json``. A rule fails only when its codebase-wide total is
both over its ``limit`` *and* higher than the count on the base it merges into,
so a change is blamed for the errors it adds, never for drift that already sits
in the base. That ``> base`` guard is what stops an unrelated PR from inheriting
a red once two PRs each land near the limit and their sum crosses it: the
bystander's count equals its base, so it is spared, while any PR that actually
grows the rule past its limit still fails.

Installed packages are part of the measurement: a typed dependency that is
present changes what basedpyright can prove (and therefore which diagnostics
fire) versus when it is absent, so counts from two differently provisioned
venvs are not comparable and their comparison produces phantom breaches no
diff hunk explains. The gate therefore provisions its own environment at
``.venv-typecheck`` (a frozen ``uv sync`` of one canonical dependency-group
set, plus a generated Prisma client) and runs every basedpyright pass from it,
so pre-commit, the CI lint job, and the artifact publisher measure one package
set by construction; re-syncs of an up-to-date env are a near-instant no-op.
The group set is folded into the cache and artifact fingerprint, so counts
recorded under a different set are never matched, only recomputed.

The gate runs basedpyright itself, for both the head and the base pass, with
``NODE_OPTIONS`` raised to the heap this repo needs: basedpyright's node
process OOMs at the ~4 GB default, and when callers had to remember the flag,
every hand-copied pipeline (Makefile, CI, a dev running the recipe by hand)
was one forgotten env line away from an 80-second crash. The base count only
matters once some rule is over its limit, so when none is the base pass is
skipped outright. When it is needed, it is a second basedpyright pass over a
detached worktree at the merge-base, run under the same environment so import
resolution matches, and its per-rule counts are cached under the repo's git
common dir keyed by merge-base commit, ``pyrightconfig.json``, ``uv.lock``,
the Prisma schema, and the dependency-group set, so re-runs against the same
branch point pay for it once. A CI workflow publishes every staging commit's counts as
an artifact (``--emit-counts-dir`` is its entry point), and on a disk-cache miss
the gate first tries to download the merge-base's artifact through the ``gh``
CLI; any fetch failure falls back silently to the local base pass, so the gate
never gets worse than it was without CI. ``--update`` ratchets each rule's ``limit`` down by the
number of errors this branch fixed relative to its branch point (the merge-base),
so the headroom you were granted shrinks by exactly what you cleared and never
grows.

``--outputjson`` is used rather than text diagnostics because the latter wrap
across lines, leaving the ``(reportRule)`` on a continuation line away from the
``- error:`` marker, so line parsing mis-attributes ~60% of errors -- the JSON
carries an unambiguous ``rule`` field.
"""

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BUDGET_PATH = REPO_ROOT / "basedpyright-code-budget.json"
PYRIGHT_CONFIG = REPO_ROOT / "pyrightconfig.json"
UV_LOCK = REPO_ROOT / "uv.lock"
DEFAULT_BASE = "origin/litellm_internal_staging"
CACHE_FILE_PREFIX = "basedpyright-base-"
CACHE_KEEP_ENTRIES = 8
ARTIFACT_NAME_PREFIX = "basedpyright-counts-"
GH_TIMEOUT_SECONDS = 10

# The one environment every basedpyright pass measures in. The group set is
# the slim one the CI publisher has always installed (not bootstrap's fatter
# --extra proxy env), so the committed budgets stay valid; changing it re-keys
# every cache and artifact fingerprint, so stale counts can never be matched.
TYPECHECK_ENV_DIR = REPO_ROOT / ".venv-typecheck"
TYPECHECK_DEP_GROUPS = ("proxy-dev", "e2e-dev")
PRISMA_GENERATE_SCRIPT = REPO_ROOT / "scripts" / "prisma_generate_if_needed.py"
PRISMA_SCHEMA = REPO_ROOT / "litellm" / "proxy" / "schema.prisma"

# basedpyright's node process needs more than the ~4 GB default heap on this
# repo; appended last so it wins node's last-flag-wins resolution over any
# caller-set value while preserving the caller's other NODE_OPTIONS flags.
NODE_HEAP_OPTION = "--max-old-space-size=8192"

# Bucket for a basedpyright diagnostic with no `rule`. Counted so it's gated.
UNCODED = "<uncoded>"

# Limit for a rule that shows up at HEAD but isn't in the budget at all -- a
# brand-new error category (new construct, or a tool/version change). The rule
# fails once it clears this many errors.
DEFAULT_LIMIT = 10


class Breach(NamedTuple):
    code: str
    total: int
    cap: int
    added: int


def _to_relative(raw: str, root: Path) -> str | None:
    path = Path(raw)
    absolute = path if path.is_absolute() else root / path
    try:
        return absolute.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def count_basedpyright(payload: str, root: Path = REPO_ROOT) -> dict[str, int]:
    """Count in-tree basedpyright errors per rule from `--outputjson`. Warnings
    and information are ignored; only `severity == "error"` is gated. Files
    outside `root` (the venv's site-packages, say) are dropped."""
    try:
        data = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        sys.stderr.write(
            f"basedpyright did not emit valid JSON ({exc}); it likely crashed or "
            f"printed text before the JSON. First 500 chars of its output:\n"
            f"{payload[:500]}\n"
        )
        raise SystemExit(1) from exc
    counts: Counter[str] = Counter()
    for diag in data.get("generalDiagnostics", []):
        if diag.get("severity") != "error":
            continue
        if _to_relative(diag.get("file", ""), root) is None:
            continue
        counts[diag.get("rule") or UNCODED] += 1
    return dict(counts)


def _run(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def node_options_with_heap(base_env: Mapping[str, str]) -> str:
    return f"{base_env.get('NODE_OPTIONS', '')} {NODE_HEAP_OPTION}".strip()


def typecheck_python_version() -> str | None:
    """The interpreter version to build the owned env with, read from
    pyrightconfig's `pythonVersion` so the packages installed for basedpyright
    to see always come from the same version it type-checks against."""
    try:
        config = json.loads(PYRIGHT_CONFIG.read_text())
    except (OSError, ValueError):
        return None
    version: Final = config.get("pythonVersion") if isinstance(config, dict) else None
    return version if isinstance(version, str) else None


def typecheck_env_commands(env_dir: Path = TYPECHECK_ENV_DIR) -> tuple[tuple[str, ...], ...]:
    python_pin: Final = typecheck_python_version()
    sync: Final = (
        "uv",
        "sync",
        "--frozen",
        *(("--python", python_pin) if python_pin else ()),
        *(flag for group in TYPECHECK_DEP_GROUPS for flag in ("--group", group)),
    )
    generate: Final = (str(env_dir / "bin" / "python"), str(PRISMA_GENERATE_SCRIPT))
    return (sync, generate)


def _run_provision_step(cmd: tuple[str, ...], env: Mapping[str, str]) -> int:
    proc = subprocess.run(
        list(cmd), cwd=REPO_ROOT, env=dict(env), capture_output=True, text=True
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def ensure_typecheck_env(
    env_dir: Path = TYPECHECK_ENV_DIR,
    run: Callable[[tuple[str, ...], Mapping[str, str]], int] = _run_provision_step,
) -> Path:
    """Sync the gate-owned venv (and its generated Prisma client) before a
    measurement pass. Unconditional on purpose: an up-to-date env makes both
    steps near-instant no-ops, and skipping them on a heuristic is how the
    measured environment and the fingerprinted one drift apart."""
    if not env_dir.exists():
        sys.stderr.write(
            f"provisioning {env_dir.name} (first run installs packages and "
            "generates the Prisma client; re-runs are near-instant no-ops)\n"
        )
    env: Final = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(env_dir)}
    for cmd in typecheck_env_commands(env_dir):
        if run(cmd, env) != 0:
            raise SystemExit(
                f"could not provision the type-check environment at {env_dir}: "
                f"`{' '.join(cmd)}` failed"
            )
    return env_dir


def run_basedpyright(cwd: Path = REPO_ROOT, env_dir: Path = TYPECHECK_ENV_DIR) -> str:
    """One basedpyright pass over `cwd` from the gate-owned venv, with the
    raised node heap exported.

    `--pythonpath` pins import resolution to the owned env's interpreter; it is
    the only pin that works, because basedpyright auto-detects a `.venv` in the
    project root and that beats both PATH order and VIRTUAL_ENV, silently
    measuring the caller's fatter venv (whose extra typed packages flip
    diagnostics) whenever the repo has one. Exit 0 (clean) and 1 (errors
    found) are both output-bearing runs; anything else is a crash and fails
    loudly instead of reading as zero errors."""
    bin_dir: Final = env_dir / "bin"
    proc = subprocess.run(
        [
            str(bin_dir / "basedpyright"),
            "--outputjson",
            "--pythonpath",
            str(bin_dir / "python"),
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        env={**os.environ, "NODE_OPTIONS": node_options_with_heap(os.environ)},
    )
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"basedpyright exited {proc.returncode}")
    return proc.stdout


def resolve_base_point(base_ref: str, cwd: Path = REPO_ROOT) -> str:
    """The snapshot commit base counts are measured at: merge-base(base_ref, HEAD),
    made aware of an in-progress merge. Mid-merge, HEAD is still the pre-merge tip,
    so its merge-base is the old branch point and every violation the base gained
    since then would be blamed on this change. While MERGE_HEAD exists, prefer
    merge-base(base_ref, MERGE_HEAD) whenever it is the newer of the two."""
    head_point: Final = _run(["git", "merge-base", base_ref, "HEAD"], cwd=cwd).strip()
    if not head_point:
        return base_ref
    merge_head: Final = _run(["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"], cwd=cwd).strip()
    if not merge_head:
        return head_point
    merge_point: Final = _run(["git", "merge-base", base_ref, merge_head], cwd=cwd).strip()
    if not merge_point:
        return head_point
    older: Final = _run(["git", "merge-base", head_point, merge_point], cwd=cwd).strip()
    return merge_point if older == head_point else head_point


@contextlib.contextmanager
def _temp_worktree(ref: str) -> Iterator[Path]:
    parent = Path(tempfile.mkdtemp(prefix="bpr_base_"))
    worktree = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref])
        yield worktree
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)


def base_counts(ref: str) -> dict[str, int]:
    """basedpyright error counts per rule for the merge-base tree. The head
    config is copied in so the base is judged by today's rules, and the run uses
    the head environment's basedpyright (on PATH) so imports resolve the same."""
    with _temp_worktree(ref) as worktree:
        shutil.copy(PYRIGHT_CONFIG, worktree / "pyrightconfig.json")
        return count_basedpyright(run_basedpyright(worktree), root=worktree)


def over_ceiling(
    head: Mapping[str, int], budget: Mapping[str, Mapping[str, int]]
) -> frozenset[str]:
    """Rules whose head count already exceeds their limit.

    A rule can only breach when it is over its limit, so when none are the base
    comparison cannot change the verdict and the base worktree pass can be skipped.
    """
    return frozenset(
        code
        for code, total in head.items()
        if total > (budget[code]["limit"] if code in budget else DEFAULT_LIMIT)
    )


def environment_fingerprints(
    dep_groups: tuple[str, ...] = TYPECHECK_DEP_GROUPS,
) -> tuple[str, ...]:
    return (
        *(
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (PYRIGHT_CONFIG, UV_LOCK, PRISMA_SCHEMA)
            if path.exists()
        ),
        "groups:" + ",".join(dep_groups),
    )


def cache_key(base_point: str, fingerprints: tuple[str, ...]) -> str:
    return hashlib.sha256("|".join((base_point, *fingerprints)).encode()).hexdigest()[
        :16
    ]


def cache_path(
    directory: Path, base_point: str, fingerprints: tuple[str, ...]
) -> Path:
    return directory / f"{CACHE_FILE_PREFIX}{cache_key(base_point, fingerprints)}.json"


def default_cache_dir() -> Path:
    common = Path(_run(["git", "rev-parse", "--git-common-dir"]).strip())
    resolved = common if common.is_absolute() else REPO_ROOT / common
    return resolved / "litellm-lint-cache"


def validated_counts(data: object) -> dict[str, int] | None:
    counts: Final = data.get("counts") if isinstance(data, dict) else None
    if not isinstance(counts, dict):
        return None
    if not all(
        isinstance(code, str) and isinstance(total, int) and not isinstance(total, bool)
        for code, total in counts.items()
    ):
        return None
    return counts


def load_cached_counts(path: Path) -> dict[str, int] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return validated_counts(data)


def scratch_path(path: Path) -> Path:
    """In-flight scratch for the tmp+rename write. Dot-prefixed so the prune
    glob in `store_counts` can never match it (a concurrent run would otherwise
    unlink it between write and rename), and pid-suffixed so two concurrent
    writers of the same entry never share a scratch."""
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def counts_payload(base_point: str, counts: Mapping[str, int]) -> str:
    return (
        json.dumps(
            {"base_point": base_point, "counts": dict(sorted(counts.items()))},
            indent=2,
        )
        + "\n"
    )


def entry_recency(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def evicted_beyond_cap(entries: Sequence[Path], keep: int) -> tuple[Path, ...]:
    newest_first: Final = sorted(entries, key=entry_recency, reverse=True)
    return tuple(newest_first[keep:])


def store_counts(
    directory: Path, path: Path, base_point: str, counts: Mapping[str, int]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    scratch = scratch_path(path)
    scratch.write_text(counts_payload(base_point, counts))
    scratch.replace(path)
    siblings: Final = tuple(
        entry for entry in directory.glob(f"{CACHE_FILE_PREFIX}*.json") if entry != path
    )
    for stale in evicted_beyond_cap(siblings, CACHE_KEEP_ENTRIES - 1):
        stale.unlink(missing_ok=True)


def parse_origin_slug(url: str) -> str | None:
    match: Final = re.fullmatch(
        r"(?:git@github\.com:|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?/?",
        url.strip(),
    )
    return match.group(1) if match else None


def origin_slug() -> str | None:
    proc: Final = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    return parse_origin_slug(proc.stdout)


def artifact_name(base_point: str) -> str:
    return f"{ARTIFACT_NAME_PREFIX}{cache_key(base_point, environment_fingerprints())}"


def _gh_output(args: list[str]) -> bytes | None:
    try:
        proc = subprocess.run(
            ["gh", *args], capture_output=True, timeout=GH_TIMEOUT_SECONDS
        )
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
            members: Final = [
                name for name in archive.namelist() if name.endswith(".json")
            ]
            if len(members) != 1:
                return None
            return json.loads(archive.read(members[0]))
    except (zipfile.BadZipFile, ValueError, OSError):
        return None


def counts_for_base(payload: object, base_point: str) -> dict[str, int] | None:
    if not isinstance(payload, dict) or payload.get("base_point") != base_point:
        return None
    counts: Final = validated_counts(payload)
    return counts if counts else None


def _fetch_fallback(reason: str) -> None:
    sys.stderr.write(f"{reason}; computing base counts locally\n")


def fetch_ci_base_counts(
    base_point: str,
    gh_output: Callable[[list[str]], bytes | None] = _gh_output,
) -> dict[str, int] | None:
    """Base counts from the CI artifact published for `base_point`, or None.

    Every failure mode (no gh, no auth, offline, expired or missing artifact,
    malformed payload, counts for a different commit) returns None so the
    caller falls back to the local base pass; the fetch is an optimization and
    must never make the gate less available than local compute alone."""
    slug: Final = origin_slug()
    if slug is None:
        return _fetch_fallback("origin remote is not a github.com URL")
    name: Final = artifact_name(base_point)
    listing: Final = gh_output(
        ["api", f"repos/{slug}/actions/artifacts?name={name}&per_page=1"]
    )
    if listing is None:
        return _fetch_fallback(f"could not list CI artifacts named {name}")
    url: Final = _artifact_download_url(_parsed_json(listing))
    if url is None:
        return _fetch_fallback(f"no usable CI artifact named {name}")
    zip_bytes: Final = gh_output(["api", url])
    if zip_bytes is None:
        return _fetch_fallback(f"download failed for CI artifact {name}")
    counts: Final = counts_for_base(_counts_json_from_zip(zip_bytes), base_point)
    if counts is None:
        return _fetch_fallback(
            f"CI artifact {name} is not valid base counts for {base_point[:12]}"
        )
    sys.stderr.write(f"base counts fetched from CI artifact {name}\n")
    return counts


def base_counts_cached(
    base_point: str,
    cache_dir: Path | None = None,
    compute: Callable[[str], dict[str, int]] = base_counts,
    fetch: Callable[[str], dict[str, int] | None] = fetch_ci_base_counts,
) -> dict[str, int]:
    """`base_counts` memoized on disk. The base tree at a given commit is
    immutable, so its counts are a pure function of the merge-base plus the
    environment fingerprints in the cache key; an empty result is never stored
    because it is the signature of a crashed pass, not a clean tree. On a disk
    miss the counts CI already published for the merge-base are fetched before
    the expensive local base pass; a fetch miss of any kind computes locally."""
    directory = default_cache_dir() if cache_dir is None else cache_dir
    path = cache_path(directory, base_point, environment_fingerprints())
    cached = load_cached_counts(path)
    if cached is not None:
        return cached
    fetched: Final = fetch(base_point)
    if fetched:
        store_counts(directory, path, base_point, fetched)
        return fetched
    counts = compute(base_point)
    if counts:
        store_counts(directory, path, base_point, counts)
    return counts


def evaluate(
    head: Mapping[str, int],
    base: Mapping[str, int],
    budget: Mapping[str, Mapping[str, int]],
) -> list[Breach]:
    breaches = []
    for code, total in head.items():
        spec = budget.get(code)
        cap = spec["limit"] if spec else DEFAULT_LIMIT
        prior = base.get(code, 0)
        if total > cap and total > prior:
            breaches.append(Breach(code, total, cap, total - prior))
    return sorted(breaches)


def is_vacuous_run(
    counts: Mapping[str, int], budget: Mapping[str, Mapping[str, int]]
) -> bool:
    """True when nothing was parsed but the budget expects errors -- the
    signature of a type checker that produced no output. `run_basedpyright`
    already fails crash exit codes, so this guards the remaining case: a run
    that exits cleanly while emitting nothing, which would otherwise clear
    every limit and pass silently."""
    return not counts and any(spec["limit"] for spec in budget.values())


def ratcheted_budget(
    budget: Mapping[str, Mapping[str, int]],
    current: Mapping[str, int],
    base: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    """Each rule's limit lowered by the errors `current` fixed vs `base`.

    `base` is the count at the branch point (the commit this branch diverged
    from). The drop is clamped to what was actually cleared (a rule that grew
    stays put), so the limit only ever falls. Rules absent from the budget are
    dropped: a genuinely new error category is added to the JSON deliberately,
    not on update.
    """
    return {
        code: {
            "limit": max(0, spec["limit"] - max(0, base.get(code, 0) - current.get(code, 0)))
        }
        for code, spec in sorted(budget.items())
    }


def cmd_update(current: Mapping[str, int], base_ref: str = DEFAULT_BASE) -> None:
    """Ratchet each rule's limit down by the errors this branch fixed.

    `current` is the working-tree count; the reference count comes
    from a second basedpyright pass over a detached worktree at the branch point
    (the merge-base with `base_ref`), so a branch's fixes tighten its own ceilings
    by exactly what they cleared since it diverged, and limits never rise.
    """
    budget = json.loads(BUDGET_PATH.read_text()) if BUDGET_PATH.exists() else {}
    base_point = resolve_base_point(base_ref)
    updated = ratcheted_budget(budget, current, base_counts_cached(base_point))
    BUDGET_PATH.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    cleared = sum(budget[code]["limit"] - updated[code]["limit"] for code in updated)
    print(
        f"Ratcheted basedpyright limits down by {cleared} errors this branch fixed "
        f"across {len(updated)} rules"
    )


def cmd_emit_counts(head: Mapping[str, int], directory: Path, head_sha: str) -> None:
    """Write HEAD's per-rule counts as the file the publisher workflow uploads.

    The filename stem is exactly the artifact name `fetch_ci_base_counts` will
    later look up for this commit, so emit and fetch cannot drift apart. Empty
    counts are refused for the same reason `is_vacuous_run` exists: a pass that
    produced nothing almost certainly crashed, and publishing it would poison
    every branch that fetches it."""
    if not head:
        print(
            "FAIL: basedpyright produced no errors; refusing to publish empty base "
            "counts because the pass almost certainly crashed or emitted nothing."
        )
        raise SystemExit(1)
    name: Final = artifact_name(head_sha)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(counts_payload(head_sha, head))
    print(
        f"Emitted base counts for {head_sha} as {name}.json "
        f"({sum(head.values())} errors total)"
    )


def cmd_check(head: Mapping[str, int], base_ref: str) -> None:
    budget = json.loads(BUDGET_PATH.read_text())
    if is_vacuous_run(head, budget):
        expected = sum(spec["limit"] for spec in budget.values())
        print(
            f"FAIL: basedpyright produced no errors, but {BUDGET_PATH.name} allows "
            f"up to ~{expected}. The type checker almost certainly crashed or emitted "
            f"nothing; refusing to certify a vacuous run."
        )
        raise SystemExit(1)
    if not over_ceiling(head, budget):
        print(
            f"OK: every rule is within its basedpyright limit ({sum(head.values())} errors total)"
        )
        return
    base_point = resolve_base_point(base_ref)
    base = base_counts_cached(base_point)
    if is_vacuous_run(base, budget):
        print(
            f"FAIL: basedpyright produced no errors for the base tree at "
            f"{base_point[:12]}, so every rule would look freshly added. The base "
            f"pass almost certainly crashed; refusing to blame this change for it."
        )
        raise SystemExit(1)
    breaches = evaluate(head, base, budget)
    if not breaches:
        print(
            f"OK: every rule is within its basedpyright limit or no higher than base ({sum(head.values())} errors total)"
        )
        return
    print("FAIL: basedpyright errors exceed the per-rule limit:")
    for breach in breaches:
        print(
            f"  {breach.code}: total {breach.total} over limit {breach.cap} (this change added {breach.added})"
        )
    print(
        "Reduce the new errors or remove an equal number elsewhere; the ceiling is "
        "the limit in basedpyright-code-budget.json."
    )
    summary = "; ".join(f"{b.code} {b.total}/{b.cap} (+{b.added})" for b in breaches)
    print(f"BREACHED RULES: {summary}")
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--emit-counts-dir", type=Path)
    args = parser.parse_args()
    from gate_slot_lock import held_slot

    with held_slot():
        ensure_typecheck_env()
        head = count_basedpyright(run_basedpyright())
        if args.emit_counts_dir is not None:
            cmd_emit_counts(
                head, args.emit_counts_dir, _run(["git", "rev-parse", "HEAD"]).strip()
            )
        elif args.update:
            cmd_update(head, args.base)
        else:
            cmd_check(head, args.base)


if __name__ == "__main__":
    main()
