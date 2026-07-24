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

Head counts are read from stdin (the caller runs basedpyright once and pipes
``--outputjson`` in). The base count only matters once some rule is over its
limit, so when none is the base pass is skipped outright. When it is needed, it
is a second basedpyright pass over a detached worktree at the merge-base, run
under the same environment so import resolution matches, and its per-rule
counts are cached under the repo's git common dir keyed by merge-base commit,
``pyrightconfig.json``, and ``uv.lock``, so re-runs against the same branch
point pay for it once. ``--update`` ratchets each rule's ``limit`` down by the
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
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BUDGET_PATH = REPO_ROOT / "basedpyright-code-budget.json"
PYRIGHT_CONFIG = REPO_ROOT / "pyrightconfig.json"
UV_LOCK = REPO_ROOT / "uv.lock"
DEFAULT_BASE = "origin/litellm_internal_staging"
CACHE_FILE_PREFIX = "basedpyright-base-"

# Bucket for a basedpyright diagnostic with no `rule`. Counted so it's gated.
UNCODED = "<uncoded>"

# Limit for a rule that shows up at HEAD but isn't in the budget at all -- a
# brand-new error category (new construct, or a tool/version change). The rule
# fails once it clears this many errors.
DEFAULT_LIMIT = 10

BASE_PASS_ATTEMPTS = 2


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


def _merge_base(base_ref: str) -> str:
    return _run(["git", "merge-base", base_ref, "HEAD"]).strip() or base_ref


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


def run_base_pass(
    run: Callable[[], "subprocess.CompletedProcess[str]"],
    attempts: int = BASE_PASS_ATTEMPTS,
) -> "subprocess.CompletedProcess[str]":
    for attempt in range(1, attempts + 1):
        proc = run()
        if proc.returncode in (0, 1):
            return proc
        sys.stderr.write(
            f"basedpyright base pass exited {proc.returncode} "
            f"(attempt {attempt}/{attempts}); stderr tail:\n{proc.stderr[-2000:]}\n"
        )
    raise SystemExit(
        f"basedpyright base pass crashed {attempts} times; its exit code and "
        f"stderr are above"
    )


def base_counts(ref: str) -> dict[str, int]:
    """basedpyright error counts per rule for the merge-base tree. The head
    config is copied in so the base is judged by today's rules, and the run uses
    the head environment's basedpyright (on PATH) so imports resolve the same."""
    exe = shutil.which("basedpyright") or "basedpyright"
    with _temp_worktree(ref) as worktree:
        shutil.copy(PYRIGHT_CONFIG, worktree / "pyrightconfig.json")
        proc = run_base_pass(
            lambda: subprocess.run(
                [exe, "--outputjson"], cwd=worktree, capture_output=True, text=True
            )
        )
        return count_basedpyright(proc.stdout, root=worktree)


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


def environment_fingerprints() -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (PYRIGHT_CONFIG, UV_LOCK)
        if path.exists()
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


def load_cached_counts(path: Path) -> dict[str, int] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    counts = data.get("counts") if isinstance(data, dict) else None
    if not isinstance(counts, dict):
        return None
    if not all(
        isinstance(code, str) and isinstance(total, int) and not isinstance(total, bool)
        for code, total in counts.items()
    ):
        return None
    return counts


def scratch_path(path: Path) -> Path:
    """In-flight scratch for the tmp+rename write. Dot-prefixed so the prune
    glob in `store_counts` can never match it (a concurrent run would otherwise
    unlink it between write and rename), and pid-suffixed so two concurrent
    writers of the same entry never share a scratch."""
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def store_counts(
    directory: Path, path: Path, base_point: str, counts: Mapping[str, int]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob(f"{CACHE_FILE_PREFIX}*.json"):
        if stale != path:
            stale.unlink(missing_ok=True)
    scratch = scratch_path(path)
    scratch.write_text(
        json.dumps(
            {"base_point": base_point, "counts": dict(sorted(counts.items()))},
            indent=2,
        )
        + "\n"
    )
    scratch.replace(path)


def base_counts_cached(
    base_point: str,
    cache_dir: Path | None = None,
    compute: Callable[[str], dict[str, int]] = base_counts,
) -> dict[str, int]:
    """`base_counts` memoized on disk. The base tree at a given commit is
    immutable, so its counts are a pure function of the merge-base plus the
    environment fingerprints in the cache key; an empty result is never stored
    because it is the signature of a crashed pass, not a clean tree."""
    directory = default_cache_dir() if cache_dir is None else cache_dir
    path = cache_path(directory, base_point, environment_fingerprints())
    cached = load_cached_counts(path)
    if cached is not None:
        return cached
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
    signature of a type checker that crashed or produced no output. The CI pipe
    swallows the tool's exit code (`tool || true`), so without this guard an
    empty run would clear every limit and pass silently."""
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

    `current` is the working-tree count (piped in); the reference count comes
    from a second basedpyright pass over a detached worktree at the branch point
    (the merge-base with `base_ref`), so a branch's fixes tighten its own ceilings
    by exactly what they cleared since it diverged, and limits never rise.
    """
    budget = json.loads(BUDGET_PATH.read_text()) if BUDGET_PATH.exists() else {}
    base_point = _merge_base(base_ref)
    updated = ratcheted_budget(budget, current, base_counts_cached(base_point))
    BUDGET_PATH.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    cleared = sum(budget[code]["limit"] - updated[code]["limit"] for code in updated)
    print(
        f"Ratcheted basedpyright limits down by {cleared} errors this branch fixed "
        f"across {len(updated)} rules"
    )


def cmd_check(
    base_ref: str,
    head_payload: Callable[[], str] = sys.stdin.read,
    base_counts_for: Callable[[str], dict[str, int]] = base_counts_cached,
    merge_base: Callable[[str], str] = _merge_base,
    budget_path: Path = BUDGET_PATH,
) -> None:
    budget = json.loads(budget_path.read_text())
    head = count_basedpyright(head_payload())
    if is_vacuous_run(head, budget):
        expected = sum(spec["limit"] for spec in budget.values())
        print(
            f"FAIL: basedpyright produced no errors, but {budget_path.name} allows "
            f"up to ~{expected}. The type checker almost certainly crashed or emitted "
            f"nothing; refusing to certify a vacuous run."
        )
        raise SystemExit(1)
    if not over_ceiling(head, budget):
        print(
            f"OK: every rule is within its basedpyright limit ({sum(head.values())} errors total)"
        )
        return
    base_point = merge_base(base_ref)
    base = base_counts_for(base_point)
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
    args = parser.parse_args()
    if args.update:
        cmd_update(count_basedpyright(sys.stdin.read()), args.base)
    else:
        cmd_check(args.base)


if __name__ == "__main__":
    main()
