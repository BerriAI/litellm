"""Daily-cron matrix publisher (GCP VM edition).

End-to-end orchestrator that runs on the dedicated cron VM
`litellm-compatibility-matrix-populator`. The flow is:

  1. Resolve the latest LiteLLM ``v*-stable`` tag via ``resolver.py``.
  2. Update a long-lived git worktree of ``BerriAI/litellm`` to that tag,
     run ``uv sync --frozen`` against it, and boot the proxy as a
     subprocess on a non-conflicting local port. Reusing the same worktree
     across runs (rather than a fresh tempdir) keeps disk footprint
     bounded — ``uv sync`` removes packages no longer pinned and ``git
     checkout`` mutates the same files in place.
  3. Run ``pytest tests/claude_code/`` against the proxy. The locally
     installed Claude Code CLI is exercised as-is — there is no
     ``npm install`` step, so the operator controls when the CLI is
     upgraded by running ``npm install -g @anthropic-ai/claude-code@latest``
     out-of-band (typically baked into the VM image or a separate cron).
  4. Build ``compatibility-matrix.json`` from the per-test results
     artifact using the Matrix JSON Builder (``matrix_builder.py``).
  5. Open (or update) a pull request against the docs repo with the JSON
     change, using a ``gh`` CLI that has been pre-authenticated on the VM
     against an account with ``pull-requests: write`` on
     ``BerriAI/litellm-docs``.

Why no Docker
-------------

The original design pulled ``ghcr.io/berriai/litellm:<tag>`` per run on
GitHub-hosted runners. The cron VM does not run docker — installing it
would just trade one set of moving parts (docker daemon, image pulls,
networking) for the simpler "one git checkout + one ``uv sync``" we
already use to start the proxy interactively. Removing the docker code
also halves the publisher's surface area.

Idempotency
-----------

Re-runs on the same UTC day with the same resolved versions land on the
same head branch (``compat-matrix/<litellm-version>-<claude-code-
version>-<UTC-date>``). If the JSON is byte-identical to the docs repo's
target branch, the script exits before pushing. If a PR is already open
for the branch, ``gh pr create`` no-ops with a non-fatal message; we
treat that as success.

The "only ``compatibility-matrix.json`` is ever committed" guarantee is
enforced by ``select_files_to_commit`` rather than by token scope, since
GitHub does not expose file-path-scoped tokens.
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from tests.claude_code.matrix_builder import build_from_paths
from tests.claude_code.resolver import latest_stable_litellm_tag

DOCS_REPO_DEFAULT = "BerriAI/litellm-docs"
DOCS_TARGET_BASENAME = "compatibility-matrix.json"
DOCS_TARGET_PATH_DEFAULT = f"static/data/{DOCS_TARGET_BASENAME}"
PR_BRANCH_PREFIX = "compat-matrix"

# Sourced from the same config the human-tended dev proxy uses, but on a
# different port so a developer running the proxy on :4000 doesn't
# collide with the cron run.
DEFAULT_PROXY_PORT = 4100
DEFAULT_PROXY_API_KEY = "sk-cron-matrix"  # the proxy never sees real auth
DEFAULT_PROXY_HEALTH_TIMEOUT_SECONDS = 90

# Location of a persistent litellm checkout that the cron mutates each
# run (``git checkout <tag>`` + ``uv sync``). Persisting across runs
# keeps disk usage bounded. Override via the ``LITELLM_WORKTREE`` env var
# or ``--worktree`` so the cron can target a deliberate path on the VM.
DEFAULT_WORKTREE = Path.home() / "litellm-cron-worktree"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "claude_code" / "manifest.yaml"
DEFAULT_RESULTS = REPO_ROOT / "compat-results.json"


# ---------------------------------------------------------------------------
# Pure helpers — unit-tested under _publisher_unit_tests/.
# ---------------------------------------------------------------------------


def commit_message_for_matrix(matrix: Mapping[str, Any]) -> str:
    """Build a deterministic commit message for the docs-repo push.

    Surfaces the three pieces of provenance the docs banner shows
    (LiteLLM version, Claude Code version, generated_at) so the docs
    repo's git log is self-describing without opening the JSON.
    """
    litellm_version = matrix.get("litellm_version", "")
    claude_code_version = matrix.get("claude_code_version", "")
    generated_at = matrix.get("generated_at", "")
    headline = "Update Claude Code compatibility matrix"
    body_lines = [
        f"litellm_version: {litellm_version}",
        f"claude_code_version: {claude_code_version}",
        f"generated_at: {generated_at}",
    ]
    return headline + "\n\n" + "\n".join(body_lines) + "\n"


def pr_branch_name(
    *, litellm_version: str, claude_code_version: str, date_utc: str
) -> str:
    """Deterministic branch name for the docs-repo PR.

    Two cron runs that resolve to the same (litellm_version,
    claude_code_version, UTC date) land on the same branch and therefore
    the same PR — the second push is a force-with-lease update of the
    existing branch and ``gh pr create`` no-ops.

    Each component is required because:
      - ``litellm_version`` distinguishes consecutive stable tags
      - ``claude_code_version`` distinguishes a Claude-Code-only refresh
      - ``date_utc`` lets us still produce a fresh branch when neither
        upstream version moved but a maintainer manually re-ran on a
        later day to recover from a transient failure
    """
    if not litellm_version:
        raise ValueError("litellm_version must be a non-empty string")
    if not claude_code_version:
        raise ValueError("claude_code_version must be a non-empty string")
    if not date_utc:
        raise ValueError("date_utc must be a non-empty string")
    return f"{PR_BRANCH_PREFIX}/{litellm_version}-{claude_code_version}-{date_utc}"


def pr_title_for_matrix(matrix: Mapping[str, Any]) -> str:
    """Single-line PR title with the resolved versions inline.

    Mirrors the commit headline so notification subject lines and the
    docs repo's PR list are immediately self-describing — maintainers
    can triage from the inbox without opening the diff.
    """
    litellm_version = matrix.get("litellm_version", "")
    claude_code_version = matrix.get("claude_code_version", "")
    return (
        "chore(compat-matrix): refresh for "
        f"{litellm_version} + claude-code {claude_code_version}"
    )


def pr_body_for_matrix(matrix: Mapping[str, Any]) -> str:
    """Markdown body summarising the matrix at a glance.

    Renders one line per feature with the per-provider statuses inline,
    so reviewers don't have to diff the JSON to see what changed since
    the last refresh. The status column ordering follows the manifest
    (which the matrix already reflects in ``providers``), keeping the
    table stable across days.
    """
    litellm_version = matrix.get("litellm_version", "")
    claude_code_version = matrix.get("claude_code_version", "")
    generated_at = matrix.get("generated_at", "")
    providers = list(matrix.get("providers", []))
    features = list(matrix.get("features", []))

    lines: List[str] = [
        "Automated daily refresh of the Claude Code compatibility matrix.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| litellm_version | `{litellm_version}` |",
        f"| claude_code_version | `{claude_code_version}` |",
        f"| generated_at | `{generated_at}` |",
        "",
        "## Per-feature results",
        "",
    ]
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        name = feature.get("name") or feature.get("id") or ""
        cells = feature.get("providers") or {}
        per_provider = ", ".join(
            f"{provider}={(cells.get(provider) or {}).get('status', 'not_tested')}"
            for provider in providers
        )
        lines.append(f"- **{name}**: {per_provider}")

    lines.extend(
        [
            "",
            "---",
            "",
            "Generated by `tests/claude_code/publisher.py`. "
            "Close without merging if the diff looks wrong; the next "
            "cron run will reopen with fresh results.",
            "",
        ]
    )
    return "\n".join(lines)


def select_files_to_commit(
    staged_paths: Sequence[str], allowed_basename: str
) -> List[str]:
    """Return only the paths whose basename matches the allowlist.

    Even though the docs-repo PR is reviewable, the publisher still
    enforces a one-file allowlist as defence in depth — a stray file in
    the working tree from a future feature must not be smuggled into the
    PR by accident.
    """
    return [p for p in staged_paths if os.path.basename(p) == allowed_basename]


# ---------------------------------------------------------------------------
# Subprocess + filesystem glue. Side-effectful, deliberately not unit-tested
# (the daily cron's failure surface is the test).
# ---------------------------------------------------------------------------


def _now_utc_iso() -> str:
    """ISO-8601 UTC timestamp with ``Z`` suffix, matching the v1 schema."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today_utc_date() -> str:
    """Fallback UTC date string used when ``generated_at`` is missing."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")


def _run(cmd: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Print + run subprocess; raise on nonzero exit unless caller opts out."""
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def _get_claude_code_version() -> str:
    """Return the version string printed by ``claude --version``."""
    completed = subprocess.run(
        ["claude", "--version"], capture_output=True, text=True, check=True
    )
    # ``claude --version`` prints e.g. "2.1.120 (Claude Code)"; the first
    # whitespace-delimited token is the version.
    out = (completed.stdout or "").strip()
    return out.split()[0] if out else ""


def _ensure_worktree(worktree: Path) -> None:
    """Make sure ``worktree`` is a working litellm checkout.

    On first run we ``git clone`` ``BerriAI/litellm`` into the worktree
    path. On subsequent runs we reuse what's already there — the run
    just needs ``git fetch && git checkout <tag>``.

    Why a clone instead of a ``git worktree`` of the dev checkout: the
    dev checkout (``~/litellm/litellm``) is where humans iterate and may
    sit on uncommitted changes or arbitrary feature branches. A separate
    clone keeps the cron's ``git checkout <tag>`` from disturbing that.
    """
    if (worktree / ".git").exists():
        return
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "https://github.com/BerriAI/litellm.git",
            str(worktree),
        ]
    )


def _checkout_tag_in_worktree(worktree: Path, tag: str) -> None:
    """Fetch and ``git checkout <tag>`` inside ``worktree``.

    Uses ``git checkout --force`` so any cruft left behind by a previous
    run (e.g. ``compat-results.json``, ``__pycache__``) is wiped before
    a fresh ``uv sync``. The cron has no use for that state — every run
    starts from the published tag.
    """
    _run(["git", "fetch", "--tags", "--force"], cwd=worktree)
    _run(["git", "reset", "--hard"], cwd=worktree)
    _run(["git", "clean", "-fdx", "-e", ".venv"], cwd=worktree)
    _run(["git", "checkout", "--force", tag], cwd=worktree)


def _uv_sync(worktree: Path) -> None:
    """Bring the worktree's ``.venv`` in line with the checked-out tag.

    ``uv sync --frozen`` is deterministic: it installs exactly what the
    lockfile says and removes anything no longer referenced. That's the
    "doesn't blow up storage" property the operator requires — the venv
    can never grow unboundedly across runs.
    """
    _run(["uv", "sync", "--frozen"], cwd=worktree)


def _start_proxy(worktree: Path, port: int, config_path: Path) -> subprocess.Popen:
    """Start the LiteLLM proxy as a subprocess; returns the Popen handle.

    Started in its own process group so we can SIGTERM the whole tree
    on shutdown — the proxy itself spawns worker subprocesses that
    don't otherwise propagate signals from a parent.
    """
    cmd = [
        "uv",
        "run",
        "litellm",
        "--config",
        str(config_path),
        "--port",
        str(port),
    ]
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.Popen(
        cmd,
        cwd=worktree,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )


def _stop_proxy(proc: subprocess.Popen) -> None:
    """SIGTERM the proxy's process group; SIGKILL if it doesn't exit."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def _wait_for_proxy(
    port: int, timeout_seconds: int = DEFAULT_PROXY_HEALTH_TIMEOUT_SECONDS
) -> None:
    """Poll ``/health/liveliness`` until it returns 200 or we time out."""
    url = f"http://127.0.0.1:{port}/health/liveliness"
    deadline = time.time() + timeout_seconds
    last_err: Optional[BaseException] = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_err = exc
        time.sleep(2)
    raise RuntimeError(
        f"proxy did not become healthy within {timeout_seconds}s: {last_err!r}"
    )


# ---------------------------------------------------------------------------
# PR creation against the docs repo.
# ---------------------------------------------------------------------------


def publish(
    *,
    docs_repo: str,
    docs_branch: str,
    docs_target_path: str,
    manifest_path: Path,
    results_path: Path,
    matrix_output_path: Path,
    litellm_version: str,
    claude_code_version: str,
    generated_at: str,
) -> None:
    """Build the matrix JSON and open a PR against the docs repo.

    Auth uses whatever ``gh auth login`` has stashed on the cron VM —
    the GCP edition does not pass an explicit token. The account ``gh``
    is logged in as must be a collaborator on ``docs_repo`` with
    ``pull-requests: write`` (the same permission ``gh pr create``
    needs interactively).
    """
    matrix = build_from_paths(
        manifest_path=manifest_path,
        results_path=results_path,
        litellm_version=litellm_version,
        claude_code_version=claude_code_version,
        generated_at=generated_at,
        output_path=matrix_output_path,
    )

    branch_name = pr_branch_name(
        litellm_version=litellm_version,
        claude_code_version=claude_code_version,
        date_utc=generated_at[:10] if generated_at else _today_utc_date(),
    )

    with tempfile.TemporaryDirectory(prefix="docs-repo-") as workdir:
        workdir_path = Path(workdir)
        # ``gh repo clone`` reuses the VM's ``gh auth`` state, so we
        # don't need to construct an authenticated URL by hand.
        _run(
            [
                "gh",
                "repo",
                "clone",
                docs_repo,
                str(workdir_path),
                "--",
                "--depth",
                "1",
                "--branch",
                docs_branch,
            ]
        )
        _run(
            ["git", "config", "user.email", "litellm-bot@berri.ai"],
            cwd=workdir_path,
        )
        _run(
            ["git", "config", "user.name", "litellm-compat-matrix-bot"],
            cwd=workdir_path,
        )
        # Branch always starts from the freshly-cloned base. If the
        # remote branch already exists from an earlier run on the same
        # day, the later ``git push --force-with-lease`` reconciles —
        # we'd rather present the latest matrix JSON than preserve a
        # stale intermediate state.
        _run(["git", "checkout", "-b", branch_name], cwd=workdir_path)

        target_in_docs = workdir_path / docs_target_path
        target_in_docs.parent.mkdir(parents=True, exist_ok=True)
        target_in_docs.write_text(matrix_output_path.read_text())

        # Defence in depth: even if some other tool dropped a file in
        # the working tree, only the matrix JSON is staged.
        staged = [docs_target_path]
        keep = select_files_to_commit(staged, DOCS_TARGET_BASENAME)
        if not keep:
            raise RuntimeError(
                "no allowed files to commit; expected "
                f"{DOCS_TARGET_BASENAME!r} but got {staged!r}"
            )
        for path in keep:
            _run(["git", "add", path], cwd=workdir_path)

        # Skip the push entirely if the JSON is byte-identical to what's
        # already on ``docs_branch`` — keeps the docs-repo PR list clean
        # during idempotent reruns of the cron.
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=workdir_path,
            check=False,
        )
        if diff.returncode == 0:
            print("matrix JSON unchanged; skipping PR", flush=True)
            return

        _run(
            ["git", "commit", "-m", commit_message_for_matrix(matrix)],
            cwd=workdir_path,
        )
        # ``--force-with-lease`` so a same-day rerun updates the existing
        # branch (and therefore the existing PR) safely; the lease check
        # ensures we never overwrite a docs-maintainer's manual fixup
        # commit on the same branch.
        _run(
            [
                "git",
                "push",
                "--force-with-lease",
                "--set-upstream",
                "origin",
                branch_name,
            ],
            cwd=workdir_path,
        )

        _open_or_update_pr(
            docs_repo=docs_repo,
            docs_branch=docs_branch,
            head_branch=branch_name,
            title=pr_title_for_matrix(matrix),
            body=pr_body_for_matrix(matrix),
            cwd=workdir_path,
        )


def _open_or_update_pr(
    *,
    docs_repo: str,
    docs_branch: str,
    head_branch: str,
    title: str,
    body: str,
    cwd: Path,
) -> None:
    """Open a PR via the ``gh`` CLI; treat 'already exists' as success."""
    completed = subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            docs_repo,
            "--base",
            docs_branch,
            "--head",
            head_branch,
            "--title",
            title,
            "--body",
            body,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        print(completed.stdout.strip(), flush=True)
        return
    stderr = completed.stderr or ""
    # ``gh pr create`` exits non-zero when a PR already exists for the
    # head branch. That's the idempotent re-run path and not an error:
    # the branch was already force-pushed above, so the existing PR now
    # carries the freshest matrix JSON.
    if "a pull request for branch" in stderr and "already exists" in stderr:
        print(
            f"PR already exists for {head_branch}; updated branch in place",
            flush=True,
        )
        return
    sys.stderr.write(stderr)
    raise RuntimeError(f"gh pr create failed with exit code {completed.returncode}")


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-repo",
        default=os.environ.get("DOCS_REPO", DOCS_REPO_DEFAULT),
        help="`owner/name` of the docs repo to publish into.",
    )
    parser.add_argument(
        "--docs-branch",
        default=os.environ.get("DOCS_BRANCH", "main"),
        help="Base branch on the docs repo to PR against.",
    )
    parser.add_argument(
        "--docs-target-path",
        default=os.environ.get("DOCS_TARGET_PATH", DOCS_TARGET_PATH_DEFAULT),
        help="Path inside the docs repo where the matrix JSON lives.",
    )
    parser.add_argument(
        "--worktree",
        type=Path,
        default=Path(os.environ.get("LITELLM_WORKTREE", str(DEFAULT_WORKTREE))),
        help=(
            "Persistent litellm checkout the cron mutates each run "
            f"(default: {DEFAULT_WORKTREE})."
        ),
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=int(os.environ.get("PROXY_PORT", DEFAULT_PROXY_PORT)),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest the matrix JSON is built against.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Path where pytest will write the compat-results.json artifact.",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=REPO_ROOT / DOCS_TARGET_BASENAME,
    )
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="Run the test pipeline but do not open a PR.",
    )
    args = parser.parse_args(argv)

    proxy_proc: Optional[subprocess.Popen] = None
    try:
        litellm_version = latest_stable_litellm_tag(
            token=os.environ.get("GITHUB_TOKEN")
        )
        print(f"resolved latest stable litellm: {litellm_version}", flush=True)

        claude_code_version = _get_claude_code_version()
        if not claude_code_version:
            print(
                "could not read 'claude --version'; is the CLI installed?",
                file=sys.stderr,
            )
            return 2
        print(f"local claude code cli: {claude_code_version}", flush=True)

        _ensure_worktree(args.worktree)
        _checkout_tag_in_worktree(args.worktree, litellm_version)
        _uv_sync(args.worktree)

        config_path = args.worktree / "tests" / "claude_code" / "test_config.yaml"
        if not config_path.exists():
            raise RuntimeError(
                f"proxy config not found at {config_path}; the resolved tag "
                f"{litellm_version} may predate the compat matrix work"
            )

        proxy_proc = _start_proxy(args.worktree, args.proxy_port, config_path)
        _wait_for_proxy(args.proxy_port)

        env = {
            **os.environ,
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{args.proxy_port}",
            "ANTHROPIC_AUTH_TOKEN": DEFAULT_PROXY_API_KEY,
            "COMPAT_RESULTS_PATH": str(args.results),
        }
        pytest_cmd = [
            "uv",
            "run",
            "pytest",
            "tests/claude_code/",
            "--ignore=tests/claude_code/_driver_unit_tests",
            "--ignore=tests/claude_code/_builder_unit_tests",
            "--ignore=tests/claude_code/_publisher_unit_tests",
            "--ignore=tests/claude_code/_pr_gate_unit_tests",
        ]
        # Operator escape hatch: PYTEST_K narrows the run to a single
        # cell or feature for first-time validation after a CLI/proxy
        # upgrade. The matrix builder fills cells we didn't touch with
        # `not_tested`, so a PYTEST_K-narrowed run is safe to publish —
        # though in practice it's used with --skip-publish.
        pytest_k = os.environ.get("PYTEST_K", "").strip()
        if pytest_k:
            pytest_cmd.extend(["-k", pytest_k])
            print(f"PYTEST_K set; narrowing pytest to: {pytest_k}", flush=True)
        # Run pytest from inside the worktree so it picks up the
        # checked-out tag's test code (and its conftest hook), not the
        # current process's working directory.
        subprocess.run(pytest_cmd, env=env, cwd=args.worktree, check=False)

        if args.skip_publish:
            print("skip-publish: not opening a PR", flush=True)
            return 0

        publish(
            docs_repo=args.docs_repo,
            docs_branch=args.docs_branch,
            docs_target_path=args.docs_target_path,
            manifest_path=args.manifest,
            results_path=args.results,
            matrix_output_path=args.matrix_output,
            litellm_version=litellm_version,
            claude_code_version=claude_code_version,
            generated_at=_now_utc_iso(),
        )
        return 0
    finally:
        if proxy_proc is not None:
            _stop_proxy(proxy_proc)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DOCS_REPO_DEFAULT",
    "DOCS_TARGET_BASENAME",
    "DOCS_TARGET_PATH_DEFAULT",
    "DEFAULT_PROXY_PORT",
    "DEFAULT_PROXY_API_KEY",
    "DEFAULT_WORKTREE",
    "PR_BRANCH_PREFIX",
    "commit_message_for_matrix",
    "select_files_to_commit",
    "pr_branch_name",
    "pr_title_for_matrix",
    "pr_body_for_matrix",
    "publish",
    "main",
]
