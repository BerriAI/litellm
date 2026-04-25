"""Daily-cron matrix publisher.

End-to-end orchestrator that runs on the isolated cron VM (per the PRD's
"Two CI environments / Daily Cron" section). The flow is:

  1. Resolve the latest LiteLLM `v*-stable` tag via `resolver.py`.
  2. Pull the corresponding Docker image and start it as the proxy.
  3. Install the absolute latest Claude Code CLI from npm.
  4. Run `pytest tests/claude_code/` against the proxy.
  5. Build `compatibility-matrix.json` from the per-test results artifact
     using the Matrix JSON Builder (`matrix_builder.py`).
  6. Direct-push the JSON to the docs repo's main branch using the
     GitHub App installation token mounted as `DOCS_REPO_TOKEN`.

The orchestration is thin glue over Docker, git, npm and subprocess — per
the PRD's "Testing Decisions" section, it intentionally ships without a
unit-test harness; the daily-cron failure surface is itself the test. The
pure helpers below (commit message, image-name builder, file allowlist)
are unit-tested under `_publisher_unit_tests/`.

The "only `compatibility-matrix.json` is ever committed" guarantee is
enforced by `select_files_to_commit` rather than by token scope, since
GitHub Apps cannot scope `contents: write` to a single file path.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from tests.claude_code.matrix_builder import build_from_paths
from tests.claude_code.resolver import latest_stable_litellm_tag

DOCS_REPO_DEFAULT = "BerriAI/litellm-docs"
DOCS_TARGET_BASENAME = "compatibility-matrix.json"
DOCS_TARGET_PATH_DEFAULT = f"static/data/{DOCS_TARGET_BASENAME}"
DOCKER_IMAGE_BASE = "ghcr.io/berriai/litellm"
DEFAULT_PROXY_PORT = 4000
DEFAULT_PROXY_API_KEY = "sk-cron-matrix"  # only used inside the ephemeral VM

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "claude_code" / "manifest.yaml"
DEFAULT_RESULTS = REPO_ROOT / "compat-results.json"


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


def docker_image_for_tag(tag: str) -> str:
    """Return the ghcr.io image reference for a `v*-stable` tag."""
    if not tag:
        raise ValueError("tag must be a non-empty string")
    return f"{DOCKER_IMAGE_BASE}:{tag}"


def select_files_to_commit(
    staged_paths: Sequence[str], allowed_basename: str
) -> List[str]:
    """Return only the paths whose basename matches the allowlist.

    The cron VM's GitHub App holds `contents: write` on the entire docs
    repo (GitHub does not support file-path-scoped tokens), so the
    "only ship the matrix JSON" property is enforced here instead. Any
    stray file in the working tree is dropped before the commit step.
    """
    return [p for p in staged_paths if os.path.basename(p) == allowed_basename]


def _now_utc_iso() -> str:
    """ISO-8601 UTC timestamp with `Z` suffix, matching the v1 schema."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(cmd: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Print + run subprocess; raise on nonzero exit unless caller opts out."""
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def _get_claude_code_version() -> str:
    """Return the version string printed by `claude --version`."""
    completed = subprocess.run(
        ["claude", "--version"], capture_output=True, text=True, check=True
    )
    # `claude --version` prints e.g. "2.1.120 (Claude Code)"; we keep the
    # raw first whitespace-delimited token, which is the version.
    out = (completed.stdout or "").strip()
    return out.split()[0] if out else ""


def _start_proxy(image: str, port: int) -> str:
    """Start the LiteLLM proxy via `docker run -d`; returns the container id."""
    completed = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            f"{port}:4000",
            "--name",
            "litellm-compat-matrix-proxy",
            image,
            "--port",
            "4000",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    container_id = (completed.stdout or "").strip()
    if not container_id:
        raise RuntimeError("docker run did not return a container id")
    return container_id


def _stop_proxy(container_id: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_id], check=False)


def _wait_for_proxy(port: int, timeout_seconds: int = 60) -> None:
    """Poll the proxy's /health endpoint until it returns 200 or we time out."""
    import urllib.error
    import urllib.request

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


def publish(
    *,
    docs_repo: str,
    docs_branch: str,
    docs_target_path: str,
    docs_token: str,
    manifest_path: Path,
    results_path: Path,
    matrix_output_path: Path,
    litellm_version: str,
    claude_code_version: str,
    generated_at: str,
) -> None:
    """Build the matrix JSON and direct-push it to the docs repo's branch.

    Only `docs_target_path` is staged from the docs-repo working tree —
    any other file produced by the build is dropped via
    `select_files_to_commit`.
    """
    matrix = build_from_paths(
        manifest_path=manifest_path,
        results_path=results_path,
        litellm_version=litellm_version,
        claude_code_version=claude_code_version,
        generated_at=generated_at,
        output_path=matrix_output_path,
    )

    with tempfile.TemporaryDirectory(prefix="docs-repo-") as workdir:
        workdir_path = Path(workdir)
        clone_url = f"https://x-access-token:{docs_token}@github.com/{docs_repo}.git"
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                docs_branch,
                clone_url,
                str(workdir_path),
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

        target_in_docs = workdir_path / docs_target_path
        target_in_docs.parent.mkdir(parents=True, exist_ok=True)
        target_in_docs.write_text(matrix_output_path.read_text())

        # Defense in depth: even if some other tool dropped a file in the
        # working tree, only the matrix JSON is staged.
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
        # already on main — keeps the docs-repo git log clean during
        # idempotent reruns of the cron.
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=workdir_path,
            check=False,
        )
        if diff.returncode == 0:
            print("matrix JSON unchanged; skipping push", flush=True)
            return

        _run(
            ["git", "commit", "-m", commit_message_for_matrix(matrix)],
            cwd=workdir_path,
        )
        _run(["git", "push", "origin", docs_branch], cwd=workdir_path)


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
        help="Branch on the docs repo to push to.",
    )
    parser.add_argument(
        "--docs-target-path",
        default=os.environ.get("DOCS_TARGET_PATH", DOCS_TARGET_PATH_DEFAULT),
        help="Path inside the docs repo where the matrix JSON lives.",
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
        "--skip-proxy",
        action="store_true",
        help=(
            "Skip Docker/proxy/CLI/pytest steps and go straight to publish — "
            "useful when the workflow runs those steps in separate jobs."
        ),
    )
    parser.add_argument(
        "--skip-publish",
        action="store_true",
        help="Run the test pipeline but do not push to the docs repo.",
    )
    args = parser.parse_args(argv)

    docs_token = os.environ.get("DOCS_REPO_TOKEN", "")
    if not args.skip_publish and not docs_token:
        print(
            "DOCS_REPO_TOKEN is required to push to the docs repo "
            "(GitHub App installation token)",
            file=sys.stderr,
        )
        return 2

    container_id: Optional[str] = None
    litellm_version: str
    claude_code_version: str
    try:
        litellm_version = latest_stable_litellm_tag(
            token=os.environ.get("GITHUB_TOKEN")
        )
        print(f"resolved latest stable litellm: {litellm_version}", flush=True)

        if not args.skip_proxy:
            image = docker_image_for_tag(litellm_version)
            _run(["docker", "pull", image])
            container_id = _start_proxy(image, args.proxy_port)
            _wait_for_proxy(args.proxy_port)

            _run(["npm", "install", "-g", "@anthropic-ai/claude-code@latest"])
            claude_code_version = _get_claude_code_version()
            print(f"installed claude code cli: {claude_code_version}", flush=True)

            env = {
                **os.environ,
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{args.proxy_port}",
                "ANTHROPIC_AUTH_TOKEN": DEFAULT_PROXY_API_KEY,
                "COMPAT_RESULTS_PATH": str(args.results),
            }
            _run(
                [
                    "pytest",
                    "tests/claude_code/",
                    "--ignore=tests/claude_code/_driver_unit_tests",
                    "--ignore=tests/claude_code/_builder_unit_tests",
                    "--ignore=tests/claude_code/_publisher_unit_tests",
                ],
                env=env,
                check=False,
            )
        else:
            claude_code_version = os.environ.get("CLAUDE_CODE_VERSION", "")

        if args.skip_publish:
            print("skip-publish: not pushing to docs repo", flush=True)
            return 0

        publish(
            docs_repo=args.docs_repo,
            docs_branch=args.docs_branch,
            docs_target_path=args.docs_target_path,
            docs_token=docs_token,
            manifest_path=args.manifest,
            results_path=args.results,
            matrix_output_path=args.matrix_output,
            litellm_version=litellm_version,
            claude_code_version=claude_code_version,
            generated_at=_now_utc_iso(),
        )
        return 0
    finally:
        if container_id is not None:
            _stop_proxy(container_id)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DOCS_REPO_DEFAULT",
    "DOCS_TARGET_BASENAME",
    "DOCS_TARGET_PATH_DEFAULT",
    "DOCKER_IMAGE_BASE",
    "commit_message_for_matrix",
    "docker_image_for_tag",
    "select_files_to_commit",
    "publish",
    "main",
]
