import base64
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

from sdk_watch.github import GitHub
from sdk_watch.report import DemoVerdict, PrDraft, pr_body

ALLOWED_FILES: Final = frozenset({"model_prices_and_context_window.json"})
ALLOWED_PREFIXES: Final = ("litellm/", "tests/")
DENIED_PREFIXES: Final = ("litellm/scripts/",)
BOT_NAME: Final = "github-actions[bot]"
BOT_EMAIL: Final = "41898282+github-actions[bot]@users.noreply.github.com"


@dataclass(frozen=True, slots=True)
class Opened:
    url: str


@dataclass(frozen=True, slots=True)
class Skipped:
    reason: str


ShipResult = Opened | Skipped


def is_allowed(path: str) -> bool:
    if path.startswith(DENIED_PREFIXES) or ".." in Path(path).parts:
        return False
    return path in ALLOWED_FILES or path.startswith(ALLOWED_PREFIXES)


def asset_path(draft: PrDraft) -> str:
    return f".github/pr-assets/sdk-watch/{draft.branch}.gif"


def staged_paths(repo: Path) -> tuple[str, ...]:
    return tuple(filter(None, _git(repo, "diff", "--cached", "--name-only", "--no-renames", "-z").split("\0")))


def ship(
    repo: Path,
    repository: str,
    base_sha: str,
    patch: Path,
    draft: PrDraft,
    verdict: DemoVerdict,
    gif: Path,
    github: GitHub,
    token: str,
    run_url: str,
) -> ShipResult:
    if not verdict.passes_after:
        return Skipped(reason="the PR's tests fail with the fix applied")
    if github.branch_exists(draft.branch):
        return Skipped(reason=f"branch {draft.branch} already exists")
    _git(repo, "switch", "--quiet", "--detach", base_sha)
    _git(repo, "apply", "--index", str(patch))
    refused: Final = tuple(path for path in staged_paths(repo) if not is_allowed(path))
    if refused:
        _git(repo, "reset", "--quiet", "--hard")
        return Skipped(reason=f"the patch touches paths the agent may not change: {', '.join(refused)}")
    _git(repo, "switch", "--quiet", "--create", draft.branch)
    _commit(repo, draft.title)
    target: Final = repo / asset_path(draft)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gif, target)
    _git(repo, "add", asset_path(draft))
    _commit(repo, "chore: attach the demo recording for review")
    recording_sha: Final = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "rm", "--quiet", asset_path(draft))
    _commit(repo, "chore: keep the demo recording out of the merged tree")
    _push(repo, draft.branch, token)
    gif_url: Final = f"https://github.com/{repository}/blob/{recording_sha}/{asset_path(draft)}?raw=true"
    return Opened(url=github.create_pr(draft.title, draft.branch, "main", pr_body(draft, verdict, gif_url, run_url)))


def ping(webhook_url: str, draft: PrDraft, opened: Opened) -> None:
    text: Final = (
        f"LiteLLM PR ready for review: <{opened.url}|{draft.title}>\n"
        f"{draft.sdk} {draft.version}: {draft.litellm_change}\n"
        "The PR has a video of its tests failing on main and passing with the fix, and a walkthrough"
    )
    httpx.post(webhook_url, json={"text": text}, timeout=30).raise_for_status()


def _push(repo: Path, branch: str, token: str) -> None:
    basic: Final = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    auth: Final = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraheader",
        "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
    }
    subprocess.run(
        ("git", "-C", str(repo), "push", "--quiet", "origin", f"HEAD:refs/heads/{branch}"),
        env={**os.environ, **auth},
        check=True,
    )


def _commit(repo: Path, message: str) -> None:
    _git(repo, "-c", f"user.name={BOT_NAME}", "-c", f"user.email={BOT_EMAIL}", "commit", "--quiet", "-m", message)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(repo), *args), stdout=subprocess.PIPE, text=True, check=True).stdout
