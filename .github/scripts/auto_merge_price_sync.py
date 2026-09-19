"""Auto-merge the provider-info-sync bot's cost-map pull requests.

Evaluates every gate (author allowlist, cost-map-only diff, required and
non-required checks, human reviews) and merges with a merge commit when
all of them hold. Every hold reason is logged; the process exits 0 on hold
and 1 only on API or programming errors.
``DRY_RUN=1`` prints the verdict without calling the merge endpoint.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

REPO_ROOT: Final = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFY_SCRIPT: Final = os.path.join(REPO_ROOT, ".circleci", "scripts", "classify_changes.sh")
API_ROOT: Final = "https://api.github.com"
CHANGED_FILE_CEILING: Final = 3000
OK_CHECK_CONCLUSIONS: Final = frozenset({"success", "skipped", "neutral"})


@dataclass(frozen=True, slots=True)
class PullRequest:
    number: int
    title: str
    author_login: str
    state: str
    draft: bool
    mergeable: bool | None
    mergeable_state: str
    head_sha: str


@dataclass(frozen=True, slots=True)
class CheckRun:
    name: str
    status: str
    conclusion: str | None
    id: int = 0


@dataclass(frozen=True, slots=True)
class CommitStatus:
    context: str
    state: str


@dataclass(frozen=True, slots=True)
class Review:
    author_login: str
    state: str
    body: str
    commit_id: str
    submitted_at: datetime


@dataclass(frozen=True, slots=True)
class Verdict:
    merge: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationInputs:
    pr: PullRequest
    changed_files: tuple[str, ...]
    required_contexts: frozenset[str]
    check_runs: tuple[CheckRun, ...]
    statuses: tuple[CommitStatus, ...]
    reviews: tuple[Review, ...]
    self_check_name: str
    author_allowlist: frozenset[str]


def _is_bot_login(login: str) -> bool:
    return login.lower().endswith("[bot]")


def latest_check_runs(check_runs: Sequence[CheckRun]) -> tuple[CheckRun, ...]:
    latest_by_name: Final[dict[str, CheckRun]] = {}
    for run in sorted(check_runs, key=lambda run: run.id):
        latest_by_name[run.name] = run
    return tuple(latest_by_name.values())


def _classify(changed_files: Sequence[str]) -> str:
    result: Final = subprocess.run(
        ["bash", CLASSIFY_SCRIPT, "cost-map-only"],
        input="\n".join(changed_files),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "error"
    return result.stdout.strip()


def evaluate(
    inputs: EvaluationInputs,
    *,
    classify: Callable[[Sequence[str]], str] = _classify,
) -> Verdict:
    pr: Final = inputs.pr
    reasons: list[str] = []

    if pr.author_login.lower() not in {login.lower() for login in inputs.author_allowlist}:
        reasons.append(f"author {pr.author_login!r} not in allowlist")
    if pr.state != "open":
        reasons.append("pr not open")
    if pr.draft:
        reasons.append("pr is a draft")
    if pr.mergeable is None:
        reasons.append("mergeability unknown")
    elif not pr.mergeable:
        reasons.append("pr not mergeable")
    if pr.mergeable_state == "dirty":
        reasons.append("pr has merge conflicts")

    if len(inputs.changed_files) > CHANGED_FILE_CEILING:
        reasons.append(f"changed file count {len(inputs.changed_files)} over {CHANGED_FILE_CEILING} ceiling")
    else:
        decision: Final = classify(inputs.changed_files)
        if decision != "run":
            reasons.append("changed files outside the cost-map-only set")

    check_runs: Final = latest_check_runs(inputs.check_runs)
    green_runs: Final = frozenset(run.name for run in check_runs if run.conclusion in OK_CHECK_CONCLUSIONS)
    green_statuses: Final = frozenset(status.context for status in inputs.statuses if status.state == "success")
    for context in sorted(inputs.required_contexts):
        if context not in green_runs and context not in green_statuses:
            reasons.append(f"required check {context!r} not green")
    for run in check_runs:
        if run.name == inputs.self_check_name:
            continue
        if run.status != "completed" or run.conclusion not in OK_CHECK_CONCLUSIONS:
            reasons.append(f"check run {run.name!r} is {run.status}/{run.conclusion}")
    for status in inputs.statuses:
        if status.state != "success":
            reasons.append(f"commit status {status.context!r} is {status.state}")

    latest_state_by_reviewer: Final[dict[str, str]] = {}
    for review in sorted(inputs.reviews, key=lambda review: review.submitted_at):
        if _is_bot_login(review.author_login):
            continue
        latest_state_by_reviewer[review.author_login] = review.state
    for reviewer, state in latest_state_by_reviewer.items():
        if state == "CHANGES_REQUESTED":
            reasons.append(f"changes requested by {reviewer}")

    return Verdict(merge=not reasons, reasons=tuple(reasons))


def _request(token: str, method: str, path: str, body: Mapping[str, object] | None = None) -> object:
    url: Final = path if path.startswith("http") else f"{API_ROOT}{path}"
    data: Final = None if body is None else json.dumps(body).encode("utf-8")
    request: Final = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_allow_fail(
    token: str, method: str, path: str, body: Mapping[str, object] | None = None
) -> tuple[int, object | None]:
    url: Final = path if path.startswith("http") else f"{API_ROOT}{path}"
    data: Final = None if body is None else json.dumps(body).encode("utf-8")
    request: Final = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _items(payload: object, key: str | None = None) -> tuple[object, ...]:
    source: Final = payload.get(key) if key and isinstance(payload, Mapping) else payload
    if not isinstance(source, list):
        return ()
    return tuple(source)


def _paginate(token: str, path: str, key: str | None = None) -> list[object]:
    separator: Final = "&" if "?" in path else "?"
    results: list[object] = []
    for page in range(1, 10_000):
        batch: Final = _items(_request(token, "GET", f"{path}{separator}per_page=100&page={page}"), key)
        results.extend(batch)
        if len(batch) < 100:
            return results
    return results


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _int(value: object) -> int:
    return value if isinstance(value, int) else 0


def _bool(value: object) -> bool:
    return value is True


def _nested(value: object, *keys: str) -> object:
    current: object = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _parse_time(value: object) -> datetime:
    text: Final = _text(value)
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _load_pr(token: str, repo: str, number: int) -> PullRequest:
    data: Final = _request(token, "GET", f"/repos/{repo}/pulls/{number}")
    if not isinstance(data, Mapping):
        raise RuntimeError(f"unexpected pull payload for #{number}")
    return PullRequest(
        number=number,
        title=_text(data.get("title")),
        author_login=_text(_nested(data, "user", "login")),
        state=_text(data.get("state")),
        draft=_bool(data.get("draft")),
        mergeable=data.get("mergeable") if isinstance(data.get("mergeable"), bool) else None,
        mergeable_state=_text(data.get("mergeable_state")),
        head_sha=_text(_nested(data, "head", "sha")),
    )


def _list_candidate_prs(token: str, repo: str, base: str, allowlist: frozenset[str]) -> list[int]:
    candidates: Final = _paginate(token, f"/repos/{repo}/pulls?state=open&base={base}")
    return [
        _int(item.get("number"))
        for item in candidates
        if isinstance(item, Mapping) and _text(_nested(item, "user", "login")).lower() in allowlist
    ]


def _changed_files(token: str, repo: str, number: int) -> tuple[str, ...]:
    files: Final = _paginate(token, f"/repos/{repo}/pulls/{number}/files")
    return tuple(_text(item.get("filename")) for item in files if isinstance(item, Mapping))


def _required_contexts(token: str, repo: str, base: str) -> frozenset[str]:
    payload: Final = _request(token, "GET", f"/repos/{repo}/rules/branches/{base}")
    contexts: set[str] = set()
    for rule in _items(payload):
        if not isinstance(rule, Mapping) or rule.get("type") != "required_status_checks":
            continue
        checks: Final = _nested(rule, "parameters", "required_status_checks")
        for check in _items(checks):
            if isinstance(check, Mapping):
                context: Final = _text(check.get("context"))
                if context:
                    contexts.add(context)
    return frozenset(contexts)


def _check_runs(token: str, repo: str, sha: str) -> tuple[CheckRun, ...]:
    runs: Final = _paginate(token, f"/repos/{repo}/commits/{sha}/check-runs", key="check_runs")
    return tuple(
        CheckRun(
            name=_text(item.get("name")),
            status=_text(item.get("status")),
            conclusion=item.get("conclusion") if isinstance(item.get("conclusion"), str) else None,
            id=item.get("id") if isinstance(item.get("id"), int) else 0,
        )
        for item in runs
        if isinstance(item, Mapping)
    )


def _statuses(token: str, repo: str, sha: str) -> tuple[CommitStatus, ...]:
    payload: Final = _request(token, "GET", f"/repos/{repo}/commits/{sha}/status")
    return tuple(
        CommitStatus(context=_text(item.get("context")), state=_text(item.get("state")))
        for item in _items(payload, "statuses")
        if isinstance(item, Mapping)
    )


def _reviews(token: str, repo: str, number: int) -> tuple[Review, ...]:
    reviews: Final = _paginate(token, f"/repos/{repo}/pulls/{number}/reviews")
    return tuple(
        Review(
            author_login=_text(_nested(item, "user", "login")),
            state=_text(item.get("state")),
            body=_text(item.get("body")),
            commit_id=_text(item.get("commit_id")),
            submitted_at=_parse_time(item.get("submitted_at")),
        )
        for item in reviews
        if isinstance(item, Mapping)
    )


def _mergeable_or_refetch(token: str, repo: str, pr: PullRequest) -> PullRequest:
    if pr.mergeable is not None:
        return pr
    time.sleep(5)
    return _load_pr(token, repo, pr.number)


def _gather_inputs(
    token: str,
    repo: str,
    number: int,
    base: str,
    self_check_name: str,
    allowlist: frozenset[str],
) -> EvaluationInputs:
    pr: Final = _mergeable_or_refetch(token, repo, _load_pr(token, repo, number))
    return EvaluationInputs(
        pr=pr,
        changed_files=_changed_files(token, repo, number),
        required_contexts=_required_contexts(token, repo, base),
        check_runs=_check_runs(token, repo, pr.head_sha),
        statuses=_statuses(token, repo, pr.head_sha),
        reviews=_reviews(token, repo, number),
        self_check_name=self_check_name,
        author_allowlist=allowlist,
    )


def merge_request_body(pr: PullRequest) -> dict[str, str]:
    return {"merge_method": "merge", "commit_title": f"{pr.title} (#{pr.number})", "sha": pr.head_sha}


def _merge(token: str, repo: str, pr: PullRequest) -> None:
    status, _ = _request_allow_fail(token, "PUT", f"/repos/{repo}/pulls/{pr.number}/merge", merge_request_body(pr))
    if status in (200, 405, 409):
        print(f"auto-merge-price-sync: PR #{pr.number} merge call returned {status}")
        return
    raise RuntimeError(f"merge call for PR #{pr.number} returned {status}")


def main() -> int:
    token: Final = os.environ.get("GH_TOKEN", "")
    repo: Final = os.environ.get("REPO", "")
    base: Final = os.environ.get("BASE_BRANCH", "main")
    dry_run: Final = os.environ.get("DRY_RUN", "") != ""
    self_check_name: Final = os.environ.get("SELF_CHECK_NAME", "auto-merge-price-sync")
    allowlist: Final = frozenset(login.lower() for login in os.environ.get("PR_AUTHOR_ALLOWLIST", "").split() if login)
    if not token:
        print("auto-merge-price-sync: app credentials not configured")
        return 0
    if not repo:
        print("auto-merge-price-sync: REPO not set", file=sys.stderr)
        return 1

    pr_number_env: Final = os.environ.get("PR_NUMBER", "")
    candidates: Final = [int(pr_number_env)] if pr_number_env else _list_candidate_prs(token, repo, base, allowlist)
    for number in candidates:
        inputs: Final = _gather_inputs(token, repo, number, base, self_check_name, allowlist)
        verdict: Final = evaluate(inputs)
        for reason in verdict.reasons:
            print(f"auto-merge-price-sync: PR #{number} hold: {reason}")
        if not verdict.merge:
            continue
        print(f"auto-merge-price-sync: PR #{number} all gates green")
        if dry_run:
            print(f"auto-merge-price-sync: DRY_RUN merge suppressed for PR #{number}")
            continue
        _merge(token, repo, inputs.pr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
