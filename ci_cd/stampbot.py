"""litellm-stampbot: approve a pull request when it is low risk by rule and by judgement.

A LiteLLM Engineering team member comments `@litellm-stampbot` on a pull request. The bot computes a floor
tier from `.github/stampbot/risk_rules.yml` (paths touched, diff size, test delta, check and review state),
asks a judge model to raise it when the rules alone leave it low, posts one comment it edits in place, sets a
`risk:<tier>` label, and approves only when the final tier is low. Every later push re-runs the same evaluation
and dismisses the bot's approval when the verdict moves off low. The judge can raise the tier and never lower it.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

API: Final = "https://api.github.com"
MARKER: Final = "<!-- litellm-stampbot -->"
MENTION: Final = "@litellm-stampbot"
LABEL_PREFIX: Final = "risk:"
LABEL_COLORS: Final[Mapping[str, str]] = {"low": "0e8a16", "medium": "fbca04", "high": "b60205"}
ALLOWED_ASSOCIATIONS: Final = frozenset(("MEMBER", "OWNER"))
DIFF_CAP_CHARS: Final = 60_000
BODY_CAP_CHARS: Final = 8_000
MAX_FILES: Final = 3_000


class Tier(IntEnum):
    low = 0
    medium = 1
    high = 2


@dataclass(frozen=True, slots=True)
class Finding:
    tier: Tier
    reason: str


class SizeCap(BaseModel):
    model_config = ConfigDict(frozen=True)
    lines: int
    files: int


class SizeRules(BaseModel):
    model_config = ConfigDict(frozen=True)
    low: SizeCap
    judge_max: SizeCap


class TestWeakening(BaseModel):
    model_config = ConfigDict(frozen=True)
    added: tuple[str, ...]
    removed: tuple[str, ...]


class Rules(BaseModel):
    model_config = ConfigDict(frozen=True)
    always_human: tuple[str, ...]
    low_by_rule: tuple[str, ...]
    test_paths: tuple[str, ...]
    code_paths: tuple[str, ...]
    test_weakening: TestWeakening
    size: SizeRules


class Repo(BaseModel):
    model_config = ConfigDict(frozen=True)
    full_name: str


class Ref(BaseModel):
    model_config = ConfigDict(frozen=True)
    sha: str
    ref: str
    repo: Repo | None = None


class User(BaseModel):
    model_config = ConfigDict(frozen=True)
    login: str
    type: str = "User"


class PullRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    number: int
    title: str
    body: str | None = None
    draft: bool = False
    state: str
    user: User
    head: Ref
    base: Ref
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0


class ChangedFile(BaseModel):
    model_config = ConfigDict(frozen=True)
    filename: str
    status: str
    additions: int = 0
    deletions: int = 0
    patch: str | None = None


class CheckRun(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str
    status: str
    conclusion: str | None = None


class CheckRunsPage(BaseModel):
    model_config = ConfigDict(frozen=True)
    check_runs: tuple[CheckRun, ...]


class CommitStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    context: str
    state: str


class CombinedStatus(BaseModel):
    model_config = ConfigDict(frozen=True)
    statuses: tuple[CommitStatus, ...]


class Review(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    user: User
    state: str
    body: str | None = None
    commit_id: str | None = None


class Comment(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: int
    body: str | None = None


class ReviewThread(BaseModel):
    model_config = ConfigDict(frozen=True)
    isResolved: bool
    isOutdated: bool = False


class ReviewThreadsPage(BaseModel):
    model_config = ConfigDict(frozen=True)
    nodes: tuple[ReviewThread, ...]


class ReviewThreadsPR(BaseModel):
    model_config = ConfigDict(frozen=True)
    reviewThreads: ReviewThreadsPage


class ReviewThreadsRepo(BaseModel):
    model_config = ConfigDict(frozen=True)
    pullRequest: ReviewThreadsPR


class ReviewThreadsData(BaseModel):
    model_config = ConfigDict(frozen=True)
    repository: ReviewThreadsRepo


class ReviewThreadsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    data: ReviewThreadsData


class TeamMembership(BaseModel):
    model_config = ConfigDict(frozen=True)
    state: str


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tier: Literal["low", "medium", "high"]
    change_kind: Literal[
        "additive", "scoped_behavior_change", "default_behavior_change", "public_api_change", "price_change"
    ]
    blast_radius: Literal["one_provider_or_page", "shared_helper", "request_hot_path", "boot_path"]
    reasons: str = Field(min_length=1)
    look_at: str = ""

    def reason_lines(self) -> tuple[str, ...]:
        return _lines(self.reasons)

    def look_at_lines(self) -> tuple[str, ...]:
        return _lines(self.look_at)


def _lines(text: str) -> tuple[str, ...]:
    return tuple(line.strip(" -*") for line in text.splitlines() if line.strip(" -*"))


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str | None = None


class ChatChoice(BaseModel):
    model_config = ConfigDict(frozen=True)
    message: ChatMessage


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    choices: tuple[ChatChoice, ...]


FILES_ADAPTER: Final = TypeAdapter(tuple[ChangedFile, ...])
REVIEWS_ADAPTER: Final = TypeAdapter(tuple[Review, ...])
COMMENTS_ADAPTER: Final = TypeAdapter(tuple[Comment, ...])
SCHEMA_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class Judgement:
    model: str
    verdict: Verdict | str


@dataclass(frozen=True, slots=True)
class Evaluation:
    pr: PullRequest
    findings: tuple[Finding, ...]
    judgement: Judgement | None

    @property
    def floor(self) -> Tier:
        return max((finding.tier for finding in self.findings), default=Tier.low)

    @property
    def final(self) -> Tier:
        if self.judgement is None or not isinstance(self.judgement.verdict, Verdict):
            return self.floor if self.judgement is None else Tier.high
        return max(self.floor, Tier[self.judgement.verdict.tier])


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    pieces: Final = re.split(r"(\*\*/|\*\*|\*|\?)", pattern)
    translated: Final = "".join(
        {"**/": "(?:.*/)?", "**": ".*", "*": "[^/]*", "?": "[^/]"}.get(piece, re.escape(piece)) for piece in pieces
    )
    return re.compile(f"^{translated}$")


def matches_any(path: str, patterns: Sequence[str]) -> bool:
    return any(glob_to_regex(pattern).match(path) for pattern in patterns)


def load_rules(path: Path) -> Rules:
    return Rules.model_validate(yaml.safe_load(path.read_text()))


def _patch_lines(patch: str | None, sign: str) -> tuple[str, ...]:
    if patch is None:
        return ()
    return tuple(line[1:] for line in patch.splitlines() if line.startswith(sign) and not line.startswith(sign * 3))


def _weakened_tests(files: Sequence[ChangedFile], rules: Rules) -> tuple[Finding, ...]:
    test_files: Final = tuple(file for file in files if matches_any(file.filename, rules.test_paths))
    removed_files: Final = tuple(f for f in test_files if f.status == "removed")
    added_files: Final = tuple(f for f in test_files if f.status == "added")
    deleted: Final = (
        tuple(Finding(Tier.high, f"deletes test file `{f.filename}`") for f in removed_files)
        if len(removed_files) > len(added_files)
        else ()
    )
    added_markers: Final = tuple(
        Finding(Tier.high, f"adds `{marker.strip()}` in `{f.filename}`")
        for f in test_files
        for marker in rules.test_weakening.added
        if any(marker in line for line in _patch_lines(f.patch, "+"))
    )
    removed_markers: Final = tuple(
        Finding(Tier.high, f"removes `{marker.strip()}` lines in `{f.filename}`")
        for f in test_files
        for marker in rules.test_weakening.removed
        if sum(marker in line for line in _patch_lines(f.patch, "-"))
        > sum(marker in line for line in _patch_lines(f.patch, "+"))
    )
    return deleted + added_markers + removed_markers


def _check_findings(checks: Sequence[CheckRun], statuses: Sequence[CommitStatus]) -> tuple[Finding, ...]:
    red_checks: Final = tuple(
        Finding(Tier.high, f"check `{c.name}` finished {c.conclusion}")
        for c in checks
        if c.status == "completed" and c.conclusion in ("failure", "timed_out", "action_required")
    )
    red_statuses: Final = tuple(
        Finding(Tier.high, f"status `{s.context}` is {s.state}") for s in statuses if s.state in ("failure", "error")
    )
    return red_checks + red_statuses


def latest_review_states(reviews: Sequence[Review]) -> Mapping[str, str]:
    decisive: Final = tuple(r for r in reviews if r.state in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED"))
    return {review.user.login: review.state for review in decisive}


def compute_findings(
    pr: PullRequest,
    repo: str,
    files: Sequence[ChangedFile],
    checks: Sequence[CheckRun],
    statuses: Sequence[CommitStatus],
    reviews: Sequence[Review],
    unresolved_threads: int,
    rules: Rules,
) -> tuple[Finding, ...]:
    head_repo: Final = pr.head.repo.full_name if pr.head.repo is not None else None
    lines, source_files = source_size(files, rules)
    paths: Final = tuple(file.filename for file in files)
    always_human: Final = tuple(
        Finding(Tier.high, f"`{path}` is on the always-human list")
        for path in paths
        if matches_any(path, rules.always_human)
    )
    code_changed: Final = any(matches_any(path, rules.code_paths) for path in paths)
    tests_touched: Final = any(matches_any(path, rules.test_paths) for path in paths)
    changes_requested: Final = tuple(
        Finding(Tier.high, f"@{login} requested changes")
        for login, state in latest_review_states(reviews).items()
        if state == "CHANGES_REQUESTED"
    )
    structural: Final = (
        ((Finding(Tier.high, f"head branch lives on fork `{head_repo}`"),) if head_repo != repo else ())
        + ((Finding(Tier.high, "pull request is a draft"),) if pr.draft else ())
        + ((Finding(Tier.high, f"pull request is {pr.state}"),) if pr.state != "open" else ())
        + (
            (
                Finding(
                    Tier.high,
                    f"{lines} source lines in {source_files} source files is over the {rules.size.judge_max.lines} line / {rules.size.judge_max.files} cap (tests and docs are not counted)",
                ),
            )
            if lines > rules.size.judge_max.lines or source_files > rules.size.judge_max.files
            else ()
        )
        + ((Finding(Tier.high, f"{unresolved_threads} unresolved review thread(s)"),) if unresolved_threads else ())
        + (
            (Finding(Tier.medium, "code changed with no test file in the pull request"),)
            if code_changed and not tests_touched
            else ()
        )
    )
    return (
        structural
        + always_human
        + changes_requested
        + _weakened_tests(files, rules)
        + _check_findings(checks, statuses)
    )


def source_size(files: Sequence[ChangedFile], rules: Rules) -> tuple[int, int]:
    source: Final = tuple(
        f
        for f in files
        if not matches_any(f.filename, rules.test_paths) and not matches_any(f.filename, rules.low_by_rule)
    )
    return sum(f.additions + f.deletions for f in source), len(source)


def low_by_rule(files: Sequence[ChangedFile], rules: Rules) -> bool:
    return bool(files) and all(matches_any(file.filename, rules.low_by_rule) for file in files)


def _clip(text: str, cap: int, what: str) -> str:
    return text if len(text) <= cap else f"{text[:cap]}\n\n[{what} truncated at {cap} characters]"


def judge_user_message(pr: PullRequest, files: Sequence[ChangedFile], rules: Rules) -> str:
    lines, source_files = source_size(files, rules)
    small: Final = lines <= rules.size.low.lines and source_files <= rules.size.low.files
    file_lines: Final = "\n".join(f"- {f.filename} ({f.status}, +{f.additions} -{f.deletions})" for f in files)
    patches: Final = "\n\n".join(f"--- {f.filename}\n{f.patch}" for f in files if f.patch is not None)
    return "\n".join(
        (
            f"Title: {pr.title}",
            f"Author: @{pr.user.login}",
            f"Size: {lines} source lines in {source_files} source files, {len(files)} files total"
            + (" (small by rule)" if small else ""),
            "",
            "## Files",
            file_lines,
            "",
            "## Body",
            _clip(pr.body or "(empty)", BODY_CAP_CHARS, "body"),
            "",
            "## Diff",
            _clip(patches, DIFF_CAP_CHARS, "diff"),
        )
    )


@dataclass(frozen=True, slots=True)
class JudgeClient:
    api_base: str
    api_key: str
    model: str
    prompt: str
    schema: Mapping[str, object]
    http: httpx.Client

    def judge(self, user_message: str) -> Judgement:
        request: Final = {
            "model": self.model,
            "messages": ({"role": "system", "content": self.prompt}, {"role": "user", "content": user_message}),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "stampbot_verdict", "strict": True, "schema": self.schema},
            },
        }
        response: Final = self.http.post(
            f"{self.api_base.rstrip('/')}/v1/chat/completions",
            json=request,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=180,
        )
        if response.status_code != 200:
            return Judgement(self.model, f"judge call failed with HTTP {response.status_code}")
        return Judgement(self.model, parse_verdict(response.text))


def parse_verdict(raw: str) -> Verdict | str:
    try:
        content: Final = ChatResponse.model_validate_json(raw).choices[0].message.content
    except (ValidationError, IndexError):
        return "judge response was not a chat completion"
    if content is None:
        return "judge returned no content"
    try:
        return Verdict.model_validate_json(content)
    except ValidationError as error:
        return f"judge verdict did not match the schema: {error.error_count()} error(s)"


class DecisionChoiceAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: str | None = None
    choice: str | bool
    confidence: float = 0.0


class DecisionsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    answers: tuple[DecisionChoiceAnswer, ...]


DECISION_QUESTIONS: Final[Mapping[str, tuple[str, tuple[str, ...]]]] = MappingProxyType(
    {
        "tier": (
            "How much human review does this pull request need before it merges? low means an additive or confined "
            "change a bot may approve, medium means a human should read it, high means a human must read it closely "
            "(request hot path, boot path, default behavior change, public API change, existing price change, money, "
            "auth or data retention).",
            ("low", "medium", "high"),
        ),
        "change_kind": (
            "What kind of change is this?",
            ("additive", "scoped_behavior_change", "default_behavior_change", "public_api_change", "price_change"),
        ),
        "blast_radius": (
            "How far does this change reach?",
            ("one_provider_or_page", "shared_helper", "request_hot_path", "boot_path"),
        ),
    }
)


def decisions_request(model: str, prompt: str, user_message: str) -> Mapping[str, object]:
    questions: Final = tuple(
        {
            "type": "choice",
            "name": name,
            "instructions": instructions,
            "choices": tuple({"value": value} for value in values),
        }
        for name, (instructions, values) in DECISION_QUESTIONS.items()
    )
    return {"model": model, "input": f"{prompt}\n\n{user_message}", "questions": questions}


def parse_decisions_verdict(raw: str) -> Verdict | str:
    try:
        answers: Final = DecisionsResponse.model_validate_json(raw).answers
    except ValidationError:
        return "judge response was not a decisions response with choice answers"
    picked: Final = {answer.name: answer for answer in answers if answer.name is not None}
    if set(picked) != set(DECISION_QUESTIONS):
        return f"judge answered {sorted(picked)} instead of {sorted(DECISION_QUESTIONS)}"
    reasons: Final = "\n".join(f"{name}: {a.choice} (confidence {a.confidence:.2f})" for name, a in picked.items())
    try:
        return Verdict.model_validate(
            {
                "tier": picked["tier"].choice,
                "change_kind": picked["change_kind"].choice,
                "blast_radius": picked["blast_radius"].choice,
                "reasons": reasons,
                "look_at": "",
            }
        )
    except ValidationError as error:
        return f"judge verdict did not match the schema: {error.error_count()} error(s)"


@dataclass(frozen=True, slots=True)
class DecisionsJudgeClient:
    api_base: str
    api_key: str
    model: str
    prompt: str
    http: httpx.Client

    def judge(self, user_message: str) -> Judgement:
        response: Final = self.http.post(
            f"{self.api_base.rstrip('/')}/v1/decisions",
            json=decisions_request(self.model, self.prompt, user_message),
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=180,
        )
        if response.status_code != 200:
            return Judgement(self.model, f"judge call failed with HTTP {response.status_code}")
        return Judgement(self.model, parse_decisions_verdict(response.text))


@dataclass(frozen=True, slots=True)
class GitHub:
    repo: str
    token: str
    http: httpx.Client

    def _headers(self) -> Mapping[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _get(self, path: str, params: Mapping[str, str] | None = None) -> httpx.Response:
        return self.http.get(f"{API}{path}", headers=self._headers(), params=params, timeout=60)

    def _send(self, method: str, path: str, body: Mapping[str, object]) -> httpx.Response:
        return self.http.request(method, f"{API}{path}", headers=self._headers(), json=body, timeout=60)

    def _get_ok(self, path: str, params: Mapping[str, str] | None = None) -> str:
        response: Final = self._get(path, params)
        response.raise_for_status()
        return response.text

    def pull_request(self, number: int) -> PullRequest:
        return PullRequest.model_validate_json(self._get_ok(f"/repos/{self.repo}/pulls/{number}"))

    def _files_page(self, number: int, page: int) -> tuple[ChangedFile, ...]:
        text: Final = self._get_ok(f"/repos/{self.repo}/pulls/{number}/files", {"per_page": "100", "page": str(page)})
        return FILES_ADAPTER.validate_json(text)

    def files(self, number: int, page: int = 1) -> tuple[ChangedFile, ...]:
        batch: Final = self._files_page(number, page)
        if len(batch) < 100 or page * 100 >= MAX_FILES:
            return batch
        return batch + self.files(number, page + 1)

    def check_runs(self, sha: str) -> tuple[CheckRun, ...]:
        text: Final = self._get_ok(f"/repos/{self.repo}/commits/{sha}/check-runs", {"per_page": "100"})
        return CheckRunsPage.model_validate_json(text).check_runs

    def statuses(self, sha: str) -> tuple[CommitStatus, ...]:
        return CombinedStatus.model_validate_json(self._get_ok(f"/repos/{self.repo}/commits/{sha}/status")).statuses

    def reviews(self, number: int) -> tuple[Review, ...]:
        text: Final = self._get_ok(f"/repos/{self.repo}/pulls/{number}/reviews", {"per_page": "100"})
        return REVIEWS_ADAPTER.validate_json(text)

    def comments(self, number: int) -> tuple[Comment, ...]:
        text: Final = self._get_ok(f"/repos/{self.repo}/issues/{number}/comments", {"per_page": "100"})
        return COMMENTS_ADAPTER.validate_json(text)

    def unresolved_threads(self, number: int) -> int:
        owner, name = self.repo.split("/", 1)
        query: Final = (
            "query($owner:String!,$name:String!,$number:Int!){repository(owner:$owner,name:$name)"
            "{pullRequest(number:$number){reviewThreads(first:100){nodes{isResolved isOutdated}}}}}"
        )
        response: Final = self._send(
            "POST", "/graphql", {"query": query, "variables": {"owner": owner, "name": name, "number": number}}
        )
        response.raise_for_status()
        threads: Final = ReviewThreadsResponse.model_validate_json(
            response.text
        ).data.repository.pullRequest.reviewThreads
        return sum(not thread.isResolved and not thread.isOutdated for thread in threads.nodes)

    def is_team_member(self, org: str, team_slug: str, login: str) -> bool | str:
        response: Final = self._get(f"/orgs/{org}/teams/{team_slug}/memberships/{login}")
        if response.status_code == 404:
            return False
        if response.status_code != 200:
            return f"team membership lookup failed with HTTP {response.status_code}"
        return TeamMembership.model_validate_json(response.text).state == "active"

    def upsert_comment(self, number: int, body: str) -> None:
        existing: Final = tuple(c for c in self.comments(number) if c.body is not None and MARKER in c.body)
        if existing:
            self._send(
                "PATCH", f"/repos/{self.repo}/issues/comments/{existing[0].id}", {"body": body}
            ).raise_for_status()
            return
        self._send("POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body}).raise_for_status()

    def has_marker_comment(self, number: int) -> bool:
        return any(c.body is not None and MARKER in c.body for c in self.comments(number))

    def set_risk_label(self, number: int, tier: Tier) -> None:
        wanted: Final = f"{LABEL_PREFIX}{tier.name}"
        created: Final = self._send(
            "POST",
            f"/repos/{self.repo}/labels",
            {"name": wanted, "color": LABEL_COLORS[tier.name], "description": "litellm-stampbot risk tier"},
        )
        if created.status_code not in (201, 422):
            created.raise_for_status()
        for other in (name for name in Tier.__members__ if name != tier.name):
            self.http.delete(
                f"{API}/repos/{self.repo}/issues/{number}/labels/{LABEL_PREFIX}{other}",
                headers=self._headers(),
                timeout=60,
            )
        self._send("POST", f"/repos/{self.repo}/issues/{number}/labels", {"labels": [wanted]}).raise_for_status()

    def bot_approvals(self, number: int) -> tuple[Review, ...]:
        return tuple(
            r for r in self.reviews(number) if r.state == "APPROVED" and r.body is not None and MARKER in r.body
        )

    def approve(self, number: int, sha: str, body: str) -> None:
        self._send(
            "POST", f"/repos/{self.repo}/pulls/{number}/reviews", {"event": "APPROVE", "commit_id": sha, "body": body}
        ).raise_for_status()

    def dismiss(self, number: int, review: Review, message: str) -> None:
        self._send(
            "PUT", f"/repos/{self.repo}/pulls/{number}/reviews/{review.id}/dismissals", {"message": message}
        ).raise_for_status()


def evaluate(github: GitHub, number: int, rules: Rules, judge: JudgeClient | DecisionsJudgeClient | None) -> Evaluation:
    pr: Final = github.pull_request(number)
    files: Final = github.files(number)
    findings: Final = compute_findings(
        pr,
        github.repo,
        files,
        github.check_runs(pr.head.sha),
        github.statuses(pr.head.sha),
        github.reviews(number),
        github.unresolved_threads(number),
        rules,
    )
    floor: Final = max((finding.tier for finding in findings), default=Tier.low)
    if floor == Tier.high or low_by_rule(files, rules):
        return Evaluation(pr, findings, None)
    if judge is None:
        return Evaluation(
            pr,
            findings,
            Judgement("(none)", "no judge configured, set LITELLM_API_BASE, LITELLM_API_KEY and STAMPBOT_JUDGE_MODEL"),
        )
    return Evaluation(pr, findings, judge.judge(judge_user_message(pr, files, rules)))


@dataclass(frozen=True, slots=True)
class Outcome:
    kind: Literal["approved", "would_approve", "needs_human"]
    note: str


def decide(evaluation: Evaluation, approvals_enabled: bool, app_token: bool) -> Outcome:
    if evaluation.final != Tier.low:
        return Outcome("needs_human", f"risk is {evaluation.final.name}, a human reviewer is needed")
    if not app_token:
        return Outcome("would_approve", "shadow mode, no GitHub App credentials, so this run only comments and labels")
    if not approvals_enabled:
        return Outcome(
            "would_approve", "shadow mode, STAMPBOT_APPROVALS_ENABLED is not true, so this run only comments and labels"
        )
    return Outcome("approved", "approved by litellm-stampbot")


def render_comment(evaluation: Evaluation, outcome: Outcome, rules_path: str) -> str:
    sha: Final = evaluation.pr.head.sha[:7]
    findings: Final = (
        "\n".join(f"- {f.tier.name}: {f.reason}" for f in evaluation.findings) or "- low: no rule raised the tier"
    )
    judge_block: Final = _render_judgement(evaluation.judgement)
    headline: Final = {
        "approved": f"Approved at `{sha}`.",
        "would_approve": f"Would approve at `{sha}`. {outcome.note}.",
        "needs_human": f"Not approved at `{sha}`. {outcome.note}.",
    }[outcome.kind]
    return "\n".join(
        (
            MARKER,
            f"### litellm-stampbot: risk {evaluation.final.name}",
            "",
            headline,
            "",
            f"Rules (`{rules_path}`), floor {evaluation.floor.name}:",
            findings,
            "",
            judge_block,
            "",
            f"A new push re-runs this check. Comment `{MENTION}` to run it again by hand.",
        )
    )


def _render_judgement(judgement: Judgement | None) -> str:
    if judgement is None:
        return "Judge: skipped, the rules alone decided."
    if not isinstance(judgement.verdict, Verdict):
        return f"Judge ({judgement.model}): {judgement.verdict}. Treated as high."
    verdict: Final = judgement.verdict
    reasons: Final = "\n".join(f"- {reason}" for reason in verdict.reason_lines())
    look_at_lines: Final = verdict.look_at_lines()
    look_at: Final = (
        ("\n\nLook at first:\n" + "\n".join(f"- `{item}`" for item in look_at_lines)) if look_at_lines else ""
    )
    return (
        f"Judge ({judgement.model}): {verdict.tier}, {verdict.change_kind.replace('_', ' ')}, "
        f"{verdict.blast_radius.replace('_', ' ')}\n{reasons}{look_at}"
    )


def _say(text: str) -> None:
    sys.stdout.write(text + "\n")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _judge_from_env(
    http: httpx.Client, prompt_path: Path, schema_path: Path
) -> JudgeClient | DecisionsJudgeClient | None:
    api_base, api_key, model = _env("LITELLM_API_BASE"), _env("LITELLM_API_KEY"), _env("STAMPBOT_JUDGE_MODEL")
    if not (api_base and api_key and model):
        return None
    if _env("STAMPBOT_JUDGE_ROUTE", "chat") == "decisions":
        return DecisionsJudgeClient(api_base, api_key, model, prompt_path.read_text(), http)
    schema: Final = SCHEMA_ADAPTER.validate_json(schema_path.read_text())
    return JudgeClient(api_base, api_key, model, prompt_path.read_text(), schema, http)


def gate_commenter(
    github: GitHub, org: str, team_slug: str, login: str, association: str, app_token: bool
) -> str | None:
    if not app_token:
        if association in ALLOWED_ASSOCIATIONS:
            return None
        return f"@{login} is not an org member (author association {association}), so this run stops here"
    membership: Final = github.is_team_member(org, team_slug, login)
    if membership is True:
        return None
    if membership is False:
        return f"@{login} is not an active member of the `{team_slug}` team, so this run stops here"
    return membership


def apply(github: GitHub, evaluation: Evaluation, outcome: Outcome, rules_path: str) -> None:
    number: Final = evaluation.pr.number
    github.upsert_comment(number, render_comment(evaluation, outcome, rules_path))
    github.set_risk_label(number, evaluation.final)
    approvals: Final = github.bot_approvals(number)
    if outcome.kind == "approved":
        if not any(r.commit_id == evaluation.pr.head.sha for r in approvals):
            github.approve(
                number,
                evaluation.pr.head.sha,
                f"{MARKER}\nLow risk by rule and by judgement at `{evaluation.pr.head.sha[:7]}`",
            )
        return
    for review in approvals:
        github.dismiss(number, review, f"risk moved to {evaluation.final.name} at {evaluation.pr.head.sha[:7]}")


def main() -> int:
    repo: Final = _env("GITHUB_REPOSITORY")
    number: Final = int(_env("PR_NUMBER"))
    event: Final = _env("STAMPBOT_EVENT", "comment")
    app_token: Final = _env("STAMPBOT_APP_TOKEN_PRESENT") == "true"
    dry_run: Final = _env("STAMPBOT_DRY_RUN") == "true"
    root: Final = Path(_env("STAMPBOT_ROOT", str(Path(__file__).resolve().parent.parent)))
    rules_path: Final = ".github/stampbot/risk_rules.yml"
    with httpx.Client() as http:
        github: Final = GitHub(repo, _env("GITHUB_TOKEN"), http)
        if event == "comment" and not dry_run:
            refusal: Final = gate_commenter(
                github,
                repo.split("/", 1)[0],
                _env("STAMPBOT_TEAM_SLUG", "litellm-engineering"),
                _env("COMMENTER"),
                _env("COMMENTER_ASSOCIATION"),
                app_token,
            )
            if refusal is not None:
                github.upsert_comment(number, f"{MARKER}\n### litellm-stampbot\n\n{refusal}.")
                _say(refusal)
                return 0
        if event == "synchronize" and not github.has_marker_comment(number):
            _say("never tagged, nothing to do")
            return 0
        judge: Final = _judge_from_env(
            http, root / ".github/stampbot/judge_prompt.md", root / ".github/stampbot/judge.schema.json"
        )
        evaluation: Final = evaluate(github, number, load_rules(root / rules_path), judge)
        outcome: Final = decide(evaluation, _env("STAMPBOT_APPROVALS_ENABLED") == "true", app_token)
        _say(render_comment(evaluation, outcome, rules_path))
        if not dry_run:
            apply(github, evaluation, outcome, rules_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
