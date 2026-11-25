from datetime import date
from typing import Final

from pydantic import BaseModel, ConfigDict

from sdk_watch.graph import WatchState
from sdk_watch.implement import Implementation, Outcome
from sdk_watch.triage import Finding


class PrDraft(BaseModel):
    model_config = ConfigDict(frozen=True)

    branch: str
    title: str
    sdk: str
    version: str
    released: date
    release_url: str
    sdk_change: str
    litellm_change: str
    walkthrough: str
    test_files: tuple[str, ...]


class DemoVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    fails_before: bool
    passes_after: bool


def pr_draft(implementation: Implementation) -> PrDraft:
    finding: Final = implementation.finding
    release: Final = finding.candidate.release
    return PrDraft(
        branch=finding.branch,
        title=implementation.report.pr_title,
        sdk=finding.candidate.sdk.name,
        version=release.version,
        released=release.released,
        release_url=finding.candidate.sdk.release_url(release.version),
        sdk_change=finding.change.sdk_change,
        litellm_change=finding.change.litellm_change,
        walkthrough=implementation.report.walkthrough,
        test_files=implementation.report.test_files,
    )


def pr_body(draft: PrDraft, verdict: DemoVerdict, gif_url: str, run_url: str) -> str:
    before: Final = "fail" if verdict.fails_before else "already pass (the tests do not catch the missing change)"
    return f"""## Handwritten 1-3 sentence description

[insert handwritten description here]

<details>
<summary>Details</summary>

## TLDR

Problem this solves:

- {draft.sdk} {draft.version} ({draft.released}) shipped an API change LiteLLM did not handle yet
- Release note: {draft.sdk_change}

How it solves it:

- {draft.litellm_change}

## Walkthrough

{draft.walkthrough}

## Screenshots / Proof of Fix

The video runs the PR's tests on main without the fix, then with it. Without the fix they {before}, with it \
they pass

![The PR's tests before and after the fix]({gif_url})

Tests: {", ".join(f"`{path}`" for path in draft.test_files)}

The provider SDK watch workflow opened this PR from the [{draft.sdk} {draft.version} release]({draft.release_url}) \
in [this run]({run_url}). GitHub starts no CI on a PR the workflow token opens, so close and reopen it to run CI

## Type

New Feature

</details>
"""


def _outcome(state: WatchState, finding: Finding) -> str:
    implemented: Final = {impl.finding.branch: impl for impl in state.implementations}
    if finding not in state.fresh:
        return "already handled by an open or recently merged PR"
    implementation: Final = implemented.get(finding.branch)
    if implementation is None:
        return "deferred to a later run (over this run's limit)"
    if implementation.ships:
        return "changed, PR draft ready"
    if implementation.report.outcome is Outcome.CHANGED:
        return "changed but left no patch or test, dropped"
    return f"{implementation.report.outcome}: {implementation.report.reason}"


def summary(state: WatchState) -> str:
    releases: Final = ", ".join(f"{c.sdk.name} {c.release.version}" for c in state.candidates) or "none"
    rows: Final = "".join(
        f"| {finding.candidate.sdk.name} {finding.candidate.release.version} | {finding.change.slug} | "
        f"{' '.join(_outcome(state, finding).split()).replace('|', '/')} |\n"
        for finding in state.findings
    )
    table: Final = (
        f"| Release | Change | Outcome |\n| --- | --- | --- |\n{rows}" if rows else "No change needs LiteLLM work\n"
    )
    return f"## Provider SDK watch\n\nReleases checked: {releases}\n\n{table}"
