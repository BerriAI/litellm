from datetime import date

import pytest
from sdk_watch.changelog import ChangelogEntry, Release
from sdk_watch.graph import Exact, OnDay, Recent, WatchDeps, run_watch
from sdk_watch.implement import Implementation, ImplementationReport, Outcome
from sdk_watch.report import summary
from sdk_watch.sources import Candidate, Sdk
from sdk_watch.triage import ChangeKind, Finding, NeededChange, TriageVerdict

OPENAI = Sdk(name="openai", repo="openai/openai-python")
ANTHROPIC = Sdk(name="anthropic", repo="anthropics/anthropic-sdk-python")

CHANGELOGS = {
    OPENAI.changelog_url: "## 3.59.0 (2026-11-24)\n\n### Features\n\n* add healthcare model IDs\n\n"
    "## 3.58.0 (2026-11-23)\n\n### Features\n\n* add a response field\n",
    ANTHROPIC.changelog_url: "## 0.105.0 (2026-11-24)\n\n### Features\n\n* mid-conversation system blocks\n",
}


def change(slug: str) -> NeededChange:
    return NeededChange(
        kind=ChangeKind.BEHAVIOR_CHANGE,
        slug=slug,
        sdk_change=f"note for {slug}",
        litellm_change=f"support {slug}",
        search_terms=(slug,),
    )


def triage_by_release(candidate: Candidate) -> TriageVerdict:
    slugs = {
        "3.59.0": ("healthcare_models",),
        "3.58.0": ("response_field",),
        "0.105.0": ("mid_conversation_system", "mid_conversation_system"),
    }[candidate.release.version]
    return TriageVerdict(changes=tuple(map(change, slugs)))


def implemented(finding: Finding) -> Implementation:
    return Implementation(
        finding=finding,
        report=ImplementationReport(
            outcome=Outcome.CHANGED,
            reason="done",
            pr_title=f"feat: {finding.change.slug}",
            walkthrough="walkthrough",
            test_files=("tests/unit/test_x.py",),
        ),
        patch="diff --git a/litellm/x.py b/litellm/x.py\n",
    )


def deps(**overrides: object) -> WatchDeps:
    defaults = {
        "fetch_text": CHANGELOGS.__getitem__,
        "triage": triage_by_release,
        "already_handled": lambda finding: None,
        "implement": implemented,
        "sdks": (OPENAI, ANTHROPIC),
        "selector": OnDay(day=date(2026, 11, 24)),
        "max_implementations": 5,
    }
    return WatchDeps(**{**defaults, **overrides})


def never(_: object) -> object:
    raise AssertionError("this stage must not run")


def slugs(findings: tuple[Finding, ...]) -> list[str]:
    return [finding.change.slug for finding in findings]


def test_a_day_implements_each_new_change_released_that_day_once() -> None:
    state = run_watch(deps())

    assert [(c.sdk.name, c.release.version) for c in state.candidates] == [
        ("openai", "3.59.0"),
        ("anthropic", "0.105.0"),
    ]
    assert slugs(tuple(impl.finding for impl in state.implementations)) == [
        "healthcare_models",
        "mid_conversation_system",
    ]
    assert all(impl.ships for impl in state.implementations)


def test_an_exact_release_ignores_the_day_and_other_sdks() -> None:
    state = run_watch(deps(selector=Exact(sdk="openai", version="3.58.0")))

    assert slugs(state.findings) == ["response_field"]


@pytest.mark.parametrize(
    ("until", "expected"),
    [
        (date(2026, 11, 25), [("openai", "3.58.0"), ("anthropic", "0.105.0")]),
        (date(2026, 11, 24), [("openai", "3.58.0")]),
    ],
    ids=["window-through-the-24th", "window-ending-before-the-24th"],
)
def test_a_recent_window_checks_each_release_in_it_that_no_earlier_run_checked(
    until: date, expected: list[tuple[str, str]]
) -> None:
    window = Recent(since=date(2026, 11, 23), until=until, checked=frozenset({("openai", "3.59.0")}))

    state = run_watch(deps(selector=window))

    assert [(c.sdk.name, c.release.version) for c in state.candidates] == expected


def test_changes_already_handled_upstream_are_never_implemented() -> None:
    pr = "https://github.com/BerriAI/litellm/pull/9"
    state = run_watch(deps(already_handled=lambda finding: pr if finding.change.slug == "healthcare_models" else None))

    assert slugs(tuple(impl.finding for impl in state.implementations)) == ["mid_conversation_system"]
    assert f"| openai 3.59.0 | healthcare_models | already handled by {pr} |" in summary(state)


def test_the_run_limit_defers_the_rest_to_a_later_run() -> None:
    state = run_watch(deps(max_implementations=1))

    assert slugs(tuple(impl.finding for impl in state.implementations)) == ["healthcare_models"]
    assert slugs(state.deferred) == ["mid_conversation_system"]
    assert "| anthropic 0.105.0 | mid_conversation_system | deferred to a later run" in summary(state)


def test_changes_an_earlier_run_deferred_are_implemented_even_with_no_new_release() -> None:
    leftover = run_watch(deps(max_implementations=1)).deferred

    state = run_watch(deps(selector=OnDay(day=date(2026, 1, 1)), triage=never, carried=leftover))

    assert slugs(tuple(impl.finding for impl in state.implementations)) == ["mid_conversation_system"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"selector": OnDay(day=date(2026, 1, 1)), "triage": never, "already_handled": never, "implement": never},
        {"triage": lambda candidate: TriageVerdict(changes=()), "already_handled": never, "implement": never},
        {"already_handled": lambda finding: f"branch {finding.branch}", "implement": never},
    ],
    ids=["no-release-that-day", "nothing-needs-litellm-work", "everything-already-handled"],
)
def test_a_stage_with_nothing_to_do_ends_the_run(overrides: dict[str, object]) -> None:
    assert run_watch(deps(**overrides)).implementations == ()


def test_the_branch_name_has_no_slash_and_names_sdk_version_and_change() -> None:
    finding = Finding(
        candidate=Candidate(
            sdk=Sdk(name="google-genai", repo="googleapis/python-genai"),
            release=Release(
                version="3.6.1", released=date(2026, 11, 18), entries=(ChangelogEntry(section="Features", text="x"),)
            ),
        ),
        change=change("tool_retrieval"),
    )

    assert finding.branch == "litellm_sdk_watch_google_genai_3_6_1_tool_retrieval"
