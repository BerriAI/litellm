from datetime import date
from pathlib import Path

from sdk_watch.changelog import Release
from sdk_watch.history import CheckedRelease, RunHistory, carried_findings, load_history, next_history, save_history
from sdk_watch.sources import Candidate, Sdk
from sdk_watch.triage import ChangeKind, Finding, NeededChange

ANTHROPIC = Sdk(name="anthropic", repo="anthropics/anthropic-sdk-python")

LEFTOVER = Finding(
    candidate=Candidate(sdk=ANTHROPIC, release=Release(version="0.105.0", released=date(2026, 11, 24), entries=())),
    change=NeededChange(
        kind=ChangeKind.BEHAVIOR_CHANGE,
        slug="mid_conversation_system",
        sdk_change="**api:** mid-conversation system blocks",
        litellm_change="Keep system blocks in place",
        search_terms=("mid_conversation_system",),
    ),
)


def test_the_saved_history_keeps_checked_releases_inside_the_window_and_the_changes_left_over(tmp_path: Path) -> None:
    before_window = CheckedRelease(sdk="openai", version="3.50.0", released=date(2026, 11, 10))
    in_window = CheckedRelease(sdk="openai", version="3.58.0", released=date(2026, 11, 23))
    path = tmp_path / "history.json"

    save_history(
        path,
        next_history(
            RunHistory(checked=(before_window, in_window)), date(2026, 11, 18), (LEFTOVER.candidate,), (LEFTOVER,)
        ),
    )
    history = load_history(path)

    assert history.checked_keys == {("openai", "3.58.0"), ("anthropic", "0.105.0")}
    assert carried_findings(history, (ANTHROPIC,)) == (LEFTOVER,)


def test_the_first_run_starts_from_an_empty_history(tmp_path: Path) -> None:
    assert load_history(tmp_path / "history.json") == RunHistory()
