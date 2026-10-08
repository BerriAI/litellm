"""Tests for scripts/lint_base_counts.py, the merge-base counting shared by the
four lint gates: the ceiling rule with its optional per-rule caps, the on-disk cache and
its eviction, the CI artifact fetch, the artifact emit, and the merge-base
resolution."""

import fnmatch
import io
import json
import os
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple, NoReturn

import pytest

import lint_base_counts as counts

_CHECKER: Final = counts.Checker("basedpyright", ("f1", "f2"))
_OTHER_CHECKER: Final = counts.Checker("ruff-strict", ("f1", "f2"))


def test_evaluate_passes_a_rule_that_did_not_grow() -> None:
    assert counts.evaluate({"LIT006": 12}, {"LIT006": 12}) == ()


def test_evaluate_blames_one_new_violation_of_an_uncapped_rule() -> None:
    assert counts.evaluate({"LIT006": 13}, {"LIT006": 12}) == (counts.Breach("LIT006", 13, 12, 1),)


def test_evaluate_lets_a_capped_rule_grow_up_to_its_cap_and_no_further() -> None:
    caps: Final = {"reportAny": 110}
    assert counts.evaluate({"reportAny": 110}, {"reportAny": 100}, caps) == ()
    assert counts.evaluate({"reportAny": 111}, {"reportAny": 100}, caps) == (counts.Breach("reportAny", 111, 110, 11),)


def test_evaluate_never_blames_a_bystander_for_a_base_already_over_the_cap() -> None:
    caps: Final = {"reportAny": 110}
    assert counts.evaluate({"reportAny": 120}, {"reportAny": 120}, caps) == ()
    assert counts.evaluate({"reportAny": 119}, {"reportAny": 120}, caps) == ()


def test_evaluate_holds_a_base_over_its_cap_to_no_growth() -> None:
    caps: Final = {"reportAny": 110}
    assert counts.evaluate({"reportAny": 121}, {"reportAny": 120}, caps) == (counts.Breach("reportAny", 121, 120, 1),)


def test_evaluate_cap_applies_only_to_the_rule_it_names() -> None:
    caps: Final = {"reportAny": 110}
    assert counts.evaluate({"LIT006": 13}, {"LIT006": 12}, caps) == (counts.Breach("LIT006", 13, 12, 1),)


def test_evaluate_counts_a_rule_absent_from_the_base_as_zero() -> None:
    assert counts.evaluate({"NEW99": 1}, {}) == (counts.Breach("NEW99", 1, 0, 1),)


def test_evaluate_never_blames_a_change_that_reduced_a_rule() -> None:
    assert counts.evaluate({"LIT006": 11}, {"LIT006": 12}) == ()


def test_evaluate_reports_every_grown_rule_sorted_by_name() -> None:
    head: Final = {"TQ008": 3, "TQ001": 2, "TQ003": 5}
    base: Final = {"TQ008": 2, "TQ001": 1, "TQ003": 5}
    assert [b.rule for b in counts.evaluate(head, base)] == ["TQ001", "TQ008"]


def test_evaluate_ignores_a_base_rule_the_head_fixed_entirely() -> None:
    assert counts.evaluate({}, {"LIT006": 12}) == ()


def test_cache_key_changes_with_base_point_and_each_fingerprint() -> None:
    key: Final = counts.cache_key("abc", ("cfg", "lock"))
    assert counts.cache_key("abc", ("cfg", "lock")) == key
    assert counts.cache_key("def", ("cfg", "lock")) != key
    assert counts.cache_key("abc", ("cfg2", "lock")) != key
    assert counts.cache_key("abc", ("cfg", "lock2")) != key


def test_checker_names_its_artifact_and_cache_file_by_the_same_key() -> None:
    key: Final = counts.cache_key("abc123", ("f1", "f2"))
    assert _CHECKER.artifact_name("abc123") == f"basedpyright-counts-{key}"
    assert _CHECKER.cache_file_name("abc123") == f"basedpyright-base-{key}.json"
    assert fnmatch.fnmatch(_CHECKER.cache_file_name("abc123"), _CHECKER.cache_glob())


def test_checkers_with_the_same_fingerprints_never_share_a_name() -> None:
    assert _CHECKER.artifact_name("abc123") != _OTHER_CHECKER.artifact_name("abc123")
    assert not fnmatch.fnmatch(_OTHER_CHECKER.cache_file_name("abc123"), _CHECKER.cache_glob())


def test_cached_counts_round_trip(tmp_path: Path) -> None:
    path: Final = counts.store_counts(tmp_path, _CHECKER, "abc123", {"reportAny": 3, "reportCall": 1})
    assert path == tmp_path / _CHECKER.cache_file_name("abc123")
    assert counts.load_cached_counts(path) == {"reportAny": 3, "reportCall": 1}


@pytest.mark.parametrize(
    "content",
    [
        None,
        "{not json",
        json.dumps(["counts"]),
        json.dumps({"base_point": "abc"}),
        json.dumps({"counts": {"reportAny": "three"}}),
        json.dumps({"counts": {"reportAny": True}}),
    ],
)
def test_missing_corrupt_or_misshapen_cache_reads_as_none(tmp_path: Path, content: str | None) -> None:
    path: Final = tmp_path / "cache.json"
    if content is not None:
        path.write_text(content)
    assert counts.load_cached_counts(path) is None


def test_scratch_is_invisible_to_the_prune_glob() -> None:
    scratch: Final = counts.scratch_path(Path("/c") / _CHECKER.cache_file_name("abc"))
    assert not fnmatch.fnmatch(scratch.name, _CHECKER.cache_glob())


def test_store_prune_spares_a_concurrent_runs_in_flight_scratch(tmp_path: Path) -> None:
    foreign: Final = counts.scratch_path(tmp_path / _CHECKER.cache_file_name("other"))
    foreign.write_text("{}")
    mine: Final = counts.store_counts(tmp_path, _CHECKER, "mine", {"reportAny": 1})
    assert foreign.exists()
    assert counts.load_cached_counts(mine) == {"reportAny": 1}


def test_store_keeps_a_concurrent_worktrees_entry_for_another_branch_point(tmp_path: Path) -> None:
    old: Final = counts.store_counts(tmp_path, _CHECKER, "old", {"reportAny": 1})
    new: Final = counts.store_counts(tmp_path, _CHECKER, "new", {"reportAny": 2})
    assert counts.load_cached_counts(old) == {"reportAny": 1}
    assert counts.load_cached_counts(new) == {"reportAny": 2}


def test_store_evicts_only_the_oldest_entries_beyond_the_cap(tmp_path: Path) -> None:
    aged: Final = tuple(
        counts.store_counts(tmp_path, _CHECKER, f"base{age}", {"reportAny": age})
        for age in range(counts.CACHE_KEEP_ENTRIES)
    )
    for age, path in enumerate(aged):
        os.utime(path, (age, age))
    newest: Final = counts.store_counts(tmp_path, _CHECKER, "newest", {"reportAny": 99})
    assert not aged[0].exists()
    assert all(path.exists() for path in aged[1:])
    assert counts.load_cached_counts(newest) == {"reportAny": 99}


def test_store_never_evicts_the_entry_it_just_wrote_even_on_mtime_ties(tmp_path: Path) -> None:
    for index in range(counts.CACHE_KEEP_ENTRIES + 2):
        os.utime(counts.store_counts(tmp_path, _CHECKER, f"base{index}", {"reportAny": 1}), (9_999_999_999,) * 2)
    mine: Final = counts.store_counts(tmp_path, _CHECKER, "mine", {"reportAny": 2})
    assert counts.load_cached_counts(mine) == {"reportAny": 2}
    assert len(list(tmp_path.glob(_CHECKER.cache_glob()))) == counts.CACHE_KEEP_ENTRIES


def test_store_eviction_never_touches_another_checkers_entries(tmp_path: Path) -> None:
    other: Final = counts.store_counts(tmp_path, _OTHER_CHECKER, "base", {"E501": 1})
    os.utime(other, (1, 1))
    for index in range(counts.CACHE_KEEP_ENTRIES + 1):
        counts.store_counts(tmp_path, _CHECKER, f"base{index}", {"reportAny": 1})
    assert counts.load_cached_counts(other) == {"E501": 1}


def _no_fetch(checker: counts.Checker, base_point: str) -> None:
    return None


def _never(reason: str) -> Callable[..., NoReturn]:
    def callback(*args: object) -> NoReturn:
        raise AssertionError(reason)

    return callback


def test_base_counts_cached_returns_the_hit_without_recomputing(tmp_path: Path) -> None:
    counts.store_counts(tmp_path, _CHECKER, "abc123", {"reportAny": 7})
    assert counts.base_counts_cached(
        _CHECKER,
        "abc123",
        _never("a cache hit must not re-run the base pass"),
        cache_dir=tmp_path,
        fetch=_never("a cache hit must not reach for CI"),
    ) == {"reportAny": 7}


def test_base_counts_cached_computes_once_then_hits(tmp_path: Path) -> None:
    calls: Final[list[str]] = []

    def fake(ref: str) -> counts.Counts:
        calls.append(ref)
        return {"reportAny": 4}

    first: Final = counts.base_counts_cached(_CHECKER, "abc123", fake, cache_dir=tmp_path, fetch=_no_fetch)
    second: Final = counts.base_counts_cached(_CHECKER, "abc123", fake, cache_dir=tmp_path, fetch=_no_fetch)
    assert first == second == {"reportAny": 4}
    assert calls == ["abc123"]


def test_base_counts_cached_keeps_each_checker_apart(tmp_path: Path) -> None:
    counts.store_counts(tmp_path, _OTHER_CHECKER, "abc123", {"E501": 7})
    assert counts.base_counts_cached(
        _CHECKER, "abc123", lambda ref: {"reportAny": 4}, cache_dir=tmp_path, fetch=_no_fetch
    ) == {"reportAny": 4}


def test_an_empty_base_pass_fails_the_gate_instead_of_blaming_the_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        counts.base_counts_cached(
            _CHECKER, "abc123def4567", lambda ref: {}, cache_dir=tmp_path, fetch=lambda checker, base_point: {}
        )
    assert exit_info.value.code == 1
    assert f"FAIL: {_CHECKER.name} produced no violations for the base tree at abc123def456," in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_base_counts_cached_uses_fetched_counts_and_persists_them(tmp_path: Path) -> None:
    fetched: Final = counts.base_counts_cached(
        _CHECKER,
        "abc123",
        _never("fetched counts must skip the local base pass"),
        cache_dir=tmp_path,
        fetch=lambda checker, base_point: {"reportAny": 9},
    )
    assert fetched == {"reportAny": 9}
    assert counts.load_cached_counts(tmp_path / _CHECKER.cache_file_name("abc123")) == {"reportAny": 9}
    assert counts.base_counts_cached(
        _CHECKER,
        "abc123",
        _never("the persisted fetch must satisfy later runs"),
        cache_dir=tmp_path,
        fetch=_never("the persisted fetch must satisfy later runs"),
    ) == {"reportAny": 9}


def test_base_counts_cached_hands_the_fetcher_the_checker_and_base_point(tmp_path: Path) -> None:
    seen: Final[list[tuple[counts.Checker, str]]] = []

    def fetch(checker: counts.Checker, base_point: str) -> None:
        seen.append((checker, base_point))

    counts.base_counts_cached(_CHECKER, "abc123", lambda ref: {"reportAny": 4}, cache_dir=tmp_path, fetch=fetch)
    assert seen == [(_CHECKER, "abc123")]


def test_base_counts_cached_falls_back_to_compute_on_a_fetch_miss(tmp_path: Path) -> None:
    calls: Final[list[str]] = []

    def local(ref: str) -> counts.Counts:
        calls.append(ref)
        return {"reportAny": 4}

    assert counts.base_counts_cached(_CHECKER, "abc123", local, cache_dir=tmp_path, fetch=_no_fetch) == {
        "reportAny": 4
    }
    assert calls == ["abc123"]


def test_base_counts_cached_treats_empty_fetched_counts_as_a_miss(tmp_path: Path) -> None:
    assert counts.base_counts_cached(
        _CHECKER,
        "abc123",
        lambda ref: {"reportAny": 2},
        cache_dir=tmp_path,
        fetch=lambda checker, base_point: {},
    ) == {"reportAny": 2}
    assert counts.load_cached_counts(tmp_path / _CHECKER.cache_file_name("abc123")) == {"reportAny": 2}


def test_origin_slug_parsing_supports_ssh_and_https_github_forms() -> None:
    assert counts.parse_origin_slug("git@github.com:BerriAI/litellm.git") == "BerriAI/litellm"
    assert counts.parse_origin_slug("git@github.com:BerriAI/litellm") == "BerriAI/litellm"
    assert counts.parse_origin_slug("https://github.com/BerriAI/litellm.git") == "BerriAI/litellm"
    assert counts.parse_origin_slug("https://github.com/BerriAI/litellm") == "BerriAI/litellm"
    assert counts.parse_origin_slug("https://github.com/BerriAI/litellm/") == "BerriAI/litellm"


def test_origin_slug_parsing_rejects_non_github_urls() -> None:
    assert counts.parse_origin_slug("https://gitlab.com/BerriAI/litellm.git") is None
    assert counts.parse_origin_slug("git@bitbucket.org:BerriAI/litellm.git") is None
    assert counts.parse_origin_slug("not a url") is None
    assert counts.parse_origin_slug("") is None


def _artifact_zip(payload: Mapping[str, object]) -> bytes:
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("counts.json", json.dumps(payload))
    return buffer.getvalue()


def _gh_stub(
    listing: Mapping[str, object], zip_bytes: bytes, seen: list[tuple[str, ...]] | None = None
) -> counts.GhOutput:
    def gh_output(args: Sequence[str]) -> bytes:
        if seen is not None:
            seen.append(tuple(args))
        if args[-1].startswith("repos/"):
            return json.dumps(listing).encode()
        return zip_bytes

    return gh_output


def _live_listing() -> Mapping[str, object]:
    return {"artifacts": [{"expired": False, "archive_download_url": "https://api.github.com/x/zip"}]}


def test_fetcher_returns_counts_from_a_matching_artifact(capsys: pytest.CaptureFixture[str]) -> None:
    payload: Final = {"base_point": "abc123", "counts": {"reportAny": 3}}
    fetched: Final = counts.fetch_ci_base_counts(
        _CHECKER, "abc123", gh=_gh_stub(_live_listing(), _artifact_zip(payload))
    )
    assert fetched == {"reportAny": 3}
    assert "fetched from CI artifact" in capsys.readouterr().err


def test_fetcher_asks_for_the_artifact_named_by_the_checker_and_base_point() -> None:
    seen: Final[list[tuple[str, ...]]] = []
    payload: Final = {"base_point": "abc123", "counts": {"reportAny": 3}}
    counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), _artifact_zip(payload), seen))
    listing_request: Final = seen[0][-1]
    assert f"name={_CHECKER.artifact_name('abc123')}" in listing_request
    assert seen[1][-1] == "https://api.github.com/x/zip"


def test_fetcher_rejects_an_artifact_for_a_different_base_point() -> None:
    payload: Final = {"base_point": "someothersha", "counts": {"reportAny": 3}}
    assert (
        counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), _artifact_zip(payload))) is None
    )


@pytest.mark.parametrize("bad_counts", [{}, {"reportAny": "three"}, {"reportAny": True}])
def test_fetcher_rejects_empty_or_misshapen_artifact_counts(bad_counts: Mapping[str, object]) -> None:
    payload: Final = {"base_point": "abc123", "counts": bad_counts}
    assert (
        counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), _artifact_zip(payload))) is None
    )


def test_fetcher_rejects_an_expired_artifact() -> None:
    listing: Final = {"artifacts": [{"expired": True, "archive_download_url": "https://api.github.com/x/zip"}]}
    payload: Final = {"base_point": "abc123", "counts": {"reportAny": 3}}
    assert counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(listing, _artifact_zip(payload))) is None


def test_fetcher_misses_when_no_artifact_is_published() -> None:
    assert counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub({"artifacts": []}, b"")) is None


def test_fetcher_misses_when_gh_is_unusable(capsys: pytest.CaptureFixture[str]) -> None:
    assert counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=lambda args: None) is None
    assert "computing base counts locally" in capsys.readouterr().err


def test_fetcher_misses_on_a_corrupt_artifact_archive() -> None:
    assert counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), b"not a zip")) is None


def test_emit_writes_the_artifact_json_named_by_the_head_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    counts.emit_counts(_CHECKER, {"reportAny": 3, "aRule": 1}, tmp_path, "deadbeef")
    name: Final = _CHECKER.artifact_name("deadbeef")
    assert json.loads((tmp_path / f"{name}.json").read_text()) == {
        "base_point": "deadbeef",
        "counts": {"aRule": 1, "reportAny": 3},
    }
    summary: Final = capsys.readouterr().out
    assert "deadbeef" in summary
    assert name in summary
    assert "4" in summary


def test_emit_refuses_to_publish_empty_counts(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        counts.emit_counts(_CHECKER, {}, tmp_path, "deadbeef")
    assert list(tmp_path.iterdir()) == []


def test_emitted_file_is_the_one_the_fetcher_looks_up(tmp_path: Path) -> None:
    written: Final = counts.emit_counts(_CHECKER, {"reportAny": 3}, tmp_path, "deadbeef")
    payload: Final = json.loads(written.read_text())
    assert counts.counts_for_base(payload, "deadbeef") == {"reportAny": 3}
    assert counts.counts_for_base(payload, "someothersha") is None
    listing_zip: Final = _artifact_zip(payload)
    assert counts.fetch_ci_base_counts(_CHECKER, "deadbeef", gh=_gh_stub(_live_listing(), listing_zip)) == {
        "reportAny": 3
    }


class _History(NamedTuple):
    parents: Mapping[str, tuple[str, ...]]
    refs: Mapping[str, str]

    def ancestry(self, commit: str) -> frozenset[str]:
        return frozenset((commit,)).union(*(self.ancestry(parent) for parent in self.parents[commit]))

    def merge_base(self, left: str, right: str) -> str:
        common: Final = self.ancestry(self.refs.get(left, left)) & self.ancestry(self.refs.get(right, right))
        return next(c for c in common if not any(c != other and c in self.ancestry(other) for other in common))


def _git_over(history: _History) -> counts.Git:
    def git(args: Sequence[str]) -> str:
        match tuple(args):
            case ("merge-base", left, right):
                return f"{history.merge_base(left, right)}\n"
            case ("rev-parse", "HEAD"):
                return f"{history.refs['HEAD']}\n"
            case ("rev-parse", "--verify", "--quiet", ref):
                return f"{history.refs[ref]}\n" if ref in history.refs else ""
            case ("rev-parse", "--path-format=absolute", "--git-common-dir"):
                return "/repo/.git\n"
            case ("rev-parse", "--path-format=absolute", "--git-dir"):
                return "/repo/.git/worktrees/feature\n"
            case _:
                raise AssertionError(f"unexpected git call: {args}")

    return git


_FEATURE_OFF_MAIN: Final = _History(
    parents={"shared": (), "feature": ("shared",), "drift": ("shared",)},
    refs={"main": "drift", "HEAD": "feature"},
)


def test_base_point_is_the_branch_point_when_no_merge_is_in_progress() -> None:
    assert counts.resolve_base_point("main", _git_over(_FEATURE_OFF_MAIN)) == "shared"


def test_base_point_mid_merge_advances_to_the_merged_in_base_tip() -> None:
    merging_main: Final = _FEATURE_OFF_MAIN._replace(refs={**_FEATURE_OFF_MAIN.refs, "MERGE_HEAD": "drift"})
    assert counts.resolve_base_point("main", _git_over(merging_main)) == "drift"


def test_base_point_mid_merge_of_an_older_side_branch_keeps_the_newer_branch_point() -> None:
    merging_old_side: Final = _History(
        parents={"shared": (), "old": ("shared",), "drift": ("shared",), "feature": ("drift",)},
        refs={"main": "drift", "HEAD": "feature", "MERGE_HEAD": "old"},
    )
    assert counts.resolve_base_point("main", _git_over(merging_old_side)) == "drift"


def test_head_sha_is_the_checked_out_commit() -> None:
    assert counts.head_sha(_git_over(_FEATURE_OFF_MAIN)) == "feature"


def test_default_cache_dir_is_shared_by_every_worktree() -> None:
    assert counts.default_cache_dir(_git_over(_FEATURE_OFF_MAIN)) == Path("/repo/.git") / counts.CACHE_DIR_NAME
