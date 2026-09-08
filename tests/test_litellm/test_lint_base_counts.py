"""Tests for scripts/lint_base_counts.py, the merge-base counting shared by the
four lint gates: the ceiling rule with per-rule headroom, the on-disk cache and
its eviction, the CI artifact fetch, the artifact emit, and the merge-base
resolution."""

import fnmatch
import io
import json
import os
import subprocess
import zipfile
from pathlib import Path

import pytest

import lint_base_counts as counts

_CHECKER = counts.Checker("basedpyright", ("f1", "f2"))
_OTHER_CHECKER = counts.Checker("ruff-strict", ("f1", "f2"))


def test_evaluate_passes_a_rule_that_did_not_grow():
    assert counts.evaluate({"LIT006": 12}, {"LIT006": 12}, {}) == ()


def test_evaluate_blames_one_new_violation_of_a_zero_headroom_rule():
    assert counts.evaluate({"LIT006": 13}, {"LIT006": 12}, {}) == (counts.Breach("LIT006", 13, 12, 1),)


def test_evaluate_lets_a_rule_grow_up_to_its_headroom_and_no_further():
    headroom = {"LIT010": 44}
    assert counts.evaluate({"LIT010": 144}, {"LIT010": 100}, headroom) == ()
    assert counts.evaluate({"LIT010": 145}, {"LIT010": 100}, headroom) == (counts.Breach("LIT010", 145, 144, 45),)


def test_evaluate_headroom_applies_only_to_the_rule_it_names():
    assert counts.evaluate({"LIT006": 13}, {"LIT006": 12}, {"LIT010": 44}) == (counts.Breach("LIT006", 13, 12, 1),)


def test_evaluate_counts_a_rule_absent_from_the_base_as_zero():
    assert counts.evaluate({"NEW99": 1}, {}, {}) == (counts.Breach("NEW99", 1, 0, 1),)


def test_evaluate_never_blames_a_change_that_reduced_a_rule():
    assert counts.evaluate({"LIT006": 11}, {"LIT006": 12}, {}) == ()


def test_evaluate_reports_every_grown_rule_sorted_by_name():
    head = {"TQ008": 3, "TQ001": 2, "TQ003": 5}
    base = {"TQ008": 2, "TQ001": 1, "TQ003": 5}
    assert [b.rule for b in counts.evaluate(head, base, {})] == ["TQ001", "TQ008"]


def test_evaluate_ignores_a_base_rule_the_head_fixed_entirely():
    assert counts.evaluate({}, {"LIT006": 12}, {}) == ()


def test_cache_key_changes_with_base_point_and_each_fingerprint():
    key = counts.cache_key("abc", ("cfg", "lock"))
    assert counts.cache_key("abc", ("cfg", "lock")) == key
    assert counts.cache_key("def", ("cfg", "lock")) != key
    assert counts.cache_key("abc", ("cfg2", "lock")) != key
    assert counts.cache_key("abc", ("cfg", "lock2")) != key


def test_checker_names_its_artifact_and_cache_file_by_the_same_key():
    key = counts.cache_key("abc123", ("f1", "f2"))
    assert _CHECKER.artifact_name("abc123") == f"basedpyright-counts-{key}"
    assert _CHECKER.cache_file_name("abc123") == f"basedpyright-base-{key}.json"
    assert fnmatch.fnmatch(_CHECKER.cache_file_name("abc123"), _CHECKER.cache_glob())


def test_checkers_with_the_same_fingerprints_never_share_a_name():
    assert _CHECKER.artifact_name("abc123") != _OTHER_CHECKER.artifact_name("abc123")
    assert not fnmatch.fnmatch(_OTHER_CHECKER.cache_file_name("abc123"), _CHECKER.cache_glob())


def test_cached_counts_round_trip(tmp_path):
    path = counts.store_counts(tmp_path, _CHECKER, "abc123", {"reportAny": 3, "reportCall": 1})
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
def test_missing_corrupt_or_misshapen_cache_reads_as_none(tmp_path, content):
    path = tmp_path / "cache.json"
    if content is not None:
        path.write_text(content)
    assert counts.load_cached_counts(path) is None


def test_scratch_is_invisible_to_the_prune_glob():
    scratch = counts.scratch_path(Path("/c") / _CHECKER.cache_file_name("abc"))
    assert not fnmatch.fnmatch(scratch.name, _CHECKER.cache_glob())


def test_store_prune_spares_a_concurrent_runs_in_flight_scratch(tmp_path):
    foreign = counts.scratch_path(tmp_path / _CHECKER.cache_file_name("other"))
    foreign.write_text("{}")
    mine = counts.store_counts(tmp_path, _CHECKER, "mine", {"reportAny": 1})
    assert foreign.exists()
    assert counts.load_cached_counts(mine) == {"reportAny": 1}


def test_store_keeps_a_concurrent_worktrees_entry_for_another_branch_point(tmp_path):
    old = counts.store_counts(tmp_path, _CHECKER, "old", {"reportAny": 1})
    new = counts.store_counts(tmp_path, _CHECKER, "new", {"reportAny": 2})
    assert counts.load_cached_counts(old) == {"reportAny": 1}
    assert counts.load_cached_counts(new) == {"reportAny": 2}


def test_store_evicts_only_the_oldest_entries_beyond_the_cap(tmp_path):
    aged = tuple(
        counts.store_counts(tmp_path, _CHECKER, f"base{age}", {"reportAny": age})
        for age in range(counts.CACHE_KEEP_ENTRIES)
    )
    for age, path in enumerate(aged):
        os.utime(path, (age, age))
    newest = counts.store_counts(tmp_path, _CHECKER, "newest", {"reportAny": 99})
    assert not aged[0].exists()
    assert all(path.exists() for path in aged[1:])
    assert counts.load_cached_counts(newest) == {"reportAny": 99}


def test_store_never_evicts_the_entry_it_just_wrote_even_on_mtime_ties(tmp_path):
    for index in range(counts.CACHE_KEEP_ENTRIES + 2):
        path = counts.store_counts(tmp_path, _CHECKER, f"base{index}", {"reportAny": 1})
        os.utime(path, (9_999_999_999, 9_999_999_999))
    mine = counts.store_counts(tmp_path, _CHECKER, "mine", {"reportAny": 2})
    assert counts.load_cached_counts(mine) == {"reportAny": 2}
    assert len(list(tmp_path.glob(_CHECKER.cache_glob()))) == counts.CACHE_KEEP_ENTRIES


def test_store_eviction_never_touches_another_checkers_entries(tmp_path):
    other = counts.store_counts(tmp_path, _OTHER_CHECKER, "base", {"E501": 1})
    os.utime(other, (1, 1))
    for index in range(counts.CACHE_KEEP_ENTRIES + 1):
        counts.store_counts(tmp_path, _CHECKER, f"base{index}", {"reportAny": 1})
    assert counts.load_cached_counts(other) == {"E501": 1}


def _no_fetch(checker, base_point):
    return None


def _never(reason):
    def callback(*args):
        raise AssertionError(reason)

    return callback


def test_base_counts_cached_returns_the_hit_without_recomputing(tmp_path):
    counts.store_counts(tmp_path, _CHECKER, "abc123", {"reportAny": 7})
    assert counts.base_counts_cached(
        _CHECKER,
        "abc123",
        _never("a cache hit must not re-run the base pass"),
        cache_dir=tmp_path,
        fetch=_never("a cache hit must not reach for CI"),
    ) == {"reportAny": 7}


def test_base_counts_cached_computes_once_then_hits(tmp_path):
    calls = []

    def fake(ref):
        calls.append(ref)
        return {"reportAny": 4}

    first = counts.base_counts_cached(_CHECKER, "abc123", fake, cache_dir=tmp_path, fetch=_no_fetch)
    second = counts.base_counts_cached(_CHECKER, "abc123", fake, cache_dir=tmp_path, fetch=_no_fetch)
    assert first == second == {"reportAny": 4}
    assert calls == ["abc123"]


def test_base_counts_cached_keeps_each_checker_apart(tmp_path):
    counts.store_counts(tmp_path, _OTHER_CHECKER, "abc123", {"E501": 7})
    assert counts.base_counts_cached(
        _CHECKER, "abc123", lambda ref: {"reportAny": 4}, cache_dir=tmp_path, fetch=_no_fetch
    ) == {"reportAny": 4}


def test_an_empty_base_pass_is_never_cached(tmp_path):
    calls = []

    def crashed(ref):
        calls.append(ref)
        return {}

    assert counts.base_counts_cached(_CHECKER, "abc123", crashed, cache_dir=tmp_path, fetch=_no_fetch) == {}
    assert counts.base_counts_cached(_CHECKER, "abc123", crashed, cache_dir=tmp_path, fetch=_no_fetch) == {}
    assert calls == ["abc123", "abc123"]
    assert list(tmp_path.iterdir()) == []


def test_base_counts_cached_uses_fetched_counts_and_persists_them(tmp_path):
    fetched = counts.base_counts_cached(
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


def test_base_counts_cached_hands_the_fetcher_the_checker_and_base_point(tmp_path):
    seen = []

    def fetch(checker, base_point):
        seen.append((checker, base_point))
        return None

    counts.base_counts_cached(_CHECKER, "abc123", lambda ref: {"reportAny": 4}, cache_dir=tmp_path, fetch=fetch)
    assert seen == [(_CHECKER, "abc123")]


def test_base_counts_cached_falls_back_to_compute_on_a_fetch_miss(tmp_path):
    calls = []

    def local(ref):
        calls.append(ref)
        return {"reportAny": 4}

    assert counts.base_counts_cached(_CHECKER, "abc123", local, cache_dir=tmp_path, fetch=_no_fetch) == {
        "reportAny": 4
    }
    assert calls == ["abc123"]


def test_base_counts_cached_treats_empty_fetched_counts_as_a_miss(tmp_path):
    assert counts.base_counts_cached(
        _CHECKER,
        "abc123",
        lambda ref: {"reportAny": 2},
        cache_dir=tmp_path,
        fetch=lambda checker, base_point: {},
    ) == {"reportAny": 2}
    assert counts.load_cached_counts(tmp_path / _CHECKER.cache_file_name("abc123")) == {"reportAny": 2}


def test_origin_slug_parsing_supports_ssh_and_https_github_forms():
    assert counts.parse_origin_slug("git@github.com:BerriAI/litellm.git") == "BerriAI/litellm"
    assert counts.parse_origin_slug("git@github.com:BerriAI/litellm") == "BerriAI/litellm"
    assert counts.parse_origin_slug("https://github.com/BerriAI/litellm.git") == "BerriAI/litellm"
    assert counts.parse_origin_slug("https://github.com/BerriAI/litellm") == "BerriAI/litellm"
    assert counts.parse_origin_slug("https://github.com/BerriAI/litellm/") == "BerriAI/litellm"


def test_origin_slug_parsing_rejects_non_github_urls():
    assert counts.parse_origin_slug("https://gitlab.com/BerriAI/litellm.git") is None
    assert counts.parse_origin_slug("git@bitbucket.org:BerriAI/litellm.git") is None
    assert counts.parse_origin_slug("not a url") is None
    assert counts.parse_origin_slug("") is None


def _artifact_zip(payload):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("counts.json", json.dumps(payload))
    return buffer.getvalue()


def _gh_stub(listing, zip_bytes, seen=None):
    def gh_output(args):
        if seen is not None:
            seen.append(tuple(args))
        if args[-1].startswith("repos/"):
            return json.dumps(listing).encode()
        return zip_bytes

    return gh_output


def _live_listing():
    return {"artifacts": [{"expired": False, "archive_download_url": "https://api.github.com/x/zip"}]}


def test_fetcher_returns_counts_from_a_matching_artifact(capsys):
    payload = {"base_point": "abc123", "counts": {"reportAny": 3}}
    fetched = counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), _artifact_zip(payload)))
    assert fetched == {"reportAny": 3}
    assert "fetched from CI artifact" in capsys.readouterr().err


def test_fetcher_asks_for_the_artifact_named_by_the_checker_and_base_point():
    seen = []
    payload = {"base_point": "abc123", "counts": {"reportAny": 3}}
    counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), _artifact_zip(payload), seen))
    listing_request = seen[0][-1]
    assert f"name={_CHECKER.artifact_name('abc123')}" in listing_request
    assert seen[1][-1] == "https://api.github.com/x/zip"


def test_fetcher_rejects_an_artifact_for_a_different_base_point():
    payload = {"base_point": "someothersha", "counts": {"reportAny": 3}}
    assert (
        counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), _artifact_zip(payload))) is None
    )


@pytest.mark.parametrize("bad_counts", [{}, {"reportAny": "three"}, {"reportAny": True}])
def test_fetcher_rejects_empty_or_misshapen_artifact_counts(bad_counts):
    payload = {"base_point": "abc123", "counts": bad_counts}
    assert (
        counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), _artifact_zip(payload))) is None
    )


def test_fetcher_rejects_an_expired_artifact():
    listing = {"artifacts": [{"expired": True, "archive_download_url": "https://api.github.com/x/zip"}]}
    payload = {"base_point": "abc123", "counts": {"reportAny": 3}}
    assert counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(listing, _artifact_zip(payload))) is None


def test_fetcher_misses_when_no_artifact_is_published():
    assert counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub({"artifacts": []}, b"")) is None


def test_fetcher_misses_when_gh_is_unusable(capsys):
    assert counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=lambda args: None) is None
    assert "computing base counts locally" in capsys.readouterr().err


def test_fetcher_misses_on_a_corrupt_artifact_archive():
    assert counts.fetch_ci_base_counts(_CHECKER, "abc123", gh=_gh_stub(_live_listing(), b"not a zip")) is None


def test_emit_writes_the_artifact_json_named_by_the_head_key(tmp_path, capsys):
    counts.emit_counts(_CHECKER, {"reportAny": 3, "aRule": 1}, tmp_path, "deadbeef")
    name = _CHECKER.artifact_name("deadbeef")
    assert json.loads((tmp_path / f"{name}.json").read_text()) == {
        "base_point": "deadbeef",
        "counts": {"aRule": 1, "reportAny": 3},
    }
    summary = capsys.readouterr().out
    assert "deadbeef" in summary
    assert name in summary
    assert "4" in summary


def test_emit_refuses_to_publish_empty_counts(tmp_path):
    with pytest.raises(SystemExit):
        counts.emit_counts(_CHECKER, {}, tmp_path, "deadbeef")
    assert list(tmp_path.iterdir()) == []


def test_emitted_file_is_the_one_the_fetcher_looks_up(tmp_path):
    written = counts.emit_counts(_CHECKER, {"reportAny": 3}, tmp_path, "deadbeef")
    payload = json.loads(written.read_text())
    assert counts.counts_for_base(payload, "deadbeef") == {"reportAny": 3}
    assert counts.counts_for_base(payload, "someothersha") is None
    listing_zip = _artifact_zip(payload)
    assert counts.fetch_ci_base_counts(_CHECKER, "deadbeef", gh=_gh_stub(_live_listing(), listing_zip)) == {
        "reportAny": 3
    }


def _git(cwd, *args):
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def _commit(cwd, name):
    (cwd / name).write_text(name)
    _git(cwd, "add", "-A")
    _git(cwd, "commit", "-q", "-m", name)
    return _git(cwd, "rev-parse", "HEAD")


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "gate@example.com")
    _git(repo, "config", "user.name", "gate")
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _branched_repo(tmp_path):
    repo = _init_repo(tmp_path)
    branch_point = _commit(repo, "shared.txt")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.txt")
    _git(repo, "checkout", "-q", "main")
    base_tip = _commit(repo, "drift.txt")
    _git(repo, "checkout", "-q", "feature")
    return repo, branch_point, base_tip


def test_base_point_is_the_branch_point_when_no_merge_is_in_progress(tmp_path):
    repo, branch_point, _ = _branched_repo(tmp_path)
    assert counts.resolve_base_point("main", cwd=repo) == branch_point


def test_base_point_mid_merge_advances_to_the_merged_in_base_tip(tmp_path):
    repo, _, base_tip = _branched_repo(tmp_path)
    _git(repo, "merge", "--no-commit", "--no-ff", "main")
    assert counts.resolve_base_point("main", cwd=repo) == base_tip


def test_base_point_mid_merge_of_an_older_side_branch_keeps_the_newer_branch_point(tmp_path):
    repo = _init_repo(tmp_path)
    _commit(repo, "shared.txt")
    _git(repo, "checkout", "-q", "-b", "old-side")
    _commit(repo, "old.txt")
    _git(repo, "checkout", "-q", "main")
    newer_point = _commit(repo, "drift.txt")
    _git(repo, "checkout", "-q", "-b", "feature")
    _commit(repo, "feature.txt")
    _git(repo, "merge", "--no-commit", "--no-ff", "old-side")
    assert counts.resolve_base_point("main", cwd=repo) == newer_point


def test_head_sha_is_the_checked_out_commit(tmp_path):
    repo, _, _ = _branched_repo(tmp_path)
    assert counts.head_sha(cwd=repo) == _git(repo, "rev-parse", "HEAD")


def test_default_cache_dir_lives_under_the_shared_git_dir(tmp_path):
    repo, _, _ = _branched_repo(tmp_path)
    assert counts.default_cache_dir(cwd=repo) == repo / ".git" / counts.CACHE_DIR_NAME
