"""Tests for .github/scripts/assert_ci_coverage.py.

Three guards share one workflow parser. The census asks whether a test file is run at
all, so an ancestor path standing in for everything below it is a valid answer. The
shard guard asks whether a sharded tree, which has no catch-all bucket, names each
child outright, so that same ancestor path must NOT be an answer. The slice guard asks
the question neither covers: whether the job that globs a file then deselects it with
`-k`, which is how a file counts as covered while running nowhere.
"""

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / ".github" / "scripts" / "assert_ci_coverage.py"
_spec = importlib.util.spec_from_file_location("assert_ci_coverage", _MODULE_PATH)
coverage = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = coverage  # @dataclass(slots=True) rebuilds via sys.modules
_spec.loader.exec_module(coverage)


def test_an_ancestor_directory_covers_a_file_but_does_not_name_it():
    # The whole point of the split: `tests/x` answers "does it run?" but not
    # "which shard owns it?" — accepting it for the latter is how a new child
    # silently drops out of a tree that has no catch-all bucket.
    assert coverage._token_covers("tests/test_litellm", "tests/test_litellm/caching") is True
    assert coverage._token_names("tests/test_litellm", "tests/test_litellm/caching") is False


def test_an_exact_token_both_covers_and_names():
    assert coverage._token_covers("tests/test_litellm/caching", "tests/test_litellm/caching") is True
    assert coverage._token_names("tests/test_litellm/caching", "tests/test_litellm/caching") is True


def test_a_glob_names_only_what_it_matches_not_what_sits_below_it():
    glob = "tests/test_litellm/test_*.py"
    assert coverage._token_names(glob, "tests/test_litellm/test_router.py") is True
    assert coverage._token_names(glob, "tests/test_litellm/test_router.py/nested.py") is False
    assert coverage._token_names(glob, "tests/test_litellm/proxy/test_router.py") is False


def test_a_glob_still_covers_the_subtree_for_the_census():
    assert coverage._token_covers("tests/llm_translation/**", "tests/llm_translation/a/b.py") is True


def test_a_directory_earns_a_shard_by_holding_tests_not_by_its_name(tmp_path):
    fixtures = tmp_path / "expected_payloads"
    fixtures.mkdir()
    (fixtures / "body.json").write_text("{}")
    tests = tmp_path / "some_area"
    tests.mkdir()
    (tests / "test_thing.py").write_text("def test_thing(): assert True\n")

    assert coverage._holds_tests(fixtures) is False
    assert coverage._holds_tests(tests) is True


def test_shard_children_lists_test_dirs_and_test_files_and_skips_fixture_dirs(tmp_path):
    root = tmp_path / "tests" / "tree"
    (root / "billing").mkdir(parents=True)
    (root / "billing" / "test_billing.py").write_text("def test_b(): assert True\n")
    (root / "test_configs").mkdir()
    (root / "test_configs" / "config.yaml").write_text("model_list: []\n")
    (root / "test_top_level.py").write_text("def test_t(): assert True\n")
    (root / "helpers.py").write_text("VALUE = 1\n")

    assert coverage._shard_children("tests/tree", tmp_path) == (
        "tests/tree/billing",
        "tests/tree/test_top_level.py",
    )


def test_an_unnamed_child_is_reported_and_a_named_one_is_not(tmp_path):
    root = tmp_path / "tests" / "tree"
    (root / "claimed").mkdir(parents=True)
    (root / "claimed" / "test_a.py").write_text("def test_a(): assert True\n")
    (root / "orphan").mkdir()
    (root / "orphan" / "test_b.py").write_text("def test_b(): assert True\n")

    findings = coverage._unassigned_shard_children(
        frozenset({"tests/tree/claimed"}), roots=("tests/tree",), repo_root=tmp_path
    )

    assert tuple(f.subject for f in findings) == ("tests/tree/orphan",)


def test_the_parent_token_alone_does_not_satisfy_any_child(tmp_path):
    root = tmp_path / "tests" / "tree"
    (root / "billing").mkdir(parents=True)
    (root / "billing" / "test_a.py").write_text("def test_a(): assert True\n")

    findings = coverage._unassigned_shard_children(
        frozenset({"tests/tree"}), roots=("tests/tree",), repo_root=tmp_path
    )

    assert tuple(f.subject for f in findings) == ("tests/tree/billing",)


def test_a_child_that_is_itself_a_sharded_root_is_checked_there_not_here(tmp_path):
    root = tmp_path / "tests" / "tree"
    (root / "proxy" / "endpoints").mkdir(parents=True)
    (root / "proxy" / "endpoints" / "test_a.py").write_text("def test_a(): assert True\n")

    findings = coverage._unassigned_shard_children(
        frozenset(), roots=("tests/tree", "tests/tree/proxy"), repo_root=tmp_path
    )

    assert tuple(f.subject for f in findings) == ("tests/tree/proxy/endpoints",)


def test_every_sharded_root_named_in_the_script_exists_on_disk():
    # A stale root would make the guard pass by checking nothing.
    missing = [root for root in coverage.SHARDED_ROOTS if not (_REPO_ROOT / root).is_dir()]
    assert missing == []


def test_the_repo_as_it_stands_has_every_shard_child_assigned():
    findings = coverage._unassigned_shard_children(coverage._invoked_test_tokens(coverage._all_scalars()))
    assert [f.subject for f in findings] == []


# --------------------------------------------------------------------------- #
# Slice guard: a job can glob a file and its -k can then throw the file out
# --------------------------------------------------------------------------- #


def _slice(**overrides):
    defaults = dict(
        job="a_job", globs=("tests/x/**/test_*.py",), named=frozenset(),
        required=(), excluded=(), understood=True,
    )
    return coverage.Slice(**{**defaults, **overrides})


def test_a_term_in_the_path_deselects_the_whole_file():
    # -k matches the module's path as well as the names inside it, so "not caching"
    # removes every test in test_caching.py, not merely the ones named for a cache.
    slice_ = _slice(excluded=("caching",))
    assert slice_.claims("tests/x/test_caching.py", frozenset({"test_get"})) is False
    assert slice_.claims("tests/x/test_router.py", frozenset({"test_get"})) is True


def test_matching_is_substring_not_word_so_cache_and_caching_are_different_terms():
    # The real config excludes both, because "cache" does not occur inside "caching";
    # collapsing them to one term would quietly let a whole file back in.
    assert _slice(excluded=("cache",)).claims("tests/x/test_caching.py", frozenset()) is True
    assert _slice(excluded=("cache",)).claims("tests/x/test_dual_cache.py", frozenset()) is False


def test_a_positive_term_can_be_satisfied_by_a_name_inside_the_file():
    # A job running -k "langfuse" claims test_logging.py when a test inside is named
    # for langfuse, so treating the path alone as the match would report a false gap.
    slice_ = _slice(required=("langfuse",))
    assert slice_.claims("tests/x/test_logging.py", frozenset({"test_langfuse_emits"})) is True
    assert slice_.claims("tests/x/test_logging.py", frozenset({"test_datadog_emits"})) is False


def test_a_file_the_job_never_globs_is_not_its_problem():
    assert _slice().claims("tests/other/test_a.py", frozenset()) is False


def test_an_explicitly_named_file_is_claimed_whatever_the_keywords_say():
    # redis_caching_unit_tests names test_dual_cache.py outright, which is what keeps
    # that file out of the report even though every -k in the globbing jobs drops it.
    slice_ = _slice(globs=(), named=frozenset({"tests/x/test_dual_cache.py"}), excluded=("cache",))
    assert slice_.claims("tests/x/test_dual_cache.py", frozenset()) is True


def test_an_unparsed_keyword_expression_claims_everything_it_globs():
    # Staying silent beats guessing: an expression this parser cannot model must never
    # be the reason a file is reported as unrun.
    assert _slice(understood=False, excluded=("cache",)).claims(
        "tests/x/test_caching.py", frozenset()
    ) is True


def test_keyword_terms_splits_an_and_chain_into_required_and_excluded():
    required, excluded, understood = coverage._keyword_terms(("langfuse and not cache and not router",))
    assert (required, excluded, understood) == (("langfuse",), ("cache", "router"), True)


def test_keyword_terms_refuses_to_model_an_or_expression():
    assert coverage._keyword_terms(("cache or router",)) == ((), (), False)


def test_keyword_terms_refuses_to_attribute_a_selector_across_several_commands():
    # A job running two pytest commands offers no way to tell which glob a -k belongs
    # to, and pairing one command's exclusion with the other's glob would invent a gap.
    assert coverage._keyword_terms(("not cache",), attributable=False) == ((), (), False)
    assert coverage._keyword_terms((), attributable=False) == ((), (), True)


def test_an_excluded_term_matching_only_an_inner_name_leaves_the_file_claimed():
    # -k "not cache" drops test_cache_key inside test_router.py and keeps the rest, so
    # the file still runs. Reporting it would be a false alarm; the guard is per-file.
    slice_ = _slice(excluded=("cache",))
    assert slice_.claims("tests/x/test_router.py", frozenset({"test_cache_key"})) is True


def test_character_class_globs_match_the_letter_shards_circleci_uses():
    # tests/local_testing is split by first letter; without character-class support every
    # file in it looks unglobbed, and the slice guard would report the whole directory.
    glob = "tests/local_testing/**/test_[a-mA-M]*.py"
    assert coverage._token_covers(glob, "tests/local_testing/test_caching.py") is True
    assert coverage._token_covers(glob, "tests/local_testing/test_router.py") is False


def test_the_repo_as_it_stands_has_no_unrecorded_slice_gap():
    findings = coverage._deselected_everywhere(coverage._load_allowlist())
    assert [f.subject for f in findings] == []


def _allowlist(*, tests: tuple[str, ...] = (), dockerfiles: tuple[str, ...] = ()):
    return coverage.Allowlist(
        test_paths=(coverage.AllowEntry(paths=tests, reason="r"),) if tests else (),
        dockerfiles=(coverage.AllowEntry(paths=dockerfiles, reason="r"),) if dockerfiles else (),
    )


def test_an_allowlist_path_whose_file_is_gone_is_reported():
    findings = coverage._stale_allowlist_paths(
        _allowlist(tests=("tests/gone/test_a.py",)),
        test_files=("tests/live/test_b.py",),
        dockerfiles=(),
    )
    assert [f.subject for f in findings] == ["tests/gone/test_a.py"]
    assert "test_paths" in findings[0].detail


def test_an_allowlist_path_that_still_matches_a_file_is_left_alone():
    findings = coverage._stale_allowlist_paths(
        _allowlist(tests=("tests/live/test_b.py",)),
        test_files=("tests/live/test_b.py",),
        dockerfiles=(),
    )
    assert findings == ()


def test_a_directory_entry_survives_while_any_file_below_it_remains():
    findings = coverage._stale_allowlist_paths(
        _allowlist(tests=("tests/live",)),
        test_files=("tests/live/nested/test_b.py",),
        dockerfiles=(),
    )
    assert findings == ()


def test_a_glob_entry_matching_nothing_is_reported_like_any_other():
    findings = coverage._stale_allowlist_paths(
        _allowlist(tests=("tests/live/test_z*.py",)),
        test_files=("tests/live/test_b.py",),
        dockerfiles=(),
    )
    assert [f.subject for f in findings] == ["tests/live/test_z*.py"]


def test_a_stale_dockerfile_entry_is_named_under_its_own_section():
    findings = coverage._stale_allowlist_paths(
        _allowlist(dockerfiles=("docker/Dockerfile.gone",)),
        test_files=(),
        dockerfiles=("docker/Dockerfile.database",),
    )
    assert [(f.subject, "dockerfiles" in f.detail) for f in findings] == [("docker/Dockerfile.gone", True)]


def test_the_repo_as_it_stands_has_no_stale_allowlist_entry():
    findings = coverage._stale_allowlist_paths(
        coverage._load_allowlist(),
        test_files=coverage._test_files(),
        dockerfiles=coverage._dockerfiles(),
    )
    assert [f.subject for f in findings] == []


def test_a_dockerfile_directory_entry_is_stale_because_only_an_exact_path_exempts_one():
    findings = coverage._stale_allowlist_paths(
        _allowlist(dockerfiles=("docker",)),
        test_files=(),
        dockerfiles=("docker/Dockerfile.database",),
    )
    assert [f.subject for f in findings] == ["docker"]


def test_a_workflow_that_names_a_file_clears_it_from_the_slice_check():
    named = coverage._workflow_named_tokens()
    assert named, "the workflows must name some test paths or the check proves nothing"
    assert any(
        coverage._token_covers(token, "tests/local_testing/test_caching_handler.py")
        for token in named
    )


def test_the_slice_check_credits_only_workflows_never_the_circleci_config():
    named = coverage._workflow_named_tokens()
    circleci_only = "tests/proxy_admin_ui_tests"
    assert not any(coverage._token_covers(token, f"{circleci_only}/test_key_management.py") for token in named), (
        "a tree only CircleCI globs must not be credited to a workflow"
    )


def test_a_file_no_workflow_names_is_still_reported_when_every_slice_drops_it():
    named = coverage._workflow_named_tokens()
    assert not any(
        coverage._token_covers(token, "tests/local_testing/test_caching.py") for token in named
    ), "test_caching.py is allowlisted, not run; crediting it would hide a real gap"
