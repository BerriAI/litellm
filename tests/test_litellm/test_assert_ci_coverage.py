"""Tests for .github/scripts/assert_ci_coverage.py.

Two guards share one workflow parser. The census asks whether a test file is run at
all, so an ancestor path standing in for everything below it is a valid answer. The
shard guard asks whether a sharded tree, which has no catch-all bucket, names each
child outright, so that same ancestor path must NOT be an answer. The pair of
matchers that splits those two questions is what these tests pin.
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
