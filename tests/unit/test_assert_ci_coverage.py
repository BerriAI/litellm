"""Tests for .github/scripts/assert_ci_coverage.py.

Three guards share one workflow parser. The census asks whether a test file is run at
all, so an ancestor path standing in for everything below it is a valid answer. The
shard guard asks whether a sharded tree, which has no catch-all bucket, names each
child outright, so that same ancestor path must NOT be an answer. The slice guard asks
the question neither covers: whether the job that globs a file then deselects it with
`-k`, which is how a file counts as covered while running nowhere.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / ".github" / "scripts" / "assert_ci_coverage.py"
_spec = importlib.util.spec_from_file_location("assert_ci_coverage", _MODULE_PATH)
coverage = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = coverage  # @dataclass(slots=True) rebuilds via sys.modules
_spec.loader.exec_module(coverage)


def _write_group_workflow(root: Path, suites: tuple[str, ...] = ("management",)) -> Path:
    workflow: Final = root / ".github/workflows/test-integration.yml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text(
        yaml.safe_dump(
            {
                "jobs": {
                    "integration": {
                        "strategy": {"matrix": {"suite": list(suites)}},
                        "steps": [{"run": "bash .github/scripts/ci/run_integration.sh $SUITE"}],
                    }
                }
            }
        )
    )
    return workflow


def test_integration_groups_require_exclusive_scheduled_github_owner(tmp_path: Path) -> None:
    test_path: Final = "tests/integration/management/test_contract.py"
    test_file: Final = tmp_path / test_path
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_contract(): pass\n")
    (tmp_path / "tests/integration/run.py").write_text(
        "from types import MappingProxyType\nfrom typing import Final\n"
        'GROUPS: Final = MappingProxyType({"management": ("management",)})\n'
    )
    paths, findings = coverage._integration_ownership(tmp_path)
    assert not paths
    assert [finding.detail for finding in findings] == ["dedicated GitHub Actions runner is missing"]
    _write_group_workflow(tmp_path)
    paths, findings = coverage._integration_ownership(tmp_path)
    assert paths == frozenset({test_path})
    assert findings == ()
    _write_group_workflow(tmp_path, ("providers",))
    _, findings = coverage._integration_ownership(tmp_path)
    assert [(finding.subject, finding.detail) for finding in findings] == [
        ("management", "canonical integration group is not scheduled by GitHub Actions")
    ]
    _write_group_workflow(tmp_path)
    workflow: Final = tmp_path / ".github/workflows/test.yml"
    workflow.write_text(yaml.safe_dump({"jobs": {"tests": {"steps": [{"run": "pytest tests/integration"}]}}}))
    _, findings = coverage._integration_ownership(tmp_path)
    assert [(finding.subject, finding.detail) for finding in findings] == [
        (test_path, "integration file outside GITHUB_FILES is selected by GitHub Actions")
    ]
    github_path: Final = "tests/integration/database/image/test_github_contract.py"
    (tmp_path / github_path).parent.mkdir(parents=True)
    (tmp_path / github_path).write_text("def test_contract(): pass\n")
    runner: Final = tmp_path / "tests/integration/run.py"
    runner.write_text(runner.read_text() + f"GITHUB_FILES: Final = frozenset({{{github_path!r}}})\n")
    workflow.unlink()
    dedicated: Final = tmp_path / ".github/workflows/image-scan.yml"
    dedicated.write_text(yaml.safe_dump({"jobs": {"tests": {"steps": [{"run": f"pytest {github_path}"}]}}}))
    github_owned, github_findings = coverage._integration_ownership(tmp_path)
    assert github_owned == frozenset({test_path, github_path})
    assert github_findings == ()
    per_pr: Final = tmp_path / ".github/workflows/test-mcp-interactions.yml"
    per_pr.write_text(yaml.safe_dump({"jobs": {"mcp": {"steps": [{"run": f"pytest {github_path}"}]}}}))
    _, shared_findings = coverage._integration_ownership(tmp_path)
    assert [(finding.subject, finding.detail) for finding in shared_findings] == [
        (github_path, "GitHub-owned integration contract runs outside a dedicated integration workflow")
    ]
    per_pr.unlink()
    dedicated.write_text(yaml.safe_dump({"jobs": {}}))
    _, missing_invocation = coverage._integration_ownership(tmp_path)
    assert [(finding.subject, finding.detail) for finding in missing_invocation] == [
        (github_path, "GitHub-owned integration contract has no invoking job")
    ]
    dedicated.write_text(yaml.safe_dump({"jobs": {"tests": {"steps": [{"run": f"pytest {github_path}"}]}}}))
    _, restored_invocation = coverage._integration_ownership(tmp_path)
    assert restored_invocation == ()


def _write_integration_tree(root: Path, github_path: str) -> None:
    contracts: Final = root / "tests/integration/management"
    contracts.mkdir(parents=True)
    (contracts / "test_contract.py").write_text("def test_contract(): pass\n")
    (root / github_path).parent.mkdir(parents=True, exist_ok=True)
    (root / github_path).write_text("def test_contract(): pass\n")
    (root / "tests/integration/run.py").write_text(
        "from types import MappingProxyType\nfrom typing import Final\n"
        'GROUPS: Final = MappingProxyType({"management": ("management",)})\n'
        f"GITHUB_FILES: Final = frozenset({{{github_path!r}}})\n"
    )
    _write_group_workflow(root)


@pytest.mark.parametrize(
    ("matrix", "expected"),
    (
        (["management", "database"], ("management", "database")),
        ([], ()),
        (None, ()),
        ("${{ inputs.suite }}", ()),
        (
            "${{ fromJSON(inputs.suite == 'all' && '[\"management\",\"database\"]' || format('[\"{0}\"]', inputs.suite)) }}",
            ("management", "database"),
        ),
    ),
)
def test_integration_matrix_accounts_for_default_suites(matrix: object, expected: tuple[str, ...]) -> None:
    assert coverage._matrix_suites(matrix) == expected


@pytest.mark.parametrize(
    ("github_path", "workflow_name", "exit_code"),
    [
        ("tests/integration/database/image/test_github_contract.py", "image-scan.yml", 0),
        ("tests/integration/database/image/test_github_contract.py", "test-unit.yml", 1),
        ("tests/integration/management/test_github_contract.py", "image-scan.yml", 1),
    ],
)
def test_census_fails_when_a_github_owned_contract_runs_outside_its_dedicated_workflow(
    tmp_path: Path, github_path: str, workflow_name: str, exit_code: int
) -> None:
    script: Final = tmp_path / ".github/scripts/assert_ci_coverage.py"
    script.parent.mkdir(parents=True)
    shutil.copy(_MODULE_PATH, script)
    _write_integration_tree(tmp_path, github_path)
    (tmp_path / ".github/workflows" / workflow_name).write_text(
        yaml.safe_dump({"jobs": {"tests": {"steps": [{"run": f"pytest {github_path}"}]}}})
    )
    result: Final = subprocess.run([sys.executable, "-I", str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == exit_code, result.stdout
    assert "Playwright" not in result.stdout


def test_an_integration_file_outside_every_group_cannot_run_on_a_github_shard(tmp_path: Path) -> None:
    _write_integration_tree(tmp_path, "tests/integration/database/image/test_github_contract.py")
    ungrouped: Final = "tests/integration/ci_probe/test_ci_probe.py"
    (tmp_path / ungrouped).parent.mkdir(parents=True)
    (tmp_path / ungrouped).write_text("def test_probe(): pass\n")
    _, before = coverage._integration_ownership(tmp_path)
    (tmp_path / ".github/workflows/test-unit.yml").write_text(
        yaml.safe_dump({"jobs": {"mcp": {"steps": [{"run": f"pytest {ungrouped}"}]}}})
    )
    _, after = coverage._integration_ownership(tmp_path)
    assert [(finding.subject, finding.detail) for finding in after if finding.subject == ungrouped] == [
        (ungrouped, "integration file outside GITHUB_FILES is selected by GitHub Actions")
    ]
    assert not [finding for finding in before if finding.subject == ungrouped]


def test_workflow_matrices_count_their_test_paths_as_invoked() -> None:
    config: Final = {
        "jobs": {
            "postgres": {
                "strategy": {
                    "matrix": {
                        "include": [
                            {"test-path": "tests/proxy_behavior"},
                            {"test-path": "tests/integration/database/test_roi_observed.py"},
                        ]
                    }
                }
            }
        }
    }
    assert coverage._invoked_test_tokens(coverage._scalars(config, "config.yml")) == frozenset(
        {"tests/proxy_behavior", "tests/integration/database/test_roi_observed.py"}
    )


def test_allowlisting_an_unrun_integration_contract_does_not_claim_it_runs(tmp_path: Path) -> None:
    path: Final = "tests/integration/database/test_roi_observed.py"
    _write_integration_tree(tmp_path, path)
    paths, findings = coverage._integration_ownership(tmp_path)
    assert path not in paths
    assert [(finding.subject, finding.detail) for finding in findings] == [
        (path, "GitHub-owned integration contract has no invoking job")
    ]
    allowlist: Final = coverage.Allowlist(
        test_paths=(coverage.AllowEntry(paths=(path,), reason="ran only in CircleCI, removed"),), dockerfiles=()
    )
    allowed_paths, allowed_findings = coverage._integration_ownership(tmp_path, allowlist=allowlist)
    assert path not in allowed_paths
    assert allowed_findings == ()
    (tmp_path / ".github/workflows/test-unit.yml").write_text(
        yaml.safe_dump({"jobs": {"tests": {"steps": [{"run": f"pytest {path}"}]}}})
    )
    _, wrong_owner = coverage._integration_ownership(tmp_path, allowlist=allowlist)
    assert [(finding.subject, finding.detail) for finding in wrong_owner] == [
        (path, "GitHub-owned integration contract runs outside a dedicated integration workflow")
    ]


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
    assert coverage._token_names(glob, "tests/test_litellm/nested/test_router.py") is False


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

    findings = coverage._unassigned_shard_children(frozenset({"tests/tree"}), roots=("tests/tree",), repo_root=tmp_path)

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
    findings = coverage._unassigned_shard_children(
        coverage._invoked_test_tokens(coverage._all_scalars())
    )
    assert [f.subject for f in findings] == []


def test_shards_are_credited_only_from_explicit_test_paths() -> None:
    scalars: Final = (
        coverage.Scalar(key="test-path", value="tests/unit/wired\ntests/unit/also_wired"),
        coverage.Scalar(key="shard", value="tests/unit/not_a_test_path"),
    )
    assert coverage._invoked_test_tokens(scalars) == frozenset({"tests/unit/wired", "tests/unit/also_wired"})


def test_an_ignored_path_is_not_credited_as_invoked() -> None:
    scalars: Final = (
        coverage.Scalar(key="test-path", value="tests/unit/a\n--ignore=tests/unit/b/test_x.py"),
    )
    assert coverage._invoked_test_tokens(scalars) == frozenset({"tests/unit/a"})


def test_a_file_its_only_shard_ignores_is_not_covered_by_that_shards_glob() -> None:
    selections: Final = coverage._invoked_selections(
        (
            coverage.Scalar(
                key="test-path",
                value="tests/unit/proxy/test_*.py --ignore=tests/unit/proxy/test_update_spend.py",
            ),
        )
    )
    assert len(selections) == 1
    selection: Final = selections[0]
    assert selection.covers("tests/unit/proxy/test_other.py") is True
    assert selection.covers("tests/unit/proxy/test_update_spend.py") is False


def test_a_deselected_file_is_treated_like_an_ignored_one() -> None:
    # --ignore is a no-op for a file the shell glob already expanded, --deselect is what drops it
    scalars: Final = (
        coverage.Scalar(
            key="test-path",
            value="tests/unit/proxy/test_[q-z]*.py\n--deselect=tests/unit/proxy/test_update_spend.py",
        ),
    )
    assert coverage._invoked_test_tokens(scalars) == frozenset({"tests/unit/proxy/test_[q-z]*.py"})
    (selection,) = coverage._invoked_selections(scalars)
    assert selection.covers("tests/unit/proxy/test_route_llm_request.py") is True
    assert selection.covers("tests/unit/proxy/test_update_spend.py") is False


def test_a_character_class_glob_in_a_test_path_is_one_token_that_names_only_its_letters() -> None:
    (selection,) = coverage._invoked_selections(
        (coverage.Scalar(key="test-path", value="tests/unit/proxy/test_[_a-p]*.py"),)
    )
    assert selection.included == frozenset({"tests/unit/proxy/test_[_a-p]*.py"})
    assert selection.covers("tests/unit/proxy/test__types.py") is True
    assert selection.covers("tests/unit/proxy/test_caching_routes.py") is True
    assert selection.covers("tests/unit/proxy/test_route_llm_request.py") is False
    assert coverage._token_names("tests/unit/proxy/test_[_a-p]*.py", "tests/unit/proxy/test_admin_mcp.py") is True
    assert coverage._token_names("tests/unit/proxy/test_[_a-p]*.py", "tests/unit/proxy/test_route_llm_request.py") is False


def test_check_shards_passes_on_the_repo_as_it_stands(capsys):
    assert coverage._check_shards() == 0


# --------------------------------------------------------------------------- #
# Slice guard: a job can glob a file and its -k can then throw the file out
# --------------------------------------------------------------------------- #


def _slice(**overrides):
    defaults = dict(
        job="a_job",
        globs=("tests/x/**/test_*.py",),
        named=frozenset(),
        required=(),
        excluded=(),
        understood=True,
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
    assert _slice(understood=False, excluded=("cache",)).claims("tests/x/test_caching.py", frozenset()) is True


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


def test_character_class_globs_match_letter_shards():
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
    assert any(coverage._token_covers(token, "tests/unit/proxy/test_proxy_server_custom_logger.py") for token in named)


def test_the_slice_check_ignores_configs_outside_the_workflow_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflows: Final = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "test.yml").write_text("jobs:\n  test:\n    steps:\n      - run: pytest tests/selected\n")
    (tmp_path / "ignored.yml").write_text("jobs:\n  test:\n    steps:\n      - run: pytest tests/unrun\n")
    monkeypatch.setattr(coverage, "WORKFLOW_DIR", workflows)
    assert coverage._workflow_named_tokens() == frozenset({"tests/selected"})


@pytest.mark.parametrize("selector", ("test_selected.py", "test_selected.py::test_redis_auth"))
def test_a_workflow_does_not_credit_a_file_it_never_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selector: str
) -> None:
    workflows: Final = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "test.yml").write_text(
        f"jobs:\n  test:\n    steps:\n      - run: uv run pytest tests/local_testing/{selector}\n"
    )
    monkeypatch.setattr(coverage, "WORKFLOW_DIR", workflows)

    named: Final = coverage._workflow_named_tokens()
    assert named == frozenset({"tests/local_testing/test_selected.py"})
    assert not any(
        coverage._token_covers(token, "tests/local_testing/test_unrun.py") for token in named
    )


def test_slice_audit_rechecks_changed_and_new_files_between_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workflows: Final = tmp_path / ".github/workflows"
    workflows.mkdir(parents=True)
    tests: Final = tmp_path / "tests/local_testing"
    tests.mkdir(parents=True)
    (workflows / "test.yml").write_text(
        yaml.safe_dump(
            {
                "jobs": {
                    selector: {
                        "steps": [
                            {
                                "run": f'pytest tests/local_testing/test_*.py -k "{selector}"'
                            }
                        ]
                    }
                    for selector in ("selected", "neighbor")
                }
            }
        )
    )
    monkeypatch.setattr(coverage, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(coverage, "TESTS_ROOT", tmp_path / "tests")
    monkeypatch.setattr(coverage, "WORKFLOW_DIR", workflows)
    allowlist: Final = coverage.Allowlist((), ())
    existing: Final = tests / "test_existing.py"
    existing.write_text("def test_selected(): pass\n")
    assert coverage._deselected_everywhere(allowlist) == ()

    existing.write_text("def test_uncovered(): pass\n")
    assert tuple(finding.subject for finding in coverage._deselected_everywhere(allowlist)) == (
        "tests/local_testing/test_existing.py",
    )

    existing.write_text("def test_selected(): pass\n")
    added: Final = tests / "test_added.py"
    added.write_text("def test_uncovered(): pass\n")
    assert tuple(finding.subject for finding in coverage._deselected_everywhere(allowlist)) == (
        "tests/local_testing/test_added.py",
    )
