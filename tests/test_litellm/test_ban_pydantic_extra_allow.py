"""Tests for the extra="allow" ban at tests/code_coverage_tests/ban_pydantic_extra_allow.py."""

import os
import subprocess
import sys

import pytest

_CODE_COVERAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_coverage_tests")
sys.path.insert(0, _CODE_COVERAGE_DIR)

import ban_pydantic_extra_allow as checker  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'class Foo(BaseModel):\n    model_config = ConfigDict(extra="allow")\n',
            id="config_dict_call",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    model_config = ConfigDict(protected_namespaces=(), extra="allow")\n',
            id="config_dict_call_with_other_kwargs",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    model_config = {"extra": "allow"}\n',
            id="plain_dict",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    model_config: ConfigDict = ConfigDict(extra="allow")\n',
            id="annotated_assignment",
        ),
        pytest.param(
            'class Foo(BaseModel, extra="allow"):\n    pass\n',
            id="class_keyword",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    class Config:\n        extra = "allow"\n',
            id="legacy_inner_config",
        ),
        pytest.param(
            "class Foo(BaseModel):\n    class Config:\n        extra = Extra.allow\n",
            id="legacy_inner_config_enum",
        ),
        pytest.param(
            'ALLOW = ConfigDict(extra="allow")\n\n\nclass Foo(BaseModel):\n    model_config = ALLOW\n',
            id="module_constant_config_dict",
        ),
        pytest.param(
            'ALLOW = {"extra": "allow"}\n\n\nclass Foo(BaseModel):\n    model_config = ALLOW\n',
            id="module_constant_dict",
        ),
        pytest.param(
            'ALLOW = {"extra": "allow"}\nALIAS = ALLOW\n\n\nclass Foo(BaseModel):\n    model_config = ALIAS\n',
            id="module_constant_chain",
        ),
        pytest.param(
            'EXTRA = "allow"\n\n\nclass Foo(BaseModel):\n    model_config = ConfigDict(extra=EXTRA)\n',
            id="module_constant_extra_value",
        ),
        pytest.param(
            'ALLOW = ConfigDict(extra="allow")\n\n\nclass Foo(BaseModel):\n'
            '    model_config = ALLOW\n\n\nALLOW = ConfigDict(extra="forbid")\n',
            id="constant_rebound_to_forbid_after_the_class",
        ),
        pytest.param(
            'ALLOW = ConfigDict(extra="forbid")\nALLOW = ConfigDict(extra="allow")\n\n\n'
            "class Foo(BaseModel):\n    model_config = ALLOW\n",
            id="constant_rebound_to_allow_above_the_class",
        ),
        pytest.param(
            'EXTRA = "allow"\n\n\nclass Foo(BaseModel):\n'
            '    model_config = ConfigDict(extra=EXTRA)\n\n\nEXTRA = "forbid"\n',
            id="extra_value_constant_rebound_after_the_class",
        ),
        pytest.param(
            'ALLOW = ConfigDict(extra="allow")\nALIAS = ALLOW\nALLOW = ConfigDict(extra="forbid")\n\n\n'
            "class Foo(BaseModel):\n    model_config = ALIAS\n",
            id="alias_captured_allow_before_its_source_was_rebound",
        ),
        pytest.param(
            'EXTRA = "allow"\n\n\nclass Foo(BaseModel, extra=EXTRA):\n    pass\n',
            id="module_constant_class_keyword",
        ),
        pytest.param(
            'if TYPE_CHECKING:\n    ALLOW = ConfigDict(extra="allow")\n\n\nclass Foo(BaseModel):\n'
            "    model_config = ALLOW\n",
            id="constant_bound_inside_an_if_block",
        ),
        pytest.param(
            'try:\n    ALLOW = ConfigDict(extra="allow")\nexcept ImportError:\n    ALLOW = None\n\n\n'
            "class Foo(BaseModel):\n    model_config = ALLOW\n",
            id="constant_bound_inside_a_try_block",
        ),
        pytest.param(
            'try:\n    class Foo(BaseModel):\n        model_config = ConfigDict(extra="allow")\n'
            "except ImportError:\n    pass\n",
            id="class_declared_inside_a_try_block",
        ),
        pytest.param(
            "if sys.version_info >= (3, 12):\n    class Foo(BaseModel):\n"
            '        model_config = ConfigDict(extra="allow")\n',
            id="class_declared_inside_an_if_block",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    if TYPE_CHECKING:\n        model_config = ConfigDict(extra="allow")\n',
            id="model_config_assigned_inside_an_if_block",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    class Config:\n        if TYPE_CHECKING:\n            extra = "allow"\n',
            id="legacy_inner_config_assigned_inside_an_if_block",
        ),
        pytest.param(
            'if IS_V2:\n    CONFIG = ConfigDict(extra="allow")\nelse:\n    CONFIG = ConfigDict(extra="forbid")\n\n\n'
            "class Foo(BaseModel):\n    model_config = CONFIG\n",
            id="one_branch_of_a_conditional_binding_allows",
        ),
    ],
)
def test_detects_extra_allow(source):
    violations = checker.find_violations_in_source(source, "litellm/types/thing.py")
    assert [violation.identifier() for violation in violations] == ["litellm/types/thing.py::Foo"]


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            'class Foo(BaseModel):\n    model_config = ConfigDict(extra="forbid")\n',
            id="extra_forbid",
        ),
        pytest.param(
            "class Foo(BaseModel):\n    model_config = ConfigDict(populate_by_name=True)\n",
            id="unrelated_config",
        ),
        pytest.param(
            "class Foo(BaseModel):\n    bar: str\n",
            id="no_config",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    """Docstring mentioning extra="allow"."""\n',
            id="docstring_mention_only",
        ),
        pytest.param(
            'def f():\n    return ConfigDict(extra="allow")\n',
            id="outside_class",
        ),
        pytest.param(
            'FORBID = ConfigDict(extra="forbid")\n\n\nclass Foo(BaseModel):\n    model_config = FORBID\n',
            id="module_constant_forbid",
        ),
        pytest.param(
            "class Foo(BaseModel):\n    model_config = IMPORTED_CONFIG\n",
            id="unresolvable_constant",
        ),
        pytest.param(
            "SELF = SELF\n\n\nclass Foo(BaseModel):\n    model_config = SELF\n",
            id="self_referencing_constant",
        ),
        pytest.param(
            'ALLOW = ConfigDict(extra="forbid")\n\n\nclass Foo(BaseModel):\n'
            '    model_config = ALLOW\n\n\nALLOW = ConfigDict(extra="allow")\n',
            id="constant_only_becomes_allow_below_the_class",
        ),
        pytest.param(
            'ALLOW = ConfigDict(extra="forbid")\nALIAS = ALLOW\nALLOW = ConfigDict(extra="allow")\n\n\n'
            "class Foo(BaseModel):\n    model_config = ALIAS\n",
            id="alias_captured_forbid_before_its_source_was_rebound",
        ),
        pytest.param(
            'if IS_V2:\n    CONFIG = ConfigDict(extra="allow")\nCONFIG = ConfigDict(extra="forbid")\n\n\n'
            "class Foo(BaseModel):\n    model_config = CONFIG\n",
            id="conditional_allow_rebound_unconditionally_before_the_class",
        ),
    ],
)
def test_ignores_non_violations(source):
    assert checker.find_violations_in_source(source, "litellm/types/thing.py") == ()


def test_reports_nested_class_with_qualified_name():
    source = 'class Outer:\n    class Inner(BaseModel):\n        model_config = ConfigDict(extra="allow")\n'
    violations = checker.find_violations_in_source(source, "litellm/types/thing.py")
    assert [violation.identifier() for violation in violations] == ["litellm/types/thing.py::Outer.Inner"]


def test_grandfathered_list_matches_repo():
    """Every grandfathered entry must still exist, and nothing new may be added."""
    found = frozenset(violation.identifier() for violation in checker.find_extra_allow_models(_REPO_ROOT))
    assert sorted(found - checker.GRANDFATHERED) == []
    assert sorted(checker.GRANDFATHERED - found) == []


def test_parse_grandfathered_reads_the_list_of_another_revision():
    source = 'OTHER = frozenset({"nope"})\nGRANDFATHERED = frozenset(\n    {\n        "a.py::A",\n        "b.py::B",\n    }\n)\n'
    assert checker.parse_grandfathered(source) == frozenset({"a.py::A", "b.py::B"})


def test_parse_grandfathered_reads_this_files_own_list():
    with open(os.path.join(_REPO_ROOT, checker.SELF_PATH), encoding="utf-8") as handle:
        assert checker.parse_grandfathered(handle.read()) == checker.GRANDFATHERED


def _baseline_commit(repo, entries):
    """A one-commit repo whose copy of the checker grandfathers exactly ``entries``."""
    target = os.path.join(repo, checker.SELF_PATH)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("GRANDFATHERED = frozenset(\n    {\n")
        handle.writelines(f'        "{entry}",\n' for entry in entries)
        handle.write("    }\n)\n")
    for command in (
        ["git", "init", "-q", "."],
        ["git", "add", checker.SELF_PATH],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "baseline"],
    ):
        subprocess.run(command, cwd=repo, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.mark.parametrize(
    "baseline, expected",
    [
        pytest.param(checker.GRANDFATHERED, [], id="unchanged_list_passes"),
        pytest.param(
            checker.GRANDFATHERED | {"litellm/types/gone.py::Gone"},
            [],
            id="removing_an_entry_passes",
        ),
        pytest.param(
            checker.GRANDFATHERED - {"litellm/types/utils.py::ImageResponse"},
            ["litellm/types/utils.py::ImageResponse"],
            id="adding_an_entry_is_reported",
        ),
    ],
)
def test_added_grandfathered_only_reports_growth(tmp_path, monkeypatch, baseline, expected):
    base_sha = _baseline_commit(str(tmp_path), sorted(baseline))
    monkeypatch.chdir(tmp_path)
    assert list(checker.added_grandfathered(base_sha)) == expected


def test_added_grandfathered_skips_a_base_without_the_checker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert checker.added_grandfathered("HEAD") == ()
