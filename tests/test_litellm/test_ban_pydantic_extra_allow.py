"""Tests for the extra="allow" ban at tests/code_coverage_tests/ban_pydantic_extra_allow.py."""

import importlib.util
import json
import os
import sys

import pytest

_CODE_COVERAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_coverage_tests")
sys.path.insert(0, _CODE_COVERAGE_DIR)

import ban_pydantic_extra_allow as checker  # noqa: E402

_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

_RATCHET_PATH = os.path.join(_REPO_ROOT, "scripts", "budget_ratchet_check.py")
_ratchet_spec = importlib.util.spec_from_file_location("budget_ratchet_check", _RATCHET_PATH)
ratchet = importlib.util.module_from_spec(_ratchet_spec)
_ratchet_spec.loader.exec_module(ratchet)


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
        pytest.param(
            'class Foo(BaseModel):\n    _CONFIG = ConfigDict(extra="allow")\n    model_config = _CONFIG\n',
            id="class_local_constant",
        ),
        pytest.param(
            'FORBID = ConfigDict(extra="forbid")\n\n\nclass Foo(BaseModel):\n'
            '    FORBID = ConfigDict(extra="allow")\n    model_config = FORBID\n',
            id="class_local_constant_shadowing_a_harmless_module_one",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    class Config:\n        _EXTRA = "allow"\n        extra = _EXTRA\n',
            id="legacy_inner_config_reading_its_own_constant",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    model_config = ConfigDict(**{"extra": "allow"})\n',
            id="config_dict_kwargs_spread",
        ),
        pytest.param(
            'ALLOW = ConfigDict(extra="allow")\n\n\nclass Foo(BaseModel):\n    model_config = {**ALLOW}\n',
            id="permissive_constant_spread_into_a_dict",
        ),
        pytest.param(
            'ALLOW = {"extra": "allow"}\n\n\nclass Foo(BaseModel, **ALLOW):\n    pass\n',
            id="permissive_constant_spread_into_the_class_keywords",
        ),
        pytest.param(
            'ALLOW = ConfigDict(extra="allow")\n\n\nclass Foo(BaseModel):\n'
            '    model_config = ALLOW\n    ALLOW = ConfigDict(extra="forbid")\n',
            id="module_constant_read_above_a_class_local_of_the_same_name",
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
        pytest.param(
            'ALLOW = ConfigDict(extra="allow")\n\n\nclass Foo(BaseModel):\n'
            '    ALLOW = ConfigDict(extra="forbid")\n    model_config = ALLOW\n',
            id="class_local_constant_shadowing_a_permissive_module_one",
        ),
        pytest.param(
            'class Foo(BaseModel):\n    model_config = LATER\n    LATER = ConfigDict(extra="allow")\n',
            id="class_local_constant_bound_below_the_line_reading_it",
        ),
        pytest.param(
            'FORBID = ConfigDict(extra="forbid")\n\n\nclass Foo(BaseModel):\n    model_config = {**FORBID}\n',
            id="harmless_constant_spread_into_a_dict",
        ),
    ],
)
def test_ignores_non_violations(source):
    assert checker.find_violations_in_source(source, "litellm/types/thing.py") == ()


def test_reports_nested_class_with_qualified_name():
    source = 'class Outer:\n    class Inner(BaseModel):\n        model_config = ConfigDict(extra="allow")\n'
    violations = checker.find_violations_in_source(source, "litellm/types/thing.py")
    assert [violation.identifier() for violation in violations] == ["litellm/types/thing.py::Outer.Inner"]


def test_budget_matches_repo():
    """Every grandfathered model must still exist, and nothing new may be added."""
    budget = checker.read_budget(os.path.join(_REPO_ROOT, checker.BUDGET_PATH))
    found = frozenset(violation.identifier() for violation in checker.find_extra_allow_models(_REPO_ROOT))
    assert sorted(found - budget.models) == []
    assert sorted(budget.models - found) == []


def test_budget_limit_matches_the_models_it_lists():
    """The limit is what the budget ratchet reads, so it can't drift from the list."""
    budget = checker.read_budget(os.path.join(_REPO_ROOT, checker.BUDGET_PATH))
    assert budget.limit == len(budget.models)


def _write_budget(path, limit, models):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({checker.BUDGET_RULE: {"limit": limit, "models": list(models)}}, handle)
    return path


def test_read_budget_reads_the_limit_and_the_models(tmp_path):
    path = _write_budget(tmp_path / "extra-allow-budget.json", 2, ["a.py::A", "b.py::B"])
    assert checker.read_budget(str(path)) == checker.Budget(limit=2, models=frozenset({"a.py::A", "b.py::B"}))


@pytest.mark.parametrize(
    "limit, expected_exit",
    [
        pytest.param(1, 0, id="limit_matching_the_single_model_passes"),
        pytest.param(2, 1, id="limit_above_the_models_listed_fails"),
    ],
)
def test_main_requires_the_limit_to_match_the_list(tmp_path, monkeypatch, limit, expected_exit):
    """A limit above the list would buy silent headroom the budget ratchet can't see."""
    model_dir = tmp_path / checker.SCAN_ROOT / "types"
    model_dir.mkdir(parents=True)
    (model_dir / "thing.py").write_text('class Foo(BaseModel):\n    model_config = ConfigDict(extra="allow")\n')
    _write_budget(tmp_path / checker.BUDGET_PATH, limit, [f"{checker.SCAN_ROOT}/types/thing.py::Foo"])
    monkeypatch.chdir(tmp_path)
    assert checker.main() == expected_exit


def test_raising_the_limit_reds_the_budget_ratchet():
    """Grandfathering one more model means raising the limit, which the shared ratchet catches."""
    budget = checker.read_budget(os.path.join(_REPO_ROOT, checker.BUDGET_PATH))
    base = {checker.BUDGET_RULE: {"limit": budget.limit}}
    head = {checker.BUDGET_RULE: {"limit": budget.limit + 1}}
    regressions = ratchet.regressions_for(checker.BUDGET_PATH, base, head)
    assert [regression.rule for regression in regressions] == [checker.BUDGET_RULE]
    assert ratchet.regressions_for(checker.BUDGET_PATH, base, {checker.BUDGET_RULE: {"limit": budget.limit - 1}}) == []


def test_the_budget_ratchet_watches_this_budget():
    assert checker.BUDGET_PATH in ratchet.DEFAULT_BUDGETS
