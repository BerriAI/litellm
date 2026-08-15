"""Tests for scripts/check_dict_usage.py (DICT001).

Two regressions are pinned. First, the rule itself: expressions whose
mypy-inferred type is a bare mutable dict are flagged, while TypedDict,
``Mapping``, and ``MappingProxyType`` usage stays silent, so the sanctioned
immutable alternatives never trip the gate. Second, the hand-rolled
traversal's node-class map: every concrete mypy AST class must be classified
in ``CHILD_ATTRS`` or ``LEAF_NODES`` (or be one of the abstract bases), so a
mypy upgrade that introduces a new node form fails here instead of being
silently skipped at scan time.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"

_checker_spec = importlib.util.spec_from_file_location("check_dict_usage", _SCRIPTS / "check_dict_usage.py")
checker = importlib.util.module_from_spec(_checker_spec)
sys.modules[_checker_spec.name] = checker
_checker_spec.loader.exec_module(checker)

_gate_spec = importlib.util.spec_from_file_location("dict_usage_gate", _SCRIPTS / "dict_usage_gate.py")
gate = importlib.util.module_from_spec(_gate_spec)
sys.modules[_gate_spec.name] = gate
_gate_spec.loader.exec_module(gate)


FIXTURE = """\
from collections import defaultdict
from types import MappingProxyType
from typing import Mapping, TypedDict


class Movie(TypedDict):
    title: str


def typed_helper() -> dict[str, int]:
    return {"a": 1}


def flagged_flows() -> int:
    inferred = typed_helper()  # flagged
    copied = inferred.copy()  # flagged
    read = inferred["a"]  # flagged
    dd = defaultdict(list)  # flagged
    walrus = len(w := typed_helper())  # flagged
    picked = inferred if read else copied  # flagged
    return len(copied) + read + walrus + len(picked) + len(dd) + len(w)  # flagged


def clean_flows(mapping: Mapping[str, int], movie: Movie) -> int:
    proxy = MappingProxyType({"b": 2})
    title = movie["title"]
    return len(mapping) + len(proxy) + len(title)
"""


@pytest.fixture(scope="module")
def fixture_package(tmp_path_factory):
    pkg = (tmp_path_factory.mktemp("dict_usage") / "pkg").resolve()
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sample.py").write_text(FIXTURE)
    return pkg


@pytest.fixture(scope="module")
def fixture_violations(fixture_package):
    return checker.collect_violations((fixture_package,), (3, 12))


def test_flags_exactly_the_mutable_dict_usage_lines(fixture_violations):
    expected = {number for number, line in enumerate(FIXTURE.splitlines(), start=1) if "# flagged" in line}
    assert {violation.line for violation in fixture_violations} == expected


def test_flagged_types_are_all_dict_variants(fixture_violations):
    assert fixture_violations
    assert {violation.type_fullname for violation in fixture_violations} <= checker.DICT_FULLNAMES


def test_typeddict_mapping_and_proxy_usage_stays_silent(fixture_violations):
    clean_region = FIXTURE.splitlines().index("def clean_flows(mapping: Mapping[str, int], movie: Movie) -> int:") + 1
    assert not [v for v in fixture_violations if v.line >= clean_region]


def test_gate_parses_the_checker_output_format(fixture_package, capsys):
    exit_code = checker.main([str(fixture_package)])
    output = capsys.readouterr().out
    parsed = gate.parse_violations(output, fixture_package.parent)
    assert exit_code == 1
    assert parsed
    assert {violation.file for violation in parsed} == {"pkg/sample.py"}
    assert {violation.code for violation in parsed} == {"DICT001"}
    assert gate.count_by_rule(parsed) == {"DICT001": len(parsed)}


def test_clean_tree_exits_zero(tmp_path, capsys):
    pkg = (tmp_path / "clean_pkg").resolve()
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "sample.py").write_text("VALUE: int = 1\n")
    assert checker.main([str(pkg)]) == 0
    assert capsys.readouterr().out == ""


def _concrete_context_classes():
    import mypy.nodes
    import mypy.patterns

    return {
        obj
        for module in (mypy.nodes, mypy.patterns)
        for obj in vars(module).values()
        if inspect.isclass(obj) and issubclass(obj, mypy.nodes.Context) and obj.__module__ == module.__name__
    }


def test_every_concrete_mypy_node_class_is_classified():
    import mypy.nodes
    import mypy.patterns

    abstract_bases = {
        mypy.nodes.Context,
        mypy.nodes.Node,
        mypy.nodes.Statement,
        mypy.nodes.Expression,
        mypy.nodes.SymbolNode,
        mypy.nodes.FuncBase,
        mypy.nodes.FuncItem,
        mypy.nodes.RefExpr,
        mypy.nodes.ImportBase,
        mypy.nodes.TypeVarLikeExpr,
        mypy.patterns.Pattern,
    }
    classified = set(checker.CHILD_ATTRS) | set(checker.LEAF_NODES)
    unclassified = _concrete_context_classes() - classified - abstract_bases
    assert not unclassified, (
        "mypy node classes missing from CHILD_ATTRS/LEAF_NODES in "
        f"scripts/check_dict_usage.py: {sorted(cls.__name__ for cls in unclassified)}"
    )


def test_child_attrs_exist_on_their_classes():
    missing = [
        f"{cls.__name__}.{attr}" for cls, attrs in checker.CHILD_ATTRS.items() for attr in attrs if attr not in dir(cls)
    ]
    assert not missing


def test_unmapped_node_class_fails_loudly(monkeypatch):
    monkeypatch.setattr(checker, "LEAF_NODES", checker.LEAF_NODES - {checker.PassStmt})
    with pytest.raises(SystemExit, match="PassStmt"):
        list(checker.iter_nodes(checker.PassStmt()))


def test_parse_python_version():
    assert checker.parse_python_version('{"pythonVersion": "3.12"}') == (3, 12)
    assert checker.parse_python_version('{"pythonVersion": "3"}') is None
    assert checker.parse_python_version('{"pythonVersion": 312}') is None
    assert checker.parse_python_version("{}") is None
    assert checker.parse_python_version("not json") is None
