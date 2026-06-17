import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_any_discipline.py"
)
_spec = importlib.util.spec_from_file_location("check_any_discipline", _MODULE_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

Violation = mod.Violation


def _v(path="litellm/x.py", line=10, code="LIT009"):
    return Violation(Path(path), line, 0, code, "Any-typed value")


def test_violation_on_a_changed_line_is_in_scope():
    assert mod._in_scope(_v(line=10), {"litellm/x.py": {10, 11}}) is True


def test_violation_on_an_unchanged_line_of_a_changed_file_is_out_of_scope():
    assert mod._in_scope(_v(line=99), {"litellm/x.py": {10, 11}}) is False


def test_whole_new_file_puts_every_line_in_scope():
    assert mod._in_scope(_v(line=99999), {"litellm/x.py": mod.ALL_LINES}) is True


def test_file_absent_from_line_map_is_out_of_scope():
    # Regression: ALL_LINES is a distinct sentinel, so a path missing from the map
    # (line_map.get -> None) is NOT mistaken for "whole file in scope".
    assert mod._in_scope(_v(path="litellm/other.py"), {"litellm/x.py": {1}}) is False


def test_no_line_map_means_no_line_filtering():
    assert mod._in_scope(_v(line=12345), None) is True


def test_build_error_is_always_in_scope():
    assert mod._in_scope(_v(code="LIT000", line=1), {"litellm/x.py": {2}}) is True


# --- per-file Any budget ------------------------------------------------------


def test_slack_is_50_percent_rounded_up():
    assert mod._slack_for(0) == 0
    assert mod._slack_for(1) == 1  # ceil(0.5): even a 1-Any file gets a little room
    assert mod._slack_for(3) == 2  # ceil(1.5)
    assert mod._slack_for(20) == 10
    assert mod._slack_for(5145) == 2573


def test_ceiling_is_baseline_plus_slack():
    assert mod._ceiling({"baseline": 20, "slack": 10}) == 30
    assert mod._ceiling({}) == 0  # an absent/empty entry means a zero ceiling


def test_lit009_counts_groups_by_file_and_ignores_other_codes():
    violations = [
        _v(path="litellm/a.py", line=1, code="LIT009"),
        _v(path="litellm/a.py", line=2, code="LIT009"),
        _v(
            path="litellm/a.py", line=3, code="LIT005"
        ),  # suppression hygiene, not an Any
        _v(path="litellm/b.py", line=1, code="LIT009"),
        _v(path="litellm/c.py", line=0, code="LIT000"),  # build error, not an Any
    ]
    assert mod.lit009_counts(violations) == {"litellm/a.py": 2, "litellm/b.py": 1}


def test_save_budget_omits_zero_count_files_and_round_trips(tmp_path):
    path = tmp_path / "any-discipline-budget.json"
    mod.save_budget(path, {"litellm/a.py": 20, "litellm/b.py": 0, "litellm/c.py": 1})
    loaded = mod.load_budget(path)
    assert loaded == {
        "litellm/a.py": {"baseline": 20, "slack": 10},
        "litellm/c.py": {"baseline": 1, "slack": 1},
    }
    assert "litellm/b.py" not in loaded  # zero-finding files are never baselined


def test_load_budget_missing_file_is_empty(tmp_path):
    assert mod.load_budget(tmp_path / "nope.json") == {}


def test_update_budget_reports_setup_error_when_git_is_unavailable():
    # all_litellm_py_files returns None when git can't list files; --update must
    # surface a clean setup error (exit 2), not crash with a raw traceback.
    assert mod.update_budget(list_files=lambda: None) == 2


# --- object discipline (LIT010) -----------------------------------------------


def test_object_and_any_disciplines_have_distinct_codes_budgets_and_keywords():
    by_code = {d.code: d for d in mod.DISCIPLINES}
    assert set(by_code) == {"LIT009", "LIT010"}
    assert by_code["LIT009"].budget_path == mod.ANY_BUDGET_PATH
    assert by_code["LIT010"].budget_path == mod.OBJECT_BUDGET_PATH
    # An object finding must never be ratcheted against the Any budget (or vice
    # versa): the two budget files and suppression keywords are separate.
    assert mod.ANY_BUDGET_PATH != mod.OBJECT_BUDGET_PATH
    assert by_code["LIT009"].ok_keyword == "any-ok"
    assert by_code["LIT010"].ok_keyword == "object-ok"


def test_counts_for_code_buckets_each_code_independently():
    violations = [
        _v(path="litellm/a.py", line=1, code="LIT009"),
        _v(path="litellm/a.py", line=2, code="LIT010"),
        _v(path="litellm/a.py", line=3, code="LIT010"),
        _v(path="litellm/b.py", line=1, code="LIT010"),
    ]
    assert mod.counts_for_code(violations, "LIT009") == {"litellm/a.py": 1}
    assert mod.counts_for_code(violations, "LIT010") == {
        "litellm/a.py": 2,
        "litellm/b.py": 1,
    }


def test_suppression_is_scoped_to_the_matching_code():
    # any-ok silences LIT009 only; object-ok silences LIT010 only. A wrong-keyword
    # suppression must NOT silence the other discipline's finding.
    src = (
        "a = 1  # any-ok: boundary\n"
        "b = 2  # object-ok: boundary\n"
        "c = 3  # any-ok\n"  # reasonless -> LIT005
    )
    suppressed, lit005 = mod.scan_suppressions(Path("litellm/x.py"), src)
    assert suppressed[1] == frozenset({"LIT009"})
    assert suppressed[2] == frozenset({"LIT010"})
    assert [v.code for v in lit005] == ["LIT005"]
    assert lit005[0].line == 3


# --- end to end: the predicate over real mypy-inferred types ------------------

_SNIPPET = (
    "import json\n"  # 1
    "def boundary(x: object) -> str:\n"  # 2
    "    y = x\n"  # 3: rvalue x is a bare object value -> LIT010
    "    return repr(y)\n"  # 4: y is object -> LIT010 (repr(...) itself is clean str)
    "coarse: dict[str, object] = {}\n"  # 5: the {} literal is dict[str, object] -> LIT010
    "clean: dict[str, int] = {}\n"  # 6: no coarse leaf -> no finding
    "loaded = json.loads('1')\n"  # 7: json.loads(...) -> Any -> LIT009
)


def _coarse_findings(src: str, cache_dir: Path):
    from mypy import build
    from mypy.modulefinder import BuildSource
    from mypy.options import Options

    opts = Options()
    opts.export_types = True
    opts.preserve_asts = True
    opts.incremental = False
    opts.cache_dir = str(cache_dir)
    opts.show_traceback = True
    res = build.build([BuildSource(None, "snippet", src)], options=opts)
    idmap = {id(expr): t for expr, t in res.types.items()}
    tree = res.graph["snippet"].tree
    assert tree is not None
    return {
        (line, code) for line, _col, code, _typ in mod.find_coarse_in_tree(tree, idmap)
    }


def test_object_in_a_container_and_bare_object_value_are_flagged(tmp_path):
    found = _coarse_findings(_SNIPPET, tmp_path / "cache")
    assert (3, "LIT010") in found  # a bare object value (the boundary param read)
    assert (5, "LIT010") in found  # dict[str, object] literal -- the coarse container
    assert (7, "LIT009") in found  # the Any gate still fires (json.loads)


def test_clean_dict_str_int_is_not_flagged_as_object(tmp_path):
    # str subclasses object, dict[str, int] has no object leaf: the rule must key
    # on the literal `object` type, never on "is a subclass of object".
    found = _coarse_findings(_SNIPPET, tmp_path / "cache")
    assert not any(line == 6 for line, _code in found)
    assert (6, "LIT010") not in found
