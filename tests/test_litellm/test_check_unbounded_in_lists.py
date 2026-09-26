"""Tests for tests/code_coverage_tests/check_unbounded_in_lists.py.

The checker reads Python rather than grepping for `IN (`, so the cases that matter are
the ones a grep gets wrong: a subquery or a literal list inside the parentheses, a
runtime value spliced in after them, a fixed display versus a name in a Prisma filter,
and where a `# bounded-ok` marker may sit for a literal a comment cannot go inside.
"""

import importlib.util
import sys
from pathlib import Path

_CHECKER_PATH = Path(__file__).resolve().parents[1] / "code_coverage_tests" / "check_unbounded_in_lists.py"
_SPEC = importlib.util.spec_from_file_location("check_unbounded_in_lists", _CHECKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)


def _check(tmp_path: Path, source: str) -> tuple:
    target = tmp_path / "module.py"
    target.write_text(source, encoding="utf-8")
    return checker.check_file(target)


def _kinds(tmp_path: Path, source: str) -> tuple:
    return tuple(finding.kind for finding in _check(tmp_path, source))


def _lines(tmp_path: Path, source: str) -> tuple:
    return tuple(finding.line for finding in _check(tmp_path, source))


class TestPrismaFilters:
    def test_a_name_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"user_id": {"in": user_ids}}\n') == ("prisma",)

    def test_a_call_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"user_id": {"in": list(user_ids)}}\n') == ("prisma",)

    def test_a_comprehension_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"id": {"in": [row.id for row in rows]}}\n') == ("prisma",)

    def test_an_attribute_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"user_id": {"in": data.user_ids}}\n') == ("prisma",)

    def test_a_starred_display_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"user_id": {"in": [*user_ids]}}\n') == ("prisma",)

    def test_not_in_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"status": {"not_in": list(statuses)}}\n') == ("prisma",)

    def test_a_filter_nested_in_a_clause_list_is_flagged(self, tmp_path):
        source = 'where = {"OR": [{"team_id": {"in": team_ids}}, {"user_id": user_id}]}\n'
        assert _kinds(tmp_path, source) == ("prisma",)

    def test_a_display_of_constants_passes(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"status": {"not_in": ["failed", "expired"]}}\n') == ()

    def test_a_display_with_a_fixed_number_of_names_passes(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"user_id": {"in": [user_id]}}\n') == ()
        assert _kinds(tmp_path, 'where = {"user_id": {"in": (owner, editor)}}\n') == ()

    def test_a_module_constant_bound_to_a_display_passes(self, tmp_path):
        constant = 'ANCHORED: Final = frozenset({"oauth2", "api_key"})\n'
        assert _kinds(tmp_path, constant + 'where = {"auth_type": {"in": ANCHORED}}\n') == ()
        assert _kinds(tmp_path, constant + 'where = {"auth_type": {"in": list(ANCHORED)}}\n') == ()
        assert _kinds(tmp_path, constant + 'where = {"auth_type": {"in": sorted(ANCHORED)}}\n') == ()

    def test_a_module_constant_built_from_another_passes(self, tmp_path):
        source = 'FIRST = ("a", "b")\nSECOND: Final = tuple(FIRST)\nwhere = {"x": {"in": SECOND}}\n'
        assert _kinds(tmp_path, source) == ()

    def test_casing_does_not_make_a_constant(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"auth_type": {"in": ANCHORED_AUTH_TYPES}}\n') == ("prisma",)
        assert _kinds(tmp_path, 'USER_IDS = load_ids()\nwhere = {"user_id": {"in": USER_IDS}}\n') == ("prisma",)
        assert _kinds(tmp_path, 'from x import STATES\nwhere = {"s": {"in": list(STATES)}}\n') == ("prisma",)
        assert _kinds(tmp_path, 'terminal = ("done", "failed")\nwhere = {"s": {"in": terminal}}\n') == ()

    def test_a_constant_spread_into_a_display_is_still_a_constant(self, tmp_path):
        base = 'BASE: Final = ("a", "b")\n'
        assert _kinds(tmp_path, base + 'MORE: Final = (*BASE, "c")\nwhere = {"s": {"not_in": list(MORE)}}\n') == ()
        assert _kinds(tmp_path, base + 'where = {"s": {"in": [*BASE, "c"]}}\n') == ()
        assert _kinds(tmp_path, base + 'where = {"s": {"in": [*BASE, *extra]}}\n') == ("prisma",)
        assert _kinds(tmp_path, 'MORE: Final = (*load(), "c")\nwhere = {"s": {"in": MORE}}\n') == ("prisma",)

    def test_a_module_value_that_could_grow_is_not_a_constant(self, tmp_path):
        assert _kinds(tmp_path, 'IDS = ["a"]\nIDS.append(late)\nwhere = {"x": {"in": IDS}}\n') == ("prisma",)
        assert _kinds(tmp_path, 'IDS = sorted(("a", "b"))\nwhere = {"x": {"in": IDS}}\n') == ("prisma",)
        assert _kinds(tmp_path, 'IDS = ("a",)\nwhere = {"x": {"in": IDS}}\n') == ()
        assert _kinds(tmp_path, 'IDS = frozenset(["a", "b"])\nwhere = {"x": {"in": IDS}}\n') == ()

    def test_an_alias_is_as_fixed_as_what_it_names(self, tmp_path):
        assert _kinds(tmp_path, 'A = load_ids()\nB = A\nwhere = {"x": {"in": B}}\n') == ("prisma",)
        assert _kinds(tmp_path, 'A = ("a",)\nB = A\nwhere = {"x": {"in": B}}\n') == ()

    def test_a_module_name_bound_twice_is_not_a_constant(self, tmp_path):
        source = 'IDS = ("a",)\nIDS = load_ids()\nwhere = {"user_id": {"in": IDS}}\n'
        assert _kinds(tmp_path, source) == ("prisma",)

    def test_a_local_binding_is_not_a_constant(self, tmp_path):
        source = 'def f():\n    ids = ("a", "b")\n    return {"user_id": {"in": ids}}\n'
        assert _kinds(tmp_path, source) == ("prisma",)

    def test_a_name_wrapped_in_a_constructor_is_still_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"token": {"in": tuple(frozenset(tokens))}}\n') == ("prisma",)

    def test_a_scalar_value_passes(self, tmp_path):
        assert _kinds(tmp_path, 'parameter = {"name": "q", "in": "query"}\n') == ()

    def test_a_dict_with_a_spread_does_not_break_the_walk(self, tmp_path):
        assert _kinds(tmp_path, 'where = {**base, "team_id": {"in": team_ids}}\n') == ("prisma",)

    def test_the_reported_line_is_the_key_line(self, tmp_path):
        source = 'where = {\n    "team_id": {\n        "in": sorted(team_ids),\n    },\n}\n'
        assert _lines(tmp_path, source) == (3,)

    def test_the_message_names_the_value(self, tmp_path):
        (finding,) = _check(tmp_path, 'where = {"user_id": {"in": list(user_ids)}}\n')
        assert "list(user_ids)" in finding.message

    def test_an_in_list_is_pointed_at_the_chunking_helper(self, tmp_path):
        (finding,) = _check(tmp_path, 'where = {"user_id": {"in": user_ids}}\n')
        assert "litellm.repositories.chunked_in" in finding.message

    def test_a_not_in_list_is_pointed_at_an_array_parameter_since_it_cannot_be_chunked(self, tmp_path):
        (finding,) = _check(tmp_path, 'where = {"user_id": {"not_in": user_ids}}\n')
        assert "<> ALL($1::text[])" in finding.message
        assert "chunked_in" not in finding.message


class TestTypedDictFieldMaps:
    """A functional TypedDict's field map names fields: its "in" key is a type, not a filter."""

    def test_a_functional_typed_dict_field_map_is_not_flagged(self, tmp_path):
        source = 'Filter = TypedDict("Filter", {"in": NotRequired[Sequence[str]], "notIn": Sequence[str]})\n'
        assert _kinds(tmp_path, source) == ()

    def test_the_typing_and_typing_extensions_attribute_forms_are_not_flagged(self, tmp_path):
        source = (
            'A = typing.TypedDict("A", {"in": Sequence[str]})\n'
            'B = typing_extensions.TypedDict("B", {"notIn": Sequence[str]})\n'
        )
        assert _kinds(tmp_path, source) == ()

    def test_a_fields_keyword_field_map_is_not_flagged(self, tmp_path):
        source = 'Filter = TypedDict("Filter", fields={"in": Sequence[str]}, total=False)\n'
        assert _kinds(tmp_path, source) == ()

    def test_a_filter_passed_to_another_call_is_still_flagged(self, tmp_path):
        source = 'rows = find_many("Filter", {"in": user_ids})\n'
        assert _kinds(tmp_path, source) == ("prisma",)

    def test_a_typed_dict_from_another_module_is_still_flagged(self, tmp_path):
        source = 'Filter = mylib.TypedDict("Filter", {"in": user_ids})\n'
        assert _kinds(tmp_path, source) == ("prisma",)

    def test_a_filter_nested_inside_a_field_map_value_is_still_flagged(self, tmp_path):
        source = 'Filter = TypedDict("Filter", {"where": {"user_id": {"in": user_ids}}})\n'
        assert _kinds(tmp_path, source) == ("prisma",)

    def test_a_filter_as_the_first_argument_of_typed_dict_is_still_flagged(self, tmp_path):
        source = 'Filter = TypedDict({"in": user_ids}, {})\n'
        assert _kinds(tmp_path, source) == ("prisma",)


class TestRawSql:
    def test_an_fstring_slice_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'sql = f"WHERE team_id IN ({placeholders})"\n') == ("raw-sql",)

    def test_not_in_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, "sql = f'\"{field}\" NOT IN ({placeholders})'\n") == ("raw-sql",)

    def test_lowercase_sql_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'sql = f"where team_id in ({placeholders})"\n') == ("raw-sql",)

    def test_a_format_slot_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'sql = "WHERE team_id IN ({})".format(placeholders)\n') == ("raw-sql",)
        assert _kinds(tmp_path, 'SQL = "WHERE team_id IN ({ids})"\n') == ("raw-sql",)

    def test_a_percent_slot_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'sql = "WHERE team_id IN (%s)" % placeholders\n') == ("raw-sql",)
        assert _kinds(tmp_path, 'sql = "WHERE team_id IN (%(ids)s)" % {"ids": placeholders}\n') == ("raw-sql",)

    def test_a_literal_that_closes_after_the_paren_is_flagged(self, tmp_path):
        assert _kinds(tmp_path, 'sql = "WHERE team_id IN (" + placeholders + ")"\n') == ("raw-sql",)

    def test_a_subquery_passes(self, tmp_path):
        source = 'sql = f"""\n    DELETE FROM "{table}"\n    WHERE id IN (\n        SELECT id FROM "{table}" LIMIT $1\n    )\n"""\n'
        assert _kinds(tmp_path, source) == ()

    def test_an_implicitly_concatenated_subquery_passes(self, tmp_path):
        source = "sql = (\n    'DELETE FROM t WHERE request_id IN ('\n    'SELECT request_id FROM t LIMIT $1)'\n)\n"
        assert _kinds(tmp_path, source) == ()

    def test_a_fixed_number_of_placeholders_passes(self, tmp_path):
        assert _kinds(tmp_path, 'sql = f"api_key NOT IN (${p}, ${p + 1})"\n') == ()

    def test_a_literal_list_passes(self, tmp_path):
        assert _kinds(tmp_path, "sql = \"status NOT IN ('failed', 'expired')\"\n") == ()

    def test_an_array_parameter_passes(self, tmp_path):
        assert _kinds(tmp_path, 'sql = "WHERE user_id = ANY($1::text[])"\n') == ()
        assert _kinds(tmp_path, 'sql = "WHERE model IN (SELECT jsonb_array_elements_text($1::jsonb))"\n') == ()

    def test_an_escaped_brace_passes(self, tmp_path):
        assert _kinds(tmp_path, 'sql = f"WHERE x IN ({{literal}}) AND y = {y}"\n') == ()

    def test_a_word_ending_in_in_passes(self, tmp_path):
        assert _kinds(tmp_path, 'sql = f"SELECT MIN ({column}) FROM t"\n') == ()
        assert _kinds(tmp_path, 'message = f"LOGIN ({user}) failed"\n') == ()

    def test_an_fstring_is_reported_once(self, tmp_path):
        assert _kinds(tmp_path, 'sql = f"WHERE a IN ({x})" + f" AND b IN ({y})"\n') == ("raw-sql", "raw-sql")

    def test_a_multiline_literal_reports_its_first_line_and_names_the_in_line(self, tmp_path):
        source = 'sql = f"""\n    SELECT 1\n    FROM t\n    WHERE team_id IN ({placeholders})\n"""\n'
        (finding,) = _check(tmp_path, source)
        assert finding.line == 1
        assert "line 4" in finding.message


class TestMarkers:
    def test_a_marker_on_the_line_suppresses(self, tmp_path):
        source = 'where = {"team_id": {"in": page_ids}}  # bounded-ok: one page of at most 100 ids\n'
        assert _kinds(tmp_path, source) == ()

    def test_a_marker_shares_the_line_with_other_suppressions(self, tmp_path):
        source = 'where = {"team_id": {"in": page_ids}}  # mutable-ok: prisma filter  # bounded-ok: one page\n'
        assert _kinds(tmp_path, source) == ()

    def test_a_marker_alone_on_the_line_above_suppresses(self, tmp_path):
        source = '# bounded-ok: the expected views are a fixed set\nsql = f"""\n    WHERE viewname IN ({views})\n"""\n'
        assert _kinds(tmp_path, source) == ()

    def test_a_marker_two_lines_above_does_not_suppress(self, tmp_path):
        source = '# bounded-ok: one page\n\nwhere = {"team_id": {"in": page_ids}}\n'
        assert _kinds(tmp_path, source) == ("prisma",)

    def test_a_marker_trailing_the_line_above_does_not_suppress(self, tmp_path):
        source = 'other = 1  # bounded-ok: one page\nwhere = {"team_id": {"in": page_ids}}\n'
        assert _kinds(tmp_path, source) == ("prisma",)

    def test_a_marker_without_a_reason_is_its_own_finding_and_suppresses_nothing(self, tmp_path):
        source = 'where = {"team_id": {"in": page_ids}}  # bounded-ok\n'
        assert _kinds(tmp_path, source) == ("marker", "prisma")

    def test_a_marker_with_a_token_reason_is_rejected(self, tmp_path):
        source = 'where = {"team_id": {"in": page_ids}}  # bounded-ok: ok\n'
        assert _kinds(tmp_path, source) == ("marker", "prisma")


class TestDriver:
    def test_the_chunking_helper_is_exempt(self):
        helper = checker.REPO_ROOT / "litellm" / "repositories" / "chunked_in.py"
        assert "prisma" in tuple(finding.kind for finding in checker.check_file(helper))
        assert checker.scan(checker.collect_paths([str(helper)])) == ()

    def test_a_copy_of_the_helper_elsewhere_is_not_exempt(self, tmp_path):
        helper = checker.REPO_ROOT / "litellm" / "repositories" / "chunked_in.py"
        copy = tmp_path / "chunked_in.py"
        copy.write_text(helper.read_text(encoding="utf-8"), encoding="utf-8")
        assert "prisma" in tuple(finding.kind for finding in checker.scan([copy]))

    def test_a_syntax_error_is_reported_not_raised(self, tmp_path):
        assert _kinds(tmp_path, "def broken(:\n") == ("unreadable",)

    def test_directories_are_walked(self, tmp_path):
        nested = tmp_path / "pkg" / "sub"
        nested.mkdir(parents=True)
        (nested / "a.py").write_text('where = {"user_id": {"in": user_ids}}\n', encoding="utf-8")
        (nested / "b.txt").write_text('where = {"user_id": {"in": user_ids}}\n', encoding="utf-8")
        findings = checker.scan(checker.collect_paths([str(tmp_path / "pkg")]))
        assert tuple(finding.path.name for finding in findings) == ("a.py",)


def _identities(tmp_path: Path, source: str) -> tuple:
    return tuple(checker.identify(_check(tmp_path, source)))


class TestIdentity:
    def test_a_finding_is_keyed_by_scope_field_and_occurrence_not_line(self, tmp_path):
        source = (
            "class Repo:\n"
            "    async def load(self):\n"
            '        a = {"user_id": {"in": ids}}\n'
            '        b = {"user_id": {"in": more}}\n'
            '        return {"team_id": {"not_in": teams}}\n'
        )
        path = (tmp_path / "module.py").resolve().as_posix()
        assert _identities(tmp_path, source) == (
            f"{path} Repo.load prisma user_id.in 0",
            f"{path} Repo.load prisma user_id.in 1",
            f"{path} Repo.load prisma team_id.not_in 0",
        )

    def test_the_field_is_read_from_a_subscript_or_keyword_or_computed_key(self, tmp_path):
        source = 'where["user_id"] = {"in": ids}\nwhere = Filter(team_id={"in": ids})\nwhere = {field: {"in": ids}}\n'
        subjects = tuple(key.split(" ")[3] for key in _identities(tmp_path, source))
        assert subjects == ("user_id.in", "team_id.in", "[field].in")

    def test_raw_sql_is_keyed_by_the_column_before_in(self, tmp_path):
        source = 'def q():\n    return f"WHERE \\"{column}\\" NOT IN ({placeholders})"\n'
        path = (tmp_path / "module.py").resolve().as_posix()
        assert _identities(tmp_path, source) == (f"{path} q raw-sql {{column}}.IN 0",)

    def test_moving_code_down_the_file_keeps_the_key(self, tmp_path):
        source = 'def f():\n    return {"user_id": {"in": ids}}\n'
        shifted = "import os\n\n\ndef g():\n    return 1\n\n\n" + source
        assert _identities(tmp_path, source) == _identities(tmp_path, shifted)


class TestBaseline:
    def _run(self, *args: str) -> int:
        return checker.main(list(args))

    def _write(self, tmp_path: Path, source: str) -> Path:
        target = tmp_path / "pkg" / "module.py"
        target.parent.mkdir(exist_ok=True)
        target.write_text(source, encoding="utf-8")
        return target

    def test_a_finding_missing_from_the_baseline_fails_the_run(self, tmp_path, capsys):
        target = self._write(tmp_path, 'where = {"user_id": {"in": user_ids}}\n')
        baseline = tmp_path / "baseline.txt"
        assert self._run(str(target), "--baseline", str(baseline)) == 1
        out = capsys.readouterr().out
        assert f"{target}:1: prisma" in out
        assert "1 new" in out

    def test_a_baselined_finding_passes_even_after_the_code_moves(self, tmp_path, capsys):
        target = self._write(tmp_path, 'def f():\n    return {"user_id": {"in": user_ids}}\n')
        baseline = tmp_path / "baseline.txt"
        assert self._run(str(target), "--baseline", str(baseline), "--update-baseline") == 0
        target.write_text("import os\n\n\n" + target.read_text(encoding="utf-8"), encoding="utf-8")
        assert self._run(str(target), "--baseline", str(baseline)) == 0
        assert "1 baselined, 0 new, 0 stale" in capsys.readouterr().out

    def test_a_new_finding_beside_a_baselined_one_fails(self, tmp_path, capsys):
        target = self._write(tmp_path, 'def f():\n    return {"user_id": {"in": user_ids}}\n')
        baseline = tmp_path / "baseline.txt"
        assert self._run(str(target), "--baseline", str(baseline), "--update-baseline") == 0
        target.write_text(
            target.read_text(encoding="utf-8") + 'def g():\n    return {"user_id": {"in": user_ids}}\n',
            encoding="utf-8",
        )
        assert self._run(str(target), "--baseline", str(baseline)) == 1
        assert f"{target}:4: prisma" in capsys.readouterr().out

    def test_a_fixed_finding_leaves_a_stale_entry_that_fails_the_run(self, tmp_path, capsys):
        target = self._write(tmp_path, 'def f():\n    return {"user_id": {"in": user_ids}}\n')
        baseline = tmp_path / "baseline.txt"
        assert self._run(str(target), "--baseline", str(baseline), "--update-baseline") == 0
        target.write_text('def f():\n    return {"user_id": {"in": [user_id]}}\n', encoding="utf-8")
        assert self._run(str(target), "--baseline", str(baseline)) == 1
        out = capsys.readouterr().out
        assert "stale entry" in out
        assert "f prisma user_id.in 0" in out

    def test_update_baseline_drops_fixed_entries_and_keeps_unscanned_ones(self, tmp_path):
        target = self._write(tmp_path, 'def f():\n    return {"user_id": {"in": user_ids}}\n')
        baseline = tmp_path / "baseline.txt"
        elsewhere = "litellm/elsewhere.py g prisma team_id.in 0"
        fixed = f"{target.resolve().as_posix()} gone prisma team_id.in 0"
        baseline.write_text(f"{elsewhere}\n{fixed}\n", encoding="utf-8")
        assert self._run(str(target), "--baseline", str(baseline), "--update-baseline") == 0
        assert checker.read_baseline(baseline) == frozenset(
            {elsewhere, f"{target.resolve().as_posix()} f prisma user_id.in 0"}
        )
        assert self._run(str(target), "--baseline", str(baseline)) == 0

    def test_entries_for_files_outside_the_scan_are_not_stale(self, tmp_path):
        target = self._write(tmp_path, "x = 1\n")
        baseline = tmp_path / "baseline.txt"
        baseline.write_text("litellm/elsewhere.py g prisma team_id.in 0\n", encoding="utf-8")
        assert self._run(str(target), "--baseline", str(baseline)) == 0

    def test_an_entry_for_a_deleted_file_under_a_scanned_directory_is_stale(self, tmp_path):
        self._write(tmp_path, "x = 1\n")
        baseline = tmp_path / "baseline.txt"
        gone = (tmp_path / "pkg" / "deleted.py").resolve().as_posix()
        baseline.write_text(f"{gone} f prisma user_id.in 0\n", encoding="utf-8")
        assert self._run(str(tmp_path / "pkg"), "--baseline", str(baseline)) == 1
