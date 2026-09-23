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

    def test_a_module_constant_passes(self, tmp_path):
        assert _kinds(tmp_path, 'where = {"auth_type": {"in": ANCHORED_AUTH_TYPES}}\n') == ()
        assert _kinds(tmp_path, 'where = {"auth_type": {"in": list(ANCHORED_AUTH_TYPES)}}\n') == ()
        assert _kinds(tmp_path, 'where = {"auth_type": {"in": sorted(_TERMINAL_STATES)}}\n') == ()

    def test_a_lowercase_name_wrapped_in_a_constructor_is_still_flagged(self, tmp_path):
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
    def test_findings_do_not_fail_the_run(self, tmp_path, capsys):
        target = tmp_path / "module.py"
        target.write_text('where = {"user_id": {"in": user_ids}}\n', encoding="utf-8")
        assert checker.main([str(target)]) == 0
        out = capsys.readouterr().out
        assert f"{target}:1: prisma" in out
        assert "1 unbounded IN list(s)" in out

    def test_a_syntax_error_is_reported_not_raised(self, tmp_path):
        assert _kinds(tmp_path, "def broken(:\n") == ("unreadable",)

    def test_directories_are_walked(self, tmp_path):
        nested = tmp_path / "pkg" / "sub"
        nested.mkdir(parents=True)
        (nested / "a.py").write_text('where = {"user_id": {"in": user_ids}}\n', encoding="utf-8")
        (nested / "b.txt").write_text('where = {"user_id": {"in": user_ids}}\n', encoding="utf-8")
        findings = checker.scan(checker.collect_paths([str(tmp_path / "pkg")]))
        assert tuple(finding.path.name for finding in findings) == ("a.py",)
