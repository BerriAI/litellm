"""Tests for the spend logs query guard at
tests/code_coverage_tests/check_spend_logs_query_budget.py.

The guard exists so a new query against LiteLLM_SpendLogs or a daily aggregate table
cannot land without editing the budget file, which is CODEOWNERS-gated. These tests pin
the two things that would silently defeat it: detecting every shape of query we use
(raw SQL, prisma model access, table name passed to a generic helper) while ignoring
prose mentions, and failing the budget comparison for files that grow or appear.
"""

import json
import os
import sys
from pathlib import Path

_CODE_COVERAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code_coverage_tests")
_REPO_ROOT = Path(_CODE_COVERAGE_DIR).resolve().parents[1]
sys.path.insert(0, _CODE_COVERAGE_DIR)
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import budget_ratchet_check as ratchet  # noqa: E402
import check_spend_logs_query_budget as guard  # noqa: E402


def _scan(tmp_path: Path, source: str) -> tuple[guard.QuerySite, ...]:
    package = tmp_path / "litellm"
    package.mkdir(exist_ok=True)
    module = package / "module.py"
    module.write_text(source, encoding="utf-8")
    return guard.scan_sites(tmp_path, ("litellm",))


def test_detects_raw_sql_select(tmp_path):
    sites = _scan(
        tmp_path,
        "rows = await prisma.db.query_raw('SELECT spend FROM \"LiteLLM_SpendLogs\" sl WHERE sl.spend > 0')\n",
    )
    assert [site.detail for site in sites] == ["raw SQL against 'LiteLLM_SpendLogs'"]
    assert sites[0].path == "litellm/module.py"


def test_detects_raw_sql_join_on_daily_aggregate(tmp_path):
    sites = _scan(tmp_path, 'sql = """\n  SELECT 1\n  JOIN LiteLLM_DailyTeamSpend d ON d.team_id = t.team_id\n"""\n')
    assert [site.detail for site in sites] == ["raw SQL against 'LiteLLM_DailyTeamSpend'"]
    assert sites[0].lineno == 1


def test_detects_prisma_model_access(tmp_path):
    sites = _scan(tmp_path, "rows = await prisma_client.db.litellm_dailyuserspend.find_many(where={})\n")
    assert [site.detail for site in sites] == ["prisma model access 'litellm_dailyuserspend'"]


def test_detects_table_name_passed_to_generic_helper(tmp_path):
    sites = _scan(tmp_path, 'result = await get_daily_activity(table_name="litellm_dailytagspend")\n')
    assert [site.detail for site in sites] == ["table name reference 'litellm_dailytagspend'"]


def test_ignores_prose_mentions_and_unguarded_tables(tmp_path):
    sites = _scan(
        tmp_path,
        '"""Spend for these calls lands in LiteLLM_SpendLogs with zero tokens."""\n'
        "rows = await prisma.db.query_raw('SELECT * FROM \"LiteLLM_TeamTable\"')\n"
        "keys = await prisma_client.db.litellm_verificationtoken.find_many()\n",
    )
    assert sites == ()


def test_counts_every_query_site_in_a_file(tmp_path):
    sites = _scan(
        tmp_path,
        "a = await prisma.db.query_raw('SELECT 1 FROM \"LiteLLM_SpendLogs\"')\n"
        "b = await prisma.db.litellm_spendlogs.count()\n"
        "c = await prisma.db.query_raw('DELETE FROM \"LiteLLM_DailyTagSpend\"')\n",
    )
    assert guard.counts_by_file(sites) == {"litellm/module.py": 3}


def test_over_budget_flags_new_file_and_growth_but_not_shrinkage():
    counts = {"a.py": 2, "b.py": 1, "c.py": 1}
    budget = {"a.py": 1, "c.py": 3}
    assert guard.over_budget(counts, budget) == {"a.py": 2, "b.py": 1}


def test_budget_file_uses_the_ratchet_schema(tmp_path):
    budget_path = tmp_path / "spend-logs-query-budget.json"
    guard.write_budget(budget_path, {"litellm/module.py": 3})
    assert json.loads(budget_path.read_text(encoding="utf-8")) == {"litellm/module.py": {"limit": 3}}
    assert guard.load_budget(budget_path) == {"litellm/module.py": 3}


def test_committed_budget_is_watched_by_the_ratchet():
    assert "spend-logs-query-budget.json" in ratchet.DEFAULT_BUDGETS
    budget = json.loads((_REPO_ROOT / "spend-logs-query-budget.json").read_text(encoding="utf-8"))
    raised = ratchet.regressions_for(
        "spend-logs-query-budget.json",
        budget,
        {path: {"limit": spec["limit"] + 1} for path, spec in budget.items()},
    )
    assert len(raised) == len(budget)


def test_committed_budget_matches_the_repository():
    counts = guard.counts_by_file(guard.scan_sites(_REPO_ROOT))
    budget = guard.load_budget(_REPO_ROOT / "spend-logs-query-budget.json")
    assert guard.over_budget(counts, budget) == {}
    assert set(budget) == set(counts), "stale entries in spend-logs-query-budget.json; rerun the guard with --update"
