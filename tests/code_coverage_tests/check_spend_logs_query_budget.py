"""LiteLLM_SpendLogs and the daily aggregate tables are the largest, hottest tables in
the proxy database, so an unreviewed query against them can take a production gateway
down. This check counts every query site against those tables in litellm/ and
enterprise/ and fails when a file exceeds the budget recorded in
spend-logs-query-budget.json.

Adding a query therefore means editing that budget file, which is owned in
.github/CODEOWNERS, so it cannot merge without a review from the spend logs owners. Run
`python tests/code_coverage_tests/check_spend_logs_query_budget.py --update` to rewrite
the budgets from the working tree once the new query is signed off."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUDGET_PATH = REPO_ROOT / "spend-logs-query-budget.json"
SCANNED_DIRS = ("litellm", "enterprise")

GUARDED_TABLES = frozenset(
    {
        "LiteLLM_SpendLogs",
        "LiteLLM_SpendLogGuardrailIndex",
        "LiteLLM_SpendLogToolIndex",
        "LiteLLM_DailyUserSpend",
        "LiteLLM_DailyTeamSpend",
        "LiteLLM_DailyTagSpend",
        "LiteLLM_DailyOrganizationSpend",
        "LiteLLM_DailyEndUserSpend",
        "LiteLLM_DailyAgentSpend",
        "LiteLLM_DailyGuardrailMetrics",
        "LiteLLM_DailyPolicyMetrics",
    }
)
GUARDED_PRISMA_ATTRS = frozenset(table.lower() for table in GUARDED_TABLES)
SQL_TABLE_REFERENCE = re.compile(
    r"\b(?:from|join|into|update|truncate|table|only)\s+\"?("
    + "|".join(sorted(GUARDED_TABLES, key=len, reverse=True))
    + r")\"?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class QuerySite:
    path: str
    lineno: int
    detail: str


def _sql_table_in(text: str) -> str | None:
    match = SQL_TABLE_REFERENCE.search(text)
    return match.group(1) if match is not None else None


def _site_for(node: ast.AST, relative: str) -> QuerySite | None:
    if isinstance(node, ast.Attribute) and node.attr in GUARDED_PRISMA_ATTRS:
        return QuerySite(relative, node.lineno, f"prisma model access '{node.attr}'")
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    if node.value.lower() in GUARDED_PRISMA_ATTRS:
        return QuerySite(relative, node.lineno, f"table name reference '{node.value}'")
    table = _sql_table_in(node.value)
    return QuerySite(relative, node.lineno, f"raw SQL against '{table}'") if table is not None else None


def sites_in_file(path: Path, root: Path) -> tuple[QuerySite, ...]:
    relative = path.relative_to(root).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(site for node in ast.walk(tree) if (site := _site_for(node, relative)) is not None)


def scan_sites(root: Path, directories: tuple[str, ...] = SCANNED_DIRS) -> tuple[QuerySite, ...]:
    return tuple(
        site
        for directory in directories
        for path in sorted((root / directory).rglob("*.py"))
        for site in sites_in_file(path, root)
    )


def counts_by_file(sites: tuple[QuerySite, ...]) -> dict[str, int]:
    return {path: sum(1 for site in sites if site.path == path) for path in sorted({site.path for site in sites})}


def over_budget(counts: dict[str, int], budget: dict[str, int]) -> dict[str, int]:
    return {path: count for path, count in counts.items() if count > budget.get(path, 0)}


def load_budget(budget_path: Path) -> dict[str, int]:
    parsed: object = json.loads(budget_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{budget_path.name} must be an object mapping file paths to {{'limit': int}}")
    return {str(path): int(spec["limit"]) for path, spec in parsed.items()}


def write_budget(budget_path: Path, counts: dict[str, int]) -> None:
    specs = {path: {"limit": count} for path, count in counts.items()}
    _ = budget_path.write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _report(sites: tuple[QuerySite, ...], violations: dict[str, int], budget: dict[str, int]) -> None:
    for path, count in violations.items():
        print(f"{path}: {count} spend logs query site(s), budget is {budget.get(path, 0)}")
        for site in sites:
            if site.path == path:
                print(f"  {site.path}:{site.lineno}: {site.detail}")
    print(
        "\nQuerying LiteLLM_SpendLogs or the daily aggregate tables is restricted. Reuse an existing "
        "repository or helper if one fits; otherwise get sign-off from the spend logs owners in "
        ".github/CODEOWNERS, run `python tests/code_coverage_tests/check_spend_logs_query_budget.py --update`, "
        "and commit the updated spend-logs-query-budget.json so the new query is reviewed explicitly"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Enforce the spend logs query budget")
    _ = parser.add_argument("--update", action="store_true", help="rewrite the budget file from the working tree")
    args = parser.parse_args()

    sites = scan_sites(REPO_ROOT)
    counts = counts_by_file(sites)

    if bool(args.update):
        write_budget(BUDGET_PATH, counts)
        print(f"wrote {BUDGET_PATH.name}: {sum(counts.values())} query site(s) across {len(counts)} file(s)")
        return 0

    budget = load_budget(BUDGET_PATH)
    violations = over_budget(counts, budget)
    if not violations:
        print(f"spend logs query budget ok: {sum(counts.values())} query site(s) across {len(counts)} file(s)")
        return 0

    _report(sites, violations, budget)
    return 1


if __name__ == "__main__":
    sys.exit(main())
