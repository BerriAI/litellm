import os
from pathlib import Path
from typing import Final

import psycopg
from psycopg import sql
from pydantic import BaseModel, Field, TypeAdapter

from litellm.proxy.management_endpoints.customer_endpoints import _CUSTOMER_SHARED_BUDGET_SQL

_REPOSITORY: Final = Path(__file__).resolve().parents[4]
_MIGRATIONS: Final = _REPOSITORY / "litellm-proxy-extras" / "litellm_proxy_extras" / "migrations"
_INDEX_MIGRATIONS: Final = (
    "20261018000100_customer_budget_reference_index",
    "20261018000200_key_budget_reference_index",
    "20261018000300_key_customer_budget_reference_index",
)
_REFERENCE_TABLES: Final = (
    "LiteLLM_OrganizationTable",
    "LiteLLM_ProjectTable",
    "LiteLLM_TagTable",
    "LiteLLM_ModelAccessGroupBudgetTable",
    "LiteLLM_TeamMembership",
    "LiteLLM_OrganizationMembership",
    "LiteLLM_TeamTable",
)


class _PlanNode(BaseModel, frozen=True):
    rows: float = Field(default=0, alias="Actual Rows")
    removed: float = Field(default=0, alias="Rows Removed by Filter")
    loops: float = Field(default=0, alias="Actual Loops")
    plans: tuple["_PlanNode", ...] = Field(default=(), alias="Plans")

    def examined_rows(self) -> float:
        return (self.rows + self.removed) * self.loops + sum(plan.examined_rows() for plan in self.plans)


class _QueryPlan(BaseModel, frozen=True):
    plan: _PlanNode = Field(alias="Plan")


def test_customer_budget_guard_keeps_generic_plan_work_bounded() -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        connection.execute('CREATE TABLE "LiteLLM_BudgetTable" (budget_id text PRIMARY KEY)')
        connection.execute('CREATE TABLE "LiteLLM_EndUserTable" (user_id text PRIMARY KEY, budget_id text)')
        connection.execute(
            'CREATE TABLE "LiteLLM_VerificationToken" (token text PRIMARY KEY, budget_id text, metadata jsonb)'
        )
        for table in _REFERENCE_TABLES:
            connection.execute(
                sql.SQL("CREATE TABLE {} (budget_id text, metadata jsonb)").format(sql.Identifier(table))
            )
        connection.execute("INSERT INTO \"LiteLLM_BudgetTable\" SELECT 'budget-'||n FROM generate_series(0,500) n")
        connection.execute(
            "INSERT INTO \"LiteLLM_EndUserTable\" SELECT 'customer-'||n,'budget-'||n FROM generate_series(0,500) n"
        )
        key_count: Final = 50_000
        connection.execute(
            "INSERT INTO \"LiteLLM_VerificationToken\" SELECT 'key-'||n,NULL,"
            "jsonb_build_object('end_user_budget_id','unrelated-budget') FROM generate_series(1,%s) n",
            (key_count,),
        )
        for migration in _INDEX_MIGRATIONS:
            connection.execute((_MIGRATIONS / migration / "migration.sql").read_text())
        connection.execute("ANALYZE")
        connection.execute("SET plan_cache_mode = force_generic_plan")
        connection.execute("PREPARE guard(text,text) AS " + _CUSTOMER_SHARED_BUDGET_SQL)
        plan_row: Final = connection.execute(
            "EXPLAIN (ANALYZE, FORMAT JSON) EXECUTE guard('budget-0','customer-0')"
        ).fetchone()
        assert plan_row is not None
        plan: Final = TypeAdapter(tuple[_QueryPlan, ...]).validate_python(plan_row[0])[0].plan
        assert plan.examined_rows() < key_count / 100
        assert connection.execute("EXECUTE guard('budget-0','customer-0')").fetchall() == []
        for metadata in ('{"end_user_budget_id": ["budget-0"]}', '{"end_user_budget_id": null}'):
            connection.execute(
                "INSERT INTO \"LiteLLM_VerificationToken\" VALUES ('malformed',NULL,%s::jsonb)", (metadata,)
            )
            assert connection.execute("EXECUTE guard('budget-0','customer-0')").fetchall() == []
            connection.execute("DELETE FROM \"LiteLLM_VerificationToken\" WHERE token='malformed'")
        connection.execute(
            'INSERT INTO "LiteLLM_VerificationToken" VALUES '
            "('metadata-budget',NULL,jsonb_build_object('end_user_budget_id','budget-0'))"
        )
        assert connection.execute("EXECUTE guard('budget-0','customer-0')").fetchall() == [("budget-0",)]
        connection.execute("DELETE FROM \"LiteLLM_VerificationToken\" WHERE token='metadata-budget'")
        connection.execute("INSERT INTO \"LiteLLM_VerificationToken\" VALUES ('direct-budget','budget-0','{}')")
        assert connection.execute("EXECUTE guard('budget-0','customer-0')").fetchall() == [("budget-0",)]
