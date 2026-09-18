"""Tests for ci_cd/migration_impact.py.

The fixtures are the real statements from the migrations that caused the rolling-upgrade
failures this report exists to announce: the `key_type` column added to
`LiteLLM_VerificationToken` (issue #36418) and the `api_key_id` -> `target_id` rename hidden
inside a `DO $$` block.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[2]
CI_CD: Final = ROOT / "ci_cd"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, CI_CD / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


impact: Final = _load("migration_impact")

AUTH_QUERY_SOURCE: Final = '''
                    sql_query = """
                        SELECT
                            v.*,
                            t.spend AS team_spend,
                            -- Added comma to separate b.* columns
                            b.max_budget AS litellm_budget_table_max_budget
                        FROM "LiteLLM_VerificationToken" AS v
                        LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id
                        LEFT JOIN "LiteLLM_BudgetTable" AS b ON v.budget_id = b.budget_id
                        WHERE v.token = $1
                    """
'''

VIEW_SOURCE: Final = '''
    await db.execute_raw("""
        CREATE OR REPLACE VIEW "LiteLLM_VerificationTokenView" AS
        SELECT v.*, t.spend AS team_spend
        FROM "LiteLLM_VerificationToken" v
        LEFT JOIN "LiteLLM_TeamTable" t ON v.team_id = t.team_id;
    """)
'''

RENAME_IN_DO_BLOCK: Final = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'LiteLLM_ShadowEvalJob' AND column_name = 'api_key_id'
    ) THEN
        ALTER TABLE "LiteLLM_ShadowEvalJob" RENAME COLUMN "api_key_id" TO "target_id";
    END IF;
END $$;

ALTER TABLE "LiteLLM_ShadowEvalJob" ADD COLUMN IF NOT EXISTS "target_type" TEXT NOT NULL DEFAULT 'key';
"""


def _auth_read() -> object:
    reads = impact.star_reads_in_source(AUTH_QUERY_SOURCE, "litellm/proxy/utils.py")
    assert len(reads) == 1
    return reads[0]


def _classify(statement: str, fresh: frozenset[str] = frozenset(), reads: tuple[object, ...] = ()) -> object:
    return impact.classify("20260713230852_add_key_type", statement, fresh, reads)


def test_split_statements_descends_into_dollar_quoted_blocks() -> None:
    statements = impact.split_statements(RENAME_IN_DO_BLOCK)
    renames = [statement for statement in statements if "RENAME COLUMN" in statement]
    assert len(renames) == 1, statements
    assert '"api_key_id" TO "target_id"' in renames[0]


def test_split_statements_drops_comments() -> None:
    statements = impact.split_statements('-- add a column\nALTER TABLE "T" ADD COLUMN "c" TEXT; /* trailing */')
    assert statements == ['ALTER TABLE "T" ADD COLUMN "c" TEXT']


def test_star_read_discovery_records_table_alias_and_joins() -> None:
    read = _auth_read()
    assert read.table == "LiteLLM_VerificationToken"
    assert read.alias == "v"
    assert set(read.joined) == {"LiteLLM_TeamTable", "LiteLLM_BudgetTable"}
    assert read.location.startswith("litellm/proxy/utils.py:")


def test_star_read_discovery_ignores_aliases_that_only_appear_in_a_comment() -> None:
    # `-- Added comma to separate b.* columns` must not register LiteLLM_BudgetTable as whole-row read.
    tables = {read.table for read in impact.star_reads_in_source(AUTH_QUERY_SOURCE, "litellm/proxy/utils.py")}
    assert tables == {"LiteLLM_VerificationToken"}


def test_star_read_discovery_ignores_view_definitions() -> None:
    # A view expands `*` when it is created, so it holds no per-connection prepared plan.
    assert impact.star_reads_in_source(VIEW_SOURCE, "litellm/proxy/db/create_views.py") == []


def test_add_column_on_a_whole_row_read_table_is_a_prepared_plan_change() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "key_type" TEXT', reads=(_auth_read(),))
    assert finding.severity == impact.PREPARED_PLAN
    assert finding.table == "LiteLLM_VerificationToken"
    assert "cached plan must not change result type" in finding.effect
    assert "litellm/proxy/utils.py" in finding.effect


def test_add_column_elsewhere_is_not_reported() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_AuditLog" ADD COLUMN "note" TEXT', reads=(_auth_read(),))
    assert finding.severity == impact.INFO


def test_drop_column_outranks_the_prepared_plan_rule() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN "key_type"', reads=(_auth_read(),))
    assert finding.severity == impact.BREAKING


def test_rename_is_breaking() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_ShadowEvalJob" RENAME COLUMN "api_key_id" TO "target_id"')
    assert finding.severity == impact.BREAKING


def test_not_null_column_is_breaking_only_without_a_default() -> None:
    without_default = _classify('ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN "created_at" TIMESTAMP(3) NOT NULL')
    with_default = _classify(
        'ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP'
    )
    assert without_default.severity == impact.BREAKING
    assert with_default.severity == impact.INFO


def test_index_builds_are_rated_by_whether_they_lock() -> None:
    blocking = _classify('CREATE INDEX "idx" ON "LiteLLM_SpendLogs"("api_key")')
    concurrent = _classify('CREATE INDEX CONCURRENTLY "idx" ON "LiteLLM_SpendLogs"("api_key")')
    assert blocking.severity == impact.LOCK
    assert concurrent.severity == impact.INFO


def test_unvalidated_constraint_is_not_a_lock() -> None:
    validating = _classify('ALTER TABLE "LiteLLM_SpendLogs" ADD CONSTRAINT "fk" FOREIGN KEY ("t") REFERENCES "T"("id")')
    deferred = _classify(
        'ALTER TABLE "LiteLLM_SpendLogs" ADD CONSTRAINT "fk" FOREIGN KEY ("t") REFERENCES "T"("id") NOT VALID'
    )
    assert validating.severity == impact.LOCK
    assert deferred.severity == impact.INFO


def test_changes_to_a_table_created_by_the_same_upgrade_are_not_reported() -> None:
    finding = _classify(
        'CREATE INDEX "idx" ON "LiteLLM_NewTable"("id")',
        fresh=frozenset({"LiteLLM_NewTable"}),
    )
    assert finding.severity == impact.INFO
    assert "created by this same upgrade" in finding.effect


def _report(findings: tuple[object, ...], migrations: tuple[str, ...] = ("20260713230852_add_key_type",)) -> object:
    return impact.Report(
        base="v1.93.0", head="v1.99.0", migrations=migrations, findings=findings, star_reads=(_auth_read(),)
    )


def test_markdown_for_an_upgrade_with_no_migrations() -> None:
    rendered = impact.render_markdown(_report(findings=(), migrations=()))
    assert "No database schema changes since `v1.93.0`." in rendered
    assert "|" not in rendered


def test_markdown_leads_with_the_worst_finding_and_its_procedure() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "key_type" TEXT', reads=(_auth_read(),))
    rendered = impact.render_markdown(_report(findings=(finding,)))
    assert "20260713230852_add_key_type" in rendered
    assert impact.SEVERITY_ICON[impact.PREPARED_PLAN] in rendered
    assert impact.PROCEDURE[impact.PREPARED_PLAN] in rendered
    assert "`SELECT v.*` on `LiteLLM_VerificationToken`" in rendered


def test_markdown_caps_the_table_and_says_where_the_rest_is() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "key_type" TEXT', reads=(_auth_read(),))
    rendered = impact.render_markdown(_report(findings=tuple([finding] * (impact.MAX_ROWS + 3))))
    assert rendered.count("| ⚠️ |") == impact.MAX_ROWS
    assert "…and 3 more" in rendered


def test_worst_severity_ranks_breaking_above_the_rest() -> None:
    lock = _classify('CREATE INDEX "idx" ON "LiteLLM_SpendLogs"("api_key")')
    breaking = _classify('ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN "key_type"', reads=(_auth_read(),))
    assert _report(findings=(lock, breaking)).worst == impact.BREAKING
    assert _report(findings=(lock,)).worst == impact.LOCK


def _git(repo: Path, *args: str) -> None:
    subprocess.run(("git", "-C", str(repo), *args), check=True, capture_output=True)


def _commit(repo: Path, message: str) -> None:
    (repo / "file.txt").write_text(message, encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-m", message)


def test_default_base_is_the_newest_stable_release_not_the_newest_tag(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _commit(repo, "first")
    _git(repo, "tag", "v1.98.0")
    _commit(repo, "second")
    _git(repo, "tag", "v1.99.0")
    _commit(repo, "third")
    _git(repo, "tag", "v1.100.0-rc.1")
    _commit(repo, "fourth")

    # The release being cut sits on top of an rc tag; the operator is upgrading from v1.99.0.
    assert impact.default_base(repo, "HEAD") == "v1.99.0"


def test_default_base_skips_the_tag_that_points_at_the_release_itself(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    _commit(repo, "first")
    _git(repo, "tag", "v1.99.0")
    _commit(repo, "second")
    _git(repo, "tag", "v1.100.0")

    assert impact.default_base(repo, "v1.100.0") == "v1.99.0"


def test_foreign_key_on_update_is_not_a_backfill() -> None:
    # `ON UPDATE CASCADE` is part of the constraint; only `UPDATE ... SET` rewrites rows.
    finding = _classify(
        'ALTER TABLE "LiteLLM_JWTKeyMapping" ADD CONSTRAINT "fk" FOREIGN KEY ("token") '
        'REFERENCES "LiteLLM_VerificationToken"("token") ON DELETE CASCADE ON UPDATE CASCADE NOT VALID'
    )
    assert finding.severity == impact.INFO
    backfill = _classify('UPDATE "LiteLLM_SpendLogs" SET "created_at" = "startTime" WHERE "created_at" IS NULL')
    assert backfill.severity == impact.LOCK


def test_index_rename_is_not_breaking() -> None:
    finding = _classify('ALTER INDEX "old_idx" RENAME TO "new_idx"')
    assert finding.severity == impact.INFO


def test_type_change_is_a_prepared_plan_change_not_a_permanent_break() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_DailyTagSpend" ALTER COLUMN "prompt_tokens" SET DATA TYPE BIGINT')
    assert finding.severity == impact.PREPARED_PLAN
    assert "cached plan must not change result type" in finding.effect


def test_not_null_is_judged_per_add_column_clause() -> None:
    one_default_is_not_enough = _classify(
        'ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN "a" TEXT NOT NULL DEFAULT \'x\', ADD COLUMN "b" TEXT NOT NULL'
    )
    assert one_default_is_not_enough.severity == impact.BREAKING


def test_drop_not_null_beside_an_add_column_is_not_breaking() -> None:
    # 20250707230009_add_mcp_namespaced_tool_name: the `NOT NULL` here belongs to `DROP NOT NULL`.
    finding = _classify(
        'ALTER TABLE "LiteLLM_DailyTagSpend" ADD COLUMN IF NOT EXISTS "mcp_namespaced_tool_name" TEXT, '
        'ALTER COLUMN "model" DROP NOT NULL'
    )
    assert finding.severity == impact.INFO


def test_insert_is_informational() -> None:
    finding = _classify('INSERT INTO "LiteLLM_Config" ("param_name") VALUES (\'x\')')
    assert finding.severity == impact.INFO


def test_markdown_escapes_pipes_inside_a_statement() -> None:
    finding = _classify("UPDATE \"LiteLLM_SpendLogs\" SET \"note\" = 'a' || 'b'")
    rendered = impact.render_markdown(_report(findings=(finding,)))
    assert "'a' \\|\\| 'b'" in rendered
    assert "'a' || 'b'" not in rendered


BASE_SOURCE: Final = '''
SQL = """
    SELECT v.*, t.spend AS team_spend
    FROM "LiteLLM_VerificationToken" AS v
    LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id
    WHERE v.token = $1
"""
'''

HEAD_SOURCE: Final = '''
SQL = """
    SELECT v.token, v.key_type, t.spend AS team_spend
    FROM "LiteLLM_VerificationToken" AS v
    LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id
    WHERE v.token = $1
"""
'''


def _write_migration(repo: Path, name: str, sql: str) -> None:
    directory = repo / impact.MIGRATIONS_DIR / name
    directory.mkdir(parents=True)
    (directory / "migration.sql").write_text(sql, encoding="utf-8")


def _two_version_repo(tmp_path: Path) -> Path:
    """A repo whose base tag reads the token table whole-row and whose head no longer does."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    (repo / "litellm" / "proxy").mkdir(parents=True)
    (repo / "litellm" / "proxy" / "utils.py").write_text(BASE_SOURCE, encoding="utf-8")
    _write_migration(
        repo,
        "20250101000000_baseline",
        'CREATE TABLE "LiteLLM_VerificationToken" ("token" TEXT PRIMARY KEY);\n'
        'CREATE TABLE "LiteLLM_TeamTable" ("team_id" TEXT PRIMARY KEY);\n',
    )
    (repo / impact.MIGRATIONS_DIR / "migration_lock.toml").write_text('provider = "postgresql"\n', encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "tag", "v1.0.0")

    (repo / "litellm" / "proxy" / "utils.py").write_text(HEAD_SOURCE, encoding="utf-8")
    _write_migration(
        repo,
        "20250201000000_add_key_type",
        # Re-declaring an existing table must not make it "fresh"; the ADD COLUMN after it is real.
        'CREATE TABLE IF NOT EXISTS "LiteLLM_VerificationToken" ("token" TEXT PRIMARY KEY);\n'
        'ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "key_type" TEXT;\n'
        'CREATE TABLE "LiteLLM_NewTable" ("id" TEXT PRIMARY KEY);\n'
        'CREATE INDEX "new_idx" ON "LiteLLM_NewTable"("id");\n',
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "head")
    _git(repo, "tag", "v1.1.0")
    return repo


def test_report_rates_migrations_against_the_queries_the_old_pods_run(tmp_path: Path) -> None:
    repo = _two_version_repo(tmp_path)
    report = impact.build_report(repo, "v1.0.0", "v1.1.0")

    assert report.migrations == ("20250201000000_add_key_type",)
    # The whole-row read exists only at v1.0.0; that is the version whose plans go stale.
    assert [read.table for read in report.star_reads] == ["LiteLLM_VerificationToken"]
    assert report.star_reads[0].location.startswith("litellm/proxy/utils.py:")

    by_statement = {finding.statement: finding for finding in report.findings}
    add_column = by_statement['ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "key_type" TEXT']
    assert add_column.severity == impact.PREPARED_PLAN
    new_index = by_statement['CREATE INDEX "new_idx" ON "LiteLLM_NewTable"("id")']
    assert new_index.severity == impact.INFO
    assert report.worst == impact.PREPARED_PLAN


def test_report_for_the_same_ref_twice_has_nothing_to_say(tmp_path: Path) -> None:
    repo = _two_version_repo(tmp_path)
    report = impact.build_report(repo, "v1.1.0", "v1.1.0")
    assert report.migrations == ()
    assert report.findings == ()
    assert "No database schema changes since `v1.1.0`." in impact.render_markdown(report)
