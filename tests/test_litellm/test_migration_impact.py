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


BASE_SCHEMA: Final = """
enum KeyType {
  llm_api
  management
}

model LiteLLM_VerificationToken {
    token      String   @id
    key_name   String?
    spend      Float    @default(0.0)
    models     String[]
    team_id    String?
    key_type   KeyType?
    litellm_budget_table LiteLLM_BudgetTable? @relation(fields: [budget_id], references: [budget_id])
    budget_id  String?
    @@index([team_id])
}

model LiteLLM_BudgetTable {
    budget_id  String @id
    max_budget Float?
    keys       LiteLLM_VerificationToken[]
}

model LiteLLM_ModelTable {
    id            Int      @id @default(autoincrement())
    model_aliases Json?    @map("aliases")
    created_at    DateTime @default(now()) @map("created_at")
}

model ClaudeCodePlugin {
    id String @id
    @@map("LiteLLM_ClaudeCodePluginTable")
}
"""


def _context(
    fresh: frozenset[str] = frozenset(),
    reads: tuple[object, ...] = (),
    schema: object = None,
    dropped: frozenset[str] = frozenset(),
    defaulted: frozenset[tuple[str, str]] = frozenset(),
    base_unique: dict[str, tuple[str, tuple[str, ...]]] | None = None,
) -> object:
    return impact.UpgradeContext(fresh, reads, schema, dropped, defaulted, base_unique or {})


def _classify(
    statement: str,
    fresh: frozenset[str] = frozenset(),
    reads: tuple[object, ...] = (),
    schema: object = None,
    dropped: frozenset[str] = frozenset(),
    defaulted: frozenset[tuple[str, str]] = frozenset(),
    base_unique: dict[str, tuple[str, tuple[str, ...]]] | None = None,
) -> object:
    context = _context(fresh, reads, schema, dropped, defaulted, base_unique)
    return impact.classify("20260713230852_add_key_type", statement, context)


def _base_schema() -> object:
    return impact.parse_prisma_schema(BASE_SCHEMA)


def test_split_statements_descends_into_dollar_quoted_blocks() -> None:
    statements = impact.split_statements(RENAME_IN_DO_BLOCK)
    renames = [statement for statement in statements if "RENAME COLUMN" in statement]
    assert len(renames) == 1, statements
    assert '"api_key_id" TO "target_id"' in renames[0]


def test_split_statements_drops_comments() -> None:
    statements = impact.split_statements('-- add a column\nALTER TABLE "T" ADD COLUMN "c" TEXT; /* trailing */')
    assert statements == ('ALTER TABLE "T" ADD COLUMN "c" TEXT',)


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
    assert impact.star_reads_in_source(VIEW_SOURCE, "litellm/proxy/db/create_views.py") == ()


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


def test_constraint_on_a_column_the_old_client_writes_can_reject_its_writes() -> None:
    # NOT VALID only skips the scan of existing rows; new writes from the old pods are still checked.
    schema = _base_schema()
    checked = _classify(
        'ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "fk" FOREIGN KEY ("budget_id") '
        'REFERENCES "LiteLLM_BudgetTable"("budget_id") NOT VALID',
        schema=schema,
    )
    assert checked.severity == impact.WRITE_REJECT
    assert "`budget_id`" in checked.effect


def test_constraint_on_a_column_the_old_client_does_not_know_is_rated_by_its_scan() -> None:
    schema = _base_schema()
    validating = _classify(
        'ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "fk" FOREIGN KEY ("project_id") '
        'REFERENCES "LiteLLM_ProjectTable"("project_id")',
        schema=schema,
    )
    deferred = _classify(
        'ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "fk" FOREIGN KEY ("project_id") '
        'REFERENCES "LiteLLM_ProjectTable"("project_id") NOT VALID',
        schema=schema,
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
    assert "| **PREPARED-PLAN** |" in rendered
    assert impact.PROCEDURE[impact.PREPARED_PLAN] in rendered
    assert "`SELECT v.*` on `LiteLLM_VerificationToken`" in rendered


def test_markdown_caps_the_table_and_says_where_the_rest_is() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "key_type" TEXT', reads=(_auth_read(),))
    rendered = impact.render_markdown(_report(findings=tuple([finding] * (impact.MAX_ROWS + 3))))
    assert rendered.count("| **PREPARED-PLAN** |") == impact.MAX_ROWS
    assert "...and 3 more" in rendered


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
    assert finding.severity == impact.WRITE_REJECT
    assert "backfill" not in finding.effect
    backfill = _classify('UPDATE "LiteLLM_SpendLogs" SET "created_at" = "startTime" WHERE "created_at" IS NULL')
    assert backfill.severity == impact.LOCK


def test_index_rename_is_not_breaking() -> None:
    finding = _classify('ALTER INDEX "old_idx" RENAME TO "new_idx"')
    assert finding.severity == impact.INFO


SPEND_SCHEMA: Final = "model LiteLLM_DailyTagSpend {\n  id String @id\n  prompt_tokens Int\n}\n"


def test_widening_a_type_is_a_prepared_plan_change() -> None:
    finding = _classify(
        'ALTER TABLE "LiteLLM_DailyTagSpend" ALTER COLUMN "prompt_tokens" SET DATA TYPE BIGINT',
        schema=impact.parse_prisma_schema(SPEND_SCHEMA),
    )
    assert finding.severity == impact.PREPARED_PLAN
    assert "cached plan must not change result type" in finding.effect


def test_a_type_the_old_client_cannot_read_is_breaking() -> None:
    finding = _classify(
        'ALTER TABLE "LiteLLM_DailyTagSpend" ALTER COLUMN "prompt_tokens" SET DATA TYPE TEXT',
        schema=impact.parse_prisma_schema(SPEND_SCHEMA),
    )
    assert finding.severity == impact.BREAKING
    assert "`prompt_tokens` to TEXT" in finding.effect


def test_type_change_without_a_schema_to_check_against_is_breaking() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_DailyTagSpend" ALTER COLUMN "prompt_tokens" SET DATA TYPE BIGINT')
    assert finding.severity == impact.BREAKING


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
    (repo / "schema.prisma").write_text(BASE_SCHEMA, encoding="utf-8")
    _write_migration(
        repo,
        "20250101000000_baseline",
        'CREATE TABLE "LiteLLM_VerificationToken" ("token" TEXT PRIMARY KEY);\n'
        'CREATE TABLE "LiteLLM_TeamTable" ("team_id" TEXT PRIMARY KEY);\n'
        'CREATE UNIQUE INDEX "LiteLLM_VerificationToken_key_name_key" ON "LiteLLM_VerificationToken"("key_name");\n',
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
        'CREATE INDEX "new_idx" ON "LiteLLM_NewTable"("id");\n'
        # A column the base schema never declared: the old client cannot miss it.
        'ALTER TABLE "LiteLLM_TeamTable" DROP COLUMN IF EXISTS "legacy_flag";\n'
        # A column it does declare: the old client selects it on every read.
        'ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN "key_name";\n',
    )
    _write_migration(
        repo,
        "20250202000000_scope_key_name_by_issuer",
        # The jwt_issuer pattern: drop the old key, add a defaulted column, recreate the key wider.
        'DROP INDEX IF EXISTS "LiteLLM_VerificationToken_key_name_key";\n'
        'ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN IF NOT EXISTS "issuer" TEXT NOT NULL DEFAULT \'\';\n'
        'CREATE UNIQUE INDEX "LiteLLM_VerificationToken_issuer_key_name_key" ON "LiteLLM_VerificationToken"'
        '("issuer", "key_name");\n',
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "head")
    _git(repo, "tag", "v1.1.0")
    return repo


def test_report_rates_migrations_against_the_queries_the_old_pods_run(tmp_path: Path) -> None:
    repo = _two_version_repo(tmp_path)
    report = impact.build_report(repo, "v1.0.0", "v1.1.0")

    assert report.migrations == ("20250201000000_add_key_type", "20250202000000_scope_key_name_by_issuer")
    # The whole-row read exists only at v1.0.0; that is the version whose plans go stale.
    assert [read.table for read in report.star_reads] == ["LiteLLM_VerificationToken"]
    assert report.star_reads[0].location.startswith("litellm/proxy/utils.py:")

    by_statement = {finding.statement: finding for finding in report.findings}
    add_column = by_statement['ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "key_type" TEXT']
    assert add_column.severity == impact.PREPARED_PLAN
    new_index = by_statement['CREATE INDEX "new_idx" ON "LiteLLM_NewTable"("id")']
    assert new_index.severity == impact.INFO
    unknown_drop = by_statement['ALTER TABLE "LiteLLM_TeamTable" DROP COLUMN IF EXISTS "legacy_flag"']
    assert unknown_drop.severity == impact.INFO
    known_drop = by_statement['ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN "key_name"']
    assert known_drop.severity == impact.BREAKING
    widened_key = by_statement[
        'CREATE UNIQUE INDEX "LiteLLM_VerificationToken_issuer_key_name_key" ON "LiteLLM_VerificationToken"'
        '("issuer", "key_name")'
    ]
    assert widened_key.severity == impact.LOCK  # widens the key it drops, so nothing new is rejected
    assert report.schema_tables == 4
    assert report.worst == impact.BREAKING


def test_report_for_the_same_ref_twice_has_nothing_to_say(tmp_path: Path) -> None:
    repo = _two_version_repo(tmp_path)
    report = impact.build_report(repo, "v1.1.0", "v1.1.0")
    assert report.migrations == ()
    assert report.findings == ()
    assert "No database schema changes since `v1.1.0`." in impact.render_markdown(report)


def test_schema_parser_reads_columns_maps_and_enums_and_skips_relations() -> None:
    schema = _base_schema()
    token = schema["LiteLLM_VerificationToken"]
    assert token["token"] == impact.Column(required=True, prisma_type="String")
    assert token["key_name"] == impact.Column(required=False, prisma_type="String")
    assert token["models"].required  # a list column is NOT NULL
    assert token["key_type"].prisma_type == "KeyType"  # enum-typed column
    assert "litellm_budget_table" not in token  # relation field, not a column
    assert "aliases" in schema["LiteLLM_ModelTable"]  # @map renames the column
    assert "model_aliases" not in schema["LiteLLM_ModelTable"]
    assert "LiteLLM_ClaudeCodePluginTable" in schema  # @@map renames the table
    assert "ClaudeCodePlugin" not in schema


def test_dropping_a_column_the_old_client_never_selects_is_not_breaking() -> None:
    schema = _base_schema()
    known = _classify('ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN "key_name"', schema=schema)
    unknown = _classify('ALTER TABLE "LiteLLM_VerificationToken" DROP COLUMN "settings_updated_at"', schema=schema)
    assert known.severity == impact.BREAKING
    assert "`key_name`" in known.effect
    assert unknown.severity == impact.INFO


def test_renaming_a_column_is_rated_by_whether_the_old_client_uses_it() -> None:
    schema = _base_schema()
    known = _classify('ALTER TABLE "LiteLLM_VerificationToken" RENAME COLUMN "team_id" TO "owner_team"', schema=schema)
    unknown = _classify('ALTER TABLE "LiteLLM_VerificationToken" RENAME COLUMN "tmp" TO "final"', schema=schema)
    assert known.severity == impact.BREAKING
    assert unknown.severity == impact.INFO


def test_set_not_null_is_breaking_only_where_the_old_client_may_write_null() -> None:
    schema = _base_schema()
    optional_at_base = _classify(
        'ALTER TABLE "LiteLLM_VerificationToken" ALTER COLUMN "key_name" SET NOT NULL', schema=schema
    )
    required_at_base = _classify(
        'ALTER TABLE "LiteLLM_VerificationToken" ALTER COLUMN "spend" SET NOT NULL', schema=schema
    )
    assert optional_at_base.severity == impact.BREAKING
    assert required_at_base.severity == impact.INFO


def test_dropping_a_table_the_old_client_does_not_know_is_informational() -> None:
    schema = _base_schema()
    known = _classify('DROP TABLE "LiteLLM_BudgetTable"', schema=schema)
    unknown = _classify('DROP TABLE "LiteLLM_TmpTable"', schema=schema)
    assert known.severity == impact.BREAKING
    assert unknown.severity == impact.INFO


def test_unique_index_on_columns_the_old_client_writes_can_reject_duplicates() -> None:
    finding = _classify(
        'CREATE UNIQUE INDEX "LiteLLM_VerificationToken_key_name_key" ON "LiteLLM_VerificationToken"("key_name")',
        schema=_base_schema(),
    )
    assert finding.severity == impact.WRITE_REJECT


TOKEN_KEY_NAME_KEY: Final = {"LiteLLM_VerificationToken_key_name_key": ("LiteLLM_VerificationToken", ("key_name",))}


def test_unique_index_that_widens_a_dropped_key_is_only_a_lock() -> None:
    # The Daily*Spend migrations drop the old unique key and create a wider one in the same file: every
    # duplicate the new key rejects, the old key already rejected.
    finding = _classify(
        'CREATE UNIQUE INDEX "LiteLLM_VerificationToken_key_name_team_id_key" ON "LiteLLM_VerificationToken"'
        '("key_name", "team_id")',
        schema=_base_schema(),
        dropped=frozenset({"LiteLLM_VerificationToken_key_name_key"}),
        base_unique=TOKEN_KEY_NAME_KEY,
    )
    assert finding.severity == impact.LOCK


def test_unique_index_that_narrows_a_dropped_key_can_reject_writes() -> None:
    # Dropping (key_name, team_id) for (key_name) is stricter: rows that only differed by team now collide.
    finding = _classify(
        'CREATE UNIQUE INDEX "LiteLLM_VerificationToken_key_name_key" ON "LiteLLM_VerificationToken"("key_name")',
        schema=_base_schema(),
        dropped=frozenset({"LiteLLM_VerificationToken_key_name_team_id_key"}),
        base_unique={
            "LiteLLM_VerificationToken_key_name_team_id_key": ("LiteLLM_VerificationToken", ("key_name", "team_id"))
        },
    )
    assert finding.severity == impact.WRITE_REJECT


def test_unique_index_over_a_new_defaulted_column_still_binds_the_old_pods() -> None:
    # Old pods insert the default, so the new column does not make their rows distinct.
    finding = _classify(
        'CREATE UNIQUE INDEX "idx" ON "LiteLLM_VerificationToken"("key_name", "jwt_issuer")',
        schema=_base_schema(),
        defaulted=frozenset({("LiteLLM_VerificationToken", "jwt_issuer")}),
    )
    assert finding.severity == impact.WRITE_REJECT


def test_check_constraint_with_unquoted_columns_is_matched_against_the_schema() -> None:
    finding = _classify(
        'ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "chk" CHECK (key_name <> \'active\') NOT VALID',
        schema=_base_schema(),
    )
    assert finding.severity == impact.WRITE_REJECT
    assert "`key_name`" in finding.effect
    assert "active" not in finding.effect  # a string literal is not a column


def test_constraint_without_a_schema_to_check_against_is_write_reject() -> None:
    finding = _classify('ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT "chk" CHECK (status = \'active\')')
    assert finding.severity == impact.WRITE_REJECT


def test_defaulted_columns_are_collected_per_clause() -> None:
    statements = [
        'ALTER TABLE "T" ADD COLUMN "a" TEXT NOT NULL DEFAULT \'\', ADD COLUMN "b" TEXT',
        'ALTER TABLE "U" ADD COLUMN IF NOT EXISTS "c" INT DEFAULT 0',
    ]
    assert impact._defaulted_columns(statements) == frozenset({("T", "a"), ("U", "c")})


def test_unique_index_on_a_new_column_is_only_a_lock() -> None:
    finding = _classify(
        'CREATE UNIQUE INDEX "idx" ON "LiteLLM_VerificationToken"("settings_updated_at")', schema=_base_schema()
    )
    assert finding.severity == impact.LOCK


def test_markdown_names_the_schema_it_used_as_evidence() -> None:
    report = impact.Report(
        base="v1.93.0", head="v1.99.0", migrations=("m",), findings=(), star_reads=(), schema_tables=65
    )
    assert "`schema.prisma` at `v1.93.0` (65 tables)" in impact.render_markdown(report)
    blind = impact.Report(base="v1.93.0", head="v1.99.0", migrations=("m",), findings=(), star_reads=())
    assert "no `schema.prisma` at `v1.93.0`" in impact.render_markdown(blind)
