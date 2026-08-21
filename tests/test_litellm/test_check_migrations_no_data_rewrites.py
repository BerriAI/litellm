"""Tests for tests/code_coverage_tests/check_migrations_no_data_rewrites.py.

The checker reads migration.sql as SQL rather than as text, so the cases that matter
are the ones a grep would get wrong: `ON DELETE CASCADE` in a foreign key (60-odd
occurrences in the shipped migrations), an `UPDATE` inside a string literal or a
comment, and an `UPDATE` hidden in the `DO $$ ... $$` block this repo uses for
conditional DDL.
"""

import importlib.util
import sys
from pathlib import Path

_CHECKER_PATH = Path(__file__).resolve().parents[1] / "code_coverage_tests" / "check_migrations_no_data_rewrites.py"
_SPEC = importlib.util.spec_from_file_location("check_migrations_no_data_rewrites", _CHECKER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
checker = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = checker
_SPEC.loader.exec_module(checker)


def _scan(tmp_path: Path, sql: str) -> tuple:
    directory = tmp_path / "20260101000000_fixture"
    directory.mkdir(exist_ok=True)
    (directory / "migration.sql").write_text(sql, encoding="utf-8")
    return checker.scan_migration(directory)


def _keywords(tmp_path: Path, sql: str) -> tuple:
    return tuple(violation.keyword for violation in _scan(tmp_path, sql))


class TestRowRewritesAreFlagged:
    def test_update_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, 'UPDATE "Foo" SET "a" = 1;') == ("UPDATE",)

    def test_delete_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, 'DELETE FROM "Foo" WHERE "a" IS NULL;') == ("DELETE",)

    def test_update_without_trailing_semicolon_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, 'UPDATE "Foo" SET "a" = 1') == ("UPDATE",)

    def test_lowercase_update_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, 'update "Foo" set "a" = 1;') == ("UPDATE",)

    def test_merge_is_flagged(self, tmp_path):
        sql = 'MERGE INTO "Foo" t USING "Bar" s ON t."id" = s."id" WHEN MATCHED THEN UPDATE SET "a" = s."a";'
        assert _keywords(tmp_path, sql) == ("MERGE",)

    def test_every_offending_statement_is_reported(self, tmp_path):
        sql = 'UPDATE "Foo" SET "a" = 1;\nALTER TABLE "Foo" ADD COLUMN "b" TEXT;\nDELETE FROM "Bar";'
        assert _keywords(tmp_path, sql) == ("UPDATE", "DELETE")

    def test_the_incident_migration_is_flagged(self, tmp_path):
        sql = (
            'UPDATE "LiteLLM_SpendLogs"\n'
            '   SET "created_at" = "endTime",\n'
            '       "updated_at" = "endTime"\n'
            ' WHERE "created_at" > "endTime" + interval \'1 hour\';\n'
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)


class TestSchemaStatementsPass:
    def test_on_delete_cascade_is_not_a_data_rewrite(self, tmp_path):
        sql = (
            'ALTER TABLE "A" ADD CONSTRAINT "A_b_fkey" FOREIGN KEY ("b") '
            'REFERENCES "B"("id") ON DELETE CASCADE ON UPDATE CASCADE;'
        )
        assert _keywords(tmp_path, sql) == ()

    def test_on_delete_set_null_is_not_a_data_rewrite(self, tmp_path):
        sql = (
            'ALTER TABLE "A" ADD CONSTRAINT "A_b_fkey" FOREIGN KEY ("b") '
            'REFERENCES "B"("id") ON DELETE SET NULL ON UPDATE CASCADE;'
        )
        assert _keywords(tmp_path, sql) == ()

    def test_add_column_with_default_passes(self, tmp_path):
        sql = 'ALTER TABLE "Foo" ADD COLUMN IF NOT EXISTS "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP;'
        assert _keywords(tmp_path, sql) == ()

    def test_drop_table_passes(self, tmp_path):
        assert _keywords(tmp_path, 'DROP TABLE IF EXISTS "Foo";') == ()

    def test_empty_file_passes(self, tmp_path):
        assert _keywords(tmp_path, "") == ()

    def test_only_comments_passes(self, tmp_path):
        assert _keywords(tmp_path, "-- nothing to do here\n") == ()


class TestInsert:
    def test_insert_values_is_bounded_and_passes(self, tmp_path):
        assert _keywords(tmp_path, "INSERT INTO \"Foo\" (\"id\") VALUES ('a'), ('b');") == ()

    def test_insert_select_scans_and_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, 'INSERT INTO "Foo" ("id") SELECT "id" FROM "Bar";') == ("INSERT ... SELECT",)


class TestCommonTableExpressions:
    def test_cte_led_update_is_flagged(self, tmp_path):
        sql = 'WITH batch AS (SELECT "id" FROM "Foo" LIMIT 100) UPDATE "Foo" SET "a" = 1 FROM batch;'
        assert _keywords(tmp_path, sql) == ("WITH ... UPDATE",)

    def test_cte_led_delete_is_flagged(self, tmp_path):
        sql = 'WITH batch AS (SELECT "id" FROM "Foo" LIMIT 100) DELETE FROM "Foo" USING batch;'
        assert _keywords(tmp_path, sql) == ("WITH ... DELETE",)

    def test_cte_led_insert_select_is_flagged(self, tmp_path):
        sql = 'WITH batch AS (SELECT "id" FROM "Bar") INSERT INTO "Foo" ("id") SELECT "id" FROM batch;'
        assert _keywords(tmp_path, sql) == ("WITH ... INSERT ... SELECT",)

    def test_read_only_cte_passes(self, tmp_path):
        sql = 'WITH batch AS (SELECT "id" FROM "Foo") SELECT count(*) FROM batch;'
        assert _keywords(tmp_path, sql) == ()


class TestDollarQuotedBlocks:
    def test_update_inside_do_block_is_flagged(self, tmp_path):
        sql = 'DO $$\nBEGIN\n    UPDATE "Foo" SET "a" = 1;\nEND $$;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_conditional_ddl_do_block_passes(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            "    IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'x') THEN\n"
            '        ALTER TABLE "Foo" DROP CONSTRAINT "x";\n'
            "    END IF;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_guarded_update_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            '    IF EXISTS (SELECT 1 FROM "Foo") THEN\n'
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    END IF;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_guard_with_a_nested_call_still_flags_the_update(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            '    IF EXISTS (SELECT 1 FROM "Foo" WHERE lower("a") = \'x\' UNION SELECT 1) THEN\n'
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    END IF;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_tagged_dollar_quote_is_scanned(self, tmp_path):
        sql = 'DO $body$\nBEGIN\n    DELETE FROM "Foo";\nEND $body$;'
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_tagged_dollar_quote_holds_an_apostrophe(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("t") VALUES ($body$don\'t$body$);\nUPDATE "Bar" SET "b" = 1;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_semicolons_inside_do_block_do_not_split_outer_statements(self, tmp_path):
        sql = 'DO $$ BEGIN PERFORM 1; END $$;\nALTER TABLE "Foo" ADD COLUMN "b" TEXT;'
        assert _keywords(tmp_path, sql) == ()


class TestQuotingAndComments:
    def test_update_inside_string_literal_passes(self, tmp_path):
        sql = 'ALTER TABLE "Foo" ADD COLUMN "note" TEXT NOT NULL DEFAULT \'UPDATE nothing\';'
        assert _keywords(tmp_path, sql) == ()

    def test_escaped_quote_inside_string_does_not_leak(self, tmp_path):
        sql = "ALTER TABLE \"Foo\" ADD COLUMN \"note\" TEXT NOT NULL DEFAULT 'it''s fine';\n"
        assert _keywords(tmp_path, sql) == ()

    def test_update_inside_line_comment_passes(self, tmp_path):
        assert _keywords(tmp_path, '-- UPDATE "Foo" SET "a" = 1;\nDROP TABLE "Bar";') == ()

    def test_update_inside_block_comment_passes(self, tmp_path):
        assert _keywords(tmp_path, '/* UPDATE "Foo" SET "a" = 1; */\nDROP TABLE "Bar";') == ()

    def test_nested_block_comment_passes(self, tmp_path):
        sql = '/* outer /* UPDATE "Foo" SET "a" = 1; */ still comment */\nDROP TABLE "Bar";'
        assert _keywords(tmp_path, sql) == ()

    def test_nested_block_comment_masks_past_the_inner_close(self, tmp_path):
        sql = '/* outer /* inner */ UPDATE "Foo" SET "a" = 1; */\nDROP TABLE "Bar";'
        assert _keywords(tmp_path, sql) == ()

    def test_update_inside_quoted_identifier_passes(self, tmp_path):
        assert _keywords(tmp_path, 'ALTER TABLE "UPDATE Foo" ADD COLUMN "b" TEXT;') == ()

    def test_select_in_a_quoted_identifier_does_not_make_an_insert_a_rewrite(self, tmp_path):
        assert _keywords(tmp_path, 'INSERT INTO "SELECT Foo" ("id") VALUES (\'a\');') == ()

    def test_update_in_a_quoted_identifier_does_not_make_a_cte_a_rewrite(self, tmp_path):
        sql = 'WITH batch AS (SELECT "id" FROM "UPDATE Foo") SELECT count(*) FROM batch;'
        assert _keywords(tmp_path, sql) == ()

    def test_positional_parameter_is_not_a_dollar_quote(self, tmp_path):
        sql = 'ALTER TABLE "Foo" ADD COLUMN "b" TEXT;\nUPDATE "Foo" SET "b" = $1;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)


class TestEscapeHatch:
    def test_marker_with_reason_exempts_the_statement(self, tmp_path):
        sql = '-- data-migration-ok: one row per tenant, at most a few hundred\nUPDATE "Foo" SET "a" = 1;'
        assert _keywords(tmp_path, sql) == ()

    def test_marker_without_reason_does_not_exempt(self, tmp_path):
        assert _keywords(tmp_path, '-- data-migration-ok:\nUPDATE "Foo" SET "a" = 1;') == ("UPDATE",)

    def test_marker_exempts_only_its_own_statement(self, tmp_path):
        sql = '-- data-migration-ok: bounded to in-flight jobs\nUPDATE "Foo" SET "a" = 1;\nUPDATE "Bar" SET "b" = 2;\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 3

    def test_marker_works_inside_a_do_block(self, tmp_path):
        sql = 'DO $$\nBEGIN\n    -- data-migration-ok: single row\n    UPDATE "Foo" SET "a" = 1;\nEND $$;'
        assert _keywords(tmp_path, sql) == ()

    def test_marker_below_the_statement_does_not_exempt_the_next_one(self, tmp_path):
        sql = 'UPDATE "Foo" SET "a" = 1;\n-- data-migration-ok: bounded\nALTER TABLE "Bar" ADD COLUMN "b" TEXT;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)


class TestReporting:
    def test_line_number_points_at_the_statement_keyword(self, tmp_path):
        sql = '-- CreateIndex\nCREATE INDEX "i" ON "Foo"("a");\n\nUPDATE "Foo" SET "a" = 1;'
        assert _scan(tmp_path, sql)[0].line == 4

    def test_render_names_the_migration_and_line(self, tmp_path):
        violation = _scan(tmp_path, '\n\nDELETE FROM "Foo";')[0]
        rendered = violation.render()
        assert "20260101000000_fixture/migration.sql:3" in rendered
        assert "DELETE" in rendered


class TestGrandfathering:
    def test_every_grandfathered_migration_still_violates(self):
        for name in sorted(checker.GRANDFATHERED):
            directory = checker.MIGRATIONS_DIR / name
            assert directory.is_dir(), f"{name} no longer exists; drop it from GRANDFATHERED"
            assert checker.scan_migration(directory), f"{name} is clean; drop it from GRANDFATHERED"

    def test_stale_entry_is_reported_when_a_migration_stops_violating(self):
        found = {name: () for name in checker.GRANDFATHERED}
        assert checker.stale_grandfathers(found) == tuple(sorted(checker.GRANDFATHERED))

    def test_missing_entry_is_reported(self):
        assert checker.stale_grandfathers({}) == tuple(sorted(checker.GRANDFATHERED))

    def test_no_stale_entries_against_the_real_tree(self):
        found = {
            path.name: checker.scan_migration(path)
            for path in checker.MIGRATIONS_DIR.iterdir()
            if (path / "migration.sql").is_file()
        }
        assert checker.stale_grandfathers(found) == ()


class TestShippedMigrations:
    def test_the_repo_is_clean(self):
        assert checker.main() == 0
