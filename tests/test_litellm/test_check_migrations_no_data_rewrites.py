"""Tests for tests/code_coverage_tests/check_migrations_no_data_rewrites.py.

The checker reads migration.sql as SQL rather than as text, so the cases that matter
are the ones a grep would get wrong: the referential actions in a foreign key, of which
the shipped migrations carry 60, an `UPDATE` inside a string literal or a comment, and
an `UPDATE` hidden in the `DO $$ ... $$` block this repo uses for conditional DDL.
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

    def test_insert_values_with_a_scalar_subquery_passes(self, tmp_path):
        sql = 'INSERT INTO "Config" ("k", "v") VALUES (\'rev\', (SELECT max("id")::text FROM "Bar"));'
        assert _keywords(tmp_path, sql) == ()

    def test_insert_values_with_a_scalar_subquery_per_row_passes(self, tmp_path):
        sql = (
            'INSERT INTO "Config" ("k", "v") VALUES\n'
            "    ('a', (SELECT \"id\" FROM \"Bar\" WHERE \"n\" = 'a')),\n"
            "    ('b', (SELECT \"id\" FROM \"Bar\" WHERE \"n\" = 'b'));"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_values_inside_a_subquery_does_not_bound_an_insert_select(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") SELECT "id" FROM (VALUES (1), (2)) AS "v"("id");'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_values_after_a_set_operation_does_not_bound_an_insert_select(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") SELECT "id" FROM "Bar" UNION ALL VALUES (1);'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_values_after_an_except_does_not_bound_an_insert_select(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") SELECT "id" FROM "Bar" EXCEPT VALUES (1);'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_select_term_after_a_values_list_is_still_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") VALUES (1), (2) UNION ALL SELECT "id" FROM "Bar";'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_parenthesised_select_row_source_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (SELECT "id" FROM "Bar");'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_parenthesised_select_row_source_without_a_column_list_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" (SELECT "id" FROM "Bar");'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_parenthesised_select_row_source_spanning_lines_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id")\n(\n    SELECT "id" FROM "Bar"\n);'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_parenthesised_select_over_a_values_list_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (SELECT * FROM (VALUES (1), (2)) AS "v"("id"));'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_set_operation_over_parenthesised_selects_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (SELECT 1) UNION (SELECT 2);'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_values_list_joined_to_a_parenthesised_select_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") VALUES (1) UNION ALL (SELECT "id" FROM "Bar");'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_values_list_excepting_a_parenthesised_select_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") VALUES (1) EXCEPT (SELECT "id" FROM "Bar");'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_values_list_joined_to_a_parenthesised_table_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") VALUES (1) UNION ALL (TABLE "Bar");'
        assert _keywords(tmp_path, sql) == ("INSERT ... TABLE",)

    def test_a_set_operation_inside_a_values_list_does_not_flag_it(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") VALUES ((SELECT 1 UNION SELECT 2 LIMIT 1));'
        assert _keywords(tmp_path, sql) == ()

    def test_a_scalar_subquery_in_a_set_operated_values_list_stays_bounded(self, tmp_path):
        sql = (
            'INSERT INTO "Foo" ("id") VALUES ((SELECT max("id") FROM "Bar"))'
            " UNION ALL VALUES (2);"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_scalar_subquery_in_a_parenthesised_values_list_stays_bounded(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (VALUES ((SELECT max("id") FROM "Bar")));'
        assert _keywords(tmp_path, sql) == ()

    def test_a_scalar_subquery_in_set_operated_parenthesised_values_lists_stays_bounded(self, tmp_path):
        sql = (
            'INSERT INTO "Foo" ("id") (VALUES ((SELECT max("id") FROM "Bar")))'
            " UNION ALL (VALUES (2));"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_set_operation_inside_a_values_list_does_not_split_the_terms(self, tmp_path):
        sql = (
            'INSERT INTO "Foo" ("id") VALUES ((SELECT max("id") FROM "Bar"'
            ' UNION SELECT max("id") FROM "Bar")) UNION ALL VALUES (2);'
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_conflict_target_after_a_parenthesised_row_source_does_not_hide_it(self, tmp_path):
        sql = (
            'INSERT INTO "Foo" ("id") (SELECT "id" FROM "Bar")'
            ' ON CONFLICT ("id") DO NOTHING;'
        )
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_returning_list_after_a_parenthesised_row_source_does_not_hide_it(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (SELECT "id" FROM "Bar") RETURNING ("id");'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_conflict_target_beside_a_bounded_values_list_stays_bounded(self, tmp_path):
        sql = (
            'INSERT INTO "Foo" ("id") (VALUES ((SELECT max("id") FROM "Bar")))'
            ' ON CONFLICT ("id") DO NOTHING;'
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_query_term_written_before_a_values_term_is_still_the_row_source(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (SELECT "id" FROM "Bar") UNION ALL (VALUES (2));'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_table_term_beside_parenthesised_values_is_still_the_row_source(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (VALUES (1)) UNION ALL (TABLE "Bar");'
        assert _keywords(tmp_path, sql) == ("INSERT ... TABLE",)

    def test_a_table_row_source_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, 'INSERT INTO "Foo" TABLE "Bar";') == ("INSERT ... TABLE",)

    def test_a_table_named_in_the_insert_target_does_not_flag_it(self, tmp_path):
        assert _keywords(tmp_path, 'INSERT INTO "audit table" ("id") VALUES (1);') == ()

    def test_a_returning_subquery_after_a_wrapped_values_list_is_not_the_row_source(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (VALUES (1)) RETURNING (SELECT count(*) FROM "Bar");'
        assert _keywords(tmp_path, sql) == ()

    def test_a_conflict_update_after_a_wrapped_values_list_stays_bounded(self, tmp_path):
        sql = (
            'INSERT INTO "Foo" ("id") (VALUES (1))'
            ' ON CONFLICT ("id") DO UPDATE SET "id" = (SELECT max("id") FROM "Bar");'
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_wrapped_values_list_of_several_rows_stays_bounded(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (VALUES (1), (2)) RETURNING (SELECT count(*) FROM "Bar");'
        assert _keywords(tmp_path, sql) == ()

    def test_the_row_source_names_its_own_keyword_not_a_later_subquery(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (TABLE "Bar") RETURNING (SELECT count(*) FROM "Baz");'
        assert _keywords(tmp_path, sql) == ("INSERT ... TABLE",)

    def test_a_select_term_wrapped_beside_a_values_term_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (VALUES (1) UNION ALL SELECT "id" FROM "Bar");'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_a_table_term_wrapped_beside_a_values_term_is_flagged(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (VALUES (1) UNION ALL TABLE "Bar");'
        assert _keywords(tmp_path, sql) == ("INSERT ... TABLE",)

    def test_a_wrapped_set_operation_of_values_lists_stays_bounded(self, tmp_path):
        sql = 'INSERT INTO "Foo" ("id") (VALUES (1) UNION ALL VALUES (2));'
        assert _keywords(tmp_path, sql) == ()


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

    def test_cte_led_insert_from_a_parenthesised_select_is_flagged(self, tmp_path):
        sql = 'WITH batch AS (SELECT "id" FROM "Bar") INSERT INTO "Foo" ("id") (SELECT "id" FROM batch);'
        assert _keywords(tmp_path, sql) == ("WITH ... INSERT ... SELECT",)

    def test_cte_led_insert_into_a_values_list_passes(self, tmp_path):
        sql = 'WITH batch AS (SELECT max("id") FROM "Bar") INSERT INTO "Foo" ("id") VALUES (1);'
        assert _keywords(tmp_path, sql) == ()

    def test_read_only_cte_passes(self, tmp_path):
        sql = 'WITH batch AS (SELECT "id" FROM "Foo") SELECT count(*) FROM batch;'
        assert _keywords(tmp_path, sql) == ()

    def test_cte_led_insert_values_is_bounded_and_passes(self, tmp_path):
        sql = 'WITH latest AS (SELECT max("id") AS "id" FROM "Bar")\nINSERT INTO "Config" ("k", "v") VALUES (\'rev\', (SELECT "id"::text FROM latest));'
        assert _keywords(tmp_path, sql) == ()

    def test_a_writable_cte_bounded_by_values_passes(self, tmp_path):
        sql = 'WITH added AS (INSERT INTO "Foo" ("id") VALUES (1) RETURNING "id") SELECT * FROM added;'
        assert _keywords(tmp_path, sql) == ()

    def test_a_writable_cte_copying_a_query_is_flagged(self, tmp_path):
        sql = (
            'WITH added AS (INSERT INTO "Foo" ("id") SELECT "id" FROM "Bar" RETURNING "id")'
            " SELECT * FROM added;"
        )
        assert _keywords(tmp_path, sql) == ("WITH ... INSERT ... SELECT",)

    def test_a_bounded_writable_cte_does_not_hide_a_copying_one_beside_it(self, tmp_path):
        sql = (
            'WITH added AS (INSERT INTO "Foo" ("id") VALUES (1) RETURNING "id"),'
            ' copied AS (INSERT INTO "Baz" ("id") SELECT "id" FROM "Bar" RETURNING "id")'
            " SELECT * FROM added, copied;"
        )
        assert _keywords(tmp_path, sql) == ("WITH ... INSERT ... SELECT",)

    def test_a_writable_cte_wrapping_its_row_source_is_flagged(self, tmp_path):
        sql = (
            'WITH added AS (INSERT INTO "Foo" ("id") (SELECT "id" FROM "Bar") RETURNING "id")'
            " SELECT * FROM added;"
        )
        assert _keywords(tmp_path, sql) == ("WITH ... INSERT ... SELECT",)


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

    def test_line_number_inside_a_do_block_counts_from_the_top_of_the_file(self, tmp_path):
        sql = (
            "-- AlterTable\n"
            'ALTER TABLE "Foo" ADD COLUMN "b" INT;\n'
            "\n"
            "DO $$\n"
            "BEGIN\n"
            '    UPDATE "Foo" SET "b" = 1;\n'
            "END $$;"
        )
        assert _scan(tmp_path, sql)[0].line == 6

    def test_line_number_inside_a_nested_body_counts_from_the_top_of_the_file(self, tmp_path):
        sql = (
            "-- CreateIndex\n"
            'CREATE INDEX "i" ON "Foo"("a");\n'
            "\n"
            "DO $outer$\n"
            "BEGIN\n"
            "    EXECUTE $inner$\n"
            '        UPDATE "Foo" SET "a" = 1\n'
            "    $inner$;\n"
            "END $outer$;"
        )
        assert _scan(tmp_path, sql)[0].line == 7


class TestStoredRoutines:
    DEFINITION = (
        "CREATE FUNCTION backfill() RETURNS void AS $$\n"
        "BEGIN\n"
        '    UPDATE "Foo" SET "a" = 1;\n'
        "END;\n"
        "$$ LANGUAGE plpgsql;\n"
    )
    PROCEDURE = (
        "CREATE OR REPLACE PROCEDURE sweep() AS $$\n"
        "BEGIN\n"
        '    DELETE FROM "Foo";\n'
        "END;\n"
        "$$ LANGUAGE plpgsql;\n"
    )

    def test_a_function_body_nothing_calls_passes(self, tmp_path):
        assert _keywords(tmp_path, self.DEFINITION) == ()

    def test_a_procedure_body_nothing_calls_passes(self, tmp_path):
        assert _keywords(tmp_path, self.PROCEDURE) == ()

    def test_a_function_the_migration_calls_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, self.DEFINITION + "SELECT backfill();\n") == ("UPDATE",)

    def test_a_procedure_the_migration_calls_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, self.PROCEDURE + "CALL sweep();\n") == ("DELETE",)

    def test_a_call_written_above_the_definition_still_counts(self, tmp_path):
        assert _keywords(tmp_path, "SELECT backfill();\n" + self.DEFINITION) == ("UPDATE",)

    def test_a_call_from_inside_a_do_block_still_counts(self, tmp_path):
        sql = self.DEFINITION + "DO $$ BEGIN PERFORM backfill(); END; $$;\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_call_through_a_quoted_identifier_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'SELECT "backfill"();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_an_unrelated_quoted_identifier_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'SELECT "other"();\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_quoted_column_sharing_the_routine_name_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'ALTER TABLE "Foo" ADD COLUMN "backfill" int;\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_quoted_index_target_sharing_the_routine_name_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'CREATE INDEX ON "Foo" ("backfill");\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_quoted_call_with_space_before_its_parenthesis_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'SELECT "backfill" ();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_quoted_call_with_a_block_comment_before_its_parenthesis_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'SELECT "backfill" /* reason */ ();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_quoted_call_with_a_line_comment_before_its_parenthesis_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'SELECT "backfill" -- run it\n();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_quoted_column_followed_by_a_comment_is_still_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'ALTER TABLE "Foo" ADD COLUMN "backfill" /* note */ int;\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_like_named_table_with_a_column_list_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'CREATE TABLE "backfill" (id int);\n'
        assert _keywords(tmp_path, sql) == ()

    def test_an_insert_into_a_like_named_table_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'INSERT INTO "backfill" ("id") VALUES (1);\n'
        assert _keywords(tmp_path, sql) == ()

    def test_an_insert_into_a_like_named_table_with_a_commented_column_list_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'INSERT INTO "backfill" /* cols */ ("id") VALUES (1);\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_foreign_key_referencing_a_like_named_table_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'CREATE TABLE "Bar" (id int REFERENCES "backfill" ("id"));\n'
        assert _keywords(tmp_path, sql) == ()

    def test_an_index_on_a_like_named_table_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'CREATE INDEX ON "backfill" ("id");\n'
        assert _keywords(tmp_path, sql) == ()

    def test_an_if_not_exists_table_named_after_the_routine_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'CREATE TABLE IF NOT EXISTS "backfill" (id int);\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_copy_into_a_like_named_table_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'COPY "backfill" ("id") FROM stdin;\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_set_returning_call_in_from_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'SELECT * FROM "backfill"();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_call_in_a_join_condition_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'SELECT 1 FROM "Bar" b JOIN "Baz" z ON "backfill"();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_call_in_an_index_predicate_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'CREATE INDEX ON "Foo" ("a") WHERE "backfill"();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_join_condition_call_after_an_earlier_index_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'CREATE INDEX ON "Foo" ("a");\nSELECT 1 FROM "Bar" b JOIN "Baz" z ON "backfill"();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_an_insert_into_a_schema_qualified_like_named_table_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'INSERT INTO public."backfill" ("id") VALUES (1);\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_schema_qualified_index_on_a_like_named_table_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'CREATE INDEX ON public."backfill" ("id");\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_schema_qualified_quoted_call_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'SELECT public."backfill"();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_spaced_schema_qualifier_on_an_insert_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'INSERT INTO public . "backfill" ("id") VALUES (1);\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_spaced_schema_qualifier_on_an_index_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + 'CREATE INDEX ON public . "backfill" ("id");\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_quoted_schema_with_spaces_round_the_dot_stays_a_relation(self, tmp_path):
        sql = self.DEFINITION + 'INSERT INTO "public" . "backfill" ("id") VALUES (1);\n'
        assert _keywords(tmp_path, sql) == ()

    def test_a_spaced_schema_qualified_call_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'SELECT public . "backfill" ();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_schema_qualified_call_in_an_index_expression_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'CREATE INDEX ON "Bar" (public."backfill"("a"));\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_spaced_schema_qualified_call_in_an_index_expression_still_counts(self, tmp_path):
        sql = self.DEFINITION + 'CREATE INDEX ON "Bar" (public . "backfill" ("a"));\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_trigger_wiring_the_function_up_counts_as_a_call(self, tmp_path):
        sql = self.DEFINITION + 'CREATE TRIGGER t AFTER INSERT ON "Foo" EXECUTE FUNCTION backfill();\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_schema_qualified_definition_nothing_calls_passes(self, tmp_path):
        assert _keywords(tmp_path, self.DEFINITION.replace("backfill()", "public.backfill()")) == ()

    def test_a_schema_qualified_function_the_migration_calls_is_flagged(self, tmp_path):
        sql = self.DEFINITION.replace("backfill()", "public.backfill()") + "SELECT public.backfill();\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_the_name_written_only_in_a_comment_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + "-- backfill() is run by hand after the deploy\n"
        assert _keywords(tmp_path, sql) == ()

    def test_the_name_written_only_in_a_do_body_comment_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + "DO $$ BEGIN\n-- backfill() is run by hand after the deploy\nPERFORM 1;\nEND; $$;\n"
        assert _keywords(tmp_path, sql) == ()

    def test_the_name_written_only_in_a_do_body_block_comment_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + "DO $$ BEGIN /* backfill() runs later */ PERFORM 1; END; $$;\n"
        assert _keywords(tmp_path, sql) == ()

    def test_the_name_written_only_in_a_nested_body_comment_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + "DO $$ BEGIN EXECUTE $q$SELECT 1 -- backfill() runs later\n$q$; END; $$;\n"
        assert _keywords(tmp_path, sql) == ()

    def test_the_name_written_in_an_executed_literal_counts_as_a_call(self, tmp_path):
        sql = self.DEFINITION + "DO $$ BEGIN EXECUTE 'SELECT backfill()'; END; $$;\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_call_after_a_literal_holding_comment_dashes_still_counts(self, tmp_path):
        sql = self.DEFINITION + "DO $$ BEGIN RAISE NOTICE '--'; PERFORM backfill(); END; $$;\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_call_from_inside_a_single_quoted_do_block_still_counts(self, tmp_path):
        sql = self.DEFINITION + "DO 'BEGIN PERFORM backfill(); END';\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_an_executed_literal_inside_a_single_quoted_do_counts_as_a_call(self, tmp_path):
        sql = self.DEFINITION + "DO 'BEGIN EXECUTE ''SELECT backfill()''; END';\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_variable_run_by_execute_in_a_single_quoted_do_counts_as_a_call(self, tmp_path):
        sql = self.DEFINITION + "DO 'DECLARE q text; BEGIN q := ''SELECT backfill()''; EXECUTE q; END';\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_call_after_a_single_quoted_literal_holding_comment_dashes_still_counts(self, tmp_path):
        sql = self.DEFINITION + "DO 'BEGIN RAISE NOTICE ''--''; PERFORM backfill(); END';\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_call_after_a_single_quoted_literal_opening_a_block_comment_still_counts(self, tmp_path):
        sql = self.DEFINITION + "DO 'BEGIN RAISE NOTICE ''/*''; PERFORM backfill(); END';\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_an_executed_literal_after_a_single_quoted_comment_dash_string_still_counts(self, tmp_path):
        sql = self.DEFINITION + "DO 'BEGIN RAISE NOTICE ''--''; EXECUTE ''SELECT backfill()''; END';\n"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_long_run_of_escaped_quotes_before_an_uncalled_definition_is_not_a_call(self, tmp_path):
        escaped_quotes = "'" * 84
        sql = f"DO 'BEGIN RAISE NOTICE ''{escaped_quotes}''; PERFORM 1; END';\n" + self.DEFINITION
        assert _keywords(tmp_path, sql) == ()

    def test_the_name_written_only_in_a_single_quoted_do_comment_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + "DO 'BEGIN\n-- backfill() runs later\nPERFORM 1; END';\n"
        assert _keywords(tmp_path, sql) == ()

    def test_the_name_written_only_in_a_non_runnable_string_is_not_a_call(self, tmp_path):
        sql = self.DEFINITION + "SELECT 'backfill() runs after the deploy';\n"
        assert _keywords(tmp_path, sql) == ()

    def test_a_recursive_call_does_not_count_as_the_migration_calling_it(self, tmp_path):
        sql = (
            "CREATE FUNCTION backfill(n int) RETURNS void AS $$\n"
            "BEGIN\n"
            '    UPDATE "Foo" SET "a" = 1;\n'
            "    PERFORM backfill(n - 1);\n"
            "END;\n"
            "$$ LANGUAGE plpgsql;\n"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_quoted_routine_name_is_read_rather_than_trusted(self, tmp_path):
        sql = self.DEFINITION.replace("backfill()", '"back fill"()')
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_do_block_is_not_a_routine_definition(self, tmp_path):
        sql = 'DO $$ BEGIN UPDATE "Foo" SET "a" = 1; END; $$;\n'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_definition_written_after_another_statement_is_still_recognised(self, tmp_path):
        sql = 'ALTER TABLE "Foo" ADD COLUMN "a" INT;\n' + self.DEFINITION
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_exempts_a_rewrite_in_a_routine_the_migration_calls(self, tmp_path):
        sql = (
            "CREATE FUNCTION backfill() RETURNS void AS $$\n"
            "BEGIN\n"
            '    UPDATE "Foo" SET "a" = 1; -- data-migration-ok: single config row\n'
            "END;\n"
            "$$ LANGUAGE plpgsql;\n"
            "SELECT backfill();\n"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_called_routine_reports_the_line_inside_its_body(self, tmp_path):
        assert _scan(tmp_path, self.DEFINITION + "SELECT backfill();\n")[0].line == 3


class TestLoopBodies:
    def test_a_rewrite_in_a_query_driven_loop_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE r record;\n"
            "BEGIN\n"
            '    FOR r IN SELECT "id" FROM "Bar" LOOP\n'
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 5

    def test_a_delete_in_a_query_driven_loop_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE r record;\n"
            "BEGIN\n"
            '    FOR r IN SELECT "id" FROM "Bar" LOOP\n'
            '        DELETE FROM "Foo" WHERE "id" = r."id";\n'
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_join_using_in_the_loop_query_does_not_hide_the_body(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE r record;\n"
            "BEGIN\n"
            '    FOR r IN SELECT a."id" FROM "A" a JOIN "B" b USING ("id") LOOP\n'
            "        EXECUTE 'UPDATE \"Foo\" SET \"a\" = 1';\n"
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_executed_in_a_query_driven_loop_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE r record;\n"
            "BEGIN\n"
            '    FOR r IN SELECT "id" FROM "Bar" LOOP\n'
            "        EXECUTE 'UPDATE \"Foo\" SET \"a\" = 1';\n"
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_nested_under_a_guard_inside_a_loop_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE r record;\n"
            "BEGIN\n"
            '    FOR r IN SELECT "id" FROM "Bar" LOOP\n'
            '        IF r."id" > 0 THEN\n'
            '            UPDATE "Foo" SET "a" = 1;\n'
            "        END IF;\n"
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_inside_a_nested_loop_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE a record;\n"
            "DECLARE b record;\n"
            "BEGIN\n"
            '    FOR a IN SELECT "id" FROM "A" LOOP FOR b IN SELECT "id" FROM "B" LOOP\n'
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    END LOOP; END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_supplying_a_nested_loop_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE a record;\n"
            "DECLARE b record;\n"
            "BEGIN\n"
            '    FOR a IN SELECT "id" FROM "A" LOOP\n'
            '    FOR b IN UPDATE "Foo" SET "x" = 1 RETURNING "id" LOOP\n'
            "        NULL;\n"
            "    END LOOP; END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 6

    def test_a_loop_running_only_ddl_passes(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE r record;\n"
            "BEGIN\n"
            '    FOR r IN SELECT "id" FROM "Bar" LOOP\n'
            '        CREATE INDEX "i" ON "Foo"("a");\n'
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_loop_over_a_rewrite_returning_rows_is_flagged_once(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE r record;\n"
            "BEGIN\n"
            '    FOR r IN UPDATE "Foo" SET "a" = 1 RETURNING "id" LOOP\n'
            "        NULL;\n"
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_marker_on_a_loop_exempts_the_rewrite_it_repeats(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE r record;\n"
            "BEGIN\n"
            "    -- data-migration-ok: one row\n"
            '    FOR r IN SELECT "id" FROM "Bar" LOOP\n'
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_select_for_update_lock_is_not_read_as_a_loop(self, tmp_path):
        sql = 'DO $$\nBEGIN\n    PERFORM 1 FROM "Foo" FOR UPDATE;\nEND $$;'
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

    def test_a_marker_written_below_its_statement_leaves_that_statement_flagged(self, tmp_path):
        sql = 'UPDATE "Foo" SET "a" = 1;\n-- data-migration-ok: bounded\nUPDATE "Bar" SET "b" = 2;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 1

    def test_marker_inside_a_do_block_below_the_first_line_exempts(self, tmp_path):
        sql = (
            "-- AlterTable\n"
            'ALTER TABLE "Foo" ADD COLUMN "b" INT;\n'
            "\n"
            "DO $$\n"
            "BEGIN\n"
            "    -- data-migration-ok: one config row\n"
            '    UPDATE "Foo" SET "b" = 1;\n'
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_trailing_a_statement_exempts_that_statement(self, tmp_path):
        sql = 'UPDATE "Foo" SET "a" = 1; -- data-migration-ok: one row\nALTER TABLE "Bar" ADD COLUMN "b" TEXT;'
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_trailing_a_statement_does_not_exempt_the_next_one(self, tmp_path):
        sql = 'UPDATE "Foo" SET "a" = 1; -- data-migration-ok: one row\nUPDATE "Bar" SET "b" = 2;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 2

    def test_a_marker_trailing_a_multiline_statement_does_not_exempt_the_next_one(self, tmp_path):
        sql = (
            'UPDATE "Foo"\n'
            '   SET "a" = 1; -- data-migration-ok: one row\n'
            'UPDATE "Bar" SET "b" = 2;'
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 3

    def test_a_marker_alone_between_two_statements_belongs_to_the_one_below_it(self, tmp_path):
        sql = 'UPDATE "Foo" SET "a" = 1;\n-- data-migration-ok: one row\nUPDATE "Bar" SET "b" = 2;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 1

    def test_a_marker_a_blank_line_above_a_statement_does_not_exempt_it(self, tmp_path):
        sql = '-- data-migration-ok: one row\n\nUPDATE "Foo" SET "a" = 1;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 3

    def test_a_trailing_marker_exempts_only_the_statement_it_follows(self, tmp_path):
        sql = 'DELETE FROM "Foo" WHERE "a" = 1; UPDATE "Bar" SET "b" = 2; -- data-migration-ok: one row'
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_marker_above_a_shared_line_exempts_only_the_first_statement_on_it(self, tmp_path):
        sql = '-- data-migration-ok: one row\nUPDATE "Foo" SET "a" = 1; DELETE FROM "Bar" WHERE "b" = 2;'
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_marker_on_the_opening_line_of_a_statement_exempts_that_statement(self, tmp_path):
        sql = 'UPDATE "Foo" -- data-migration-ok: one row\n   SET "a" = 1;\nDELETE FROM "Bar";'
        assert _keywords(tmp_path, sql) == ("DELETE",)
        assert _scan(tmp_path, sql)[0].line == 3

    def test_marker_above_a_do_block_does_not_exempt_a_rewrite_inside_it(self, tmp_path):
        sql = (
            "-- data-migration-ok: bounded, this belongs to the insert below\n"
            "INSERT INTO \"Config\" (\"k\") VALUES ('x');\n"
            "\n"
            'DO $$ BEGIN UPDATE "Foo" SET "b" = 1; END $$;'
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 4

    def test_marker_directly_above_a_do_block_does_not_exempt_its_body(self, tmp_path):
        sql = (
            "-- data-migration-ok: seeding two default rows\n"
            "DO $$\n"
            "BEGIN\n"
            '    INSERT INTO "Foo" ("a") VALUES (1);\n'
            '    UPDATE "Foo" SET "a" = 1;\n'
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 5

    def test_marker_directly_above_a_one_line_do_block_does_not_exempt_its_body(self, tmp_path):
        sql = '-- data-migration-ok: seeding one default row\nDO $$ BEGIN UPDATE "Foo" SET "a" = 1; END $$;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 2

    def test_marker_on_the_do_line_does_not_exempt_its_body(self, tmp_path):
        sql = (
            "DO $$  -- data-migration-ok: bounded to one row\n"
            "BEGIN\n"
            '    IF EXISTS (SELECT 1 FROM "Foo") THEN\n'
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    END IF;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 4

    def test_a_marked_rewrite_does_not_exempt_a_later_one_in_the_same_block(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            "    -- data-migration-ok: bounded to one row\n"
            '    UPDATE "Foo" SET "a" = 1;\n'
            '    UPDATE "Bar" SET "b" = 2;\n'
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 5


class TestDynamicSql:
    def test_execute_of_a_quoted_update_is_flagged(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 3

    def test_execute_of_a_quoted_delete_is_flagged(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'DELETE FROM \"Foo\"';\nEND $$;"
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_execute_of_a_formatted_update_is_flagged(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE format('UPDATE %I SET \"a\" = 1', 'Foo');\nEND $$;"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_execute_of_a_dollar_quoted_update_is_flagged(self, tmp_path):
        sql = 'DO $outer$\nBEGIN\n    EXECUTE $q$UPDATE "Foo" SET "a" = 1$q$;\nEND $outer$;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_doubled_quote_inside_executed_sql_does_not_hide_the_rewrite(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'UPDATE \"Foo\" SET \"a\" = date_trunc(''day'', \"t\")';\nEND $$;"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_quoted_as_data_inside_executed_sql_is_not_run(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'SELECT ''UPDATE \"Foo\" SET \"a\" = 1''';\nEND $$;"
        assert _keywords(tmp_path, sql) == ()

    def test_a_doubled_quote_does_not_split_the_literal_it_sits_in(self, tmp_path):
        sql = "INSERT INTO \"Foo\" (\"note\") VALUES ('a''UPDATE \"Bar\" SET \"a\" = 1''b');"
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_following_a_doubled_quote_in_the_same_payload_is_flagged(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'SELECT ''x''; UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_comment_dash_inside_a_doubled_quote_does_not_hide_a_later_rewrite(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'SELECT ''--''; UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 3

    def test_a_block_comment_open_inside_a_doubled_quote_does_not_hide_a_later_rewrite(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'SELECT ''/*''; DELETE FROM \"Foo\"';\nEND $$;"
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_rewrite_genuinely_commented_out_inside_executed_sql_is_not_run(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'SELECT 1 -- UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_below_escaped_quotes_in_a_multiline_payload_reports_its_own_line(self, tmp_path):
        sql = (
            "DO $$\nBEGIN\n    EXECUTE '\n"
            "SELECT ''a'', ''b'', ''c'', ''d'', ''e''\n"
            "; UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 5

    def test_a_rewrite_in_a_later_command_before_bind_values_is_flagged(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'SELECT 1; DELETE FROM \"Foo\" WHERE \"a\" = $1' USING 1;\nEND $$;"
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_bind_value_naming_a_rewrite_is_not_run(self, tmp_path):
        sql = (
            "DO $$\nBEGIN\n    EXECUTE 'INSERT INTO \"Audit\" (\"note\") VALUES ($1)'"
            " USING 'DELETE FROM \"Foo\"';\nEND $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_executed_with_bind_values_is_still_flagged(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'DELETE FROM \"Foo\" WHERE \"a\" = $1' USING 1;\nEND $$;"
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_using_written_inside_the_command_does_not_end_it(self, tmp_path):
        sql = (
            "DO $$\nBEGIN\n    EXECUTE 'DELETE FROM \"Foo\" USING \"Bar\""
            " WHERE \"Foo\".\"a\" = \"Bar\".\"a\"';\nEND $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_join_using_in_a_subquery_building_the_command_does_not_end_it(self, tmp_path):
        sql = (
            "DO $$\nDECLARE v text;\nBEGIN\n    EXECUTE (SELECT v FROM \"A\" x JOIN \"A\" y"
            " USING (\"id\")) || 'UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_join_using_does_not_take_the_place_of_the_real_bind_values(self, tmp_path):
        sql = (
            "DO $$\nDECLARE v text;\nBEGIN\n    EXECUTE (SELECT v FROM \"A\" x JOIN \"A\" y"
            " USING (\"id\")) || 'UPDATE \"Foo\" SET \"a\" = $1' USING 2;\nEND $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_bind_value_naming_a_rewrite_after_a_subquery_join_is_not_run(self, tmp_path):
        sql = (
            "DO $$\nDECLARE v text;\nBEGIN\n    EXECUTE (SELECT v FROM \"A\" x JOIN \"A\" y"
            " USING (\"id\")) USING 'UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_execute_of_ddl_passes(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'ALTER TABLE \"Foo\" ADD COLUMN \"b\" TEXT';\nEND $$;"
        assert _keywords(tmp_path, sql) == ()

    def test_execute_of_a_read_only_query_passes(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'SELECT 1';\nEND $$;"
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_exempts_an_executed_rewrite(self, tmp_path):
        sql = "DO $$\nBEGIN\n    -- data-migration-ok: one row\n    EXECUTE 'UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        assert _keywords(tmp_path, sql) == ()

    def test_a_literal_that_is_not_executed_is_still_inert(self, tmp_path):
        sql = "INSERT INTO \"Foo\" (\"note\") VALUES ('UPDATE \"Bar\" SET \"a\" = 1');"
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_declared_into_a_variable_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text := 'UPDATE \"Foo\" SET \"a\" = 1';\n"
            "BEGIN\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 3

    def test_a_rewrite_assigned_in_the_body_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    stmt := 'DELETE FROM \"Foo\" WHERE \"a\" = 1';\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)
        assert _scan(tmp_path, sql)[0].line == 5

    def test_a_rewrite_selected_into_a_variable_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    SELECT 'UPDATE \"Foo\" SET \"a\" = 1' INTO stmt;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 5

    def test_a_rewrite_selected_into_a_strict_target_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    SELECT 'DELETE FROM \"Foo\"' INTO STRICT stmt;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_rewrite_assigned_with_a_bare_equals_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    stmt = 'UPDATE \"Foo\" SET \"a\" = 1';\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 5

    def test_a_rewrite_assigned_with_a_bare_equals_after_then_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    IF true THEN stmt = 'DELETE FROM \"Foo\"'; END IF;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)
        assert _scan(tmp_path, sql)[0].line == 5

    def test_a_rewrite_declared_with_a_bare_equals_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text = 'UPDATE \"Foo\" SET \"a\" = 1';\n"
            "BEGIN\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_an_insert_target_table_is_not_read_as_an_assignment(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    audit text := 'ALTER TABLE \"Foo\" ADD COLUMN \"b\" TEXT';\n"
            "BEGIN\n"
            "    INSERT INTO audit (note) VALUES ('DELETE FROM \"Foo\" is left to the app');\n"
            "    EXECUTE audit;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_returned_into_a_variable_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    INSERT INTO \"Log\" (\"sql\") VALUES ('UPDATE \"Foo\" SET \"a\" = 1')\n"
            "        RETURNING \"sql\" INTO stmt;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_assigned_past_an_earlier_comparison_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "    total int := 1;\n"
            "BEGIN\n"
            "    IF total = 1 THEN stmt = 'DELETE FROM \"Foo\" WHERE true'; END IF;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)
        assert _scan(tmp_path, sql)[0].line == 6

    def test_a_rewrite_assigned_past_a_loop_comparison_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "    total int := 3;\n"
            "BEGIN\n"
            "    WHILE total >= 1 LOOP stmt = 'UPDATE \"Foo\" SET \"a\" = 1'; END LOOP;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_selected_into_a_target_a_line_down_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    SELECT 'UPDATE \"Foo\" SET \"a\" = 1'\n"
            "        INTO\n"
            "        stmt;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_selected_into_a_strict_target_a_line_down_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    SELECT 'DELETE FROM \"Foo\"' INTO STRICT\n"
            "        stmt;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_rewrite_executed_by_a_name_a_line_down_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text := 'UPDATE \"Foo\" SET \"a\" = 1';\n"
            "BEGIN\n"
            "    EXECUTE\n"
            "        stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_rewrite_compared_against_is_not_an_assignment(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text := 'ALTER TABLE \"Foo\" ADD COLUMN \"b\" TEXT';\n"
            "BEGIN\n"
            "    IF stmt = 'UPDATE \"Foo\" SET \"a\" = 1' THEN\n"
            "        RAISE NOTICE 'the application owns that one';\n"
            "    END IF;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_passed_to_execute_as_a_parameter_is_not_run(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text := 'UPDATE \"Foo\" SET \"a\" = 1';\n"
            "BEGIN\n"
            "    EXECUTE 'INSERT INTO \"Log\" (\"sql\") VALUES ($1)' USING stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_compared_beside_an_assignment_is_not_run(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "    total int := 1;\n"
            "    ok boolean;\n"
            "BEGIN\n"
            "    stmt := 'CREATE INDEX IF NOT EXISTS \"ix_a\" ON \"Foo\" (\"a\")';\n"
            "    ok := total = 1 AND stmt = 'DELETE FROM \"Foo\"';\n"
            "    RAISE NOTICE 'purge script? %', ok;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_named_as_an_argument_beside_an_assignment_is_not_run(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "    ok boolean;\n"
            "BEGIN\n"
            "    stmt := 'CREATE INDEX IF NOT EXISTS \"ix_a\" ON \"Foo\" (\"a\")';\n"
            "    ok := probe_match(subject => stmt, wanted => 'DELETE FROM \"Foo\"');\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_compared_after_a_wider_comparison_is_not_run(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "    total int := 1;\n"
            "    ok boolean;\n"
            "BEGIN\n"
            "    stmt := 'CREATE INDEX IF NOT EXISTS \"ix_a\" ON \"Foo\" (\"a\")';\n"
            "    ok := total >= 1 AND stmt = 'DELETE FROM \"Foo\"';\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_assigned_through_a_case_expression_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "    total int := 1;\n"
            "BEGIN\n"
            "    stmt := CASE WHEN total = 1 THEN 'DELETE FROM \"Foo\"' ELSE 'SELECT 1' END;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_query_reaching_into_past_an_execute_is_not_a_rewrite(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "    total int;\n"
            "BEGIN\n"
            "    stmt := 'CREATE INDEX IF NOT EXISTS \"ix_a\" ON \"Foo\" (\"a\")';\n"
            "    EXECUTE 'SELECT count(*) FROM \"Foo\"'\n"
            "        INTO total;\n"
            "    SELECT (CASE WHEN total > 0 THEN 1 ELSE 2 END) INTO total\n"
            "        FROM \"Foo\"\n"
            "       WHERE \"a\" = 'DELETE FROM \"Foo\"';\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_query_reaching_using_past_an_execute_is_not_a_rewrite(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "    total int;\n"
            "BEGIN\n"
            "    stmt := 'CREATE INDEX IF NOT EXISTS \"ix_a\" ON \"Foo\" (\"a\")';\n"
            "    EXECUTE 'SELECT count(*) FROM \"Foo\" WHERE \"a\" = $1'\n"
            "        USING 'k1';\n"
            "    SELECT (CASE WHEN true THEN 1 ELSE 2 END) INTO total\n"
            "        FROM \"Foo\" x JOIN \"Foo\" y USING (\"a\")\n"
            "       WHERE x.\"a\" = 'DELETE FROM \"Foo\"';\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_rewrite_walked_by_a_loop_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    FOR stmt IN SELECT 'DELETE FROM \"Foo\"' LOOP\n"
            "        EXECUTE stmt;\n"
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)
        assert _scan(tmp_path, sql)[0].line == 5

    def test_a_rewrite_walked_by_a_foreach_loop_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text;\n"
            "BEGIN\n"
            "    FOREACH stmt IN ARRAY ARRAY['UPDATE \"Foo\" SET \"a\" = 1'] LOOP\n"
            "        EXECUTE stmt;\n"
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_a_loop_over_a_query_running_nothing_is_inert(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    rec record;\n"
            "BEGIN\n"
            "    FOR rec IN SELECT \"a\" FROM \"Foo\" LOOP\n"
            "        RAISE NOTICE 'the DELETE FROM \"Foo\" path is the application''s: %', rec;\n"
            "    END LOOP;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_literal_selected_into_a_variable_nothing_runs_is_inert(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    msg text;\n"
            "BEGIN\n"
            "    SELECT 'UPDATE of legacy rows is skipped' INTO msg;\n"
            "    RAISE NOTICE '%', msg;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_comparing_an_executed_variable_does_not_flag_the_comparison(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    stmt text := 'ALTER TABLE \"Foo\" ADD COLUMN \"b\" TEXT';\n"
            "BEGIN\n"
            "    IF stmt = 'DELETE FROM \"Foo\"' THEN RAISE NOTICE 'never'; END IF;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_ddl_assigned_to_a_variable_passes(self, tmp_path):
        sql = "DO $$\nDECLARE\n    stmt text := 'ALTER TABLE \"Foo\" ADD COLUMN \"b\" TEXT';\nBEGIN\n    EXECUTE stmt;\nEND $$;"
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_exempts_a_rewrite_held_in_a_variable(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    -- data-migration-ok: one config row, keyed by its primary key\n"
            "    stmt text := 'UPDATE \"Config\" SET \"v\" = 1 WHERE \"k\" = ''rev''';\n"
            "BEGIN\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_exempts_an_execute_whose_sql_starts_on_a_later_line(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            "    -- data-migration-ok: one config row, keyed by its primary key\n"
            "    EXECUTE '\n"
            "        UPDATE \"Config\" SET \"v\" = 1 WHERE \"k\" = ''rev''';\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_exempts_an_assignment_whose_sql_starts_on_a_later_line(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    -- data-migration-ok: one config row, keyed by its primary key\n"
            "    stmt text := '\n"
            "        UPDATE \"Config\" SET \"v\" = 1 WHERE \"k\" = ''rev''';\n"
            "BEGIN\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_an_unmarked_execute_whose_sql_starts_on_a_later_line_is_flagged(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            "    EXECUTE '\n"
            "        UPDATE \"Foo\" SET \"a\" = 1';\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 4

    def test_a_message_assigned_but_never_executed_passes(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    msg text := 'UPDATE of legacy rows skipped, the application backfills them';\n"
            "BEGIN\n"
            "    RAISE NOTICE '%', msg;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_notice_naming_a_delete_it_never_runs_passes(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    note text := 'DELETE FROM legacy rows is handled by the application';\n"
            "BEGIN\n"
            "    RAISE NOTICE '%', note;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_do_body_in_single_quotes_is_scanned(self, tmp_path):
        sql = "DO 'BEGIN UPDATE \"Foo\" SET \"a\" = 1; END';"
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 1

    def test_a_quoted_do_body_with_a_language_clause_is_scanned(self, tmp_path):
        sql = "DO LANGUAGE plpgsql 'BEGIN DELETE FROM \"Foo\"; END';"
        assert _keywords(tmp_path, sql) == ("DELETE",)

    def test_a_quoted_do_body_holding_only_ddl_passes(self, tmp_path):
        sql = "DO 'BEGIN ALTER TABLE \"Foo\" ADD COLUMN \"b\" TEXT; END';"
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_exempts_a_quoted_do_body(self, tmp_path):
        sql = (
            "-- data-migration-ok: one config row, keyed by its primary key\n"
            "DO 'BEGIN UPDATE \"Config\" SET \"v\" = 1 WHERE \"k\" = ''rev''; END';"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_concatenated_sql_is_flagged_when_the_keyword_leads_a_fragment(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'UPDATE ' || quote_ident('Foo') || ' SET \"a\" = 1';\nEND $$;"
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 3

    def test_concatenated_sql_is_flagged_when_the_keyword_leads_a_later_fragment(self, tmp_path):
        sql = "DO $$\nBEGIN\n    EXECUTE 'WITH x AS (SELECT 1) ' || 'UPDATE \"Foo\" SET \"a\" = 1';\nEND $$;"
        assert _keywords(tmp_path, sql) == ("UPDATE",)

    def test_only_the_variable_that_is_executed_is_read_as_sql(self, tmp_path):
        sql = (
            "DO $$\n"
            "DECLARE\n"
            "    msg text := 'UPDATE of legacy rows skipped';\n"
            "    stmt text := 'DELETE FROM \"Foo\" WHERE \"a\" = 1';\n"
            "BEGIN\n"
            "    RAISE NOTICE '%', msg;\n"
            "    EXECUTE stmt;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("DELETE",)
        assert _scan(tmp_path, sql)[0].line == 4

    def test_a_marker_on_an_execute_covers_its_single_quoted_payload(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            "    EXECUTE '  -- data-migration-ok: bounded to one row\n"
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    ';\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()

    def test_a_marker_on_an_execute_does_not_reach_into_a_dollar_quoted_payload(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            "    EXECUTE $x$  -- data-migration-ok: bounded to one row\n"
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    $x$;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 4

    def test_a_marker_inside_a_dollar_quoted_payload_exempts_its_rewrite(self, tmp_path):
        sql = (
            "DO $$\n"
            "BEGIN\n"
            "    EXECUTE $x$\n"
            "        -- data-migration-ok: bounded to one row\n"
            '        UPDATE "Foo" SET "a" = 1;\n'
            "    $x$;\n"
            "END $$;"
        )
        assert _keywords(tmp_path, sql) == ()


class TestExplain:
    def test_explain_analyze_over_an_update_is_flagged(self, tmp_path):
        sql = 'EXPLAIN ANALYZE UPDATE "Foo" SET "a" = 1;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 1

    def test_explain_analyze_verbose_over_a_delete_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, 'EXPLAIN ANALYZE VERBOSE DELETE FROM "Foo";') == ("DELETE",)

    def test_explain_with_a_parenthesised_analyze_is_flagged(self, tmp_path):
        assert _keywords(tmp_path, 'EXPLAIN (ANALYZE, BUFFERS) UPDATE "Foo" SET "a" = 1;') == ("UPDATE",)

    def test_explain_analyze_over_an_insert_select_is_flagged(self, tmp_path):
        sql = 'EXPLAIN ANALYZE INSERT INTO "Foo" SELECT "a" FROM "Bar";'
        assert _keywords(tmp_path, sql) == ("INSERT ... SELECT",)

    def test_explain_analyze_over_a_select_passes(self, tmp_path):
        assert _keywords(tmp_path, 'EXPLAIN ANALYZE SELECT * FROM "Foo";') == ()

    def test_a_marker_exempts_an_explained_rewrite(self, tmp_path):
        sql = '-- data-migration-ok: one config row\nEXPLAIN ANALYZE UPDATE "Foo" SET "a" = 1;'
        assert _keywords(tmp_path, sql) == ()

    def test_an_analyze_of_its_own_passes(self, tmp_path):
        assert _keywords(tmp_path, 'ANALYZE "Foo";') == ()

    def test_a_vacuum_analyze_passes(self, tmp_path):
        assert _keywords(tmp_path, 'VACUUM ANALYZE "Foo";') == ()

    def test_an_explained_rewrite_inside_a_block_reports_its_line(self, tmp_path):
        sql = 'DO $$\nBEGIN\n    EXPLAIN ANALYZE UPDATE "Foo" SET "a" = 1;\nEND $$;'
        assert _keywords(tmp_path, sql) == ("UPDATE",)
        assert _scan(tmp_path, sql)[0].line == 3


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


CLEAN = 'ALTER TABLE "Foo" ADD COLUMN "a" INT;'
DIRTY = 'UPDATE "Foo" SET "a" = 1;'
FIXTURE = "20260101000000_fixture"


def _tree(monkeypatch, tmp_path: Path, sql: str, grandfathered: frozenset = frozenset()) -> None:
    """Stand a migrations directory holding one fixture migration in for the repo's own. The
    root moves with it, since a rendered violation names the migration relative to the root and
    the two are read off the same checkout everywhere but here."""
    directory = tmp_path / "migrations" / FIXTURE
    directory.mkdir(parents=True)
    (directory / "migration.sql").write_text(sql, encoding="utf-8")
    monkeypatch.setattr(checker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(checker, "MIGRATIONS_DIR", tmp_path / "migrations")
    monkeypatch.setattr(checker, "GRANDFATHERED", grandfathered)


class TestExitCode:
    def test_a_clean_tree_passes(self, tmp_path, monkeypatch):
        _tree(monkeypatch, tmp_path, CLEAN)
        assert checker.main() == 0

    def test_a_violation_fails_the_check(self, tmp_path, monkeypatch):
        _tree(monkeypatch, tmp_path, DIRTY)
        assert checker.main() == 1

    def test_a_stale_grandfather_alone_fails_the_check(self, tmp_path, monkeypatch):
        _tree(monkeypatch, tmp_path, CLEAN, frozenset({FIXTURE}))
        assert checker.main() == 1

    def test_a_grandfathered_violation_passes(self, tmp_path, monkeypatch):
        _tree(monkeypatch, tmp_path, DIRTY, frozenset({FIXTURE}))
        assert checker.main() == 0

    def test_a_missing_migrations_directory_is_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(checker, "MIGRATIONS_DIR", tmp_path / "absent")
        assert checker.main() == 2

    def test_the_failure_names_the_migration_the_line_and_the_keyword(
        self, tmp_path, monkeypatch, capsys
    ):
        _tree(monkeypatch, tmp_path, DIRTY)
        checker.main()
        printed = capsys.readouterr().out
        assert f"migrations/{FIXTURE}/migration.sql:1" in printed
        assert "UPDATE rewrites existing rows at boot" in printed
        assert checker.GUIDANCE in printed

    def test_a_stale_grandfather_is_named(self, tmp_path, monkeypatch, capsys):
        _tree(monkeypatch, tmp_path, CLEAN, frozenset({FIXTURE}))
        checker.main()
        assert f"{FIXTURE}: listed in GRANDFATHERED" in capsys.readouterr().out

    def test_a_missing_directory_is_reported_on_stderr(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(checker, "MIGRATIONS_DIR", tmp_path / "absent")
        checker.main()
        assert "migrations directory not found" in capsys.readouterr().err
