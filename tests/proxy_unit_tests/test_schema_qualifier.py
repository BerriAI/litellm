"""Unit tests for ``litellm.proxy.db.schema_qualifier``.

Covers the env-var / URL resolution and the SQL rewrite. The rewrite is the
non-obvious part: it must be idempotent, must be a no-op when no schema is
configured, and must not touch identifiers that look like ours but aren't
ours (e.g. ``"LITELLM_TABLE"`` uppercase, ``LiteLLM_Whatever`` without
quotes, schema-already-qualified references).
"""

import os
from unittest import mock

import pytest

from litellm.proxy.db import schema_qualifier
from litellm.proxy.db.schema_qualifier import get_schema, qualify


@pytest.fixture(autouse=True)
def _reset_cache():
    """``get_schema`` caches its result. Reset between tests so each one
    sees the fresh env state."""
    schema_qualifier.reset_cache()
    yield
    schema_qualifier.reset_cache()


class TestGetSchema:
    def test_returns_none_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert get_schema() is None

    def test_reads_database_schema_env(self):
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "litellm"}, clear=True):
            assert get_schema() == "litellm"

    def test_strips_whitespace(self):
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "  litellm  "}, clear=True):
            assert get_schema() == "litellm"

    def test_falls_back_to_database_url_query_param(self):
        url = "postgresql://u:p@h/db?schema=litellm&sslmode=require"
        with mock.patch.dict(os.environ, {"DATABASE_URL": url}, clear=True):
            assert get_schema() == "litellm"

    def test_database_schema_takes_precedence_over_url(self):
        with mock.patch.dict(
            os.environ,
            {
                "DATABASE_SCHEMA": "from_env",
                "DATABASE_URL": "postgresql://u:p@h/db?schema=from_url",
            },
            clear=True,
        ):
            assert get_schema() == "from_env"

    def test_returns_none_for_public_schema(self):
        # `public` is the Postgres default — no qualifier required, and
        # forcing one would just be noise. Treat it as unset.
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "public"}, clear=True):
            assert get_schema() is None


class TestQualify:
    def test_noop_when_unset(self):
        sql = 'SELECT * FROM "LiteLLM_VerificationToken"'
        with mock.patch.dict(os.environ, {}, clear=True):
            assert qualify(sql) == sql

    def test_prefixes_a_single_litellm_table(self):
        sql = 'SELECT * FROM "LiteLLM_VerificationToken" WHERE token = $1'
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "litellm"}, clear=True):
            out = qualify(sql)
        assert out == (
            'SELECT * FROM "litellm"."LiteLLM_VerificationToken" '
            "WHERE token = $1"
        )

    def test_prefixes_every_table_in_a_multi_table_join(self):
        sql = (
            'SELECT v.*, t.spend FROM "LiteLLM_VerificationToken" AS v '
            'LEFT JOIN "LiteLLM_TeamTable" AS t ON v.team_id = t.team_id '
            'LEFT JOIN "LiteLLM_BudgetTable" AS b ON v.budget_id = b.budget_id'
        )
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "litellm"}, clear=True):
            out = qualify(sql)
        assert '"litellm"."LiteLLM_VerificationToken"' in out
        assert '"litellm"."LiteLLM_TeamTable"' in out
        assert '"litellm"."LiteLLM_BudgetTable"' in out
        # The aliases and join conditions must be untouched
        assert "AS v " in out
        assert "v.team_id = t.team_id" in out

    def test_handles_create_view_statements(self):
        sql = (
            'CREATE VIEW "LiteLLM_VerificationTokenView" AS '
            'SELECT v.*, t.spend FROM "LiteLLM_VerificationToken" v '
            'LEFT JOIN "LiteLLM_TeamTable" t ON v.team_id = t.team_id'
        )
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "litellm"}, clear=True):
            out = qualify(sql)
        assert '"litellm"."LiteLLM_VerificationTokenView"' in out
        assert '"litellm"."LiteLLM_VerificationToken"' in out
        assert '"litellm"."LiteLLM_TeamTable"' in out

    def test_handles_regclass_cast_in_string_literal(self):
        # `'"LiteLLM_SpendLogs"'::regclass` — the inner double-quoted
        # identifier needs qualifying even though it's embedded in a
        # single-quoted string. Postgres accepts the qualified form.
        sql = (
            "SELECT reltuples FROM pg_class "
            "WHERE oid = '\"LiteLLM_SpendLogs\"'::regclass;"
        )
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "litellm"}, clear=True):
            out = qualify(sql)
        assert "'\"litellm\".\"LiteLLM_SpendLogs\"'::regclass" in out

    def test_does_not_touch_unquoted_table_references(self):
        # Token names, column names, and other unquoted occurrences of
        # `LiteLLM_*` should NOT be rewritten — only the double-quoted
        # identifier form is rewritten.
        sql = "-- comment about LiteLLM_VerificationToken table behavior"
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "litellm"}, clear=True):
            assert qualify(sql) == sql

    def test_idempotent(self):
        # Applying the qualifier twice must produce the same result as
        # applying it once. Important because callers should be able to
        # apply ``qualify()`` defensively without checking whether the
        # SQL has already been rewritten.
        sql = 'SELECT * FROM "LiteLLM_VerificationToken"'
        with mock.patch.dict(os.environ, {"DATABASE_SCHEMA": "litellm"}, clear=True):
            once = qualify(sql)
            twice = qualify(once)
        assert once == twice

    def test_url_with_schema_param(self):
        sql = 'SELECT * FROM "LiteLLM_TeamTable"'
        url = "postgresql://u:p@h/db?schema=acme&sslmode=require"
        with mock.patch.dict(os.environ, {"DATABASE_URL": url}, clear=True):
            assert qualify(sql) == 'SELECT * FROM "acme"."LiteLLM_TeamTable"'
