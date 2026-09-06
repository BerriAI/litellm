from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
EXTRAS_ROOT = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"
SCHEMAS = (
    REPO_ROOT / "schema.prisma",
    REPO_ROOT / "litellm" / "proxy" / "schema.prisma",
    EXTRAS_ROOT / "schema.prisma",
)

INDEXES = {
    "api_key": (
        EXTRAS_ROOT / "migrations" / "20260804233000_add_spend_log_api_key_window_covering_index" / "migration.sql"
    ),
    "team_id": (
        EXTRAS_ROOT / "migrations" / "20260804233100_add_spend_log_team_window_covering_index" / "migration.sql"
    ),
}


def test_spend_log_budget_window_indexes_are_declared_in_all_schemas():
    schemas = [schema.read_text(encoding="utf-8") for schema in SCHEMAS]

    assert all(schema == schemas[0] for schema in schemas[1:])
    for schema in schemas:
        for field in INDEXES:
            assert f"@@index([{field}, startTime])" in schema


def test_spend_log_budget_window_migrations_create_covering_indexes_concurrently():
    for field, migration_path in INDEXES.items():
        migration = migration_path.read_text(encoding="utf-8")

        assert migration.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 1
        assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in migration
        assert f'("{field}", "startTime") INCLUDE ("spend")' in migration
