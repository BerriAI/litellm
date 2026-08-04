from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
EXTRAS_ROOT = REPO_ROOT / "litellm-proxy-extras" / "litellm_proxy_extras"

INDEXES = {
    "api_key": (
        EXTRAS_ROOT / "migrations" / "20260804233000_add_spend_log_api_key_window_covering_index" / "migration.sql"
    ),
    "team_id": (
        EXTRAS_ROOT / "migrations" / "20260804233100_add_spend_log_team_window_covering_index" / "migration.sql"
    ),
}


def test_spend_log_budget_window_indexes_are_declared_in_both_schemas():
    root_schema = (REPO_ROOT / "schema.prisma").read_text(encoding="utf-8")
    packaged_schema = (EXTRAS_ROOT / "schema.prisma").read_text(encoding="utf-8")

    for field in INDEXES:
        declaration = f"@@index([{field}, startTime])"
        assert declaration in root_schema
        assert declaration in packaged_schema


def test_spend_log_budget_window_migrations_create_covering_indexes_concurrently():
    for field, migration_path in INDEXES.items():
        migration = migration_path.read_text(encoding="utf-8")

        assert migration.count("CREATE INDEX CONCURRENTLY IF NOT EXISTS") == 1
        assert "CREATE INDEX CONCURRENTLY IF NOT EXISTS" in migration
        assert f'("{field}", "startTime") INCLUDE ("spend")' in migration
