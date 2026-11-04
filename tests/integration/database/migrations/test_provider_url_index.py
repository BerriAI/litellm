import os
import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from litellm_proxy_extras.utils import MCP_SERVER_TABLE, PROVIDER_URL_INDEX, ProxyExtrasDBManager

psycopg = pytest.importorskip("psycopg")

pytestmark = pytest.mark.timeout(120)

requires_db: Final = pytest.mark.skipif(
    "DATABASE_URL" not in os.environ,
    reason="requires a postgres database (DATABASE_URL)",
)


def _base_url() -> str:
    return os.environ["DATABASE_URL"].split("?")[0]


def _scratch_schema(monkeypatch: pytest.MonkeyPatch, columns: str) -> Iterator[str]:
    schema: Final = f"provider_url_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'CREATE SCHEMA "{schema}"')
        conn.execute(f'CREATE TABLE "{schema}"."{MCP_SERVER_TABLE}" ({columns})')
    monkeypatch.delenv("DIRECT_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"{_base_url()}?schema={schema}")
    yield schema
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def pushed_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """What `prisma db push` or schema adoption leaves: the column, but not the SQL-only index."""
    yield from _scratch_schema(monkeypatch, 'server_id TEXT PRIMARY KEY, url TEXT, execution_mode TEXT NOT NULL')


@pytest.fixture
def older_schema(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    yield from _scratch_schema(monkeypatch, "server_id TEXT PRIMARY KEY, url TEXT")


def _insert(schema: str, server_id: str, url: str, mode: str) -> None:
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        conn.execute(
            f'INSERT INTO "{schema}"."{MCP_SERVER_TABLE}" (server_id, url, execution_mode) VALUES (%s, %s, %s)',
            (server_id, url, mode),
        )


def _index_exists(schema: str) -> bool:
    with psycopg.connect(_base_url(), autocommit=True) as conn:
        row = conn.execute("SELECT to_regclass(%s)", (f'"{schema}"."{PROVIDER_URL_INDEX}"',)).fetchone()
    return row is not None and row[0] is not None


@requires_db
def test_setup_creates_the_provider_url_index_a_pushed_database_lacks(pushed_schema: str) -> None:
    assert _index_exists(pushed_schema) is False

    assert ProxyExtrasDBManager.ensure_provider_url_index() is True
    assert ProxyExtrasDBManager.ensure_provider_url_index() is True

    assert _index_exists(pushed_schema) is True
    _insert(pushed_schema, "gateway-a", "https://same.example/mcp", "litellm")
    _insert(pushed_schema, "gateway-b", "https://same.example/mcp", "litellm")
    _insert(pushed_schema, "provider-a", "https://same.example/mcp", "provider")
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert(pushed_schema, "provider-b", "https://same.example/mcp", "provider")


@requires_db
def test_setup_skips_the_provider_url_index_before_the_execution_mode_column_exists(older_schema: str) -> None:
    assert ProxyExtrasDBManager.ensure_provider_url_index() is False
    assert _index_exists(older_schema) is False
