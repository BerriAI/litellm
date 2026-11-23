import json
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql
from pydantic import TypeAdapter

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    ALGO_AES_GCM,
    ENCRYPTION_ALGORITHM_SETTING,
    V2_GCM_PREFIX,
    decrypt_if_encrypted_with,
    encrypt_value_helper,
)
from litellm.proxy.management_endpoints.credential_migration import check_encryption, migrate_encryption
from litellm.proxy.utils import PrismaClient
from tests.integration._support.prisma_clients import connected_proxy_prisma

SALT_KEY: Final = "sk-integration-profile-migration-salt"
SERVER: Final = "tickets"
PROFILES: Final = TypeAdapter(list[dict[str, str]])
ADMIN: Final = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)


@dataclass(frozen=True, slots=True)
class _Database:
    client: PrismaClient
    database_url: str
    schema: str

    def stored_profiles(self) -> list[dict[str, str]]:
        with psycopg.connect(self.database_url) as connection:
            row: Final = connection.execute(
                sql.SQL("SELECT additional_auth FROM {} WHERE server_id = %s").format(
                    sql.Identifier(self.schema, "LiteLLM_MCPServerTable")
                ),
                (SERVER,),
            ).fetchone()
        assert row is not None
        return PROFILES.validate_python(row[0])


@contextmanager
def _private_copy_of_the_tables(database_url: str, schema: str) -> Iterator[None]:
    with psycopg.connect(database_url, autocommit=True) as setup:
        setup.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            for (table,) in setup.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() AND tablename LIKE 'LiteLLM\\_%'"
            ).fetchall():
                setup.execute(
                    sql.SQL("CREATE TABLE {} (LIKE {} INCLUDING DEFAULTS INCLUDING CONSTRAINTS)").format(
                        sql.Identifier(schema, table), sql.Identifier(table)
                    )
                )
            yield
        finally:
            setup.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _scoped_url(database_url: str, schema: str) -> str:
    parsed: Final = urlsplit(database_url)
    return urlunsplit(parsed._replace(query=urlencode({**dict(parse_qsl(parsed.query)), "schema": schema})))


@asynccontextmanager
async def _server_with_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[_Database, tuple[dict[str, str], dict[str, str]]]]:
    database_url: Final = os.environ["DATABASE_URL"]
    monkeypatch.setenv("LITELLM_SALT_KEY", SALT_KEY)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    legacy_profile: Final = {
        "id": "Build",
        "auth_type": "bearer_token",
        "authentication_token": encrypt_value_helper("build-secret", new_encryption_key=SALT_KEY),
    }
    monkeypatch.setattr(proxy_server, "general_settings", {ENCRYPTION_ALGORITHM_SETTING: ALGO_AES_GCM})
    current_profile: Final = {
        "id": "Read",
        "auth_type": "bearer_token",
        "authentication_token": encrypt_value_helper("read-secret", new_encryption_key=SALT_KEY),
    }
    schema: Final = f"integration_{uuid.uuid4().hex}"
    with _private_copy_of_the_tables(database_url, schema):
        with psycopg.connect(database_url, autocommit=True) as connection:
            connection.execute(
                sql.SQL("INSERT INTO {} (server_id, additional_auth) VALUES (%s, %s::jsonb)").format(
                    sql.Identifier(schema, "LiteLLM_MCPServerTable")
                ),
                (SERVER, json.dumps([legacy_profile, current_profile])),
            )
        monkeypatch.setenv("DATABASE_URL", _scoped_url(database_url, schema))
        async with connected_proxy_prisma() as client:
            yield _Database(client, database_url, schema), (legacy_profile, current_profile)


@pytest.mark.asyncio
async def test_the_inventory_reads_profile_secrets_from_the_database_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _server_with_profiles(monkeypatch) as (database, seeded):
        check: Final = await check_encryption(database.client)
        dry_run: Final = await migrate_encryption(database.client, ADMIN, dry_run=True)
        stored: Final = database.stored_profiles()

    for report in (check, dry_run):
        mcp_server = next(location for location in report.locations if location.location == "mcp_server")
        assert (mcp_server.scanned, mcp_server.legacy, mcp_server.already_v2) == (2, 1, 1)
        assert report.residual_legacy == 1
    assert stored == list(seeded)


@pytest.mark.asyncio
async def test_the_aes_migration_rewrites_profile_secrets_and_leaves_the_rest_of_the_profile_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with _server_with_profiles(monkeypatch) as (database, (legacy_profile, current_profile)):
        report: Final = await migrate_encryption(database.client, ADMIN)
        migrated, untouched = database.stored_profiles()
        after_the_check: Final = await check_encryption(database.client)

    assert migrated["authentication_token"].startswith(V2_GCM_PREFIX)
    assert {
        **migrated,
        "authentication_token": decrypt_if_encrypted_with(migrated["authentication_token"], SALT_KEY),
    } == {"id": "Build", "auth_type": "bearer_token", "authentication_token": "build-secret"}
    assert untouched == current_profile
    mcp_server: Final = next(location for location in report.locations if location.location == "mcp_server")
    assert (mcp_server.migrated, mcp_server.legacy, mcp_server.already_v2) == (1, 0, 2)
    assert (report.residual_legacy, after_the_check.residual_legacy) == (0, 0)
    assert legacy_profile["authentication_token"] != migrated["authentication_token"]


@pytest.mark.asyncio
async def test_running_the_aes_migration_again_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    async with _server_with_profiles(monkeypatch) as (database, _):
        await migrate_encryption(database.client, ADMIN)
        after_the_first_run: Final = database.stored_profiles()
        await migrate_encryption(database.client, ADMIN)
        after_the_second_run: Final = database.stored_profiles()

    assert after_the_second_run == after_the_first_run
