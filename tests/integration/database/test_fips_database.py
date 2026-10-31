import asyncio
from contextlib import nullcontext
from typing import Final, LiteralString

import pytest

from litellm.proxy.common_utils.fips import FipsModeError, enforce_fips_database
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.fips_repository import FipsRepository
from tests.integration._support.database import read_rows, scratch_database, write_rows

_CONFIG_TABLE: Final = (
    'CREATE TABLE "LiteLLM_Config" (param_name TEXT PRIMARY KEY, param_value JSONB, last_run_at TIMESTAMP(3), '
    "reload_revision BIGINT NOT NULL DEFAULT 0)"
)
_KEY_TABLE: Final = 'CREATE TABLE "LiteLLM_VerificationToken" (token TEXT PRIMARY KEY)'
_MARKER: Final = "INSERT INTO \"LiteLLM_Config\" (param_name, param_value) VALUES ('fips_mode', 'true')"
_KEY: Final = "INSERT INTO \"LiteLLM_VerificationToken\" (token) VALUES ('hashed-key')"
_SETTING: Final = "INSERT INTO \"LiteLLM_Config\" (param_name, param_value) VALUES ('general_settings', '{}')"


def _marker_rows(database_url: str) -> int:
    return len(
        read_rows('SELECT 1 FROM "LiteLLM_Config" WHERE param_name = %s', ("fips_mode",), database_url=database_url)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fips_mode, setup, refusal, marked_after",
    [
        (True, (_CONFIG_TABLE, _KEY_TABLE), None, 1),
        (True, (_CONFIG_TABLE, _KEY_TABLE, _MARKER, _KEY), None, 1),
        (True, (_CONFIG_TABLE, _KEY_TABLE, _KEY), "already holds data written without it", 0),
        (True, (_CONFIG_TABLE, _KEY_TABLE, _SETTING), "already holds data written without it", 0),
        (True, (), "could not be checked", None),
        (False, (_CONFIG_TABLE, _KEY_TABLE, _MARKER), "set up with general_settings.fips_mode: true", 1),
        (False, (_CONFIG_TABLE, _KEY_TABLE, _KEY), None, 0),
        (False, (), None, None),
    ],
    ids=[
        "fips-on-new-database",
        "fips-on-fips-database",
        "fips-on-database-with-a-key",
        "fips-on-database-with-a-setting",
        "fips-on-database-without-tables",
        "fips-off-fips-database",
        "fips-off-database-with-a-key",
        "fips-off-database-without-tables",
    ],
)
async def test_fips_database_check(
    monkeypatch: pytest.MonkeyPatch,
    fips_mode: bool,
    setup: tuple[LiteralString, ...],
    refusal: str | None,
    marked_after: int | None,
) -> None:
    with scratch_database() as database_url:
        for statement in setup:
            write_rows(statement, (), database_url=database_url)
        monkeypatch.setenv("DATABASE_URL", database_url)
        client: Final = PrismaClient(database_url, ProxyLogging(UserApiKeyCache()))
        await client.connect()
        try:
            with pytest.raises(FipsModeError, match=refusal) if refusal else nullcontext():
                await enforce_fips_database(FipsRepository(client), fips_mode=fips_mode, database_is_configured=True)
        finally:
            await client.aclose()
        if marked_after is not None:
            assert _marker_rows(database_url) == marked_after


@pytest.mark.asyncio
async def test_pods_starting_together_in_fips_mode_on_a_new_database_both_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with scratch_database() as database_url:
        for statement in (_CONFIG_TABLE, _KEY_TABLE):
            write_rows(statement, (), database_url=database_url)
        monkeypatch.setenv("DATABASE_URL", database_url)
        pods: Final = tuple(PrismaClient(database_url, ProxyLogging(UserApiKeyCache())) for _ in range(4))
        await asyncio.gather(*(pod.connect() for pod in pods))
        try:
            await asyncio.gather(
                *(
                    enforce_fips_database(FipsRepository(pod), fips_mode=True, database_is_configured=True)
                    for pod in pods
                )
            )
        finally:
            await asyncio.gather(*(pod.aclose() for pod in pods))
        assert _marker_rows(database_url) == 1
