from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.db.pgbouncer import PGBOUNCER_POOLED_ENV_VAR
from litellm.repositories.database import Database

WRITER_URL: Final = "postgresql://litellm:secret@db.invalid:5432/proxy?connection_limit=3"


def _prisma(aclose: AsyncMock | None = None) -> MagicMock:
    return MagicMock(aclose=aclose or AsyncMock())


@pytest.fixture
def static_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAM_TOKEN_DB_AUTH", raising=False)
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    monkeypatch.delenv(PGBOUNCER_POOLED_ENV_VAR, raising=False)


@pytest.mark.asyncio
@pytest.mark.usefixtures("static_password")
async def test_closing_the_database_closes_prisma() -> None:
    aclose: Final = AsyncMock()
    database: Final = await Database.open(_prisma(aclose), WRITER_URL)

    aclose.assert_not_awaited()
    await database.aclose()

    aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("static_password")
async def test_static_password_needs_no_token_refresh() -> None:
    database: Final = await Database.open(_prisma(), WRITER_URL)

    assert database.token_refresh is None
    await database.aclose()


@pytest.mark.asyncio
async def test_closing_the_database_stops_the_token_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.delenv("AZURE_POSTGRESQL_AUTH", raising=False)
    monkeypatch.delenv(PGBOUNCER_POOLED_ENV_VAR, raising=False)
    database: Final = await Database.open(_prisma(), WRITER_URL)
    token_refresh: Final = database.token_refresh
    assert token_refresh is not None and not token_refresh.done()

    await database.aclose()

    assert token_refresh.cancelled()


@pytest.mark.asyncio
async def test_rejected_native_settings_still_close_prisma() -> None:
    aclose: Final = AsyncMock()

    with pytest.raises(ValueError, match="must name a host and a user"):
        await Database.open(_prisma(aclose), "postgresql://db.invalid/proxy")

    aclose.assert_awaited_once()
