import asyncio
import os
import socket
from datetime import timedelta
from typing import Final
from urllib.parse import urlsplit

import pytest
from prisma import Prisma
from prisma.engine.errors import AlreadyConnectedError, EngineConnectionError

from litellm.caching.caching import DualCache
from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.utils import PrismaClient, ProxyLogging
from tests.integration._support.database import scratch_database


@pytest.mark.asyncio
@pytest.mark.parametrize("occupied", (False, True), ids=("normal", "occupied-port"))
async def test_prisma_connect_owns_its_listening_port(occupied: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
        competitor.bind(("127.0.0.1", 0))  # test-quality-ok: keep the socket bound to force Prisma's port collision
        port: Final = competitor.getsockname()[1]
        if occupied:
            monkeypatch.setattr("prisma.engine.utils.get_open_port", lambda: port)
        monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
        client: Final = PrismaClient(os.environ["DATABASE_URL"], ProxyLogging(DualCache()))
        try:
            await client.db.connect(timeout=timedelta(seconds=5))
            with pytest.raises(AlreadyConnectedError):
                await client.db.connect(timeout=timedelta(seconds=5))
            assert await client.db.query_raw("SELECT 1 AS result") == [{"result": 1}]
            assert client.db.is_connected() is True
        finally:
            await client.db.disconnect()


@pytest.mark.asyncio
async def test_concurrent_prisma_engines_query_their_own_databases() -> None:
    with scratch_database("engine_first") as first, scratch_database("engine_second") as second:
        clients: Final = tuple(PrismaWrapper(Prisma(datasource={"url": url})) for url in (first, second))
        try:
            await asyncio.gather(*(client.connect() for client in clients))
            results: Final = await asyncio.gather(
                *(client.query_raw("SELECT current_database() AS name") for client in clients)
            )
            assert results == [[{"name": urlsplit(url).path.lstrip("/")}] for url in (first, second)]
        finally:
            await asyncio.gather(*(client.disconnect() for client in clients))


@pytest.mark.asyncio
async def test_prisma_startup_database_failure_closes_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as unavailable:
        unavailable.bind(  # test-quality-ok: socket remains bound throughout the connection attempt
            ("127.0.0.1", 0)
        )
        url: Final = (
            f"postgresql://invalid:invalid@127.0.0.1:{unavailable.getsockname()[1]}/unavailable?connect_timeout=1"
        )
        original: Final = Prisma(datasource={"url": url})
        client: Final = PrismaWrapper(original)
        connecting: Final = asyncio.create_task(client.connect(timeout=timedelta(seconds=5)))
        try:
            await asyncio.sleep(0)
            engine: Final = original._engine
            with pytest.raises(EngineConnectionError):
                await connecting
            assert original.is_connected() is False
            assert engine.process is None
            assert engine.session.should_close() is False
            monkeypatch.setattr(original, "_datasource", {"url": os.environ["DATABASE_URL"]})
            await client.connect(timeout=timedelta(seconds=5))
            assert await client.query_raw("SELECT 1 AS result") == [{"result": 1}]
            assert original.is_connected() is True
        finally:
            await client.disconnect()


@pytest.mark.asyncio
async def test_cancelled_prisma_startup_stops_its_engine() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stalled_database:
        stalled_database.bind(("127.0.0.1", 0))  # test-quality-ok: socket remains bound during the stalled startup
        stalled_database.listen()
        url: Final = f"postgresql://invalid:invalid@127.0.0.1:{stalled_database.getsockname()[1]}/stalled"
        original: Final = Prisma(datasource={"url": url})
        client: Final = PrismaWrapper(original)
        connecting: Final = asyncio.create_task(client.connect())
        try:
            await asyncio.sleep(0)
            process: Final = original._engine.process
            assert process is not None
            connecting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await connecting
            assert process.poll() is not None
            assert original.is_connected() is False
        finally:
            await client.disconnect()
