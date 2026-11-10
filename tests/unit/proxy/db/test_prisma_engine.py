import asyncio
from typing import Final, NoReturn

import pytest
from prisma import Prisma
from prisma.engine.errors import EngineConnectionError
from prisma.engine.query import QueryEngine

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.prisma_engine import _listening_url


@pytest.mark.parametrize(
    ("line", "expected"),
    (
        (
            b'{"target":"query_engine::server","fields":{"ip":"127.0.0.1","port":"31415",'
            b'"message":"Started query engine http server on http://127.0.0.1:31415"}}',
            "http://127.0.0.1:31415",
        ),
        (b"", None),
        (b"query-engine version", None),
        (b'{"target":"query_engine","fields":{"message":"PANIC","reason":"Address already in use"}}', None),
        (
            b'{"target":"other","fields":{"ip":"127.0.0.1","port":"31415",'
            b'"message":"Started query engine http server on http://127.0.0.1:31415"}}',
            None,
        ),
        (
            b'{"target":"query_engine::server","fields":{"ip":"0.0.0.0","port":"31415",'
            b'"message":"Started query engine http server on http://0.0.0.0:31415"}}',
            None,
        ),
        (
            b'{"target":"query_engine::server","fields":{"ip":"127.0.0.1","port":"0",'
            b'"message":"Started query engine http server on http://127.0.0.1:0"}}',
            None,
        ),
        (
            b'{"target":"query_engine::server","fields":{"ip":"127.0.0.1","port":"65536",'
            b'"message":"Started query engine http server on http://127.0.0.1:65536"}}',
            None,
        ),
        (
            b'{"target":"query_engine::server","fields":{"ip":"127.0.0.1","port":"31415",'
            b'"message":"Other server event"}}',
            None,
        ),
    ),
)
def test_listening_url_only_accepts_an_engine_listening_event(line: bytes, expected: str | None) -> None:
    assert _listening_url(line) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", (EngineConnectionError("engine startup failed"), asyncio.CancelledError()))
async def test_failed_engine_startup_clears_connection_state(
    fault: EngineConnectionError | asyncio.CancelledError, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable_engine(_: QueryEngine) -> NoReturn:
        raise fault

    monkeypatch.setattr(QueryEngine, "_ensure_file", unavailable_engine)
    original: Final = Prisma(datasource={"url": "postgresql://unused@localhost/unused"})
    client: Final = PrismaWrapper(original)
    try:
        with pytest.raises(type(fault)) as caught:
            await client.connect()
        assert caught.value is fault
        assert original.is_connected() is False
    finally:
        await client.disconnect()
