import asyncio
import io
import json
import subprocess
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Final, Literal, NoReturn

import pytest
from prisma import Prisma
from prisma.engine.errors import EngineConnectionError
from prisma.engine.query import QueryEngine
from prisma.types import DatasourceOverride

from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.db.prisma_engine import (
    _connect_when_listening,
    _engine_http_options,
    _listening_url,
    initialize_query_engine,
)


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
    original: Final = Prisma(http={"timeout": 5}, datasource={"url": "postgresql://unused@localhost/unused"})
    client: Final = PrismaWrapper(original)
    try:
        with pytest.raises(type(fault)) as caught:
            await client.connect()
        assert caught.value is fault
        assert original.is_connected() is False
    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_the_engine_connects_to_the_address_it_publishes() -> None:
    loop: Final = asyncio.get_running_loop()
    ready: Final[asyncio.Future[str | None]] = loop.create_future()
    reached: Final[asyncio.Future[str]] = loop.create_future()
    ready.set_result("http://127.0.0.1:31415")

    async def connect(address: str) -> None:
        reached.set_result(address)

    assert await _connect_when_listening(ready, connect, timedelta(seconds=30)) == "connected"
    assert reached.result() == "http://127.0.0.1:31415"


@pytest.mark.asyncio
async def test_an_engine_that_exits_before_publishing_is_never_contacted() -> None:
    loop: Final = asyncio.get_running_loop()
    ready: Final[asyncio.Future[str | None]] = loop.create_future()
    reached: Final[asyncio.Future[str]] = loop.create_future()
    ready.set_result(None)

    async def connect(address: str) -> None:
        reached.set_result(address)

    assert await _connect_when_listening(ready, connect, timedelta(seconds=30)) == "exited"
    assert reached.done() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("stalls_at", ("publishing", "status"))
async def test_an_engine_that_stalls_times_out(stalls_at: Literal["publishing", "status"]) -> None:
    loop: Final = asyncio.get_running_loop()
    ready: Final[asyncio.Future[str | None]] = loop.create_future()
    never: Final[asyncio.Future[None]] = loop.create_future()
    if stalls_at == "status":
        ready.set_result("http://127.0.0.1:31415")

    async def connect(_: str) -> None:
        await never

    assert await _connect_when_listening(ready, connect, timedelta(milliseconds=10)) == "timed out"


class _EngineTakingAnHttpConfig:
    def __init__(self, *, dml_path: Path, http_config: Mapping[str, object] | None = None) -> None:
        self.http: Final = dict(http_config or {})


class _EngineTakingHttpKeywords:
    def __init__(self, *, dml_path: Path, **http: object) -> None:
        self.http: Final = http


@pytest.mark.parametrize("engine", (_EngineTakingAnHttpConfig, _EngineTakingHttpKeywords))
def test_the_client_http_options_reach_either_engine_constructor(
    engine: type[_EngineTakingAnHttpConfig] | type[_EngineTakingHttpKeywords],
) -> None:
    built: Final = engine(dml_path=Path("schema.prisma"), **_engine_http_options(engine, {"timeout": 5}))
    assert built.http == {"timeout": 5}


class _EngineThatExitsWithoutListening(subprocess.Popen[bytes]):
    def __init__(self, argv: tuple[str, ...], environment: Mapping[str, str]) -> None:
        self.argv: Final = argv
        self.environment: Final = environment
        self.stdout = io.BytesIO(b"query-engine banner that is not JSON\n")
        self.returncode = 1


@pytest.mark.asyncio
async def test_spawn_hands_the_engine_its_datasource_override_and_reports_an_exit_before_listening() -> None:
    client: Final = Prisma(http={"timeout": 5}, datasource={"url": "postgresql://unused@localhost/unused"})
    initialize_query_engine(client, launch=_EngineThatExitsWithoutListening)
    engine: Final = client._engine  # pyright: ignore[reportPrivateUsage]  # the engine initialize_query_engine installed
    assert isinstance(engine, QueryEngine)
    override: Final = DatasourceOverride(name="db", url="postgresql://litellm@db.internal:5432/litellm")

    with pytest.raises(EngineConnectionError, match="exited before publishing its listening address"):
        await engine.spawn(Path("query-engine"), timedelta(seconds=30), [override])

    process: Final = engine.process
    assert isinstance(process, _EngineThatExitsWithoutListening)
    assert process.argv[:3] == (str(Path("query-engine").absolute()), "-p", "0")
    assert json.loads(process.environment["OVERWRITE_DATASOURCES"]) == [override]
