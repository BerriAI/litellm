import asyncio
import inspect
import json
import os
import signal
import subprocess
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from threading import Thread
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma.types import DatasourceOverride


class _AddressFields(BaseModel):
    model_config = ConfigDict(frozen=True)
    ip: Literal["127.0.0.1"]
    port: int = Field(gt=0, le=65535)
    message: str


class _ListeningEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    target: Literal["query_engine::server"]
    fields: _AddressFields


def _listening_url(line: bytes) -> str | None:
    try:
        event: Final = _ListeningEvent.model_validate_json(line)
    except ValidationError:
        return None
    if not event.fields.message.startswith("Started query engine http server on "):
        return None
    return f"http://{event.fields.ip}:{event.fields.port}"


def _publish_address(ready: asyncio.Future[str | None], address: str | None) -> None:
    if not ready.done():
        ready.set_result(address)


def _forward_output(stream: BinaryIO, loop: asyncio.AbstractEventLoop, ready: asyncio.Future[str | None]) -> None:
    def publish(address: str | None) -> None:
        try:
            loop.call_soon_threadsafe(_publish_address, ready, address)
        except RuntimeError:
            pass

    try:
        with stream:
            for line in stream:
                sys.stdout.write(line.decode("utf-8", errors="replace"))
                sys.stdout.flush()
                if (address := _listening_url(line)) is not None:
                    publish(address)
    finally:
        publish(None)


async def _connect_when_listening(
    ready: asyncio.Future[str | None], connect: Callable[[str], Awaitable[None]], timeout: timedelta
) -> Literal["connected", "exited", "timed out"]:
    async def attempt() -> Literal["connected", "exited"]:
        address: Final = await ready
        if address is None:
            return "exited"
        await connect(address)
        return "connected"

    try:
        return await asyncio.wait_for(attempt(), timeout.total_seconds())
    except asyncio.TimeoutError:
        return "timed out"


def _engine_http_options(engine: type[object], http_config: Mapping[str, object]) -> Mapping[str, object]:
    if "http_config" in inspect.signature(engine).parameters:
        return MappingProxyType({"http_config": http_config})
    return http_config


def _unblock_signals() -> None:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGINT, signal.SIGTERM))


def _launch_engine(argv: tuple[str, ...], environment: Mapping[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        argv,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        preexec_fn=None if sys.platform == "win32" else _unblock_signals,
    )


def initialize_query_engine(
    client: "Prisma",
    launch: Callable[[tuple[str, ...], Mapping[str, str]], subprocess.Popen[bytes]] = _launch_engine,
) -> None:
    from prisma._constants import DEFAULT_CONNECT_TIMEOUT
    from prisma.client import PACKAGED_SCHEMA_PATH
    from prisma.engine import errors
    from prisma.engine.query import QueryEngine
    from prisma.utils import (
        DEBUG,
        _env_bool,  # pyright: ignore[reportPrivateUsage]  # Preserve Prisma's playground flag semantics
    )

    class PortOwningQueryEngine(QueryEngine):
        async def spawn(
            self,
            file: Path,
            timeout: timedelta = DEFAULT_CONNECT_TIMEOUT,
            datasources: Sequence["DatasourceOverride"] | None = None,
        ) -> None:
            playground: Final = _env_bool("__PRISMA_PY_PLAYGROUND")
            environment: Final = {
                **os.environ,
                "PRISMA_DML_PATH": str(self.dml_path.absolute()),
                "RUST_LOG": "info" if DEBUG or playground else "error,query_engine::server=info",
                "RUST_LOG_FORMAT": "json",
                "PRISMA_CLIENT_ENGINE_TYPE": "binary",
                "PRISMA_ENGINE_PROTOCOL": "graphql",
                **(
                    {"OVERWRITE_DATASOURCES": json.dumps(datasources, ensure_ascii=False)}
                    if datasources is not None
                    else {}
                ),
                **({"LOG_QUERIES": "y"} if self._log_queries else {}),
            }
            self.process = launch(
                (
                    str(file.absolute()),
                    "-p",
                    "0",
                    "--enable-metrics",
                    "--enable-raw-queries",
                    *(("--enable-playground",) if playground else ()),
                ),
                environment,
            )
            stream: Final = self.process.stdout
            if stream is None:
                raise errors.EngineConnectionError("Query engine output pipe is unavailable")
            loop: Final = asyncio.get_running_loop()
            ready: Final[asyncio.Future[str | None]] = loop.create_future()
            Thread(target=_forward_output, args=(stream, loop, ready), daemon=True).start()

            async def connect(address: str) -> None:
                self.url = address
                await self.request("GET", "/status")

            match await _connect_when_listening(ready, connect, timeout):
                case "connected":
                    return
                case "exited":
                    raise errors.EngineConnectionError("Query engine exited before publishing its listening address")
                case "timed out":
                    raise errors.EngineConnectionError("Timed out waiting for the query engine's listening address")

    client._engine = PortOwningQueryEngine(  # pyright: ignore[reportPrivateUsage]  # Prisma exposes engine injection only through this setter  # rebind-ok: install the lifecycle-owned transport
        dml_path=PACKAGED_SCHEMA_PATH,
        log_queries=client._log_queries,  # pyright: ignore[reportPrivateUsage]  # Preserve the generated client's query logging
        **_engine_http_options(
            PortOwningQueryEngine,
            client._http_config,  # pyright: ignore[reportPrivateUsage]  # Preserve the generated client's HTTP options
        ),
    )
