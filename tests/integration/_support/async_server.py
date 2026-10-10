from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from threading import Thread
from typing import Final

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


async def close_writer(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with suppress(ConnectionError, OSError):
        await writer.wait_closed()


class LoopbackServer:
    def __init__(self, port: int, handler: Handler) -> None:
        self.loop: Final = asyncio.new_event_loop()
        self._port: Final = port
        self._handler: Final = handler
        self._server: asyncio.Server | None = None
        self._thread: Final = Thread(target=self._run, name=f"integration-relay-{port}", daemon=True)
        self._connections: tuple[asyncio.Task[None], ...] = ()

    def start(self) -> None:
        try:
            self._thread.start()
            asyncio.run_coroutine_threadsafe(self._bind(), self.loop).result(timeout=10)
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if not self._thread.is_alive():
            if not self.loop.is_closed():
                self.loop.close()
            return
        try:
            asyncio.run_coroutine_threadsafe(self._close(), self.loop).result(timeout=10)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)
            self._thread.join(10)
            assert not self._thread.is_alive(), f"Database relay thread on port {self._port} did not stop"

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.close()

    async def _bind(self) -> None:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", self._port)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task: Final = asyncio.current_task()
        assert task is not None
        self._connections = (*self._connections, task)
        try:
            await self._handler(reader, writer)
        finally:
            await close_writer(writer)
            self._connections = tuple(connection for connection in self._connections if connection is not task)

    async def _close(self) -> None:
        server: Final = self._server
        if server is not None:
            server.close()
        connections: Final = self._connections
        for task in connections:
            task.cancel()
        await asyncio.gather(*connections, return_exceptions=True)
        current: Final = asyncio.current_task()
        remaining: Final = tuple(task for task in asyncio.all_tasks(self.loop) if task is not current)
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        if server is not None:
            await server.wait_closed()
