import asyncio
from typing import Final

import pytest

from litellm.rust_bridge.trace.generated.types import TraceScope
from litellm.tracing.remote import BACKGROUND_READ_CONNECTIONS, READ_CLASS_HEADER, LensConnection, RemoteTraceStore


@pytest.mark.asyncio
async def test_control_requests_reuse_connections_without_retaining_another_service_credential() -> None:
    requests: Final[asyncio.Queue[tuple[str, bytes]]] = asyncio.Queue()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                headers: Final = await reader.readuntil(b"\r\n\r\n")
                requests.put_nowait((str(writer.get_extra_info("peername")), headers))
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        finally:
            writer.close()
            await writer.wait_closed()

    async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
        port: Final = server.sockets[0].getsockname()[1]
        first: Final = LensConnection(f"http://127.0.0.1:{port}/one", "first-service-token")
        second: Final = LensConnection(f"http://127.0.0.1:{port}/two", "second-service-token")
        try:
            for connection in (first, second):
                response: Final = await connection.control_client().get(
                    connection.endpoint("/internal/status"), headers=connection.headers
                )
                assert response.json() == {}
            first_peer, first_request = await asyncio.wait_for(requests.get(), 2)
            second_peer, second_request = await asyncio.wait_for(requests.get(), 2)
            assert first_peer == second_peer
            assert b"GET /one/internal/status " in first_request
            assert b"GET /two/internal/status " in second_request
            assert b"Bearer first-service-token" in first_request
            assert b"Bearer second-service-token" not in first_request
            assert b"Bearer second-service-token" in second_request
            assert b"Bearer first-service-token" not in second_request
        finally:
            await second.control_client().aclose()


@pytest.mark.asyncio
async def test_saturated_background_reads_leave_interactive_connections_free() -> None:
    release: Final = asyncio.Event()
    stalled: Final[asyncio.Queue[bytes]] = asyncio.Queue()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            headers: Final = (await reader.readuntil(b"\r\n\r\n")).lower()
            length: Final = int(headers.split(b"content-length: ")[1].split(b"\r\n")[0])
            await reader.readexactly(length)
            if f"{READ_CLASS_HEADER}: background".encode() in headers:
                stalled.put_nowait(headers)
                await release.wait()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            await writer.drain()
        finally:
            writer.close()

    async with await asyncio.start_server(serve, "127.0.0.1", 0) as server:
        port: Final = server.sockets[0].getsockname()[1]
        connection: Final = LensConnection(f"http://127.0.0.1:{port}", "s" * 32)
        scope: Final = TraceScope(all_teams=1, user_id="", team_ids=())
        async with connection.lifespan_client() as interactive, connection.background_client() as background:
            backlog: Final = tuple(
                asyncio.create_task(RemoteTraceStore(background).query("content", {"id": str(index)}))
                for index in range(BACKGROUND_READ_CONNECTIONS * 3)
            )
            for _ in range(BACKGROUND_READ_CONNECTIONS):
                await asyncio.wait_for(stalled.get(), 2)
            await asyncio.sleep(0.1)
            assert stalled.empty()
            trace: Final = await asyncio.wait_for(RemoteTraceStore(interactive).get_trace("t", scope, "r"), 2)
            assert trace == {}
            assert not any(task.done() for task in backlog)
            release.set()
            assert await asyncio.gather(*backlog) == ["{}"] * len(backlog)
