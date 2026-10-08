import asyncio
from typing import Final

import pytest

from litellm.tracing.remote import LensConnection


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
