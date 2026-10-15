import asyncio
import contextlib
import json
import socket
import ssl
import struct
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from integration._support.tls import server_context, write_self_signed_cert

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, MaskedHTTPStatusError
from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api import GenericGuardrailAPI

Cut = Literal["close", "reset"]
Transport = Literal["aiohttp", "httpx"]


async def _read_one_request_body(reader: asyncio.StreamReader) -> bytes:
    head: Final = await reader.readuntil(b"\r\n\r\n")
    body_length: Final = next(
        (int(line.split(b":", 1)[1]) for line in head.lower().split(b"\r\n") if line.startswith(b"content-length:")),
        0,
    )
    return await reader.readexactly(body_length)


def _drop(writer: asyncio.StreamWriter, cut: Cut) -> None:
    if cut == "close":
        writer.close()
        return
    writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    writer.transport.abort()


@contextlib.asynccontextmanager
async def _upstream_that_cuts(
    status_line: bytes,
    body: bytes,
    tls: ssl.SSLContext | None = None,
    cut: Cut = "close",
    cut_every_connection: bool = False,
) -> AsyncIterator[tuple[str, "asyncio.Queue[bytes]"]]:
    """A load balancer that drops an idle keepalive connection answers the next request
    with a clean close ("close") or a TCP reset ("reset") and no response. The first
    connection here is cut that way, and later ones get the reply, with a pause before
    the body so a streamed read has to go back to the socket after the call returns."""
    received_bodies: Final = asyncio.Queue[bytes]()

    async def cut_then_answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        received_bodies.put_nowait(await _read_one_request_body(reader))
        if cut_every_connection or received_bodies.qsize() == 1:
            _drop(writer, cut)
            return
        writer.write(
            b"HTTP/1.1 %s\r\nContent-Type: application/json\r\nConnection: close\r\n"
            b"Content-Length: %d\r\n\r\n" % (status_line, len(body))
        )
        await writer.drain()
        await asyncio.sleep(0.2)
        writer.write(body)
        await writer.drain()
        writer.close()
        with contextlib.suppress(ConnectionError, OSError):
            await writer.wait_closed()

    server: Final = await asyncio.start_server(cut_then_answer, "127.0.0.1", 0, ssl=tls)
    scheme: Final = "https" if tls else "http"
    async with server:
        yield f"{scheme}://127.0.0.1:{server.sockets[0].getsockname()[1]}", received_bodies


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """aiohttp already resends a cut PUT or DELETE once on its own, so litellm's resend of
    those methods is only the one doing the work on the httpx transport."""
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")


@pytest.fixture(params=["aiohttp", "httpx"])
def transport(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[Transport]:
    if request.param == "httpx":
        monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield request.param
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
async def test_server_disconnect_before_a_delete_response_is_resent_on_a_fresh_connection(stream: bool) -> None:
    async with _upstream_that_cuts(b"200 OK", b'{"ok": true}') as (base_url, received):
        handler: Final = AsyncHTTPHandler(timeout=5.0)
        try:
            response: Final = await handler.delete(f"{base_url}/", json={"hello": "world"}, stream=stream)
            await response.aread()
        finally:
            await handler.close()

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert received.qsize() == 2


@pytest.mark.usefixtures("httpx_transport")
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
async def test_status_error_on_the_resend_is_masked_like_the_first_attempt(stream: bool) -> None:
    async with _upstream_that_cuts(b"429 Too Many Requests", b'{"error": "slow down"}') as (base_url, received):
        handler: Final = AsyncHTTPHandler(timeout=5.0)
        try:
            with pytest.raises(MaskedHTTPStatusError) as raised:
                await handler.delete(f"{base_url}/v1/x?key=SUPERSECRET", json={"hello": "world"}, stream=stream)
        finally:
            await handler.close()

    assert raised.value.status_code == 429
    assert raised.value.response.json() == {"error": "slow down"}
    assert "SUPERSECRET" not in str(raised.value)
    assert "SUPERSECRET" not in str(raised.value.request.url)
    assert received.qsize() == 2


@pytest.mark.usefixtures("httpx_transport")
async def test_resend_keeps_the_handlers_tls_settings(tmp_path: Path) -> None:
    cert_file, key_file = write_self_signed_cert(tmp_path)
    async with _upstream_that_cuts(b"200 OK", b'{"ok": true}', tls=server_context(cert_file, key_file)) as (
        base_url,
        received,
    ):
        handler: Final = AsyncHTTPHandler(timeout=5.0, ssl_verify=False)
        try:
            response: Final = await handler.delete(f"{base_url}/")
        finally:
            await handler.close()

    assert response.json() == {"ok": True}
    assert received.qsize() == 2


@pytest.mark.parametrize("method", ["put", "delete"])
async def test_a_put_or_delete_to_an_upstream_that_keeps_cutting_is_sent_at_most_twice(
    transport: Transport, method: str
) -> None:
    async with _upstream_that_cuts(b"200 OK", b"{}", cut_every_connection=True) as (base_url, received):
        handler: Final = AsyncHTTPHandler(timeout=5.0)
        try:
            with pytest.raises(httpx.TransportError):
                await getattr(handler, method)(f"{base_url}/", json={"hello": "world"})
        finally:
            await handler.close()

    assert received.qsize() == 2


@pytest.mark.parametrize("cut", ["close", "reset"])
async def test_a_cut_post_is_not_replayed(transport: Transport, cut: Cut) -> None:
    async with _upstream_that_cuts(b"200 OK", b'{"ok": true}', cut=cut) as (base_url, received):
        handler: Final = AsyncHTTPHandler(timeout=5.0)
        try:
            with pytest.raises(httpx.TransportError):
                await handler.post(f"{base_url}/", json={"hello": "world"})
        finally:
            await handler.close()

    assert received.qsize() == 1


@pytest.mark.parametrize("cut", ["close", "reset"])
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
async def test_a_cut_idempotent_post_is_resent_on_a_fresh_connection(
    transport: Transport, cut: Cut, stream: bool
) -> None:
    async with _upstream_that_cuts(b"200 OK", b'{"ok": true}', cut=cut) as (base_url, received):
        handler: Final = AsyncHTTPHandler(timeout=5.0)
        try:
            response: Final = await handler.post(
                f"{base_url}/", json={"hello": "world"}, stream=stream, idempotent=True
            )
            await response.aread()
        finally:
            await handler.close()

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert received.qsize() == 2


@pytest.mark.parametrize("cut", ["close", "reset"])
async def test_guardrail_check_cut_before_a_response_is_resent_once(transport: Transport, cut: Cut) -> None:
    async with _upstream_that_cuts(b"200 OK", b'{"action": "NONE"}', cut=cut) as (api_base, received):
        result: Final = await GenericGuardrailAPI(
            api_base=api_base, guardrail_name="cut-once-guardrail", event_hook="pre_call"
        ).apply_guardrail(
            inputs={"texts": ["hi"]},
            request_data={},
            input_type="request",
        )

    assert result == {"texts": ["hi"]}
    assert received.qsize() == 2
    first_body: Final = received.get_nowait()
    assert json.loads(first_body)["texts"] == ["hi"]
    assert received.get_nowait() == first_body
