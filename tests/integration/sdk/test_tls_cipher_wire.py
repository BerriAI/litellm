from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter

if TYPE_CHECKING:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from tests.integration._support.client import eventually
from tests.integration._support.tls import TlsPeer, tls_peer, write_self_signed_cert

CHACHA20: Final = "ECDHE-RSA-CHACHA20-POLY1305"
AES256_GCM: Final = "AES256-GCM-SHA384"
AES128_GCM: Final = "AES128-GCM-SHA256"
CBC_SHA256: Final = "ECDHE-RSA-AES128-SHA256"
CBC_SHA384: Final = "ECDHE-RSA-AES256-SHA384"
PFS_GCM: Final = "ECDHE-RSA-AES256-GCM-SHA384"
BURST_SIZE: Final = 30
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

OVERRIDE_EXCHANGE: Final = textwrap.dedent(
    """
    import sys
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    handler = HTTPHandler(ssl_verify=sys.argv[2])
    try:
        response = handler.post(f"{sys.argv[1]}/v1/chat/completions", json={"user": "u-override"})
        print(response.json()["id"])
    finally:
        handler.close()
    """
)


@pytest.fixture(scope="module")
def tls_cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("tlscerts"))


@dataclass(frozen=True, slots=True)
class BurstOutcome:
    user: str
    when: float
    response_id: str | None
    error: Exception | None


def _sync_post(cert_file: Path, url: str, user: str) -> str:
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    handler: Final = HTTPHandler(ssl_verify=str(cert_file))
    try:
        response: Final = handler.client.post(f"{url}/v1/chat/completions", json={"user": user})
        return str(JSON_OBJECT.validate_json(response.content)["id"])
    finally:
        handler.close()


def _async_post(cert_file: Path, url: str, user: str) -> str:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    async def _run() -> str:
        handler: Final = AsyncHTTPHandler(ssl_verify=str(cert_file))
        try:
            response: Final = await handler.client.post(f"{url}/v1/chat/completions", json={"user": user})
            return str(JSON_OBJECT.validate_json(response.content)["id"])
        finally:
            await handler.close()

    return asyncio.run(_run())


def _assert_served(peer: TlsPeer, user: str, response_id: str) -> None:
    records: Final = peer.received()
    assert response_id == f"chatcmpl-{user}"
    assert [(record.user, record.cipher) for record in records] == [(user, peer.cipher)], records


def test_sync_handler_refuses_chacha20_only_peer(tls_cert: tuple[Path, Path]) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(CHACHA20, cert_file, key_file) as peer:
        with pytest.raises(httpx.ConnectError, match=r"(?i)ssl|handshake"):
            _sync_post(cert_file, peer.url, "u-cell1")
        assert peer.received() == (), peer.received()


def test_async_httpx_handler_refuses_chacha20_only_peer(
    monkeypatch: pytest.MonkeyPatch, tls_cert: tuple[Path, Path]
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    cert_file, key_file = tls_cert
    with tls_peer(CHACHA20, cert_file, key_file) as peer:
        with pytest.raises(httpx.ConnectError, match=r"(?i)ssl|handshake"):
            _async_post(cert_file, peer.url, "u-cell2")
        assert peer.received() == (), peer.received()


def test_async_aiohttp_handler_refuses_chacha20_only_peer(tls_cert: tuple[Path, Path]) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(CHACHA20, cert_file, key_file) as peer:
        with pytest.raises(httpx.ProtocolError, match=r"(?i)ssl|handshake"):
            _async_post(cert_file, peer.url, "u-cell3")
        assert peer.received() == (), peer.received()


def test_sync_handler_refuses_non_pfs_aes256_gcm_peer(tls_cert: tuple[Path, Path]) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(AES256_GCM, cert_file, key_file) as peer:
        with pytest.raises(httpx.ConnectError, match=r"(?i)ssl|handshake"):
            _sync_post(cert_file, peer.url, "u-cell4")
        assert peer.received() == (), peer.received()


def test_async_httpx_handler_refuses_non_pfs_aes128_gcm_peer(
    monkeypatch: pytest.MonkeyPatch, tls_cert: tuple[Path, Path]
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    cert_file, key_file = tls_cert
    with tls_peer(AES128_GCM, cert_file, key_file) as peer:
        with pytest.raises(httpx.ConnectError, match=r"(?i)ssl|handshake"):
            _async_post(cert_file, peer.url, "u-cell5")
        assert peer.received() == (), peer.received()


def test_sync_handler_accepts_cbc_peer_by_default(tls_cert: tuple[Path, Path]) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(CBC_SHA256, cert_file, key_file) as peer:
        _assert_served(peer, "u-cell6", _sync_post(cert_file, peer.url, "u-cell6"))


def test_sync_handler_refuses_cbc_peer_in_fips_mode(
    monkeypatch: pytest.MonkeyPatch, tls_cert: tuple[Path, Path]
) -> None:
    monkeypatch.setenv("LITELLM_FIPS_MODE", "true")
    cert_file, key_file = tls_cert
    with tls_peer(CBC_SHA256, cert_file, key_file) as peer:
        with pytest.raises(httpx.ConnectError, match=r"(?i)ssl|handshake"):
            _sync_post(cert_file, peer.url, "u-cell7")
        assert peer.received() == (), peer.received()


def test_async_httpx_handler_refuses_cbc_sha384_peer_in_fips_mode(
    monkeypatch: pytest.MonkeyPatch, tls_cert: tuple[Path, Path]
) -> None:
    monkeypatch.setenv("LITELLM_FIPS_MODE", "true")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    cert_file, key_file = tls_cert
    with tls_peer(CBC_SHA384, cert_file, key_file) as peer:
        with pytest.raises(httpx.ConnectError, match=r"(?i)ssl|handshake"):
            _async_post(cert_file, peer.url, "u-cell8")
        assert peer.received() == (), peer.received()


def test_handlers_accept_pfs_gcm_peer_in_default_and_fips_modes(
    monkeypatch: pytest.MonkeyPatch, tls_cert: tuple[Path, Path]
) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(PFS_GCM, cert_file, key_file) as peer:
        served: Final = _sync_post(cert_file, peer.url, "u-cell9-sync")
        assert served == "chatcmpl-u-cell9-sync"
        served_async: Final = _async_post(cert_file, peer.url, "u-cell9-async")
        assert served_async == "chatcmpl-u-cell9-async"
        monkeypatch.setenv("LITELLM_FIPS_MODE", "true")
        served_fips: Final = _sync_post(cert_file, peer.url, "u-cell9-fips")
        assert served_fips == "chatcmpl-u-cell9-fips"
        served_async_fips: Final = _async_post(cert_file, peer.url, "u-cell9-async-fips")
        assert served_async_fips == "chatcmpl-u-cell9-async-fips"
        assert {record.cipher for record in peer.received()} == {PFS_GCM}


def _override_exchange(url: str, cert_file: Path, overrides: Mapping[str, str]) -> str:
    completed: Final = subprocess.run(
        [sys.executable, "-P", "-c", OVERRIDE_EXCHANGE, url, str(cert_file)],
        env={**os.environ, **overrides},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def test_ssl_ciphers_env_override_keeps_chacha20(tls_cert: tuple[Path, Path]) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(CHACHA20, cert_file, key_file) as peer:
        assert _override_exchange(peer.url, cert_file, {"LITELLM_SSL_CIPHERS": CHACHA20}) == "chatcmpl-u-override"
        assert [record.cipher for record in peer.received()] == [CHACHA20]


def test_ssl_ciphers_override_wins_over_fips_mode(tls_cert: tuple[Path, Path]) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(CHACHA20, cert_file, key_file) as peer:
        result: Final = _override_exchange(
            peer.url, cert_file, {"LITELLM_SSL_CIPHERS": CHACHA20, "LITELLM_FIPS_MODE": "true"}
        )
        assert result == "chatcmpl-u-override"
        assert [record.cipher for record in peer.received()] == [CHACHA20]


def test_cipher_list_follows_fips_mode_toggle_in_one_process(
    monkeypatch: pytest.MonkeyPatch, tls_cert: tuple[Path, Path]
) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(CBC_SHA256, cert_file, key_file) as peer:
        assert _sync_post(cert_file, peer.url, "u-cell12-before") == "chatcmpl-u-cell12-before"
        monkeypatch.setenv("LITELLM_FIPS_MODE", "true")
        with pytest.raises(httpx.ConnectError, match=r"(?i)ssl|handshake"):
            _sync_post(cert_file, peer.url, "u-cell12-fips")
        monkeypatch.delenv("LITELLM_FIPS_MODE")
        assert _sync_post(cert_file, peer.url, "u-cell12-after") == "chatcmpl-u-cell12-after"
        assert [record.user for record in peer.received()] == ["u-cell12-before", "u-cell12-after"]


def _assert_burst_outcomes(outcomes: Mapping[int, BurstOutcome]) -> None:
    for index, outcome in outcomes.items():
        if outcome.error is None:
            assert outcome.response_id == f"chatcmpl-user-{index}", outcome
            continue
        message: str = str(outcome.error).lower()
        assert "handshake" not in message and "cipher" not in message, (
            f"user-{index} failed with a cipher class error: {outcome.error!r}"
        )
        assert isinstance(outcome.error, httpx.HTTPError), f"user-{index}: {outcome.error!r}"


def _peer_ready(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def test_burst_survives_peer_restart_without_cipher_errors(tls_cert: tuple[Path, Path]) -> None:
    cert_file, key_file = tls_cert
    with tls_peer(PFS_GCM, cert_file, key_file) as peer:
        second_wave: Final = threading.Event()
        outcomes: Final[dict[int, BurstOutcome]] = {}
        outcomes_lock: Final = threading.Lock()

        def _record(index: int, outcome: BurstOutcome) -> None:
            with outcomes_lock:
                outcomes[index] = outcome

        def _snapshot() -> tuple[BurstOutcome, ...]:
            with outcomes_lock:
                return tuple(outcomes.values())

        def _one(index: int) -> None:
            if index >= 10:
                second_wave.wait(timeout=30)
            try:
                response_id: Final = _sync_post(cert_file, peer.url, f"user-{index}")
                _record(index, BurstOutcome(f"user-{index}", time.monotonic(), response_id, None))
            except Exception as error:
                _record(index, BurstOutcome(f"user-{index}", time.monotonic(), None, error))

        async def _async_burst() -> None:
            from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

            handler: Final = AsyncHTTPHandler(ssl_verify=str(cert_file))
            try:
                await asyncio.to_thread(second_wave.wait, 30)
                await asyncio.gather(*(_async_one(handler, index) for index in range(15, BURST_SIZE)))
            finally:
                await handler.close()

        async def _async_one(handler: AsyncHTTPHandler, index: int) -> None:
            try:
                response: Final = await handler.client.post(
                    f"{peer.url}/v1/chat/completions", json={"user": f"user-{index}"}
                )
                _record(
                    index,
                    BurstOutcome(
                        f"user-{index}", time.monotonic(), str(JSON_OBJECT.validate_json(response.content)["id"]), None
                    ),
                )
            except Exception as error:
                _record(index, BurstOutcome(f"user-{index}", time.monotonic(), None, error))

        def _async_thread() -> None:
            asyncio.run(_async_burst())

        async_thread: Final = threading.Thread(target=_async_thread, daemon=True)
        with ThreadPoolExecutor(max_workers=15) as pool:
            async_thread.start()
            futures: Final = tuple(pool.submit(_one, index) for index in range(15))
            eventually(
                lambda: sum(1 for outcome in _snapshot() if outcome.error is None) >= 3,
                lambda ready: ready,
                seconds=15,
            )
            stopped_at: Final = time.monotonic()
            peer.stop()
            peer.start()
            eventually(lambda: _peer_ready(peer.port), lambda ready: ready, seconds=15)
            restarted_at: Final = time.monotonic()
            second_wave.set()
            for future in futures:
                future.result(timeout=60)
            async_thread.join(timeout=60)

        finished: Final = _snapshot()
        assert len(finished) == BURST_SIZE
        _assert_burst_outcomes({int(outcome.user.removeprefix("user-")): outcome for outcome in finished})
        succeeded: Final = {outcome.user for outcome in finished if outcome.error is None}
        recorded: Final = peer.received()
        assert sorted(record.user or "" for record in recorded) == sorted(succeeded)
        assert all(record.cipher == PFS_GCM for record in recorded)
        for user in succeeded:
            assert sum(1 for record in recorded if record.user == user) == 1, user
        assert any(outcome.error is None and outcome.when < stopped_at for outcome in finished), (
            "no success before restart"
        )
        assert any(outcome.error is None and outcome.when > restarted_at for outcome in finished), (
            "no success after restart"
        )
