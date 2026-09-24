from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from openai import APIStatusError, AsyncOpenAI, OpenAI
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.tls import TlsPeer, tls_peer, write_self_signed_cert

CHACHA20: Final = "ECDHE-RSA-CHACHA20-POLY1305"
NONPFS_GCM: Final = "AES256-GCM-SHA384"
PFS_GCM: Final = "ECDHE-RSA-AES256-GCM-SHA384"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class Peers:
    gcm: TlsPeer
    chacha: TlsPeer
    nonpfs: TlsPeer


@pytest.fixture(scope="module")
def tls_cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("tlscerts"))


@pytest.fixture(scope="module")
def tls_peers(tls_cert: tuple[Path, Path]) -> Iterator[Peers]:
    cert_file, key_file = tls_cert
    with (
        tls_peer(PFS_GCM, cert_file, key_file) as gcm,
        tls_peer(CHACHA20, cert_file, key_file) as chacha,
        tls_peer(NONPFS_GCM, cert_file, key_file) as nonpfs,
    ):
        yield Peers(gcm=gcm, chacha=chacha, nonpfs=nonpfs)


@pytest.fixture(scope="module")
def cipher_proxy(
    tls_cert: tuple[Path, Path], tls_peers: Peers, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    cert_file, _ = tls_cert
    directory: Final = tmp_path_factory.mktemp("tlsproxy")
    base_config: Final = JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    config: Final = {
        **base_config,
        "model_list": [
            {
                "model_name": "tls-peer",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{tls_peers.gcm.url}/v1",
                    "api_key": "sk-peer",
                },
            },
            {
                "model_name": "tls-chacha-peer",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{tls_peers.chacha.url}/v1",
                    "api_key": "sk-peer",
                },
            },
            {
                "model_name": "tls-nonpfs-peer",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{tls_peers.nonpfs.url}/v1",
                    "api_key": "sk-peer",
                },
            },
        ],
    }
    config_path: Final = directory / "tls_cipher_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with gateway_from_environment() as gateway:
        with owned_proxy(gateway, directory, {"SSL_CERT_FILE": str(cert_file)}, config=config_path) as candidate:
            yield candidate


def _sync_completion(proxy: Gateway, model: str, user: str, *, stream: bool = False) -> str:
    with OpenAI(api_key=proxy.key, base_url=f"{proxy.client.base_url}/v1", max_retries=0, timeout=60) as client:
        if stream:
            chunks: Final = list(
                client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "tls"}], user=user, stream=True
                )
            )
            assert chunks, "stream produced no chunks"
            assert all(chunk.id == f"chatcmpl-{user}" for chunk in chunks), chunks[0].id
            return str(chunks[0].id)
        response: Final = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "tls"}], user=user
        )
        return str(response.id)


async def _async_completion(proxy: Gateway, model: str, user: str, *, stream: bool = False) -> str:
    async with AsyncOpenAI(
        api_key=proxy.key, base_url=f"{proxy.client.base_url}/v1", max_retries=0, timeout=60
    ) as client:
        if stream:
            chunk_ids: Final = [
                chunk.id
                async for chunk in await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": "tls"}], user=user, stream=True
                )
            ]
            assert chunk_ids, "stream produced no chunks"
            assert all(chunk_id == f"chatcmpl-{user}" for chunk_id in chunk_ids), chunk_ids[0]
            return str(chunk_ids[0])
        response: Final = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "tls"}], user=user
        )
        return str(response.id)


def test_proxy_round_trips_over_pfs_gcm_peer_and_writes_spend(cipher_proxy: Gateway, tls_peers: Peers) -> None:
    marker: Final = uuid.uuid4().hex[:8]
    sync_id: Final = _sync_completion(cipher_proxy, "tls-peer", f"user13-sync-{marker}")
    stream_id: Final = _sync_completion(cipher_proxy, "tls-peer", f"user13-stream-{marker}", stream=True)
    async_id: Final = asyncio.run(_async_completion(cipher_proxy, "tls-peer", f"user13-async-{marker}"))
    async_stream_id: Final = asyncio.run(
        _async_completion(cipher_proxy, "tls-peer", f"user13-astream-{marker}", stream=True)
    )
    users: Final = {f"user13-{leg}-{marker}" for leg in ("sync", "stream", "async", "astream")}
    recorded: Final = tls_peers.gcm.received()
    assert {record.user for record in recorded} == users, recorded
    assert all(record.cipher == PFS_GCM for record in recorded), recorded
    spend_rows: Final = tuple(
        eventually(
            lambda response_id=response_id: read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        for response_id in (sync_id, stream_id, async_id, async_stream_id)
    )
    assert [rows[0]["request_id"] for rows in spend_rows] == [sync_id, stream_id, async_id, async_stream_id]


def test_proxy_reports_ssl_error_for_chacha20_only_peer(cipher_proxy: Gateway, tls_peers: Peers) -> None:
    marker: Final = uuid.uuid4().hex[:8]
    before: Final = len(tls_peers.chacha.received())
    with pytest.raises(APIStatusError) as raised:
        _sync_completion(cipher_proxy, "tls-chacha-peer", f"user14-{marker}")
    assert raised.value.status_code >= 400
    assert "ssl" in str(raised.value).lower() or "connection error" in str(raised.value).lower(), str(raised.value)
    assert len(tls_peers.chacha.received()) == before
    readiness: Final = cipher_proxy.client.get("/health/readiness")
    assert readiness.status_code == 200, readiness.text
    recovered: Final = _sync_completion(cipher_proxy, "tls-peer", f"user14-ok-{marker}")
    assert recovered == f"chatcmpl-user14-ok-{marker}"


def test_proxy_reports_ssl_error_for_non_pfs_peer(cipher_proxy: Gateway, tls_peers: Peers) -> None:
    marker: Final = uuid.uuid4().hex[:8]
    before: Final = len(tls_peers.nonpfs.received())
    response: Final = cipher_proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "tls-nonpfs-peer",
            "messages": [{"role": "user", "content": "tls"}],
            "user": f"user15-{marker}",
        },
    )
    assert response.status_code >= 400, response.text
    assert "ssl" in response.text.lower() or "connection error" in response.text.lower(), response.text
    assert len(tls_peers.nonpfs.received()) == before
    readiness: Final = cipher_proxy.client.get("/health/readiness")
    assert readiness.status_code == 200, readiness.text
    recovered: Final = _sync_completion(cipher_proxy, "tls-peer", f"user15-ok-{marker}")
    assert recovered == f"chatcmpl-user15-ok-{marker}"
