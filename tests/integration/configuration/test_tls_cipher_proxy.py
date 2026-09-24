from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import yaml
from anthropic import Anthropic
from anthropic import APIStatusError as AnthropicAPIStatusError
from openai import APIStatusError, AsyncOpenAI, OpenAI
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, eventually, gateway_from_environment
from tests.integration._support.database import read_rows
from tests.integration._support.process import OwnedProxy, owned_proxy, owned_proxy_process
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
    base_settings: Final = JSON_OBJECT.validate_python(base_config.get("litellm_settings", {}))
    config: Final = {
        **base_config,
        "litellm_settings": {**base_settings, "cache": False},
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


@pytest.fixture(scope="module")
def cipher_owned_proxy(
    tls_cert: tuple[Path, Path], tls_peers: Peers, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[OwnedProxy]:
    cert_file, _ = tls_cert
    directory: Final = tmp_path_factory.mktemp("tlsproxyworkers")
    base_config: Final = JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    base_settings: Final = JSON_OBJECT.validate_python(base_config.get("litellm_settings", {}))
    config: Final = {
        **base_config,
        "litellm_settings": {**base_settings, "cache": False},
        "model_list": [
            {
                "model_name": "tls-peer",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{tls_peers.gcm.url}/v1",
                    "api_key": "sk-peer",
                },
            }
        ],
    }
    config_path: Final = directory / "tls_cipher_workers_config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    with gateway_from_environment() as gateway:
        with owned_proxy_process(
            gateway, directory, {"SSL_CERT_FILE": str(cert_file)}, config=config_path, workers=2
        ) as owned:
            yield owned


def _responses_request(proxy: Gateway, model: str, user: str) -> str:
    response: Final = proxy.request("POST", "/v1/responses", {"model": model, "input": "tls", "user": user})
    assert response.status_code == 200, response.text
    return str(JSON_OBJECT.validate_json(response.text)["id"])


def _messages_request(proxy: Gateway, model: str) -> str:
    with Anthropic(api_key=proxy.key, base_url=str(proxy.client.base_url), max_retries=0, timeout=60) as client:
        message: Final = client.messages.create(
            model=model, max_tokens=8, messages=[{"role": "user", "content": "tls"}]
        )
        return message.id


def test_proxy_round_trips_responses_over_pfs_gcm_peer_and_writes_spend(
    cipher_proxy: Gateway, tls_peers: Peers
) -> None:
    marker: Final = uuid.uuid4().hex[:8]
    user: Final = f"user19-resp-{marker}"
    before: Final = len(tls_peers.gcm.received())
    response_id: Final = _responses_request(cipher_proxy, "tls-peer", user)
    assert response_id.startswith("resp_"), response_id
    recorded: Final = tuple(
        record for record in tls_peers.gcm.received()[before:] if record.path.endswith("/responses")
    )
    assert [record.user for record in recorded] == [user], recorded
    assert all(record.cipher == PFS_GCM for record in recorded), recorded
    spend_rows: Final = eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (response_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert spend_rows[0]["request_id"] == response_id


def test_proxy_reports_ssl_error_for_responses_on_chacha20_only_peer(cipher_proxy: Gateway, tls_peers: Peers) -> None:
    before: Final = len(tls_peers.chacha.received())
    response: Final = cipher_proxy.request(
        "POST",
        "/v1/responses",
        {"model": "tls-chacha-peer", "input": "tls", "user": f"user20-{uuid.uuid4().hex[:8]}"},
    )
    assert response.status_code >= 400, response.text
    assert "ssl" in response.text.lower() or "connection error" in response.text.lower(), response.text
    assert len(tls_peers.chacha.received()) == before
    readiness: Final = cipher_proxy.client.get("/health/readiness")
    assert readiness.status_code == 200, readiness.text


def test_proxy_round_trips_messages_over_pfs_gcm_peer(cipher_proxy: Gateway, tls_peers: Peers) -> None:
    before: Final = len(tls_peers.gcm.received())
    message_id: Final = _messages_request(cipher_proxy, "tls-peer")
    assert message_id, "messages response carried no id"
    recorded: Final = tls_peers.gcm.received()
    assert len(recorded) - before == 1, recorded[before:]
    assert recorded[-1].cipher == PFS_GCM, recorded[-1]


def test_proxy_reports_ssl_error_for_messages_on_chacha20_only_peer(cipher_proxy: Gateway, tls_peers: Peers) -> None:
    before: Final = len(tls_peers.chacha.received())
    with pytest.raises(AnthropicAPIStatusError) as raised:
        _messages_request(cipher_proxy, "tls-chacha-peer")
    assert raised.value.status_code >= 400
    assert "ssl" in str(raised.value).lower() or "connection error" in str(raised.value).lower(), str(raised.value)
    assert len(tls_peers.chacha.received()) == before
    readiness: Final = cipher_proxy.client.get("/health/readiness")
    assert readiness.status_code == 200, readiness.text


@dataclass(frozen=True, slots=True)
class _BurstOutcome:
    user: str | None
    status: int
    body: str
    finished_at: float


def _chaos_request(proxy: Gateway, kind: str, index: int, marker: str) -> _BurstOutcome:
    user: Final = f"chaos-{kind}-{index}-{marker}"
    if kind in ("chat", "slow"):
        try:
            chat_response: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": "tls-peer",
                    "messages": [{"role": "user", "content": "tls"}],
                    "user": user,
                    "num_retries": 0,
                },
            )
            return _BurstOutcome(
                user=user, status=chat_response.status_code, body=chat_response.text, finished_at=time.monotonic()
            )
        except httpx.HTTPError:
            return _BurstOutcome(user=user, status=0, body="", finished_at=time.monotonic())
    if kind == "stream":
        try:
            stream_response: Final = proxy.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": "tls-peer",
                    "messages": [{"role": "user", "content": "tls"}],
                    "user": user,
                    "stream": True,
                    "num_retries": 0,
                },
            )
            return _BurstOutcome(
                user=user, status=stream_response.status_code, body=stream_response.text, finished_at=time.monotonic()
            )
        except httpx.HTTPError:
            return _BurstOutcome(user=user, status=0, body="", finished_at=time.monotonic())
    if kind == "responses":
        try:
            responses_response: Final = proxy.request(
                "POST",
                "/v1/responses",
                {"model": "tls-peer", "input": "tls", "user": user, "num_retries": 0},
            )
            return _BurstOutcome(
                user=user,
                status=responses_response.status_code,
                body=responses_response.text,
                finished_at=time.monotonic(),
            )
        except httpx.HTTPError:
            return _BurstOutcome(user=user, status=0, body="", finished_at=time.monotonic())
    try:
        messages_response: Final = proxy.request(
            "POST",
            "/v1/messages",
            {
                "model": "tls-peer",
                "max_tokens": 8,
                "messages": [{"role": "user", "content": f"tls {user}"}],
                "num_retries": 0,
            },
            headers={"anthropic-version": "2023-06-01"},
        )
        return _BurstOutcome(
            user=None,
            status=messages_response.status_code,
            body=messages_response.text,
            finished_at=time.monotonic(),
        )
    except httpx.HTTPError:
        return _BurstOutcome(user=None, status=0, body="", finished_at=time.monotonic())


def test_proxy_burst_survives_peer_restart_with_every_outcome_recorded(cipher_proxy: Gateway, tls_peers: Peers) -> None:
    marker: Final = uuid.uuid4().hex[:8]
    kinds: Final = ("chat", "stream", "responses", "messages", "chat", "stream")
    before: Final = len(tls_peers.gcm.received())
    executor: Final = ThreadPoolExecutor(max_workers=30)
    with executor:
        futures: Final = tuple(
            executor.submit(_chaos_request, cipher_proxy, kind, index, marker) for index in range(5) for kind in kinds
        )
        eventually(
            lambda: sum(1 for future in futures if future.done() and future.result().status == 200),
            lambda successes: successes >= 3,
            seconds=60,
        )
        tls_peers.gcm.stop()
        tls_peers.gcm.start()
        finished: Final = tuple(future.result(timeout=120) for future in futures)
    assert len(finished) == 30, finished
    assert not any(outcome.status == 0 for outcome in finished), finished
    records: Final = tls_peers.gcm.received()[before:]
    assert all(record.cipher == PFS_GCM for record in records), records
    submitted_users: Final = {outcome.user for outcome in finished if outcome.user is not None}
    successful_users: Final = {
        outcome.user for outcome in finished if outcome.status == 200 and outcome.user is not None
    }
    assert all(any(record.user == user for record in records) for user in successful_users), (successful_users, records)
    recorded_users: Final = {record.user for record in records if record.user is not None}
    assert recorded_users <= submitted_users, (recorded_users - submitted_users, records)
    assert all(sum(1 for record in records if record.user == user) <= 3 for user in submitted_users), (
        submitted_users,
        records,
    )
    anonymous: Final = sum(1 for record in records if record.user is None)
    messages_outcomes: Final = sum(1 for outcome in finished if outcome.user is None)
    messages_ok: Final = sum(1 for outcome in finished if outcome.user is None and outcome.status == 200)
    assert messages_ok <= anonymous <= messages_outcomes * 3, (anonymous, messages_ok, records)
    after_id: Final = _sync_completion(cipher_proxy, "tls-peer", f"chaos-after-{marker}")
    assert after_id == f"chatcmpl-chaos-after-{marker}"
    last_record: Final = tls_peers.gcm.received()[-1]
    assert last_record.cipher == PFS_GCM and last_record.user == f"chaos-after-{marker}", last_record
    readiness: Final = cipher_proxy.client.get("/health/readiness")
    assert readiness.status_code == 200, readiness.text


def test_proxy_burst_survives_losing_one_worker(cipher_owned_proxy: OwnedProxy, tls_peers: Peers) -> None:
    proxy: Final = cipher_owned_proxy.gateway
    marker: Final = uuid.uuid4().hex[:8]
    executor: Final = ThreadPoolExecutor(max_workers=12)
    with executor:
        futures: Final = tuple(
            executor.submit(_chaos_request, proxy, "slow" if index % 2 else "chat", index, marker)
            for index in range(12)
        )
        eventually(
            lambda: sum(1 for future in futures if future.done() and future.result().status == 200),
            lambda successes: successes >= 2,
            seconds=60,
        )
        children: Final = psutil.Process(cipher_owned_proxy.process.pid).children()
        assert children, "two-worker proxy has no child processes to kill"
        killed_at: Final = time.monotonic()
        children[0].kill()
        finished: Final = tuple(future.result(timeout=120) for future in futures)
    assert len(finished) == 12, finished
    assert any(outcome.finished_at > killed_at for outcome in finished), finished
    post_kill_id: Final = _sync_completion(proxy, "tls-peer", f"post-kill-{marker}")
    assert post_kill_id == f"chatcmpl-post-kill-{marker}"

    def _ready() -> bool:
        try:
            return proxy.client.get("/health/readiness").status_code == 200
        except httpx.TransportError:
            return False

    eventually(_ready, lambda ready: ready, seconds=60)
