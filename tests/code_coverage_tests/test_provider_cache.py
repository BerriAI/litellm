from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest
from e2e_http import NetworkError, PreparedForward, RawResponse, StreamChunk, StreamHead, forward, prepare_forward
from models import LiteLLMParamsBody
from provider_cache import CacheEdge, CacheHit, CaptureLease, exact_key, successful_response
from provider_cache_redis import PUBLISH, RedisCommands, RedisResponseStore, configured_cache, redis_store
from provider_cache_routing import LIVE_PROVIDER_REQUIRED, route_cache_model
from provider_edge import configured_cache_backend, start_provider_edge
from redis.exceptions import ConnectionError as RedisConnectionError

SECRET: Final = b"synthetic-cache-hmac-key-for-tests"
BODY: Final = b'{"model":"test","messages":[{"role":"user","content":"hello"}]}'
SUCCESS: Final = b'{"id":"provider-fixed-id","choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
HEADERS: Final = {"content-type": "application/json", "authorization": "Bearer synthetic-account-one"}


class Provider(ThreadingHTTPServer):
    hits: tuple[tuple[str, bytes], ...] = ()
    response: bytes = SUCCESS
    status: int = 200
    delay: float = 0
    stream: bool = False
    truncated: bool = False


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        server: Final = self.server
        assert isinstance(server, Provider)
        body: Final = self.rfile.read(int(self.headers.get("content-length", "0")))
        server.hits += ((self.path, body),)
        time.sleep(server.delay)
        self.send_response(server.status)
        if server.stream:
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(server.response), server.response))
            if server.truncated:
                self.close_connection = True
                return
            self.wfile.write(b"0\r\n\r\n")
            return
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(server.response)))
        self.end_headers()
        self.wfile.write(server.response)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def provider() -> Generator[Provider, None, None]:
    server: Final = Provider(("127.0.0.1", 0), Handler)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def redis_url(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    configured: Final = os.environ.get("E2E_CACHE_TEST_REDIS_URL")
    if configured:
        yield configured
        return
    binary: Final = shutil.which("redis-server")
    assert binary is not None, "Set E2E_CACHE_TEST_REDIS_URL or install Redis for cache integration checks"
    root: Final = tmp_path_factory.mktemp("provider-cache-redis")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: Final = probe.getsockname()[1]
    with (root / "redis.log").open("wb") as log:
        process: Final = subprocess.Popen(
            [binary, "--bind", "127.0.0.1", "--port", str(port), "--save", "", "--appendonly", "no", "--dir", str(root)],
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            deadline: Final = time.monotonic() + 5
            while True:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                        break
                except OSError:
                    assert process.poll() is None and time.monotonic() < deadline
                    time.sleep(0.02)
            yield f"redis://127.0.0.1:{port}/0"
        finally:
            process.terminate()
            process.wait(timeout=5)


@pytest.fixture
def store(redis_url: str) -> RedisResponseStore:
    return redis_store(redis_url, "test-" + uuid.uuid4().hex)


@contextmanager
def edge(cache: CacheEdge, provider: Provider) -> Generator[str, None, None]:
    upstream: Final = f"http://127.0.0.1:{provider.server_port}"
    running: Final = start_provider_edge(cache, mounts={"openai": upstream})
    try:
        yield running.edge.api_base("openai") + "/v1/chat/completions"
    finally:
        running.shutdown()


def call(url: str, body: bytes = BODY, headers: dict[str, str] = HEADERS) -> RawResponse:
    result: Final = forward("POST", url, headers=headers, body=body, timeout=5)
    assert isinstance(result, RawResponse), result
    return result


def test_success_is_reusable_across_fresh_edges(store: RedisResponseStore, provider: Provider) -> None:
    with edge(CacheEdge(store, SECRET), provider) as url:
        assert call(url).body == SUCCESS
        assert call(url).body == SUCCESS
    with edge(CacheEdge(store, SECRET), provider) as other:
        assert call(other).body == SUCCESS
    assert len(provider.hits) == 1


@pytest.mark.parametrize("body", [BODY + b" ", BODY.replace(b"hello", b"Hello"), BODY.replace(b"test", b"test2")])
def test_any_body_change_calls_live(store: RedisResponseStore, provider: Provider, body: bytes) -> None:
    with edge(CacheEdge(store, SECRET), provider) as url:
        call(url)
        call(url, body)
        call(url, body)
    assert len(provider.hits) == 2


@pytest.mark.parametrize("name,value", [("authorization", "Bearer another-account"), ("x-request-id", "one"), ("anthropic-version", "new")])
def test_changed_header_cannot_reuse(store: RedisResponseStore, provider: Provider, name: str, value: str) -> None:
    with edge(CacheEdge(store, SECRET), provider) as url:
        call(url)
        call(url, headers=HEADERS | {name: value})
        call(url + "?x=1")
    assert len(provider.hits) == 3


@pytest.mark.parametrize("status,response", [(429, b'{"error":"rate limited"}'), (500, b'failed'), (200, b'{"error":"bad"}'), (200, b'not json')])
def test_failed_provider_responses_never_enter_cache(store: RedisResponseStore, provider: Provider, status: int, response: bytes) -> None:
    provider.status = status
    provider.response = response
    with edge(CacheEdge(store, SECRET), provider) as url:
        assert call(url).status_code == status
        assert call(url).body == response
    assert len(provider.hits) == 2


def test_expiry_does_not_slide(store: RedisResponseStore, provider: Provider) -> None:
    short: Final = replace(store, lifetime_ms=250)
    with edge(CacheEdge(short, SECRET), provider) as url:
        call(url)
        call(url)
        time.sleep(0.3)
        call(url)
        call(url)
    assert len(provider.hits) == 2


def test_concurrent_requests_publish_atomically(store: RedisResponseStore, provider: Provider) -> None:
    provider.delay = 0.15
    with edge(CacheEdge(store, SECRET), provider) as url:
        with ThreadPoolExecutor(max_workers=5) as executor:
            replies: Final = tuple(executor.map(lambda _: call(url).body, range(5)))
    assert replies == (SUCCESS,) * 5
    assert len(provider.hits) == 1


@pytest.mark.parametrize("truncated", [False, True])
def test_stream_completion_controls_publication(store: RedisResponseStore, provider: Provider, truncated: bool) -> None:
    provider.stream = True
    provider.truncated = truncated
    provider.response = b'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    with edge(CacheEdge(store, SECRET), provider) as url:
        for _ in range(2):
            result: Final = forward("POST", url, headers=HEADERS, body=BODY, timeout=5)
            if truncated:
                assert isinstance(result, NetworkError)
            else:
                assert isinstance(result, RawResponse) and result.body == provider.response
    assert len(provider.hits) == (2 if truncated else 1)


def test_store_outage_preserves_provider_success(provider: Provider) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: Final = probe.getsockname()[1]
        unavailable: Final = redis_store(f"redis://127.0.0.1:{port}/0", "unavailable")
        with edge(CacheEdge(unavailable, SECRET), provider) as url:
            assert call(url).body == SUCCESS
            assert call(url).body == SUCCESS
    assert len(provider.hits) == 2


def test_old_lease_cannot_overwrite_new_owner(store: RedisResponseStore) -> None:
    short: Final = replace(store, lease_ms=50)
    old: Final = short.lookup("key")
    assert isinstance(old, CaptureLease)
    time.sleep(0.08)
    current: Final = short.lookup("key")
    assert isinstance(current, CaptureLease)
    assert not short.publish("key", old, b"old")
    assert short.publish("key", current, b"new")
    hit: Final = short.lookup("key")
    assert isinstance(hit, CacheHit) and hit.payload == b"new"


def test_identity_preserves_values_and_never_contains_credentials() -> None:
    variants: Final = (b'{}', b'{"a":null}', b'{"a":false}', b'{"a":0}', b'{"a":0.0}', b'{"a":"0"}', b' { }', None, b'')
    keys: Final = tuple(exact_key(SECRET, "POST", "https://example.invalid/v1/chat/completions", HEADERS, body) for body in variants)
    assert len(set(keys)) == len(variants)
    assert all(len(key) == 64 and "synthetic-account" not in key for key in keys)


@pytest.mark.parametrize("payload", [b"corrupt response", '{"response":"{}","signature":"é"}'.encode()])
def test_corrupt_entry_is_replaced_by_same_successful_request(store: RedisResponseStore, provider: Provider, payload: bytes) -> None:
    upstream: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"
    prepared: Final = prepare_forward("POST", upstream, HEADERS, BODY)
    assert isinstance(prepared, PreparedForward)
    key: Final = exact_key(SECRET, "POST", upstream, prepared.headers, BODY)
    lease: Final = store.lookup(key)
    assert isinstance(lease, CaptureLease)
    assert store.publish(key, lease, payload)
    cache: Final = CacheEdge(store, SECRET)
    for _ in range(2):
        head = cache.forward("POST", upstream, HEADERS, BODY, 5)
        assert isinstance(head, StreamHead)
        assert b"".join(step.data for step in head.steps if isinstance(step, StreamChunk)) == SUCCESS
    assert len(provider.hits) == 1
    assert dict(cache.counters.counts) == {
        "corrupt": 1, "misses": 1, "upstream_attempts": 1, "writes": 1, "hits": 1,
    }


@pytest.mark.parametrize("payload", [
    b'data: {}\n\ndata: [DONE]\n\n',
    b'data: {"choices":[{"index":0,"delta":{}}]}\n\ndata: [DONE]\n\n',
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]',
    b'data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n',
])
def test_malformed_success_stream_is_never_cached(store: RedisResponseStore, provider: Provider, payload: bytes) -> None:
    provider.stream = True
    provider.response = payload
    with edge(CacheEdge(store, SECRET), provider) as url:
        assert call(url).body == payload
        assert call(url).body == payload
    assert len(provider.hits) == 2


def test_anthropic_stream_requires_start_finish_and_stop() -> None:
    start: Final = b'data: {"type":"message_start","message":{}}\n\n'
    finish: Final = b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
    stop: Final = b'data: {"type":"message_stop"}\n\n'
    url: Final = "https://example.invalid/v1/messages"
    headers: Final = {"content-type": "text/event-stream"}
    assert successful_response(url, 200, headers, start + finish + stop)
    assert not successful_response(url, 200, headers, start + stop)
    assert not successful_response(url, 200, headers, finish + stop)
    assert not successful_response(url, 200, headers, start + finish)


@pytest.mark.parametrize("provider,suffix", [("openai", "/v1"), ("anthropic", "")])
def test_normal_registration_routes_supported_providers(provider: str, suffix: str) -> None:
    params: Final = LiteLLMParamsBody(model=f"{provider}/test", api_key="os.environ/SYNTHETIC_KEY", timeout=12)
    routed: Final = route_cache_model(params, lambda mount: f"http://edge.invalid/{mount}", enabled=True)
    assert routed.api_base == f"http://edge.invalid/{provider}{suffix}"
    assert routed.model_dump(exclude={"api_base"}) == params.model_dump(exclude={"api_base"})
    assert params.api_base is None


@pytest.mark.parametrize("params", [
    LiteLLMParamsBody(model="bedrock/test"),
    LiteLLMParamsBody(model="azure/test"),
    LiteLLMParamsBody(model="openai/test", api_base="https://custom.invalid/v1"),
    LiteLLMParamsBody(model="openai/test", api_base=""),
    LiteLLMParamsBody(model="openai/test", litellm_credential_name="named-credential"),
    LiteLLMParamsBody(model="openai/test", mock_response="synthetic"),
])
def test_registration_preserves_unsupported_or_explicit_routes(params: LiteLLMParamsBody) -> None:
    def unexpected_edge(mount: str) -> str:
        pytest.fail(f"should not start edge for {mount}")
    assert route_cache_model(params, unexpected_edge, enabled=True) is params


def test_rollback_and_live_only_policy_keep_direct_provider_route() -> None:
    params: Final = LiteLLMParamsBody(model="openai/test")
    assert route_cache_model(params, lambda _: "http://edge.invalid", enabled=False) is params
    assert route_cache_model(params, lambda _: "http://edge.invalid", enabled=True, mode="realtime") is params
    token: Final = LIVE_PROVIDER_REQUIRED.set(True)
    try:
        assert route_cache_model(params, lambda _: "http://edge.invalid", enabled=True) is params
    finally:
        LIVE_PROVIDER_REQUIRED.reset(token)
    assert route_cache_model(params, lambda _: "http://edge.invalid", enabled=True).api_base == "http://edge.invalid/v1"


@dataclass(frozen=True)
class PublishOutage:
    client: RedisCommands

    def eval(self, script: str, numkeys: int, *args: str | bytes | int) -> object:
        if script == PUBLISH:
            raise RedisConnectionError("synthetic publication outage")
        return self.client.eval(script, numkeys, *args)


def test_write_outage_preserves_success_without_hidden_retry(store: RedisResponseStore, provider: Provider) -> None:
    unavailable: Final = replace(store, client=PublishOutage(store.client))
    cache: Final = CacheEdge(unavailable, SECRET)
    with edge(cache, provider) as url:
        assert call(url).body == SUCCESS
        assert call(url).body == SUCCESS
    assert len(provider.hits) == 2
    assert dict(cache.counters.counts)["write_failures"] == 2
    with edge(CacheEdge(store, SECRET), provider) as url:
        assert call(url).body == SUCCESS
        assert call(url).body == SUCCESS
    assert len(provider.hits) == 3


def test_connection_failure_releases_capture_lease(store: RedisResponseStore) -> None:
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        url: Final = f"http://127.0.0.1:{unavailable.getsockname()[1]}/v1/chat/completions"
        cache: Final = CacheEdge(store, SECRET)
        assert isinstance(cache.forward("POST", url, HEADERS, BODY, 0.2), NetworkError)
        prepared: Final = prepare_forward("POST", url, HEADERS, BODY)
        assert isinstance(prepared, PreparedForward)
        key: Final = exact_key(SECRET, "POST", url, prepared.headers, BODY)
        slot: Final = store.lookup(key)
        assert isinstance(slot, CaptureLease)
        assert store.release(key, slot)
        assert dict(cache.counters.counts)["rejected"] == 1


def test_close_before_first_chunk_releases_lease(store: RedisResponseStore, provider: Provider) -> None:
    url: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"
    cache: Final = CacheEdge(store, SECRET)
    head: Final = cache.forward("POST", url, HEADERS, BODY, 5)
    assert isinstance(head, StreamHead)
    head.steps.close()
    prepared: Final = prepare_forward("POST", url, HEADERS, BODY)
    assert isinstance(prepared, PreparedForward)
    key: Final = exact_key(SECRET, "POST", url, prepared.headers, BODY)
    slot: Final = store.lookup(key)
    assert isinstance(slot, CaptureLease)
    assert store.release(key, slot)


def test_effective_account_change_cannot_reuse_cache(
    store: RedisResponseStore, provider: Provider, monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    url: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"
    cache: Final = CacheEdge(store, SECRET)
    for account in ("account-a", "account-b", "account-b"):
        netrc = tmp_path / account
        netrc.write_text(f"machine 127.0.0.1 login {account} password synthetic\n")
        monkeypatch.setenv("NETRC", str(netrc))
        head = cache.forward("POST", url, HEADERS, BODY, 5)
        assert isinstance(head, StreamHead)
        assert b"".join(step.data for step in head.steps if isinstance(step, StreamChunk)) == SUCCESS
    assert len(provider.hits) == 2
    assert dict(cache.counters.counts)["hits"] == 1


def test_enabled_environment_reuses_store_across_fresh_backends(
    redis_url: str, provider: Provider, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2E_PROVIDER_CACHE", "1")
    monkeypatch.setenv("E2E_PROVIDER_CACHE_REDIS_URL", redis_url)
    monkeypatch.setenv("E2E_PROVIDER_CACHE_HMAC_KEY", SECRET.decode())
    monkeypatch.setenv("E2E_PROVIDER_CACHE_NAMESPACE", "environment-" + uuid.uuid4().hex)
    configured_cache.cache_clear()
    try:
        for _ in range(2):
            backend = configured_cache_backend()
            assert isinstance(backend, CacheEdge)
            with edge(backend, provider) as url:
                assert call(url).body == SUCCESS
            configured_cache.cache_clear()
        assert len(provider.hits) == 1
        monkeypatch.setenv("E2E_PROVIDER_CACHE", "0")
        assert configured_cache_backend() is None
    finally:
        configured_cache.cache_clear()
