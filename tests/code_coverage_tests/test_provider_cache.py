from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from urllib.parse import urlsplit

import pytest
from e2e_http import NetworkError, PreparedForward, RawResponse, StreamChunk, StreamHead, forward, prepare_forward
from models import LiteLLMParamsBody, ModelMode
from botocore.credentials import Credentials
from provider_cache import (
    SIGNATURE_HEADERS,
    CacheEdge,
    CacheHit,
    CaptureLease,
    MountPolicy,
    ResponseStore,
    cacheable_endpoint,
    request_identity,
    slotted_key,
    successful_response,
)
from provider_cache_redis import PUBLISH, RedisCommands, RedisResponseStore, configured_cache, redis_store
from provider_cache_routing import LIVE_PROVIDER_REQUIRED, route_cache_model
from fixture_mode import SESSION_TEST_KEY
from provider_edge import EDGE_MOUNTS, configured_cache_backend, resolve_mount, start_provider_edge
from provider_edge_bedrock import bedrock_signer
from redis.exceptions import ConnectionError as RedisConnectionError

SECRET: Final = b"synthetic-cache-hmac-key-for-tests"
BODY: Final = b'{"model":"test","messages":[{"role":"user","content":"hello"}]}'
SUCCESS: Final = b'{"id":"provider-fixed-id","choices":[{"message":{"content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}'
HEADERS: Final = {"content-type": "application/json", "authorization": "Bearer synthetic-account-one"}
TEST_KEY: Final = "tests/e2e/synthetic_suite.py::TestCase::test_case"
OTHER_TEST_KEY: Final = "tests/e2e/synthetic_suite.py::TestCase::test_other_case"


def marked(marker: str) -> bytes:
    """One request body shaped like the suite's own: a fixed prompt salted with a
    12-lowercase-hex ``unique_marker()`` token, fresh on every run."""
    return b'{"model":"test","messages":[{"role":"user","content":"hello %s"}]}' % marker.encode()


MARKED: Final = marked("0a1b2c3d4e5f")
BEDROCK_MOUNT: Final = "bedrock/us-east-1"
BEDROCK_MODEL: Final = "us.anthropic.claude-haiku-4-5-20251001-v1%3A0"
BEDROCK_BODY: Final = b'{"messages":[{"role":"user","content":[{"text":"hello 0a1b2c3d4e5f"}]}]}'
CONVERSE_SUCCESS: Final = (
    b'{"output":{"message":{"role":"assistant","content":[{"text":"hi"}]}},'
    b'"stopReason":"end_turn","usage":{"inputTokens":1,"outputTokens":1,"totalTokens":2}}'
)
INVOKE_SUCCESS: Final = (
    b'{"id":"msg_synthetic","type":"message","role":"assistant",'
    b'"content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn"}'
)
STATIC_CREDENTIALS: Final = Credentials("AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")


class Provider(ThreadingHTTPServer):
    hits: tuple[tuple[str, bytes], ...] = ()
    authorizations: tuple[str, ...] = ()
    response: bytes = SUCCESS
    status: int = 200
    delay: float = 0
    stream: bool = False
    truncated: bool = False
    cookie: str = ""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        server: Final = self.server
        assert isinstance(server, Provider)
        body: Final = self.rfile.read(int(self.headers.get("content-length", "0")))
        server.hits += ((self.path, body),)
        server.authorizations += (self.headers.get("authorization", ""),)
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
        if server.cookie:
            self.send_header("set-cookie", server.cookie)
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


def cache_edge(store: ResponseStore, test_key: str = TEST_KEY) -> CacheEdge:
    """A cache edge standing in for one pytest process. A fresh instance over the
    same store is the next build running the same test: the recordings survive,
    the per-test FIFO slot counters start over."""
    return CacheEdge(store, SECRET, test_key=lambda: test_key)


def slot_key(
    url: str, slot: int = 0, body: bytes | None = BODY,
    headers: dict[str, str] = HEADERS, test_key: str = TEST_KEY,
) -> str:
    prepared: Final = prepare_forward("POST", url, headers, body)
    assert isinstance(prepared, PreparedForward)
    return slotted_key(SECRET, request_identity(SECRET, test_key, "POST", url, prepared.headers, body), slot)


def bedrock_cache_edge(store: ResponseStore, test_key: str = TEST_KEY) -> CacheEdge:
    return CacheEdge(
        store, SECRET, test_key=lambda: test_key,
        policies={BEDROCK_MOUNT: MountPolicy(
            sign=bedrock_signer("us-east-1", lambda: STATIC_CREDENTIALS), unkeyed_headers=SIGNATURE_HEADERS,
        )},
    )


@contextmanager
def edge(cache: CacheEdge, provider: Provider) -> Generator[str, None, None]:
    upstream: Final = f"http://127.0.0.1:{provider.server_port}"
    running: Final = start_provider_edge(cache, mounts={"openai": upstream})
    try:
        yield running.edge.api_base("openai") + "/v1/chat/completions"
    finally:
        running.shutdown()


@contextmanager
def bedrock_edge(cache: CacheEdge, provider: Provider, action: str = "converse") -> Generator[str, None, None]:
    upstream: Final = f"http://127.0.0.1:{provider.server_port}"
    running: Final = start_provider_edge(cache, mounts={BEDROCK_MOUNT: upstream})
    try:
        yield f"{running.edge.api_base(BEDROCK_MOUNT)}/model/{BEDROCK_MODEL}/{action}"
    finally:
        running.shutdown()


def call(url: str, body: bytes = BODY, headers: dict[str, str] = HEADERS) -> RawResponse:
    result: Final = forward("POST", url, headers=headers, body=body, timeout=5)
    assert isinstance(result, RawResponse), result
    return result


def test_repeated_call_takes_its_own_slot_and_both_replay_next_run(
    store: RedisResponseStore, provider: Provider,
) -> None:
    with edge(cache_edge(store), provider) as url:
        assert call(url).body == SUCCESS
        assert call(url).body == SUCCESS
    assert len(provider.hits) == 2
    with edge(cache_edge(store), provider) as other:
        assert call(other).body == SUCCESS
        assert call(other).body == SUCCESS
    assert len(provider.hits) == 2


@pytest.mark.parametrize("body", [BODY + b" ", BODY.replace(b"hello", b"Hello"), BODY.replace(b"test", b"test2")])
def test_any_body_change_calls_live(store: RedisResponseStore, provider: Provider, body: bytes) -> None:
    with edge(cache_edge(store), provider) as url:
        call(url)
    assert len(provider.hits) == 1
    with edge(cache_edge(store), provider) as url:
        call(url, body)
    assert len(provider.hits) == 2
    with edge(cache_edge(store), provider) as url:
        call(url, body)
    assert len(provider.hits) == 2


@pytest.mark.parametrize("name,value", [("authorization", "Bearer another-account"), ("x-request-id", "one"), ("anthropic-version", "new")])
def test_changed_header_cannot_reuse(store: RedisResponseStore, provider: Provider, name: str, value: str) -> None:
    with edge(cache_edge(store), provider) as url:
        call(url)
    assert len(provider.hits) == 1
    with edge(cache_edge(store), provider) as url:
        call(url, headers=HEADERS | {name: value})
        call(url + "?x=1")
    assert len(provider.hits) == 3


@pytest.mark.parametrize("status,response", [(429, b'{"error":"rate limited"}'), (500, b'failed'), (200, b'{"error":"bad"}'), (200, b'not json')])
def test_failed_provider_responses_never_enter_cache(store: RedisResponseStore, provider: Provider, status: int, response: bytes) -> None:
    provider.status = status
    provider.response = response
    with edge(cache_edge(store), provider) as url:
        assert call(url).status_code == status
    assert len(provider.hits) == 1
    with edge(cache_edge(store), provider) as url:
        assert call(url).body == response
    assert len(provider.hits) == 2


def test_cookie_setting_success_is_reused_without_the_cookie(store: RedisResponseStore, provider: Provider) -> None:
    provider.cookie = "__cf_bm=synthetic-bot-management; Path=/; HttpOnly; Secure"
    with edge(cache_edge(store), provider) as url:
        live: Final = call(url)
    with edge(cache_edge(store), provider) as url:
        replayed: Final = call(url)
    assert len(provider.hits) == 1
    assert all(reply.body == SUCCESS and "set-cookie" not in reply.headers for reply in (live, replayed))


def test_expiry_does_not_slide(store: RedisResponseStore, provider: Provider) -> None:
    short: Final = replace(store, lifetime_ms=250)
    url: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"

    def drain() -> None:
        head = cache_edge(short).forward("openai", "POST", url, dict(HEADERS), BODY, 5)
        assert isinstance(head, StreamHead)
        assert b"".join(step.data for step in head.steps if isinstance(step, StreamChunk)) == SUCCESS

    drain()
    assert len(provider.hits) == 1
    drain()
    assert len(provider.hits) == 1
    time.sleep(0.3)
    drain()
    assert len(provider.hits) == 2


def test_concurrent_builds_publish_one_recording_atomically(
    store: RedisResponseStore, provider: Provider,
) -> None:
    """Five processes running the same test at the same time all reach slot 0 of
    one key, which is the only way the capture lease is contended now that a
    repeat inside a single test takes its own slot."""
    provider.delay = 0.15
    url: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"
    edges: Final = tuple(cache_edge(store) for _ in range(5))

    def drain(cache: CacheEdge) -> bytes:
        head = cache.forward("openai", "POST", url, dict(HEADERS), BODY, 5)
        assert isinstance(head, StreamHead)
        return b"".join(step.data for step in head.steps if isinstance(step, StreamChunk))

    with ThreadPoolExecutor(max_workers=5) as executor:
        replies: Final = tuple(executor.map(drain, edges))
    assert replies == (SUCCESS,) * 5
    assert len(provider.hits) == 1


@pytest.mark.parametrize("age_past_expiry_ms", [0, 1])
def test_expired_response_is_rejected_without_physical_eviction(
    store: RedisResponseStore, age_past_expiry_ms: int,
) -> None:
    response_key: Final = store.keys("expired")[0]
    retained: Final = store.client.eval(
        """
local clock = redis.call('TIME')
local expires = clock[1] * 1000 + math.floor(clock[2] / 1000) - tonumber(ARGV[1])
redis.call('HSET', KEYS[1], 'captured', expires - 86400000, 'expires', expires, 'payload', 'old-response')
return redis.call('PTTL', KEYS[1])
""",
        1, response_key, age_past_expiry_ms,
    )
    assert retained == -1
    replacement: Final = store.lookup("expired")
    assert isinstance(replacement, CaptureLease)
    assert replacement.expires_at_ms - replacement.captured_at_ms == 86_400_000
    assert store.publish("expired", replacement, b"fresh-response")
    hit: Final = store.lookup("expired")
    assert isinstance(hit, CacheHit) and hit.payload == b"fresh-response"


@pytest.mark.parametrize("truncated", [False, True])
def test_stream_completion_controls_publication(store: RedisResponseStore, provider: Provider, truncated: bool) -> None:
    provider.stream = True
    provider.truncated = truncated
    provider.response = b'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    for _ in range(2):
        with edge(cache_edge(store), provider) as url:
            result = forward("POST", url, headers=HEADERS, body=BODY, timeout=5)
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
        for _ in range(2):
            with edge(cache_edge(unavailable), provider) as url:
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
    url: Final = "https://example.invalid/v1/chat/completions"
    keys: Final = tuple(request_identity(SECRET, TEST_KEY, "POST", url, HEADERS, body) for body in variants)
    assert len(set(keys)) == len(variants)
    assert all(len(key) == 64 and "synthetic-account" not in key for key in keys)


@pytest.mark.parametrize("payload", [b"corrupt response", '{"response":"{}","signature":"é"}'.encode()])
def test_corrupt_entry_is_replaced_by_same_successful_request(store: RedisResponseStore, provider: Provider, payload: bytes) -> None:
    upstream: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"
    key: Final = slot_key(upstream)
    lease: Final = store.lookup(key)
    assert isinstance(lease, CaptureLease)
    assert store.publish(key, lease, payload)
    caches: Final = tuple(cache_edge(store) for _ in range(2))
    for cache in caches:
        head = cache.forward("openai", "POST", upstream, dict(HEADERS), BODY, 5)
        assert isinstance(head, StreamHead)
        assert b"".join(step.data for step in head.steps if isinstance(step, StreamChunk)) == SUCCESS
    assert len(provider.hits) == 1
    assert dict(caches[0].counters.counts) == {
        "corrupt": 1, "mount:openai:corrupt": 1, "misses": 1, "mount:openai:misses": 1,
        "upstream_attempts": 1, "mount:openai:upstream_attempts": 1,
        "writes": 1, "mount:openai:writes": 1,
    }
    assert dict(caches[1].counters.counts) == {"hits": 1, "mount:openai:hits": 1}


@pytest.mark.parametrize("payload", [
    b'data: {}\n\ndata: [DONE]\n\n',
    b'data: {"choices":[{"index":0,"delta":{}}]}\n\ndata: [DONE]\n\n',
    b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]',
    b'data: {"error":{"message":"failed"}}\n\ndata: [DONE]\n\n',
])
def test_malformed_success_stream_is_never_cached(store: RedisResponseStore, provider: Provider, payload: bytes) -> None:
    provider.stream = True
    provider.response = payload
    for _ in range(2):
        with edge(cache_edge(store), provider) as url:
            assert call(url).body == payload
    assert len(provider.hits) == 2


def test_requests_differing_only_by_marker_share_one_recording_per_slot(
    store: RedisResponseStore, provider: Provider,
) -> None:
    """The whole point of the canonical key. Every e2e test salts its prompt with
    a fresh ``unique_marker()``, so before this the same test could never reuse
    anything across builds. The second run mints markers it has never sent, which
    is what a later build actually does, and must still serve both from the two
    slots the first run recorded."""
    with edge(cache_edge(store), provider) as url:
        assert call(url, MARKED).body == SUCCESS
        assert call(url, marked("f5e4d3c2b1a0")).body == SUCCESS
    assert len(provider.hits) == 2
    with edge(cache_edge(store), provider) as url:
        assert call(url, marked("7c6b5a493827")).body == SUCCESS
        assert call(url, marked("1122334455ff")).body == SUCCESS
    assert len(provider.hits) == 2


@pytest.mark.parametrize("body", [
    b'{"model":"test","messages":[{"role":"user","content":"hello 0a1b2c3d4e5"}]}',
    b'{"model":"test","messages":[{"role":"user","content":"hello 0a1b2c3d4e5f0"}]}',
    b'{"model":"test","messages":[{"role":"user","content":"hello 0A1B2C3D4E5F"}]}',
    b'{"model":"0a1b2c3d4e5f","messages":[{"role":"user","content":"hello"}]}',
])
def test_a_token_that_is_not_a_marker_keeps_its_own_key(
    store: RedisResponseStore, provider: Provider, body: bytes,
) -> None:
    """Too short, too long, upper case, or in another field: none of these is the
    12-lowercase-hex token ``unique_marker`` mints, so none may fold onto it."""
    with edge(cache_edge(store), provider) as url:
        call(url, MARKED)
    assert len(provider.hits) == 1
    with edge(cache_edge(store), provider) as url:
        call(url, body)
    assert len(provider.hits) == 2


def test_another_test_never_reuses_this_tests_recording(
    store: RedisResponseStore, provider: Provider,
) -> None:
    with edge(cache_edge(store), provider) as url:
        call(url)
    assert len(provider.hits) == 1
    with edge(cache_edge(store, OTHER_TEST_KEY), provider) as url:
        call(url)
    assert len(provider.hits) == 2
    with edge(cache_edge(store, OTHER_TEST_KEY), provider) as url:
        call(url)
    assert len(provider.hits) == 2


def test_calls_outside_any_test_are_never_cached(
    store: RedisResponseStore, provider: Provider,
) -> None:
    url: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"
    cache: Final = CacheEdge(store, SECRET, test_key=lambda: SESSION_TEST_KEY)
    for _ in range(2):
        head = cache.forward("openai", "POST", url, dict(HEADERS), BODY, 5)
        assert isinstance(head, StreamHead)
        assert b"".join(step.data for step in head.steps if isinstance(step, StreamChunk)) == SUCCESS
    assert len(provider.hits) == 2
    assert dict(cache.counters.counts) == {
        "bypass": 2, "mount:openai:bypass": 2,
        "upstream_attempts": 2, "mount:openai:upstream_attempts": 2,
    }


def test_counters_attribute_every_outcome_to_its_mount(
    store: RedisResponseStore, provider: Provider,
) -> None:
    """The build report needs per-provider hit counts, and the flat totals cannot
    supply them. Anthropic is served a chat-shaped body here, which its validator
    rejects, so one mount writes and the other does not."""
    upstream: Final = f"http://127.0.0.1:{provider.server_port}"
    cache: Final = cache_edge(store)
    running: Final = start_provider_edge(cache, mounts={"openai": upstream, "anthropic": upstream})
    try:
        call(running.edge.api_base("openai") + "/v1/chat/completions")
        call(running.edge.api_base("anthropic") + "/v1/messages")
    finally:
        running.shutdown()
    counts: Final = dict(cache.counters.counts)
    assert counts["misses"] == 2
    assert counts["mount:openai:misses"] == 1 and counts["mount:anthropic:misses"] == 1
    assert counts["mount:openai:writes"] == 1 and "mount:anthropic:writes" not in counts
    assert counts["mount:anthropic:rejected"] == 1 and "mount:openai:rejected" not in counts


EMBEDDING_SUCCESS: Final = (
    b'{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2]}],'
    b'"model":"text-embedding-3-small","usage":{"prompt_tokens":2,"total_tokens":2}}'
)
RESPONSE_SUCCESS: Final = b'{"id":"resp_synthetic","object":"response","status":"completed","output":[]}'
RESPONSE_STREAM_SUCCESS: Final = (
    b'data: {"type":"response.created","response":{"id":"resp_synthetic"}}\n\n'
    b'data: {"type":"response.completed","response":{"id":"resp_synthetic","status":"completed"}}\n\n'
)


@contextmanager
def openai_edge(cache: CacheEdge, provider: Provider, path: str) -> Generator[str, None, None]:
    upstream: Final = f"http://127.0.0.1:{provider.server_port}"
    running: Final = start_provider_edge(cache, mounts={"openai": upstream})
    try:
        yield running.edge.api_base("openai") + path
    finally:
        running.shutdown()


class TestNonChatOpenAiEndpoints:
    """Chat and messages were the only cacheable paths. Embeddings and responses
    are the other two JSON endpoints the suite drives through the same mount, and
    each needs its own completeness rule: a chat response's ``choices`` check
    would reject a perfectly good embedding."""

    @pytest.mark.parametrize("path,response", [
        ("/v1/embeddings", EMBEDDING_SUCCESS),
        ("/v1/responses", RESPONSE_SUCCESS),
    ])
    def test_complete_responses_replay_on_the_next_run(
        self, store: RedisResponseStore, provider: Provider, path: str, response: bytes,
    ) -> None:
        provider.response = response
        for _ in range(2):
            with openai_edge(cache_edge(store), provider, path) as url:
                assert call(url, MARKED).body == response
        assert len(provider.hits) == 1

    def test_a_completed_response_stream_replays(
        self, store: RedisResponseStore, provider: Provider,
    ) -> None:
        provider.stream = True
        provider.response = RESPONSE_STREAM_SUCCESS
        for _ in range(2):
            with openai_edge(cache_edge(store), provider, "/v1/responses") as url:
                assert call(url, MARKED).body == RESPONSE_STREAM_SUCCESS
        assert len(provider.hits) == 1

    @pytest.mark.parametrize("path,response", [
        ("/v1/embeddings", b'{"object":"list","data":[],"usage":{"prompt_tokens":0}}'),
        ("/v1/embeddings", b'{"object":"list","data":[{"object":"embedding","index":0,"embedding":[]}],"usage":{}}'),
        ("/v1/embeddings", b'{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1]}]}'),
        ("/v1/responses", b'{"id":"resp_x","object":"response","status":"incomplete","output":[]}'),
        ("/v1/responses", b'{"id":"resp_x","object":"response","status":"in_progress","output":[]}'),
        ("/v1/responses", b'{"id":"resp_x","object":"response","output":[]}'),
    ])
    def test_incomplete_bodies_never_enter_the_cache(
        self, store: RedisResponseStore, provider: Provider, path: str, response: bytes,
    ) -> None:
        provider.response = response
        for _ in range(2):
            with openai_edge(cache_edge(store), provider, path) as url:
                assert call(url, MARKED).body == response
        assert len(provider.hits) == 2

    @pytest.mark.parametrize("payload", [
        b'data: {"type":"response.created","response":{"id":"resp_x"}}\n\n',
        b'data: {"type":"response.created","response":{"id":"resp_x"}}\n\ndata: {"type":"response.failed"}\n\n',
        b'data: {"type":"response.completed","response":{"id":"resp_x"}}\n\ndata: {"type":"response.created"}\n\n',
    ])
    def test_a_response_stream_that_never_completed_is_never_cached(
        self, store: RedisResponseStore, provider: Provider, payload: bytes,
    ) -> None:
        provider.stream = True
        provider.response = payload
        for _ in range(2):
            with openai_edge(cache_edge(store), provider, "/v1/responses") as url:
                assert call(url, MARKED).body == payload
        assert len(provider.hits) == 2

    @pytest.mark.parametrize("path,cacheable", [
        ("/v1/chat/completions", True), ("/v1/messages", True),
        ("/v1/embeddings", True), ("/v1/responses", True),
        ("/v1/audio/speech", False), ("/v1/images/generations", False),
        ("/v1/files", False), ("/v1/batches", False),
    ])
    def test_only_the_json_endpoints_are_cacheable(self, path: str, cacheable: bool) -> None:
        assert cacheable_endpoint("openai", "POST", f"https://api.openai.com{path}", MARKED) is cacheable


class TestBedrockSigning:
    """Bedrock is the reason the edge could not mount it before: SigV4 covers the
    Host header, so forwarding through a rewritten api_base invalidates the
    proxy's signature. The edge mints its own over the upstream URL instead."""

    def test_the_proxys_signature_is_replaced_not_forwarded(self) -> None:
        signer: Final = bedrock_signer("us-east-1", lambda: STATIC_CREDENTIALS)
        signed: Final = signer(
            "POST",
            f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{BEDROCK_MODEL}/converse",
            {"content-type": "application/json", "Authorization": "AWS4-HMAC-SHA256 Credential=PROXY/...",
             "X-Amz-Date": "19700101T000000Z", "X-Amz-Security-Token": "proxy-session-token"},
            BEDROCK_BODY,
        )
        assert "PROXY" not in str(signed) and "proxy-session-token" not in str(signed)
        assert signed["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/")
        assert "/us-east-1/bedrock/aws4_request" in signed["Authorization"]
        assert signed["X-Amz-Date"] != "19700101T000000Z"
        assert signed["content-type"] == "application/json"

    def test_the_signed_url_reaches_the_wire_byte_for_byte(self) -> None:
        """SigV4 hashes the canonical URI, so if the HTTP layer re-encoded the
        colon in an inference-profile id after signing, every call would fail
        with a signature mismatch rather than anything that names the cause."""
        url: Final = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{BEDROCK_MODEL}/converse"
        signer: Final = bedrock_signer("us-east-1", lambda: STATIC_CREDENTIALS)
        prepared: Final = prepare_forward("POST", url, signer("POST", url, dict(HEADERS), BEDROCK_BODY), BEDROCK_BODY)
        assert isinstance(prepared, PreparedForward)
        assert urlsplit(prepared.url).path == urlsplit(url).path

    def test_signature_headers_are_excluded_from_the_key(
        self, store: RedisResponseStore, provider: Provider,
    ) -> None:
        """A real signature is fresh on every call, so keying on it would make
        every Bedrock request a permanent miss. The stub signer here varies its
        stamp per call on purpose: the real one only varies once a second, which
        would let this pass by luck when it should fail."""
        provider.response = CONVERSE_SUCCESS
        stamps: Final = iter(("20260101T000000Z", "20260102T111111Z"))

        def varying(method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> dict[str, str]:
            return dict(headers) | {"authorization": f"AWS4-HMAC-SHA256 {url}", "x-amz-date": next(stamps)}

        def signing_edge() -> CacheEdge:
            return CacheEdge(
                store, SECRET, test_key=lambda: TEST_KEY,
                policies={BEDROCK_MOUNT: MountPolicy(sign=varying, unkeyed_headers=SIGNATURE_HEADERS)},
            )

        for _ in range(2):
            with bedrock_edge(signing_edge(), provider) as url:
                assert call(url, BEDROCK_BODY).body == CONVERSE_SUCCESS
        assert len(provider.hits) == 1
        assert provider.authorizations[0] == (
            f"AWS4-HMAC-SHA256 http://127.0.0.1:{provider.server_port}/model/{BEDROCK_MODEL}/converse"
        ), "the signature must cover the upstream URL the edge calls, not the edge URL the proxy called"

    def test_a_mount_without_a_signer_still_keys_on_its_credentials(
        self, store: RedisResponseStore, provider: Provider,
    ) -> None:
        """The exclusion is per mount. Dropping authorization globally would let
        one OpenAI account read another's recording."""
        cache: Final = bedrock_cache_edge(store)
        assert "authorization" in SIGNATURE_HEADERS
        assert "authorization" in cache.keyed("openai", HEADERS)
        assert "authorization" not in cache.keyed(BEDROCK_MOUNT, HEADERS)
        with edge(cache, provider) as url:
            call(url)
        with edge(bedrock_cache_edge(store), provider) as url:
            call(url, headers=HEADERS | {"authorization": "Bearer synthetic-account-two"})
        assert len(provider.hits) == 2

    @pytest.mark.parametrize("action,response", [("converse", CONVERSE_SUCCESS), ("invoke", INVOKE_SUCCESS)])
    def test_complete_responses_replay_on_the_next_run(
        self, store: RedisResponseStore, provider: Provider, action: str, response: bytes,
    ) -> None:
        provider.response = response
        for _ in range(2):
            with bedrock_edge(bedrock_cache_edge(store), provider, action) as url:
                assert call(url, BEDROCK_BODY).body == response
        assert len(provider.hits) == 1

    @pytest.mark.parametrize("action,response", [
        ("converse", b'{"output":{"message":{}}}'),
        ("converse", b'{"stopReason":"end_turn"}'),
        ("converse", b'{"message":"The provided model identifier is invalid."}'),
        ("converse", CONVERSE_SUCCESS[:-20]),
        ("invoke", b'{"id":"msg_x","type":"message","content":[{"type":"text","text":"hi"}]}'),
        ("invoke", b'{"id":"msg_x","type":"message","stop_reason":"end_turn"}'),
        ("invoke", b'{"message":"Too many requests, please wait before trying again."}'),
    ])
    def test_incomplete_or_error_bodies_never_enter_the_cache(
        self, store: RedisResponseStore, provider: Provider, action: str, response: bytes,
    ) -> None:
        provider.response = response
        for _ in range(2):
            with bedrock_edge(bedrock_cache_edge(store), provider, action) as url:
                assert call(url, BEDROCK_BODY).body == response
        assert len(provider.hits) == 2

    @pytest.mark.parametrize("action", ["converse-stream", "invoke-with-response-stream"])
    def test_streaming_endpoints_go_live_every_time(
        self, store: RedisResponseStore, provider: Provider, action: str,
    ) -> None:
        """An eventstream's completeness cannot be proven without parsing its
        frames, so these bypass rather than risk recording a truncated answer.
        They are still signed: a bypass is a forward, not a passthrough."""
        provider.response = CONVERSE_SUCCESS
        cache: Final = bedrock_cache_edge(store)
        for _ in range(2):
            with bedrock_edge(cache, provider, action) as url:
                assert call(url, BEDROCK_BODY).body == CONVERSE_SUCCESS
        assert len(provider.hits) == 2
        assert dict(cache.counters.counts)[f"mount:{BEDROCK_MOUNT}:bypass"] == 2
        assert all(
            sent.startswith("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/")
            for sent in provider.authorizations
        ), provider.authorizations

    @pytest.mark.parametrize("action,cacheable", [
        ("converse", True), ("invoke", True),
        ("converse-stream", False), ("invoke-with-response-stream", False),
    ])
    def test_only_the_unary_bedrock_actions_are_cacheable(self, action: str, cacheable: bool) -> None:
        url: Final = f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{BEDROCK_MODEL}/{action}"
        assert cacheable_endpoint(BEDROCK_MOUNT, "POST", url, BEDROCK_BODY) is cacheable

    def test_a_region_mount_resolves_whole(self) -> None:
        resolved: Final = resolve_mount(f"/{BEDROCK_MOUNT}/model/{BEDROCK_MODEL}/converse", EDGE_MOUNTS)
        assert resolved is not None
        assert resolved.mount == BEDROCK_MOUNT
        assert resolved.upstream_base == "https://bedrock-runtime.us-east-1.amazonaws.com"
        assert resolved.upstream_path == f"model/{BEDROCK_MODEL}/converse"


def test_anthropic_stream_requires_start_finish_and_stop() -> None:
    start: Final = b'data: {"type":"message_start","message":{}}\n\n'
    finish: Final = b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n'
    stop: Final = b'data: {"type":"message_stop"}\n\n'
    url: Final = "https://example.invalid/v1/messages"
    headers: Final = {"content-type": "text/event-stream"}
    assert successful_response("anthropic", url, 200, headers, start + finish + stop)
    assert not successful_response("anthropic", url, 200, headers, start + stop)
    assert not successful_response("anthropic", url, 200, headers, finish + stop)
    assert not successful_response("anthropic", url, 200, headers, start + finish)


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


@pytest.mark.parametrize("model", [
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "bedrock/converse/us.anthropic.claude-sonnet-5",
    "bedrock/invoke/us.anthropic.claude-haiku-4-5-20251001-v1:0",
])
def test_anthropic_on_bedrock_registers_the_edge_as_its_runtime_endpoint(model: str) -> None:
    params: Final = LiteLLMParamsBody(model=model)
    routed: Final = route_cache_model(params, lambda mount: f"http://edge.invalid/{mount}", enabled=True)
    assert routed.aws_bedrock_runtime_endpoint == "http://edge.invalid/bedrock/us-east-1"
    assert routed.api_base is None
    assert routed.model_dump(exclude={"aws_bedrock_runtime_endpoint"}) == params.model_dump(
        exclude={"aws_bedrock_runtime_endpoint"}
    )


@pytest.mark.parametrize("params", [
    LiteLLMParamsBody(model="bedrock/amazon.titan-embed-text-v2:0"),
    LiteLLMParamsBody(model="bedrock/amazon.nova-canvas-v1:0"),
    LiteLLMParamsBody(model="bedrock/amazon.nova-sonic-v1:0"),
    LiteLLMParamsBody(model="bedrock/arn:aws:bedrock:us-east-1::foundation-model/cohere.rerank-v3-5:0"),
    LiteLLMParamsBody(model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", aws_role_name="arn:aws:iam::1:role/x"),
    LiteLLMParamsBody(model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", aws_access_key_id="AKIA"),
    LiteLLMParamsBody(model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", api_base="https://custom.invalid"),
    LiteLLMParamsBody(
        model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        aws_bedrock_runtime_endpoint="https://custom.invalid",
    ),
    LiteLLMParamsBody(model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0", aws_region_name="eu-west-1"),
])
def test_bedrock_deployments_the_edge_must_not_touch_keep_their_direct_route(params: LiteLLMParamsBody) -> None:
    """Non-Anthropic models the runner role cannot invoke, deployments carrying
    their own AWS identity (routing those would replace the assume-role chain the
    batch suite exists to prove), explicit endpoints, and unmounted regions."""
    routed: Final = route_cache_model(
        params, lambda mount: None if mount not in EDGE_MOUNTS else f"http://edge.invalid/{mount}", enabled=True,
    )
    assert routed is params or routed.aws_bedrock_runtime_endpoint == params.aws_bedrock_runtime_endpoint


@pytest.mark.parametrize("mode", ["batch", "realtime", "image_generation"])
def test_a_bedrock_deployment_with_a_mode_keeps_its_direct_route(mode: ModelMode) -> None:
    params: Final = LiteLLMParamsBody(model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
    assert route_cache_model(
        params, lambda mount: f"http://edge.invalid/{mount}", enabled=True, mode=mode,
    ) is params


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
    cache: Final = cache_edge(unavailable)
    with edge(cache, provider) as url:
        assert call(url).body == SUCCESS
        assert call(url).body == SUCCESS
    assert len(provider.hits) == 2
    assert dict(cache.counters.counts)["write_failures"] == 2
    with edge(cache_edge(store), provider) as url:
        assert call(url).body == SUCCESS
    assert len(provider.hits) == 3
    with edge(cache_edge(store), provider) as url:
        assert call(url).body == SUCCESS
    assert len(provider.hits) == 3


def test_connection_failure_releases_capture_lease(store: RedisResponseStore) -> None:
    with socket.socket() as unavailable:
        unavailable.bind(("127.0.0.1", 0))
        url: Final = f"http://127.0.0.1:{unavailable.getsockname()[1]}/v1/chat/completions"
        cache: Final = cache_edge(store)
        assert isinstance(cache.forward("openai", "POST", url, dict(HEADERS), BODY, 0.2), NetworkError)
        key: Final = slot_key(url)
        lease: Final = store.lookup(key)
        assert isinstance(lease, CaptureLease)
        assert store.release(key, lease)
        assert dict(cache.counters.counts)["rejected"] == 1


def test_close_before_first_chunk_releases_lease(store: RedisResponseStore, provider: Provider) -> None:
    url: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"
    cache: Final = cache_edge(store)
    head: Final = cache.forward("openai", "POST", url, dict(HEADERS), BODY, 5)
    assert isinstance(head, StreamHead)
    head.steps.close()
    key: Final = slot_key(url)
    lease: Final = store.lookup(key)
    assert isinstance(lease, CaptureLease)
    assert store.release(key, lease)


def test_effective_account_change_cannot_reuse_cache(
    store: RedisResponseStore, provider: Provider, monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    url: Final = f"http://127.0.0.1:{provider.server_port}/v1/chat/completions"
    caches: Final = tuple(cache_edge(store) for _ in range(3))
    for account, cache in zip(("account-a", "account-b", "account-b"), caches, strict=True):
        netrc = tmp_path / account
        netrc.write_text(f"machine 127.0.0.1 login {account} password synthetic\n")
        monkeypatch.setenv("NETRC", str(netrc))
        head = cache.forward("openai", "POST", url, dict(HEADERS), BODY, 5)
        assert isinstance(head, StreamHead)
        assert b"".join(step.data for step in head.steps if isinstance(step, StreamChunk)) == SUCCESS
    assert len(provider.hits) == 2
    assert dict(caches[2].counters.counts)["hits"] == 1


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


@pytest.mark.parametrize("known_mount", (True, False))
def test_duplicate_headers_bypass_cache_and_count_live_calls(
    store: RedisResponseStore, provider: Provider, known_mount: bool,
) -> None:
    cache: Final = cache_edge(store)
    with edge(cache, provider) as url:
        parsed: Final = urlsplit(url)
        for _ in range(2):
            connection = HTTPConnection(str(parsed.hostname), parsed.port, timeout=5)
            try:
                connection.putrequest("POST", parsed.path if known_mount else "/unknown/v1/chat/completions")
                connection.putheader("content-length", str(len(BODY)))
                connection.putheader("content-type", "application/json")
                connection.putheader("x-duplicate", "first")
                connection.putheader("x-duplicate", "second")
                connection.endheaders(BODY)
                response = connection.getresponse()
                assert response.status == (200 if known_mount else 404)
                payload = response.read()
                assert payload == SUCCESS if known_mount else b"unknown provider mount" in payload
            finally:
                connection.close()
    assert len(provider.hits) == (2 if known_mount else 0)
    assert dict(cache.counters.counts)["duplicate_header_bypass"] == 2
    assert dict(cache.counters.counts).get("upstream_attempts", 0) == (2 if known_mount else 0)
