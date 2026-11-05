import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.utils import CredentialItem
from litellm.proxy.credential_endpoints import user_provider_credentials as upc
from litellm.proxy.credential_endpoints.github_copilot_token_refresh import (
    GithubCopilotTokenRefreshSingleFlight,
    arefresh_expiring_connections,
)
from litellm.proxy.credential_endpoints.user_provider_credentials import GithubCopilotUserConnectionPayload
from litellm.llms.github_copilot.per_user_auth import aattach_github_copilot_user_session

_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
_API_KEY_URL = "https://api.github.com/copilot_internal/v2/token"


class _Store:
    def __init__(self, payload: GithubCopilotUserConnectionPayload):
        self.ciphertext: str | None = upc._encode(payload)
        self.exists = True

    async def find_unique(self, *, where: object) -> SimpleNamespace | None:
        return SimpleNamespace(credential_b64=self.ciphertext) if self.exists and self.ciphertext is not None else None

    async def update_many(self, *, where: dict[str, object], data: dict[str, object]) -> int:
        if (
            not self.exists
            or self.ciphertext is None
            or self.ciphertext != where["credential_b64"]
            or where["user_id"] != "user-a"
            or where["credential_name"] != "copilot-cred"
        ):
            return 0
        self.ciphertext = str(data["credential_b64"])
        return 1


class _RacingStore(_Store):
    def __init__(self, payload: GithubCopilotUserConnectionPayload):
        super().__init__(payload)
        self._first_reads = 0
        self._both_read = asyncio.Event()

    async def find_unique(self, *, where: object) -> SimpleNamespace | None:
        if self._first_reads < 2:
            ciphertext: Final = self.ciphertext
            self._first_reads += 1
            if self._first_reads == 2:
                self._both_read.set()
            await self._both_read.wait()
            return SimpleNamespace(credential_b64=ciphertext)
        return await super().find_unique(where=where)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def monotonic(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.value += seconds
        await asyncio.sleep(0)


class _HeldLockClient:
    async def set(self, name: str, value: str, *, nx: bool = False, px: int | None = None) -> object | None:
        return None

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object:
        return 1

    async def exists(self, *names: str) -> int:
        return 0


class _RedisCache:
    def __init__(self) -> None:
        self.lock_client = _HeldLockClient()

    def init_async_client(self) -> _HeldLockClient:
        return self.lock_client

    def check_and_fix_namespace(self, key: str) -> str:
        return key


def _cache_with_fake_redis(monkeypatch: pytest.MonkeyPatch) -> tuple[DualCache, RedisCache]:
    import fakeredis

    redis_client: Final = fakeredis.FakeAsyncRedis()
    redis_cache: Final = RedisCache(host="localhost", port=6379)
    monkeypatch.setattr(redis_cache, "init_async_client", lambda: redis_client)
    return DualCache(redis_cache=redis_cache), redis_cache


def _credential() -> CredentialItem:
    return CredentialItem(
        credential_name="copilot-cred",
        credential_values={"github_copilot_auth_type": "per_user_oauth"},
        credential_info={},
    )


def _payload(
    access_token: str = "gho_old",
    refresh_token: str | None = "ghr_old",
    access_expires_at: float | None = 1.0,
    refresh_expires_at: float | None = None,
) -> GithubCopilotUserConnectionPayload:
    return GithubCopilotUserConnectionPayload(
        access_token=access_token,
        github_login="octo",
        refresh_token=refresh_token,
        access_token_expires_at=access_expires_at,
        refresh_token_expires_at=refresh_expires_at,
    )


def _prisma(store: _Store) -> tuple[MagicMock, MagicMock]:
    table: Final = MagicMock()
    table.find_unique = AsyncMock(side_effect=store.find_unique)
    table.update_many = AsyncMock(side_effect=store.update_many)
    table.upsert = AsyncMock()
    prisma_client: Final = MagicMock()
    prisma_client.db.litellm_userprovidercredentials = table
    prisma_client.writer_db = prisma_client.db
    return prisma_client, table


@contextmanager
def _github_http(
    respond: Callable[[httpx.Request], httpx.Response],
) -> Iterator[list[httpx.Request]]:
    requests: Final[list[httpx.Request]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return respond(request)

    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(capture))
    with (
        patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=client),
        patch("litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client", return_value=client),
    ):
        yield requests


@pytest.fixture(autouse=True)
def _environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from litellm.llms.github_copilot.per_user_auth import _SESSION_CACHE

    _SESSION_CACHE.flush_cache()
    monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-test-master")
    monkeypatch.setattr(litellm, "credential_list", [_credential()])
    yield
    _SESSION_CACHE.flush_cache()


@pytest.mark.asyncio
async def test_proactive_refresh_persists_rotated_pair_and_rewrites_redis_cache(monkeypatch):
    store: Final = _Store(_payload())
    prisma_client, _ = _prisma(store)
    import fakeredis

    from litellm.caching.redis_cache import RedisCache

    redis_client: Final = fakeredis.FakeAsyncRedis()
    redis_cache: Final = RedisCache(host="localhost", port=6379)
    monkeypatch.setattr(redis_cache, "init_async_client", lambda: redis_client)
    cache: Final = DualCache(redis_cache=redis_cache)

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == _ACCESS_TOKEN_URL
        return httpx.Response(
            200,
            json={
                "access_token": "gho_rotated",
                "refresh_token": "ghr_rotated",
                "expires_in": 3600,
                "refresh_token_expires_in": 7200,
            },
            request=request,
        )

    with _github_http(respond) as requests:
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            cache,
            "user-a",
            {"copilot-cred": _payload()},
        )
    persisted: Final = await upc.get_user_provider_credential(prisma_client, "user-a", "copilot-cred")
    assert result["copilot-cred"].access_token == "gho_rotated"
    assert result["copilot-cred"].refresh_token == "ghr_rotated"
    assert persisted == result["copilot-cred"]
    assert persisted.access_token_expires_at is not None
    assert persisted.refresh_token_expires_at is not None
    cached_ciphertext: Final = await redis_cache.async_get_cache(upc._cache_key("user-a", "copilot-cred"))
    assert upc.decode_user_provider_credential(cached_ciphertext) == persisted
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_same_worker_concurrent_refreshes_share_one_refresh_request():
    store: Final = _Store(_payload())
    prisma_client, _ = _prisma(store)
    single_flight: Final = GithubCopilotTokenRefreshSingleFlight()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "gho_rotated", "refresh_token": "ghr_rotated"},
            request=request,
        )

    with _github_http(respond) as requests:
        results: Final = await asyncio.gather(
            arefresh_expiring_connections(
                prisma_client,
                DualCache(),
                "user-a",
                {"copilot-cred": _payload()},
                single_flight=single_flight,
            ),
            arefresh_expiring_connections(
                prisma_client,
                DualCache(),
                "user-a",
                {"copilot-cred": _payload()},
                single_flight=single_flight,
            ),
        )
    assert len(requests) == 1
    assert [result["copilot-cred"].access_token for result in results] == ["gho_rotated", "gho_rotated"]


@pytest.mark.asyncio
async def test_single_flight_reuses_inflight_refresh_task():
    payload: Final = _payload()
    single_flight: Final = GithubCopilotTokenRefreshSingleFlight()
    refresh_started: Final = asyncio.Event()
    finish_refresh: Final = asyncio.Event()

    async def wait_for_refresh() -> GithubCopilotUserConnectionPayload:
        refresh_started.set()
        await finish_refresh.wait()
        return payload

    refresh_call: Final = AsyncMock(side_effect=wait_for_refresh)

    async def refresh() -> GithubCopilotUserConnectionPayload | None:
        return await refresh_call()

    first: Final = asyncio.create_task(single_flight.run(("user-a", "copilot-cred"), refresh))
    await refresh_started.wait()
    second: Final = asyncio.create_task(single_flight.run(("user-a", "copilot-cred"), refresh))
    await asyncio.sleep(0)
    assert refresh_call.await_count == 1
    finish_refresh.set()
    results: Final = await asyncio.gather(first, second)
    assert results[0] is payload
    assert results[1] is payload
    assert refresh_call.await_count == 1


@pytest.mark.asyncio
async def test_completed_single_flight_runs_second_refresh_with_rotated_token(monkeypatch):
    from litellm.proxy.credential_endpoints import github_copilot_token_refresh as refresh_module

    monkeypatch.setattr(refresh_module.time, "time", lambda: 1000.0)
    store: Final = _Store(_payload())
    prisma_client, _ = _prisma(store)
    single_flight: Final = GithubCopilotTokenRefreshSingleFlight()
    refresh_tokens: Final[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        refresh_tokens.append(body["refresh_token"])
        token_number: Final = len(refresh_tokens)
        return httpx.Response(
            200,
            json={
                "access_token": f"gho_{token_number}",
                "refresh_token": f"ghr_{token_number}",
                "expires_in": 100,
                "refresh_token_expires_in": 3600,
            },
            request=request,
        )

    with _github_http(respond) as requests:
        first: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": _payload()},
            single_flight=single_flight,
        )
        second: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            first,
            single_flight=single_flight,
        )
    assert len(requests) == 2
    assert refresh_tokens == ["ghr_old", "ghr_1"]
    assert second["copilot-cred"].access_token == "gho_2"
    assert second["copilot-cred"].refresh_token == "ghr_2"


@pytest.mark.asyncio
async def test_nonexpiring_connection_and_expired_refresh_token_skip_token_calls():
    store: Final = _Store(_payload())
    prisma_client, _ = _prisma(store)
    connections: Final = {
        "copilot-cred": _payload(refresh_token=None),
        "expired-refresh": _payload(refresh_expires_at=1.0),
    }
    with _github_http(lambda request: pytest.fail(f"unexpected GitHub request to {request.url}")) as requests:
        result: Final = await arefresh_expiring_connections(prisma_client, DualCache(), "user-a", connections)
    assert result == connections
    assert not requests


@pytest.mark.asyncio
async def test_reactive_exchange_401_refreshes_the_github_token():
    store: Final = _Store(_payload(access_expires_at=9999999999.0))
    prisma_client, _ = _prisma(store)

    def respond(request: httpx.Request) -> httpx.Response:
        if str(request.url) == _API_KEY_URL:
            return httpx.Response(401, json={"message": "expired"}, request=request)
        return httpx.Response(
            200,
            json={"access_token": "gho_reactive", "refresh_token": "ghr_reactive"},
            request=request,
        )

    with _github_http(respond) as requests:
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": _payload(access_expires_at=9999999999.0)},
        )
    assert [str(request.url) for request in requests] == [_API_KEY_URL, _ACCESS_TOKEN_URL]
    assert result["copilot-cred"].access_token == "gho_reactive"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 404])
async def test_exchange_permission_failures_do_not_refresh_github_token(status: int) -> None:
    stale: Final = _payload(access_expires_at=9999999999.0)
    store: Final = _Store(stale)
    original_ciphertext: Final = store.ciphertext
    prisma_client, table = _prisma(store)

    def respond(request: httpx.Request) -> httpx.Response:
        if str(request.url) == _API_KEY_URL:
            return httpx.Response(status, json={"message": "Permission denied"}, request=request)
        return httpx.Response(
            200,
            json={"access_token": "gho_rotated", "refresh_token": "ghr_rotated"},
            request=request,
        )

    with _github_http(respond) as requests:
        first: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": stale},
        )
        second: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": first["copilot-cred"]},
        )

    stored: Final = await upc.get_user_provider_credential(prisma_client, "user-a", "copilot-cred")
    assert [str(request.url) for request in requests] == [_API_KEY_URL, _API_KEY_URL]
    assert table.update_many.await_count == 0
    assert store.ciphertext == original_ciphertext
    assert stored == stale
    assert first["copilot-cred"] == stale
    assert second["copilot-cred"] == stale


@pytest.mark.asyncio
async def test_redis_lock_loser_rereads_the_persisted_connection():
    old: Final = _payload()
    rotated: Final = _payload("gho_winner", "ghr_winner", 9999999999.0)
    store: Final = _Store(rotated)
    prisma_client, _ = _prisma(store)
    cache: Final = DualCache(redis_cache=_RedisCache())
    with _github_http(lambda request: pytest.fail(f"lock loser refreshed at {request.url}")) as requests:
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            cache,
            "user-a",
            {"copilot-cred": old},
        )
    assert result["copilot-cred"] == rotated
    assert not requests


@pytest.mark.asyncio
async def test_compare_and_set_loss_does_not_resurrect_disconnected_connection():
    store: Final = _Store(_payload())
    prisma_client, table = _prisma(store)

    def respond(request: httpx.Request) -> httpx.Response:
        store.exists = False
        return httpx.Response(
            200,
            json={"access_token": "gho_rotated", "refresh_token": "ghr_rotated"},
            request=request,
        )

    with _github_http(respond):
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": _payload()},
        )
    assert result == {}
    table.update_many.assert_awaited_once()
    table.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_refresh_deleted_after_cache_set_restores_disconnect_tombstone(monkeypatch: pytest.MonkeyPatch):
    store: Final = _Store(_payload())
    prisma_client, _ = _prisma(store)
    cache, redis_cache = _cache_with_fake_redis(monkeypatch)
    await upc.invalidate_user_provider_credential_cache(cache, "user-a", "copilot-cred")
    original_set_cache: Final = redis_cache.async_set_cache

    async def set_then_delete_row(key: str | None, value: object, **kwargs: object) -> None:
        await original_set_cache(key, value, **kwargs)
        if isinstance(value, str) and value != upc._NOT_CONNECTED:
            store.exists = False

    monkeypatch.setattr(redis_cache, "async_set_cache", set_then_delete_row)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "gho_rotated", "refresh_token": "ghr_rotated"},
            request=request,
        )

    with _github_http(respond):
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            cache,
            "user-a",
            {"copilot-cred": _payload()},
        )
    cached: Final = await redis_cache.async_get_cache(upc._cache_key("user-a", "copilot-cred"))
    assert result == {}
    assert cached == upc._NOT_CONNECTED


@pytest.mark.asyncio
async def test_refresh_returns_replacement_written_after_cache_set_and_drops_stale_cache(
    monkeypatch: pytest.MonkeyPatch,
):
    store: Final = _Store(_payload())
    prisma_client, _ = _prisma(store)
    cache, redis_cache = _cache_with_fake_redis(monkeypatch)
    replacement: Final = _payload("gho_reconnected", "ghr_reconnected", 9999999999.0)
    original_set_cache: Final = redis_cache.async_set_cache

    async def set_then_replace_row(key: str | None, value: object, **kwargs: object) -> None:
        await original_set_cache(key, value, **kwargs)
        if isinstance(value, str) and value != upc._NOT_CONNECTED:
            store.ciphertext = upc._encode(replacement)

    monkeypatch.setattr(redis_cache, "async_set_cache", set_then_replace_row)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"access_token": "gho_rotated", "refresh_token": "ghr_rotated"},
            request=request,
        )

    with _github_http(respond):
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            cache,
            "user-a",
            {"copilot-cred": _payload()},
        )
    cached: Final = await redis_cache.async_get_cache(upc._cache_key("user-a", "copilot-cred"))
    assert result["copilot-cred"] == replacement
    assert cached is None


@pytest.mark.asyncio
async def test_rejected_refresh_waits_for_winner_then_returns_winner_token():
    old: Final = _payload()
    store: Final = _RacingStore(old)
    prisma_client, _ = _prisma(store)
    clock: Final = _Clock()
    attempts: Final[list[str]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        attempts.append(body["refresh_token"])
        if len(attempts) == 1:
            return httpx.Response(
                200,
                json={"access_token": "gho_winner", "refresh_token": "ghr_winner"},
                request=request,
            )
        return httpx.Response(400, json={"error": "invalid_grant"}, request=request)

    with _github_http(respond) as requests:
        results: Final = await asyncio.gather(
            arefresh_expiring_connections(
                prisma_client,
                DualCache(),
                "user-a",
                {"copilot-cred": old},
                single_flight=GithubCopilotTokenRefreshSingleFlight(),
                sleep=clock.sleep,
                clock=clock.monotonic,
            ),
            arefresh_expiring_connections(
                prisma_client,
                DualCache(),
                "user-a",
                {"copilot-cred": old},
                single_flight=GithubCopilotTokenRefreshSingleFlight(),
                sleep=clock.sleep,
                clock=clock.monotonic,
            ),
        )
    assert attempts == ["ghr_old", "ghr_old"]
    assert len(requests) == 2
    assert [result["copilot-cred"].access_token for result in results] == ["gho_winner", "gho_winner"]


@pytest.mark.asyncio
async def test_rejected_refresh_times_out_then_exchange_returns_reconnect_401():
    store: Final = _Store(_payload())
    prisma_client, _ = _prisma(store)
    clock: Final = _Clock()

    def respond(request: httpx.Request) -> httpx.Response:
        if str(request.url) == _ACCESS_TOKEN_URL:
            return httpx.Response(400, json={"error": "invalid_grant"}, request=request)
        return httpx.Response(401, json={"message": "expired"}, request=request)

    with _github_http(respond):
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": _payload()},
            sleep=clock.sleep,
            clock=clock.monotonic,
        )
        kwargs: Final[dict[str, object]] = {
            "litellm_credential_name": "copilot-cred",
            "secret_fields": {
                "user_provider_credentials_user_id": "user-a",
                "user_provider_credentials": {
                    "copilot-cred": {
                        "access_token": result["copilot-cred"].access_token,
                        "github_host": "github.com",
                    }
                },
            },
        }
        with pytest.raises(litellm.CallerCredentialAuthenticationError):
            await aattach_github_copilot_user_session(kwargs)
    assert clock.value == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_unavailable_refresh_returns_stale_without_polling():
    store: Final = _Store(_payload())
    prisma_client, _ = _prisma(store)
    clock: Final = _Clock()

    with _github_http(lambda request: httpx.Response(500, request=request)):
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": _payload()},
            sleep=clock.sleep,
            clock=clock.monotonic,
        )
    assert result["copilot-cred"] == _payload()
    assert clock.value == 0.0


@pytest.mark.asyncio
async def test_rejected_refresh_clears_metadata_for_later_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.credential_endpoints import github_copilot_token_refresh as refresh_module

    monkeypatch.setattr(refresh_module.time, "time", lambda: 1000.0)
    stale: Final = _payload(access_expires_at=900.0, refresh_expires_at=2000.0)
    store: Final = _Store(stale)
    prisma_client, _ = _prisma(store)
    clock: Final = _Clock()

    def reject_refresh(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad_refresh_token"}, request=request)

    with _github_http(reject_refresh) as requests:
        first: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": stale},
            sleep=clock.sleep,
            clock=clock.monotonic,
        )
        stored: Final = await upc.get_user_provider_credential(prisma_client, "user-a", "copilot-cred")
        assert stored is not None
        assert stored.access_token == stale.access_token
        assert stored.access_token_expires_at == stale.access_token_expires_at
        assert stored.refresh_token is None
        assert stored.refresh_token_expires_at is None
        assert first["copilot-cred"] == stored
        before_second: Final = clock.monotonic()
        second: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": stored},
            sleep=clock.sleep,
            clock=clock.monotonic,
        )
        assert clock.monotonic() == before_second

    assert second["copilot-cred"] == stored
    assert len(requests) == 1
    assert json.loads(requests[0].content)["refresh_token"] == stale.refresh_token


@pytest.mark.asyncio
async def test_unavailable_refresh_preserves_metadata_and_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.credential_endpoints import github_copilot_token_refresh as refresh_module

    monkeypatch.setattr(refresh_module.time, "time", lambda: 1000.0)
    stale: Final = _payload(access_expires_at=900.0, refresh_expires_at=2000.0)
    store: Final = _Store(stale)
    prisma_client, _ = _prisma(store)
    clock: Final = _Clock()

    def unavailable(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "temporarily_unavailable"}, request=request)

    with _github_http(unavailable) as requests:
        first: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": stale},
            sleep=clock.sleep,
            clock=clock.monotonic,
        )
        stored: Final = await upc.get_user_provider_credential(prisma_client, "user-a", "copilot-cred")
        assert stored is not None
        assert stored == stale
        second: Final = await arefresh_expiring_connections(
            prisma_client,
            DualCache(),
            "user-a",
            {"copilot-cred": stored},
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    assert first["copilot-cred"] == stale
    assert second["copilot-cred"] == stale
    assert [json.loads(request.content)["refresh_token"] for request in requests] == ["ghr_old", "ghr_old"]
    assert clock.value == 0.0


@pytest.mark.asyncio
async def test_rejected_refresh_cas_loss_returns_replacement_without_cache_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.credential_endpoints import github_copilot_token_refresh as refresh_module

    monkeypatch.setattr(refresh_module.time, "time", lambda: 1000.0)
    stale: Final = _payload(access_expires_at=900.0, refresh_expires_at=2000.0)
    replacement: Final = _payload(
        access_token="gho_reconnected",
        refresh_token="ghr_reconnected",
        access_expires_at=3000.0,
        refresh_expires_at=4000.0,
    )
    store: Final = _Store(stale)
    prisma_client, table = _prisma(store)
    cache, redis_cache = _cache_with_fake_redis(monkeypatch)
    clock: Final = _Clock()
    cache_writes: Final[list[object]] = []
    original_update_many: Final = store.update_many
    original_set_cache: Final = redis_cache.async_set_cache

    async def replace_before_cas(*, where: dict[str, object], data: dict[str, object]) -> int:
        store.ciphertext = upc._encode(replacement)
        return await original_update_many(where=where, data=data)

    async def record_cache_write(key: str | None, value: object, **kwargs: object) -> None:
        cache_writes.append(value)
        await original_set_cache(key, value, **kwargs)

    table.update_many = AsyncMock(side_effect=replace_before_cas)
    monkeypatch.setattr(redis_cache, "async_set_cache", record_cache_write)

    def reject_refresh(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad_refresh_token"}, request=request)

    with _github_http(reject_refresh):
        result: Final = await arefresh_expiring_connections(
            prisma_client,
            cache,
            "user-a",
            {"copilot-cred": stale},
            sleep=clock.sleep,
            clock=clock.monotonic,
        )

    stored: Final = await upc.get_user_provider_credential(prisma_client, "user-a", "copilot-cred")
    assert result["copilot-cred"] == replacement
    assert stored == replacement
    assert cache_writes == []
