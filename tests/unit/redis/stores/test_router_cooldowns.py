import ast
import gc
import json
import logging
import time
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import pytest

import litellm
from litellm._internal_context import current_service_target
from litellm.caching.redis_batch import Batch, request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache, redis_breaker_state_counts
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.redis.stores.connection import RedisStores, connect
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, RouterCooldownStore
from litellm.redis.store_settings import RedisBreakerSettings, RedisStoreSettings
from litellm.rust_bridge._native import NativeRedisClient, NativeRouterCooldowns
from litellm.rust_bridge.host import settings
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem
from tests.unit.caching.redis_batch_double import (
    FakeNativePending,
    FakeNatives,
    ReadyPending,
    fake_pending,
    fake_send,
)


@dataclass(frozen=True, slots=True)
class _WireCooldown:
    """One cooldown as main's ``RedisCache`` writes it (``str(value)``) and reads it (``json.loads`` then
    ``ast.literal_eval``). The same literals sit in ``litellm/rust/crates/redis-store/tests/support/mod.rs``,
    where real Redis is checked to hold them after the Rust store writes."""

    cooldown: CooldownCacheValue
    str_dict: str
    json: str
    ttl_seconds: float
    redis_ttl_seconds: int


_WIRE_COOLDOWNS: Final = (
    _WireCooldown(
        cooldown={
            "exception_received": "RateLimitError: limited é 'quoted'",
            "status_code": "429",
            "timestamp": 1730000000.25,
            "cooldown_time": 60,
        },
        str_dict="{'exception_received': \"RateLimitError: limited é 'quoted'\", 'status_code': '429', 'timestamp': 1730000000.25, 'cooldown_time': 60}",
        json='{"exception_received": "RateLimitError: limited \\u00e9 \'quoted\'", "status_code": "429", "timestamp": 1730000000.25, "cooldown_time": 60}',
        ttl_seconds=90.5,
        redis_ttl_seconds=90,
    ),
    _WireCooldown(
        cooldown={
            "exception_received": "Timeout",
            "status_code": "408",
            "timestamp": 1730000000.5,
            "cooldown_time": 0.5,
        },
        str_dict="{'exception_received': 'Timeout', 'status_code': '408', 'timestamp': 1730000000.5, 'cooldown_time': 0.5}",
        json='{"exception_received": "Timeout", "status_code": "408", "timestamp": 1730000000.5, "cooldown_time": 0.5}',
        ttl_seconds=30.0,
        redis_ttl_seconds=30,
    ),
)


def _entry(model_id: str = "dep") -> CooldownCacheValue:
    return {"exception_received": model_id, "status_code": "429", "timestamp": 1000.0, "cooldown_time": 120.0}


@dataclass
class FakeCooldowns:
    has_redis: bool = False
    stored: dict[str, CooldownCacheValue] = field(default_factory=dict)
    fails: dict[str, int] = field(default_factory=dict)
    targets: list[str | None] = field(default_factory=list)
    smallest: float | None = None
    declared: list[tuple[str, ...]] = field(default_factory=list)

    def attach(self, client: NativeRedisClient, /) -> bool:
        attached: Final = not self.has_redis
        self.has_redis = True
        return attached

    def put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None:
        self.targets.append(current_service_target())
        self.stored[model_id] = CooldownCacheValue(**cooldown)  # pyright: ignore[reportArgumentType]  # test double

    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int:
        self.targets.append(current_service_target())
        self.fails[model_id] = self.fails.get(model_id, 0) + 1
        return self.fails[model_id]

    async def async_put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None:
        self.put(model_id, cooldown, ttl_seconds)

    async def async_increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int:
        return self.increment_allowed_fails(model_id, suffix, ttl_seconds)

    def active_sync(self, model_ids: Sequence[str]) -> list[tuple[str, CooldownCacheValue]]:
        self.targets.append(current_service_target())
        return [(model_id, self.stored[model_id]) for model_id in model_ids if model_id in self.stored]

    def active(
        self, model_ids: Sequence[str], send: object
    ) -> FakeNativePending[list[tuple[str, CooldownCacheValue]]] | ReadyPending[list[tuple[str, CooldownCacheValue]]]:
        self.targets.append(current_service_target())
        self.declared.append(tuple(model_ids))
        return fake_pending(send, lambda: self.active_sync(model_ids))

    def minimum(self, model_ids: Sequence[str]) -> float | None:
        return self.smallest

    def delete_local(self, model_id: str) -> None:
        self.stored.pop(model_id, None)

    def flush_local_counters(self) -> None:
        self.fails.clear()

    def snapshot(self) -> dict[str, CooldownCacheValue]:
        return dict(self.stored)


def _client(namespace: str | None = None) -> RedisStores:
    redis: Final = connect(RedisCache(host="redis.internal", port=6380, namespace=namespace))
    assert redis.client is not None
    return redis


def test_late_attach_keeps_the_cooldowns_this_router_already_holds() -> None:
    store: Final = RouterCooldownStore(connect(None).router_cooldowns())
    entry: Final[CooldownCacheValue] = {
        "exception_received": "dep",
        "status_code": "429",
        "timestamp": time.time(),
        "cooldown_time": 120.0,
    }
    store.put("dep", entry, 60.0)
    store.attach_redis(connect(None))
    store.attach_redis(_client())
    assert store.active_sync(["dep"]) == (("dep", entry),)


def test_writes_and_reads_carry_the_router_cooldowns_service_target() -> None:
    handle: Final = FakeCooldowns(has_redis=True)
    store: Final = RouterCooldownStore(handle)
    store.put("dep", _entry(), 60.0)
    assert store.increment_allowed_fails("dep", "timeout", 30.0) == 1
    assert store.active_sync(["dep", "other"]) == (("dep", _entry()),)
    assert handle.targets == ["router_cooldowns"] * 3
    assert current_service_target() is None


@pytest.mark.parametrize(("smallest", "expected"), ((None, 60.0), (0, 60.0), (5, 5)))
def test_minimum_falls_back_to_the_default_when_nothing_or_zero_is_stored(
    smallest: float | None, expected: float
) -> None:
    store: Final = RouterCooldownStore(FakeCooldowns(smallest=smallest))
    assert store.minimum(["dep"], 60.0) == expected


@pytest.mark.parametrize("case", _WIRE_COOLDOWNS, ids=("rate-limit", "timeout"))
def test_the_wire_cases_are_what_mains_codec_writes_and_reads(case: _WireCooldown) -> None:
    assert str(case.cooldown) == case.str_dict
    assert ast.literal_eval(case.str_dict) == case.cooldown
    assert json.loads(case.json) == case.cooldown
    assert int(case.ttl_seconds) == case.redis_ttl_seconds


@pytest.mark.parametrize(
    "setting",
    ({"ssl_ciphers": "HIGH"}, {"credential_provider": "not-a-provider"}),
)
def test_a_redis_setting_the_native_client_cannot_honor_never_fails_construction(
    setting: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    redis: Final = RedisCache(host="127.0.0.1", port=1, **setting)
    with caplog.at_level(logging.WARNING):
        store: Final = RouterCooldownStore(connect(redis).router_cooldowns())
    cooldown: Final = CooldownCacheValue(
        exception_received="model", status_code="429", timestamp=time.time(), cooldown_time=120.0
    )
    store.put("model", cooldown, 60.0)

    assert store.active_sync(["model"]) == (("model", cooldown),)
    assert next(iter(setting)) in caplog.text


def test_an_ocsp_check_the_native_client_cannot_perform_raises_like_the_async_redis_client() -> None:
    redis: Final = RedisCache(host="127.0.0.1", port=1, ssl=True, ssl_validate_ocsp=True)

    with pytest.raises(TypeError):
        connect(redis)


class _RecordingSecrets(CustomSecretManager):
    def __init__(self, values: Mapping[str, str]) -> None:
        super().__init__(secret_manager_name="router_cooldowns_test")  # pyright: ignore[reportUnknownMemberType]  # CustomSecretManager takes untyped **kwargs
        self.values: Final = values
        self.asked: tuple[str, ...] = ()

    async def async_read_secret(
        self, secret_name: str, optional_params: dict[str, object] | None = None, timeout: object = None
    ) -> str | None:
        raise AssertionError("get_secret reads custom managers synchronously")

    def sync_read_secret(
        self, secret_name: str, optional_params: dict[str, object] | None = None, timeout: object = None
    ) -> str | None:
        self.asked = (*self.asked, secret_name)
        return self.values.get(secret_name)


def test_the_native_client_reads_redis_secrets_through_the_secret_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    manager: Final = _RecordingSecrets({"REDIS_PASSWORD": "vault-password"})
    monkeypatch.setattr(litellm, "secret_manager_client", manager)
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))
    configured: Final = settings.secret_manager
    monkeypatch.setattr(settings, "secret_manager", lambda: replace(configured(), native=False))

    _ = connect(RedisCache(host="127.0.0.1", port=1, namespace=f"secret-manager-test-{uuid.uuid4()}"))

    assert "REDIS_PASSWORD" in manager.asked


def test_the_native_breaker_counts_in_the_redis_breaker_state_gauge() -> None:
    open_before: Final = redis_breaker_state_counts().get("open", 0)
    store: Final = NativeRouterCooldowns(
        NativeRedisClient.connect(
            RedisStoreSettings(
                redis_kwargs={"host": f"breaker-gauge-{uuid.uuid4()}.invalid", "port": 1},
                namespace=None,
                default_ttl=60,
                breaker=RedisBreakerSettings(failure_threshold=1, recovery_timeout=60.0, timeout_min_duration=5.0),
            )
        )
    )

    store.put("model", _entry("model"), 60.0)

    assert redis_breaker_state_counts().get("open", 0) == open_before + 1


@pytest.mark.asyncio
async def test_a_declared_active_read_answers_after_the_native_pre_flush_under_the_router_cooldowns_target() -> None:
    handle: Final = FakeCooldowns(has_redis=True)
    store: Final = RouterCooldownStore(handle)
    natives: Final = FakeNatives()
    store.put("dep", _entry(), 60.0)

    with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        declared: Final = store.active(["dep", "other"], batch=Batch.PRE_CALL)
        assert natives.built[0].events == []
        answer: Final = tuple(await declared)

    assert answer == (("dep", _entry()),)
    assert natives.built[0].events == ["flush_pre", "resolve"]
    assert handle.declared == [("dep", "other")]
    assert handle.targets[1:] == ["router_cooldowns", "router_cooldowns"]


@pytest.mark.asyncio
async def test_an_active_read_outside_a_request_goes_out_on_its_own() -> None:
    store: Final = RouterCooldownStore(FakeCooldowns(has_redis=True))
    store.put("dep", _entry(), 60.0)

    assert await store.active(["dep", "other"], batch=Batch.NONE) == (("dep", _entry()),)


@pytest.mark.asyncio
async def test_an_active_read_declared_on_the_pre_call_batch_and_never_awaited_leaves_no_coroutine() -> None:
    """An armed cooldown prefetch the request never routes, because the limiter or a budget check rejected it first,
    is dropped unawaited; that must not leave a coroutine behind to warn when it is collected."""
    store: Final = RouterCooldownStore(FakeCooldowns(has_redis=True))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
            _ = store.active(["dep"], batch=Batch.PRE_CALL)
        _ = gc.collect()

    assert [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)] == []
