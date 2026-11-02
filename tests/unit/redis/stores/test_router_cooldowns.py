import ast
import asyncio
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Final

import pytest

import litellm
from litellm._internal_context import current_service_target
from litellm.caching.redis_cache import RedisCache, redis_breaker_state_counts
from litellm.integrations.custom_secret_manager import CustomSecretManager
from litellm.redis.stores._connection import redis_kwargs
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, RouterCooldownStore
from litellm.rust_bridge._native import NativeRouterCooldowns
from litellm.rust_bridge.host import settings
from litellm.types.secret_managers.main import KeyManagementSettings, KeyManagementSystem


@dataclass(frozen=True, slots=True)
class _WireCooldown:
    """One cooldown as main's ``RedisCache`` writes it (``str(value)``) and reads it (``json.loads`` then
    ``ast.literal_eval``). The same literals sit in ``litellm-rust/crates/redis-store/tests/support/mod.rs``,
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
    redis_kwargs: Mapping[str, object] | None
    namespace: str | None
    stored: dict[str, CooldownCacheValue] = field(default_factory=dict)
    fails: dict[str, int] = field(default_factory=dict)
    targets: list[str | None] = field(default_factory=list)
    smallest: float | None = None

    def put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None:
        self.targets.append(current_service_target())
        self.stored[model_id] = CooldownCacheValue(**cooldown)  # pyright: ignore[reportArgumentType]  # test double

    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int:
        self.targets.append(current_service_target())
        self.fails[model_id] = self.fails.get(model_id, 0) + 1
        return self.fails[model_id]

    def active(self, model_ids: Sequence[str]) -> list[tuple[str, CooldownCacheValue]]:
        self.targets.append(current_service_target())
        return [(model_id, self.stored[model_id]) for model_id in model_ids if model_id in self.stored]

    def aactive(self, model_ids: Sequence[str]) -> "asyncio.Future[list[tuple[str, CooldownCacheValue]]]":
        future: Final[asyncio.Future[list[tuple[str, CooldownCacheValue]]]] = asyncio.get_running_loop().create_future()
        future.set_result(self.active(model_ids))
        return future

    def minimum(self, model_ids: Sequence[str]) -> float | None:
        return self.smallest

    def evict_local(self, model_id: str) -> None:
        self.stored.pop(model_id, None)

    def flush_local_counters(self) -> None:
        self.fails.clear()

    def snapshot(self) -> dict[str, CooldownCacheValue]:
        return dict(self.stored)


@dataclass
class Handles:
    built: list[FakeCooldowns] = field(default_factory=list)

    def __call__(self, connection: RedisCache | None) -> FakeCooldowns:
        handle: Final = FakeCooldowns(
            None if connection is None else redis_kwargs(connection),
            None if connection is None else connection.namespace,
        )
        self.built.append(handle)
        return handle


def _redis(namespace: str | None = None) -> RedisCache:
    return RedisCache(host="redis.internal", port=6380, namespace=namespace)


def test_the_store_hands_the_native_handle_redis_kwargs_and_namespace_not_a_client() -> None:
    handles: Final = Handles()
    RouterCooldownStore(_redis("tenant"), handle=handles)
    assert handles.built[0].redis_kwargs == {"host": "redis.internal", "port": 6380, "socket_timeout": 5.0}
    assert handles.built[0].namespace == "tenant"


def test_late_attach_rebuilds_once_and_later_attaches_are_ignored() -> None:
    handles: Final = Handles()
    store: Final = RouterCooldownStore(None, handle=handles)
    store.put("dep", _entry(), 60.0)
    store.attach_redis(None)
    store.attach_redis(_redis())
    store.attach_redis(_redis("other"))
    assert [handle.redis_kwargs is not None for handle in handles.built] == [False, True]
    assert handles.built[1].namespace is None
    assert store.active_sync(["dep"], None) == ()


def test_writes_and_reads_carry_the_router_cooldowns_service_target() -> None:
    handles: Final = Handles()
    store: Final = RouterCooldownStore(_redis(), handle=handles)
    store.put("dep", _entry(), 60.0)
    assert store.increment_allowed_fails("dep", "timeout", 30.0) == 1
    assert store.active_sync(["dep", "other"], None) == (("dep", _entry()),)
    assert handles.built[0].targets == ["router_cooldowns"] * 3
    assert current_service_target() is None


@pytest.mark.parametrize(("smallest", "expected"), ((None, 60.0), (0, 60.0), (5, 5)))
def test_minimum_falls_back_to_the_default_when_nothing_or_zero_is_stored(
    smallest: float | None, expected: float
) -> None:
    handles: Final = Handles()
    store: Final = RouterCooldownStore(None, handle=handles)
    handles.built[0].smallest = smallest
    assert store.minimum(["dep"], 60.0, None) == expected


@pytest.mark.parametrize("case", _WIRE_COOLDOWNS, ids=("rate-limit", "timeout"))
def test_the_wire_cases_are_what_mains_codec_writes_and_reads(case: _WireCooldown) -> None:
    assert str(case.cooldown) == case.str_dict
    assert ast.literal_eval(case.str_dict) == case.cooldown
    assert json.loads(case.json) == case.cooldown
    assert int(case.ttl_seconds) == case.redis_ttl_seconds


@pytest.mark.parametrize(
    "setting",
    ({"ssl_ciphers": "HIGH"}, {"ssl": True, "ssl_validate_ocsp": True}, {"credential_provider": "not-a-provider"}),
)
def test_a_redis_setting_the_native_client_cannot_honor_never_fails_construction(
    setting: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    redis: Final = RedisCache(host="127.0.0.1", port=1, **setting)
    with caplog.at_level(logging.WARNING):
        store: Final = RouterCooldownStore(redis)
    cooldown: Final = CooldownCacheValue(
        exception_received="model", status_code="429", timestamp=time.time(), cooldown_time=120.0
    )
    store.put("model", cooldown, 60.0)

    assert store.active_sync(["model"], None) == (("model", cooldown),)
    assert next(iter(setting)) in caplog.text


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


def test_the_native_store_reads_redis_secrets_through_the_secret_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    manager: Final = _RecordingSecrets({"REDIS_PASSWORD": "vault-password"})
    monkeypatch.setattr(litellm, "secret_manager_client", manager)
    monkeypatch.setattr(litellm, "_key_management_system", KeyManagementSystem.CUSTOM)
    monkeypatch.setattr(litellm, "_key_management_settings", KeyManagementSettings(access_mode="read_only"))
    configured: Final = settings.secret_manager
    monkeypatch.setattr(settings, "secret_manager", lambda: replace(configured(), native=False))

    NativeRouterCooldowns({"host": "127.0.0.1", "port": 1}, namespace="secret-manager-test")

    assert "REDIS_PASSWORD" in manager.asked


def test_the_native_breaker_counts_in_the_redis_breaker_state_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.setenv("REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "1")
    open_before: Final = redis_breaker_state_counts().get("open", 0)
    store: Final = NativeRouterCooldowns({"host": "127.0.0.1", "port": 1}, namespace="breaker-gauge-test")

    store.put("model", _entry("model"), 60.0)

    assert redis_breaker_state_counts().get("open", 0) == open_before + 1
