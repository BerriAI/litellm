from __future__ import annotations

import asyncio
import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import pytest
from fastapi import Request

import litellm
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, SourcedFields
from litellm.integrations.custom_logger_v2.legacy import PROXY_REQUEST_KEYS, KwargsFields
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_TeamTable, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import _authorize_authenticated_request
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.spend_tracking import budget_reservation, native_budgets
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.proxy.v2_proxy_fields import PRE_CALL_SOURCES
from litellm.redis.store_settings import RedisStoreSettings
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.router import Router
from litellm.rust_bridge._native import NativeRedisClient, NativeSpendCounters


def native_budgets_enabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "callbacks", [])
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # each test installs its owner over its own store
    yield
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # later tests run without a native owner


def install_native_owner(store: SpendCounterStore) -> native_budgets.NativeBudgetsLogger:
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # the owner binds the store the test installed
    for stale in [
        callback for callback in litellm.callbacks if isinstance(callback, native_budgets.NativeBudgetsLogger)
    ]:
        litellm.callbacks.remove(stale)
    native_budgets.install_native_budgets(store, litellm.callbacks)
    return native_owner()


def native_owner() -> native_budgets.NativeBudgetsLogger:
    if native_budgets.native_budgets() is None:
        return install_native_owner(proxy_server.spend_counter_store)
    owner: Final = next(
        (callback for callback in litellm.callbacks if isinstance(callback, native_budgets.NativeBudgetsLogger)), None
    )
    assert owner is not None, "the native budgets owner is not in litellm.callbacks"
    return owner


async def native_pre_call(auth: UserAPIKeyAuth, data: Mapping[str, object]) -> Patch | Abort | None:
    request: Final = {**data, "litellm_call_id": uuid.uuid4().hex}
    fields: Final = SourcedFields(
        KwargsFields(request, PROXY_REQUEST_KEYS), PRE_CALL_SOURCES, HookArgs(auth=auth, request=request)
    )
    outcome: Final = await native_owner().run_rust(
        Hook.pre_call,
        frozenset({Field.budget_plan}),
        FieldView(fields),
        CallContext[object](hook=Hook.pre_call, call=None, state=None),
    )
    if isinstance(outcome, Patch):
        _ = fields.apply(outcome)
    return outcome


async def admit_native_plan(auth: UserAPIKeyAuth, data: Mapping[str, object]) -> None:
    if auth.budget_plan is None:
        return
    outcome: Final = await native_pre_call(auth, data)
    if isinstance(outcome, Abort) and outcome.exception is not None:
        raise outcome.exception


async def authorize_and_admit(
    token: UserAPIKeyAuth,
    request: Request,
    request_data: dict[str, object],
    route: str,
    api_key: str,
    *,
    force_virtual_key_checks: bool = False,
) -> UserAPIKeyAuth | None:
    billed: Final = await _authorize_authenticated_request(
        token, request, request_data, route, api_key, force_virtual_key_checks=force_virtual_key_checks
    )
    await admit_native_plan(billed or token, request_data)
    return billed


def refused_redis_client() -> NativeRedisClient:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: Final[int] = probe.getsockname()[1]
    client: Final = NativeRedisClient.connect(
        RedisStoreSettings(
            redis_kwargs={"host": "127.0.0.1", "port": port}, namespace=None, default_ttl=60, breaker=None
        )
    )
    assert client is not None
    return client


SET_MAX_SCRIPT_MARKER: Final = "tonumber(cur) < tonumber(ARGV[1])"


def _bulk(value: float | None) -> bytes:
    if value is None:
        return b"$-1\r\n"
    encoded: Final = repr(value).encode()
    return b"$%d\r\n%s\r\n" % (len(encoded), encoded)


async def _bulk_string(reader: asyncio.StreamReader) -> str:
    size: Final = int((await reader.readline())[1:])
    return (await reader.readexactly(size + 2))[:-2].decode()


@dataclass
class RespRedis:
    values: dict[str, float]
    failing: frozenset[str] = frozenset()
    expires_after_increment: str | None = None
    ttl: float | None = None
    fail_once: list[Callable[[tuple[str, ...]], bool]] = field(default_factory=list)
    commands: list[tuple[str, ...]] = field(default_factory=list)
    expires_at: dict[str, float] = field(default_factory=dict)
    refresh_attempts: int = 0
    refresh_count: int = 0

    def get(self, key: str) -> float | None:
        self._evict(key)
        return self.values.get(key)

    def _evict(self, key: str) -> None:
        if self.expires_at.get(key, float("inf")) <= time.monotonic():
            _ = self.values.pop(key, None)
            _ = self.expires_at.pop(key, None)

    def _store(self, key: str, value: float) -> None:
        self.values[key] = value
        if self.ttl is not None:
            self.expires_at[key] = time.monotonic() + self.ttl

    def _is_refresh(self, key: str) -> bool:
        previous: Final = self.commands[-2] if len(self.commands) > 1 else ()
        return previous[:2] != ("INCRBYFLOAT", key)

    def _set(self, key: str, value: str, options: Sequence[str]) -> bytes:
        if "NX" in (option.upper() for option in options) and key in self.values:
            return b"$-1\r\n"
        self._store(key, float(value))
        return b"+OK\r\n"

    def _set_max(self, key: str, spend: str) -> bytes:
        current: Final = self.values.get(key)
        if current is None or current < float(spend):
            self._store(key, float(spend))
            return _bulk(float(spend))
        return _bulk(current)

    def _refused_once(self, command: tuple[str, ...]) -> bool:
        refusal: Final = next((matches for matches in self.fail_once if matches(command)), None)
        if refusal is None:
            return False
        self.fail_once.remove(refusal)
        return True

    def _expire(self, key: str) -> bytes:
        refresh: Final = self._is_refresh(key)
        if self.ttl is None:
            return b":1\r\n"
        if key not in self.values:
            return b":0\r\n"
        self.refresh_count += refresh
        self.expires_at[key] = time.monotonic() + self.ttl
        return b":1\r\n"

    def answer(self, command: tuple[str, ...]) -> bytes:
        self.commands.append(command)
        name, *args = command
        for key in self.values.keys() & set(args):
            self._evict(key)
        if name.upper() == "EXPIRE":
            self.refresh_attempts += self._is_refresh(args[0])
        if name.upper() in self.failing or self._refused_once(command):
            return b"-ERR refused by the test\r\n"
        match name.upper():
            case "GET":
                return _bulk(self.values.get(args[0]))
            case "MGET":
                return b"*%d\r\n" % len(args) + b"".join(_bulk(self.values.get(key)) for key in args)
            case "INCRBYFLOAT":
                total: Final = self.values.get(args[0], 0.0) + float(args[1])
                self._store(args[0], total)
                if args[0] == self.expires_after_increment:
                    del self.values[args[0]]
                return _bulk(total)
            case "DEL":
                return b":%d\r\n" % sum(self.values.pop(key, None) is not None for key in args)
            case "EXPIRE":
                return self._expire(args[0])
            case "SET":
                return self._set(args[0], args[1], args[2:])
            case "EVAL" if SET_MAX_SCRIPT_MARKER in args[0]:
                return self._set_max(args[2], args[3])
            case "EVAL":
                return b"-ERR script not emulated by the test\r\n"
            case _:
                return b"+OK\r\n"

    def writes(self) -> list[tuple[str, ...]]:
        return [command for command in self.commands if command[0].upper() not in ("GET", "MGET")]

    def deleted(self) -> set[str]:
        return {key for command in self.commands if command[0].upper() == "DEL" for key in command[1:]}

    def count(self, name: str) -> int:
        return sum(command[0].upper() == name for command in self.commands)

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while header := await reader.readline():
                command: Final = [await _bulk_string(reader) for _ in range(int(header[1:]))]
                writer.write(self.answer(tuple(command)))
                await writer.drain()
        finally:
            writer.close()


async def _stop_serving(server: asyncio.Server) -> None:
    server.close()
    connections: Final = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    for connection in connections:
        _ = connection.cancel()
    _ = await asyncio.gather(*connections, return_exceptions=True)


@contextmanager
def resp_store(redis: RespRedis, default_ttl: float = 60) -> Iterator[SpendCounterStore]:
    loop: Final = asyncio.new_event_loop()
    thread: Final = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    server: Final = asyncio.run_coroutine_threadsafe(asyncio.start_server(redis.serve, "127.0.0.1", 0), loop).result(
        timeout=5
    )
    port: Final[int] = server.sockets[0].getsockname()[1]
    try:
        client: Final = NativeRedisClient.connect(
            RedisStoreSettings(
                redis_kwargs={"host": "127.0.0.1", "port": port},
                namespace=None,
                default_ttl=default_ttl,
                breaker=None,
            )
        )
        assert client is not None
        yield SpendCounterStore(NativeSpendCounters(client))
    finally:
        asyncio.run_coroutine_threadsafe(_stop_serving(server), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


UNLIMITED_BUDGET: Final = 1e12


async def native_reservation(
    store: SpendCounterStore,
    auth: UserAPIKeyAuth,
    entries: Sequence[Mapping[str, object]],
    reserved_cost: float,
    input_cost: float = 0.0,
    route: str = "/v1/chat/completions",
) -> dict[str, object]:
    _ = install_native_owner(store)
    plan: Final = {
        "counters": [
            {
                "counter_key": entry["counter_key"],
                "max_budget": entry.get("max_budget", UNLIMITED_BUDGET),
                "fallback_spend": entry.get("fallback_spend", 0.0),
                "entity_type": entry.get("entity_type"),
                "entity_id": entry.get("entity_id"),
                "spend_log_entity_id": entry.get("spend_log_entity_id", entry.get("entity_id")),
            }
            for entry in entries
        ],
        "reservation_cost": float(reserved_cost),
        "input_cost": float(input_cost),
        "input_tokens": None,
        "throttle_counter": None,
        "fail_closed": False,
        "status_code": int(litellm.budget_exceeded_status_code),
        "reservation_id": native_budgets.new_reservation_id(),
    }
    _ = native_budgets.pending_native_reservation(plan, auth, route, {}, {})
    reservation: Final = await native_budgets.admit_native_reservation_now(auth)
    assert reservation is not None and native_budgets.is_native_reservation(reservation)
    return reservation


async def reserve_budget_for_request(
    request_body: dict,
    route: str,
    llm_router: Router | None,
    valid_token: UserAPIKeyAuth | None,
    team_object: LiteLLM_TeamTable | None,
    user_object: LiteLLM_UserTable | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
    end_user_id: str | None = None,
    end_user_object: object = None,
    apply_user_budget_to_team_keys: bool = False,
    fail_closed_budget_enforcement: bool = False,
    raw_body: bytes | None = None,
    path_params: Mapping[str, object] | None = None,
) -> dict | None:
    _ = native_owner()
    plan: Final = await budget_reservation.reserve_budget_for_request(
        request_body=request_body,
        route=route,
        llm_router=llm_router,
        valid_token=valid_token,
        team_object=team_object,
        user_object=user_object,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        proxy_logging_obj=proxy_logging_obj,
        end_user_id=end_user_id,
        end_user_object=end_user_object,
        apply_user_budget_to_team_keys=apply_user_budget_to_team_keys,
        fail_closed_budget_enforcement=fail_closed_budget_enforcement,
        raw_body=raw_body,
        path_params=path_params,
    )
    if plan is None or valid_token is None:
        return plan
    assert valid_token.budget_plan is not None, f"{route} reserved through the Python engine, not the native owner"
    outcome: Final = await native_pre_call(valid_token, request_body)
    if isinstance(outcome, Abort) and outcome.exception is not None:
        raise outcome.exception
    return plan if native_budgets.is_native_reservation(plan) else None
