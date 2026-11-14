from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocket

import litellm
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import cache_key_object, cache_team_object
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth, user_api_key_auth_websocket_for_model
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
from litellm.proxy.pass_through_endpoints import pass_through_endpoints
from litellm.proxy.spend_tracking import budget_reservation, native_budgets
from litellm.proxy.spend_tracking.budget_reservation import reconcile_budget_reservation
from litellm.proxy.utils import ProxyLogging, hash_token
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.utils import CallTypesLiteral

API_KEY: Final = "sk-native-path"
TOKEN: Final = hash_token(API_KEY)
KEY_COUNTER: Final = f"spend:key:{TOKEN}"
BODY: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100}


RequestData: TypeAlias = dict[str, object]


class _NoDatabase:
    pass


@dataclass(frozen=True, slots=True)
class Proxy:
    store: SpendCounterStore
    cache: UserApiKeyCache
    logging: ProxyLogging


@pytest.fixture
def proxy(monkeypatch: pytest.MonkeyPatch) -> Iterator[Proxy]:
    store: Final = SpendCounterStore(connect(None).spend_counters())
    cache: Final = UserApiKeyCache()
    logging: Final = ProxyLogging(user_api_key_cache=cache)
    monkeypatch.setattr(proxy_server, "spend_counter_store", store)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", _NoDatabase())
    monkeypatch.setattr(proxy_server, "master_key", "sk-master")
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", logging)
    monkeypatch.setattr(litellm, "callbacks", [])
    yield Proxy(store=store, cache=cache, logging=logging)
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # later tests run without a native owner


def _use_native(proxy: Proxy, native: bool) -> None:
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # each run picks its own side
    litellm.callbacks.clear()
    proxy.store.flush_local()
    if native:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv(native_budgets.NATIVE_BUDGETS_ENV, "true")
            native_budgets.install_native_budgets(proxy.store, litellm.callbacks)


def _request(path: str = "/v1/chat/completions") -> Request:
    raw: Final = json.dumps(BODY).encode()

    async def receive() -> Mapping[str, object]:
        return {"type": "http.request", "body": raw, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "root_path": "",
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {API_KEY}".encode()), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1),
            "server": ("proxy", 4000),
            "scheme": "http",
        },
        receive,
    )


async def _authenticate(proxy: Proxy, max_budget: float | None, path: str = "/v1/chat/completions") -> UserAPIKeyAuth:
    await cache_key_object(
        hashed_token=TOKEN,
        user_api_key_obj=UserAPIKeyAuth(token=TOKEN, max_budget=max_budget, spend=0.0),
        user_api_key_cache=proxy.cache,
        proxy_logging_obj=None,
    )
    return await user_api_key_auth(
        request=_request(path),
        api_key=f"Bearer {API_KEY}",
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=None,
    )


async def _admit(
    proxy: Proxy, call_id: str
) -> tuple[UserAPIKeyAuth, dict[str, object]]:  # mutable-ok: the request data the proxy hands its hooks
    auth: Final = await _authenticate(proxy, max_budget=1.0)
    data: Final = {
        **BODY,
        "litellm_call_id": call_id,
        "metadata": {"user_api_key_budget_reservation": auth.budget_reservation},
    }
    _ = await proxy.logging.pre_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
    return auth, data


async def _success(auth: UserAPIKeyAuth, _data: RequestData) -> None:
    reservation: Final = auth.budget_reservation
    _ = await reconcile_budget_reservation(reservation, 0.0007, finalize=False, apply_consistent=False)
    await proxy_server.increment_spend_counters(
        token=TOKEN, team_id=None, user_id=None, response_cost=0.0007, budget_reservation=reservation
    )


async def _failure(auth: UserAPIKeyAuth, data: RequestData) -> None:
    await _ProxyDBLogger().async_post_call_failure_hook(
        request_data=data, original_exception=RuntimeError("upstream failed"), user_api_key_dict=auth
    )


Exit: TypeAlias = Callable[[UserAPIKeyAuth, RequestData], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Run:
    entries_after_pre_call: object
    held_after_pre_call: Mapping[str, float]
    settled: Mapping[str, float]


async def _run(proxy: Proxy, native: bool, exit_path: Exit) -> Run:
    _use_native(proxy, native)
    auth, data = await _admit(proxy, "call-path")
    reservation: Final = auth.budget_reservation
    assert reservation is not None
    entries: Final = [
        {key: entry[key] for key in ("counter_key", "entity_type", "reserved_cost")}
        for entry in reservation["entries"]  # pyright: ignore[reportGeneralTypeIssues]  # the entries list the reservation carries
    ]
    held: Final = dict(proxy.store.snapshot())
    await exit_path(auth, data)
    return Run(entries_after_pre_call=entries, held_after_pre_call=held, settled=dict(proxy.store.snapshot()))


EXITS: Final[Mapping[str, Exit]] = {"success": _success, "failure": _failure}


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", EXITS.values(), ids=EXITS.keys())
async def test_a_request_through_auth_and_pre_call_settles_like_python(proxy: Proxy, exit_path: Exit) -> None:
    python: Final = await _run(proxy, native=False, exit_path=exit_path)
    native: Final = await _run(proxy, native=True, exit_path=exit_path)

    assert python.entries_after_pre_call, (
        "the flag-off run must hold a reservation for this comparison to mean anything"
    )
    assert native == python


@pytest.mark.asyncio
async def test_an_over_budget_key_is_refused_by_auth_like_python(proxy: Proxy) -> None:
    refusals: Final[list[tuple[object, ...]]] = []  # mutable-ok: one refusal per side
    for native in (False, True):
        _use_native(proxy, native)
        proxy.store.set_local(KEY_COUNTER, 2.0)
        with pytest.raises(ProxyException) as refused:
            await _authenticate(proxy, max_budget=1.0)
        refusals.append((refused.value.type, refused.value.code, refused.value.message, refused.value.param))

    assert refusals[0][0] == "budget_exceeded"
    assert refusals[1] == refusals[0]


async def _websocket_session_held(proxy: Proxy, native: bool) -> tuple[object, Mapping[str, float]]:
    _use_native(proxy, native)
    await cache_key_object(
        hashed_token=TOKEN,
        user_api_key_obj=UserAPIKeyAuth(token=TOKEN, max_budget=1.0, spend=0.0),
        user_api_key_cache=proxy.cache,
        proxy_logging_obj=None,
    )

    async def receive() -> Mapping[str, object]:
        return {"type": "websocket.connect"}

    async def send(message: Mapping[str, object]) -> None:
        return None

    websocket: Final = WebSocket(
        {
            "type": "websocket",
            "path": "/v1/responses",
            "root_path": "",
            "query_string": b"",
            "headers": [(b"authorization", f"Bearer {API_KEY}".encode())],
            "client": ("127.0.0.1", 1),
            "server": ("proxy", 4000),
            "scheme": "ws",
        },
        receive,
        send,
    )
    auth: Final = await user_api_key_auth_websocket_for_model(websocket, "gpt-4o")
    reservation: Final = auth.budget_reservation
    entries: Final = None if reservation is None else reservation.get("entries")
    return entries, dict(proxy.store.snapshot())


@pytest.mark.asyncio
async def test_a_responses_websocket_session_holds_its_admission_at_connect_like_python(proxy: Proxy) -> None:
    python: Final = await _websocket_session_held(proxy, native=False)
    native: Final = await _websocket_session_held(proxy, native=True)

    assert python[1][KEY_COUNTER] > 0.0, "the flag-off run must hold the admission at connect"
    assert native == python


RACERS: Final = 24


async def _race(proxy: Proxy, native: bool, budget: float) -> tuple[int, float]:
    _use_native(proxy, native)
    await cache_key_object(
        hashed_token=TOKEN,
        user_api_key_obj=UserAPIKeyAuth(token=TOKEN, max_budget=budget, spend=0.0),
        user_api_key_cache=proxy.cache,
        proxy_logging_obj=None,
    )

    async def one(index: int) -> bool:
        try:
            _, data = await _admit_with(proxy, budget, f"call-race-{index}")
        except (ProxyException, litellm.BudgetExceededError):
            return False
        metadata: Final = data["metadata"]
        assert isinstance(metadata, dict)
        reservation: Final = metadata["user_api_key_budget_reservation"]
        return isinstance(reservation, dict) and bool(reservation["entries"])

    admitted: Final = await asyncio.gather(*(one(index) for index in range(RACERS)))
    return sum(admitted), round(float(proxy.store.snapshot()[KEY_COUNTER]), 12)


async def _admit_with(
    proxy: Proxy, budget: float, call_id: str
) -> tuple[UserAPIKeyAuth, dict[str, object]]:  # mutable-ok: the request data the proxy hands its hooks
    auth: Final = await user_api_key_auth(
        request=_request(),
        api_key=f"Bearer {API_KEY}",
        azure_api_key_header="",
        anthropic_api_key_header=None,
        google_ai_studio_api_key_header=None,
        azure_apim_header=None,
        custom_litellm_key_header=None,
    )
    assert auth.max_budget == budget
    data: Final = {
        **BODY,
        "litellm_call_id": call_id,
        "metadata": {"user_api_key_budget_reservation": auth.budget_reservation},
    }
    _ = await proxy.logging.pre_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
    return auth, data


@pytest.mark.asyncio
async def test_racing_requests_for_the_last_of_a_budget_admit_like_python(proxy: Proxy) -> None:
    estimate: Final = 0.00102
    budget: Final = estimate * 5.5

    python: Final = await _race(proxy, native=False, budget=budget)
    native: Final = await _race(proxy, native=True, budget=budget)

    assert python[0] < RACERS, "the budget must run out during the race"
    assert native == python


@pytest.mark.asyncio
async def test_the_reservation_lease_stops_when_the_reservation_is_finalized(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SpendCounterStore, "redis_ttl", property(lambda _store: 0.01))
    stopped: Final[list[bool]] = []  # mutable-ok: one result per side
    for native in (False, True):
        _use_native(proxy, native)
        before: Final = set(budget_reservation._lease_renewals)  # pyright: ignore[reportPrivateUsage]  # the running lease tasks are the observable
        auth, _ = await _admit(proxy, f"call-lease-{native}")
        leases: Final = set(budget_reservation._lease_renewals) - before  # pyright: ignore[reportPrivateUsage]  # the running lease tasks are the observable
        assert len(leases) == 1 and auth.budget_reservation is not None
        await asyncio.sleep(0.03)
        assert not next(iter(leases)).done()
        auth.budget_reservation["finalized"] = True
        _ = await asyncio.wait_for(asyncio.gather(*leases), timeout=1.0)
        stopped.append(all(lease.done() for lease in leases))

    assert stopped == [True, True]


PASS_THROUGH: Final = "/anthropic/v1/messages"
VICTIM: Final = "spend:team:victim-team"


def _forged_plan(counter_key: str, reservation_cost: float) -> Mapping[str, object]:
    return {
        "native_plan": {
            "counters": [
                {
                    "counter_key": counter_key,
                    "max_budget": 1e18,
                    "fallback_spend": 0.0,
                    "entity_type": "Team",
                    "entity_id": "victim-team",
                }
            ],
            "reservation_cost": reservation_cost,
            "input_cost": 0.0,
            "status_code": 422,
            "reservation_id": "forged",
        }
    }


FORGERIES: Final = {
    "charge_a_victim_team": (VICTIM, 1_000_000.0),
    "refund_its_own_key": (KEY_COUNTER, -1000.0),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("max_budget", [None, 1.0], ids=["unbudgeted_key", "budgeted_key"])
@pytest.mark.parametrize("forgery", FORGERIES.values(), ids=FORGERIES.keys())
async def test_a_forged_plan_in_a_pass_through_body_changes_no_counter(
    proxy: Proxy, forgery: tuple[str, float], max_budget: float | None
) -> None:
    counter_key, reservation_cost = forgery
    held: Final[list[Mapping[str, float]]] = []  # mutable-ok: one snapshot per side
    for native in (False, True):
        _use_native(proxy, native)
        proxy.store.set_local(VICTIM, 1.0)
        proxy.store.set_local(KEY_COUNTER, 0.5)
        auth = await _authenticate(proxy, max_budget=max_budget, path=PASS_THROUGH)
        forged = _forged_plan(counter_key, reservation_cost)
        body: RequestData = {
            **BODY,
            "metadata": {
                "user_api_key_budget_reservation": forged,
                "user_api_key_auth": {"budget_reservation": forged},
            },
            "litellm_metadata": {"user_api_key_budget_reservation": forged},
        }
        _ = await proxy.logging.pre_call_hook(user_api_key_dict=auth, data=body, call_type="pass_through_endpoint")
        held.append(dict(proxy.store.snapshot()))

    assert held[0][VICTIM] == 1.0
    assert held[1] == held[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", EXITS.values(), ids=EXITS.keys())
async def test_a_pass_through_llm_route_reserves_and_settles_like_python(proxy: Proxy, exit_path: Exit) -> None:
    runs: Final[list[Run]] = []  # mutable-ok: one run per side
    for native in (False, True):
        _use_native(proxy, native)
        auth = await _authenticate(proxy, max_budget=1.0, path=PASS_THROUGH)
        body: RequestData = {**BODY, "litellm_call_id": "call-pass-through"}
        _ = await proxy.logging.pre_call_hook(user_api_key_dict=auth, data=body, call_type="pass_through_endpoint")
        reservation = auth.budget_reservation
        assert reservation is not None
        entries = [
            {key: entry[key] for key in ("counter_key", "entity_type", "reserved_cost")}
            for entry in reservation["entries"]  # pyright: ignore[reportGeneralTypeIssues]  # the entries list the reservation carries
        ]
        held = dict(proxy.store.snapshot())
        await exit_path(auth, {**body, "metadata": {"user_api_key_budget_reservation": reservation}})
        runs.append(Run(entries_after_pre_call=entries, held_after_pre_call=held, settled=dict(proxy.store.snapshot())))

    assert runs[0].entries_after_pre_call, (
        "the flag-off run must hold a reservation for this comparison to mean anything"
    )
    assert runs[1] == runs[0]


MESSAGES_BODY: Final = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "hi"}],
}
MESSAGES_REPLY: Final = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 8, "output_tokens": 5},
}


@dataclass(frozen=True, slots=True)
class _Upstream:
    client: httpx.AsyncClient


@dataclass(frozen=True, slots=True)
class HttpRun:
    status: int
    held_upstream: tuple[Mapping[str, float], ...]


def _over_http(proxy: Proxy, native: bool, monkeypatch: pytest.MonkeyPatch, body: Mapping[str, object]) -> HttpRun:
    _use_native(proxy, native)
    asyncio.run(
        cache_key_object(
            hashed_token=TOKEN,
            user_api_key_obj=UserAPIKeyAuth(token=TOKEN, max_budget=1.0, spend=0.0),
            user_api_key_cache=proxy.cache,
            proxy_logging_obj=None,
        )
    )
    proxy.store.set_local(VICTIM, 1.0)
    proxy.store.set_local(KEY_COUNTER, 0.5)
    held: Final[list[Mapping[str, float]]] = []  # mutable-ok: what the upstream saw held

    def upstream(_request: httpx.Request) -> httpx.Response:
        held.append(dict(proxy.store.snapshot()))
        return httpx.Response(200, json=MESSAGES_REPLY)

    monkeypatch.setattr(
        pass_through_endpoints,
        "get_async_httpx_client",
        lambda **_: _Upstream(httpx.AsyncClient(transport=httpx.MockTransport(upstream))),
    )
    response: Final = TestClient(proxy_server.app, raise_server_exceptions=False).post(
        PASS_THROUGH, json=body, headers={"Authorization": f"Bearer {API_KEY}"}
    )
    return HttpRun(status=response.status_code, held_upstream=tuple(held))


FORGED_BODIES: Final[Mapping[str, Mapping[str, object]]] = {
    "plain": MESSAGES_BODY,
    **{
        name: {
            **MESSAGES_BODY,
            "metadata": {"user_api_key_budget_reservation": _forged_plan(counter_key, cost)},
            "litellm_metadata": {"user_api_key_budget_reservation": _forged_plan(counter_key, cost)},
        }
        for name, (counter_key, cost) in FORGERIES.items()
    },
}


@pytest.mark.parametrize("body", FORGED_BODIES.values(), ids=FORGED_BODIES.keys())
def test_the_anthropic_pass_through_route_holds_the_reservation_like_python_over_http(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch, body: Mapping[str, object]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    python: Final = _over_http(proxy, False, monkeypatch, body)
    native: Final = _over_http(proxy, True, monkeypatch, body)

    assert python.status == 200
    assert python.held_upstream[0][KEY_COUNTER] > 0.5, "the flag-off run must hold a reservation upstream"
    assert python.held_upstream[0][VICTIM] == 1.0
    assert native == python


TEAM: Final = "team-at-max"
TEAM_COUNTER: Final = f"spend:team:{TEAM}"


@dataclass(frozen=True, slots=True)
class Refusal:
    status: int
    body: str
    headers: Mapping[str, str]
    failure_hooks: tuple[Mapping[str, object], ...]
    counters: Mapping[str, float]


def _refused_over_http(
    proxy: Proxy, native: bool, monkeypatch: pytest.MonkeyPatch, path: str, body: Mapping[str, object]
) -> Refusal:
    _use_native(proxy, native)

    async def seed() -> None:
        await cache_key_object(
            hashed_token=TOKEN,
            user_api_key_obj=UserAPIKeyAuth(token=TOKEN, team_id=TEAM, spend=0.0),
            user_api_key_cache=proxy.cache,
            proxy_logging_obj=None,
        )
        await cache_team_object(
            team_id=TEAM,
            team_table=LiteLLM_TeamTableCachedObj(team_id=TEAM, max_budget=1.0, spend=0.0),
            user_api_key_cache=proxy.cache,
            proxy_logging_obj=None,
        )

    asyncio.run(seed())
    proxy.store.set_local(TEAM_COUNTER, 1.0)
    hooks: Final[list[Mapping[str, object]]] = []  # mutable-ok: one entry per failure hook call
    original: Final = proxy.logging.post_call_failure_hook

    async def recording(
        request_data: dict[str, object],  # mutable-ok: the hook's own signature
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        error_type: ProxyErrorTypes | None = None,
        route: str | None = None,
        traceback_str: str | None = None,
        dispatched_call_type: str | None = None,
    ) -> HTTPException | None:
        hooks.append(
            {
                "exception": (type(original_exception).__name__, str(original_exception)),
                "llm_provider": getattr(original_exception, "llm_provider", None),
                "error_type": error_type,
                "route": route,
                "dispatched_call_type": dispatched_call_type,
                "identity": (user_api_key_dict.token, user_api_key_dict.team_id, user_api_key_dict.request_route),
                "budget_reservation": user_api_key_dict.budget_reservation,
                "request_keys": sorted(request_data),
            }
        )
        return await original(
            request_data=request_data,
            original_exception=original_exception,
            user_api_key_dict=user_api_key_dict,
            error_type=error_type,
            route=route,
            traceback_str=traceback_str,
            dispatched_call_type=dispatched_call_type,
        )

    monkeypatch.setattr(proxy.logging, "post_call_failure_hook", recording)
    response: Final = TestClient(proxy_server.app, raise_server_exceptions=False).post(
        path, json=body, headers={"Authorization": f"Bearer {API_KEY}"}
    )
    return Refusal(
        status=response.status_code,
        body=response.text,
        headers=dict(response.headers),
        failure_hooks=tuple(hooks),
        counters=dict(proxy.store.snapshot()),
    )


REFUSED_ROUTES: Final[Mapping[str, tuple[str, Mapping[str, object]]]] = {
    "chat_completions": ("/v1/chat/completions", BODY),
    "anthropic_pass_through": (PASS_THROUGH, MESSAGES_BODY),
    "messages": ("/v1/messages", MESSAGES_BODY),
    "responses": ("/v1/responses", {"model": "gpt-4o", "input": "hello", "max_output_tokens": 100}),
    "text_completions": ("/v1/completions", {"model": "gpt-3.5-turbo-instruct", "prompt": "hi", "max_tokens": 10}),
}


@pytest.mark.parametrize("route", REFUSED_ROUTES.values(), ids=REFUSED_ROUTES.keys())
def test_a_reservation_refusal_reaches_the_client_like_the_auth_refusal(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch, route: tuple[str, Mapping[str, object]]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    path, body = route
    python: Final = _refused_over_http(proxy, False, monkeypatch, path, body)
    native: Final = _refused_over_http(proxy, True, monkeypatch, path, body)

    assert python.status == 422
    assert native == python


@dataclass(frozen=True, slots=True)
class _GatedHandle:
    handled: Callable[..., Awaitable[object]] | None
    entered: asyncio.Event
    opened: asyncio.Event

    async def __call__(self, *arguments: object) -> object:
        assert self.handled is not None
        returned: Final = await self.handled(*arguments)
        self.entered.set()
        _ = await self.opened.wait()
        return returned


async def _cancel_during_pre_call(proxy: Proxy, native: bool, monkeypatch: pytest.MonkeyPatch) -> Mapping[str, float]:
    _use_native(proxy, native)
    auth: Final = await _authenticate(proxy, max_budget=1.0)
    data: Final = {**BODY, "litellm_call_id": f"call-cancel-{native}"}
    gate: Final = _GatedHandle(
        getattr(litellm.callbacks[0], "_handled", None) if native else None, asyncio.Event(), asyncio.Event()
    )
    if native:
        monkeypatch.setattr(litellm.callbacks[0], "_handled", gate)
    pre_call: Final = asyncio.ensure_future(
        proxy.logging.pre_call_hook(user_api_key_dict=auth, data=data, call_type="completion")
    )
    if native:
        _ = await gate.entered.wait()
    _ = pre_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pre_call
    release: Final = asyncio.ensure_future(
        budget_reservation.release_budget_reservation_on_cancel(auth.budget_reservation)
    )
    await asyncio.sleep(0)
    gate.opened.set()
    await release
    for _ in range(3):
        await asyncio.sleep(0)
    return dict(proxy.store.snapshot())


@pytest.mark.asyncio
async def test_a_client_that_disconnects_during_pre_call_leaves_no_hold_like_python(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch
) -> None:
    python: Final = await _cancel_during_pre_call(proxy, False, monkeypatch)
    native: Final = await _cancel_during_pre_call(proxy, True, monkeypatch)

    assert python[KEY_COUNTER] > 0.0, "the flag-off run must charge the input cost on cancel"
    assert native == python


async def _admit_twice(proxy: Proxy, native: bool) -> Run:
    _use_native(proxy, native)
    auth, data = await _admit(proxy, f"call-fallback-{native}")
    reservation: Final = auth.budget_reservation
    assert reservation is not None
    _ = await proxy.logging.pre_call_hook(user_api_key_dict=auth, data=dict(data), call_type="completion")
    assert auth.budget_reservation is reservation
    return Run(
        entries_after_pre_call=[
            {key: entry[key] for key in ("counter_key", "entity_type", "reserved_cost")}
            for entry in reservation["entries"]  # pyright: ignore[reportGeneralTypeIssues]  # the entries list the reservation carries
        ],
        held_after_pre_call=dict(proxy.store.snapshot()),
        settled={},
    )


@pytest.mark.asyncio
async def test_a_second_pre_call_for_the_same_auth_reserves_once_like_python(proxy: Proxy) -> None:
    python: Final = await _admit_twice(proxy, native=False)
    native: Final = await _admit_twice(proxy, native=True)

    assert python.entries_after_pre_call, (
        "the flag-off run must hold a reservation for this comparison to mean anything"
    )
    assert native == python
    assert native.held_after_pre_call[KEY_COUNTER] == pytest.approx(
        sum(float(entry["reserved_cost"]) for entry in native.entries_after_pre_call)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownArgumentType, reportUnknownVariableType]  # entries are the dicts built above
    )


class _NextSubscriber(CustomLogger):
    def __init__(self) -> None:
        self.seen: list[tuple[object, object]] = []  # mutable-ok: what each pre_call saw

    async def async_pre_call_hook(
        self, user_api_key_dict: UserAPIKeyAuth, cache: DualCache, data: dict, call_type: CallTypesLiteral
    ) -> None:
        reservation: Final = user_api_key_dict.budget_reservation
        self.seen.append((reservation, None if reservation is None else list(reservation["entries"])))  # pyright: ignore[reportArgumentType]  # the entries list the reservation carries


async def _seen_by_the_next_subscriber(proxy: Proxy, native: bool) -> tuple[object, object]:
    _use_native(proxy, native)
    subscriber: Final = _NextSubscriber()
    litellm.callbacks.append(subscriber)
    auth: Final = await _authenticate(proxy, max_budget=1.0)
    shared: Final = auth.budget_reservation
    assert shared is not None
    _ = await proxy.logging.pre_call_hook(
        user_api_key_dict=auth, data={**BODY, "litellm_call_id": f"call-next-{native}"}, call_type="completion"
    )
    assert len(subscriber.seen) == 1
    reservation, entries = subscriber.seen[0]
    assert reservation is shared
    assert auth.budget_reservation is shared
    return entries, shared["reserved_cost"]


@pytest.mark.asyncio
async def test_the_next_pre_call_subscriber_sees_the_reservation_written_into_the_auth_dict(proxy: Proxy) -> None:
    python: Final = await _seen_by_the_next_subscriber(proxy, native=False)
    native: Final = await _seen_by_the_next_subscriber(proxy, native=True)

    assert python[0], "the flag-off run must hold a reservation before the next subscriber runs"
    assert native == python


@pytest.mark.parametrize("route", REFUSED_ROUTES.values(), ids=REFUSED_ROUTES.keys())
def test_a_runtime_change_to_the_budget_exceeded_status_code_reaches_the_next_refusal_like_python(
    proxy: Proxy, monkeypatch: pytest.MonkeyPatch, route: tuple[str, Mapping[str, object]]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    path, body = route
    refusals: Final[dict[int, tuple[Refusal, Refusal]]] = {}  # mutable-ok: one pair per status code
    for code in (402, 429):
        monkeypatch.setattr(litellm, "budget_exceeded_status_code", code)
        refusals[code] = (
            _refused_over_http(proxy, False, monkeypatch, path, body),
            _refused_over_http(proxy, True, monkeypatch, path, body),
        )

    assert {code: python.status for code, (python, _) in refusals.items()} == {402: 402, 429: 429}
    assert {code: native for code, (_, native) in refusals.items()} == {
        code: python for code, (python, _) in refusals.items()
    }
