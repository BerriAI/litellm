from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Final

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.mcp_server.contracts import OperationContext
from litellm.proxy.mcp_server.events.access import EventGrant, principal_of
from litellm.proxy.mcp_server.events.admin import AdminServices, admin_services, router
from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse
from litellm.proxy.mcp_server.events.webhooks.signing import new_secret
from litellm.proxy.mcp_server.events.webhooks.store import SubscriptionRow, SubscriptionStatus
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import CallbackFailure, EventType
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig

DEPLOY_DONE: Final = EventType(name="deploy.done", delivery=("poll",))

from .fakes import BUILD_FAILED, runtime, server
from .fixtures import virtual_key
from .webhooks.memory_store import MemoryStore

ALICE: Final = virtual_key("sk-alice", user_id="alice")
BOB: Final = virtual_key("sk-bob", user_id="bob")
ADMIN: Final = virtual_key("sk-admin", user_role=LitellmUserRoles.PROXY_ADMIN)
VIEWER: Final = virtual_key("sk-viewer", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
SECRET: Final = new_secret()


def _row(subscription_id: str, owner: UserAPIKeyAuth, status: SubscriptionStatus = "active") -> SubscriptionRow:
    return SubscriptionRow(
        id=subscription_id,
        principal=principal_of(owner) or "",
        key_hash=owner.token,
        server_id="srv-ci",
        event_name="build.failed",
        client_url="https://client.example/hook",
        client_secret=SECRET,
        cursor="mcp_state_v1.subscribe-time",
        upstream_secret=new_secret(),
        status=status,
        user_id=owner.user_id,
    )


@dataclass
class Callback:
    failure: CallbackFailure | None = None
    status: int = 200
    sent: list[tuple[str, bytes, Mapping[str, str]]] = field(default_factory=list)  # mutable-ok: records posts

    async def post(
        self, url: str, body: bytes, headers: Mapping[str, str]
    ) -> Result[CallbackResponse, CallbackFailure]:
        self.sent.append((url, body, headers))
        return Error(self.failure) if self.failure is not None else Ok(CallbackResponse(status=self.status, body=b""))


async def _no_alias(_alias: str) -> str | None:
    return None


def _lookup(server_id: str) -> MCPServer | None:
    return server() if server_id == "srv-ci" else None


def _client(caller: UserAPIKeyAuth, store: MemoryStore | None, callback: Callback) -> httpx.AsyncClient:
    app: Final = FastAPI()
    app.include_router(router)
    services: Final = AdminServices(
        store=store, runtime=runtime((server(),)), post=callback.post, lookup=_lookup, key_for_alias=_no_alias
    )
    app.dependency_overrides[user_api_key_auth] = lambda: caller
    app.dependency_overrides[admin_services] = lambda: services
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")


def _store() -> MemoryStore:
    return MemoryStore(rows={"sub_a": _row("sub_a", ALICE), "sub_b": _row("sub_b", BOB)})


@pytest.mark.parametrize(
    ("caller", "visible"),
    ((ALICE, ["sub_a"]), (BOB, ["sub_b"]), (ADMIN, ["sub_a", "sub_b"]), (VIEWER, ["sub_a", "sub_b"])),
)
async def test_callers_list_only_their_own_subscriptions_and_admins_list_all(
    caller: UserAPIKeyAuth, visible: list[str]
) -> None:
    async with _client(caller, _store(), Callback()) as client:
        response = await client.get("/events/subscriptions")
    assert response.status_code == 200
    assert sorted(item["id"] for item in response.json()["subscriptions"]) == visible
    assert {item["callback_host"] for item in response.json()["subscriptions"]} == {"client.example"}


@pytest.mark.parametrize(("server_id", "status", "count"), (("srv-ci", 200, 1), ("srv-other", 404, None)))
async def test_event_types_come_from_servers_the_caller_may_reach(
    server_id: str, status: int, count: int | None
) -> None:
    async with _client(ALICE, None, Callback()) as client:
        response = await client.get("/events/types", params={"server_id": server_id})
    assert response.status_code == status
    if count is not None:
        assert len(response.json()) == count


@pytest.mark.parametrize(
    ("config", "grant", "names"),
    (
        (MCPServerEventsConfig(enabled=True), None, ["build.failed", "deploy.done"]),
        (MCPServerEventsConfig(enabled=True, allowed_events=("build.*",)), None, ["build.failed"]),
        (MCPServerEventsConfig(enabled=True), (frozenset({"deploy.done"}),), ["deploy.done"]),
    ),
)
@pytest.mark.asyncio
async def test_event_types_list_only_what_the_server_publishes_and_the_caller_is_granted(
    config: MCPServerEventsConfig, grant: EventGrant, names: list[str]
) -> None:
    app: Final = FastAPI()
    app.include_router(router)
    events_runtime: Final = runtime((server(events=config),), (BUILD_FAILED, DEPLOY_DONE), grant=grant)
    services: Final = AdminServices(
        store=None, runtime=events_runtime, post=Callback().post, lookup=_lookup, key_for_alias=_no_alias
    )
    app.dependency_overrides[user_api_key_auth] = lambda: ALICE
    app.dependency_overrides[admin_services] = lambda: services
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw") as client:
        response = await client.get("/events/types", params={"server_id": "srv-ci"})
    assert [item["name"] for item in response.json()] == names


@pytest.mark.asyncio
async def test_the_subscription_list_pages_with_a_continuation_offset() -> None:
    store: Final = MemoryStore(rows={f"sub_{index}": _row(f"sub_{index}", ALICE) for index in range(5)})
    async with _client(ADMIN, store, Callback()) as client:
        first = (await client.get("/events/subscriptions", params={"limit": 2})).json()
        last = (await client.get("/events/subscriptions", params={"limit": 2, "offset": 4})).json()
    assert (len(first["subscriptions"]), first["next_offset"]) == (2, 2)
    assert (len(last["subscriptions"]), last["next_offset"]) == (1, None)


@pytest.mark.asyncio
async def test_event_types_filter_servers_by_the_callers_network_address() -> None:
    seen: Final[list[str | None]] = []  # mutable-ok: records the client IP each lookup saw

    async def allowed(context: OperationContext) -> tuple[MCPServer, ...]:
        seen.append(context.client_ip)
        return (server(),)

    app: Final = FastAPI()
    app.include_router(router)
    events_runtime: Final = replace(runtime((server(),)), allowed_servers=allowed)
    services: Final = AdminServices(
        store=None, runtime=events_runtime, post=Callback().post, lookup=_lookup, key_for_alias=_no_alias
    )
    app.dependency_overrides[user_api_key_auth] = lambda: ALICE
    app.dependency_overrides[admin_services] = lambda: services
    transport: Final = httpx.ASGITransport(app=app, client=("203.0.113.9", 4100))
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        await client.get("/events/types", params={"server_id": "srv-ci"})
    assert seen == ["203.0.113.9"]


@pytest.mark.asyncio
async def test_a_key_alias_filter_from_a_non_admin_is_ignored_and_shows_their_own_rows() -> None:
    store: Final = MemoryStore(rows={"sub_1": _row("sub_1", ALICE)})
    async with _client(ALICE, store, Callback()) as client:
        listed = (await client.get("/events/subscriptions", params={"key_alias": "someone-else"})).json()
    assert [item["id"] for item in listed["subscriptions"]] == ["sub_1"]


def test_keys_limited_to_mcp_routes_reach_the_event_subscription_endpoints() -> None:
    from litellm.proxy._types import LiteLLMRoutes
    from litellm.proxy.auth.route_checks import RouteChecks

    for route in ("/v1/mcp/events/types", "/v1/mcp/events/subscriptions", "/v1/mcp/events/subscriptions/sub_1/test"):
        assert RouteChecks.check_route_access(route=route, allowed_routes=LiteLLMRoutes.mcp_routes.value)


@pytest.mark.asyncio
async def test_admin_catalog_removes_webhook_mode_when_there_is_no_public_origin() -> None:
    async with _client(ALICE, None, Callback()) as client:
        response = await client.get("/events/types", params={"server_id": "srv-ci"})
    assert response.status_code == 200
    assert response.json()[0]["delivery"] == ["poll", "push"]


@pytest.mark.parametrize(("with_store", "modes"), ((False, ["poll", "push"]), (True, ["poll", "push", "webhook"])))
async def test_catalog_requires_storage_to_offer_webhooks(
    monkeypatch: pytest.MonkeyPatch, with_store: bool, modes: list[str]
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setitem(proxy_server.general_settings, "mcp_events_public_base_url", "https://gateway.example")
    async with _client(ALICE, _store() if with_store else None, Callback()) as client:
        response = await client.get("/events/types", params={"server_id": "srv-ci"})
    assert response.status_code == 200 and response.json()[0]["delivery"] == modes
