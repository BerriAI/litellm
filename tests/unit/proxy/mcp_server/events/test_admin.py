import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.mcp_server.events.access import principal_of
from litellm.proxy.mcp_server.events.admin import AdminServices, admin_services, router
from litellm.proxy.mcp_server.events.webhooks.delivery import CallbackResponse
from litellm.proxy.mcp_server.events.webhooks.signing import new_secret, parse_secret, verify
from litellm.proxy.mcp_server.events.webhooks.store import SubscriptionRow, SubscriptionStatus
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok, Result
from litellm.types.mcp_server.events import CallbackFailure
from litellm.types.mcp_server.mcp_server_manager import MCPServer

from .fakes import runtime, server
from .webhooks.memory_store import MemoryStore

ALICE: Final = UserAPIKeyAuth(api_key="sk-alice", user_id="alice")
BOB: Final = UserAPIKeyAuth(api_key="sk-bob", user_id="bob")
ADMIN: Final = UserAPIKeyAuth(api_key="sk-admin", user_role=LitellmUserRoles.PROXY_ADMIN)
VIEWER: Final = UserAPIKeyAuth(api_key="sk-viewer", user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
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


def _lookup(server_id: str) -> MCPServer | None:
    return server() if server_id == "srv-ci" else None


def _client(caller: UserAPIKeyAuth, store: MemoryStore | None, callback: Callback) -> httpx.AsyncClient:
    app: Final = FastAPI()
    app.include_router(router)
    services: Final = AdminServices(store=store, runtime=runtime((server(),)), post=callback.post, lookup=_lookup)
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


@pytest.mark.parametrize(
    ("caller", "status"),
    ((ALICE, 200), (ADMIN, 200), (BOB, 404), (VIEWER, 403)),
)
async def test_only_the_owner_or_an_admin_terminates_a_subscription(caller: UserAPIKeyAuth, status: int) -> None:
    store: Final = _store()
    callback: Final = Callback()
    async with _client(caller, store, callback) as client:
        response = await client.delete("/events/subscriptions/sub_a")
    assert response.status_code == status
    terminated: Final = status == 200
    assert (store.rows["sub_a"].status == "terminated") is terminated
    assert [json.loads(body)["type"] for _, body, _ in callback.sent] == (["terminated"] if terminated else [])


async def test_a_test_event_is_signed_with_the_subscription_secret_and_carries_the_body_data() -> None:
    callback: Final = Callback()
    async with _client(ALICE, _store(), callback) as client:
        response = await client.post("/events/subscriptions/sub_a/test", json={"data": {"build": 7}})
    assert response.json() == {"delivered": True, "status": 200, "error": None}
    url, body, headers = callback.sent[0]
    secret: Final = parse_secret(SECRET)
    assert secret is not None
    assert url == "https://client.example/hook"
    assert verify((secret,), headers, body, int(time.time()))
    assert json.loads(body)["data"] == {"build": 7}
    assert json.loads(body)["name"] == "ci-build.failed"


async def test_a_test_event_without_a_body_sends_the_default_payload() -> None:
    callback: Final = Callback()
    async with _client(ALICE, _store(), callback) as client:
        response = await client.post("/events/subscriptions/sub_a/test")
    assert response.status_code == 200
    assert json.loads(callback.sent[0][1])["data"] == {"test": True}


@pytest.mark.parametrize(
    ("callback", "expected"),
    (
        (Callback(status=500), {"delivered": False, "status": 500, "error": None}),
        (Callback(failure="timeout"), {"delivered": False, "status": None, "error": "timeout"}),
    ),
)
async def test_a_failing_callback_is_reported_not_raised(callback: Callback, expected: dict[str, object]) -> None:
    async with _client(ALICE, _store(), callback) as client:
        response = await client.post("/events/subscriptions/sub_a/test")
    assert response.json() == expected


@pytest.mark.parametrize(
    ("caller", "row_status", "status"),
    ((ALICE, "terminated", 409), (ALICE, "pending", 409), (BOB, "active", 404), (VIEWER, "active", 403)),
)
async def test_test_events_need_an_active_subscription_the_caller_may_manage(
    caller: UserAPIKeyAuth, row_status: SubscriptionStatus, status: int
) -> None:
    callback: Final = Callback()
    store: Final = MemoryStore(rows={"sub_a": _row("sub_a", ALICE, row_status)})
    async with _client(caller, store, callback) as client:
        response = await client.post("/events/subscriptions/sub_a/test")
    assert response.status_code == status
    assert callback.sent == []


@pytest.mark.parametrize(
    ("method", "path"),
    (
        ("GET", "/events/subscriptions"),
        ("DELETE", "/events/subscriptions/sub_a"),
        ("POST", "/events/subscriptions/sub_a/test"),
    ),
)
async def test_subscription_routes_need_a_database(method: str, path: str) -> None:
    async with _client(ADMIN, None, Callback()) as client:
        response = await client.request(method, path)
    assert response.status_code == 503


@pytest.mark.parametrize(("server_id", "status", "count"), (("srv-ci", 200, 1), ("srv-other", 404, None)))
async def test_event_types_come_from_servers_the_caller_may_reach(
    server_id: str, status: int, count: int | None
) -> None:
    async with _client(ALICE, None, Callback()) as client:
        response = await client.get("/events/types", params={"server_id": server_id})
    assert response.status_code == status
    if count is not None:
        assert len(response.json()) == count
