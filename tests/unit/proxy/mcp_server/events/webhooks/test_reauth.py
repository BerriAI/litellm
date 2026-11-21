from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.webhooks.reauth import (
    Authorized,
    Reauthorization,
    Revoked,
    Unavailable,
    _key_revoked,  # pyright: ignore[reportPrivateUsage]  # the key-state rule is the unit under test
    authorize_row,
)
from litellm.proxy.mcp_server.events.webhooks.store import SubscriptionRow
from litellm.types.mcp_server.mcp_server_manager import MCPServer, MCPServerEventsConfig

from ..fakes import runtime, server

ROW: Final = SubscriptionRow(
    id="sub_1",
    principal="key:1",
    key_hash="abc",
    server_id="srv-ci",
    event_name="build.failed",
    client_url="https://client.example/hook",
    client_secret="whsec_x",
    upstream_secret="whsec_y",
    status="active",
)
CALLER: Final = UserAPIKeyAuth(api_key="abc")


def _load(outcome: Reauthorization):  # noqa: ANN202  # test helper
    async def load(_row: SubscriptionRow) -> Reauthorization:
        return outcome

    return load


@pytest.mark.parametrize(
    ("loaded", "servers", "grant", "expected"),
    (
        (Authorized(CALLER), (server(),), None, Authorized(CALLER)),
        (Revoked("key"), (server(),), None, Revoked("key")),
        (Unavailable("db"), (server(),), None, Unavailable("db")),
        (Authorized(CALLER), (), None, Revoked("server")),
        (Authorized(CALLER), (server(events=MCPServerEventsConfig(enabled=False)),), None, Revoked("server")),
        (Authorized(CALLER), (server(events=MCPServerEventsConfig(enabled=True, allowed_events=("x.*",))),), None, Revoked("server")),
        (Authorized(CALLER), (server(),), frozenset({"deploy.done"}), Revoked("event")),
    ),
)
@pytest.mark.asyncio
async def test_reauthorization_outcomes(
    loaded: Reauthorization, servers: tuple[MCPServer, ...], grant: frozenset[str] | None, expected: Reauthorization
) -> None:
    assert await authorize_row(ROW, runtime(servers, grant=grant), _load(loaded)) == expected


@pytest.mark.asyncio
async def test_a_failing_permission_lookup_is_unavailable_not_revoked() -> None:
    base: Final = runtime((server(),))

    async def broken(_: object) -> tuple[MCPServer, ...]:
        raise ConnectionError("db down")

    from dataclasses import replace

    result: Final = await authorize_row(ROW, replace(base, allowed_servers=broken), _load(Authorized(CALLER)))  # pyright: ignore[reportArgumentType]  # test double
    assert isinstance(result, Unavailable)


@pytest.mark.parametrize(
    ("key", "revoked"),
    (
        (UserAPIKeyAuth(api_key="a"), False),
        (UserAPIKeyAuth(api_key="a", blocked=True), True),
        (UserAPIKeyAuth(api_key="a", expires=datetime.now(timezone.utc) - timedelta(minutes=1)), True),
        (UserAPIKeyAuth(api_key="a", expires=datetime.now(timezone.utc) + timedelta(days=1)), False),
    ),
)
def test_blocked_or_expired_keys_are_revoked(key: UserAPIKeyAuth, revoked: bool) -> None:
    assert _key_revoked(key) is revoked
