from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.events.webhooks.reauth import (
    Authorized,
    Reauthorization,
    Revoked,
    Unavailable,
    _key_revoked,  # pyright: ignore[reportPrivateUsage]  # the key-state rule is the unit under test
    admitted_user,
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
        (
            Authorized(CALLER),
            (server(events=MCPServerEventsConfig(enabled=True, allowed_events=("x.*",))),),
            None,
            Revoked("server"),
        ),
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


_LIVE_USER: Final = UserAPIKeyAuth(user_id="u-1")


async def _returns_user(user_id: str) -> UserAPIKeyAuth:
    return _LIVE_USER


async def _raises_unauthorized(user_id: str) -> UserAPIKeyAuth:
    raise HTTPException(status_code=401, detail="Invalid or expired credential")


async def _raises_unavailable(user_id: str) -> UserAPIKeyAuth:
    raise HTTPException(status_code=503, detail="database unavailable")


async def _raises_unexpected(user_id: str) -> UserAPIKeyAuth:
    raise RuntimeError("connection reset")


@pytest.mark.parametrize(
    ("reload", "expected"),
    (
        (_returns_user, Authorized(_LIVE_USER)),
        (_raises_unauthorized, Revoked("user")),
        (_raises_unavailable, Unavailable("database unavailable")),
        (_raises_unexpected, Unavailable("connection reset")),
    ),
)
async def test_a_keyless_subscription_follows_the_live_user(
    reload: Callable[[str], Awaitable[UserAPIKeyAuth]], expected: Reauthorization
) -> None:
    assert await admitted_user("u-1", reload) == expected
