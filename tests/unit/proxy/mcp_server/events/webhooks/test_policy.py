from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.mcp_server.events.webhooks.policy import (
    MAX_TTL_MS,
    MIN_TTL_MS,
    REQUEST_SCOPED_TTL_MS,
    client_grant,
    credential_outlives_request,
    granted_ttl_ms,
    iso,
    parse_time,
    subscription_id,
)
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.events import SubscribeEventsParams, WebhookDelivery

from ..fakes import server

NOW: Final = datetime(2026, 11, 20, tzinfo=timezone.utc)


def _params(**ttl: int | None) -> SubscribeEventsParams:
    return SubscribeEventsParams(
        name="ci-build.failed", delivery=WebhookDelivery(url="https://hook.example", secret="whsec_x"), **ttl
    )


def test_the_subscription_id_is_derived_from_principal_url_name_and_arguments() -> None:
    base: Final = subscription_id("key:1", "https://a", "ci-x", {"repo": "a"})
    assert base == subscription_id("key:1", "https://a", "ci-x", {"repo": "a"})
    variations: Final = (
        subscription_id("key:2", "https://a", "ci-x", {"repo": "a"}),
        subscription_id("key:1", "https://b", "ci-x", {"repo": "a"}),
        subscription_id("key:1", "https://a", "ci-y", {"repo": "a"}),
        subscription_id("key:1", "https://a", "ci-x", {"repo": "b"}),
    )
    assert base not in variations and len(set(variations)) == 4


@pytest.mark.parametrize(
    ("auth_type", "extra", "outlives"),
    (
        (None, {}, True),
        (MCPAuth.bearer_token, {}, True),
        (MCPAuth.aws_sigv4, {}, True),
        (MCPAuth.oauth2, {"oauth2_flow": "client_credentials"}, True),
        (MCPAuth.oauth2, {}, False),
        (MCPAuth.oauth_delegate, {}, False),
        (MCPAuth.true_passthrough, {}, False),
        (MCPAuth.oauth2_token_exchange, {}, False),
        (MCPAuth.oauth2_id_jag, {}, False),
        (MCPAuth.bearer_token, {"extra_headers": ["X-Team"]}, False),
        (MCPAuth.none, {"is_byok": True}, False),
        (MCPAuth.none, {"oauth_passthrough": True}, False),
    ),
)
def test_only_server_held_credentials_outlive_the_request(
    auth_type: MCPAuth | None, extra: dict[str, object], outlives: bool
) -> None:
    assert credential_outlives_request(server().model_copy(update={"auth_type": auth_type, **extra})) is outlives


@pytest.mark.parametrize(
    ("ttl", "auth_type", "expected"),
    (
        ({}, None, 60 * 60 * 1000),
        ({"ttl_ms": 1}, None, MIN_TTL_MS),
        ({"ttl_ms": 10 * MAX_TTL_MS}, None, MAX_TTL_MS),
        ({"ttl_ms": None}, None, None),
        ({"ttl_ms": None}, MCPAuth.oauth_delegate, REQUEST_SCOPED_TTL_MS),
        ({"ttl_ms": 10 * MAX_TTL_MS}, MCPAuth.oauth_delegate, REQUEST_SCOPED_TTL_MS),
        ({"ttl_ms": 120_000}, MCPAuth.oauth_delegate, 120_000),
    ),
)
def test_granted_ttl(ttl: dict[str, int | None], auth_type: MCPAuth | None, expected: int | None) -> None:
    assert granted_ttl_ms(_params(**ttl), server().model_copy(update={"auth_type": auth_type})) == expected


@pytest.mark.parametrize(
    ("ttl_ms", "upstream", "expected"),
    (
        (60_000, None, NOW + timedelta(minutes=1)),
        (60_000, NOW + timedelta(seconds=30), NOW + timedelta(seconds=30)),
        (None, NOW + timedelta(hours=2), NOW + timedelta(hours=2)),
        (None, None, None),
    ),
)
def test_the_client_grant_never_outlives_the_upstream_grant(
    ttl_ms: int | None, upstream: datetime | None, expected: datetime | None
) -> None:
    assert client_grant(NOW, ttl_ms, upstream) == expected


def test_times_round_trip_as_utc_iso() -> None:
    assert iso(NOW) == "2026-11-20T00:00:00Z"
    assert parse_time("2026-11-20T00:00:00Z") == NOW
    assert parse_time("2026-11-20T00:00:00") == NOW
    assert parse_time("garbage") is None
    assert parse_time(None) is None
