"""Tests for the client_credentials (M2M) token source and its retrying bearer auth.

These are the behavior-contract spec: grant shape (scopes / audience / client auth method),
rotation-aware cache keying, expires_in-driven expiry, error classification, and the
401 -> discard -> refetch -> retry-once recovery in ``ClientCredentialsBearerAuth``.
"""

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.client_credentials import (
    ClientCredentialsBearerAuth,
    ClientCredentialsTokenSource,
    TokenEndpointDenied,
    TokenEndpointOutcome,
    TokenEndpointSuccess,
    TokenEndpointUnreachable,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.result import Error, Ok
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ClientCredentialsConfig,
)


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class _FakePoster:
    """Records every grant POST and returns canned outcomes (last one repeats)."""

    def __init__(self, outcomes: "list[TokenEndpointOutcome]") -> None:
        self._outcomes = outcomes
        self.calls: "list[tuple[str, dict[str, str], dict[str, str]]]" = []

    async def __call__(self, url: str, form: "dict[str, str]", headers: "dict[str, str]") -> TokenEndpointOutcome:
        self.calls.append((url, dict(form), dict(headers)))
        index = min(len(self.calls) - 1, len(self._outcomes) - 1)
        return self._outcomes[index]


def _success(access_token: str = "m2m-token", **extra: object) -> TokenEndpointSuccess:
    return TokenEndpointSuccess(body={"access_token": access_token, **extra})


def _config(**overrides: object) -> ClientCredentialsConfig:
    fields: "dict[str, object]" = {
        "client_id": "cid",
        "client_secret": SecretStr("csec"),
        "token_url": "https://idp.example.com/token",
        **overrides,
    }
    return ClientCredentialsConfig.model_validate(fields)


@pytest.mark.asyncio
async def test_grant_posts_client_credentials_with_scopes_and_audience():
    poster = _FakePoster([_success()])
    source = ClientCredentialsTokenSource(poster)
    result = await source.get("s", _config(scopes=("read", "write"), audience="https://api.example.com"))
    assert isinstance(result, Ok)
    assert result.ok.access_token == "m2m-token"
    url, form, _headers = poster.calls[0]
    assert url == "https://idp.example.com/token"
    assert form["grant_type"] == "client_credentials"
    assert form["scope"] == "read write"
    assert form["audience"] == "https://api.example.com"
    assert form["client_id"] == "cid"
    assert form["client_secret"] == "csec"


@pytest.mark.asyncio
async def test_grant_omits_scope_and_audience_when_not_configured():
    poster = _FakePoster([_success()])
    await ClientCredentialsTokenSource(poster).get("s", _config())
    _url, form, _headers = poster.calls[0]
    assert "scope" not in form
    assert "audience" not in form


@pytest.mark.asyncio
async def test_grant_honors_client_secret_basic():
    poster = _FakePoster([_success()])
    await ClientCredentialsTokenSource(poster).get("s", _config(token_endpoint_auth_method="client_secret_basic"))
    _url, form, headers = poster.calls[0]
    assert headers["Authorization"].startswith("Basic ")
    assert "client_secret" not in form
    assert "client_id" not in form


@pytest.mark.asyncio
async def test_missing_grant_fields_are_misconfigured_and_never_posted():
    poster = _FakePoster([_success()])
    result = await ClientCredentialsTokenSource(poster).get(
        "s", ClientCredentialsConfig(client_id="cid", client_secret=SecretStr("csec"))
    )
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"
    assert "token_url" in result.error.summary
    assert poster.calls == []


@pytest.mark.asyncio
async def test_token_is_cached_across_gets():
    poster = _FakePoster([_success(expires_in=3600)])
    source = ClientCredentialsTokenSource(poster)
    first = await source.get("s", _config())
    second = await source.get("s", _config())
    assert isinstance(first, Ok) and isinstance(second, Ok)
    assert second.ok.access_token == first.ok.access_token
    assert len(poster.calls) == 1


@pytest.mark.asyncio
async def test_expires_in_bounds_the_cache_lifetime():
    clock = _Clock(1000.0)
    poster = _FakePoster([_success("t1", expires_in=120), _success("t2", expires_in=120)])
    source = ClientCredentialsTokenSource(poster, expiry_skew_seconds=60.0, clock=clock)
    first = await source.get("s", _config())
    assert isinstance(first, Ok)
    assert first.ok.expires_at == 1120.0
    clock.t = 1059.0  # within expires_in - skew
    assert len(poster.calls) == 1
    within = await source.get("s", _config())
    assert isinstance(within, Ok) and within.ok.access_token == "t1"
    clock.t = 1061.0  # past expires_in - skew: the entry lapsed before the real token does
    lapsed = await source.get("s", _config())
    assert isinstance(lapsed, Ok) and lapsed.ok.access_token == "t2"
    assert len(poster.calls) == 2


@pytest.mark.asyncio
async def test_short_lived_token_is_never_served_past_its_expiry():
    # expires_in below the skew must not be floored into serving an expired token: the cache
    # entry lapses with the token itself, and the next get re-fetches.
    clock = _Clock(1000.0)
    poster = _FakePoster([_success("t1", expires_in=5), _success("t2", expires_in=5)])
    source = ClientCredentialsTokenSource(poster, expiry_skew_seconds=60.0, min_cache_seconds=10.0, clock=clock)
    first = await source.get("s", _config())
    assert isinstance(first, Ok) and first.ok.access_token == "t1"
    clock.t = 1004.0  # still within the token's real lifetime
    within = await source.get("s", _config())
    assert isinstance(within, Ok) and within.ok.access_token == "t1"
    clock.t = 1006.0  # past expires_at: the floor must not keep serving t1
    lapsed = await source.get("s", _config())
    assert isinstance(lapsed, Ok) and lapsed.ok.access_token == "t2"
    assert len(poster.calls) == 2


class _RecordingBackend:
    """A TokenCacheBackend spy: records every write so a test can assert none happened."""

    def __init__(self) -> None:
        self.set_ttls: list[float] = []

    async def get(self, identity_key: str, server_id: str):
        return None

    async def set(self, identity_key: str, server_id: str, token, ttl_seconds: float) -> None:
        self.set_ttls.append(ttl_seconds)

    async def delete(self, identity_key: str, server_id: str) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_in", [0, -30])
async def test_non_positive_expires_in_writes_no_cache_entry(expires_in):
    # A dead-on-arrival entry (ttl 0) must not be written at all: it can never be served, but it
    # would occupy a slot in the bounded backend and could evict a live token. The mint itself
    # still succeeds for the current request, and the next get re-fetches.
    backend = _RecordingBackend()
    poster = _FakePoster([_success("t1", expires_in=expires_in), _success("t2", expires_in=expires_in)])
    source = ClientCredentialsTokenSource(poster, backend=backend)
    first = await source.get("s", _config())
    assert isinstance(first, Ok) and first.ok.access_token == "t1"
    again = await source.get("s", _config())
    assert isinstance(again, Ok) and again.ok.access_token == "t2"
    assert backend.set_ttls == []
    assert len(poster.calls) == 2


@pytest.mark.asyncio
async def test_lock_dict_is_bounded_for_ephemeral_server_ids():
    poster = _FakePoster([_success()])
    source = ClientCredentialsTokenSource(poster, max_locks=8)
    for index in range(20):
        result = await source.get(f"ephemeral-{index}", _config())
        assert isinstance(result, Ok)
    assert len(source._locks) <= 8


@pytest.mark.asyncio
async def test_missing_expires_in_is_cached_briefly_not_an_hour():
    clock = _Clock(1000.0)
    poster = _FakePoster([_success("t1"), _success("t2")])
    source = ClientCredentialsTokenSource(poster, default_ttl_seconds=300.0, clock=clock)
    first = await source.get("s", _config())
    assert isinstance(first, Ok)
    assert first.ok.expires_at is None
    clock.t = 1301.0  # past the default TTL; v1 would still be serving its 3600s-cached token
    second = await source.get("s", _config())
    assert isinstance(second, Ok) and second.ok.access_token == "t2"
    assert len(poster.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rotation",
    [
        {"client_secret": SecretStr("rotated")},
        {"client_id": "cid-2"},
        {"scopes": ("admin",)},
        {"audience": "https://other.example.com"},
        {"token_url": "https://idp2.example.com/token"},
    ],
)
async def test_credential_rotation_invalidates_the_cached_token(rotation):
    poster = _FakePoster([_success("old", expires_in=3600), _success("new", expires_in=3600)])
    source = ClientCredentialsTokenSource(poster)
    before = await source.get("s", _config())
    after = await source.get("s", _config(**rotation))
    assert isinstance(before, Ok) and before.ok.access_token == "old"
    assert isinstance(after, Ok) and after.ok.access_token == "new"
    assert len(poster.calls) == 2


@pytest.mark.asyncio
async def test_idp_4xx_is_misconfigured_and_5xx_is_unavailable():
    denied = await ClientCredentialsTokenSource(
        _FakePoster([TokenEndpointDenied(status_code=401, detail="HTTP 401")])
    ).get("s", _config())
    assert isinstance(denied, Error) and denied.error.tag == "misconfigured"
    down = await ClientCredentialsTokenSource(
        _FakePoster([TokenEndpointDenied(status_code=503, detail="HTTP 503")])
    ).get("s", _config())
    assert isinstance(down, Error) and down.error.tag == "upstream_unavailable"
    unreachable = await ClientCredentialsTokenSource(_FakePoster([TokenEndpointUnreachable(detail="dns")])).get(
        "s", _config()
    )
    assert isinstance(unreachable, Error) and unreachable.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_response_without_access_token_is_misconfigured():
    poster = _FakePoster([TokenEndpointSuccess(body={"token_type": "Bearer"})])
    result = await ClientCredentialsTokenSource(poster).get("s", _config())
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


@pytest.mark.asyncio
async def test_error_results_are_not_cached():
    poster = _FakePoster([TokenEndpointUnreachable(detail="down"), _success("recovered")])
    source = ClientCredentialsTokenSource(poster)
    first = await source.get("s", _config())
    second = await source.get("s", _config())
    assert isinstance(first, Error)
    assert isinstance(second, Ok) and second.ok.access_token == "recovered"


@pytest.mark.asyncio
async def test_refetch_discards_the_failed_token_and_mints_a_fresh_one():
    poster = _FakePoster([_success("stale", expires_in=3600), _success("fresh", expires_in=3600)])
    source = ClientCredentialsTokenSource(poster)
    first = await source.get("s", _config())
    assert isinstance(first, Ok)
    fresh = await source.refetch("s", _config(), failed_access_token="stale")
    assert fresh == "fresh"
    assert len(poster.calls) == 2
    after = await source.get("s", _config())
    assert isinstance(after, Ok) and after.ok.access_token == "fresh"
    assert len(poster.calls) == 2


@pytest.mark.asyncio
async def test_refetch_reuses_a_concurrent_replacement_without_a_second_grant():
    poster = _FakePoster([_success("replacement", expires_in=3600)])
    source = ClientCredentialsTokenSource(poster)
    seeded = await source.get("s", _config())
    assert isinstance(seeded, Ok)
    result = await source.refetch("s", _config(), failed_access_token="some-older-token")
    assert result == "replacement"
    assert len(poster.calls) == 1


@pytest.mark.asyncio
async def test_refetch_returns_none_when_the_grant_fails():
    poster = _FakePoster([_success("stale"), TokenEndpointUnreachable(detail="down")])
    source = ClientCredentialsTokenSource(poster)
    await source.get("s", _config())
    assert await source.refetch("s", _config(), failed_access_token="stale") is None


def _upstream(responses: "list[httpx.Response]") -> "tuple[httpx.MockTransport, list[str]]":
    # The auth flow re-yields the same Request object on retry, so snapshot the Authorization
    # value per send; holding the Request would show the post-retry mutation for both entries.
    seen: "list[str]" = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization", ""))
        return responses[min(len(seen) - 1, len(responses) - 1)]

    return httpx.MockTransport(handler), seen


@pytest.mark.asyncio
async def test_bearer_auth_sends_the_token_and_leaves_a_success_alone():
    transport, seen = _upstream([httpx.Response(200)])

    async def refetch(failed: str) -> "str | None":
        raise AssertionError("must not refetch on success")

    auth = ClientCredentialsBearerAuth("m2m-token", refetch)
    async with httpx.AsyncClient(transport=transport, auth=auth) as client:
        response = await client.get("https://upstream.example.com/mcp")
    assert response.status_code == 200
    assert seen == ["Bearer m2m-token"]


@pytest.mark.asyncio
async def test_bearer_auth_retries_a_401_once_with_a_fresh_token():
    transport, seen = _upstream([httpx.Response(401), httpx.Response(200)])
    refetched: "list[str]" = []

    async def refetch(failed: str) -> "str | None":
        refetched.append(failed)
        return "fresh-token"

    auth = ClientCredentialsBearerAuth("stale-token", refetch)
    async with httpx.AsyncClient(transport=transport, auth=auth) as client:
        response = await client.get("https://upstream.example.com/mcp")
    assert response.status_code == 200
    assert refetched == ["stale-token"]
    assert seen == ["Bearer stale-token", "Bearer fresh-token"]


@pytest.mark.asyncio
async def test_bearer_auth_surfaces_the_401_when_the_refetch_fails():
    transport, seen = _upstream([httpx.Response(401)])

    async def refetch(failed: str) -> "str | None":
        return None

    auth = ClientCredentialsBearerAuth("stale-token", refetch)
    async with httpx.AsyncClient(transport=transport, auth=auth) as client:
        response = await client.get("https://upstream.example.com/mcp")
    assert response.status_code == 401
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_bearer_auth_gives_up_after_a_second_401():
    transport, seen = _upstream([httpx.Response(401), httpx.Response(401)])
    refetched: "list[str]" = []

    async def refetch(failed: str) -> "str | None":
        refetched.append(failed)
        return "fresh-token"

    auth = ClientCredentialsBearerAuth("stale-token", refetch)
    async with httpx.AsyncClient(transport=transport, auth=auth) as client:
        response = await client.get("https://upstream.example.com/mcp")
    assert response.status_code == 401
    assert len(seen) == 2
    assert refetched == ["stale-token"]


def test_bearer_auth_rejects_sync_clients():
    async def refetch(failed: str) -> "str | None":
        return None

    auth = ClientCredentialsBearerAuth("token", refetch)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)), auth=auth) as client:
        with pytest.raises(RuntimeError):
            client.get("https://upstream.example.com/mcp")
