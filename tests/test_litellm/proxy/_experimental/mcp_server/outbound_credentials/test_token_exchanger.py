"""Tests for the pure RFC 8693 token exchanger: the OBO swap, caching, and single-flight.

Drives `Rfc8693TokenExchanger` through an injected fake HTTP post and clock, so the exchange, the
form it sends, the per-caller-token cache, and the failure mapping are pinned without a live IdP.
"""

import asyncio

import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials import (
    Error,
    Ok,
    ServerSpec,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
    Rfc8693TokenExchanger,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    TokenExchangeConfig,
)

_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"

_CONFIG = TokenExchangeConfig(
    token_exchange_endpoint="https://idp.example.com/token",
    audience="https://up.example.com",
    client_id="cid",
    client_secret=SecretStr("csec"),
    scopes=("s1", "s2"),
)
_SERVER = ServerSpec(server_id="srv", resource="https://up.example.com", config=_CONFIG)


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _RecordingPost:
    def __init__(self, body: dict[str, object] | None) -> None:
        self._body = body
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.headers: list[dict[str, str]] = []

    async def __call__(self, url: str, form: dict[str, str], headers: dict[str, str]) -> dict[str, object] | None:
        self.calls.append((url, dict(form)))
        self.headers.append(dict(headers))
        return self._body


def _spec(config: TokenExchangeConfig) -> ServerSpec:
    return ServerSpec(server_id="srv", resource="https://up.example.com", config=config)


@pytest.mark.asyncio
async def test_exchange_emits_token_and_sends_rfc8693_form():
    post = _RecordingPost({"access_token": "exchanged", "expires_in": 3600})
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("caller-jwt", _SERVER, _CONFIG)
    assert isinstance(result, Ok)
    assert result.ok.access_token == "exchanged"
    url, form = post.calls[0]
    assert url == "https://idp.example.com/token"
    assert form == {
        "grant_type": _GRANT,
        "subject_token": "caller-jwt",
        "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        "client_id": "cid",
        "client_secret": "csec",
        "audience": "https://up.example.com",
        "scope": "s1 s2",
    }
    # client_secret_post is the default: creds in the body, no client-auth header
    assert post.headers[0] == {}


@pytest.mark.asyncio
async def test_client_secret_basic_sends_authorization_header_and_omits_body_creds():
    import base64

    config = TokenExchangeConfig(
        token_exchange_endpoint="https://idp.example.com/token",
        client_id="cid",
        client_secret=SecretStr("csec"),
        token_endpoint_auth_method="client_secret_basic",
    )
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _spec(config), config)
    assert isinstance(result, Ok)
    _, form = post.calls[0]
    assert "client_id" not in form and "client_secret" not in form
    assert post.headers[0]["Authorization"] == "Basic " + base64.b64encode(b"cid:csec").decode()


@pytest.mark.asyncio
async def test_client_secret_post_keeps_creds_in_body_with_no_auth_header():
    config = TokenExchangeConfig(
        token_exchange_endpoint="https://idp.example.com/token",
        client_id="cid",
        client_secret=SecretStr("csec"),
        token_endpoint_auth_method="client_secret_post",
    )
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _spec(config), config)
    _, form = post.calls[0]
    assert form["client_id"] == "cid" and form["client_secret"] == "csec"
    assert "Authorization" not in post.headers[0]


@pytest.mark.asyncio
async def test_exchange_maps_idp_rejection_to_unauthorized():
    """An IdP 4xx (surfaced as SubjectTokenRejected by the post adapter) is non-retryable: it maps
    to ``unauthorized`` (the 401 OBO challenge), not the retryable ``upstream_unavailable`` (503)."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
        SubjectTokenRejected,
    )

    async def _rejecting_post(url, form, headers):
        raise SubjectTokenRejected("IdP rejected the token exchange (HTTP 400)")

    result = await Rfc8693TokenExchanger(_rejecting_post, clock=_Clock()).exchange("bad-jwt", _SERVER, _CONFIG)
    assert isinstance(result, Error)
    assert result.error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_exchange_maps_gateway_fault_to_misconfigured():
    """A gateway-fault RFC 6749 code (invalid_client / invalid_target / ..., surfaced as
    TokenExchangeClientError) is the gateway's problem, not the caller's, so it maps to misconfigured
    (500) rather than the retryable 503 or the 401 OBO challenge the caller can't act on."""
    from litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger import (
        TokenExchangeClientError,
    )

    async def _client_error_post(url, form, headers):
        raise TokenExchangeClientError("invalid_client")

    result = await Rfc8693TokenExchanger(_client_error_post, clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


@pytest.mark.asyncio
async def test_exchange_maps_transport_failure_to_upstream_unavailable():
    """A post returning None (5xx / network / timeout / malformed body) stays retryable: 503."""
    result = await Rfc8693TokenExchanger(_RecordingPost(None), clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_exchange_caches_per_caller_token():
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    exchanger = Rfc8693TokenExchanger(post, clock=_Clock())
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    second = await exchanger.exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(second, Ok) and second.ok.access_token == "x"
    assert len(post.calls) == 1


@pytest.mark.asyncio
async def test_rotated_caller_token_re_exchanges():
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    exchanger = Rfc8693TokenExchanger(post, clock=_Clock())
    await exchanger.exchange("jwt-1", _SERVER, _CONFIG)
    await exchanger.exchange("jwt-2", _SERVER, _CONFIG)
    assert len(post.calls) == 2


@pytest.mark.asyncio
async def test_same_token_different_tenant_does_not_share_cache():
    # Two tenants presenting the same opaque token (e.g. a shared/service token) must not collide on
    # one cache entry: tenant_id is part of the key, so each tenant gets its own exchange.
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    exchanger = Rfc8693TokenExchanger(post, clock=_Clock())
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="acme")
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="globex")
    assert len(post.calls) == 2
    # Same tenant + token still hits the cache.
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="acme")
    assert len(post.calls) == 2


@pytest.mark.asyncio
async def test_invalidate_forces_re_exchange():
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    exchanger = Rfc8693TokenExchanger(post, clock=_Clock())
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="acme")
    await exchanger.invalidate("jwt", _SERVER, _CONFIG, tenant_id="acme")
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="acme")
    assert len(post.calls) == 2


@pytest.mark.asyncio
async def test_invalidate_targets_only_the_matching_tenant():
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    exchanger = Rfc8693TokenExchanger(post, clock=_Clock())
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="acme")
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="globex")
    await exchanger.invalidate("jwt", _SERVER, _CONFIG, tenant_id="acme")
    # globex's entry survives; only acme re-exchanges.
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="globex")
    await exchanger.exchange("jwt", _SERVER, _CONFIG, tenant_id="acme")
    assert len(post.calls) == 3


@pytest.mark.asyncio
async def test_rotated_config_re_exchanges_before_ttl():
    # Same caller token + server, but the operator rotated the audience/scope: the cached token was
    # minted for the old config, so it must re-exchange (not serve the stale token) before TTL.
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    exchanger = Rfc8693TokenExchanger(post, clock=_Clock())
    rotated = _CONFIG.model_copy(update={"audience": "https://new.example.com", "scopes": ("s3",)})
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    await exchanger.exchange("jwt", _SERVER, rotated)
    assert len(post.calls) == 2
    _, second_form = post.calls[1]
    assert second_form["audience"] == "https://new.example.com"
    assert second_form["scope"] == "s3"


@pytest.mark.asyncio
async def test_rotated_token_endpoint_auth_method_re_exchanges_before_ttl():
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    exchanger = Rfc8693TokenExchanger(post, clock=_Clock())
    rotated = _CONFIG.model_copy(update={"token_endpoint_auth_method": "client_secret_basic"})
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    await exchanger.exchange("jwt", _SERVER, rotated)
    assert len(post.calls) == 2
    _, second_form = post.calls[1]
    assert "client_id" not in second_form
    assert "client_secret" not in second_form
    assert "Authorization" in post.headers[1]


@pytest.mark.asyncio
async def test_concurrent_callers_single_flight_one_exchange():
    release = asyncio.Event()

    class _Blocking:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, url, form, headers):
            self.calls += 1
            await release.wait()
            return {"access_token": "x", "expires_in": 3600}

    post = _Blocking()
    exchanger = Rfc8693TokenExchanger(post, clock=_Clock())
    first = asyncio.create_task(exchanger.exchange("jwt", _SERVER, _CONFIG))
    second = asyncio.create_task(exchanger.exchange("jwt", _SERVER, _CONFIG))
    await asyncio.sleep(0.02)
    release.set()
    r1, r2 = await asyncio.gather(first, second)
    assert post.calls == 1
    assert isinstance(r1, Ok) and isinstance(r2, Ok)


@pytest.mark.asyncio
async def test_idp_failure_is_upstream_unavailable():
    result = await Rfc8693TokenExchanger(_RecordingPost(None), clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_missing_access_token_is_upstream_unavailable():
    post = _RecordingPost({"token_type": "Bearer"})
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("token_type", ["N_A", "n_a", "DPoP", "mac"])
async def test_non_bearer_token_type_is_refused(token_type):
    # RFC 8693 2.2.1: the resolver forwards the exchanged token as `Bearer`. A non-Bearer token_type
    # (e.g. N_A = not a standalone access token) must fail closed, not be minted as a bogus Bearer.
    post = _RecordingPost({"access_token": "x", "token_type": token_type, "expires_in": 3600})
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_non_bearer_token_type_is_logged():
    from unittest.mock import patch

    post = _RecordingPost({"access_token": "x", "token_type": "N_A", "expires_in": 3600})
    with patch(
        "litellm.proxy._experimental.mcp_server.outbound_credentials.token_exchanger.verbose_logger"
    ) as mock_logger:
        await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert mock_logger.warning.called
    assert "N_A" in repr(mock_logger.warning.call_args)


@pytest.mark.asyncio
@pytest.mark.parametrize("token_type", ["Bearer", "bearer", "BEARER"])
async def test_bearer_token_type_is_accepted_case_insensitively(token_type):
    post = _RecordingPost({"access_token": "x", "token_type": token_type, "expires_in": 3600})
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Ok) and result.ok.access_token == "x"


@pytest.mark.asyncio
async def test_absent_token_type_defaults_to_bearer():
    # Many IdPs omit token_type; absence must not fail the exchange (RFC 6750 default is Bearer).
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Ok) and result.ok.access_token == "x"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "issued_token_type",
    [
        "urn:ietf:params:oauth:token-type:refresh_token",
        "urn:ietf:params:oauth:token-type:id_token",
        "urn:ietf:params:oauth:token-type:saml2",
    ],
)
async def test_non_access_issued_token_type_is_refused_even_when_bearer(issued_token_type):
    # A malformed STS could mint a refresh/id/saml token but label it Bearer; issued_token_type must
    # still fail it closed rather than forward a non-access token as an upstream access credential.
    post = _RecordingPost(
        {"access_token": "x", "token_type": "Bearer", "issued_token_type": issued_token_type, "expires_in": 3600}
    )
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "issued_token_type",
    ["urn:ietf:params:oauth:token-type:access_token", "urn:ietf:params:oauth:token-type:jwt", "custom-unknown", None],
)
async def test_access_or_unknown_issued_token_type_is_accepted(issued_token_type):
    # access_token / jwt are usable; an absent or unrecognized type is accepted (lenient), so real
    # IdPs that omit issued_token_type or use a custom URN keep working.
    body: dict[str, object] = {"access_token": "x", "token_type": "Bearer", "expires_in": 3600}
    if issued_token_type is not None:
        body["issued_token_type"] = issued_token_type
    result = await Rfc8693TokenExchanger(_RecordingPost(body), clock=_Clock()).exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Ok) and result.ok.access_token == "x"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        TokenExchangeConfig(token_exchange_endpoint="https://idp/token", client_secret=SecretStr("s")),
        TokenExchangeConfig(token_exchange_endpoint="https://idp/token", client_id="c"),
    ],
)
async def test_incomplete_config_is_misconfigured_without_hitting_idp(config):
    post = _RecordingPost({"access_token": "x"})
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _spec(config), config)
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"
    assert post.calls == []


@pytest.mark.asyncio
async def test_missing_endpoint_is_precondition_required_without_hitting_idp():
    # No endpoint configured (and none discoverable): fail closed with a 412-mapped precondition
    # rather than guessing an IdP or falling back. The subject token is never POSTed anywhere.
    config = TokenExchangeConfig(client_id="c", client_secret=SecretStr("s"))
    post = _RecordingPost({"access_token": "x"})
    result = await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _spec(config), config)
    assert isinstance(result, Error)
    assert result.error.tag == "precondition_required"
    assert post.calls == []


@pytest.mark.asyncio
async def test_cached_token_expires_after_its_ttl():
    clock = _Clock(1000.0)
    # expires_in=120, buffer=60 -> ttl 60 -> cached until t=1060.
    post = _RecordingPost({"access_token": "x", "expires_in": 120})
    exchanger = Rfc8693TokenExchanger(post, clock=clock)
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    clock.now = 1059.0
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    assert len(post.calls) == 1
    clock.now = 1061.0
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    assert len(post.calls) == 2


@pytest.mark.asyncio
async def test_audience_and_scope_omitted_when_unset():
    config = TokenExchangeConfig(
        token_exchange_endpoint="https://idp/token",
        client_id="cid",
        client_secret=SecretStr("csec"),
    )
    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    await Rfc8693TokenExchanger(post, clock=_Clock()).exchange("jwt", _spec(config), config)
    _, form = post.calls[0]
    assert "audience" not in form
    assert "scope" not in form


@pytest.mark.asyncio
@pytest.mark.parametrize("expires_in", ["120", 120.0, "120.0"], ids=["str", "float", "str_float"])
async def test_numeric_expires_in_is_honored(expires_in):
    # A JSON int/float/numeric-string expires_in must drive the TTL, not fall back to the default.
    clock = _Clock(1000.0)
    post = _RecordingPost({"access_token": "x", "expires_in": expires_in})
    exchanger = Rfc8693TokenExchanger(post, clock=clock)  # ttl = max(120-60, 10) = 60 -> until 1060
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    clock.now = 1061.0
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    assert len(post.calls) == 2


@pytest.mark.asyncio
async def test_short_lived_token_is_not_cached_past_its_expiry():
    # expires_in (5s) below the buffer/min floor must NOT be served stale: cache only until expiry.
    clock = _Clock(1000.0)
    post = _RecordingPost({"access_token": "x", "expires_in": 5})
    exchanger = Rfc8693TokenExchanger(post, clock=clock)
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    clock.now = 1004.0  # within the 5s lifetime -> still cached
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    assert len(post.calls) == 1
    clock.now = 1006.0  # past expiry -> re-exchange, not a stale bearer
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    assert len(post.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"access_token": "x"},
        {"access_token": "x", "expires_in": "not-a-number"},
        {"access_token": "x", "expires_in": True},
    ],
    ids=["missing", "unparseable", "bool"],
)
async def test_unusable_expires_in_falls_back_to_default_ttl(body):
    clock = _Clock(1000.0)
    post = _RecordingPost(body)
    exchanger = Rfc8693TokenExchanger(post, clock=clock, default_ttl_seconds=300.0)
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    clock.now = 1299.0
    await exchanger.exchange("jwt", _SERVER, _CONFIG)
    assert len(post.calls) == 1


@pytest.mark.asyncio
async def test_distributed_coordinator_refresh_and_reread_use_the_cache():
    # Mimics the cross-replica coordinator contract: the winner's refresh populates the cache, a
    # re-entrant refresh sees the fresh entry, and a loser reads it back via reread, all without a
    # second IdP call.
    class _ReplayCoordinator:
        async def run(self, user_id, server_id, refresh, reread):
            first = await refresh()
            second = await refresh()
            via_reread = await reread()
            assert first is not None and second is not None and via_reread is not None
            return via_reread

    post = _RecordingPost({"access_token": "x", "expires_in": 3600})
    exchanger = Rfc8693TokenExchanger(post, coordinator=_ReplayCoordinator(), clock=_Clock())
    result = await exchanger.exchange("jwt", _SERVER, _CONFIG)
    assert isinstance(result, Ok) and result.ok.access_token == "x"
    assert len(post.calls) == 1
