"""Tests for renewing the stored SSO identity assertion behind the ID-JAG arm.

Pins the contract an unattended agent depends on: an assertion that has run out is renewed from the
refresh token captured beside it instead of stranding the agent until its user signs in again, the
IdP sees one redemption per user no matter how many tool calls arrive at once, a rotation is written
back without overwriting a sign-in that landed mid-renewal, and the two failure kinds stay
distinguishable - a dead refresh token still challenges the user, an unreachable IdP does not.
"""

import asyncio
import base64
import itertools
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone

import httpx
import jwt as pyjwt
import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_refresher import (
    HttpxTokenEndpointTransport,
    RefreshFailure,
    RefreshingSSOAssertionStore,
    SSOAssertionRefresher,
    SSOClientConfig,
    default_sso_assertion_store,
    sso_client_config,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.sso_assertion_store import (
    AssertionStoreUnavailable,
    SSOIdentityAssertion,
)

SIGNING_KEY = "test-idp-signing-key-32-bytes-long-xxxx"
ISSUER = "https://idp.example.com"
TOKEN_ENDPOINT = "https://idp.example.com/token"

_CLIENT = SSOClientConfig(
    token_endpoint=TOKEN_ENDPOINT,
    client_id="litellm",
    client_secret=SecretStr("s3cret"),
    auth_method="client_secret_basic",
)
_POST_CLIENT = SSOClientConfig(
    token_endpoint=TOKEN_ENDPOINT,
    client_id="litellm",
    client_secret=SecretStr("s3cret"),
    auth_method="client_secret_post",
)


_MINTED = itertools.count()


def _id_token(subject: str = "u1", exp_offset: int = 3600) -> str:
    """A distinct token per call. Two mints with the same claims in the same second would encode
    identically, which would let a test that means "the renewed token replaced the old one" pass
    while comparing a value to itself."""
    return pyjwt.encode(
        {"iss": ISSUER, "sub": subject, "exp": int(time.time()) + exp_offset, "jti": f"t{next(_MINTED)}"},
        SIGNING_KEY,
        algorithm="HS256",
    )


def _stored(id_token: str, *, expires_in: int, refresh_token: str | None = "rt_1") -> SSOIdentityAssertion:
    """A row as the SSO callback wrote it: ``expires_in`` seconds from now, mirroring the id_token."""
    return SSOIdentityAssertion(
        id_token=SecretStr(id_token),
        refresh_token=SecretStr(refresh_token) if refresh_token else None,
        issuer=ISSUER,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
    )


class _FakeRows:
    """The one assertion row per user: the inner read seam and the refresher's read/write pair."""

    def __init__(self, rows: dict[str, SSOIdentityAssertion] | None = None) -> None:
        self.rows: dict[str, SSOIdentityAssertion] = dict(rows or {})
        self.reads: list[str] = []
        self.writes: list[tuple[str, SSOIdentityAssertion]] = []

    async def fetch(self, user_id: str) -> SSOIdentityAssertion | None:
        self.reads.append(user_id)
        # A real suspension point, so concurrent callers interleave here instead of running to
        # completion one at a time and never actually racing.
        await asyncio.sleep(0)
        return self.rows.get(user_id)

    async def write(self, user_id: str, assertion: SSOIdentityAssertion) -> None:
        self.writes.append((user_id, assertion))
        self.rows[user_id] = assertion


class _FakeTransport:
    """Answers every refresh with the same canned result, optionally holding until ``gate`` opens."""

    def __init__(
        self,
        response: Result[Mapping[str, object], RefreshFailure],
        *,
        gate: asyncio.Event | None = None,
        on_call: Callable[[], None] | None = None,
    ) -> None:
        self._response = response
        self._gate = gate
        self._on_call = on_call
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.headers: list[dict[str, str]] = []

    async def post(
        self, url: str, form: Mapping[str, str], headers: Mapping[str, str]
    ) -> Result[Mapping[str, object], RefreshFailure]:
        self.calls.append((url, dict(form)))
        self.headers.append(dict(headers))
        if self._on_call is not None:
            self._on_call()
        if self._gate is not None:
            await self._gate.wait()
        return self._response


def _renewal(id_token: str, refresh_token: str | None = None) -> Result[Mapping[str, object], RefreshFailure]:
    body: dict[str, object] = {"access_token": "at", "id_token": id_token, "token_type": "Bearer"}
    return Ok({**body, "refresh_token": refresh_token} if refresh_token else body)


def _store(
    rows: _FakeRows,
    transport: _FakeTransport,
    *,
    client_config: Callable[[], SSOClientConfig | None] = lambda: _CLIENT,
    coordinator_factory: Callable[[], object] = lambda: None,
) -> RefreshingSSOAssertionStore:
    refresher = SSOAssertionRefresher(
        transport, client_config=client_config, read=rows.fetch, write=rows.write
    )
    return RefreshingSSOAssertionStore(
        rows,
        refresher,
        coordinator_factory=coordinator_factory,  # pyright: ignore[reportArgumentType]  # test doubles stand in for the runtime factory
    )


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(2000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_an_expiring_assertion_is_renewed_and_the_renewal_is_what_the_reader_gets():
    """The whole point: an agent calling after its user's id_token ran out keeps working."""
    stale, fresh = _id_token(exp_offset=-1), _id_token()
    rows = _FakeRows({"alice": _stored(stale, expires_in=-1)})
    transport = _FakeTransport(_renewal(fresh))

    served = await _store(rows, transport).fetch("alice")

    assert served is not None
    assert served.id_token.get_secret_value() == fresh
    assert len(transport.calls) == 1
    url, form = transport.calls[0]
    assert url == TOKEN_ENDPOINT
    assert form["grant_type"] == "refresh_token"
    assert form["refresh_token"] == "rt_1"


@pytest.mark.asyncio
async def test_a_basic_auth_login_gets_a_basic_auth_refresh():
    """The non-PKCE login always sends HTTP Basic, so the renewal must too; credentials in the body
    would 401 against an IdP application registered for Basic."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token()))

    await _store(rows, transport).fetch("alice")

    expected = base64.b64encode(b"litellm:s3cret").decode()
    assert transport.headers[0]["Authorization"] == f"Basic {expected}"
    _url, form = transport.calls[0]
    assert "client_secret" not in form
    assert "client_id" not in form


@pytest.mark.asyncio
async def test_a_body_credential_login_gets_a_body_credential_refresh():
    """The mirror case. A PKCE deployment with GENERIC_INCLUDE_CLIENT_ID set signs in with the
    credentials in the body, so Basic here would 401 against an application registered for post; the
    renewal has to follow the login rather than a constant."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token()))

    await _store(rows, transport, client_config=lambda: _POST_CLIENT).fetch("alice")

    assert "Authorization" not in transport.headers[0]
    _url, form = transport.calls[0]
    assert form["client_id"] == "litellm"
    assert form["client_secret"] == "s3cret"


@pytest.mark.parametrize(
    ("include_client_id", "expected"),
    [
        (None, "client_secret_basic"),
        ("false", "client_secret_basic"),
        ("TRUE", "client_secret_post"),
        ("true", "client_secret_post"),
    ],
)
def test_the_auth_method_follows_the_flag_the_login_reads(include_client_id, expected):
    """``GENERIC_INCLUDE_CLIENT_ID`` is what the PKCE login branches on, parsed the same way it
    parses it, so the renewal cannot pick a method the sign-in did not use."""
    env = {
        "GENERIC_TOKEN_ENDPOINT": TOKEN_ENDPOINT,
        "GENERIC_CLIENT_ID": "litellm",
        "GENERIC_CLIENT_SECRET": "s3cret",
        **({"GENERIC_INCLUDE_CLIENT_ID": include_client_id} if include_client_id is not None else {}),
    }

    config = sso_client_config(env)

    assert config is not None
    assert config.auth_method == expected


@pytest.mark.asyncio
async def test_an_assertion_well_inside_its_lifetime_never_reaches_the_idp():
    """The common path must cost exactly what it did before this store existed."""
    current = _id_token()
    rows = _FakeRows({"alice": _stored(current, expires_in=1800)})
    transport = _FakeTransport(_renewal(_id_token()))

    served = await _store(rows, transport).fetch("alice")

    assert served is not None
    assert served.id_token.get_secret_value() == current
    assert transport.calls == []
    assert rows.writes == []


@pytest.mark.asyncio
async def test_renewal_starts_inside_the_skew_rather_than_after_expiry():
    """A token that would die between resolution and the second exchange leg is replaced first."""
    about_to_expire, fresh = _id_token(), _id_token()
    assert about_to_expire != fresh
    rows = _FakeRows({"alice": _stored(about_to_expire, expires_in=30)})
    transport = _FakeTransport(_renewal(fresh))

    served = await _store(rows, transport).fetch("alice")

    assert served is not None
    assert served.id_token.get_secret_value() == fresh


@pytest.mark.asyncio
async def test_a_user_with_no_stored_assertion_is_still_absent():
    rows = _FakeRows()
    transport = _FakeTransport(_renewal(_id_token()))

    assert await _store(rows, transport).fetch("nobody") is None
    assert transport.calls == []


@pytest.mark.asyncio
async def test_a_refused_refresh_leaves_the_expired_assertion_for_the_reader_to_reject():
    """A dead refresh token is the user's problem, and the reader's expiry guard is what tells them;
    swapping in a renewed-looking value or hiding the row would break that challenge."""
    stale = _id_token(exp_offset=-1)
    rows = _FakeRows({"alice": _stored(stale, expires_in=-1)})
    transport = _FakeTransport(Error(RefreshFailure.of_rejected("the IdP refused the refresh with status 400")))

    served = await _store(rows, transport).fetch("alice")

    assert served is not None
    assert served.id_token.get_secret_value() == stale
    assert rows.writes == []


@pytest.mark.asyncio
async def test_an_unreachable_idp_is_a_store_outage_not_a_sign_in_again_challenge():
    """503, not 412: the user has nothing to fix by signing in again while the IdP is down."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(Error(RefreshFailure.of_unavailable("the IdP token endpoint is unreachable")))

    with pytest.raises(AssertionStoreUnavailable):
        await _store(rows, transport).fetch("alice")


@pytest.mark.asyncio
async def test_a_missing_refresh_token_names_the_scope_the_operator_has_to_set(caplog):
    """Nothing to redeem is the default state of a deployment, so the log has to say what to change
    or the feature stays silently inert."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1, refresh_token=None)})
    transport = _FakeTransport(_renewal(_id_token()))

    with caplog.at_level(logging.WARNING):
        served = await _store(rows, transport).fetch("alice")

    assert served is not None
    assert transport.calls == []
    assert "GENERIC_SCOPE" in caplog.text
    assert "offline_access" in caplog.text


@pytest.mark.asyncio
async def test_an_unconfigured_sso_client_never_calls_the_idp(caplog):
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token()))

    with caplog.at_level(logging.WARNING):
        served = await _store(rows, transport, client_config=lambda: None).fetch("alice")

    assert served is not None
    assert transport.calls == []
    assert "GENERIC_TOKEN_ENDPOINT" in caplog.text


@pytest.mark.asyncio
async def test_a_refresh_response_carrying_no_id_token_is_refused(caplog):
    """An access token is not an identity assertion, so there is nothing to assert upstream."""
    stale = _id_token(exp_offset=-1)
    rows = _FakeRows({"alice": _stored(stale, expires_in=-1)})
    transport = _FakeTransport(Ok({"access_token": "at", "token_type": "Bearer"}))

    with caplog.at_level(logging.WARNING):
        served = await _store(rows, transport).fetch("alice")

    assert served is not None
    assert served.id_token.get_secret_value() == stale
    assert rows.writes == []
    assert "openid" in caplog.text


@pytest.mark.asyncio
async def test_a_rotated_refresh_token_replaces_the_stored_one():
    """An IdP that rotates invalidates the old token, so keeping it would cost a sign-in next time."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token(), refresh_token="rt_2"))

    await _store(rows, transport).fetch("alice")

    stored = rows.rows["alice"]
    assert stored.refresh_token is not None
    assert stored.refresh_token.get_secret_value() == "rt_2"


@pytest.mark.asyncio
async def test_an_omitted_refresh_token_carries_the_previous_one_forward():
    """An IdP that does not rotate expects the original to keep working; dropping it would strand
    the user after exactly one renewal."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token()))

    await _store(rows, transport).fetch("alice")

    stored = rows.rows["alice"]
    assert stored.refresh_token is not None
    assert stored.refresh_token.get_secret_value() == "rt_1"


@pytest.mark.asyncio
async def test_the_renewed_expiry_moves_forward_so_the_next_read_does_not_refresh_again():
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token(exp_offset=3600)))
    store = _store(rows, transport)

    await store.fetch("alice")
    await store.fetch("alice")

    assert len(transport.calls) == 1


async def _explode(user_id: str, assertion: SSOIdentityAssertion) -> None:
    raise RuntimeError("write failed")


@pytest.mark.asyncio
async def test_a_renewal_that_cannot_be_recorded_is_reported_as_transient():
    """The store is what every caller reads, so a renewal nobody can see is not a success. Calling it
    one would hand back a token the gateway failed to record."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    refresher = SSOAssertionRefresher(
        _FakeTransport(_renewal(_id_token())), client_config=lambda: _CLIENT, read=rows.fetch, write=_explode
    )

    outcome = await refresher.refresh("alice", rows.rows["alice"])

    assert isinstance(outcome, Error)
    assert outcome.error.kind == "unavailable"


@pytest.mark.asyncio
async def test_a_failed_write_does_not_tell_the_user_to_sign_in_again():
    """A database that cannot take the write is not something signing in again fixes, so the reader
    has to see an outage rather than the stale row's expiry."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token()))
    refresher = SSOAssertionRefresher(
        transport, client_config=lambda: _CLIENT, read=rows.fetch, write=_explode
    )
    store = RefreshingSSOAssertionStore(
        rows,
        refresher,
        coordinator_factory=lambda: None,  # pyright: ignore[reportArgumentType]  # test double stands in for the runtime factory
    )

    with pytest.raises(AssertionStoreUnavailable):
        await store.fetch("alice")

    assert len(transport.calls) == 1


@pytest.mark.asyncio
async def test_concurrent_reads_for_one_user_redeem_the_refresh_token_once():
    """A burst of tool calls must not replay one refresh token N times: an IdP that rotates reads
    that as reuse and can revoke the whole grant chain."""
    gate = asyncio.Event()
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    fresh = _id_token()
    transport = _FakeTransport(_renewal(fresh), gate=gate)
    store = _store(rows, transport)

    callers = [asyncio.create_task(store.fetch("alice")) for _ in range(8)]
    await _until(lambda: len(transport.calls) >= 1 and len(rows.reads) >= 8)
    # Guards against a vacuous pass: every caller must have read the expired row and entered the
    # renewal branch while the winner is still blocked, otherwise they never raced at all.
    assert len(rows.reads) >= 8
    assert not any(task.done() for task in callers)

    gate.set()
    served = await asyncio.gather(*callers)

    assert len(transport.calls) == 1
    assert {assertion.id_token.get_secret_value() for assertion in served if assertion is not None} == {fresh}


@pytest.mark.asyncio
async def test_concurrent_reads_for_different_users_each_get_their_own_refresh():
    """Single-flight is per user; collapsing across users would leave everyone but one stranded."""
    gate = asyncio.Event()
    rows = _FakeRows(
        {
            "alice": _stored(_id_token("alice", exp_offset=-1), expires_in=-1),
            "bob": _stored(_id_token("bob", exp_offset=-1), expires_in=-1),
        }
    )
    transport = _FakeTransport(_renewal(_id_token()), gate=gate)
    store = _store(rows, transport)

    callers = [asyncio.create_task(store.fetch(user)) for user in ("alice", "bob")]
    await _until(lambda: len(transport.calls) >= 2)
    gate.set()
    await asyncio.gather(*callers)

    assert len(transport.calls) == 2
    assert {form["refresh_token"] for _url, form in transport.calls} == {"rt_1"}


@pytest.mark.asyncio
async def test_a_renewal_writes_back_when_the_row_did_not_move():
    """The refresh-then-sign-in ordering: nothing displaced the row, so the rotation must land."""
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    fresh = _id_token()
    transport = _FakeTransport(_renewal(fresh, refresh_token="rt_2"))

    served = await _store(rows, transport).fetch("alice")

    assert [user_id for user_id, _assertion in rows.writes] == ["alice"]
    assert rows.rows["alice"].id_token.get_secret_value() == fresh
    assert served is not None
    assert served.id_token.get_secret_value() == fresh


@pytest.mark.asyncio
async def test_a_sign_in_landing_mid_renewal_is_not_overwritten():
    """The sign-in-then-refresh ordering. The login wrote a newer assertion while the IdP call was in
    flight; overwriting it would put back a refresh token the IdP has already rotated away, costing
    that user a sign-in later."""
    from_login = _stored(_id_token("alice"), expires_in=3600, refresh_token="rt_from_login")
    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})

    def _login_lands() -> None:
        rows.rows["alice"] = from_login

    transport = _FakeTransport(_renewal(_id_token(), refresh_token="rt_2"), on_call=_login_lands)

    served = await _store(rows, transport).fetch("alice")

    assert rows.writes == []
    stored = rows.rows["alice"]
    assert stored.refresh_token is not None
    assert stored.refresh_token.get_secret_value() == "rt_from_login"
    assert served is not None
    assert served.id_token.get_secret_value() == from_login.id_token.get_secret_value()


class _RecordingCoordinator:
    """Stands in for the cross-replica coordinator, running the winner's refresh inline."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, str]] = []

    async def run(
        self,
        user_id: str,
        server_id: str,
        refresh: Callable[[], Awaitable[None]],
        reread: Callable[[], Awaitable[None]],
    ) -> None:
        self.runs.append((user_id, server_id))
        return await refresh()


@pytest.mark.asyncio
async def test_the_cross_replica_coordinator_is_used_and_built_once():
    """Redis elects one refresher across the fleet; rebuilding its client per renewal would open a
    connection every time."""
    coordinator = _RecordingCoordinator()
    builds: list[int] = []

    def _factory() -> object:
        builds.append(1)
        return coordinator

    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token(exp_offset=-1)))
    store = _store(rows, transport, coordinator_factory=_factory)

    await store.fetch("alice")
    await store.fetch("alice")

    assert len(builds) == 1
    assert coordinator.runs == [("alice", "sso_identity_assertion"), ("alice", "sso_identity_assertion")]


@pytest.mark.asyncio
async def test_the_in_process_coordinator_is_retried_until_redis_appears():
    """A proxy that gains Redis after boot must stop electing a winner per worker."""
    coordinator = _RecordingCoordinator()
    available: list[bool] = [False]

    def _factory() -> object | None:
        return coordinator if available[0] else None

    rows = _FakeRows({"alice": _stored(_id_token(exp_offset=-1), expires_in=-1)})
    transport = _FakeTransport(_renewal(_id_token(exp_offset=-1)))
    store = _store(rows, transport, coordinator_factory=_factory)

    await store.fetch("alice")
    assert coordinator.runs == []

    available[0] = True
    await store.fetch("alice")
    assert coordinator.runs == [("alice", "sso_identity_assertion")]


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"GENERIC_CLIENT_ID": "litellm", "GENERIC_CLIENT_SECRET": "s"},
        {"GENERIC_TOKEN_ENDPOINT": TOKEN_ENDPOINT, "GENERIC_CLIENT_SECRET": "s"},
        {"GENERIC_TOKEN_ENDPOINT": TOKEN_ENDPOINT, "GENERIC_CLIENT_ID": "litellm"},
        {"GENERIC_TOKEN_ENDPOINT": "", "GENERIC_CLIENT_ID": "litellm", "GENERIC_CLIENT_SECRET": "s"},
    ],
)
def test_a_partial_sso_client_is_no_client(env):
    """Redeeming against a half-configured client would post credentials nowhere useful; the arm
    treats it as "cannot renew" and falls back to the sign-in challenge."""
    assert sso_client_config(env) is None


def test_the_configured_sso_client_is_the_one_the_login_used():
    config = sso_client_config(
        {
            "GENERIC_TOKEN_ENDPOINT": TOKEN_ENDPOINT,
            "GENERIC_CLIENT_ID": "litellm",
            "GENERIC_CLIENT_SECRET": "s3cret",
        }
    )

    assert config is not None
    assert config.token_endpoint == TOKEN_ENDPOINT
    assert config.client_id == "litellm"
    assert config.client_secret.get_secret_value() == "s3cret"


def test_the_live_store_renews_over_the_database_reader():
    """The composition root has to produce a renewing store, or none of this runs in production."""
    assert isinstance(default_sso_assertion_store(), RefreshingSSOAssertionStore)


def _responding(response: httpx.Response | None) -> HttpxTokenEndpointTransport:
    async def _post(url: str, form: Mapping[str, str], headers: Mapping[str, str]) -> httpx.Response | None:
        return response

    return HttpxTokenEndpointTransport(_post)


def _json_response(status: int, payload: dict[str, object]) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("POST", TOKEN_ENDPOINT))


@pytest.mark.parametrize("status", [400, 401, 403])
@pytest.mark.asyncio
async def test_the_idp_declining_the_grant_is_a_refusal_the_user_must_act_on(status):
    """A 4xx means this refresh token is finished; calling that an outage would sit the user behind a
    503 forever instead of telling them to sign in."""
    outcome = await _responding(_json_response(status, {"error": "invalid_grant"})).post(TOKEN_ENDPOINT, {}, {})

    assert isinstance(outcome, Error)
    assert outcome.error.kind == "rejected"


@pytest.mark.parametrize("status", [500, 502, 503])
@pytest.mark.asyncio
async def test_a_failing_idp_is_an_outage_not_a_refusal(status):
    """The refresh token is probably fine; telling the user to sign in again would blame them for
    someone else's outage, and would burn their session for nothing."""
    outcome = await _responding(_json_response(status, {})).post(TOKEN_ENDPOINT, {}, {})

    assert isinstance(outcome, Error)
    assert outcome.error.kind == "unavailable"


@pytest.mark.asyncio
async def test_an_unreachable_endpoint_is_an_outage():
    async def _post(url: str, form: Mapping[str, str], headers: Mapping[str, str]) -> httpx.Response | None:
        raise httpx.ConnectError("connection refused")

    outcome = await HttpxTokenEndpointTransport(_post).post(TOKEN_ENDPOINT, {}, {})

    assert isinstance(outcome, Error)
    assert outcome.error.kind == "unavailable"


@pytest.mark.asyncio
async def test_a_non_json_body_is_an_outage():
    response = httpx.Response(200, text="<html>maintenance</html>", request=httpx.Request("POST", TOKEN_ENDPOINT))

    outcome = await _responding(response).post(TOKEN_ENDPOINT, {}, {})

    assert isinstance(outcome, Error)
    assert outcome.error.kind == "unavailable"


@pytest.mark.asyncio
async def test_a_missing_response_is_an_outage():
    outcome = await _responding(None).post(TOKEN_ENDPOINT, {}, {})

    assert isinstance(outcome, Error)
    assert outcome.error.kind == "unavailable"


@pytest.mark.asyncio
async def test_a_successful_grant_is_handed_back_as_the_parsed_body():
    outcome = await _responding(_json_response(200, {"access_token": "at", "id_token": "idt"})).post(
        TOKEN_ENDPOINT, {"grant_type": "refresh_token"}, {}
    )

    assert isinstance(outcome, Ok)
    assert outcome.ok["id_token"] == "idt"
