import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import MagicMock

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Request

from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.models.team import LiteLLM_TeamTableCachedObj, Member
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_TeamTable
from litellm.proxy.auth import core
from litellm.proxy.auth import denial as auth_denial
from litellm.proxy.auth import jwt as proxy_jwt
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken
from litellm.proxy.auth.host.facts import (
    AliasMissing,
    EndUserRow,
    FoundTeam,
    RegisteredAgentFacts,
    TeamMissing,
)
from litellm.proxy.auth.host.python import AgentLookup, HttpCall, NoRegisteredAgents, PythonAuthHost, SharedFetches
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.model_max_budget_limiter import PROXY_VirtualKeyModelMaxBudgetLimiter
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.redis.stores.auth_cache import AuthObject
from litellm.redis.stores.router_cooldowns import RouterCooldownStore
from litellm.router_utils.cooldown_cache import CooldownCache
from litellm.rust_bridge._native import RustAuth, RustAuthSettings, RustIdentity
from litellm.types.agents import AgentResponse
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles
from tests.unit.caching.redis_batch_double import FakeNatives, fake_send
from tests.unit.proxy.rust_auth_stores import FakeTeamStore, RustAuthStores, install_rust_auth_stores
from tests.unit.redis.stores.auth_cache_double import shared_cache
from tests.unit.redis.stores.test_router_cooldowns import FakeCooldowns

JWKS_URL: Final = "https://idp.example/jwks"
JWT_AUTH: Final = LiteLLM_JWTAuth(user_id_jwt_field="sub")
SIGNING_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _request(path: str) -> Request:
    return Request({"type": "http", "method": "POST", "path": path, "headers": [], "query_string": b""})


def _host(
    *,
    route: str = "/chat/completions",
    request_data: dict[str, object] | None = None,  # mutable-ok: the host takes the request body it rewrites
    cache: UserApiKeyCache | None = None,
    jwks_cache: DualCache | None = None,
    http_handler: HTTPHandler | None = None,
    shared_fetches: SharedFetches | None = None,
    api_key: str | None = None,
    with_request: bool = True,
    agent_lookup: AgentLookup | None = None,
    prisma_client: PrismaClient | None = None,
) -> PythonAuthHost:
    user_api_key_cache: Final = cache or UserApiKeyCache()
    return PythonAuthHost(
        route=route,
        parent_otel_span=None,
        http=HttpCall(
            request=_request(route),
            request_data=request_data or {},
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        if with_request
        else None,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        jwks_cache=jwks_cache or DualCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=user_api_key_cache),
        llm_router=None,
        llm_model_list=None,
        model_max_budget_limiter=PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache()),
        litellm_proxy_admin_name="default_user_id",
        litellm_jwtauth=JWT_AUTH,
        agent_lookup=agent_lookup or NoRegisteredAgents(),
        http_handler=http_handler or HTTPHandler(),
        shared_fetches=shared_fetches or SharedFetches(),
        api_key=api_key,
    )


@pytest.mark.asyncio
async def test_end_user_row_carries_the_id_it_was_asked_for_not_the_request_bodys() -> None:
    end_user: Final = await _host(request_data={"user": "from-body"}).end_user("from-key")

    assert (end_user.end_user_id, dict(end_user.params)) == ("from-key", {"end_user_id": "from-key"})


@pytest.mark.asyncio
async def test_a_role_change_is_stored_and_the_stored_user_is_cached_and_returned(
    rust_auth_stores: RustAuthStores,
) -> None:
    rust_auth_stores.user.rows["u1"] = {
        "user_id": "u1",
        "user_role": "internal_user",
        "user_email": "stored@example.com",
    }
    cache: Final = UserApiKeyCache()
    stale: Final = LiteLLM_UserTable(user_id="u1", user_role="internal_user", user_email="stale@example.com")

    updated: Final = await _host(cache=cache, prisma_client=MagicMock(spec=PrismaClient)).set_user_role(
        stale, "proxy_admin"
    )

    cached: Final = await cache.async_get_cache(key=AuthObject.user("u1"), model_type=LiteLLM_UserTable)
    assert rust_auth_stores.user.rows["u1"]["user_role"] == "proxy_admin"
    assert (updated.user_role, updated.user_email) == ("proxy_admin", "stored@example.com")
    assert cached == updated


@pytest.mark.asyncio
@pytest.mark.parametrize("user_role", ["proxy_admin", "internal_user"])
async def test_a_session_token_carries_the_email_on_its_user_row(
    rust_auth_stores: RustAuthStores, monkeypatch: pytest.MonkeyPatch, user_role: str
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-salt-session-email")
    rust_auth_stores.user.rows["u1"] = {"user_id": "u1", "user_role": user_role, "user_email": "stored@example.com"}
    minted: Final = ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        user_info=LiteLLM_UserTable(user_id="u1", user_role=user_role)
    )

    key: Final = await _host(prisma_client=MagicMock(spec=PrismaClient)).ui_key(minted)

    assert key is not None
    assert (key.user_id, key.user_role, key.user_email) == ("u1", user_role, "stored@example.com")


@pytest.mark.asyncio
async def test_an_end_user_id_is_kept_when_no_database_can_validate_it() -> None:
    assert await _host().validated_end_user_id("from-body") == "from-body"


@pytest.mark.asyncio
async def test_without_an_end_user_no_limits_are_stamped() -> None:
    end_user: Final = await _host().end_user(None)

    assert (end_user.end_user_id, end_user.end_user, dict(end_user.params)) == (None, None, {})


@pytest.mark.asyncio
async def test_newer_cached_team_values_replace_the_keys_copy() -> None:
    key: Final = UserAPIKeyAuth(token="hashed", team_id="t", team_models=["old"], team_max_budget=1.0)
    team: Final = LiteLLM_TeamTableCachedObj(team_id="t", models=["new"], max_budget=5.0)

    token: Final = await _host().with_team(key, team)

    assert (token.token, token.team_id, token.team_models, token.team_max_budget) == ("hashed", "t", ["new"], 5.0)


def _host_whose_team_read_fails(monkeypatch: pytest.MonkeyPatch, error: Exception) -> PythonAuthHost:
    install_rust_auth_stores(monkeypatch, RustAuthStores(team=FakeTeamStore(rows={"team-a": error})))
    return replace(_host(), prisma_client=SimpleNamespace())  # pyright: ignore[reportArgumentType]  # the team read goes through the repository


@pytest.mark.asyncio
async def test_a_team_read_the_database_cannot_serve_raises_the_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outage: Final = httpx.ConnectError("connection refused")

    with pytest.raises(httpx.ConnectError) as raised:
        await _host_whose_team_read_fails(monkeypatch, outage).team("team-a", upsert=False)

    assert raised.value is outage


@pytest.mark.asyncio
async def test_a_team_read_that_fails_for_another_reason_reads_as_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    read: Final = await _host_whose_team_read_fails(monkeypatch, ValueError("row does not validate")).team(
        "team-a", upsert=False
    )

    assert read is TeamMissing.UNREADABLE


@pytest.mark.asyncio
async def test_object_permission_is_not_read_without_a_database() -> None:
    assert await _host().object_permission("op-1") is None


class _Jwks:
    """Serves the signing key's JWKS and counts fetches. A fetch suspends once like a real request, so
    concurrent callers interleave."""

    def __init__(self, status: int = 200) -> None:
        self.fetches = 0
        self.status: Final = status

    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        self.fetches += 1
        await asyncio.sleep(0)
        key: Final = jwt.algorithms.RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(self.status, json={"keys": [{**key, "kid": "k1"}]}, request=httpx.Request("GET", url))


def _token(claims: dict[str, object]) -> str:
    return jwt.encode({"exp": int(time.time()) + 600, **claims}, SIGNING_KEY, algorithm="RS256", headers={"kid": "k1"})


async def _authorize(
    token: str,
    route: str,
    cache: UserApiKeyCache,
    jwks: _Jwks,
    shared_fetches: SharedFetches | None = None,
    jwks_cache: DualCache | None = None,
):
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True},
        JWT_AUTH,
        core.HostFacts(master_key=None, premium=True, has_db=False),
    )
    host: Final = _host(
        route=route,
        cache=cache,
        jwks_cache=jwks_cache,
        http_handler=jwks,  # pyright: ignore[reportArgumentType]  # serves only the GET the host makes
        shared_fetches=shared_fetches,
        api_key=token,
        with_request=False,
    )
    return await core.drive(
        RustAuth.authorize_jwt(core.request(token, route, "POST"), settings, provisioning=True), host
    )


@pytest.fixture(autouse=True)
def _jwks_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", JWKS_URL)


@pytest.mark.asyncio
async def test_admin_scope_authorizes_a_proxy_admin_from_the_fetched_jwks() -> None:
    jwks: Final = _Jwks()

    outcome: Final = await _authorize(
        _token({"sub": "admin-1", "scope": "litellm_proxy_admin"}), "/key/generate", UserApiKeyCache(), jwks
    )

    assert not outcome.denied
    identity: Final = outcome.identity
    assert isinstance(identity, RustIdentity.Jwt)
    assert identity.is_proxy_admin is True
    assert proxy_jwt.auth_from_result(proxy_jwt.jwt_result(identity, None)).user_role == LitellmUserRoles.PROXY_ADMIN
    assert jwks.fetches == 1


@pytest.mark.asyncio
async def test_a_cached_jwks_is_not_fetched_again() -> None:
    jwks: Final = _Jwks()
    cache: Final = UserApiKeyCache()
    jwks_cache: Final = DualCache()
    token: Final = _token({"sub": "admin-1", "scope": "litellm_proxy_admin"})

    _ = await _authorize(token, "/key/generate", cache, jwks, jwks_cache=jwks_cache)
    _ = await _authorize(token, "/key/generate", cache, jwks, jwks_cache=jwks_cache)

    assert jwks.fetches == 1


@pytest.mark.asyncio
async def test_concurrent_requests_share_one_jwks_fetch() -> None:
    jwks: Final = _Jwks()
    cache: Final = UserApiKeyCache()
    shared_fetches: Final = SharedFetches()
    token: Final = _token({"sub": "admin-1", "scope": "litellm_proxy_admin"})

    outcomes: Final = await asyncio.gather(
        _authorize(token, "/key/generate", cache, jwks, shared_fetches),
        _authorize(token, "/key/generate", cache, jwks, shared_fetches),
    )

    assert not any(outcome.denied for outcome in outcomes)
    assert jwks.fetches == 1


@pytest.mark.asyncio
async def test_concurrent_requests_share_a_failed_jwks_fetch_and_the_next_one_fetches_again() -> None:
    jwks: Final = _Jwks(status=503)
    cache: Final = UserApiKeyCache()
    shared_fetches: Final = SharedFetches()
    token: Final = _token({"sub": "admin-1", "scope": "litellm_proxy_admin"})

    failures: Final = await asyncio.gather(
        _authorize(token, "/key/generate", cache, jwks, shared_fetches),
        _authorize(token, "/key/generate", cache, jwks, shared_fetches),
        return_exceptions=True,
    )

    assert all(
        str(failure).startswith(f"JWT Auth: JWKS endpoint {JWKS_URL} returned status 503") for failure in failures
    )
    assert jwks.fetches == 1
    with pytest.raises(Exception, match="returned status 503"):
        _ = await _authorize(token, "/key/generate", cache, jwks, shared_fetches)
    assert jwks.fetches == 2


class _ParkedFetch:
    """A fetch that waits to be released and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0
        self.release: Final = asyncio.Event()

    async def __call__(self, url: str) -> object:
        self.calls += 1
        _ = await self.release.wait()
        return {"keys": []}


@pytest.mark.asyncio
async def test_a_caller_sharing_a_cancelled_fetch_takes_it_over() -> None:
    shared_fetches: Final = SharedFetches()
    fetch: Final = _ParkedFetch()
    first: Final = asyncio.create_task(shared_fetches.fetch(JWKS_URL, fetch))
    second: Final = asyncio.create_task(shared_fetches.fetch(JWKS_URL, fetch))
    await asyncio.sleep(0)

    _ = first.cancel()
    fetch.release.set()

    assert await asyncio.wait_for(second, timeout=1) == {"keys": []}
    assert first.cancelled()
    assert fetch.calls == 2


@pytest.mark.asyncio
async def test_a_user_token_takes_its_role_from_the_user_row_the_core_returns() -> None:
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(
        key=AuthObject.user("user-1"),
        value=LiteLLM_UserTable(user_id="user-1", user_role="internal_user_viewer"),
        model_type=LiteLLM_UserTable,
    )
    jwt_token: Final = _token({"sub": "user-1"})

    outcome: Final = await _authorize(jwt_token, "/chat/completions", cache, _Jwks())

    identity: Final = outcome.identity
    assert isinstance(identity, RustIdentity.Jwt)
    assert (identity.is_proxy_admin, identity.user_id) == (False, "user-1")
    assert identity.user is not None and identity.user.user_id == "user-1"
    result: Final = proxy_jwt.jwt_result(identity, jwt_token)
    assert result["token"] == jwt_token
    assert proxy_jwt.auth_from_result(result).user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


@pytest.mark.asyncio
async def test_a_token_signed_by_another_key_is_denied_with_the_proxy_message() -> None:
    other: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    forged: Final = jwt.encode(
        {"sub": "admin-1", "scope": "litellm_proxy_admin", "exp": int(time.time()) + 600},
        other,
        algorithm="RS256",
        headers={"kid": "k1"},
    )

    outcome: Final = await _authorize(forged, "/key/generate", UserApiKeyCache(), _Jwks())

    assert outcome.denied
    with pytest.raises(Exception, match="Validation fails"):
        auth_denial.raise_denial(outcome)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "error_class", "message"),
    [
        ({"kid": "rotated-away"}, auth_denial.NoMatchingJWTPublicKeyError, "kid=rotated-away"),
        (
            {"kid": "k1", "typ": "oauth-id-jag+jwt"},
            auth_denial.IdJagIsNotABearerTokenError,
            "an ID-JAG is an authorization grant, not a bearer token",
        ),
    ],
    ids=["unknown_kid", "id_jag_as_bearer"],
)
async def test_a_refused_jwt_is_raised_as_the_error_class_its_failure_row_records(
    headers: dict[str, str], error_class: type[Exception], message: str
) -> None:
    token: Final = jwt.encode({"sub": "admin-1"}, SIGNING_KEY, algorithm="RS256", headers=headers)

    outcome: Final = await _authorize(token, "/key/generate", UserApiKeyCache(), _Jwks())

    assert outcome.denial is not None
    error: Final = auth_denial.denial_error(outcome.denial)
    assert type(error) is error_class
    assert message in str(error)


class _ScriptedHttp:
    """Answers each GET with the next scripted reply, raising it when it is an exception, and records the
    headers it was sent."""

    def __init__(self, *replies: httpx.Response | Exception) -> None:
        self.replies: Final = list(replies)  # mutable-ok: consumed one reply per GET
        self.headers: Final[list[object]] = []  # mutable-ok: records each GET's headers

    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        self.headers.append(headers)
        reply: Final = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _reply(status: int, url: str, *, json: object = None, text: str | None = None) -> httpx.Response:
    request: Final = httpx.Request("GET", url)
    if text is not None:
        return httpx.Response(status, text=text, request=request)
    return httpx.Response(status, json=json, request=request)


def _scripted_host(http: _ScriptedHttp) -> PythonAuthHost:
    return _host(http_handler=http)  # pyright: ignore[reportArgumentType]  # serves only the GETs the host makes


@pytest.mark.asyncio
async def test_a_jwks_fetch_retries_transport_errors_until_one_answers() -> None:
    http: Final = _ScriptedHttp(
        httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), _reply(200, JWKS_URL, json={"keys": []})
    )

    assert await _scripted_host(http).fetch_json(JWKS_URL) == {"keys": []}
    assert http.replies == []


@pytest.mark.asyncio
async def test_a_jwks_fetch_that_never_connects_reads_as_no_document() -> None:
    http: Final = _ScriptedHttp(httpx.ConnectError("1"), httpx.ConnectError("2"), httpx.ConnectError("3"))

    assert await _scripted_host(http).fetch_json(JWKS_URL) is None
    assert http.replies == []


DISCOVERY_URL: Final = "https://idp.example/.well-known/openid-configuration"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "reply", "message"),
    [
        (DISCOVERY_URL, _reply(500, DISCOVERY_URL, text="down"), f"OIDC discovery endpoint {DISCOVERY_URL} returned"),
        (DISCOVERY_URL, _reply(200, DISCOVERY_URL, text="<html>"), "Failed to parse OIDC discovery document"),
        (JWKS_URL, _reply(200, JWKS_URL, text="<html>"), "Error parsing response"),
    ],
    ids=["discovery-status", "discovery-not-json", "jwks-not-json"],
)
async def test_an_unusable_jwt_endpoint_reply_names_the_endpoint(url: str, reply: httpx.Response, message: str) -> None:
    with pytest.raises(Exception, match=message):
        _ = await _scripted_host(_ScriptedHttp(reply)).fetch_json(url)


USERINFO_URL: Final = "https://idp.example/userinfo"


@pytest.mark.asyncio
async def test_userinfo_is_fetched_with_the_callers_token() -> None:
    http: Final = _ScriptedHttp(_reply(200, USERINFO_URL, json={"sub": "user-1"}))

    assert await _scripted_host(http).fetch_userinfo(USERINFO_URL, "opaque-token") == {"sub": "user-1"}
    assert http.headers == [{"Authorization": "Bearer opaque-token", "Accept": "application/json"}]


@pytest.mark.asyncio
async def test_a_refused_userinfo_call_fails_with_the_endpoints_status() -> None:
    http: Final = _ScriptedHttp(_reply(401, USERINFO_URL, text="bad token"))

    with pytest.raises(Exception, match="Failed to fetch OIDC UserInfo: OIDC UserInfo endpoint returned status 401"):
        _ = await _scripted_host(http).fetch_userinfo(USERINFO_URL, "opaque-token")


@pytest.mark.asyncio
async def test_a_team_alias_cannot_be_resolved_without_a_database() -> None:
    assert await _host().team_by_alias("platform") == AliasMissing.NO_DB


@pytest.mark.asyncio
@pytest.mark.parametrize("by_alias", [False, True], ids=["team-id", "team-alias"])
async def test_jwt_projects_only_the_effective_member_from_the_team_it_read(by_alias: bool) -> None:
    cache: Final = UserApiKeyCache()
    key: Final = AuthObject.team_alias("platform") if by_alias else AuthObject.team("team-a")
    caller: Final = Member(user_id="canonical-user", role="admin")
    roster: Final = [Member(user_id="claim-subject", role="user"), caller]
    team: Final = LiteLLM_TeamTableCachedObj(team_id="team-a", team_alias="platform", members_with_roles=roster)
    await cache.async_set_cache(key, team, model_type=LiteLLM_TeamTableCachedObj)
    host: Final = _host(cache=cache, prisma_client=MagicMock(spec=PrismaClient))

    result: Final = await host.team_by_alias("platform") if by_alias else await host.team("team-a", upsert=False)
    assert isinstance(result, FoundTeam)
    assert (result.row.members_with_roles, result.facts.members) == ([], ())
    assert result.facts.roster is not None
    cache.delete_local(key.key)
    replacement: Final = team.model_copy(update={"members_with_roles": roster[:1]})
    await cache.async_set_cache(key, replacement, model_type=LiteLLM_TeamTableCachedObj)

    first: Final = await host.bind_team_member(result.row, result.facts.roster, "canonical-user")
    assert (first.row.members_with_roles, first.facts.members) == ([caller], ("canonical-user",))
    first.row.members_with_roles[0].role = "user"
    second: Final = await host.bind_team_member(result.row, result.facts.roster, "canonical-user")
    missing: Final = await host.bind_team_member(result.row, result.facts.roster, "nonmember")
    assert second.row.members_with_roles == [caller]
    assert missing.facts.members == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "allowed"), [("gpt-4o", True), ("claude-opus", False)])
async def test_team_model_access_follows_the_teams_model_list(model: str, allowed: bool) -> None:
    team: Final = LiteLLM_TeamTable(team_id="t", models=["gpt-4o"])

    assert await _host().team_model_access(team, model) is allowed


class _OneAgent:
    def __init__(self, agent: AgentResponse) -> None:
        self.agent: Final = agent

    def get_agent_by_id(self, agent_id: str) -> AgentResponse | None:
        return None

    def get_agent_by_name(self, agent_name: str) -> AgentResponse | None:
        return self.agent if agent_name == self.agent.agent_name else None


@pytest.mark.asyncio
async def test_a_registered_agent_is_found_by_name_when_no_id_matches() -> None:
    agent: Final = AgentResponse(agent_id="a-1", agent_name="billing-bot", agent_card_params={}, identity_managed=True)
    host: Final = _host(agent_lookup=_OneAgent(agent))

    assert await host.registered_agent("billing-bot") == RegisteredAgentFacts(
        agent_id="a-1", identity_managed=True, has_identity=False, has_legacy_identity=False
    )
    assert await host.registered_agent("someone-else") is None
    assert await _host().registered_agent("billing-bot") is None


def _router_recording_cooldown_reads() -> tuple[Router, FakeCooldowns]:
    handle: Final = FakeCooldowns(has_redis=True)
    router: Final = Router(
        model_list=[
            {"model_name": "claude", "litellm_params": {"model": "anthropic/claude-x"}, "model_info": {"id": "dep-a"}},
            {"model_name": "other", "litellm_params": {"model": "anthropic/other"}, "model_info": {"id": "dep-other"}},
        ]
    )
    router.cooldown_cache = CooldownCache(store=RouterCooldownStore(handle), default_cooldown_time=60)
    return router, handle


async def _identity_prefetch(host: PythonAuthHost) -> None:
    await host.prefetch_identity("sk-test", None)


async def _auth_object_prefetch(host: PythonAuthHost) -> None:
    await host.prefetch_auth_objects(
        UserAPIKeyAuth(token="hashed-key", user_id="user-1"), EndUserRow(end_user_id=None, end_user=None, params={})
    )


_PREFETCHES: Final = pytest.mark.parametrize(
    "prefetch", [_identity_prefetch, _auth_object_prefetch], ids=["identity", "auth_objects"]
)


async def _run(prefetch: Callable[[PythonAuthHost], Awaitable[None]], host: PythonAuthHost) -> None:
    with request_redis_batch_scope(native_factory=FakeNatives(), native_send_factory=fake_send):  # pyright: ignore[reportArgumentType]  # test double
        await prefetch(host)


def _declared(handle: FakeCooldowns) -> list[frozenset[str]]:
    return [frozenset(ids) for ids in handle.declared]


@pytest.mark.asyncio
@_PREFETCHES
@pytest.mark.parametrize(
    ("request_data", "reads"),
    [
        pytest.param({"model": "claude"}, [frozenset({"dep-a"})], id="requested_model"),
        pytest.param({}, [frozenset({"dep-a", "dep-other"})], id="no_model"),
        pytest.param({"model": []}, [frozenset({"dep-a", "dep-other"})], id="non_string_model"),
    ],
)
async def test_the_auth_prefetch_reads_the_cooldowns_of_the_deployments_the_requested_model_may_route_to(
    monkeypatch: pytest.MonkeyPatch,
    prefetch: Callable[[PythonAuthHost], Awaitable[None]],
    request_data: dict[str, object],  # mutable-ok: the host takes the request body it rewrites
    reads: list[frozenset[str]],
) -> None:
    router, handle = _router_recording_cooldown_reads()
    monkeypatch.setattr(proxy_server, "llm_router", router)
    cache, _ = shared_cache()

    await _run(prefetch, _host(request_data=request_data, cache=cache))

    assert _declared(handle) == reads


@pytest.mark.asyncio
@_PREFETCHES
async def test_the_auth_prefetch_reads_no_cooldowns_without_a_proxy_router(
    monkeypatch: pytest.MonkeyPatch, prefetch: Callable[[PythonAuthHost], Awaitable[None]]
) -> None:
    captured, handle = _router_recording_cooldown_reads()
    monkeypatch.setattr(proxy_server, "llm_router", None)
    cache, _ = shared_cache()

    await _run(prefetch, replace(_host(request_data={"model": "claude"}, cache=cache), llm_router=captured))

    assert _declared(handle) == []


@pytest.mark.asyncio
@_PREFETCHES
async def test_the_auth_prefetch_reads_from_the_proxy_router_current_when_it_reads_not_the_one_the_host_captured(
    monkeypatch: pytest.MonkeyPatch, prefetch: Callable[[PythonAuthHost], Awaitable[None]]
) -> None:
    captured, captured_handle = _router_recording_cooldown_reads()
    current, current_handle = _router_recording_cooldown_reads()
    monkeypatch.setattr(proxy_server, "llm_router", current)
    cache, _ = shared_cache()

    await _run(prefetch, replace(_host(request_data={"model": "claude"}, cache=cache), llm_router=captured))

    assert (_declared(captured_handle), _declared(current_handle)) == ([], [frozenset({"dep-a"})])


@pytest.mark.asyncio
async def test_team_model_access_normalizes_claude_picker_hex_id_on_messages_route() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
            }
        ]
    )
    team: Final = LiteLLM_TeamTable(team_id="team-1", models=["gpt-4o"])
    picker_id: Final = f"claude-router-{b'gpt-4o'.hex()}"
    host: Final = replace(_host(route="/v1/messages"), llm_router=router)

    assert await host.team_model_access(team, picker_id) is True


@pytest.mark.asyncio
async def test_team_model_access_normalizes_claude_1m_suffix_with_team_alias() -> None:
    from litellm.proxy._types import LiteLLM_ModelTable

    router: Final = Router(
        model_list=[
            {
                "model_name": "claude-sonnet-4-6",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-6", "api_key": "sk-test"},
                "model_info": {"max_input_tokens": 1_000_000},
            }
        ]
    )
    team: Final = LiteLLM_TeamTable(
        team_id="team-1",
        models=["claude-sonnet-4-6"],
        litellm_model_table=LiteLLM_ModelTable(
            model_aliases={"sonnet": "claude-sonnet-4-6"},
            created_by="test",
            updated_by="test",
        ),
    )
    host: Final = replace(_host(route="/v1/messages"), llm_router=router)

    assert await host.team_model_access(team, "sonnet[1m]") is True


@pytest.mark.asyncio
async def test_team_model_access_and_normalize_skip_provider_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.auth.user_api_key_auth import _normalize_claude_model

    router: Final = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-test"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "llm_router", router)
    team: Final = LiteLLM_TeamTable(team_id="team-1", models=["gpt-4o"])
    picker_id: Final = f"claude-router-{b'gpt-4o'.hex()}"
    passthrough_req: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/messages",
            "headers": [],
            "query_string": b"",
            "path_params": {"endpoint": "v1/messages"},
        }
    )
    request_data: dict[str, object] = {"model": picker_id}
    host: Final = replace(
        _host(route="/v1/messages"),
        llm_router=router,
        http=HttpCall(
            request=passthrough_req,
            request_data=request_data,
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ),
    )

    assert await host.team_model_access(team, picker_id) is False
    await _normalize_claude_model(
        request_data,
        UserAPIKeyAuth(token="t", team_id="team-1"),
        passthrough_req,
        "/v1/messages",
    )
    assert request_data["model"] == picker_id
