import asyncio
import time
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Final

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Request

from litellm.caching.dual_cache import DualCache
from litellm.llms.custom_httpx.httpx_handler import HTTPHandler
from litellm.models.team import LiteLLM_TeamTableCachedObj
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_TeamTable
from litellm.proxy.auth import core
from litellm.proxy.auth import denial as auth_denial
from litellm.proxy.auth import jwt as proxy_jwt
from litellm.proxy.auth.host.facts import AliasMissing, RegisteredAgentFacts, TeamMissing
from litellm.proxy.auth.host.python import AgentLookup, HttpCall, NoRegisteredAgents, PythonAuthHost, SharedFetches
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.model_max_budget_limiter import PROXY_VirtualKeyModelMaxBudgetLimiter
from litellm.proxy.utils import ProxyLogging
from litellm.rust_bridge._native import RustAuth, RustAuthSettings, RustIdentity
from litellm.types.agents import AgentResponse
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.proxy.auth.user_roles import LitellmUserRoles

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
    http_handler: HTTPHandler | None = None,
    shared_fetches: SharedFetches | None = None,
    api_key: str | None = None,
    with_request: bool = True,
    agent_lookup: AgentLookup | None = None,
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
        prisma_client=None,
        user_api_key_cache=user_api_key_cache,
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


class _FailingTeams:
    def __init__(self, error: Exception) -> None:
        self.error: Final = error

    async def find_unique(self, **_: object) -> None:
        raise self.error


def _host_whose_team_read_fails(error: Exception) -> PythonAuthHost:
    tables: Final = SimpleNamespace(litellm_teamtable=_FailingTeams(error))
    return replace(_host(), prisma_client=SimpleNamespace(db=tables, writer_db=tables))  # pyright: ignore[reportArgumentType]  # only the team table is read


@pytest.mark.asyncio
async def test_a_team_read_the_database_cannot_serve_raises_the_connection_error() -> None:
    outage: Final = httpx.ConnectError("connection refused")

    with pytest.raises(httpx.ConnectError) as raised:
        await _host_whose_team_read_fails(outage).team("team-a", upsert=False)

    assert raised.value is outage


@pytest.mark.asyncio
async def test_a_team_read_that_fails_for_another_reason_reads_as_unreadable() -> None:
    read: Final = await _host_whose_team_read_fails(ValueError("row does not validate")).team("team-a", upsert=False)

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
    token: str, route: str, cache: UserApiKeyCache, jwks: _Jwks, shared_fetches: SharedFetches | None = None
):
    settings: Final = RustAuthSettings(
        {"enable_jwt_auth": True},
        JWT_AUTH,
        core.HostFacts(master_key=None, premium=True, has_db=False),
    )
    host: Final = _host(
        route=route,
        cache=cache,
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
    token: Final = _token({"sub": "admin-1", "scope": "litellm_proxy_admin"})

    _ = await _authorize(token, "/key/generate", cache, jwks)
    _ = await _authorize(token, "/key/generate", cache, jwks)

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
        key="user-1",
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
