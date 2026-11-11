import time
from datetime import datetime, timezone
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
from litellm.proxy._types import LiteLLM_JWTAuth
from litellm.proxy.auth import core, rust_jwt
from litellm.proxy.auth import denial as auth_denial
from litellm.proxy.auth import identity as auth_identity
from litellm.proxy.auth.host.python import HttpCall, NoRegisteredAgents, PythonAuthHost
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.hooks.model_max_budget_limiter import PROXY_VirtualKeyModelMaxBudgetLimiter
from litellm.proxy.utils import ProxyLogging
from litellm.rust_bridge._native import RustAuth, RustIdentity
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
    api_key: str | None = None,
    with_request: bool = True,
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
        agent_lookup=NoRegisteredAgents(),
        http_handler=http_handler or HTTPHandler(),
        api_key=api_key,
    )


@pytest.mark.asyncio
async def test_mapped_jwt_end_user_wins_over_the_request_body() -> None:
    resolved: Final = UserAPIKeyAuth(token="hashed", end_user_id="from-jwt")

    end_user: Final = await _host(request_data={"user": "from-body"}).end_user("hashed", resolved)

    assert (end_user.end_user_id, dict(end_user.params)) == ("from-jwt", {"end_user_id": "from-jwt"})


@pytest.mark.asyncio
async def test_end_user_comes_from_the_request_body_for_a_presented_key() -> None:
    end_user: Final = await _host(request_data={"user": "from-body"}).end_user("sk-1", None)

    assert end_user.end_user_id == "from-body"


@pytest.mark.asyncio
async def test_without_an_end_user_no_limits_are_stamped() -> None:
    end_user: Final = await _host().end_user("sk-1", None)

    assert (end_user.end_user_id, end_user.end_user, dict(end_user.params)) == (None, None, {})


@pytest.mark.asyncio
async def test_newer_cached_team_values_replace_the_keys_copy() -> None:
    key: Final = UserAPIKeyAuth(token="hashed", team_id="t", team_models=["old"], team_max_budget=1.0)
    team: Final = LiteLLM_TeamTableCachedObj(team_id="t", models=["new"], max_budget=5.0)

    token: Final = await _host().with_team(key, team)

    assert (token.token, token.team_id, token.team_models, token.team_max_budget) == ("hashed", "t", ["new"], 5.0)


@pytest.mark.asyncio
async def test_object_permission_is_not_read_without_a_database() -> None:
    assert await _host().object_permission("op-1") is None


class _Jwks:
    """Serves the signing key's JWKS and counts fetches."""

    def __init__(self) -> None:
        self.fetches = 0

    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        self.fetches += 1
        key: Final = jwt.algorithms.RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(200, json={"keys": [{**key, "kid": "k1"}]}, request=httpx.Request("GET", url))


def _token(claims: dict[str, object]) -> str:
    return jwt.encode({"exp": int(time.time()) + 600, **claims}, SIGNING_KEY, algorithm="RS256", headers={"kid": "k1"})


async def _authorize(token: str, route: str, cache: UserApiKeyCache, jwks: _Jwks):
    settings: Final = core.build_settings(
        {"enable_jwt_auth": True},
        JWT_AUTH,
        master_key=None,
        premium_user=True,
        has_db=False,
    )
    host: Final = _host(
        route=route,
        cache=cache,
        http_handler=jwks,  # pyright: ignore[reportArgumentType]  # serves only the GET the host makes
        api_key=token,
        with_request=False,
    )
    return await core.drive(
        RustAuth.authorize_jwt(rust_jwt.request(token, route, "POST"), settings, provisioning=True), host
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
    assert (
        auth_identity.auth_from_result(auth_identity.jwt_result(identity, None)).user_role
        == LitellmUserRoles.PROXY_ADMIN
    )
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
    result: Final = auth_identity.jwt_result(identity, jwt_token)
    assert result["token"] == jwt_token
    assert auth_identity.auth_from_result(result).user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY


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
