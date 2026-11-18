"""`user_api_key_auth` for a deployment whose teams come from a JWT `roles` claim: users are created on first sight,
model access is per team, pass-through endpoints require auth, and requests are let through a database outage. A
second JWT configuration maps a client claim to a virtual key and names the end user by a claim. The auth core is
the real one, so every case crosses from the Python host into the core and back."""

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Request
from pydantic import JsonValue

from litellm.proxy import proxy_server
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_JWTAuth,
    LiteLLM_TeamTable,
    ProxyException,
    TeamMemberAddRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, end_user_cache_key
from litellm.proxy.management_endpoints import team_endpoints
from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pass_through
from litellm.repositories import rust_database
from litellm.rust_bridge._native import RustJwtKeyMappingRepository
from tests.unit.proxy.auth._user_table import UserTable
from tests.unit.proxy.rust_auth_stores import FakeJwtKeyMappingStore, JwtMapping

JWKS_URL: Final = "https://idp.invalid/keys"
SIGNING_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
YEAR_2100: Final = 4102444800
MAPPED_KEY: Final = "sk-team-a-key-0123456789abcdef"
MAPPED_CLIENT: Final = "client-team-a"
SOLVER: Final = "/v1/solver/solution"

EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
JWT_AUTH: Final[Mapping[str, object]] = MappingProxyType(
    {
        "user_id_jwt_field": "oid",
        "user_email_jwt_field": "upn",
        "team_ids_jwt_field": "roles",
        "user_id_upsert": True,
        "enforce_team_based_model_access": True,
        "team_allowed_routes": ["openai_routes", "info_routes", "mcp_routes", "anthropic_routes"],
    }
)
MAPPED: Final[Mapping[str, object]] = MappingProxyType(
    {"virtual_key_claim_field": "azp", "end_user_id_jwt_field": "customer"}
)
CHAT: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"model": "model-a", "messages": [{"role": "user", "content": "hi"}]}
)
MEMBER: Final[Mapping[str, JsonValue]] = MappingProxyType({"oid": "new-oid", "upn": "new@corp.invalid"})
TEAMS: Final = (
    LiteLLM_TeamTable(team_id="team-a", models=["model-a"]),
    LiteLLM_TeamTable(team_id="team-all", team_alias="Team All", models=[]),
    LiteLLM_TeamTable(team_id="team-pt", models=["model-a"], metadata={"allowed_passthrough_routes": [SOLVER]}),
    LiteLLM_TeamTable(team_id="team-blocked", models=["model-a"], blocked=True),
)
BUDGET: Final = LiteLLM_BudgetTable(max_budget=1.0)
CUSTOMERS: Final = (
    LiteLLM_EndUserTable(user_id="customer-ok", blocked=False, spend=0.5, litellm_budget_table=BUDGET),
    LiteLLM_EndUserTable(user_id="customer-over", blocked=False, spend=5.0, litellm_budget_table=BUDGET),
)


@dataclass(frozen=True, slots=True)
class Call:
    """One request carrying a JWT signed over `claims`."""

    claims: Mapping[str, JsonValue]
    path: str = "/v1/chat/completions"
    method: str = "POST"
    body: Mapping[str, JsonValue] | None = CHAT
    headers: Mapping[str, str] = MappingProxyType({})
    jwt_auth: Mapping[str, object] = EMPTY
    settings: Mapping[str, object] = EMPTY
    database_up: bool = True


@dataclass(frozen=True, slots=True)
class Served:
    user_id: str | None
    team_id: str | None
    user_email: str | None = None
    end_user_id: str | None = None


@dataclass(frozen=True, slots=True)
class Refused:
    """`logged_as` is the class the failure spend row records."""

    code: str
    error_type: str
    logged_as: str
    message: str


@dataclass(frozen=True, slots=True)
class Outcome:
    """What the client got, the users auth created and the `(team, user)` memberships it added."""

    reply: Served | Refused
    created: tuple[str, ...] = ()
    added: tuple[tuple[str, str], ...] = ()


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class _Jwks:
    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        key: Final = jwt.algorithms.RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(200, json={"keys": [{**key, "kid": "k1"}]}, request=httpx.Request("GET", url))


def _cache(database_up: bool) -> UserApiKeyCache:
    """Teams, customers and the mapped key, or nothing once the database is down."""
    cache: Final = UserApiKeyCache()
    if not database_up:
        return cache
    for team in TEAMS:
        cache.set_cache(f"team_id:{team.team_id}", team, model_type=LiteLLM_TeamTable)
    for customer in CUSTOMERS:
        cache.set_cache(end_user_cache_key(customer.user_id), customer, model_type=LiteLLM_EndUserTable)
    cache.set_cache(
        _hash(MAPPED_KEY),
        UserAPIKeyAuth(
            token=_hash(MAPPED_KEY), team_id="team-a", user_id="key-owner", models=[], team_models=["model-a"]
        ),
        model_type=UserAPIKeyAuth,
    )
    return cache


class _UnreachableUsers(UserTable):
    async def find_unique(self, where: Mapping[str, object], include: object = None) -> None:
        raise httpx.ConnectError("connection refused")


async def _teams_by_alias(where: Mapping[str, object], **_: object) -> list[LiteLLM_TeamTable]:
    return [team for team in TEAMS if team.team_alias is not None and team.team_alias == where.get("team_alias")]


class _Tables(SimpleNamespace):
    """Every table is empty except the ones it is built with."""

    def __getattr__(self, table: str) -> SimpleNamespace:
        return SimpleNamespace(
            find_unique=AsyncMock(return_value=None),
            find_first=AsyncMock(return_value=None),
            find_many=AsyncMock(return_value=[]),
        )


def _prisma(users: UserTable, database_up: bool) -> MagicMock:
    """A database holding only users, whose key, team and user reads fail once it is down."""
    read: Final = (
        AsyncMock(return_value=None) if database_up else AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    )
    tables: Final = _Tables(
        litellm_teamtable=SimpleNamespace(find_unique=read, find_first=read, find_many=_teams_by_alias),
        litellm_usertable=users,
    )
    prisma: Final = MagicMock()
    prisma.get_data = read
    prisma.attempt_db_reconnect = AsyncMock(return_value=False)
    prisma.db = tables
    prisma.writer_db = tables
    return prisma


def _request(call: Call) -> Request:
    headers: Final = (("content-type", "application/json"), *call.headers.items())
    request: Final = Request(
        {
            "type": "http",
            "method": call.method,
            "path": call.path,
            "headers": [(name.lower().encode(), value.encode()) for name, value in headers],
            "query_string": b"",
        }
    )
    request._body = b"" if call.body is None else json.dumps(dict(call.body)).encode()
    return request


def _bearer(claims: Mapping[str, JsonValue]) -> str:
    token: Final = jwt.encode({"exp": YEAR_2100, **claims}, SIGNING_KEY, algorithm="RS256", headers={"kid": "k1"})
    return f"Bearer {token}"


async def _reply(call: Call, failure_hook: AsyncMock) -> Served | Refused:
    try:
        auth: Final = await user_api_key_auth(request=_request(call), api_key=_bearer(call.claims))
    except ProxyException as error:
        return Refused(
            code=str(error.code),
            error_type=str(getattr(error.type, "value", error.type)),
            logged_as=type(failure_hook.call_args.kwargs["original_exception"]).__name__,
            message=str(error.message),
        )
    return Served(auth.user_id, auth.team_id, user_email=auth.user_email, end_user_id=auth.end_user_id)


async def _outcome(monkeypatch: pytest.MonkeyPatch, call: Call) -> Outcome:
    users: Final = UserTable() if call.database_up else _UnreachableUsers()
    mappings: Final = FakeJwtKeyMappingStore([JwtMapping("azp", MAPPED_CLIENT, _hash(MAPPED_KEY))])
    if not call.database_up:
        mappings.error = httpx.ConnectError("connection refused")
    database: Final = rust_database.repository

    def repository(kind: Callable[..., object]) -> object:
        return mappings if kind is RustJwtKeyMappingRepository else database(kind)

    monkeypatch.setattr(rust_database, "repository", repository)
    added: Final[list[tuple[str, str]]] = []  # mutable-ok: records the team members auth adds
    failure_hook: Final = AsyncMock(return_value=None)

    async def add_member(data: TeamMemberAddRequest, **_: object) -> None:
        assert not isinstance(data.member, list)
        added.append((data.team_id, str(data.member.user_id)))

    monkeypatch.setenv("JWT_PUBLIC_KEY_URL", JWKS_URL)
    monkeypatch.delenv("JWT_ISSUER", raising=False)
    monkeypatch.delenv("JWT_AUDIENCE", raising=False)
    monkeypatch.setattr(team_endpoints, "team_member_add", add_member)
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "post_call_failure_hook", failure_hook)
    for name, value in {
        "prisma_client": _prisma(users, call.database_up),
        "user_api_key_cache": _cache(call.database_up),
        "premium_user": True,
        "master_key": "sk-profile-master-key-0123456789",
        "user_custom_auth": None,
        "llm_router": None,
        "general_settings": {"enable_jwt_auth": True, "allow_requests_on_db_unavailable": True, **call.settings},
        "config_passthrough_endpoints": None,
        "jwt_http_handler": _Jwks(),
        "litellm_jwtauth": LiteLLM_JWTAuth(**{**JWT_AUTH, **call.jwt_auth}),
    }.items():
        monkeypatch.setattr(proxy_server, name, value)
    await pass_through.initialize_pass_through_endpoints(
        [{"path": SOLVER, "target": "https://host-005.invalid/solver/solution", "auth": True, "include_subpath": True}]
    )
    try:
        reply: Final = await _reply(call, failure_hook)
    finally:
        await pass_through.initialize_pass_through_endpoints([])
    return Outcome(reply, created=tuple(row.user_id for row in users.rows), added=tuple(added))


def _joined(team_id: str) -> Outcome:
    """A first-time user, created and added to `team_id`, whose request that team serves."""
    return Outcome(
        Served("new-oid", team_id, user_email="new@corp.invalid"), created=("new-oid",), added=((team_id, "new-oid"),)
    )


CASES: Final = (
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}),
        _joined("team-a"),
        id="a_new_user_is_created_and_joins_the_claimed_team",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["no-such-team", "team-a"]}),
        _joined("team-a"),
        id="a_claimed_team_the_proxy_does_not_have_is_skipped",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a", "team-all"]}, headers={"x-litellm-team-id": "Team All"}),
        _joined("team-all"),
        id="the_team_header_picks_a_claimed_team_by_its_alias",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, headers={"x-litellm-team-id": "team-all"}),
        Outcome(
            Refused(
                "403",
                "auth_error",
                "HTTPException",
                "x-litellm-team-id 'team-all' does not resolve to a team id or a unique team alias in your JWT's "
                'allowed teams. Allowed team ids: ["team-a"]',
            )
        ),
        id="a_team_header_outside_the_claim_is_refused",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-blocked"]}),
        Outcome(
            Refused(
                "401",
                "auth_error",
                "Exception",
                "Authentication Error, Team=team-blocked is blocked. Update via `/team/unblock` if you're an admin.",
            ),
            created=("new-oid",),
            added=(("team-blocked", "new-oid"),),
        ),
        id="a_blocked_team_is_refused_after_the_user_joined_it",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, body={**CHAT, "user": "customer-over"}),
        Outcome(
            Refused(
                "422",
                "budget_exceeded",
                "BudgetExceededError",
                "ExceededBudget: End User=customer-over over budget. Spend=5.0, Budget=1.0",
            ),
            created=("new-oid",),
            added=(("team-a", "new-oid"),),
        ),
        id="a_body_end_user_over_budget_is_refused",
    ),
    pytest.param(
        Call(
            claims={"azp": MAPPED_CLIENT, "customer": "customer-ok"},
            body={**CHAT, "user": "customer-over"},
            jwt_auth=MAPPED,
        ),
        Outcome(Served("key-owner", "team-a", end_user_id="customer-ok")),
        id="a_mapped_jwt_is_checked_for_the_end_user_its_claim_names_over_the_body_one",
    ),
    pytest.param(
        Call(claims={**MEMBER, "azp": "client-unknown", "roles": ["team-a"]}, jwt_auth=MAPPED),
        _joined("team-a"),
        id="a_client_without_a_mapped_key_falls_back_to_the_claimed_team",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-pt"]}, path=f"{SOLVER}/job-1", method="GET", body=None),
        _joined("team-pt"),
        id="a_granting_team_reaches_a_path_under_a_pass_through_endpoint",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, path=f"{SOLVER}/job-1", method="GET", body=None),
        Outcome(
            Refused(
                "403",
                "auth_error",
                "HTTPException",
                f"Team not allowed to access passthrough route {SOLVER}/job-1. Configure `allowed_passthrough_routes` "
                "on the team.",
            )
        ),
        id="a_team_without_the_grant_is_refused_under_a_pass_through_endpoint",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, database_up=False),
        Outcome(Served("__db_unavailable_fallback__", None)),
        id="a_team_read_during_an_outage_is_let_through",
    ),
    pytest.param(
        Call(
            claims={**MEMBER, "roles": ["team-a"]},
            settings={"allow_requests_on_db_unavailable": False},
            database_up=False,
        ),
        Outcome(
            Refused(
                "503",
                "no_db_connection",
                "ConnectError",
                "Service Unavailable, the authentication database is temporarily unreachable. Please retry shortly.",
            )
        ),
        id="a_team_read_during_an_outage_is_a_503_when_outages_are_not_let_through",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("call", "expected"), CASES)
async def test_a_jwt_request_is_served_or_refused(
    monkeypatch: pytest.MonkeyPatch, call: Call, expected: Outcome
) -> None:
    assert await _outcome(monkeypatch, call) == expected
