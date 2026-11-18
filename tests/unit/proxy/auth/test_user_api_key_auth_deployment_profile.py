"""Auth outcomes for a deployment configured like one production install: teams come from a JWT `roles` claim,
users are created on first sight, model access is per team, pass-through endpoints require auth, and requests are
let through a database outage. A second JWT configuration maps a client claim to a virtual key and names the end
user by a claim."""

import hashlib
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType, SimpleNamespace
from typing import Final, Literal
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
    LiteLLM_UserTable,
    ProxyException,
    TeamMemberAddRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, end_user_cache_key
from litellm.proxy.management_endpoints import team_endpoints
from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pass_through
from tests.unit.proxy.auth._user_table import UserTable

JWKS_URL: Final = "https://idp.invalid/keys"
SIGNING_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)

MASTER_KEY: Final = "sk-profile-master-key-0123456789"
TEAM_KEY: Final = "sk-team-a-key-0123456789abcdef"
PERSONAL_KEY: Final = "sk-personal-key-0123456789abcdef"
GRANTED_KEY: Final = "sk-pt-key-0123456789abcdef0"
BLOCKED_KEY: Final = "sk-blocked-key-0123456789abcdef"
EXPIRED_KEY: Final = "sk-expired-key-0123456789abcdef"
DELETED_KEY: Final = "sk-deleted-key-0123456789abcdef"
UNKNOWN_KEY: Final = "sk-unknown-key-0123456789abcdef"

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
EMBEDDINGS: Final = "/v1/embeddings/vendor/embed-model"
GRANTS: Final = (EMBEDDINGS, "/vendor/v1/systemone", "/v1/solver/solution")
PASS_THROUGH: Final = (
    {
        "path": "/vendor/v1/systemone",
        "target": "https://host-001.invalid/v1/systemone",
        "auth": True,
        "methods": ["POST"],
    },
    {"path": EMBEDDINGS, "target": "https://host-003.invalid/v1/embeddings", "auth": True},
    {"path": f"{EMBEDDINGS}/models", "target": "https://host-003.invalid/v1/models", "auth": True},
    {"path": f"{EMBEDDINGS}/health", "target": "https://host-003.invalid/health", "auth": True},
    {
        "path": "/v1/solver/solution",
        "target": "https://host-005.invalid/solver/solution",
        "auth": True,
        "include_subpath": True,
    },
    {
        "path": "/v1/images/generations/image-model",
        "target": "https://host-024.invalid/v1/images/generations",
        "auth": True,
    },
    {
        "path": "/v1/audio/vendor/speech-model/transcriptions",
        "target": "https://host-010.invalid/v1/audio/transcriptions",
        "auth": True,
    },
)
GENERAL_SETTINGS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "disable_budget_reservation": True,
        "forward_client_headers_to_llm_api": True,
        "allow_requests_on_db_unavailable": True,
        "store_model_in_db": False,
        "use_x_forwarded_for": True,
        "enable_jwt_auth": True,
    }
)
MAPPINGS: Final[Mapping[str, str]] = MappingProxyType(
    {
        **{f"client-team-a{suffix}": TEAM_KEY for suffix in ("", "-2", "-3", "-4", "-5", "-6", "-7")},
        "client-blocked": BLOCKED_KEY,
        "client-expired": EXPIRED_KEY,
        "client-pt": GRANTED_KEY,
        "client-gone": DELETED_KEY,
    }
)

CHAT: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"model": "model-a", "messages": [{"role": "user", "content": "hi"}]}
)
MEMBER: Final[Mapping[str, JsonValue]] = MappingProxyType({"oid": "new-oid", "upn": "new@corp.invalid"})

Database = Literal["up", "down", "down_teams_uncached"]


@dataclass(frozen=True, slots=True)
class Call:
    """One request. `claims` are signed into a JWT, or `key` is sent as the bearer token."""

    claims: Mapping[str, JsonValue] | None = None
    key: str | None = None
    path: str = "/v1/chat/completions"
    method: str = "POST"
    body: Mapping[str, JsonValue] | None = CHAT
    headers: Mapping[str, str] = MappingProxyType({})
    token: Mapping[str, object] = EMPTY
    jwt_auth: Mapping[str, object] = EMPTY
    settings: Mapping[str, object] = EMPTY
    database: Database = "up"
    user_cached: bool = False


@dataclass(frozen=True, slots=True)
class Served:
    user_id: str | None
    team_id: str | None
    user_email: str | None = None
    end_user_id: str | None = None
    end_user_rpm_limit: int | None = None
    user_role: str = "internal_user"
    by_key: bool = False
    created: tuple[str, ...] = ()
    added: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Refused:
    """`logged_as` is the class the failure spend row records, and `message` how the client's message opens."""

    code: str
    error_type: str
    logged_as: str
    message: str
    created: tuple[str, ...] = ()
    added: tuple[tuple[str, str], ...] = ()


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _signed(claims: Mapping[str, JsonValue], options: Mapping[str, object]) -> str:
    expires_in: Final = options.get("expires_in", 600)
    typ: Final = options.get("typ")
    assert isinstance(expires_in, int)
    return jwt.encode(
        {"exp": int(time.time()) + expires_in, **claims},
        OTHER_KEY if options.get("key") == "OTHER_KEY" else SIGNING_KEY,
        algorithm="RS256",
        headers={"kid": str(options.get("kid", "k1")), **({"typ": str(typ)} if typ else {})},
    )


class _Jwks:
    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        key: Final = jwt.algorithms.RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(200, json={"keys": [{**key, "kid": "k1"}]}, request=httpx.Request("GET", url))


TEAMS: Final = (
    LiteLLM_TeamTable(team_id="team-a", team_alias="Team A", models=["model-a"]),
    LiteLLM_TeamTable(team_id="team-b", team_alias="Team B", models=["model-b"]),
    LiteLLM_TeamTable(team_id="team-pt", models=["model-a"], metadata={"allowed_passthrough_routes": list(GRANTS)}),
    LiteLLM_TeamTable(team_id="team-blocked", models=["model-a"], blocked=True),
    LiteLLM_TeamTable(team_id="team-all", models=[]),
)
EXISTING_USER: Final = LiteLLM_UserTable(
    user_id="existing-oid", user_email="existing@corp.invalid", user_role="internal_user", organization_memberships=[]
)


def _keys() -> Mapping[str, UserAPIKeyAuth]:
    return {
        TEAM_KEY: UserAPIKeyAuth(
            token=_hash(TEAM_KEY), team_id="team-a", user_id="key-owner", models=[], team_models=["model-a"]
        ),
        PERSONAL_KEY: UserAPIKeyAuth(token=_hash(PERSONAL_KEY), user_id="key-owner", models=["model-a"]),
        GRANTED_KEY: UserAPIKeyAuth(
            token=_hash(GRANTED_KEY),
            team_id="team-pt",
            models=[],
            team_models=["model-a"],
            team_metadata={"allowed_passthrough_routes": list(GRANTS)},
        ),
        BLOCKED_KEY: UserAPIKeyAuth(token=_hash(BLOCKED_KEY), blocked=True, models=["model-a"]),
        EXPIRED_KEY: UserAPIKeyAuth(
            token=_hash(EXPIRED_KEY), expires=datetime.now(timezone.utc) - timedelta(hours=1), models=["model-a"]
        ),
    }


def _cache(database: Database, user_cached: bool) -> UserApiKeyCache:
    cache: Final = UserApiKeyCache()
    if database != "down_teams_uncached":
        for team in TEAMS:
            cache.set_cache(f"team_id:{team.team_id}", team, model_type=LiteLLM_TeamTable)
    if database == "up":
        cache.set_cache(
            end_user_cache_key("customer-over"),
            LiteLLM_EndUserTable(
                user_id="customer-over",
                blocked=False,
                spend=5.0,
                litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
            ),
            model_type=LiteLLM_EndUserTable,
        )
        cache.set_cache(
            end_user_cache_key("customer-known"),
            LiteLLM_EndUserTable(
                user_id="customer-known",
                blocked=False,
                litellm_budget_table=LiteLLM_BudgetTable(rpm_limit=3, tpm_limit=9),
            ),
            model_type=LiteLLM_EndUserTable,
        )
        for key, row in _keys().items():
            cache.set_cache(_hash(key), row, model_type=UserAPIKeyAuth)
    if database == "up" or user_cached:
        cache.set_cache(EXISTING_USER.user_id, EXISTING_USER, model_type=LiteLLM_UserTable)
    return cache


class _UnreachableUsers(UserTable):
    async def find_unique(self, where: Mapping[str, object], include: object = None) -> LiteLLM_UserTable | None:
        raise httpx.ConnectError("connection refused")


async def _mapping(where: Mapping[str, object], **_: object) -> SimpleNamespace | None:
    key: Final = MAPPINGS.get(str(where.get("jwt_claim_value")))
    return None if key is None or where.get("jwt_issuer") else SimpleNamespace(token=_hash(key))


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


def _prisma(users: UserTable, database: Database) -> MagicMock:
    """A database holding only users and key mappings. Teams, keys and customers are served from the cache, and an
    unreachable database fails the key, team and user reads."""
    read: Final = (
        AsyncMock(return_value=None)
        if database == "up"
        else AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    )
    tables: Final = _Tables(
        litellm_teamtable=SimpleNamespace(find_unique=read, find_first=read, find_many=_teams_by_alias),
        litellm_usertable=users,
        litellm_jwtkeymapping=SimpleNamespace(find_first=AsyncMock(side_effect=_mapping)),
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


def _credential(call: Call) -> str:
    if call.claims is not None:
        return f"Bearer {_signed(call.claims, call.token)}"
    return "" if call.key is None else f"Bearer {call.key}"


async def _outcome(monkeypatch: pytest.MonkeyPatch, call: Call) -> Served | Refused:
    users: Final = (UserTable if call.database == "up" else _UnreachableUsers)(EXISTING_USER)
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
        "prisma_client": _prisma(users, call.database),
        "user_api_key_cache": _cache(call.database, call.user_cached),
        "premium_user": True,
        "master_key": MASTER_KEY,
        "user_custom_auth": None,
        "llm_router": None,
        "general_settings": {**GENERAL_SETTINGS, **call.settings},
        "config_passthrough_endpoints": None,
        "jwt_http_handler": _Jwks(),
        "litellm_jwtauth": LiteLLM_JWTAuth(**{**JWT_AUTH, **call.jwt_auth}),
    }.items():
        monkeypatch.setattr(proxy_server, name, value)
    await pass_through.initialize_pass_through_endpoints([dict(endpoint) for endpoint in PASS_THROUGH])
    try:
        auth: Final = await user_api_key_auth(request=_request(call), api_key=_credential(call))
    except ProxyException as error:
        return Refused(
            code=str(error.code),
            error_type=str(getattr(error.type, "value", error.type)),
            logged_as=type(failure_hook.call_args.kwargs["original_exception"]).__name__,
            message=str(error.message),
            created=tuple(row.user_id for row in users.rows[1:]),
            added=tuple(added),
        )
    finally:
        await pass_through.initialize_pass_through_endpoints([])
    return Served(
        user_id=auth.user_id,
        team_id=auth.team_id,
        user_email=auth.user_email,
        end_user_id=auth.end_user_id,
        end_user_rpm_limit=auth.end_user_rpm_limit,
        user_role=str(auth.user_role.value) if auth.user_role is not None else "",
        by_key=auth.token is not None,
        created=tuple(row.user_id for row in users.rows[1:]),
        added=tuple(added),
    )


NEW_MEMBER_OF_TEAM_A: Final = Served(
    "new-oid", "team-a", user_email="new@corp.invalid", created=("new-oid",), added=(("team-a", "new-oid"),)
)
NEW_MEMBER_OF_GRANTING_TEAM: Final = Served(
    "new-oid", "team-pt", user_email="new@corp.invalid", created=("new-oid",), added=(("team-pt", "new-oid"),)
)
OUTAGE_FALLBACK: Final = Served("__db_unavailable_fallback__", None, by_key=True)
PROXY_ADMIN: Final = Served("default_user_id", None, user_role="proxy_admin", by_key=True)
MODEL_A_DENIED: Final = (
    "The requested model 'model-a' is not available for this API key, or the model name is invalid. Check the models "
    "available to you and try again."
)
MODEL_B_DENIED: Final = MODEL_A_DENIED.replace("model-a", "model-b")
NO_TEAMS_IN_TOKEN: Final = (
    "No teams found in token. `enforce_team_based_model_access` is set to True. Token must belong to a team."
)
CUSTOMER_OVER_BUDGET: Final = "ExceededBudget: End User=customer-over over budget. Spend=5.0, Budget=1.0"
DATABASE_UNREACHABLE: Final = (
    "Service Unavailable, the authentication database is temporarily unreachable. Please retry shortly."
)
KEY_BLOCKED: Final = "Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin."
INVALID_KEY: Final = "Authentication Error, Invalid API key provided."
ADMIN_ONLY_ROUTE: Final = (
    "Authentication Error, Only proxy admin can be used to generate, delete, update info for new keys/users/teams. "
    "Route=/key/generate. Your role=unknown. Your user_id=*******"
)

CASES: Final = (
    pytest.param(Call(claims={**MEMBER, "roles": ["team-a"]}), NEW_MEMBER_OF_TEAM_A, id="jwt_one_team_with_access"),
    pytest.param(
        Call(claims={"oid": "existing-oid", "upn": "existing@corp.invalid", "roles": ["team-a"]}),
        Served("existing-oid", "team-a", user_email="existing@corp.invalid", added=(("team-a", "existing-oid"),)),
        id="jwt_existing_user_one_team",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-b", "team-a"]}), NEW_MEMBER_OF_TEAM_A, id="jwt_second_team_has_the_model"
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-b"]}),
        Refused(
            "403",
            "team_model_access_denied",
            "ModelAccessDeniedProxyException",
            MODEL_A_DENIED,
            created=("new-oid",),
            added=(("team-b", "new-oid"),),
        ),
        id="jwt_no_team_has_the_model",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["no-such-team", "team-a"]}),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_unknown_team_then_known",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["no-such-team"]}),
        Refused("403", "auth_error", "HTTPException", "No team has access to the requested model: model-a."),
        id="jwt_only_unknown_team",
    ),
    pytest.param(
        Call(claims=MEMBER), Refused("403", "auth_error", "HTTPException", NO_TEAMS_IN_TOKEN), id="jwt_no_roles_claim"
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": []}),
        Refused("403", "auth_error", "HTTPException", NO_TEAMS_IN_TOKEN),
        id="jwt_empty_roles",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": "team-a"}),
        Refused("403", "auth_error", "HTTPException", NO_TEAMS_IN_TOKEN),
        id="jwt_roles_as_string",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": [7, None, "team-a"]}), NEW_MEMBER_OF_TEAM_A, id="jwt_roles_with_non_string"
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-all"]}),
        Served(
            "new-oid", "team-all", user_email="new@corp.invalid", created=("new-oid",), added=(("team-all", "new-oid"),)
        ),
        id="jwt_all_models_team",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-blocked"]}),
        Refused(
            "401",
            "auth_error",
            "Exception",
            "Authentication Error, Team=team-blocked is blocked. Update via `/team/unblock` if you're an admin.",
            created=("new-oid",),
            added=(("team-blocked", "new-oid"),),
        ),
        id="jwt_blocked_team",
    ),
    pytest.param(
        Call(claims={"upn": "new@corp.invalid", "roles": ["team-a"]}),
        Served(
            "new@corp.invalid",
            "team-a",
            user_email="new@corp.invalid",
            created=("new@corp.invalid",),
            added=(("team-a", "new@corp.invalid"),),
        ),
        id="jwt_no_oid",
    ),
    pytest.param(
        Call(claims={"oid": "new-oid", "roles": ["team-a"]}),
        Served("new-oid", "team-a", created=("new-oid",), added=(("team-a", "new-oid"),)),
        id="jwt_no_upn",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-b", "team-a"]}, body={"messages": []}),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_no_model_in_body",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-all", "team-a"]}, headers={"x-litellm-team-id": "team-a"}),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_header_picks_allowed_team",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, headers={"x-litellm-team-id": "team-b"}),
        Refused(
            "403",
            "auth_error",
            "HTTPException",
            "x-litellm-team-id 'team-b' does not resolve to a team id or a unique team alias in your JWT's allowed teams.",
        ),
        id="jwt_header_team_not_in_roles",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a", "team-b"]}, headers={"x-litellm-team-id": "team-b"}),
        Refused(
            "403",
            "team_model_access_denied",
            "ModelAccessDeniedProxyException",
            MODEL_A_DENIED,
            created=("new-oid",),
            added=(("team-b", "new-oid"),),
        ),
        id="jwt_header_team_without_model",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, headers={"x-litellm-team-id": "Team A"}),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_header_alias",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-all", "team-a"]}, headers={"x-litellm-team-id": "Team A"}),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_header_alias_picks_second_role",
    ),
    pytest.param(
        Call(
            claims={**MEMBER, "roles": ["team-a"]},
            body={**CHAT, "model": "model-b"},
            headers={"x-litellm-team-id": "Team B"},
        ),
        Refused(
            "403",
            "auth_error",
            "HTTPException",
            "x-litellm-team-id 'Team B' does not resolve to a team id or a unique team alias in your JWT's allowed teams.",
        ),
        id="jwt_header_alias_outside_roles",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, path="/v1/messages", body={**CHAT, "max_tokens": 5}),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_anthropic_route",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, path="/v1/responses", body={"model": "model-a", "input": "hi"}),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_responses_route",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, path="/v1/models", method="GET", body=None),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_info_route",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, path="/key/generate", body={}),
        Served("new-oid", None, user_email="new@corp.invalid", created=("new-oid",)),
        id="jwt_management_route",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, path="/team/info", method="GET", body=None),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_team_info_route",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, path="/user/info", method="GET", body=None),
        NEW_MEMBER_OF_TEAM_A,
        id="jwt_user_info_route",
    ),
    pytest.param(
        Call(claims={**MEMBER, "scope": "litellm_proxy_admin"}, path="/key/generate", body={}),
        Served("new-oid", None, user_email="new@corp.invalid", user_role="proxy_admin"),
        id="jwt_admin_scope_management_route",
    ),
    pytest.param(
        Call(claims={**MEMBER, "scope": "litellm_proxy_admin"}),
        Refused(
            "401",
            "auth_error",
            "Exception",
            "Authentication Error, Admin not allowed to access this route. Route=/v1/chat/completions,",
        ),
        id="jwt_admin_scope_chat",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, body={**CHAT, "user": "customer-known"}),
        Served(
            "new-oid",
            "team-a",
            user_email="new@corp.invalid",
            end_user_id="customer-known",
            end_user_rpm_limit=3,
            created=("new-oid",),
            added=(("team-a", "new-oid"),),
        ),
        id="jwt_body_user_known_customer",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, body={**CHAT, "user": "customer-new"}),
        Served(
            "new-oid",
            "team-a",
            user_email="new@corp.invalid",
            end_user_id="customer-new",
            created=("new-oid",),
            added=(("team-a", "new-oid"),),
        ),
        id="jwt_body_user_unknown_customer",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, body={**CHAT, "user": "customer-over"}),
        Refused(
            "422",
            "budget_exceeded",
            "BudgetExceededError",
            CUSTOMER_OVER_BUDGET,
            created=("new-oid",),
            added=(("team-a", "new-oid"),),
        ),
        id="jwt_body_user_over_budget",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, headers={"x-litellm-end-user-id": "customer-known"}),
        Served(
            "new-oid",
            "team-a",
            user_email="new@corp.invalid",
            end_user_id="customer-known",
            end_user_rpm_limit=3,
            created=("new-oid",),
            added=(("team-a", "new-oid"),),
        ),
        id="jwt_header_end_user",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, path="/v1/embeddings/vendor/embed-model", body={"input": "x"}),
        Refused(
            "403",
            "auth_error",
            "HTTPException",
            "Team not allowed to access passthrough route /v1/embeddings/vendor/embed-model. Configure `allowed_passthrough_routes` on the team.",
        ),
        id="jwt_pass_through_team_without_grant",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-pt"]}, path="/v1/embeddings/vendor/embed-model", body={"input": "x"}),
        NEW_MEMBER_OF_GRANTING_TEAM,
        id="jwt_pass_through_team_with_grant",
    ),
    pytest.param(
        Call(
            claims={**MEMBER, "roles": ["team-pt"]},
            path="/v1/embeddings/vendor/embed-model/models",
            method="GET",
            body=None,
        ),
        NEW_MEMBER_OF_GRANTING_TEAM,
        id="jwt_pass_through_child_path_of_grant",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-pt"]}, path="/v1/images/generations/image-model", body={"prompt": "x"}),
        Refused(
            "403",
            "auth_error",
            "HTTPException",
            "Team not allowed to access passthrough route /v1/images/generations/image-model. Configure `allowed_passthrough_routes` on the team.",
        ),
        id="jwt_pass_through_ungranted_endpoint",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-pt"]}, path="/vendor/v1/systemone", body={"x": 1}),
        NEW_MEMBER_OF_GRANTING_TEAM,
        id="jwt_pass_through_post_only_with_post",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-pt"]}, path="/vendor/v1/systemone", method="GET", body=None),
        NEW_MEMBER_OF_GRANTING_TEAM,
        id="jwt_pass_through_post_only_with_get",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-pt"]}, path="/v1/solver/solution/job-1", method="GET", body=None),
        Refused(
            "403",
            "auth_error",
            "HTTPException",
            "Key/team not allowed to access passthrough route /v1/solver/solution/job-1. Configure `allowed_passthrough_routes` on the team or key.",
            created=("new-oid",),
        ),
        id="jwt_pass_through_subpath",
    ),
    pytest.param(
        Call(
            claims={**MEMBER, "roles": ["team-a", "team-pt"]},
            path="/v1/embeddings/vendor/embed-model",
            body={"input": "x"},
        ),
        NEW_MEMBER_OF_GRANTING_TEAM,
        id="jwt_pass_through_second_team_grants",
    ),
    pytest.param(
        Call(
            claims={**MEMBER, "roles": ["team-a", "team-pt"]},
            path="/v1/embeddings/vendor/embed-model",
            body={"input": "x"},
            headers={"x-litellm-team-id": "team-pt"},
        ),
        NEW_MEMBER_OF_GRANTING_TEAM,
        id="jwt_pass_through_header_team_grants",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-pt"]}, path="/v1/audio/vendor/speech-model/transcriptions", body={}),
        Refused(
            "403",
            "auth_error",
            "HTTPException",
            "Team not allowed to access passthrough route /v1/audio/vendor/speech-model/transcriptions. Configure `allowed_passthrough_routes` on the team.",
        ),
        id="jwt_pass_through_audio",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, token={"expires_in": -60}),
        Refused("401", "expired_key", "ProxyException", "Token Expired"),
        id="jwt_expired",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, token={"key": "OTHER_KEY"}),
        Refused("401", "auth_error", "Exception", "Authentication Error, Validation fails: "),
        id="jwt_wrong_signature",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, token={"kid": "rotated"}),
        Refused(
            "401", "auth_error", "NoMatchingJWTPublicKeyError", "Authentication Error, No matching public key found."
        ),
        id="jwt_unknown_kid",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, token={"typ": "oauth-id-jag+jwt"}),
        Refused(
            "401",
            "auth_error",
            "IdJagIsNotABearerTokenError",
            "Authentication Error, an ID-JAG is an authorization grant, not a bearer token; redeem it at the token endpoint",
        ),
        id="jwt_id_jag",
    ),
    pytest.param(
        Call(
            claims={"azp": "client-team-a", "customer": "customer-known"},
            body={**CHAT, "user": "customer-over"},
            jwt_auth=MAPPED,
        ),
        Served("key-owner", "team-a", end_user_id="customer-known", end_user_rpm_limit=3, by_key=True),
        id="mapped_claim_end_user_wins_over_over_budget_body_user",
    ),
    pytest.param(
        Call(
            claims={"azp": "client-team-a-2", "customer": "customer-new"},
            headers={"x-litellm-end-user-id": "customer-known"},
            jwt_auth=MAPPED,
        ),
        Served("key-owner", "team-a", end_user_id="customer-new", by_key=True),
        id="mapped_claim_end_user_wins_over_header_user",
    ),
    pytest.param(
        Call(
            claims={"azp": "client-team-a-3", "customer": "customer-over"},
            body={**CHAT, "user": "customer-known"},
            jwt_auth=MAPPED,
        ),
        Refused("422", "budget_exceeded", "BudgetExceededError", CUSTOMER_OVER_BUDGET),
        id="mapped_claim_over_budget_is_refused",
    ),
    pytest.param(
        Call(claims={"azp": "client-team-a-4"}, body={**CHAT, "user": "customer-known"}, jwt_auth=MAPPED),
        Served("key-owner", "team-a", end_user_id="customer-known", end_user_rpm_limit=3, by_key=True),
        id="mapped_no_claim_uses_body_user",
    ),
    pytest.param(
        Call(
            claims={"azp": "client-team-a-5", "customer": ""}, body={**CHAT, "user": "customer-known"}, jwt_auth=MAPPED
        ),
        Served("key-owner", "team-a", end_user_id="customer-known", end_user_rpm_limit=3, by_key=True),
        id="mapped_empty_claim_uses_body_user",
    ),
    pytest.param(
        Call(claims={**MEMBER, "azp": "client-team-a-6"}, jwt_auth=MAPPED),
        Served("key-owner", "team-a", by_key=True),
        id="mapped_no_end_user_anywhere",
    ),
    pytest.param(
        Call(claims={"azp": "client-team-a-7"}, body={**CHAT, "model": "model-b"}, jwt_auth=MAPPED),
        Refused("403", "team_model_access_denied", "ModelAccessDeniedProxyException", MODEL_B_DENIED),
        id="mapped_wrong_model",
    ),
    pytest.param(
        Call(claims={"azp": "client-blocked"}, jwt_auth=MAPPED),
        Refused("401", "auth_error", "ProxyException", KEY_BLOCKED),
        id="mapped_blocked_key",
    ),
    pytest.param(
        Call(claims={"azp": "client-expired"}, jwt_auth=MAPPED),
        Refused("401", "expired_key", "ProxyException", "Authentication Error - Expired Key. Key Expiry time "),
        id="mapped_expired_key",
    ),
    pytest.param(
        Call(claims={"azp": "client-gone"}, jwt_auth=MAPPED),
        Refused("401", "token_not_found_in_db", "KeyNotFoundError", INVALID_KEY),
        id="mapped_deleted_key",
    ),
    pytest.param(
        Call(
            claims={"azp": "client-pt"}, path="/v1/embeddings/vendor/embed-model", body={"input": "x"}, jwt_auth=MAPPED
        ),
        Served(None, "team-pt", by_key=True),
        id="mapped_pass_through",
    ),
    pytest.param(
        Call(claims={**MEMBER, "azp": "client-unknown", "roles": ["team-a"]}, jwt_auth=MAPPED),
        NEW_MEMBER_OF_TEAM_A,
        id="mapped_unmapped_client_falls_back_to_roles",
    ),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, jwt_auth=MAPPED),
        NEW_MEMBER_OF_TEAM_A,
        id="mapped_no_client_claim_falls_back_to_roles",
    ),
    pytest.param(Call(key=TEAM_KEY), Served("key-owner", "team-a", by_key=True), id="key_team"),
    pytest.param(
        Call(key=TEAM_KEY, body={**CHAT, "model": "model-b"}),
        Refused("403", "team_model_access_denied", "ModelAccessDeniedProxyException", MODEL_B_DENIED),
        id="key_team_wrong_model",
    ),
    pytest.param(Call(key=PERSONAL_KEY), Served("key-owner", None, by_key=True), id="key_personal"),
    pytest.param(
        Call(key=TEAM_KEY, body={**CHAT, "user": "customer-known"}),
        Served("key-owner", "team-a", end_user_id="customer-known", end_user_rpm_limit=3, by_key=True),
        id="key_body_user_known",
    ),
    pytest.param(
        Call(key=TEAM_KEY, body={**CHAT, "user": "customer-over"}),
        Refused("422", "budget_exceeded", "BudgetExceededError", CUSTOMER_OVER_BUDGET),
        id="key_body_user_over_budget",
    ),
    pytest.param(
        Call(key=GRANTED_KEY, path="/v1/embeddings/vendor/embed-model", body={"input": "x"}),
        Served(None, "team-pt", by_key=True),
        id="key_pass_through_granted",
    ),
    pytest.param(
        Call(key=TEAM_KEY, path="/v1/embeddings/vendor/embed-model", body={"input": "x"}),
        Refused(
            "403",
            "auth_error",
            "HTTPException",
            "Key/team not allowed to access passthrough route /v1/embeddings/vendor/embed-model. Configure `allowed_passthrough_routes` on the team or key.",
        ),
        id="key_pass_through_not_granted",
    ),
    pytest.param(
        Call(key=TEAM_KEY, path="/key/generate", body={}),
        Refused("401", "auth_error", "ProxyException", ADMIN_ONLY_ROUTE),
        id="key_management_route",
    ),
    pytest.param(Call(key=BLOCKED_KEY), Refused("401", "auth_error", "ProxyException", KEY_BLOCKED), id="key_blocked"),
    pytest.param(
        Call(key=EXPIRED_KEY),
        Refused("401", "expired_key", "ProxyException", "Authentication Error - Expired Key. Key Expiry time "),
        id="key_expired",
    ),
    pytest.param(
        Call(key=UNKNOWN_KEY),
        Refused("401", "token_not_found_in_db", "KeyNotFoundError", INVALID_KEY),
        id="key_unknown",
    ),
    pytest.param(
        Call(key="not-a-key"),
        Refused(
            "401",
            "auth_error",
            "HTTPException",
            "LiteLLM Virtual Key expected. Received=not-****-key, expected to start with 'sk-'.",
        ),
        id="key_not_virtual",
    ),
    pytest.param(
        Call(key=None),
        Refused(
            "401",
            "auth_error",
            "ProxyException",
            "Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix.",
        ),
        id="key_missing",
    ),
    pytest.param(Call(key=MASTER_KEY), PROXY_ADMIN, id="master_key_chat"),
    pytest.param(Call(key=MASTER_KEY, path="/key/generate", body={}), PROXY_ADMIN, id="master_key_management"),
    pytest.param(
        Call(key=MASTER_KEY, body={**CHAT, "user": "customer-known"}),
        Served(
            "default_user_id",
            None,
            end_user_id="customer-known",
            end_user_rpm_limit=3,
            user_role="proxy_admin",
            by_key=True,
        ),
        id="master_key_body_user",
    ),
    pytest.param(
        Call(key=MASTER_KEY, path="/v1/embeddings/vendor/embed-model", body={"input": "x"}),
        PROXY_ADMIN,
        id="master_key_pass_through",
    ),
    pytest.param(Call(key=TEAM_KEY, database="down"), OUTAGE_FALLBACK, id="db_down_key"),
    pytest.param(Call(claims={**MEMBER, "roles": ["team-a"]}, database="down"), OUTAGE_FALLBACK, id="db_down_jwt"),
    pytest.param(
        Call(claims={**MEMBER, "roles": ["team-a"]}, database="down_teams_uncached"),
        Refused("403", "auth_error", "HTTPException", "No team has access to the requested model: model-a."),
        id="db_down_jwt_teams_uncached",
    ),
    pytest.param(
        Call(
            claims={"oid": "existing-oid", "upn": "existing@corp.invalid", "roles": ["team-a"]},
            database="down",
            user_cached=True,
        ),
        Served("existing-oid", "team-a", user_email="existing@corp.invalid", added=(("team-a", "existing-oid"),)),
        id="db_down_jwt_known_user_cached",
    ),
    pytest.param(
        Call(
            claims={**MEMBER, "roles": ["team-a"]},
            settings={"allow_requests_on_db_unavailable": False},
            database="down_teams_uncached",
        ),
        Refused("403", "auth_error", "HTTPException", "No team has access to the requested model: model-a."),
        id="db_down_jwt_teams_uncached_not_allowed",
    ),
    pytest.param(
        Call(key=TEAM_KEY, path="/key/generate", body={}, database="down"),
        Refused("401", "auth_error", "ProxyException", ADMIN_ONLY_ROUTE),
        id="db_down_key_unknown_route",
    ),
    pytest.param(Call(key=MASTER_KEY, database="down"), PROXY_ADMIN, id="db_down_master_key"),
    pytest.param(
        Call(key=TEAM_KEY, settings={"allow_requests_on_db_unavailable": False}, database="down"),
        Refused("503", "no_db_connection", "ConnectError", DATABASE_UNREACHABLE),
        id="db_down_key_not_allowed",
    ),
    pytest.param(
        Call(
            claims={**MEMBER, "roles": ["team-a"]},
            settings={"allow_requests_on_db_unavailable": False},
            database="down",
        ),
        Refused("503", "no_db_connection", "ConnectError", DATABASE_UNREACHABLE),
        id="db_down_jwt_not_allowed",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("call", "expected"), CASES)
async def test_a_request_is_served_or_refused_as_on_the_python_auth_path(
    monkeypatch: pytest.MonkeyPatch, call: Call, expected: Served | Refused
) -> None:
    outcome: Final = await _outcome(monkeypatch, call)

    if isinstance(outcome, Refused) and isinstance(expected, Refused):
        assert outcome.message.startswith(expected.message), outcome.message
        assert replace(outcome, message=expected.message) == expected
    else:
        assert outcome == expected
