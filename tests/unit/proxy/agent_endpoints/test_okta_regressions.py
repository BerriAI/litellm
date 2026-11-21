import json
import os
from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.requests import Request

from litellm.proxy import proxy_server
from litellm.proxy._types import (
    IdJagRequestingApp,
    JWTIssuerConfig,
    LiteLLM_JWTAuth,
    LiteLLM_UserTable,
    ProxyException,
)
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken
from litellm.proxy.auth.jwt import ID_JAG_TYP
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_builder
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server.gateway_dcr_flow import AssertionIdentity, SubjectIdentity
from litellm.proxy.mcp_server.id_jag_grant import AssertionRedemption, IdJagPrerequisites, identity_from_id_jag
from tests.unit.proxy._prisma import prisma_with_empty_config
from tests.unit.proxy.rust_auth_stores import RustAuthStores, install_rust_auth_stores

CUSTOM: Final = "https://acme.okta.example/oauth2/aus1agents"
ORG_SERVER: Final = "https://acme.okta.example"
AUDIENCE: Final = "api://litellm"
APP: Final = "0oa1agentapp"
MARIA: Final = "maria@example.com"
JWKS_URL: Final = "https://acme.okta.example/jwks"
SIGNING_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
CLIENT_CREDENTIALS: Final = {"sub": APP, "cid": APP, "scp": ["invoices.read"]}
SIGN_IN: Final = {"sub": MARIA, "uid": "00u1maria", "cid": APP, "email": MARIA, "scp": ["invoices.read"]}
RECORD: Final = os.environ.get("OKTA_REGRESSION_RECORD")


class _Jwks:
    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        key: Final = jwt.algorithms.RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(200, json={"keys": [{**key, "kid": "k1"}]}, request=httpx.Request("GET", url))


def _token(claims: Mapping[str, object]) -> str:
    return jwt.encode(
        {"iss": CUSTOM, "aud": AUDIENCE, "exp": 4_102_444_800, **claims},
        SIGNING_KEY,
        algorithm="RS256",
        headers={"kid": "k1"},
    )


def _request() -> Request:
    body: Final = {"model": "gpt-test", "messages": []}
    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "path_params": {},
            "parsed_body": (tuple(body), body),
        }
    )
    request._body = json.dumps(body).encode()
    return request


class Plain:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, stores: RustAuthStores) -> None:
        self.client: Final = prisma_with_empty_config()
        self.client.writer_db.litellm_agentidentity.find_unique = AsyncMock(return_value=None)
        self.client.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
        stores.user.rows[MARIA] = {
            "user_id": MARIA,
            "user_email": MARIA,
            "user_role": "internal_user",
            "teams": [],
            "metadata": {},
        }
        for name, value in {
            "prisma_client": self.client,
            "user_api_key_cache": UserApiKeyCache(),
            "proxy_logging_obj": MagicMock(post_call_failure_hook=AsyncMock(return_value=None)),
            "general_settings": {"enable_jwt_auth": True},
            "premium_user": True,
            "master_key": "sk-test-master",
            "llm_router": None,
            "user_custom_auth": None,
            "litellm_proxy_admin_name": "admin",
            "jwt_http_handler": _Jwks(),
        }.items():
            monkeypatch.setattr(proxy_server, name, value)
        monkeypatch.setenv("JWT_PUBLIC_KEY_URL", JWKS_URL)
        self.monkeypatch: Final = monkeypatch

    def configure(self, **settings: object) -> None:
        issuers: Final = [
            JWTIssuerConfig(issuer=issuer, jwks_url=JWKS_URL, audience=AUDIENCE) for issuer in (CUSTOM, ORG_SERVER)
        ]
        self.monkeypatch.setattr(proxy_server, "litellm_jwtauth", LiteLLM_JWTAuth(issuers=issuers, **settings))

    async def outcome(self, claims: Mapping[str, object]) -> list[object]:
        try:
            auth: Final = await user_api_key_auth_builder(
                request=_request(),
                api_key=f"Bearer {_token(claims)}",
                azure_api_key_header="",
                anthropic_api_key_header=None,
                google_ai_studio_api_key_header=None,
                azure_apim_header=None,
                request_data={"model": "gpt-test", "messages": []},
            )
        except ProxyException as refused:
            return ["refused", refused.code, refused.message]
        return ["admitted", auth.agent_id, auth.user_id, str(auth.user_role), auth.team_id]

    def identity_reads(self) -> int:
        return self.client.writer_db.litellm_agentidentity.find_unique.await_count


@pytest.fixture
def plain(monkeypatch: pytest.MonkeyPatch) -> Plain:
    return Plain(monkeypatch, install_rust_auth_stores(monkeypatch))


_TOKENS: Final = {
    "client_credentials": CLIENT_CREDENTIALS,
    "person": SIGN_IN,
    "actor": {**SIGN_IN, "act": {"sub": APP, "act": {"sub": "0oa1service"}}},
    "org_server_person": {**SIGN_IN, "iss": ORG_SERVER},
}
_SETTINGS: Final = {"no_user_fields": {}, "email_user_field": {"user_email_jwt_field": "email"}}
MAIN: Final[dict[str, list[object]]] = {
    "actor/email_user_field": ["admitted", None, "maria@example.com", "internal_user", None],
    "actor/no_user_fields": ["admitted", None, None, "internal_user", None],
    "client_credentials/email_user_field": ["admitted", None, None, "internal_user", None],
    "client_credentials/no_user_fields": ["admitted", None, None, "internal_user", None],
    "org_server_person/email_user_field": ["admitted", None, "maria@example.com", "internal_user", None],
    "org_server_person/no_user_fields": ["admitted", None, None, "internal_user", None],
    "person/email_user_field": ["admitted", None, "maria@example.com", "internal_user", None],
    "person/no_user_fields": ["admitted", None, None, "internal_user", None],
}


def _record(case: str, outcome: object) -> None:
    if RECORD:
        with open(RECORD, "a") as out:
            out.write(json.dumps({"case": case, "outcome": outcome}) + "\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("settings", sorted(_SETTINGS))
@pytest.mark.parametrize("token", sorted(_TOKENS))
async def test_okta_tokens_from_an_issuer_not_opted_in_get_mains_outcome_and_no_identity_read(
    plain: Plain, token: str, settings: str
) -> None:
    plain.configure(**_SETTINGS[settings])
    case: Final = f"{token}/{settings}"

    outcome: Final = await plain.outcome(_TOKENS[token])

    _record(case, outcome)
    assert outcome == MAIN[case]
    assert plain.identity_reads() == 0


NOW: Final = 1_800_000_000.0
GATEWAY: Final = "https://gateway.example.com"


async def _maria(claims: Mapping[str, object], team: str | None) -> SubjectIdentity:
    return SubjectIdentity(user_id=MARIA, team_id=team)


@pytest.mark.asyncio
@pytest.mark.parametrize("issuer", [CUSTOM, ORG_SERVER])
async def test_an_id_jag_from_an_issuer_not_opted_in_redeems_as_on_main(plain: Plain, issuer: str) -> None:
    plain.configure()
    claims: Final = {
        "iss": issuer,
        "sub": "00u1maria",
        "aud": GATEWAY,
        "exp": NOW + 300,
        "iat": NOW - 5,
        "jti": "jti-1",
        "client_id": APP,
        "scope": "litellm:mcp",
    }

    async def decode(assertion: str, audience: object, required: object) -> Mapping[str, object]:
        return claims

    identity: Final = await identity_from_id_jag(
        AssertionRedemption(
            assertion=jwt.encode(
                claims, "unused-by-the-fake-verifier-but-long-enough-for-hs256", headers={"typ": ID_JAG_TYP}
            ),
            client_id=APP,
            client_secret=None,
            authorization_header=None,
            team_id=None,
            audiences=(GATEWAY,),
            now=NOW,
            leeway_seconds=0,
        ),
        prerequisites=IdJagPrerequisites(
            jwt_auth_enabled=True,
            has_database=True,
            licensed=True,
            maps_jwts_to_virtual_keys=False,
            shared_replay_store=True,
            base_url=GATEWAY,
            requesting_apps=(IdJagRequestingApp(client_id=APP),),
        ),
        decode=decode,
        resolve_subject=_maria,
    )

    assert isinstance(identity, AssertionIdentity)
    dumped: Final = {key: value for key, value in identity.model_dump().items() if value is not None}
    _record(f"id_jag/{issuer}", dumped)
    assert dumped == {
        "user_id": MARIA,
        "replay_key": dumped["replay_key"],
        "replay_ttl_seconds": 300,
        "scope": "litellm:mcp",
        "allowed_routes": ("mcp_inference_routes",),
    }
    assert plain.identity_reads() == 0


_SESSION_FIELDS: Final = ("user_id", "team_id", "user_role", "is_session_token", "allowed_routes", "agent_id", "models")


def test_an_unstamped_session_credential_mints_the_fields_main_mints(plain: Plain) -> None:
    session: Final = ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        LiteLLM_UserTable(user_id=MARIA, user_email=MARIA, user_role="internal_user")
    )
    key: Final = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(session)
    assert key is not None
    fields: Final = {name: getattr(key, name) for name in _SESSION_FIELDS}
    serialized: Final = sorted(json.loads(key.model_dump_json(exclude_none=True)))

    _record("session/fields", [repr(fields), serialized])
    assert fields == {
        "user_id": MARIA,
        "team_id": None,
        "user_role": key.user_role,
        "is_session_token": True,
        "allowed_routes": [],
        "agent_id": None,
        "models": [],
    }
    assert serialized == [
        "aliases",
        "allowed_cache_controls",
        "allowed_routes",
        "auto_rotate",
        "budget_fallbacks",
        "config",
        "expires",
        "is_session_token",
        "key_alias",
        "key_name",
        "metadata",
        "model_max_budget",
        "model_spend",
        "models",
        "permissions",
        "rotation_count",
        "soft_budget_cooldown",
        "spend",
        "team_blocked",
        "team_models",
        "token",
        "total_spend",
        "user_id",
        "user_role",
    ]


MAIN_SESSION: Final = "litellm_login_W8cRcwGqm64hMKo4qmD-Xj3ln0Lh-MYsfRPdoV1egLNPEFpdxfwluvJcaOXQPB-Zh-B_q9XP8-HtFZNrziz-TlQtNANLIqnWGmlP-6lJX_t1afiN67Ca2J6cs8Kk5J_Lo35qD3Zg-XbU_0n02KNOLnz2cIrxNErdrG-VpL2OWDLsWGOuOqb71qLWAPqZXF1NQuow6slH-ZM63vApFlSqaJW-Ci5umkvNaSpCfLi4UKgYhc7ErcjwM5JUskcKk6l3-fn2kmOBcsusVDtLqOox5AQ1b-5BZ5R9B3PGLK553qM1tbX-tnaU_-n5GFaQF9356ye4scgtsAUqmcjAUKGbRIB6sz8FAGW-sI7HazpHYPqjGlSppmx0KJ-13eKCYaGwitn5f4KTAaUsg5LDkPdK2BeO3gWTM1hvVUiNuapnMXFemXDya0SoUTpCiRw1brn-CMyQI6FQLRWtJRKi6QNL4rcXreTLf3qVt-fWFZEK67xXFJ64PYQeE8JL6eTtzgh5QjgoYO-EVLW2ggMK3DsI4Rt6n5jzGTXzvV83D_z-e2pQFOPa3LKC5DC9URNoV-5Wo7SW10Qutagnb7Sbdd4fc4kfM_lUldE2gfzyWekCprF0d1OKh8R0lP5hAnM7B5d2ZbkE5HnIhIJc9-_zLESqPDmcVo5eVjRlU45k1Q3LwFmAGiLzK5bT6G0hSCXswZLj7JcRNfre4j8zhuA4Gb4X_azUBaTf9-d60o40T6XpeVABmm371zzbwEIHTwXfV0_5qfQ"


def test_a_session_credential_main_minted_still_decodes_to_the_same_fields(plain: Plain) -> None:
    key: Final = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(MAIN_SESSION)

    assert key is not None
    assert {name: getattr(key, name) for name in _SESSION_FIELDS} == {
        "user_id": MARIA,
        "team_id": None,
        "user_role": key.user_role,
        "is_session_token": True,
        "allowed_routes": [],
        "agent_id": None,
        "models": [],
    }
    assert getattr(key, "agent_binding_revision", None) is None


_CONFIGS: Final = {
    "global_fields": {"user_email_jwt_field": "email", "issuers": [{"issuer": CUSTOM, "audience": AUDIENCE}]},
    "per_issuer_fields": {
        "issuers": [
            {"issuer": CUSTOM, "audience": [AUDIENCE, "api://other"], "user_id_jwt_field": "uid"},
            {"issuer": ORG_SERVER, "disable_audience_validation": True, "user_email_jwt_field": "email"},
        ]
    },
    "entra": {
        "user_id_jwt_field": "oid",
        "issuers": [
            {
                "issuer": "https://login.microsoftonline.com/11111111-1111-4111-8111-111111111111/v2.0",
                "jwks_url": "https://login.microsoftonline.com/common/discovery/v2.0/keys",
                "audience": "api://litellm",
            }
        ],
    },
}
MAIN_READINGS: Final[dict[str, list[object]]] = {
    "entra": ["oid-1", None],
    "global_fields": [MARIA, MARIA],
    "per_issuer_fields": [None, None],
}


@pytest.mark.parametrize("config", sorted(_CONFIGS))
def test_an_existing_jwt_config_loads_in_both_configs_and_reads_claims_as_on_main(config: str) -> None:
    from litellm.proxy.auth.jwt import claims

    jwtauth: Final = LiteLLM_JWTAuth.model_validate(_CONFIGS[config])
    issuer: Final = _CONFIGS[config]["issuers"][0]["issuer"]
    reading: Final = claims(
        {"iss": issuer, "sub": "00u1maria", "uid": "00u1maria", "oid": "oid-1", "email": MARIA}, jwtauth
    )
    observed: Final = [reading.user_id, reading.user_email]

    _record(f"config/{config}", observed)
    assert observed == MAIN_READINGS[config]
