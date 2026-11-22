import json
from collections.abc import Callable, Mapping
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from starlette.requests import Request

from litellm.proxy import proxy_server
from litellm.proxy._types import (
    JWTIssuerConfig,
    LiteLLM_JWTAuth,
    LiteLLM_ObjectPermissionTable,
    ProxyException,
    ScopeMapping,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth_builder
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import AgentExecutionMode, AgentIdentityBinding
from tests.unit.proxy._prisma import prisma_with_empty_config
from tests.unit.proxy.rust_auth_stores import RustAuthStores, install_rust_auth_stores

ORG: Final = "acme"
CUSTOM: Final = "https://acme.okta.example/oauth2/aus1agents"
ORG_SERVER: Final = "https://acme.okta.example"
AUDIENCE: Final = "api://litellm"
APP: Final = "0oa1agentapp"
SIGNING_KEY: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
JWKS_URL: Final = "https://acme.okta.example/jwks"
TOKEN_EXPIRY: Final = 4_102_444_800


class _Jwks:
    async def get(self, url: str, params: object = None, headers: object = None) -> httpx.Response:
        key: Final = jwt.algorithms.RSAAlgorithm.to_jwk(SIGNING_KEY.public_key(), as_dict=True)
        return httpx.Response(200, json={"keys": [{**key, "kid": "k1"}]}, request=httpx.Request("GET", url))


def _token(claims: Mapping[str, object]) -> str:
    return jwt.encode(
        {"iss": CUSTOM, "aud": AUDIENCE, "exp": TOKEN_EXPIRY, **claims},
        SIGNING_KEY,
        algorithm="RS256",
        headers={"kid": "k1"},
    )


def _request(
    route: str = "/v1/chat/completions", method: str = "POST", headers: Mapping[str, str] | None = None
) -> Request:
    body: Final = {"model": "gpt-test", "messages": []}
    extra: Final = tuple((name.encode(), value.encode()) for name, value in (headers or {}).items())
    request: Final = Request(
        {
            "type": "http",
            "method": method,
            "path": route,
            "headers": [(b"content-type", b"application/json"), *extra],
            "query_string": b"",
            "path_params": {},
            "parsed_body": (tuple(body), body),
        }
    )
    request._body = json.dumps(body).encode()
    return request


def _issuers(okta_org: str | None = ORG) -> tuple[JWTIssuerConfig, ...]:
    return tuple(
        JWTIssuerConfig(
            issuer=issuer, jwks_url=JWKS_URL, audience=AUDIENCE, agent_identity_provider="okta", okta_org=okta_org
        )
        for issuer in (CUSTOM, ORG_SERVER)
    )


def okta_agent(
    *,
    mode: AgentExecutionMode = "autonomous",
    enabled: bool = True,
    active: bool = True,
    client: str = APP,
    scopes: tuple[str, ...] = (),
) -> AgentResponse:
    binding: Final = AgentIdentityBinding(
        agent_id="invoice-bot",
        active=active,
        provider="okta",
        issuer=CUSTOM,
        tenant_id=ORG,
        client_id=client,
        required_scopes=scopes,
        revision="r1",
    )
    return AgentResponse(
        agent_id="invoice-bot",
        agent_name="invoice-bot",
        agent_card_params={},
        identity_managed=True,
        identity=binding,
        enabled=enabled,
        execution_mode=mode,
    )


class Gateway:
    def __init__(self, monkeypatch: pytest.MonkeyPatch, stores: RustAuthStores) -> None:
        self.monkeypatch: Final = monkeypatch
        self.client: Final = prisma_with_empty_config()
        self.stores: Final = stores
        self.client.writer_db.litellm_agentidentity.find_unique = AsyncMock(return_value=None)
        self.client.writer_db.litellm_retiredagentidentity.find_unique = AsyncMock(return_value=None)
        self.client.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
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
        }.items():
            monkeypatch.setattr(proxy_server, name, value)
        monkeypatch.setenv("JWT_PUBLIC_KEY_URL", JWKS_URL)
        monkeypatch.setattr(proxy_server, "jwt_http_handler", _Jwks())
        monkeypatch.setattr(proxy_server, "litellm_jwtauth", LiteLLM_JWTAuth(issuers=list(_issuers())))

    def bind(self, agent: AgentResponse) -> None:
        assert agent.identity is not None
        self.client.writer_db.litellm_agentidentity.find_unique = AsyncMock(return_value=agent.identity)
        self.client.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=agent)
        self.stores.agent_identity.current.add((agent.agent_id, agent.identity.revision))

    async def send(
        self,
        claims: Mapping[str, object],
        route: str = "/v1/chat/completions",
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
    ) -> UserAPIKeyAuth:
        return await user_api_key_auth_builder(
            request=_request(route, method, headers),
            api_key=f"Bearer {_token(claims)}",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            request_data={"model": "gpt-test", "messages": []},
        )

    async def refusal(self, claims: Mapping[str, object], route: str = "/v1/chat/completions") -> ProxyException:
        with pytest.raises(ProxyException) as refused:
            await self.send(claims, route)
        return refused.value

    async def outcome(
        self,
        claims: Mapping[str, object],
        route: str = "/v1/chat/completions",
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
    ) -> tuple[object, ...]:
        try:
            auth: Final = await self.send(claims, route, method, headers)
        except ProxyException as refused:
            return ("refused", refused.code, refused.message)
        context: Final = auth.managed_agent_context
        return ("admitted", auth.agent_id, context.mode if context else None, auth.user_id)

    def configure(self, **settings: object) -> None:
        current: Final = proxy_server.litellm_jwtauth
        self.monkeypatch.setattr(proxy_server, "litellm_jwtauth", current.model_copy(update=settings))

    def opt_out(self) -> None:
        self.configure(
            issuers=[
                issuer.model_copy(update={"agent_identity_provider": None, "okta_org": None}) for issuer in _issuers()
            ]
        )

    def lookups(self) -> int:
        return self.client.writer_db.litellm_agentidentity.find_unique.await_count


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Gateway:
    return Gateway(monkeypatch, install_rust_auth_stores(monkeypatch))


CLIENT_CREDENTIALS: Final = {"sub": APP, "cid": APP, "scp": ["invoices.read"]}


@pytest.mark.asyncio
async def test_row_12_client_credentials_token_is_the_autonomous_agent(gateway: Gateway) -> None:
    gateway.bind(okta_agent())

    auth: Final = await gateway.send(CLIENT_CREDENTIALS)

    assert auth.agent_id == "invoice-bot"
    assert auth.managed_agent_context is not None
    assert auth.managed_agent_context.mode == "autonomous"


async def _plain_today(gateway: Gateway, claims: Mapping[str, object]) -> tuple[object, ...]:
    gateway.opt_out()
    today: Final = await gateway.outcome(claims)
    gateway.configure(issuers=list(_issuers()))
    return today


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("claims", "iss"),
    [
        pytest.param({"sub": "someone", "cid": APP}, CUSTOM, id="row_04_neither_person_nor_client_credentials"),
        pytest.param(CLIENT_CREDENTIALS, ORG_SERVER, id="row_02_access_token_from_the_org_server"),
    ],
)
async def test_tokens_okta_never_names_an_agent_in_take_the_plain_path_without_a_lookup(
    gateway: Gateway, claims: Mapping[str, object], iss: str
) -> None:
    gateway.bind(okta_agent())
    token_claims: Final = {**claims, "iss": iss}
    today: Final = await _plain_today(gateway, token_claims)
    before: Final = gateway.lookups()

    assert await gateway.outcome(token_claims) == today
    assert gateway.lookups() == before


@pytest.mark.asyncio
async def test_row_01_issuer_that_has_not_opted_in_never_looks_up_a_binding(gateway: Gateway) -> None:
    gateway.bind(okta_agent())
    gateway.opt_out()

    outcome: Final = await gateway.outcome(CLIENT_CREDENTIALS)

    assert outcome[0] == "refused" or outcome[1] is None
    assert gateway.lookups() == 0


@pytest.mark.asyncio
async def test_row_06_unbound_client_credentials_token_takes_the_plain_path(gateway: Gateway) -> None:
    today: Final = await _plain_today(gateway, CLIENT_CREDENTIALS)

    assert await gateway.outcome(CLIENT_CREDENTIALS) == today
    assert gateway.lookups() == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["autonomous", "both"])
async def test_row_12_client_credentials_token_runs_as_the_agent_in_autonomous_or_both_mode(
    gateway: Gateway, mode: AgentExecutionMode
) -> None:
    gateway.bind(okta_agent(mode=mode, scopes=("invoices.read",)))

    assert await gateway.outcome(CLIENT_CREDENTIALS) == ("admitted", "invoice-bot", "autonomous", None)
    assert gateway.stores.agent_identity.recorded == [("invoice-bot", "r1")]


def _retired(gateway: Gateway) -> None:
    gateway.client.writer_db.litellm_retiredagentidentity.find_unique = AsyncMock(return_value=MagicMock())


def _identity_read_fails(gateway: Gateway) -> None:
    gateway.client.writer_db.litellm_agentidentity.find_unique = AsyncMock(side_effect=RuntimeError("db down"))


def _bind(
    *, mode: AgentExecutionMode = "autonomous", enabled: bool = True, active: bool = True, scopes: tuple[str, ...] = ()
) -> Callable[[Gateway], None]:
    def bind(gateway: Gateway) -> None:
        gateway.bind(okta_agent(mode=mode, enabled=enabled, active=active, scopes=scopes))

    return bind


def _bind_other_issuer(gateway: Gateway) -> None:
    agent: Final = okta_agent()
    assert agent.identity is not None
    moved: Final = agent.model_copy(
        update={"identity": agent.identity.model_copy(update={"issuer": "https://acme.okta.example/oauth2/other"})}
    )
    gateway.bind(moved)


def _recording_fails(gateway: Gateway) -> None:
    gateway.bind(okta_agent())
    gateway.stores.agent_identity.error = RuntimeError("writer down")


def _revision_moved(gateway: Gateway) -> None:
    gateway.bind(okta_agent())
    gateway.stores.agent_identity.current.clear()
    gateway.stores.agent_identity.current.add(("invoice-bot", "r2"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "claims", "route", "code", "message"),
    [
        pytest.param(
            _identity_read_fails, CLIENT_CREDENTIALS, None, "503", "Agent identity could not be loaded", id="row_05"
        ),
        pytest.param(
            _retired, CLIENT_CREDENTIALS, None, "403", "This agent identity binding has been retired", id="row_07"
        ),
        pytest.param(
            _bind(enabled=False),
            CLIENT_CREDENTIALS,
            None,
            "403",
            "Agent is disabled or no longer bound to an identity",
            id="row_09_disabled",
        ),
        pytest.param(
            _bind(active=False),
            CLIENT_CREDENTIALS,
            None,
            "403",
            "Agent is disabled or no longer bound to an identity",
            id="row_09_unbound",
        ),
        pytest.param(
            _bind_other_issuer,
            CLIENT_CREDENTIALS,
            None,
            "403",
            "Token does not match the registered Okta application",
            id="row_10",
        ),
        pytest.param(
            _bind(mode="delegated"),
            CLIENT_CREDENTIALS,
            None,
            "403",
            "Agent is not enabled for this execution mode",
            id="row_11",
        ),
        pytest.param(
            _bind(scopes=("invoices.write",)),
            CLIENT_CREDENTIALS,
            None,
            "403",
            "Token lacks the required scopes",
            id="row_15",
        ),
        pytest.param(
            _bind(), {**CLIENT_CREDENTIALS, "scp": "invoices.read"}, None, "403", "Invalid scope claim", id="row_22"
        ),
        pytest.param(
            _bind(),
            CLIENT_CREDENTIALS,
            "/key/generate",
            "403",
            "Agent identities can only access inference and agent discovery routes",
            id="row_23",
        ),
        pytest.param(
            _recording_fails,
            CLIENT_CREDENTIALS,
            None,
            "503",
            "Agent authentication could not be recorded",
            id="row_26_write_fails",
        ),
        pytest.param(
            _revision_moved,
            CLIENT_CREDENTIALS,
            None,
            "403",
            "Agent identity changed during authentication; retry",
            id="row_26_revision_moved",
        ),
    ],
)
async def test_client_credentials_refusals(
    gateway: Gateway,
    setup: Callable[[Gateway], None],
    claims: Mapping[str, object],
    route: str | None,
    code: str,
    message: str,
) -> None:
    setup(gateway)

    outcome: Final = await gateway.outcome(claims, route or "/v1/chat/completions")

    assert outcome[:2] == ("refused", code)
    assert message in str(outcome[2])


def refuse_every_token(token: dict[str, object]) -> Literal[True]:
    raise ValueError("custom validation refused this token")


@pytest.mark.asyncio
async def test_row_27_custom_validation_refuses_an_agent_token_exactly_as_a_plain_token(gateway: Gateway) -> None:
    gateway.bind(okta_agent())
    gateway.configure(custom_validate=refuse_every_token)
    plain: Final = await _plain_today(gateway, CLIENT_CREDENTIALS)

    outcome: Final = await gateway.outcome(CLIENT_CREDENTIALS)

    assert outcome[0] == "refused"
    assert outcome == plain


@pytest.mark.asyncio
async def test_row_28_virtual_key_mapping_refuses_a_token_that_resolves_to_an_agent(gateway: Gateway) -> None:
    gateway.bind(okta_agent())
    gateway.configure(virtual_key_claim_field="sub")

    outcome: Final = await gateway.outcome(CLIENT_CREDENTIALS)

    assert outcome[:2] == ("refused", "403")
    assert "Managed agents require direct JWT authentication without virtual-key mapping" in str(outcome[2])


@pytest.mark.asyncio
async def test_row_29_scope_mappings_do_not_apply_to_an_autonomous_agent_as_for_entra(gateway: Gateway) -> None:
    gateway.bind(okta_agent())
    gateway.configure(
        enforce_scope_based_access=True,
        scope_mappings=[ScopeMapping(scope="other.scope", models=["another-model"])],
    )

    assert await gateway.outcome(CLIENT_CREDENTIALS) == ("admitted", "invoice-bot", "autonomous", None)


@pytest.mark.asyncio
async def test_row_09_agent_deleted_while_its_binding_is_still_cached_on_this_pod(gateway: Gateway) -> None:
    gateway.bind(okta_agent())
    assert (await gateway.outcome(CLIENT_CREDENTIALS))[0] == "admitted"
    gateway.client.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

    outcome: Final = await gateway.outcome(CLIENT_CREDENTIALS)

    assert outcome[:2] == ("refused", "403")
    assert "Agent is disabled or no longer bound to an identity" in str(outcome[2])


MARIA: Final = "maria@example.com"
SIGN_IN: Final = {"sub": MARIA, "uid": "00u1maria", "cid": APP, "email": MARIA, "scp": ["invoices.read"]}


def _person(gateway: Gateway, *, granted: bool = True, active: bool = True) -> None:
    gateway.configure(user_email_jwt_field="email")
    permission: Final = LiteLLM_ObjectPermissionTable(
        object_permission_id="maria-grants", agents=["invoice-bot"] if granted else []
    )
    gateway.stores.object_permission.seed("maria-grants", permission.model_dump())
    gateway.stores.user.rows[MARIA] = {
        "user_id": MARIA,
        "user_email": MARIA,
        "user_role": "internal_user",
        "teams": [],
        "metadata": {} if active else {"scim_active": False},
        "object_permission_id": "maria-grants",
        "object_permission": permission,
    }


@pytest.mark.asyncio
async def test_row_20_person_signed_in_to_a_delegated_agents_app_runs_as_that_agent_for_that_person(
    gateway: Gateway,
) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway)

    assert await gateway.outcome(SIGN_IN) == ("admitted", "invoice-bot", "delegated", MARIA)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup",
    [
        pytest.param(lambda gateway: None, id="row_08_no_binding"),
        pytest.param(_bind(mode="autonomous"), id="row_08_autonomous_only"),
        pytest.param(_bind(mode="autonomous", enabled=False), id="row_08_autonomous_only_disabled"),
        pytest.param(_retired, id="row_08_retired"),
    ],
)
async def test_row_08_person_token_without_a_delegated_binding_keeps_todays_plain_path(
    gateway: Gateway, setup: Callable[[Gateway], None]
) -> None:
    setup(gateway)
    _person(gateway)
    today: Final = await _plain_today(gateway, SIGN_IN)

    assert await gateway.outcome(SIGN_IN) == today
    assert today[0] == "admitted" and today[1] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["delegated", "both"])
async def test_row_20_person_token_runs_as_the_agent_in_delegated_or_both_mode(
    gateway: Gateway, mode: AgentExecutionMode
) -> None:
    gateway.bind(okta_agent(mode=mode))
    _person(gateway)

    assert await gateway.outcome(SIGN_IN) == ("admitted", "invoice-bot", "delegated", MARIA)


def _delegated(
    *, enabled: bool = True, active: bool = True, scopes: tuple[str, ...] = (), granted: bool = True
) -> Callable[[Gateway], None]:
    def setup(gateway: Gateway) -> None:
        gateway.bind(okta_agent(mode="delegated", enabled=enabled, active=active, scopes=scopes))
        _person(gateway, granted=granted)

    return setup


def _delegated_other_issuer(gateway: Gateway) -> None:
    _bind_other_issuer(gateway)
    agent: Final = okta_agent(mode="delegated")
    assert agent.identity is not None
    moved: Final = agent.model_copy(
        update={"identity": agent.identity.model_copy(update={"issuer": "https://acme.okta.example/oauth2/other"})}
    )
    gateway.bind(moved)
    _person(gateway)


def _delegated_without_user(gateway: Gateway) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    gateway.configure(user_email_jwt_field="email")


def _delegated_without_user_fields(gateway: Gateway) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway)
    gateway.configure(user_email_jwt_field=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "claims", "route", "code", "message"),
    [
        pytest.param(_delegated(enabled=False), SIGN_IN, None, "403", "Agent is disabled or no longer bound to an identity", id="row_09_person_disabled"),
        pytest.param(_delegated(active=False), SIGN_IN, None, "403", "Agent is disabled or no longer bound to an identity", id="row_09_person_unbound"),
        pytest.param(_delegated_other_issuer, SIGN_IN, None, "403", "Token does not match the registered Okta application", id="row_10_person"),
        pytest.param(_delegated(scopes=("invoices.write",)), SIGN_IN, None, "403", "Token lacks the required scopes", id="row_15_person"),
        pytest.param(_delegated_without_user, SIGN_IN, None, "403", "The delegated person has no LiteLLM user", id="row_16_no_user"),
        pytest.param(_delegated_without_user_fields, SIGN_IN, None, "403", "The delegated person has no LiteLLM user", id="row_16_no_user_claim"),
        pytest.param(_delegated(granted=False), SIGN_IN, None, "403", "Select a team that grants access to this agent using x-litellm-team-id", id="row_19"),
        pytest.param(_delegated(), {**SIGN_IN, "scp": "invoices.read"}, None, "403", "Invalid scope claim", id="row_22_person"),
        pytest.param(_delegated(), SIGN_IN, "/key/generate", "403", "Agent identities can only access inference and agent discovery routes", id="row_23_person"),
    ],
)
async def test_person_token_refusals(
    gateway: Gateway,
    setup: Callable[[Gateway], None],
    claims: Mapping[str, object],
    route: str | None,
    code: str,
    message: str,
) -> None:
    setup(gateway)

    outcome: Final = await gateway.outcome(claims, route or "/v1/chat/completions")

    assert outcome[:2] == ("refused", code)
    assert message in str(outcome[2])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configure", "person"),
    [
        pytest.param({"user_allowed_email_domain": "corp.example"}, {}, id="row_17_email_domain"),
        pytest.param({}, {"active": False}, id="row_24_deactivated"),
    ],
)
async def test_person_refusals_carry_mains_exact_plain_path_refusal(
    gateway: Gateway, configure: Mapping[str, object], person: Mapping[str, bool]
) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway, active=person.get("active", True))
    gateway.configure(**configure)
    plain: Final = await _plain_today(gateway, SIGN_IN)

    outcome: Final = await gateway.outcome(SIGN_IN)

    assert outcome[0] == "refused"
    assert outcome == plain


@pytest.mark.asyncio
async def test_row_18_enforce_rbac_refuses_a_delegated_agent_as_it_does_for_entra(gateway: Gateway) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway)
    gateway.configure(enforce_rbac=True)

    outcome: Final = await gateway.outcome(SIGN_IN)

    assert outcome[:2] == ("refused", "403")
    assert "Unmatched token passed in. enforce_rbac is set to True" in str(outcome[2])


def _with_user_fields(
    gateway: Gateway, issuer_fields: Mapping[str, str], global_email: str | None, okta: bool
) -> None:
    gateway.configure(
        user_email_jwt_field=global_email,
        issuers=[
            issuer.model_copy(
                update={**issuer_fields, **({} if okta else {"agent_identity_provider": None, "okta_org": None})}
            )
            for issuer in _issuers()
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("issuer_fields", "global_email", "user_id"),
    [
        pytest.param({"user_id_jwt_field": "uid"}, None, "00u1maria", id="issuer_user_id_field"),
        pytest.param({}, "email", MARIA, id="global_email_fallback"),
        pytest.param({"user_id_jwt_field": "uid"}, "email", "00u1maria", id="both"),
        pytest.param({}, None, None, id="neither"),
    ],
)
async def test_the_delegated_person_is_the_user_the_plain_path_names_for_the_same_claims(
    gateway: Gateway, issuer_fields: Mapping[str, str], global_email: str | None, user_id: str | None
) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway)
    if user_id is not None:
        gateway.stores.user.rows[user_id] = {**gateway.stores.user.rows[MARIA], "user_id": user_id}
    _with_user_fields(gateway, issuer_fields, global_email, okta=False)
    plain: Final = await gateway.outcome(SIGN_IN)
    _with_user_fields(gateway, issuer_fields, global_email, okta=True)

    delegated: Final = await gateway.outcome(SIGN_IN)

    assert plain == ("admitted", None, None, user_id)
    if user_id is None:
        assert delegated[0] == "refused"
        assert "The delegated person has no LiteLLM user" in str(delegated[2])
    else:
        assert delegated == ("admitted", "invoice-bot", "delegated", user_id)


@pytest.mark.asyncio
async def test_row_29_scope_mappings_refuse_a_delegated_agent_with_mains_plain_path_refusal(gateway: Gateway) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway)
    gateway.configure(
        enforce_scope_based_access=True,
        scope_mappings=[ScopeMapping(scope="other.scope", models=["another-model"])],
    )
    plain: Final = await _plain_today(gateway, SIGN_IN)

    outcome: Final = await gateway.outcome(SIGN_IN)

    assert outcome[0] == "refused"
    assert outcome == plain


@pytest.mark.asyncio
async def test_row_25_a_grant_from_a_team_the_token_did_not_name_is_refused_on_a_route_teams_cannot_call(
    gateway: Gateway,
) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway, granted=False)
    team_grant: Final = LiteLLM_ObjectPermissionTable(object_permission_id="finance-grants", agents=["invoice-bot"])
    gateway.stores.object_permission.seed("finance-grants", team_grant.model_dump())
    gateway.stores.user.rows[MARIA] = {**gateway.stores.user.rows[MARIA], "teams": ["finance"]}
    gateway.stores.team.rows["finance"] = {
        "team_id": "finance",
        "members_with_roles": [{"user_id": MARIA, "role": "user"}],
        "object_permission_id": "finance-grants",
        "object_permission": team_grant,
    }
    gateway.stores.team_membership.rows[(MARIA, "finance")] = {"user_id": MARIA, "team_id": "finance"}

    outcome: Final = await gateway.outcome(SIGN_IN, "/v1/agents", "GET")

    assert outcome[:2] == ("refused", "403")
    assert "The granting team is not allowed to access this route" in str(outcome[2])


@pytest.mark.asyncio
async def test_row_19_a_team_header_naming_a_team_that_does_not_grant_the_agent_is_refused(gateway: Gateway) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway, granted=False)
    team_grant: Final = LiteLLM_ObjectPermissionTable(object_permission_id="finance-grants", agents=["invoice-bot"])
    gateway.stores.object_permission.seed("finance-grants", team_grant.model_dump())
    gateway.stores.user.rows[MARIA] = {**gateway.stores.user.rows[MARIA], "teams": ["finance", "sales"]}
    for team in ("finance", "sales"):
        gateway.stores.team.rows[team] = {
            "team_id": team,
            "members_with_roles": [{"user_id": MARIA, "role": "user"}],
            **({"object_permission_id": "finance-grants", "object_permission": team_grant} if team == "finance" else {}),
        }
        gateway.stores.team_membership.rows[(MARIA, team)] = {"user_id": MARIA, "team_id": team}

    gateway.configure(team_ids_jwt_field="groups")

    outcome: Final = await gateway.outcome(
        {**SIGN_IN, "groups": ["finance", "sales"]}, headers={"x-litellm-team-id": "sales"}
    )

    assert outcome[:2] == ("refused", "403")
    assert "Select a team that grants access to this agent using x-litellm-team-id" in str(outcome[2])
