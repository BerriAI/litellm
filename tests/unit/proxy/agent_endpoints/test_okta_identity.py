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
from litellm.proxy._types import JWTIssuerConfig, LiteLLM_JWTAuth, ProxyException, ScopeMapping, UserAPIKeyAuth
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


def _request(route: str = "/v1/chat/completions") -> Request:
    body: Final = {"model": "gpt-test", "messages": []}
    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": route,
            "headers": [(b"content-type", b"application/json")],
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

    async def send(self, claims: Mapping[str, object], route: str = "/v1/chat/completions") -> UserAPIKeyAuth:
        return await user_api_key_auth_builder(
            request=_request(route),
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

    async def outcome(self, claims: Mapping[str, object], route: str = "/v1/chat/completions") -> tuple[object, ...]:
        try:
            auth: Final = await self.send(claims, route)
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
