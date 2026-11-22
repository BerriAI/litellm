import logging
from collections.abc import Callable, Mapping
from typing import Final

import jwt
import pytest

from litellm.proxy._types import IdJagRequestingApp, LiteLLM_UserTable, ProxyException
from litellm.proxy.auth.auth_checks import LITELLM_SESSION_TOKEN_PREFIX, ExperimentalUIJWTToken
from litellm.proxy.auth.jwt import ID_JAG_TYP
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_bearer_token
from litellm.proxy.mcp_server.gateway_dcr_flow import AssertionIdentity, SubjectIdentity, SubjectTokenRefusal
from litellm.proxy.mcp_server.id_jag_grant import (
    AssertionRedemption,
    IdJagPrerequisites,
    _proxy_agent_stamp,
    identity_from_id_jag,
)
from litellm.types.proxy.agent_identity import AgentStamp
from tests.unit.proxy.agent_endpoints.test_okta_identity import (
    APP,
    MARIA,
    ORG_SERVER,
    Gateway,
    _person,
    _request,
    gateway,
    okta_agent,
)

NOW: Final = 1_800_000_000.0
AUDIENCE: Final = "https://gateway.example.com"
PREREQUISITES: Final = IdJagPrerequisites(
    jwt_auth_enabled=True,
    has_database=True,
    licensed=True,
    maps_jwts_to_virtual_keys=False,
    shared_replay_store=True,
    base_url=AUDIENCE,
    requesting_apps=(IdJagRequestingApp(client_id=APP),),
)
ASSERTION: Final = {
    "iss": ORG_SERVER,
    "sub": "00u1maria",
    "aud": AUDIENCE,
    "exp": NOW + 300,
    "iat": NOW - 5,
    "jti": "jti-1",
    "client_id": APP,
    "scope": "litellm:mcp",
}
__all__ = ("gateway",)


async def _verified(claims: Mapping[str, object]) -> Callable[..., object]:
    async def decode(assertion: str, audience: object, required: object) -> Mapping[str, object]:
        return claims

    return decode


async def _maria(claims: Mapping[str, object], team: str | None) -> SubjectIdentity:
    return SubjectIdentity(user_id=MARIA, team_id=team)


async def _redeem(claims: Mapping[str, object] = ASSERTION) -> AssertionIdentity | SubjectTokenRefusal:
    return await identity_from_id_jag(
        AssertionRedemption(
            assertion=jwt.encode(
                dict(claims), "unused-by-the-fake-verifier-but-long-enough-for-hs256", headers={"typ": ID_JAG_TYP}
            ),
            client_id=APP,
            client_secret=None,
            authorization_header=None,
            team_id=None,
            audiences=(AUDIENCE,),
            now=NOW,
            leeway_seconds=0,
        ),
        prerequisites=PREREQUISITES,
        decode=await _verified(claims),
        resolve_subject=_maria,
        stamp_agent=_proxy_agent_stamp,
    )


def _delegated(gateway: Gateway, **binding: object) -> None:
    gateway.bind(okta_agent(mode="delegated", **binding))
    _person(gateway)


@pytest.mark.asyncio
async def test_j7_an_assertion_from_a_delegated_app_is_stamped_with_its_agent(gateway: Gateway) -> None:
    _delegated(gateway)

    identity: Final = await _redeem()

    assert isinstance(identity, AssertionIdentity)
    assert (identity.user_id, identity.agent) == (MARIA, AgentStamp(agent_id="invoice-bot", binding_revision="r1"))


def _unbound(gateway: Gateway) -> None:
    _person(gateway)


def _autonomous_only(gateway: Gateway) -> None:
    gateway.bind(okta_agent())
    _person(gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "claims", "looks_up"),
    [
        pytest.param(_delegated, {**ASSERTION, "iss": "https://other.example"}, False, id="j1_issuer_not_opted_in"),
        pytest.param(_unbound, ASSERTION, True, id="j3_no_binding"),
        pytest.param(_autonomous_only, ASSERTION, True, id="j3_autonomous_only"),
    ],
)
async def test_assertions_without_a_delegated_binding_redeem_exactly_as_today(
    gateway: Gateway, setup: Callable[[Gateway], None], claims: Mapping[str, object], looks_up: bool
) -> None:
    setup(gateway)

    identity: Final = await _redeem(claims)

    assert isinstance(identity, AssertionIdentity)
    assert identity.agent is None
    assert identity.model_dump(exclude={"agent"}) == (
        await identity_from_id_jag(
            AssertionRedemption(
                assertion=jwt.encode(
                    dict(claims), "unused-by-the-fake-verifier-but-long-enough-for-hs256", headers={"typ": ID_JAG_TYP}
                ),
                client_id=APP,
                client_secret=None,
                authorization_header=None,
                team_id=None,
                audiences=(AUDIENCE,),
                now=NOW,
                leeway_seconds=0,
            ),
            prerequisites=PREREQUISITES,
            decode=await _verified(claims),
            resolve_subject=_maria,
        )
    ).model_dump(exclude={"agent"})
    assert (gateway.lookups() > 0) is looks_up


def _no_grant(gateway: Gateway) -> None:
    gateway.bind(okta_agent(mode="delegated"))
    _person(gateway, granted=False)


def _revision_moves(gateway: Gateway) -> None:
    _delegated(gateway)
    gateway.stores.agent_identity.current.clear()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "logged"),
    [
        pytest.param(
            lambda gateway: _delegated(gateway, enabled=False), "agent invoice-bot is disabled", id="j4_disabled"
        ),
        pytest.param(
            lambda gateway: _delegated(gateway, active=False), "agent invoice-bot is disabled", id="j4_unbound"
        ),
        pytest.param(_no_grant, f"user {MARIA} has no grant for agent invoice-bot", id="j5_no_grant"),
        pytest.param(_revision_moves, "Agent identity changed during authentication", id="j6_revision_moved"),
    ],
)
async def test_agent_refusals_at_redemption_are_invalid_grant_with_the_reason_logged(
    gateway: Gateway, setup: Callable[[Gateway], None], logged: str, caplog: pytest.LogCaptureFixture
) -> None:
    setup(gateway)

    with caplog.at_level(logging.WARNING):
        refusal: Final = await _redeem()

    assert isinstance(refusal, SubjectTokenRefusal)
    assert refusal.error == "invalid_grant"
    assert logged in caplog.text


@pytest.mark.asyncio
async def test_j2_main_refusals_run_first_and_never_look_up_a_binding(gateway: Gateway) -> None:
    _delegated(gateway)

    refusal: Final = await _redeem({**ASSERTION, "scope": "openid"})

    assert isinstance(refusal, SubjectTokenRefusal)
    assert gateway.lookups() == 0


@pytest.mark.asyncio
async def test_a_binding_read_that_fails_at_redemption_is_retryable(gateway: Gateway) -> None:
    _delegated(gateway)
    gateway.client.writer_db.litellm_agentidentity.find_unique.side_effect = RuntimeError("db down")

    refusal: Final = await _redeem()

    assert isinstance(refusal, SubjectTokenRefusal)
    assert refusal.error == "temporarily_unavailable"


def _session(stamp: AgentStamp | None = AgentStamp(agent_id="invoice-bot", binding_revision="r1")) -> str:
    return ExperimentalUIJWTToken.get_cli_jwt_auth_token(
        LiteLLM_UserTable(user_id=MARIA, user_email=MARIA, user_role="internal_user"), agent=stamp
    )


async def _send(session: str, route: str = "/v1/chat/completions") -> tuple[object, ...]:
    try:
        auth: Final = await user_api_key_auth(
            request=_request(route),
            api_key=f"Bearer {session}",
            azure_api_key_header="",
            anthropic_api_key_header=None,
            google_ai_studio_api_key_header=None,
            azure_apim_header=None,
            custom_litellm_key_header=None,
        )
    except ProxyException as refused:
        return ("refused", refused.code, refused.message)
    context: Final = auth.managed_agent_context
    return ("admitted", auth.agent_id, context.mode if context else None, auth.user_id)


@pytest.mark.asyncio
async def test_a_stamped_session_runs_as_the_delegated_agent_for_its_person(gateway: Gateway) -> None:
    agent: Final = okta_agent(mode="delegated")
    gateway.bind(agent.model_copy(update={"object_permission": {"models": ["gpt-test"]}}))
    _person(gateway)

    assert await _send(_session()) == ("admitted", "invoice-bot", "delegated", MARIA)


@pytest.mark.asyncio
async def test_an_unstamped_session_takes_todays_path(gateway: Gateway) -> None:
    _delegated(gateway)

    assert await _send(_session(None)) == ("admitted", None, None, MARIA)
    assert gateway.lookups() == 0


def _deleted(gateway: Gateway) -> None:
    _person(gateway)


def _moved(gateway: Gateway) -> None:
    agent: Final = okta_agent(mode="delegated")
    assert agent.identity is not None
    gateway.bind(agent.model_copy(update={"identity": agent.identity.model_copy(update={"revision": "r2"})}))
    _person(gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("setup", "route", "message"),
    [
        pytest.param(_deleted, "/v1/chat/completions", "Agent no longer exists", id="j8_deleted"),
        pytest.param(
            lambda gateway: _delegated(gateway, enabled=False),
            "/v1/chat/completions",
            "Agent execution is disabled",
            id="j8_disabled",
        ),
        pytest.param(
            lambda gateway: _delegated(gateway, active=False),
            "/v1/chat/completions",
            "Agent execution is disabled",
            id="j8_unbound",
        ),
        pytest.param(_moved, "/v1/chat/completions", "Agent binding changed; redeem a new assertion", id="j8_rebound"),
        pytest.param(
            _autonomous_only,
            "/v1/chat/completions",
            "Agent is not enabled for this execution mode",
            id="j8_now_autonomous",
        ),
        pytest.param(
            _no_grant,
            "/v1/chat/completions",
            "The delegated user is not permitted to invoke this agent",
            id="j8_grant_revoked",
        ),
        pytest.param(
            _delegated,
            "/key/generate",
            "Agent identities can only access inference and agent discovery routes",
            id="j9",
        ),
    ],
)
async def test_a_stamped_session_rechecks_its_agent_on_every_request(
    gateway: Gateway, setup: Callable[[Gateway], None], route: str, message: str
) -> None:
    setup(gateway)

    outcome: Final = await _send(_session(), route)

    assert outcome[:2] == ("refused", "403")
    assert message in str(outcome[2])


@pytest.mark.asyncio
async def test_a_session_naming_a_managed_agent_without_a_revision_is_refused_never_served_as_the_person(
    gateway: Gateway,
) -> None:
    _delegated(gateway)
    stamped: Final = ExperimentalUIJWTToken.get_key_object_from_ui_hash_key(_session())
    assert stamped is not None
    older_reader: Final = encrypt_bearer_token(
        stamped.model_copy(update={"agent_binding_revision": None}).model_dump_json(exclude_none=True),
        prefix=LITELLM_SESSION_TOKEN_PREFIX,
    )

    outcome: Final = await _send(older_reader)

    assert outcome[:2] == ("refused", "403")
    assert "bound identity provider token" in str(outcome[2])


@pytest.mark.asyncio
async def test_a_grant_lookup_that_cannot_finish_at_redemption_is_retryable(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.agent_endpoints.auth import agent_permission_handler

    _delegated(gateway)

    async def unreachable(user_id: str | None, team_id: str | None = None) -> frozenset[str]:
        raise RuntimeError("db down")

    monkeypatch.setattr(agent_permission_handler, "verified_human_agent_grants", unreachable)

    refusal: Final = await _redeem()

    assert isinstance(refusal, SubjectTokenRefusal)
    assert refusal.error == "temporarily_unavailable"


@pytest.mark.asyncio
async def test_a_redemption_with_no_database_is_never_stamped(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", None)

    assert await _proxy_agent_stamp(ASSERTION, APP, SubjectIdentity(user_id=MARIA)) is None


@pytest.mark.asyncio
async def test_a_stamped_session_whose_agent_cannot_be_read_is_retryable(gateway: Gateway) -> None:
    _delegated(gateway)
    gateway.client.writer_db.litellm_agentstable.find_unique.side_effect = RuntimeError("db down")

    outcome: Final = await _send(_session())

    assert outcome[:2] == ("refused", "503")

@pytest.mark.asyncio
async def test_redemption_checks_the_current_app_binding(gateway: Gateway) -> None:
    _delegated(gateway, client="replacement-client")

    result = await _redeem()

    assert isinstance(result, SubjectTokenRefusal)
    assert result.error == "invalid_grant"
