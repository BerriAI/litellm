import base64
import uuid
from collections.abc import Mapping
from urllib.parse import parse_qs

import httpx
import pytest
import respx
from fastapi import HTTPException

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.a2a_endpoints import _resolve_backend_auth
from litellm.proxy.agent_endpoints.token_exchange import exchange_agent_token
from litellm.proxy.mcp_server.outbound_credentials.token_exchanger import (
    OboTokenExchanger,
    SubjectTokenRejected,
    TokenExchangeClientError,
)
from litellm.types.agents import AgentResponse

_SECRET = "obo-client-secret"
_SUBJECT = "eyJ.caller.jwt"
_CLAIMS = '{"acrs":"c1"}'
_BLOCK = {
    "token_exchange_endpoint": "https://idp.example.com/token",
    "client_id": "agent-client",
    "client_secret": _SECRET,
    "audience": "api://agent",
}


class _IdP:
    def __init__(self, outcome: dict[str, object] | None | Exception = None) -> None:
        self.outcome = outcome
        self.forms: list[dict[str, str]] = []

    async def __call__(self, url: str, form: dict[str, str], headers: dict[str, str]) -> dict[str, object] | None:
        self.forms.append(form)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if self.outcome is not None:
            return self.outcome
        target = form.get("audience") or form.get("scope", "")
        return {"access_token": f"exchanged:{form.get('subject_token') or form.get('assertion')}:{target}"}


def _jwt_caller(team_id: str = "team-1") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="hashed-mapped-key", team_id=team_id, jwt_claims={"sub": "user-1"})


async def _exchange(
    idp: _IdP,
    block: Mapping[str, object] = _BLOCK,
    *,
    agent_id: str = "agent-1",
    headers: Mapping[str, str] | None = None,
    caller: UserAPIKeyAuth | None = None,
    exchanger: OboTokenExchanger | None = None,
) -> Mapping[str, str]:
    return await exchange_agent_token(
        {"token_exchange": dict(block), "model": "agent"},
        agent_id,
        "https://agent.example.com",
        {"authorization": f"Bearer {_SUBJECT}"} if headers is None else headers,
        caller or _jwt_caller(),
        exchanger or OboTokenExchanger(idp),
    )


async def test_the_callers_token_is_exchanged_for_the_agent_bearer_with_env_expanded_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_OBO_SECRET", _SECRET)
    idp = _IdP()

    header = await _exchange(idp, {**_BLOCK, "client_secret": "os.environ/AGENT_OBO_SECRET", "header_name": "X-Other"})

    assert header == {"Authorization": f"Bearer exchanged:{_SUBJECT}:api://agent"}
    assert idp.forms == [
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": _SUBJECT,
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
            "audience": "api://agent",
            "client_id": "agent-client",
            "client_secret": _SECRET,
        }
    ]


async def test_the_entra_obo_profile_sends_the_callers_token_as_the_assertion() -> None:
    idp = _IdP()

    header = await _exchange(idp, {**_BLOCK, "profile": "entra_obo", "scopes": ["api://agent/.default"]})

    assert header == {"Authorization": f"Bearer exchanged:{_SUBJECT}:api://agent/.default"}
    assert idp.forms[0]["assertion"] == _SUBJECT
    assert idp.forms[0]["requested_token_use"] == "on_behalf_of"


async def test_exchanges_are_cached_per_agent_and_per_tenant() -> None:
    idp = _IdP()
    exchanger = OboTokenExchanger(idp)

    await _exchange(idp, exchanger=exchanger)
    await _exchange(idp, exchanger=exchanger)
    await _exchange(idp, agent_id="agent-2", exchanger=exchanger)
    await _exchange(idp, caller=_jwt_caller("team-2"), exchanger=exchanger)

    assert len(idp.forms) == 3


@pytest.mark.parametrize(
    "headers",
    [{}, {"authorization": "Bearer sk-virtual-key"}, {"authorization": "Bearer opaque", "x-litellm-api-key": "opaque"}],
)
async def test_a_caller_without_an_identity_token_gets_a_401_before_the_idp(headers: Mapping[str, str]) -> None:
    idp = _IdP()

    with pytest.raises(HTTPException) as refused:
        await _exchange(idp, headers=headers, caller=UserAPIKeyAuth(api_key="hashed-key"))

    assert refused.value.status_code == 401
    assert refused.value.headers == {"WWW-Authenticate": 'Bearer error="invalid_request"'}
    assert idp.forms == []


@pytest.mark.parametrize(
    ("outcome", "status", "challenge"),
    [
        (
            SubjectTokenRejected("rejected", claims=_CLAIMS),
            401,
            f'Bearer error="insufficient_claims", claims="{base64.b64encode(_CLAIMS.encode()).decode()}"',
        ),
        (SubjectTokenRejected("rejected"), 401, 'Bearer error="invalid_token"'),
        (TokenExchangeClientError("invalid_client"), 500, None),
        ({"token_type": "Bearer"}, 503, None),
    ],
)
async def test_an_idp_refusal_maps_to_the_public_status_without_leaking_secrets(
    outcome: dict[str, object] | Exception, status: int, challenge: str | None
) -> None:
    with pytest.raises(HTTPException) as refused:
        await _exchange(_IdP(outcome))

    assert refused.value.status_code == status
    assert (refused.value.headers or {}).get("WWW-Authenticate") == challenge
    assert _SECRET not in str(refused.value.detail) and _SUBJECT not in str(refused.value.detail)


@pytest.mark.parametrize("block", ["not-a-mapping", {**_BLOCK, "profile": "unknown"}])
async def test_an_invalid_block_is_a_500_that_never_echoes_the_secret(block: object) -> None:
    idp = _IdP()

    with pytest.raises(HTTPException) as refused:
        await exchange_agent_token(
            {"token_exchange": block}, "agent-1", "https://agent.example.com", {}, _jwt_caller(), OboTokenExchanger(idp)
        )

    assert refused.value.status_code == 500
    assert _SECRET not in str(refused.value.detail)
    assert idp.forms == []


@respx.mock
async def test_backend_auth_sends_the_exchanged_bearer_and_strips_the_block_from_the_forwarded_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    subject = f"eyJ.{uuid.uuid4().hex}"
    idp = respx.post("https://idp.example.com/token").mock(
        return_value=httpx.Response(200, json={"access_token": "agent-bound", "token_type": "Bearer"})
    )
    agent = AgentResponse(
        agent_id=f"agent-{uuid.uuid4().hex}",
        agent_name="obo-agent",
        agent_card_params={"url": "https://agent.example.com"},
        litellm_params={"token_exchange": _BLOCK, "api_key": "static-key"},
    )

    auth = await _resolve_backend_auth(
        {"token_exchange": _BLOCK, "api_key": "static-key"},
        None,
        agent=agent,
        request_headers={"authorization": f"Bearer {subject}"},
        user_api_key_auth=_jwt_caller(),
    )

    assert auth.header == {"Authorization": "Bearer agent-bound"}
    assert auth.litellm_params == {"api_key": "static-key"}
    assert parse_qs(idp.calls.last.request.content.decode())["subject_token"] == [subject]


async def test_an_unset_env_reference_is_a_500_naming_the_field_and_never_reaches_the_idp() -> None:
    idp = _IdP()

    with pytest.raises(HTTPException) as refused:
        await _exchange(idp, {**_BLOCK, "audience": "os.environ/A2A_OBO_AUDIENCE_NOT_SET"})

    assert refused.value.status_code == 500
    assert "audience" in str(refused.value.detail) and _SECRET not in str(refused.value.detail)
    assert idp.forms == []


@pytest.mark.parametrize("disabled", [{}, False, None])
async def test_a_falsy_token_exchange_block_keeps_the_agents_static_auth(disabled: object) -> None:
    agent = AgentResponse(
        agent_id="agent-plain",
        agent_name="plain-agent",
        agent_card_params={"url": "https://agent.example.com"},
        litellm_params={"token_exchange": disabled, "api_key": "static-key"},
    )

    auth = await _resolve_backend_auth(
        {"token_exchange": disabled, "api_key": "static-key"},
        None,
        agent=agent,
        request_headers={},
        user_api_key_auth=UserAPIKeyAuth(api_key="hashed-key"),
    )

    assert auth.header == {"Authorization": "Bearer static-key"}
