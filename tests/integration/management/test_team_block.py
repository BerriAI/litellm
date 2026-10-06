import uuid
from collections.abc import Iterator
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, JsonValue

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows


class _ObservedRequest(BaseModel):
    body: dict[str, JsonValue]


class _Observations(BaseModel):
    requests: list[_ObservedRequest]


class _TeamState(BaseModel):
    team_id: str
    blocked: bool


class _TeamInfo(BaseModel):
    team_info: _TeamState


class _ProxyError(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _ProxyError


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=15, trust_env=False) as client:
        response: Final = client.get("/__observations")
        assert response.status_code == 200, response.text
        yield client


def _chat(proxy: Gateway, model: str, key: str, text: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": text}]},
        key=key,
    )


def _models(client: httpx.Client) -> tuple[str, ...]:
    response: Final = client.get("/__observations")
    assert response.status_code == 200, response.text
    observations: Final = _Observations.model_validate_json(response.text)
    return tuple(string_value(request.body["model"]) for request in observations.requests)


def test_team_block_and_unblock_update_endpoint_info_and_db_state(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        provider_model: Final = f"team-block-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team)

        primary_before: Final = _chat(gateway, model, key, "team block primary warm")
        assert primary_before.status_code == 200, primary_before.text
        peer_before: Final = _chat(peer, model, key, "team block peer warm")
        assert peer_before.status_code == 200, peer_before.text
        observed_before_block: Final = _models(upstream)
        assert observed_before_block == (provider_model, provider_model), repr(observed_before_block)

        blocked_response: Final = gateway.request("POST", "/team/block", {"team_id": team})
        assert blocked_response.status_code == 200, blocked_response.text
        blocked: Final = _TeamState.model_validate_json(blocked_response.text)
        assert blocked.blocked is True, blocked_response.text

        info_response: Final = gateway.request("GET", "/team/info", params={"team_id": team})
        assert info_response.status_code == 200, info_response.text
        info: Final = _TeamInfo.model_validate_json(info_response.text)
        assert info.team_info.blocked is True, info_response.text
        assert info.team_info.team_id == team, info_response.text
        assert read_rows(
            'SELECT blocked FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [{"blocked": True}], info_response.text

        cold_key: Final = scenario.key(team_id=team)
        cold_denied: Final = _chat(gateway, model, cold_key, "cold team key after block")
        assert cold_denied.status_code == 401, cold_denied.text
        assert _ErrorResponse.model_validate_json(cold_denied.text).error.type == "auth_error", cold_denied.text
        after_cold_denied: Final = _models(upstream)
        assert after_cold_denied == (), repr(after_cold_denied)

        unblocked_response: Final = gateway.request("POST", "/team/unblock", {"team_id": team})
        assert unblocked_response.status_code == 200, unblocked_response.text
        restored_key: Final = scenario.key(team_id=team)
        restored: Final = _chat(gateway, model, restored_key, "new team key after unblock")
        assert restored.status_code == 200, restored.text
        after_unblock: Final = _models(upstream)
        assert after_unblock == (provider_model,), repr(after_unblock)
        unblocked: Final = _TeamState.model_validate_json(unblocked_response.text)
        assert unblocked.blocked is False, unblocked_response.text
        unblocked_info_response: Final = gateway.request("GET", "/team/info", params={"team_id": team})
        assert unblocked_info_response.status_code == 200, unblocked_info_response.text
        unblocked_info: Final = _TeamInfo.model_validate_json(unblocked_info_response.text)
        assert unblocked_info.team_info.blocked is False, unblocked_info_response.text
        assert read_rows(
            'SELECT blocked FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [{"blocked": False}], unblocked_info_response.text

        assert observed_before_block + after_cold_denied + after_unblock == (
            provider_model,
            provider_model,
            provider_model,
        ), repr((observed_before_block, after_cold_denied, after_unblock))


def test_unblocking_a_team_restores_warmed_keys_on_both_proxies(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client
) -> None:
    pytest.skip("BUG: /team/unblock does not restore warmed team-key access after blocked auth is cached")

    with gateway.scenario() as scenario:
        provider_model: Final = f"team-block-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team)

        primary_before: Final = _chat(gateway, model, key, "team unblock primary warm")
        assert primary_before.status_code == 200, primary_before.text
        peer_before: Final = _chat(peer, model, key, "team unblock peer warm")
        assert peer_before.status_code == 200, peer_before.text
        observed_before_block: Final = _models(upstream)
        assert observed_before_block == (provider_model, provider_model), repr(observed_before_block)

        blocked_response: Final = gateway.request("POST", "/team/block", {"team_id": team})
        assert blocked_response.status_code == 200, blocked_response.text
        cold_key: Final = scenario.key(team_id=team)
        cold_denied: Final = _chat(gateway, model, cold_key, "cold team key after block")
        assert cold_denied.status_code == 401, cold_denied.text
        assert _ErrorResponse.model_validate_json(cold_denied.text).error.type == "auth_error", cold_denied.text
        after_cold_denied: Final = _models(upstream)
        assert after_cold_denied == (), repr(after_cold_denied)

        unblocked_response: Final = gateway.request("POST", "/team/unblock", {"team_id": team})
        assert unblocked_response.status_code == 200, unblocked_response.text

        primary_after: Final = _chat(gateway, model, key, "team unblock primary")
        assert primary_after.status_code == 200, primary_after.text
        peer_after: Final = _chat(peer, model, key, "team unblock peer")
        assert peer_after.status_code == 200, peer_after.text
        after_unblock: Final = _models(upstream)
        assert after_unblock == (provider_model, provider_model), repr(after_unblock)


def test_warmed_team_keys_are_refused_on_both_proxies_after_block(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client
) -> None:
    pytest.skip("BUG: blocking a team does not refresh warmed team-key auth state")

    with gateway.scenario() as scenario:
        provider_model: Final = f"team-block-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team)

        primary_before: Final = _chat(gateway, model, key, "team block primary warm")
        assert primary_before.status_code == 200, primary_before.text
        peer_before: Final = _chat(peer, model, key, "team block peer warm")
        assert peer_before.status_code == 200, peer_before.text
        observed_before_block: Final = _models(upstream)
        assert observed_before_block == (provider_model, provider_model), repr(observed_before_block)

        blocked_response: Final = gateway.request("POST", "/team/block", {"team_id": team})
        assert blocked_response.status_code == 200, blocked_response.text

        primary_denied: Final = _chat(gateway, model, key, "blocked team primary")
        peer_denied: Final = _chat(peer, model, key, "blocked team peer")
        after_block: Final = _models(upstream)
        assert (primary_denied.status_code, peer_denied.status_code) == (401, 401), (
            f"primary={primary_denied.text}; peer={peer_denied.text}"
        )
        assert _ErrorResponse.model_validate_json(primary_denied.text).error.type == "auth_error", primary_denied.text
        assert _ErrorResponse.model_validate_json(peer_denied.text).error.type == "auth_error", peer_denied.text
        assert after_block == (), repr(after_block)
        assert observed_before_block + after_block == (provider_model, provider_model), repr(
            (observed_before_block, after_block)
        )


def test_a_new_team_key_is_refused_on_a_peer_that_cached_the_team_before_block(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client
) -> None:
    pytest.skip("BUG: a peer proxy that cached the team before /team/block admits a never-seen team key")

    with gateway.scenario() as scenario:
        provider_model: Final = f"team-block-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        team: Final = scenario.team(models=[model])
        warm_key: Final = scenario.key(team_id=team)

        peer_before: Final = _chat(peer, model, warm_key, "team block peer warm")
        assert peer_before.status_code == 200, peer_before.text
        observed_before_block: Final = _models(upstream)
        assert observed_before_block == (provider_model,), repr(observed_before_block)

        blocked_response: Final = gateway.request("POST", "/team/block", {"team_id": team})
        assert blocked_response.status_code == 200, blocked_response.text
        assert read_rows(
            'SELECT blocked FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [{"blocked": True}], blocked_response.text

        cold_key: Final = scenario.key(team_id=team)
        peer_denied: Final = _chat(peer, model, cold_key, "new team key on peer after block")
        after_block: Final = _models(upstream)
        assert peer_denied.status_code == 401, peer_denied.text
        assert _ErrorResponse.model_validate_json(peer_denied.text).error.type == "auth_error", peer_denied.text
        assert after_block == (), repr(after_block)
