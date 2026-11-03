import itertools
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, JsonValue

from tests.integration._support.client import Gateway, eventually, held, string_value
from tests.integration._support.database import read_rows

IN_FLIGHT_CALLERS_PER_PROXY: Final = 8
BLOCK_CYCLES: Final = 4
PROBES_PER_PROXY: Final = 6


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


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True)
    content: str


class _ChatBody(BaseModel):
    model_config = ConfigDict(frozen=True)
    messages: tuple[_Message, ...]


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=15, trust_env=False) as client:
        client.delete("/__observations").raise_for_status()
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


def _texts(client: httpx.Client) -> tuple[str, ...]:
    response: Final = client.get("/__observations")
    assert response.status_code == 200, response.text
    observations: Final = _Observations.model_validate_json(response.text)
    return tuple(
        message.content
        for message in itertools.chain.from_iterable(
            _ChatBody.model_validate(request.body).messages for request in observations.requests
        )
    )


def _keep_calling(proxy: Gateway, model: str, key: str, stop: threading.Event) -> tuple[int, ...]:
    return tuple(
        _chat(proxy, model, key, f"in flight {uuid.uuid4().hex}").status_code
        for _ in itertools.takewhile(lambda _: not stop.is_set(), itertools.count())
    )


def _statuses_on_fresh_connections(proxies: tuple[Gateway, ...], model: str, key: str, marker: str) -> tuple[int, ...]:
    return tuple(
        proxy.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"{marker} {uuid.uuid4().hex}"}]},
            key=key,
            headers={"connection": "close"},
        ).status_code
        for proxy in proxies * PROBES_PER_PROXY
    )


def _block_while_team_traffic_is_in_flight(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client, model: str, key: str, team: str
) -> tuple[httpx.Response, tuple[int, ...]]:
    stop: Final = threading.Event()
    with ThreadPoolExecutor(max_workers=2 * IN_FLIGHT_CALLERS_PER_PROXY) as pool:
        callers: Final = tuple(
            pool.submit(_keep_calling, proxy, model, key, stop)
            for proxy in (gateway, peer) * IN_FLIGHT_CALLERS_PER_PROXY
        )
        try:
            eventually(
                lambda: len(_models(upstream)), lambda served: served >= 4 * IN_FLIGHT_CALLERS_PER_PROXY, seconds=30
            )
            blocked: Final = gateway.request("POST", "/team/block", {"team_id": team})
        finally:
            stop.set()
        statuses: Final = tuple(itertools.chain.from_iterable(caller.result() for caller in callers))
    return blocked, statuses


def _block_and_unblock_under_team_traffic(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client, model: str, key: str, team: str, cycle: int
) -> None:
    upstream.delete("/__observations").raise_for_status()
    blocked, in_flight = _block_while_team_traffic_is_in_flight(gateway, peer, upstream, model, key, team)
    assert blocked.status_code == 200, blocked.text
    assert in_flight.count(200) >= 4 * IN_FLIGHT_CALLERS_PER_PROXY, repr(in_flight)
    assert set(in_flight) <= {200, 401}, repr(in_flight)

    marker: Final = f"after block {cycle} {uuid.uuid4().hex}"
    refused: Final = held(
        lambda: _statuses_on_fresh_connections((gateway, peer), model, key, marker),
        lambda statuses: set(statuses) == {401},
        holding=5,
        seconds=15,
    )
    assert refused == (401,) * (2 * PROBES_PER_PROXY), repr(refused)
    leaked: Final = tuple(text for text in _texts(upstream) if text.startswith(marker))
    assert leaked == (), repr(leaked)

    unblocked: Final = gateway.request("POST", "/team/unblock", {"team_id": team})
    assert unblocked.status_code == 200, unblocked.text
    eventually(
        lambda: _statuses_on_fresh_connections((gateway, peer), model, key, f"after unblock {cycle}"),
        lambda statuses: set(statuses) == {200},
        seconds=15,
    )


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
        upstream.delete("/__observations").raise_for_status()

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
        upstream.delete("/__observations").raise_for_status()

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
        upstream.delete("/__observations").raise_for_status()

        blocked_response: Final = gateway.request("POST", "/team/block", {"team_id": team})
        assert blocked_response.status_code == 200, blocked_response.text
        cold_key: Final = scenario.key(team_id=team)
        cold_denied: Final = _chat(gateway, model, cold_key, "cold team key after block")
        assert cold_denied.status_code == 401, cold_denied.text
        assert _ErrorResponse.model_validate_json(cold_denied.text).error.type == "auth_error", cold_denied.text
        after_cold_denied: Final = _models(upstream)
        assert after_cold_denied == (), repr(after_cold_denied)
        upstream.delete("/__observations").raise_for_status()

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
        upstream.delete("/__observations").raise_for_status()

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
    with gateway.scenario() as scenario:
        provider_model: Final = f"team-block-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        team: Final = scenario.team(models=[model])
        warm_key: Final = scenario.key(team_id=team)

        peer_before: Final = _chat(peer, model, warm_key, "team block peer warm")
        assert peer_before.status_code == 200, peer_before.text
        observed_before_block: Final = _models(upstream)
        assert observed_before_block == (provider_model,), repr(observed_before_block)
        upstream.delete("/__observations").raise_for_status()

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


def test_a_team_model_alias_still_routes_on_both_proxies_after_block_and_unblock(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        provider_model: Final = f"team-block-{uuid.uuid4().hex}"
        model: Final = scenario.model(model=f"openai/{provider_model}")
        alias: Final = f"alias-{uuid.uuid4().hex}"
        team: Final = scenario.team(models=[model], model_aliases={alias: model})
        key: Final = scenario.key(team_id=team, models=[model])

        warmed: Final = tuple(
            _chat(proxy, alias, key, f"team alias warm {uuid.uuid4().hex}").status_code for proxy in (gateway, peer)
        )
        assert warmed == (200, 200), repr(warmed)
        assert _models(upstream) == (provider_model, provider_model)
        upstream.delete("/__observations").raise_for_status()

        blocked: Final = gateway.request("POST", "/team/block", {"team_id": team})
        assert blocked.status_code == 200, blocked.text
        unblocked: Final = gateway.request("POST", "/team/unblock", {"team_id": team})
        assert unblocked.status_code == 200, unblocked.text

        after_unblock: Final = _statuses_on_fresh_connections((gateway, peer), alias, key, "team alias after unblock")
        assert after_unblock == (200,) * (2 * PROBES_PER_PROXY), repr(after_unblock)
        assert _models(upstream) == (provider_model,) * (2 * PROBES_PER_PROXY)


def test_a_block_issued_while_team_traffic_is_in_flight_holds_on_every_worker(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        provider_model: Final = f"team-block-{uuid.uuid4().hex}"
        model: Final = scenario.model(
            model=f"openai/{provider_model}", input_cost_per_token=0.000001, output_cost_per_token=0.000002
        )
        team: Final = scenario.team(models=[model])
        key: Final = scenario.key(team_id=team)

        for cycle in range(BLOCK_CYCLES):
            _block_and_unblock_under_team_traffic(gateway, peer, upstream, model, key, team, cycle)
