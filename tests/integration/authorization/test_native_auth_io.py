import os
import uuid
from hashlib import sha256
from typing import Final

import httpx
import psycopg
import pytest
from pydantic import JsonValue, TypeAdapter
from redis import Redis

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import paused_prisma_engines

_OBSERVATIONS: Final = TypeAdapter(dict[str, list[dict[str, JsonValue]]])


def _worker_pid(gateway: Gateway) -> int:
    response: Final = gateway.request("GET", "/health/integration-readiness")
    assert response.status_code == 200, response.text
    return int(response.headers["x-integration-worker-pid"])


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"native auth storage {uuid.uuid4().hex}"}]},
        key=key,
    )


def _observations(gateway: Gateway) -> list[dict[str, JsonValue]]:
    response: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=5, trust_env=False)
    response.raise_for_status()
    return _OBSERVATIONS.validate_json(response.content)["requests"]


def test_cold_and_warm_context_reads_do_not_need_prisma(gateway: Gateway, peer: Gateway) -> None:
    primary_pid: Final = _worker_pid(gateway)
    peer_pid: Final = _worker_pid(peer)
    assert primary_pid != peer_pid
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        owner: Final = scenario.member(team)
        project: Final = scenario.project(team, models=[model])
        key: Final = scenario.key(team_id=team, user_id=owner, project_id=project, models=[model])
        with paused_prisma_engines(peer_pid):
            cold: Final = _chat(peer, model, key)
            assert cold.status_code == 200, cold.text
            warm: Final = _chat(peer, model, key)
            assert warm.status_code == 200, warm.text
        assert len(_observations(gateway)) == 2


def test_warm_key_read_does_not_wait_for_the_key_table(gateway: Gateway, peer: Gateway) -> None:
    database_url: Final = gateway.database_url or os.environ["DATABASE_URL"]
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert _chat(peer, model, key).status_code == 200
        with psycopg.connect(database_url) as connection:
            connection.execute('LOCK TABLE "LiteLLM_VerificationToken" IN ACCESS EXCLUSIVE MODE')
            warm: Final = _chat(peer, model, key)
            assert warm.status_code == 200, warm.text
        assert len(_observations(gateway)) == 2


def test_revocation_reaches_a_warmed_peer_without_prisma(gateway: Gateway, peer: Gateway) -> None:
    peer_pid: Final = _worker_pid(peer)
    assert _worker_pid(gateway) != peer_pid
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        for worker in (gateway, peer):
            response: Final = _chat(worker, model, key)
            assert response.status_code == 200, response.text
        with paused_prisma_engines(peer_pid):
            gateway.post("/key/block", {"key": key})
            denied: Final = eventually(
                lambda: _chat(peer, model, key), lambda response: response.status_code == 401, seconds=10
            )
            before: Final = _observations(gateway)
            assert len(before) >= 2
            assert _chat(peer, model, key).status_code == 401
            assert _observations(gateway) == before
            assert denied.json()["error"]["type"] == "auth_error", denied.text
        rows: Final = gateway.rows(
            'SELECT blocked FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
        )
        assert rows == [{"blocked": True}]


@pytest.mark.parametrize("context", ("team", "project", "user"))
def test_context_changes_reach_a_warmed_peer(gateway: Gateway, peer: Gateway, context: str) -> None:
    peer_pid: Final = _worker_pid(peer)
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = scenario.team(models=[model])
        owner: Final = scenario.member(team)
        project: Final = scenario.project(team, models=[model])
        key: Final = scenario.key(team_id=team, user_id=owner, project_id=project, models=[model])
        warm: Final = _chat(peer, model, key)
        assert warm.status_code == 200, warm.text
        with paused_prisma_engines(peer_pid):
            if context == "team":
                gateway.post("/team/update", {"team_id": team, "blocked": True})
            elif context == "project":
                gateway.post("/project/update", {"project_id": project, "blocked": True})
            else:
                gateway.post("/user/update", {"user_id": owner, "metadata": {"scim_active": False}})
            denied: Final = eventually(
                lambda: _chat(peer, model, key), lambda response: response.status_code in (401, 422), seconds=10
            )
            before: Final = _observations(gateway)
            assert before
            repeated: Final = _chat(peer, model, key)
            assert repeated.status_code == denied.status_code, repeated.text
            assert _observations(gateway) == before


@pytest.mark.parametrize("source", ("redis", "database"))
def test_spend_reads_share_native_counters_and_reseed_without_prisma(
    gateway: Gateway, peer: Gateway, source: str
) -> None:
    database_url: Final = gateway.database_url or os.environ["DATABASE_URL"]
    peer_pid: Final = _worker_pid(peer)
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        control: Final = scenario.key(models=[model])
        allowed: Final = _chat(peer, model, control)
        assert allowed.status_code == 200, allowed.text
        key: Final = scenario.key(models=[model], max_budget=1)
        hashed: Final = sha256(key.encode()).hexdigest()
        counter: Final = f"spend:key:{hashed}"
        with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), socket_timeout=2) as cache:
            if source == "redis":
                cache.set(counter, 2, ex=60)
            else:
                with psycopg.connect(database_url) as connection:
                    connection.execute('UPDATE "LiteLLM_VerificationToken" SET spend=2 WHERE token=%s', (hashed,))
                cache.delete(counter)
            with paused_prisma_engines(peer_pid):
                denied: Final = _chat(peer, model, key)
                assert denied.status_code == 422, denied.text
            value: Final = cache.get(counter)
            assert value is not None and float(value) == 2
        assert len(_observations(gateway)) == 1
