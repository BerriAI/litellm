import threading
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.process import graceful_stop_seconds
from tests.integration._support.wire import wire_server
from tests.integration.authorization._jwt_team_upsert import (
    UpsertRig,
    delete_team_if_present,
    marker_of,
    provider_reply,
    team_models,
    upserting_jwt_rig,
)

pytestmark: Final = pytest.mark.shares_worker_state

BURST_TEAMS: Final = 10
KILL_AFTER_RESPONSES: Final = 5
ROUTES: Final = (
    ("/v1/chat/completions", False),
    ("/v1/messages", True),
    ("/v1/responses", False),
    ("/v1/chat/completions", True),
    ("/v1/messages", False),
    ("/v1/responses", True),
)


@dataclass(frozen=True, slots=True)
class _Attempt:
    team: str
    marker: str
    status: int | None
    text: str


def _body(route: str, stream: bool, model: str, marker: str) -> dict[str, JsonValue]:
    if route == "/v1/responses":
        return {"model": model, "input": marker, "stream": stream}
    return {
        "model": model,
        "messages": [{"role": "user", "content": marker}],
        "stream": stream,
        **({"max_tokens": 16} if route == "/v1/messages" else {}),
    }


def _call(candidate: Gateway, token: str, team: str, route: str, stream: bool, model: str) -> _Attempt:
    marker: Final = f"marker-{uuid.uuid4().hex}"
    try:
        response: Final = candidate.request("POST", route, _body(route, stream, model, marker), key=token)
    except httpx.TransportError as error:
        return _Attempt(team, marker, None, repr(error))
    return _Attempt(team, marker, response.status_code, response.text)


def _served_request_ids(team: str) -> list[str]:
    rows: Final = read_rows(
        'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE team_id = %s AND status = %s', (team, "success")
    )
    return [str(row["request_id"]) for row in rows]


def _burst(rig: UpsertRig, tokens: Sequence[tuple[str, str]], model: str, victim: int) -> tuple[_Attempt, ...]:
    answered: Final = threading.Semaphore(0)

    def team_calls(index: int, team: str, token: str) -> tuple[_Attempt, ...]:
        def one(offset: int) -> _Attempt:
            route, stream = ROUTES[(index + offset) % len(ROUTES)]
            attempt: Final = _call(rig.gateway, token, team, route, stream, model)
            answered.release()
            return attempt

        return tuple(one(offset) for offset in range(3))

    with ThreadPoolExecutor(max_workers=len(tokens)) as pool, rig.owned.replacing_worker(victim) as kill:
        futures: Final = tuple(pool.submit(team_calls, index, *pair) for index, pair in enumerate(tokens))
        for _ in range(KILL_AFTER_RESPONSES):
            assert answered.acquire(timeout=120), "The burst stalled before the worker was killed"
        kill()
        return tuple(chain.from_iterable(future.result() for future in futures))


def test_a_burst_of_first_requests_survives_a_worker_kill_with_every_team_on_the_default_models(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(provider_reply) as provider, gateway.scenario() as scenario:
        approved: Final = scenario.model(api_base=f"{provider.url}/v1")
        unapproved: Final = scenario.model(api_base=f"{provider.url}/v1")
        subject: Final = scenario.user(user_role="internal_user")
        teams: Final = tuple(f"integration-dept-{uuid.uuid4().hex}" for _ in range(BURST_TEAMS))
        with upserting_jwt_rig(
            gateway, tmp_path, {"team_id_jwt_field": "dept"}, {"models": [approved]}, workers=2
        ) as rig:
            for team in teams:
                scenario.cleanups.callback(delete_team_if_present, scenario, team)
            tokens: Final = tuple((team, rig.sign({"sub": subject, "dept": team})) for team in teams)
            workers: Final = eventually(
                rig.owned.worker_pids, lambda found: len(found) == 2, seconds=graceful_stop_seconds() + 60
            )

            attempts: Final = _burst(rig, tokens, approved, min(workers))
            recovered: Final = tuple(
                _call(rig.gateway, token, team, "/v1/chat/completions", False, approved) for team, token in tokens
            )
            refused: Final = tuple(
                _call(rig.gateway, token, team, "/v1/chat/completions", False, unapproved) for team, token in tokens
            )

            answered: Final = tuple(attempt for attempt in attempts if attempt.status is not None)
            assert len(answered) >= KILL_AFTER_RESPONSES, attempts
            assert [(attempt.status, attempt.text) for attempt in answered if attempt.status != 200] == []
            assert [attempt.status for attempt in recovered] == [200] * BURST_TEAMS, recovered
            assert [attempt.status for attempt in refused] == [403] * BURST_TEAMS, refused
            assert [team_models(team) for team in teams] == [[[approved]]] * BURST_TEAMS
            reached_provider: Final = frozenset(marker_of(request) for request in provider.drain())
            served: Final = tuple(attempt for attempt in (*answered, *recovered))
            assert frozenset(attempt.marker for attempt in served) <= reached_provider
            assert not frozenset(attempt.marker for attempt in refused) & reached_provider
            for team in teams:
                after_recovery: Final = frozenset(
                    f"chatcmpl-{attempt.marker}" for attempt in recovered if attempt.team == team
                )
                logged: Final = eventually(
                    lambda team=team: _served_request_ids(team),
                    lambda rows, after_recovery=after_recovery: after_recovery <= frozenset(rows),
                    seconds=70,
                )
                matches: Final = [
                    [attempt.marker for attempt in served if attempt.team == team and request_id in attempt.text]
                    for request_id in logged
                ]
                assert ([len(match) for match in matches], len(frozenset(logged))) == (
                    [1] * len(logged),
                    len(logged),
                ), (
                    team,
                    logged,
                )
