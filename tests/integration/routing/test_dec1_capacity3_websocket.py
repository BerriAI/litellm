"""Model capacity on Responses API WebSocket sessions (``/v1/responses`` in websocket mode).

Session start. Opening the socket runs the proxy's pre-call logic once with ``{"model": <m>, "websocket": ...}``
(``_responses_websocket_session``, litellm/proxy/response_api_endpoints/endpoints.py:2626), so the dynamic limiter's
``async_pre_call_hook`` (dynamic_rate_limiter_v3.py:1638) takes one request of the model-wide window for the whole
session. A refusal there is sent as one error frame (``turn_refusal_frame``: ``type: error``, ``status: 429``,
``error.type: rate_limit_exceeded``) and the socket is closed with 1008 ``Rate limit exceeded`` (endpoints.py:2700).

Per turn. Each ``response.create`` frame is admitted by ``enforce_token_limits_for_frame`` (:1024), which only checks
the caller's class share slice and borrower shares on their token windows once the model saturates, and borrow caps
always (:1062-1068). It never checks the model-wide window, so turns are not held to the model's ``rpm``. A refused
turn gets the same error frame and the socket stays open. When the share windows cannot be read and
``fail_closed_rate_limit_enforcement`` is on, the turn is refused with a 503 ``server_error`` frame (:1079-1082,
streaming_iterator.py:2241). Each turn is logged on its own, so settlement (:1785) charges each turn's usage
separately (the scripted peer reports 7 tokens per turn).

The provider is the scripted TLS peer in ``_support/responses_ws_peer.py``; the proxy trusts its certificate through
``SSL_CERT_FILE``, as ``test_responses_websocket_turn_admission.py`` does.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import pytest
import websockets
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, object_value
from tests.integration._support.process import graceful_stop_seconds
from tests.integration._support.redis_process import OwnedRedis, owned_redis
from tests.integration._support.responses_ws_peer import Peer, create, drain, responses_peer, turn
from tests.integration._support.tls import write_self_signed_cert
from tests.integration.authorization.dec1_limcap4_support import counter, counter_settles
from tests.integration.routing.dec1_capacity3_support import (
    CLASS_SHARE_SCOPE,
    WS_TURN_TOKENS,
    SocketOutcome,
    refused_at_open,
    socket_turns,
    turn_types,
    ws_entry,
)
from tests.integration.routing.dec1_capacity_support import MODEL_SCOPE, capacity_proxy, group_name

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

UNVERIFIABLE: Final = "Rate limit enforcement unavailable"


@pytest.fixture(scope="module")
def cert(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    return write_self_signed_cert(tmp_path_factory.mktemp("capacity3-ws-cert"))


@contextmanager
def _ws_rig(
    gateway: Gateway,
    directory: Path,
    cert: tuple[Path, Path],
    *,
    limits: Mapping[str, JsonValue],
    priority_settings: Mapping[str, JsonValue] | None = None,
    reservation: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    redis: OwnedRedis | None = None,
) -> Iterator[tuple[Gateway, Peer, str, Scenario]]:
    group: Final = group_name("c3ws")
    environment: Final = {
        "SSL_CERT_FILE": str(cert[0]),
        **({"REDIS_HOST": redis.host, "REDIS_PORT": str(redis.port)} if redis is not None else {}),
    }
    with (
        responses_peer(cert) as peer,
        capacity_proxy(
            gateway,
            directory,
            model_list=[ws_entry(group, peer.url, **limits)],
            priority_reservation=reservation,
            priority_settings=priority_settings,
            general_settings=general_settings,
            environment=environment,
            shared=redis is None,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        yield candidate, peer, group, scenario


def _forwarded(peer: Peer) -> int:
    return sum(frame.get("type") == "response.create" for frame in drain(peer.frames))


def _refusal(outcome: SocketOutcome) -> dict[str, JsonValue]:
    refused: Final = next(events[-1] for events in outcome.turns if events[-1].get("type") == "error")
    return refused


def test_opening_a_responses_socket_takes_one_request_of_model_capacity(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """A model with ``rpm: 10``. One socket carries one completed turn, and ``{model_saturation_check:<m>}:requests``
    holds 1: the session's pre-call admission counted it (endpoints.py:2683, dynamic_rate_limiter_v3.py:1292)."""
    with _ws_rig(gateway, tmp_path, cert, limits={"rpm": 10}) as (candidate, _peer, group, _scenario):
        outcome: Final = socket_turns(candidate, candidate.key, group, 1)
        assert turn_types(outcome) == ["response.completed"], outcome.turns
        assert counter_settles(MODEL_SCOPE, group, "requests", 1) == 1


def test_three_turns_on_one_socket_take_one_request_of_model_capacity(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """Three completed turns on one socket leave ``{model_saturation_check:<m>}:requests`` at 1. Turn admission
    (:1024) never writes the model-wide window, so capacity counts sessions, not turns. A port that admitted each
    frame through the pre-call path would count 3."""
    with _ws_rig(gateway, tmp_path, cert, limits={"rpm": 10}) as (candidate, _peer, group, _scenario):
        outcome: Final = socket_turns(candidate, candidate.key, group, 3)
        assert turn_types(outcome) == ["response.completed"] * 3, outcome.turns
        assert counter_settles(MODEL_SCOPE, group, "tokens", 3 * WS_TURN_TOKENS) == 3 * WS_TURN_TOKENS
        assert counter(MODEL_SCOPE, group, "requests") == 1


def test_each_turn_settles_its_own_usage_on_the_model_token_window(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """A model with ``tpm: 10000``. After two turns ``{model_saturation_check:<m>}:tokens`` settles to 14, the 7 tokens
    of each turn, because every turn is logged on its own and settlement reads that turn's usage (``_usage_of``,
    :176). A port that settled once at socket close from the last turn would leave 7."""
    with _ws_rig(gateway, tmp_path, cert, limits={"tpm": 10_000}) as (candidate, _peer, group, _scenario):
        outcome: Final = socket_turns(candidate, candidate.key, group, 2)
        assert turn_types(outcome) == ["response.completed"] * 2, outcome.turns
        assert counter_settles(MODEL_SCOPE, group, "tokens", 2 * WS_TURN_TOKENS) == 2 * WS_TURN_TOKENS


def test_a_socket_opened_on_a_full_model_is_refused_with_the_capacity_text_and_closed(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """A model with ``rpm: 1``. The first socket takes the window. A second socket's first frame gets one error frame
    with ``status: 429``, ``error.type: rate_limit_exceeded`` and a message containing ``Model capacity reached for
    <m>``, then the socket is closed with 1008 ``Rate limit exceeded``; the peer sees no turn for it."""
    with _ws_rig(gateway, tmp_path, cert, limits={"rpm": 1}) as (candidate, peer, group, _scenario):
        assert turn_types(socket_turns(candidate, candidate.key, group, 1)) == ["response.completed"]
        assert _forwarded(peer) == 1
        frame, code, reason = refused_at_open(candidate, candidate.key, group)
        assert (frame["type"], frame["status"]) == ("error", 429), frame
        error: Final = object_value(frame["error"])
        assert error["type"] == "rate_limit_exceeded", frame
        assert f"Model capacity reached for {group}" in str(error["message"]), frame
        assert (code, reason) == (1008, "Rate limit exceeded")
        assert _forwarded(peer) == 0


def test_turns_past_the_models_rpm_on_an_open_socket_are_still_served(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """A model with ``rpm: 1``. One socket sends three turns and all three complete and reach the peer, although the
    model's request window was full after the session opened: the model-wide window is checked only at session start
    (:1024-1068 build no ``model_saturation_check`` descriptor)."""
    with _ws_rig(gateway, tmp_path, cert, limits={"rpm": 1}) as (candidate, peer, group, _scenario):
        outcome: Final = socket_turns(candidate, candidate.key, group, 3)
        assert turn_types(outcome) == ["response.completed"] * 3, outcome.turns
        assert _forwarded(peer) == 3


def test_a_borrowing_teams_turn_past_its_share_is_refused_on_the_open_socket(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """Teams borrow on a saturated proxy (``saturation_threshold: 0.0``), the model has ``tpm: 10000`` and the team
    ``model_tpm_limit: {<m>: 10}``. Turns settle 7 tokens each on ``{model_per_team_saturated:<team>:<m>}:tokens``, so
    within four turns one is refused with an error frame ``status: 429``, ``error.type: rate_limit_exceeded`` and a
    message containing ``Team model_tpm_limit reached for <m>`` (``_team_limit_error``, :953), while the socket stays
    open: the turn after it gets an error frame too, not a closed socket. UNVERIFIED which turn is the first refused
    (2 or 3): it depends on whether the read-only check refuses at or above the limit."""
    with _ws_rig(
        gateway,
        tmp_path,
        cert,
        limits={"tpm": 10_000},
        priority_settings={
            "saturation_threshold": 0.0,
            "saturation_check_cache_ttl": 0,
            "model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]},
        },
    ) as (candidate, peer, group, scenario):
        team: Final = scenario.team(metadata={"model_tpm_limit": {group: 10}})
        outcome: Final = socket_turns(candidate, scenario.key(team_id=team), group, 4)
        types: Final = turn_types(outcome)
        assert types[0] == "response.completed", outcome.turns
        assert types[-1] == "error" and outcome.close_code is None, (types, outcome.close_code)
        refusal: Final = _refusal(outcome)
        assert refusal["status"] == 429, refusal
        assert f"Team model_tpm_limit reached for {group}" in str(object_value(refusal["error"])["message"]), refusal
        assert _forwarded(peer) == types.count("response.completed")


def test_a_teams_class_share_turn_refusal_names_the_class_share(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """Teams split ``{"interactive": 1.0}`` on a saturated ``tpm: 40`` model. A first team completes one turn (7
    tokens) so two teams are active and each slice is ``int(40 x 0.5) = 20``. A second team's socket then runs turns of
    7 tokens until one is refused with a 429 frame whose message contains ``Priority class share reached for <m>``
    (``_class_share_error`` through ``_throttled``, :1087-1092). UNVERIFIED which turn (the fourth if the check refuses
    at or above 20)."""
    with _ws_rig(
        gateway,
        tmp_path,
        cert,
        limits={"tpm": 40},
        reservation={"interactive": 1.0},
        priority_settings={
            "saturation_threshold": 0.0,
            "saturation_check_cache_ttl": 0,
            "priority_class_share_config": {"split_equally_among_active": ["team"]},
        },
    ) as (candidate, _peer, group, scenario):
        first: Final = scenario.key(team_id=scenario.team(metadata={"priority": "interactive"}))
        second: Final = scenario.key(team_id=scenario.team(metadata={"priority": "interactive"}))
        assert turn_types(socket_turns(candidate, first, group, 1)) == ["response.completed"]
        outcome: Final = socket_turns(candidate, second, group, 5)
        assert "error" in turn_types(outcome), outcome.turns
        message: Final = str(object_value(_refusal(outcome)["error"])["message"])
        assert f"Priority class share reached for {group}" in message, message


def test_a_teams_class_share_slice_counts_the_socket_session_once(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """#51952 on sockets. Teams split on an idle (``saturation_threshold: 0.9``) ``rpm: 100`` model. A team's socket
    carries two turns. ``{priority_class_share_saturated:<len>:<m>:<team_id>}:requests`` holds 1: the session's
    pre-call admission tracked the slice once and turn admission adds no requests (:1071-1078 is read-only)."""
    with _ws_rig(
        gateway,
        tmp_path,
        cert,
        limits={"rpm": 100},
        reservation={"interactive": 1.0},
        priority_settings={
            "saturation_threshold": 0.9,
            "priority_class_share_config": {"split_equally_among_active": ["team"]},
        },
    ) as (candidate, _peer, group, scenario):
        team: Final = scenario.team(metadata={"priority": "interactive"})
        outcome: Final = socket_turns(candidate, scenario.key(team_id=team), group, 2)
        assert turn_types(outcome) == ["response.completed"] * 2, outcome.turns
        value: Final = f"{len(group)}:{group}:{team}"
        assert counter_settles(CLASS_SHARE_SCOPE, value, "requests", 1) == 1


def test_a_borrowing_teams_turn_is_refused_with_a_503_frame_when_fail_closed_cannot_read_redis(
    gateway: Gateway, tmp_path: Path, cert: tuple[Path, Path]
) -> None:
    """#51705 made websocket turns fail closed. A proxy on a Redis this test owns, with
    ``fail_closed_rate_limit_enforcement: true``, teams borrowing on a saturated model and a team limit on it. The
    first turn completes, Redis is stopped, and the next turn on the same socket gets an error frame with ``status:
    503``, ``error.type: server_error`` and a message containing ``Rate limit enforcement unavailable`` (the
    ``ServiceUnavailableError`` of :1080 framed by ``_turn_rejection_frame``, streaming_iterator.py:2241); the peer sees
    no second turn. UNVERIFIED: the regular limiter's own turn check may refuse first with the same text."""
    with (
        owned_redis(tmp_path / "redis") as redis,
        _ws_rig(
            gateway,
            tmp_path,
            cert,
            limits={"tpm": 100_000},
            priority_settings={
                "saturation_threshold": 0.0,
                "model_limit_borrowing_config": {"borrow_idle_capacity": ["team"]},
            },
            general_settings={"fail_closed_rate_limit_enforcement": True},
            redis=redis,
        ) as (candidate, peer, group, scenario),
    ):
        team: Final = scenario.team(metadata={"model_tpm_limit": {group: 10_000}})
        key: Final = scenario.key(team_id=team)
        outcome: Final = _turn_then_outage(candidate, key, group, redis)
        assert turn_types(outcome) == ["response.completed", "error"], outcome.turns
        refusal: Final = outcome.turns[1][-1]
        assert refusal["status"] == 503, refusal
        error: Final = object_value(refusal["error"])
        assert error["type"] == "server_error" and UNVERIFIABLE in str(error["message"]), refusal
        assert _forwarded(peer) == 1
        redis.start()


def _turn_then_outage(gateway: Gateway, key: str, model: str, redis: OwnedRedis) -> SocketOutcome:
    """One socket: a completed turn, Redis stopped, then a second turn."""

    async def run() -> SocketOutcome:
        url: Final = (
            f"{str(gateway.client.base_url).rstrip('/').replace('http://', 'ws://')}/v1/responses?model={model}"
        )
        async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"}) as connection:
            first: Final = await turn(connection, create(model, "capacity3 before outage"))
            await asyncio.to_thread(redis.stop)
            second: Final = await turn(connection, create(model, "capacity3 after outage"))
            return SocketOutcome((first, second), None, None)

    return asyncio.run(run())
