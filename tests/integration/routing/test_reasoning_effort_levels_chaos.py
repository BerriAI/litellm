from __future__ import annotations

import dataclasses
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.routing._reasoning_effort_levels_support import (
    CLAUDE_LEVELS,
    KIMI_LEVELS,
    answer,
    chat_request,
    deployment,
    marker_of,
    messages_request,
    new_marker,
    owned_config,
    provider_reply,
    responses_request,
    sent_effort,
    yaml_deployment,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(3 * graceful_stop_seconds() + 180)

_KIMI_ROUNDED: Final = MappingProxyType({"minimal": "low", "medium": "high", "xhigh": "max"})
_CLAUDE_ROUNDED: Final = MappingProxyType({"minimal": "low", "medium": "high", "xhigh": "high"})
_KINDS: Final = ("chat", "chat-stream", "responses", "responses-stream", "messages", "messages-stream")
_REQUESTED: Final = ("minimal", "medium", "xhigh")
_BURST: Final = 30


@dataclass(frozen=True, slots=True)
class Models:
    kimi: str
    claude: str


@dataclass(frozen=True, slots=True)
class Shot:
    marker: str
    kind: str
    requested: str
    rounded: str


@dataclass(frozen=True, slots=True)
class Outcome:
    shot: Shot
    status: int | None
    text: str


def _shot(index: int) -> Shot:
    kind: Final = _KINDS[index % len(_KINDS)]
    requested: Final = _REQUESTED[index // len(_KINDS) % len(_REQUESTED)]
    rounded: Final = _CLAUDE_ROUNDED if kind.startswith("messages") else _KIMI_ROUNDED
    return Shot(new_marker(), kind, requested, rounded[requested])


def _route_and_body(shot: Shot, models: Models) -> tuple[str, dict[str, JsonValue]]:
    stream: Final = shot.kind.endswith("-stream")
    level: Final = shot.requested
    match shot.kind.removesuffix("-stream"):
        case "chat":
            return "/v1/chat/completions", chat_request(models.kimi, shot.marker, reasoning_effort=level, stream=stream)
        case "responses":
            return "/v1/responses", responses_request(
                models.kimi, shot.marker, reasoning={"effort": level}, stream=stream
            )
        case "messages":
            return "/v1/messages", messages_request(
                models.claude,
                shot.marker,
                thinking={"type": "adaptive"},
                output_config={"effort": level},
                stream=stream,
            )
        case other:
            raise AssertionError(other)


def _fire(gateway: Gateway, models: Models, shot: Shot) -> Outcome:
    route, body = _route_and_body(shot, models)
    try:
        response: Final = gateway.request("POST", route, body)
    except httpx.HTTPError as error:
        return Outcome(shot, None, repr(error))
    return Outcome(shot, response.status_code, response.text)


def _burst(
    gateway: Gateway, models: Models, shots: tuple[Shot, ...], during: Callable[[], None]
) -> tuple[Outcome, ...]:
    with ThreadPoolExecutor(max_workers=len(shots)) as pool:
        futures: Final[tuple[Future[Outcome], ...]] = tuple(pool.submit(_fire, gateway, models, shot) for shot in shots)
        during()
        return tuple(future.result() for future in futures)


def _assert_every_upstream_effort_rounded(shots: tuple[Shot, ...], received: tuple[Request, ...]) -> None:
    by_marker: Final = MappingProxyType({shot.marker: shot for shot in shots})
    observed: Final = tuple((marker_of(request), sent_effort(request)) for request in received)
    assert all(marker in by_marker for marker, _ in observed), observed
    wrong: Final = tuple((marker, effort) for marker, effort in observed if effort != by_marker[marker].rounded)
    assert wrong == (), wrong


def _assert_answered_once(outcomes: tuple[Outcome, ...], received: tuple[Request, ...]) -> None:
    upstream: Final = tuple(marker_of(request) for request in received)
    answered: Final = tuple(outcome for outcome in outcomes if outcome.status == 200)
    assert all(answer(outcome.shot.marker) in outcome.text for outcome in answered), [
        (outcome.shot, outcome.text[:300]) for outcome in answered if answer(outcome.shot.marker) not in outcome.text
    ]
    assert all(upstream.count(outcome.shot.marker) == 1 for outcome in answered), upstream


def _assert_all_answered(outcomes: tuple[Outcome, ...]) -> None:
    failed: Final = tuple(
        (outcome.shot, outcome.status, outcome.text[:300]) for outcome in outcomes if outcome.status != 200
    )
    assert failed == (), failed


def _shots(count: int) -> tuple[Shot, ...]:
    return tuple(_shot(index) for index in range(count))


def _yaml_models(wire_url: str) -> tuple[JsonValue, ...]:
    return (
        yaml_deployment("kimi-levels", "hosted_vllm", wire_url, {"reasoning_effort_levels": list(KIMI_LEVELS)}),
        yaml_deployment("claude-levels", "anthropic", wire_url, {"reasoning_effort_levels": list(CLAUDE_LEVELS)}),
    )


_OWNED_MODELS: Final = Models("kimi-levels", "claude-levels")


@contextmanager
def _owned_proxy(
    directory: Path, model_list: tuple[JsonValue, ...], settings: Mapping[str, JsonValue]
) -> Iterator[OwnedProxy]:
    directory.mkdir(parents=True, exist_ok=True)
    with gateway_from_environment() as shared:
        config: Final = owned_config(directory, model_list, litellm_settings=settings)
        with owned_proxy_process(shared, directory, {"LITELLM_LOG": "INFO"}, config=config, workers=2) as owned:
            yield owned


def _port(wire: Wire) -> int:
    port: Final = urlsplit(wire.url).port
    assert port is not None, wire.url
    return port


def test_every_request_in_a_burst_is_rounded_while_a_worker_is_killed(tmp_path: Path) -> None:
    shots: Final = _shots(_BURST)
    with wire_server(provider_reply) as wire:
        with _owned_proxy(tmp_path, _yaml_models(wire.url), {}) as owned:
            workers: Final = eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=30)
            with owned.replacing_worker(min(workers)) as kill:
                outcomes: Final = _burst(owned.gateway, _OWNED_MODELS, shots, kill)
            probes: Final = _shots(len(_KINDS))
            probe_outcomes: Final = tuple(_fire(owned.gateway, _OWNED_MODELS, shot) for shot in probes)
        received: Final = wire.drain()
    assert any(outcome.status == 200 for outcome in outcomes), outcomes
    _assert_every_upstream_effort_rounded((*shots, *probes), received)
    _assert_answered_once((*outcomes, *probe_outcomes), received)
    _assert_all_answered(probe_outcomes)


def test_every_request_in_a_burst_is_rounded_across_a_proxy_restart(tmp_path: Path) -> None:
    shots: Final = _shots(_BURST)
    probes: Final = _shots(len(_KINDS))
    with wire_server(provider_reply) as wire:
        with _owned_proxy(tmp_path / "first", _yaml_models(wire.url), {}) as first:
            outcomes: Final = _burst(first.gateway, _OWNED_MODELS, shots, first.begin_shutdown)
        with _owned_proxy(tmp_path / "second", _yaml_models(wire.url), {}) as second:
            probe_outcomes: Final = tuple(_fire(second.gateway, _OWNED_MODELS, shot) for shot in probes)
        received: Final = wire.drain()
    _assert_every_upstream_effort_rounded((*shots, *probes), received)
    _assert_answered_once((*outcomes, *probe_outcomes), received)
    _assert_all_answered(probe_outcomes)


def _held_streams(gate: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        reply: Final = provider_reply(request)
        return reply if reply.chunks is None else dataclasses.replace(reply, gate_after_first=gate)

    return respond


def _slow_streams(request: Request) -> Reply:
    reply: Final = provider_reply(request)
    return reply if reply.chunks is None else dataclasses.replace(reply, pause_between_chunks=0.2)


def test_requests_stay_rounded_when_the_upstream_stops_mid_burst_and_comes_back(gateway: Gateway) -> None:
    gate: Final = threading.Event()
    before: Final = _shots(_BURST)
    during: Final = tuple(shot for shot in _shots(12) if not shot.kind.endswith("-stream"))
    after: Final = _shots(_BURST)
    with gateway.scenario() as scenario:
        with wire_server(_held_streams(gate)) as first:
            models: Final = Models(
                deployment(scenario, "hosted_vllm", first, {"reasoning_effort_levels": list(KIMI_LEVELS)}),
                deployment(scenario, "anthropic", first, {"reasoning_effort_levels": list(CLAUDE_LEVELS)}),
            )
            port: Final = _port(first)
            with ThreadPoolExecutor(max_workers=_BURST) as pool:
                pending: Final = tuple(pool.submit(_fire, gateway, models, shot) for shot in before)
                eventually(first.received.qsize, lambda count: count == _BURST, seconds=30)
                first_received: Final = first.drain()
            held_outcomes: Final = tuple(future.result() for future in pending)
        stopped_outcomes: Final = tuple(_fire(gateway, models, shot) for shot in during)
        with wire_server(provider_reply, port=port) as second:
            after_outcomes: Final = _burst(gateway, models, after, lambda: None)
            second_received: Final = second.drain()
    _assert_every_upstream_effort_rounded(before, first_received)
    _assert_answered_once(held_outcomes, first_received)
    assert all(outcome.status is not None and outcome.status >= 500 for outcome in stopped_outcomes), [
        (outcome.status, outcome.text[:200]) for outcome in stopped_outcomes
    ]
    assert all("error" in outcome.text for outcome in stopped_outcomes), [
        outcome.text[:200] for outcome in stopped_outcomes
    ]
    _assert_every_upstream_effort_rounded(after, second_received)
    _assert_answered_once(after_outcomes, second_received)
    _assert_all_answered(after_outcomes)


def test_a_slow_streaming_upstream_neither_deadlocks_nor_duplicates_rounded_requests(gateway: Gateway) -> None:
    shots: Final = _shots(_BURST)
    with wire_server(_slow_streams) as wire, gateway.scenario() as scenario:
        models: Final = Models(
            deployment(scenario, "hosted_vllm", wire, {"reasoning_effort_levels": list(KIMI_LEVELS)}),
            deployment(scenario, "anthropic", wire, {"reasoning_effort_levels": list(CLAUDE_LEVELS)}),
        )
        outcomes: Final = _burst(gateway, models, shots, lambda: None)
        received: Final = wire.drain()
    _assert_all_answered(outcomes)
    _assert_every_upstream_effort_rounded(shots, received)
    _assert_answered_once(outcomes, received)
    assert len(received) == len(shots)


def _remap_lines(log: Path, needle: str) -> tuple[str, ...]:
    return tuple(line for line in log.read_text(errors="replace").splitlines() if needle in line)


def test_yaml_global_rounding_yields_to_a_deployment_override_and_logs_each_remap(tmp_path: Path) -> None:
    markers: Final = MappingProxyType({name: new_marker() for name in ("global", "override", "forged")})
    with wire_server(provider_reply) as wire:
        model_list: Final = (
            yaml_deployment("kimi-global", "hosted_vllm", wire.url, {"reasoning_effort_levels": list(KIMI_LEVELS)}),
            yaml_deployment(
                "kimi-override",
                "hosted_vllm",
                wire.url,
                {"reasoning_effort_levels": list(KIMI_LEVELS), "reasoning_effort_rounding": "up"},
            ),
        )
        with _owned_proxy(tmp_path, model_list, {"reasoning_effort_rounding": "down"}) as owned:
            global_response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                chat_request("kimi-global", markers["global"], reasoning_effort="medium"),
            )
            override_response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                chat_request("kimi-override", markers["override"], reasoning_effort="medium"),
            )
            forged_response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                chat_request(
                    "kimi-global",
                    markers["forged"],
                    reasoning_effort="turbo\nreasoning_effort high sent as forged",
                    drop_params=True,
                ),
            )
            remaps: Final = eventually(
                lambda: _remap_lines(owned.log, "reasoning_effort "),
                lambda lines: sum(" sent as " in line for line in lines) >= 2,
                seconds=30,
            )
        received: Final = MappingProxyType({marker_of(request): request for request in wire.drain()})
    assert [global_response.status_code, override_response.status_code, forged_response.status_code] == [200] * 3, (
        global_response.text,
        override_response.text,
        forged_response.text,
    )
    assert sent_effort(received[markers["global"]]) == "low"
    assert sent_effort(received[markers["override"]]) == "high"
    assert sent_effort(received[markers["forged"]]) is None
    assert any(
        "reasoning_effort medium sent as low (rounding down)" in line and "kimi-global" in line for line in remaps
    ), remaps
    assert any(
        "reasoning_effort medium sent as high (rounding up)" in line and "kimi-override" in line for line in remaps
    ), remaps
    assert not any(line.lstrip().startswith("reasoning_effort high sent as forged") for line in remaps), remaps
