import json
import signal
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import psutil
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.process import owned_proxy_process
from integration._support.redis_process import owned_redis
from integration._support.wire import Reply, Request, wire_server
from prometheus_client.parser import text_string_to_metric_families
from test_cache_hit_guardrail_metrics import (
    DEPLOYMENT_FAILURE,
    GUARDRAIL_PATH,
    PROXY_FAILED,
    Rig,
    _blocking_sink,
    _chat_body,
    _guardrail_config,
    _provider,
    _rig,
)

BURST: Final = 10


def _deployment_id(candidate: Gateway, model_name: str) -> str:
    entries: Final = candidate.get("/model/info")["data"]
    assert isinstance(entries, list)
    entry: Final = next(item for item in entries if object_value(item)["model_name"] == model_name)
    return string_value(object_value(object_value(entry)["model_info"])["id"])


def _stall_sink(stall: threading.Event, release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert request.target == GUARDRAIL_PATH, request.target
        if stall.is_set():
            release.wait(timeout=60)
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic block"}).encode())

    return respond


def _samples(candidate: Gateway, model_names: tuple[str, ...]) -> tuple:
    response: Final = candidate.client.request(
        "GET", "/metrics", headers={"Authorization": f"Bearer {candidate.key}"}, follow_redirects=True
    )
    assert response.status_code == 200, f"GET /metrics: {response.status_code}"
    return tuple(
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
        if sample.labels.get("requested_model") in model_names
    )


def _populated(samples: tuple, deployment_id: str) -> float:
    return float(
        sum(
            sample.value
            for sample in samples
            if sample.name == DEPLOYMENT_FAILURE and sample.labels.get("model_id") == deployment_id
        )
    )


def _blank(samples: tuple) -> float:
    return float(
        sum(
            sample.value
            for sample in samples
            if sample.name == DEPLOYMENT_FAILURE and sample.labels.get("model_id") == ""
        )
    )


def _proxy_failed(samples: tuple) -> float:
    return float(sum(sample.value for sample in samples if sample.name == PROXY_FAILED))


def _burst_bodies(rig: Rig, marker: str, anthropic_name: str | None) -> tuple[tuple[str, dict], ...]:
    chat: Final = tuple(
        ("/v1/chat/completions", _chat_body(rig.model_name, f"burst {marker} {index}", rig.guardrail_name))
        for index in range(BURST)
    )
    responses: Final = tuple(
        (
            "/v1/responses",
            {"model": rig.model_name, "input": f"burst {marker} r{index}", "guardrails": [rig.guardrail_name]},
        )
        for index in range(BURST)
    )
    messages: Final = (
        tuple(
            (
                "/v1/messages",
                {
                    "model": anthropic_name,
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": f"burst {marker} m{index}"}],
                    "guardrails": [rig.guardrail_name],
                },
            )
            for index in range(BURST)
        )
        if anthropic_name is not None
        else ()
    )
    return chat + responses + messages


def _warm(rig: Rig, bodies: tuple[tuple[str, dict], ...]) -> None:
    for path, body in bodies:
        warmed: Final = dict(body)
        warmed.pop("guardrails", None)
        response: Final = rig.candidate.request("POST", path, warmed)
        assert response.status_code == 200, f"warm {path}: {response.status_code} {response.text}"


def _fire(rig: Rig, bodies: tuple[tuple[str, dict], ...]) -> tuple[tuple[int, str | None], ...]:
    def call(item: tuple[str, dict]) -> tuple[int, str | None]:
        path, body = item
        try:
            response: Final = rig.candidate.request("POST", path, body)
            return response.status_code, response.headers.get("x-litellm-call-id")
        except httpx.HTTPError:
            return -1, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        return tuple(pool.map(call, bodies))


def _expect_counted_within(
    rig: Rig, model_names: tuple[str, ...], deployment_ids: tuple[str, ...], low: int, high: int
) -> None:
    def converged() -> tuple:
        samples: Final = _samples(rig.candidate, model_names)
        populated: Final = sum(_populated(samples, deployment) for deployment in deployment_ids)
        if low <= populated <= high and _blank(samples) == 0:
            return samples
        return ()

    eventually(converged, bool, seconds=70)


def _expect_exactly_once(rig: Rig, model_names: tuple[str, ...], deployment_ids: tuple[str, ...], four_xx: int) -> None:
    _expect_counted_within(rig, model_names, deployment_ids, four_xx, four_xx)


def test_burst_cache_hit_rejects_count_exactly_once(gateway: Gateway, tmp_path: Path) -> None:
    """X0: 30 mixed-endpoint cache-hit rejects across two deployments, each counted once."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker) as rig:
        anthropic_name: Final = rig.scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929", api_base=rig.provider.url, api_key="synthetic-provider-key"
        )
        anthropic_id: Final = _deployment_id(rig.candidate, anthropic_name)
        bodies: Final = _burst_bodies(rig, marker, anthropic_name)
        _warm(rig, bodies)
        outcomes: Final = _fire(rig, bodies)
        rejected: Final = sum(1 for status, _ in outcomes if status >= 400)
        assert all(status == 400 for status, _ in outcomes), outcomes
        _expect_exactly_once(rig, (rig.model_name, anthropic_name), (rig.deployment_id, anthropic_id), rejected)


def test_stalled_guardrail_sink_recovers_and_counts(gateway: Gateway, tmp_path: Path) -> None:
    """X1: guardrail sink stalls mid-burst; requests fail exactly once, then recovery counts again."""
    marker: Final = uuid.uuid4().hex
    stall: Final = threading.Event()
    release: Final = threading.Event()
    with _rig(gateway, tmp_path, marker, sink=_stall_sink(stall, release)) as rig:
        bodies: Final = _burst_bodies(rig, marker, None)
        _warm(rig, bodies)
        stall.set()
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures: Final = tuple(
                pool.submit(lambda b: rig.candidate.request("POST", b[0], b[1]), body) for body in bodies
            )
            eventually(lambda: rig.policy.received.qsize() >= 5, bool, seconds=30)
            release.set()
            outcomes: Final = tuple(
                (future.result().status_code, future.result().headers.get("x-litellm-call-id")) for future in futures
            )
        assert all(status >= 400 for status, _ in outcomes), outcomes
        blocked: Final = sum(1 for status, _ in outcomes if status == 400)
        outages: Final = sum(1 for status, _ in outcomes if status >= 500)
        assert blocked + outages == len(bodies), outcomes
        samples: Final = eventually(
            lambda: _samples(rig.candidate, (rig.model_name,)),
            lambda observed: _proxy_failed(observed) == blocked + outages,
            seconds=70,
        )
        assert _proxy_failed(samples) == blocked + outages, (samples, outcomes)
        follow_up: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            _chat_body(rig.model_name, "post stall unrelated " + marker, None),
        )
        assert follow_up.status_code == 200, follow_up.text
        _expect_exactly_once(rig, (rig.model_name,), (rig.deployment_id,), blocked)


def test_redis_outage_keeps_serving_in_memory_hits(gateway: Gateway, tmp_path: Path) -> None:
    """X2: the redis cache keeps an in-memory shadow, so a redis outage does not stop cache-hit rejects."""
    marker: Final = uuid.uuid4().hex
    with owned_redis(tmp_path) as cache:
        with _rig(gateway, tmp_path, marker, env={"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)}) as rig:
            bodies: Final = _burst_bodies(rig, marker, None)[:BURST]
            _warm(rig, bodies)
            reject: Final = rig.candidate.request("POST", *bodies[0])
            assert reject.status_code == 400, reject.text
            warmed_hits: Final = rig.provider.received.qsize()
            cache.stop()
            outcomes: Final = _fire(rig, bodies[1:])
            assert all(status == 400 for status, _ in outcomes), outcomes
            assert rig.provider.received.qsize() == warmed_hits, (
                "redis outage reached the provider",
                warmed_hits,
                rig.provider.received.qsize(),
            )
            cache.start()
            recovered: Final = rig.candidate.request(
                "POST",
                "/v1/chat/completions",
                _chat_body(rig.model_name, "x2 rehit " + marker, rig.guardrail_name),
            )
            assert recovered.status_code == 400, recovered.text
            _expect_exactly_once(rig, (rig.model_name,), (rig.deployment_id,), 1 + len(bodies))


def test_worker_kill_mid_burst_keeps_counting(gateway: Gateway, tmp_path: Path) -> None:
    """X3: workers=2, SIGKILL one uvicorn child mid-burst; survivors keep rejecting; the count is answered plus at most the in-flight requests the killed worker had already counted."""
    marker: Final = uuid.uuid4().hex
    with _rig(gateway, tmp_path, marker, workers=2) as rig:
        bodies: Final = _burst_bodies(rig, marker, None)
        _warm(rig, bodies)
        children: Final = psutil.Process(rig.process.pid).children(recursive=True)
        assert children, "no uvicorn worker children found"
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures: Final = tuple(
                pool.submit(lambda b: rig.candidate.request("POST", b[0], b[1]), body) for body in bodies
            )
            eventually(lambda: rig.policy.received.qsize() >= 3, bool, seconds=30)
            children[0].send_signal(signal.SIGKILL)
            statuses: list[int] = []  # mutable-ok: collect per-request outcomes from concurrent futures
            for future in futures:
                try:
                    statuses.append(future.result().status_code)
                except httpx.HTTPError:
                    statuses.append(-1)
            answered: Final = sum(1 for status in statuses if status >= 0)
            transport_lost: Final = sum(1 for status in statuses if status == -1)
        assert all(status == 400 for status in statuses if status >= 0), (
            statuses,
            transport_lost,
        )
        _expect_counted_within(rig, (rig.model_name,), (rig.deployment_id,), answered, answered + transport_lost)


def test_proxy_restart_mid_burst_keeps_counting(gateway: Gateway, tmp_path: Path) -> None:
    """X4: restart the owned proxy between the two halves; pre-restart count asserted, then recounted."""
    marker: Final = uuid.uuid4().hex
    prom_dir: Final = tmp_path / "prom"
    prom_dir.mkdir()
    with wire_server(_blocking_sink) as policy, wire_server(_provider(marker)) as provider:
        config: Final = _guardrail_config(tmp_path, "guardrail-" + marker, policy.url)
        bodies: Final = tuple(
            (
                "/v1/chat/completions",
                _chat_body("pending-model", f"burst {marker} {index}", "guardrail-" + marker),
            )
            for index in range(BURST)
        )
        with owned_proxy_process(
            gateway, tmp_path, {"PROMETHEUS_MULTIPROC_DIR": str(prom_dir)}, config=config
        ) as owned_one:
            model: Final = "restart-" + marker
            owned_one.gateway.post(
                "/model/new",
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_base": provider.url + "/v1",
                        "api_key": "synthetic-provider-key",
                    },
                },
            )
            deployment: Final = _deployment_id(owned_one.gateway, model)
            named: Final = tuple((path, {**body, "model": model}) for path, body in bodies)
            first_half, second_half = named[: BURST // 2], named[BURST // 2 :]
            for path, body in named:
                warmed: Final = dict(body)
                warmed.pop("guardrails", None)
                assert owned_one.gateway.request("POST", path, warmed).status_code == 200
            outcomes_one: Final = tuple(owned_one.gateway.request("POST", path, body) for path, body in first_half)
            assert all(response.status_code == 400 for response in outcomes_one), [r.text for r in outcomes_one]
            pre: Final = eventually(
                lambda: (
                    _populated(_samples(owned_one.gateway, (model,)), deployment),
                    _blank(_samples(owned_one.gateway, (model,))),
                ),
                lambda observed: observed[0] == len(first_half) and observed[1] == 0,
                seconds=70,
            )
        with owned_proxy_process(
            gateway, tmp_path, {"PROMETHEUS_MULTIPROC_DIR": str(prom_dir)}, config=config
        ) as owned_two:
            outcomes_two: Final = tuple(owned_two.gateway.request("POST", path, body) for path, body in second_half)
            assert all(response.status_code == 400 for response in outcomes_two), (
                pre,
                [(r.status_code, r.text[:200]) for r in outcomes_two],
            )
            post: Final = eventually(
                lambda: (
                    _populated(_samples(owned_two.gateway, (model,)), deployment),
                    _blank(_samples(owned_two.gateway, (model,))),
                ),
                lambda observed: observed[0] == len(named) and observed[1] == 0,
                seconds=70,
            )
            assert post[0] == len(named), (pre, post, outcomes_two)
            owned_two.gateway.post("/model/delete", {"id": deployment})
