import json
import signal
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, JsonValue, eventually, gateway_from_environment, object_value
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

BURST: Final = 30


def _provider(request: Request) -> Reply:
    try:
        body: Final = json.loads(request.body)
    except json.JSONDecodeError:
        return Reply(status=404, body=b"{}")
    text: Final = (
        body["messages"][-1]["content"] if "messages" in body else str(body.get("input", json.dumps(body)[:200]))
    )
    return Reply(
        status=400,
        body=json.dumps(
            {"error": {"type": "invalid_request_error", "message": f"Unsupported content: {text}"}}
        ).encode(),
    )


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    process: OwnedProxy
    provider: Wire
    sink: Wire
    model: str
    outage: threading.Event
    slow: threading.Event
    batches: list[Request]

    def failure_events(self) -> tuple[dict[str, JsonValue], ...]:
        self.batches.extend(self.sink.drain())  # mutable-ok: drain consumes, polls keep earlier batches
        return tuple(
            object_value(event)
            for batch in self.batches
            for event in json.loads(batch.body)
            if self.model in json.dumps(event)
        )


def _bodies(model: str, marker: str) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    chat: Final = tuple(
        (
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": f"burst {marker} {index}"}],
                **({"stream": True} if index % 2 else {}),
            },
        )
        for index in range(BURST // 3 * 2)
    )
    messages: Final = tuple(
        (
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": f"burst {marker} m{index}"}],
            },
        )
        for index in range(BURST // 6)
    )
    responses: Final = tuple(
        ("/v1/responses", {"model": model, "input": f"burst {marker} r{index}"})
        for index in range(BURST - len(chat) - len(messages))
    )
    return chat + messages + responses


def _fire(rig: Rig, bodies: tuple[tuple[str, dict[str, JsonValue]], ...]) -> tuple[tuple[int, str | None], ...]:
    def call(item: tuple[str, dict[str, JsonValue]]) -> tuple[int, str | None]:
        try:
            response: Final = rig.proxy.request("POST", item[0], item[1])
            response.read()
            return response.status_code, response.headers.get("x-litellm-call-id")
        except httpx.HTTPError:
            return -1, None

    with ThreadPoolExecutor(max_workers=8) as pool:
        return tuple(pool.map(call, bodies))


@pytest.mark.timeout(280)
def test_g1_sink_outage_mid_burst_lands_each_call_id_once_redacted(
    tmp_path: Path,
) -> None:
    marker: Final = uuid.uuid4().hex
    outage: Final = threading.Event()

    def sink(request: Request) -> Reply:
        if outage.is_set():
            return Reply(status=503, body=b'{"error":"sink down"}')
        return Reply()

    with wire_server(_provider) as provider, wire_server(sink) as endpoint:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update(
            {"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1, "turn_off_message_logging": True}
        )
        path: Final = tmp_path / "chaos.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway, tmp_path, {"GENERIC_LOGGER_ENDPOINT": endpoint.url}, config=path, workers=2
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            rig: Final = Rig(owned.gateway, owned, provider, endpoint, model, outage, threading.Event(), [])
            bodies: Final = _bodies(model, marker)
            outage.set()
            first_half: Final = _fire(rig, bodies[: BURST // 2])
            outage.clear()
            second_half: Final = _fire(rig, bodies[BURST // 2 :])
            outcomes: Final = first_half + second_half
            answered: Final = tuple(outcome for outcome in outcomes if outcome[0] >= 0)
            assert all(status == 400 for status, _ in answered), outcomes
            events: Final = eventually(rig.failure_events, lambda values: len(values) >= len(second_half), seconds=70)
            seen: Final = tuple(str(event.get("litellm_call_id")) for event in events)
            assert len(seen) == len(set(seen)), ("duplicate failure events", seen)
            for event in events:
                assert marker not in json.dumps(event), json.dumps(event)[:400]


@pytest.mark.timeout(280)
def test_g2_slow_sink_no_deadlock_no_duplicates(tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    slow: Final = threading.Event()

    def sink(request: Request) -> Reply:
        if slow.is_set():
            time.sleep(1)
        return Reply()

    with wire_server(_provider) as provider, wire_server(sink) as endpoint:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update(
            {"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1, "turn_off_message_logging": True}
        )
        path: Final = tmp_path / "chaos_slow.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway, tmp_path, {"GENERIC_LOGGER_ENDPOINT": endpoint.url}, config=path, workers=2
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            rig: Final = Rig(owned.gateway, owned, provider, endpoint, model, threading.Event(), slow, [])
            slow.set()
            bodies: Final = _bodies(model, marker)[:6]
            outcomes: Final = _fire(rig, bodies)
            assert all(status == 400 for status, _ in outcomes), outcomes
            call_ids: Final = tuple(cid for _, cid in outcomes if cid)
            events: Final = eventually(rig.failure_events, lambda values: len(values) >= len(bodies), seconds=120)
            landed: Final = tuple(str(event.get("litellm_call_id")) for event in events)
            assert len(landed) == len(set(landed)), ("duplicate failure events", landed)
            for event in events:
                assert marker not in json.dumps(event), json.dumps(event)[:400]
            assert set(call_ids) <= set(landed), (call_ids, landed)


@pytest.mark.timeout(280)
def test_g3_proxy_restart_mid_burst(tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_provider) as provider, wire_server(lambda _: Reply()) as endpoint:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update(
            {"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1, "turn_off_message_logging": True}
        )
        path: Final = tmp_path / "chaos_restart.yaml"
        path.write_text(yaml.safe_dump(config))
        overrides: Final = {"GENERIC_LOGGER_ENDPOINT": endpoint.url}
        with gateway_from_environment() as gateway:
            bodies: Final = _bodies("restart-model", marker)
            with owned_proxy_process(gateway, tmp_path, overrides, config=path, workers=2) as owned_one:
                owned_one.gateway.post(
                    "/model/new",
                    {
                        "model_name": "restart-model",
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_base": provider.url + "/v1",
                            "api_key": "synthetic-provider-key",
                        },
                    },
                )
                first: Final = _fire(
                    Rig(
                        owned_one.gateway,
                        owned_one,
                        provider,
                        endpoint,
                        "restart-model",
                        threading.Event(),
                        threading.Event(),
                        [],
                    ),
                    bodies[: BURST // 2],
                )
            with owned_proxy_process(gateway, tmp_path, overrides, config=path, workers=2) as owned_two:
                rig_two: Final = Rig(
                    owned_two.gateway,
                    owned_two,
                    provider,
                    endpoint,
                    "restart-model",
                    threading.Event(),
                    threading.Event(),
                    [],
                )

                def served() -> tuple[int, ...]:
                    probe: Final = rig_two.proxy.request("POST", *bodies[BURST // 2])
                    return (probe.status_code,)

                eventually(served, lambda statuses: statuses[0] == 400, seconds=60)
                second: Final = _fire(rig_two, bodies[BURST // 2 :])
            answered: Final = first + second
            assert all(status in (400, 500) for status, _ in answered), answered
            assert any(status == 400 for status, _ in answered), answered
            events: Final = tuple(object_value(event) for batch in endpoint.drain() for event in json.loads(batch.body))
            call_ids: Final = tuple(str(event.get("litellm_call_id")) for event in events if event)
            assert len(call_ids) == len(set(call_ids)), ("duplicate events after restart", call_ids)
            for event in events:
                assert marker not in json.dumps(event), json.dumps(event)[:400]


@pytest.mark.timeout(280)
def test_g4_worker_kill_keeps_serving_redacted(tmp_path: Path) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_provider) as provider, wire_server(lambda _: Reply()) as endpoint:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update(
            {"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1, "turn_off_message_logging": True}
        )
        path: Final = tmp_path / "chaos_worker.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            gateway_from_environment() as gateway,
            owned_proxy_process(
                gateway, tmp_path, {"GENERIC_LOGGER_ENDPOINT": endpoint.url}, config=path, workers=2
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key="synthetic-provider-key")
            rig: Final = Rig(owned.gateway, owned, provider, endpoint, model, threading.Event(), threading.Event(), [])
            bodies: Final = _bodies(model, marker)[:12]
            children: Final = psutil.Process(owned.process.pid).children(recursive=True)
            assert children, "no uvicorn worker children found"
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures: Final = tuple(
                    pool.submit(lambda b: rig.proxy.request("POST", b[0], b[1]), body) for body in bodies
                )
                eventually(lambda: provider.received.qsize() >= 3, bool, seconds=30)
                children[0].send_signal(signal.SIGKILL)
                statuses: list[int] = []  # mutable-ok: collect per-request outcomes from concurrent futures
                for future in futures:
                    try:
                        statuses.append(future.result().status_code)
                    except httpx.HTTPError:
                        statuses.append(-1)
            assert all(status == 400 for status in statuses if status >= 0), statuses
            events: Final = eventually(rig.failure_events, lambda values: len(values) >= 1, seconds=70)
            landed: Final = tuple(str(event.get("litellm_call_id")) for event in events)
            assert len(landed) == len(set(landed)), ("duplicate failure events", landed)
            for event in events:
                assert marker not in json.dumps(event), json.dumps(event)[:400]
