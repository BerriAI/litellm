import concurrent.futures
import os
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Final

import httpx
from _langfuse_otel import (
    _dedupe_spans,
    _drained_spans,
    _langfuse_proxy,
    _marker_from_body,
    _proxy_config,
    _sink,
    _span_containing_marker,
    _upstream_reply_for,
)
from integration._support.client import Gateway, eventually
from integration._support.process import group_members, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server


def _endpoints(size: int) -> tuple[str, ...]:
    return tuple(
        "chat" if index < 10 else "chat_stream" if index < 20 else "messages" if index < 25 else "responses"
        for index in range(size)
    )


def _burst_body(endpoint: str, model: str, marker: str) -> dict[str, object]:
    if endpoint == "responses":
        return {"model": model, "input": marker, "cache": {"no-cache": True}}
    body: Final = {"model": model, "messages": [{"role": "user", "content": marker}], "cache": {"no-cache": True}}
    if endpoint == "messages":
        return {**body, "max_tokens": 5}
    if endpoint == "chat_stream":
        return {**body, "stream": True}
    return body


def _burst_path(endpoint: str) -> str:
    if endpoint == "responses":
        return "/v1/responses"
    if endpoint == "messages":
        return "/v1/messages"
    return "/v1/chat/completions"


def _send(candidate: Gateway, model: str, endpoint: str, marker: str) -> dict[str, object]:
    headers: Final = {
        "Authorization": f"Bearer {candidate.key}",
        "x-litellm-end-user-id": f"end-user-{marker}",
    }
    try:
        if endpoint == "chat_stream":
            with candidate.client.stream(
                "POST", "/v1/chat/completions", json=_burst_body(endpoint, model, marker), headers=headers
            ) as response:
                return {"marker": marker, "status": response.status_code, "body": response.read().decode()}
        response: Final = candidate.client.request(
            "POST", _burst_path(endpoint), json=_burst_body(endpoint, model, marker), headers=headers
        )
        return {"marker": marker, "status": response.status_code, "body": response.text}
    except httpx.HTTPError as error:
        return {"marker": marker, "status": 0, "body": repr(error)}


def _send_to_live_gateway(holder: dict[str, Gateway], model: str, endpoint: str, marker: str) -> dict[str, object]:
    deadline: Final = time.monotonic() + 60
    outcome: Final[dict[str, object]] = {"marker": marker, "status": 0, "body": "no live proxy"}
    attempts: Final = {"count": 0}
    while time.monotonic() < deadline:
        candidate: Final = holder.get("gateway")
        if candidate is None:
            time.sleep(0.25)
            continue
        result: Final = _send(candidate, model, endpoint, marker)
        attempts["count"] += 1
        if result["status"] != 0:
            result["attempts"] = attempts["count"]
            return result
        outcome.update(result)
        time.sleep(0.25)
    outcome["attempts"] = attempts["count"]
    return outcome


def _landed_counts(sink: Wire, batches: list[bytes], markers: tuple[str, ...]) -> dict[str, int]:
    spans: Final = _dedupe_spans(_drained_spans(sink, batches))
    generations: Final = tuple(span for span in spans if span[1].get("langfuse.observation.type") == "generation")
    return {marker: len(_span_containing_marker(generations, marker)) for marker in markers}


def test_langfuse_otel_sink_outage_mid_burst_never_duplicates(gateway: Gateway, tmp_path: Path) -> None:
    outage: Final = threading.Event()

    def outage_sink(request: Request) -> Reply:
        if outage.is_set():
            return Reply(status=503)
        return _sink(request)

    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(outage_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        markers: Final = tuple(uuid.uuid4().hex for _ in range(30))
        endpoints: Final = _endpoints(30)
        results: Final[dict[str, dict[str, object]]] = {}
        window: Final[set[str]] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures: Final = {
                executor.submit(_send, candidate, model, endpoints[index], markers[index]): markers[index]
                for index in range(30)
            }
            for future in concurrent.futures.as_completed(futures):
                marker: Final = futures[future]
                results[marker] = future.result()
                if len(results) == 10:
                    outage.set()
                if len(results) == 20:
                    outage.clear()
                if outage.is_set():
                    window.add(marker)
        failures: Final = {m: r for m, r in results.items() if r["status"] != 200}
        assert not failures, failures
        batches: Final[list[bytes]] = []
        counts: Final = eventually(
            lambda: _landed_counts(collector, batches, markers),
            lambda values: all(value == 1 for value in values.values()),
            seconds=60,
            return_last_on_timeout=True,
        )
        missing: Final = {m for m, c in counts.items() if c == 0}
        duplicates: Final = {m: c for m, c in counts.items() if c > 1}
        assert not duplicates, duplicates
        assert missing <= window, (missing, window, counts)


def test_langfuse_otel_slow_sink_never_duplicates_or_loses(gateway: Gateway, tmp_path: Path) -> None:
    def slow_sink(request: Request) -> Reply:
        time.sleep(1.5)
        return _sink(request)

    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(slow_sink) as collector,
        _langfuse_proxy(gateway, tmp_path, collector.url) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        markers: Final = tuple(uuid.uuid4().hex for _ in range(30))
        endpoints: Final = _endpoints(30)
        results: Final[dict[str, dict[str, object]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures: Final = {
                executor.submit(_send, candidate, model, endpoints[index], markers[index]): markers[index]
                for index in range(30)
            }
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
        failures: Final = {m: r for m, r in results.items() if r["status"] != 200}
        assert not failures, failures
        batches: Final[list[bytes]] = []
        counts: Final = eventually(
            lambda: _landed_counts(collector, batches, markers),
            lambda values: all(value == 1 for value in values.values()),
            seconds=60,
            return_last_on_timeout=True,
        )
        assert counts == {marker: 1 for marker in markers}, counts


def test_langfuse_otel_worker_kill_mid_burst_keeps_every_marker(gateway: Gateway, tmp_path: Path) -> None:
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
        owned_proxy_process(
            gateway,
            tmp_path,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-integration",
                "LANGFUSE_SECRET_KEY": "sk-integration",
                "LANGFUSE_HOST": collector.url,
                "OTEL_BSP_SCHEDULE_DELAY": "100",
            },
            config=_proxy_config(tmp_path, "langfuse_otel.yaml", ("langfuse_otel",)),
            workers=2,
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        candidate: Final = owned.gateway
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=provider.url + "/v1")
        markers: Final = tuple(uuid.uuid4().hex for _ in range(30))
        endpoints: Final = _endpoints(30)
        results: Final[dict[str, dict[str, object]]] = {}
        killed: Final = {"done": False}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures: Final = {
                executor.submit(_send, candidate, model, endpoints[index], markers[index]): markers[index]
                for index in range(30)
            }
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
                if len(results) >= 10 and not killed["done"]:
                    children: Final = tuple(
                        process for process in group_members(owned.process.pid) if process.pid != owned.process.pid
                    )
                    assert children, "no uvicorn worker child found"
                    os.kill(children[0].pid, signal.SIGKILL)
                    killed["done"] = True
        assert killed["done"], "worker kill never fired"
        accepted: Final = {m for m, r in results.items() if r["status"] == 200}
        failures: Final = {m: r for m, r in results.items() if r["status"] != 200}
        survivor: Final = _send(candidate, model, "chat", uuid.uuid4().hex)
        assert survivor["status"] == 200, survivor
        accepted_markers: Final = tuple(sorted(accepted))
        batches: Final[list[bytes]] = []
        counts: Final = eventually(
            lambda: _landed_counts(collector, batches, accepted_markers),
            lambda values: all(value == 1 for value in values.values()),
            seconds=60,
            return_last_on_timeout=True,
        )
        assert counts == {marker: 1 for marker in accepted_markers}, (counts, failures)


def test_langfuse_otel_proxy_restart_mid_burst_keeps_every_marker(gateway: Gateway, tmp_path: Path) -> None:
    with (
        wire_server(lambda request: _upstream_reply_for(request, _marker_from_body(request))) as provider,
        wire_server(_sink) as collector,
    ):
        holder: Final[dict[str, Gateway]] = {}
        markers: Final = tuple(uuid.uuid4().hex for _ in range(30))
        endpoints: Final = _endpoints(30)
        results: Final[dict[str, dict[str, object]]] = {}
        restarted: Final = {"done": False}
        one: Final = tmp_path / "one"
        two: Final = tmp_path / "two"
        one.mkdir()
        two.mkdir()
        context_one: Final = _langfuse_proxy(gateway, one, collector.url)
        context_two: Final = _langfuse_proxy(gateway, two, collector.url)
        candidate_one: Final = context_one.__enter__()
        candidate_two: Final = {"gateway": None}
        model_name: Final = uuid.uuid4().hex
        created: Final = candidate_one.post(
            "/model/new",
            {
                "model_name": model_name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": provider.url + "/v1",
                },
            },
        )
        model_id: Final = str(created["model_info"]["id"])
        holder["gateway"] = candidate_one
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures: Final = {
                    executor.submit(
                        _send_to_live_gateway, holder, model_name, endpoints[index], markers[index]
                    ): markers[index]
                    for index in range(30)
                }
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
                    if len(results) < 8 or restarted["done"]:
                        continue
                    restarted["done"] = True
                    holder.pop("gateway")
                    context_one.__exit__(None, None, None)
                    candidate_two["gateway"] = context_two.__enter__()
                    holder["gateway"] = candidate_two["gateway"]
        finally:
            if candidate_two["gateway"] is not None:
                candidate_two["gateway"].post("/model/delete", {"id": model_id})
                context_two.__exit__(None, None, None)
            context_one.__exit__(None, None, None)
        assert restarted["done"], "restart never fired"
        accepted: Final = {m for m, r in results.items() if r["status"] == 200}
        failures: Final = {m: r for m, r in results.items() if r["status"] != 200}
        accepted_markers: Final = tuple(sorted(accepted))
        batches: Final[list[bytes]] = []
        counts: Final = eventually(
            lambda: _landed_counts(collector, batches, accepted_markers),
            lambda values: all(value >= 1 for value in values.values()),
            seconds=60,
            return_last_on_timeout=True,
        )
        missing: Final = {m for m, c in counts.items() if c == 0}
        over_counted: Final = {m: c for m, c in counts.items() if c > int(results[m].get("attempts", 1))}
        assert not missing and not over_counted, (missing, over_counted, failures)
