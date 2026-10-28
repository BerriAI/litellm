import json
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Final

import httpx
from _langfuse_otel_metadata_support import (
    assert_key_alias,
    langfuse_rig,
    scenario_identity,
    span_attributes,
    span_response_id,
    watched_spans,
)
from integration._support.client import Gateway, eventually
from integration._support.wire import Reply, Request


def test_langfuse_otel_repeated_identical_requests_emit_one_span_each(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-repeat-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        watch: Final = watched_spans(rig.collector)
        responses: Final = tuple(
            rig.candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": identity["model"],
                    "messages": [{"role": "user", "content": marker}],
                    "cache": {"no-cache": True},
                },
                key=key,
            )
            for _ in range(3)
        )
        assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
        response_ids: Final = tuple(response.json()["id"] for response in responses)
        assert len(set(response_ids)) == 3, response_ids
        settled: Final = eventually(
            watch,
            lambda spans: all(sum(span_response_id(s) == rid for s in spans) == 1 for rid in response_ids),
            seconds=40,
        )
        for rid in response_ids:
            assert sum(span_response_id(s) == rid for s in settled) == 1


def test_langfuse_otel_sink_outage_mid_burst_lands_every_response_id_once(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-chaos-outage-" + uuid.uuid4().hex
    outage: Final = threading.Event()
    rejected: Final[list[bytes]] = []  # mutable-ok: the wire server thread records each batch it answers 503

    def sink_reply(request: Request) -> Reply:
        if outage.is_set():
            rejected.append(request.body)
            return Reply(status=503, body=b'{"error": "sink outage"}')
        return Reply()

    with (
        langfuse_rig(gateway, tmp_path, {}, marker, sink_reply=sink_reply) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        anthropic_model: Final = scenario.model(model="anthropic/claude-sonnet-4-6", api_base=rig.provider.url)
        received: Final[list[bytes]] = []  # mutable-ok: drain() consumes the wire queue, so batches accumulate

        def watch() -> tuple[dict[str, object], ...]:
            # The wire also records the batches the sink rejected; only the accepted ones landed.
            received.extend(batch.body for batch in rig.collector.drain())
            accepted: Final = Counter(received) - Counter(rejected)
            return tuple(attributes for body in accepted.elements() for attributes in span_attributes(body))

        def call(index: int) -> str:
            path: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")[index % 3]
            streaming: Final = index % 2 == 0
            body: Final = {
                "/v1/chat/completions": {
                    "model": identity["model"],
                    "messages": [{"role": "user", "content": f"{marker}-{index}"}],
                    "stream": streaming,
                    "cache": {"no-cache": True},
                },
                "/v1/messages": {
                    "model": anthropic_model,
                    "messages": [{"role": "user", "content": f"{marker}-{index}"}],
                    "max_tokens": 64,
                    "stream": streaming,
                },
                "/v1/responses": {"model": identity["model"], "input": f"{marker}-{index}", "stream": streaming},
            }[path]
            if streaming:
                with rig.candidate.client.stream(
                    "POST", path, json=body, headers={"Authorization": f"Bearer {key}"}
                ) as stream:
                    assert stream.status_code == 200, f"burst {index}: {stream.status_code}"
                    events: Final = tuple(
                        line.removeprefix("data: ")
                        for line in stream.iter_lines()
                        if line.startswith("data: ") and not line.endswith("[DONE]")
                    )
                payloads: Final = tuple(json.loads(line) for line in events)
                found: Final = next(
                    payload.get("response_id") or (payload.get("message") or {}).get("id") or payload.get("id")
                    for payload in payloads
                    if payload.get("response_id") or payload.get("message") or payload.get("id")
                )
                return str(found)
            response: Final = rig.candidate.client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
            assert response.status_code == 200, f"burst {index}: {response.text}"
            return response.json()["id"]

        with ThreadPoolExecutor(max_workers=30) as pool:
            futures: Final = tuple(pool.submit(call, index) for index in range(30))
            while sum(f.done() for f in futures) < 10:
                wait(futures, timeout=0.05)
            outage.set()
            eventually(
                lambda: (sum(f.done() for f in futures), len(rejected)),
                lambda progress: progress[0] >= 25 and progress[1] >= 1,
                seconds=20,
            )
            outage.clear()
            response_ids: Final = tuple(f.result() for f in futures)
        assert len(set(response_ids)) == 30, response_ids
        settled: Final = eventually(
            watch,
            lambda spans: all(sum(span_response_id(s) == rid for s in spans) == 1 for rid in response_ids),
            seconds=80,
            return_last_on_timeout=True,
        )
        landed: Final = sorted(
            str(span_response_id(span)) for span in settled if span_response_id(span) in response_ids
        )
        assert landed == sorted(response_ids), sorted(set(response_ids) - set(landed))
        assert len(rejected) >= 1, "sink outage never served a 503"
        assert_key_alias([span for span in settled if span_response_id(span) in response_ids], identity["key_alias"])


def test_langfuse_otel_worker_kill_mid_burst_keeps_serving_and_exports(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-chaos-kill-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        watch: Final = watched_spans(rig.collector)

        def call(index: int) -> tuple[int, str]:
            try:
                response: Final = rig.candidate.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": identity["model"],
                        "messages": [{"role": "user", "content": f"{marker}-{index}"}],
                        "cache": {"no-cache": True},
                    },
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=30,
                )
            except httpx.HTTPError as error:
                return 0, repr(error)
            return response.status_code, response.json()["id"] if response.status_code == 200 else response.text

        workers: Final = rig.proxy.worker_pids()
        assert len(workers) >= 2, f"expected multiple uvicorn workers, got {sorted(workers)!r}"
        victim: Final = min(workers)
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures: Final = tuple(pool.submit(call, index) for index in range(20))
            while sum(f.done() for f in futures) < 5:
                wait(futures, timeout=0.05)
            replacement: Final = rig.proxy.kill_worker(victim)
            outcomes: Final = tuple(f.result() for f in futures)
        assert replacement != victim
        succeeded: Final = tuple(identifier for status, identifier in outcomes if status == 200)
        assert succeeded, outcomes
        post_kill: Final = tuple(call(20 + index) for index in range(5))
        post_kill_ids: Final = tuple(identifier for _, identifier in post_kill)
        assert all(status == 200 for status, _ in post_kill), post_kill
        settled: Final = eventually(
            watch,
            lambda spans: all(sum(span_response_id(s) == rid for s in spans) == 1 for rid in post_kill_ids),
            seconds=60,
        )
        for rid in succeeded:
            assert sum(span_response_id(s) == rid for s in settled) <= 1, rid
        assert_key_alias([span for span in settled if span_response_id(span) in post_kill_ids], identity["key_alias"])


def test_langfuse_otel_slow_sink_does_not_deadlock_exports(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-chaos-slow-" + uuid.uuid4().hex

    def sink_reply(request: Request) -> Reply:
        gate: Final = threading.Event()
        threading.Timer(2, gate.set).start()
        return Reply(chunks=(b"{}",), gate_after_first=gate)

    with (
        langfuse_rig(gateway, tmp_path, {}, marker, sink_reply=sink_reply) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        watch: Final = watched_spans(rig.collector)
        responses: Final = tuple(
            rig.candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": identity["model"],
                    "messages": [{"role": "user", "content": f"{marker}-{index}"}],
                    "cache": {"no-cache": True},
                },
                key=key,
            )
            for index in range(10)
        )
        assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
        response_ids: Final = tuple(response.json()["id"] for response in responses)
        settled: Final = eventually(
            watch,
            lambda spans: all(sum(span_response_id(s) == rid for s in spans) == 1 for rid in response_ids),
            seconds=60,
        )
        assert sorted(span_response_id(span) for span in settled if span_response_id(span) in response_ids) == sorted(
            response_ids
        )
        assert_key_alias([span for span in settled if span_response_id(span) in response_ids], identity["key_alias"])
