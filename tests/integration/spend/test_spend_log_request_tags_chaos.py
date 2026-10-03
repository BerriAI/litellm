import json
import threading
import uuid
from collections.abc import Mapping, Sequence, Set
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import Final

import psutil
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration.spend._request_tag_helpers import (
    ANTHROPIC_MODEL,
    OPENAI_MODEL,
    T3,
    provider_env,
    provider_reply,
    write_config,
)

HEADERS: Final = {"user-agent": "claude-cli/2.0.0", "x-tenant-id": "tenant-a"}
ANTHROPIC_HEADERS: Final = {**HEADERS, "anthropic-version": "2023-06-01"}
EXPECTED: Final = T3
ROUTES: Final = (
    "/anthropic/v1/messages",
    "/openai/v1/chat/completions",
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/responses",
)


def _ids(response) -> str:
    for prefix in ("msg_", "chatcmpl_", "resp_"):
        if prefix in response.text:
            return prefix + response.text.split(prefix)[1].split('"')[0]
    raise AssertionError(f"no upstream id in {response.text[:200]}")


def _tagged_requests(
    candidate: Gateway, key: str, anthropic_model: str, openai_model: str, stream: bool, index: int
) -> tuple:
    """One call per route in ROUTES order with the same client headers."""
    marker: Final = f"burst {index} {uuid.uuid4().hex}"
    return (
        candidate.request(
            "POST",
            "/anthropic/v1/messages",
            {
                "model": ANTHROPIC_MODEL,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": marker}],
                "stream": stream,
            },
            key=key,
            headers=ANTHROPIC_HEADERS,
        ),
        candidate.request(
            "POST",
            "/openai/v1/chat/completions",
            {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": marker}], "stream": stream},
            key=key,
            headers=HEADERS,
        ),
        candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": openai_model, "messages": [{"role": "user", "content": marker}]},
            key=key,
            headers=HEADERS,
        ),
        candidate.request(
            "POST",
            "/v1/messages",
            {
                "model": anthropic_model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": marker}],
            },
            key=key,
            headers=ANTHROPIC_HEADERS,
        ),
        candidate.request("POST", "/v1/responses", {"model": openai_model, "input": marker}, key=key, headers=HEADERS),
    )


def _deployments(scenario, url: str) -> tuple[str, str]:
    return (
        scenario.model(model=f"anthropic/{ANTHROPIC_MODEL}", api_base=url, api_key="synthetic-anthropic-key"),
        scenario.model(model=f"openai/{OPENAI_MODEL}", api_base=f"{url}/v1"),
    )


def _landed_tags(key: str, satisfied) -> Sequence[Mapping]:
    digest: Final = sha256(key.encode()).hexdigest()
    landed: Final = eventually(
        lambda: read_rows('SELECT request_id, request_tags FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)),
        satisfied,
        seconds=60,
    )
    assert len({row["request_id"] for row in landed}) == len(landed)
    for row in landed:
        value: Final = row["request_tags"]
        assert (json.loads(value) if isinstance(value, str) else value) == EXPECTED
    return landed


def _worker_pids(owned) -> tuple[int, ...]:
    workers: Final = tuple(
        child
        for child in psutil.Process(owned.process.pid).children(recursive=True)
        if any(marker in " ".join(child.cmdline()) for marker in ("spawn_main", "integration._support.proxy"))
    )
    return tuple(worker.pid for worker in workers)


def test_burst_across_routes_records_tags_once_per_response(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy_process(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            candidate: Final = owned.gateway
            anthropic_model, openai_model = _deployments(scenario, wire.url)
            key: Final = scenario.key()

            def burst(index: int) -> tuple:
                return _tagged_requests(
                    candidate, key, anthropic_model, openai_model, stream=index % 2 == 1, index=index
                )

            with ThreadPoolExecutor(max_workers=10) as pool:
                responses: Final = tuple(chain.from_iterable(pool.map(burst, range(10))))
            assert len(responses) == 50
            assert all(response.status_code == 200 for response in responses), [
                (response.status_code, response.text[:200]) for response in responses
            ]
            ids: Final = [_ids(response) for response in responses]
            assert len(set(ids)) == 50, "duplicate upstream id in burst"
            assert len(wire.drain()) == 50
            pids: Final = _worker_pids(owned)
            assert len(set(pids)) == 2, f"expected two uvicorn workers, found {pids}"
            assert all(psutil.Process(pid).is_running() for pid in pids)
            _landed_tags(key, lambda values: len(values) == 50)


@pytest.mark.timeout(240)
def test_sink_outage_does_not_lose_spend_log_tags(gateway: Gateway, tmp_path: Path) -> None:
    down: Final = threading.Event()
    delivered: Final = []
    rejected: Final = []

    def stoppable_sink(request: Request) -> Reply:
        if down.is_set():
            rejected.append(request)
            return Reply(status=503)
        delivered.append(request)
        return Reply()

    with wire_server(provider_reply) as wire, wire_server(stoppable_sink) as endpoint:
        config: Final = write_config(
            tmp_path,
            {
                "litellm_settings": {
                    "extra_spend_tag_headers": ["x-tenant-id"],
                    "callbacks": ["generic_api"],
                    "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
                }
            },
        )
        with (
            owned_proxy_process(
                gateway,
                tmp_path,
                {
                    **provider_env(wire.url),
                    "GENERIC_LOGGER_ENDPOINT": endpoint.url,
                    "GENERIC_LOGGER_HEADERS": "Authorization=Bearer synthetic-sink-secret",
                },
                config=config,
                workers=2,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            candidate: Final = owned.gateway
            anthropic_model, openai_model = _deployments(scenario, wire.url)
            key: Final = scenario.key()

            def burst(index: int) -> tuple:
                return _tagged_requests(candidate, key, anthropic_model, openai_model, stream=False, index=index)

            with ThreadPoolExecutor(max_workers=5) as pool:
                first: Final = tuple(chain.from_iterable(pool.map(burst, range(3))))
            assert all(response.status_code == 200 for response in first), [
                (response.status_code, response.text[:200]) for response in first
            ]

            def call_ids(responses: Sequence) -> Set[str]:
                return {response.headers["x-litellm-call-id"] for response in responses}

            first_ids: Final = call_ids(first)
            first_landed: Final = _landed_tags(key, lambda values: len(values) == len(first))
            assert len(first_landed) == len(first)

            def events_for(ids: Set[str]) -> Sequence[Mapping]:
                events: Final = chain.from_iterable(json.loads(batch.body) for batch in delivered)
                return [event for event in events if event.get("litellm_call_id") in ids]

            eventually(lambda: events_for(first_ids), lambda found: len(found) >= 1, seconds=70)
            first_events: Final = events_for(first_ids)
            first_occurrences: Final = [event["litellm_call_id"] for event in first_events]
            assert set(first_occurrences) <= first_ids
            assert len(first_occurrences) == len(set(first_occurrences)), "duplicate burst-1 delivery"
            for event in first_events:
                assert event["request_tags"] == EXPECTED
            down.set()
            with ThreadPoolExecutor(max_workers=5) as pool:
                second: Final = tuple(chain.from_iterable(pool.map(lambda i: burst(100 + i), range(3))))
            second_ids: Final = call_ids(second)
            outage_probe: Final = eventually(
                lambda: (len(rejected), {event["litellm_call_id"] for event in events_for(second_ids)}),
                lambda state: state[0] >= 1 and state[1] == set(),
                seconds=30,
            )
            assert outage_probe[0] >= 1, "sink saw no rejection during the outage window"
            assert outage_probe[1] == set(), "burst-2 event delivered to a down sink"
            down.clear()
            probe: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": openai_model,
                    "messages": [{"role": "user", "content": f"recovery probe {uuid.uuid4().hex}"}],
                },
                key=key,
                headers=HEADERS,
            )
            assert probe.status_code == 200, probe.text
            probe_id: Final = probe.headers["x-litellm-call-id"]
            probe_events: Final = eventually(
                lambda: events_for({probe_id}),
                lambda found: len(found) >= 1,
                seconds=70,
            )
            assert len(probe_events) == 1, "recovery probe delivered to the sink more than once"
            assert probe_events[0]["request_tags"] == EXPECTED
            responses: Final = [*first, *second, probe]
            second_events: Final = events_for(second_ids)
            second_occurrences: Final = [event["litellm_call_id"] for event in second_events]
            assert len(second_occurrences) == len(set(second_occurrences)), (
                "duplicate burst-2 delivery after the outage"
            )

            _landed_tags(key, lambda values: len(values) == len(responses))


def test_worker_kill_mid_burst_loses_no_spend_rows(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy_process(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            candidate: Final = owned.gateway
            anthropic_model, openai_model = _deployments(scenario, wire.url)
            key: Final = scenario.key()

            def burst(index: int) -> tuple:
                return _tagged_requests(candidate, key, anthropic_model, openai_model, stream=False, index=index)

            with ThreadPoolExecutor(max_workers=5) as pool:
                first: Final = tuple(chain.from_iterable(pool.map(burst, range(3))))

            workers: Final = [
                child
                for child in psutil.Process(owned.process.pid).children(recursive=True)
                if any(marker in " ".join(child.cmdline()) for marker in ("spawn_main", "integration._support.proxy"))
            ]
            assert len(workers) == 2, (
                f"expected two uvicorn workers, found {[(w.pid, w.cmdline()[:3]) for w in workers]}"
            )
            first_landed: Final = _landed_tags(key, lambda values: len(values) == len(first))
            workers[0].kill()
            psutil.wait_procs(workers[:1], timeout=10)
            assert not workers[0].is_running()

            with ThreadPoolExecutor(max_workers=5) as pool:
                second: Final = tuple(chain.from_iterable(pool.map(lambda i: burst(100 + i), range(3))))
            responses: Final = [*first, *second]
            for position in range(len(ROUTES)):
                statuses: Final = {
                    responses[offset + position].status_code for offset in range(0, len(responses), len(ROUTES))
                }
                assert 200 in statuses, f"no surviving 200 for route {ROUTES[position]}: {statuses}"
            second_ok: Final = [response for response in second if response.status_code == 200]
            assert second_ok, "surviving worker served no second-burst request"
            ok: Final = [response for response in responses if response.status_code == 200]
            ids: Final = [_ids(response) for response in ok]
            assert len(set(ids)) == len(ids), "duplicate upstream id in burst"
            _landed_tags(key, lambda values: len(values) == len(first_landed) + len(second_ok))
