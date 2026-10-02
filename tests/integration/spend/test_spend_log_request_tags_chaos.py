import json
import threading
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import psutil
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration.spend._request_tag_helpers import (
    OPENAI_MODEL,
    T3,
    provider_env,
    provider_reply,
    write_config,
)

from tests.integration._support.database import read_rows

MODEL: Final = "claude-sonnet-4-5-20250929"
HEADERS: Final = {"user-agent": "claude-cli/2.0.0", "x-tenant-id": "tenant-a"}
ANTHROPIC_HEADERS: Final = {**HEADERS, "anthropic-version": "2023-06-01"}
EXPECTED: Final = T3


def _ids(response) -> str:
    for prefix in ("msg_", "chatcmpl_", "resp_"):
        if prefix in response.text:
            return prefix + response.text.split(prefix)[1].split('"')[0]
    raise AssertionError(f"no upstream id in {response.text[:200]}")


def _tagged_requests(candidate: Gateway, key: str, model: str, stream: bool, index: int) -> tuple:
    """One call per route shape, all with the same client headers; returns (response, request_id)."""
    marker: Final = f"burst {index}"
    anthropic: Final = candidate.request(
        "POST",
        "/anthropic/v1/messages",
        {"model": MODEL, "max_tokens": 16, "messages": [{"role": "user", "content": marker}], "stream": stream},
        key=key,
        headers=ANTHROPIC_HEADERS,
    )
    openai: Final = candidate.request(
        "POST",
        "/openai/v1/chat/completions",
        {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": marker}], "stream": stream},
        key=key,
        headers=HEADERS,
    )
    unified: Final = candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": marker}]},
        key=key,
        headers=HEADERS,
    )
    return anthropic, openai, unified


# C1: 10 concurrent bursts x 3 routes; each response id lands exactly one spend row with the tags
def test_burst_across_routes_records_tags_once_per_response(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model=f"openai/{OPENAI_MODEL}", api_base=f"{wire.url}/v1")
            key: Final = scenario.key()

            def burst(index: int) -> tuple:
                return _tagged_requests(candidate, key, model, stream=index % 2 == 1, index=index)

            with ThreadPoolExecutor(max_workers=10) as pool:
                responses: Final = [response for group in pool.map(burst, range(10)) for response in group]
            assert len(responses) == 30
            assert all(response.status_code == 200 for response in responses), [
                (response.status_code, response.text[:200]) for response in responses
            ]
            ids: Final = [_ids(response) for response in responses]
            assert len(set(ids)) == 30, "duplicate upstream id in burst"
            assert len(wire.drain()) == 30
            digest: Final = sha256(key.encode()).hexdigest()
            landed: Final = eventually(
                lambda: read_rows(
                    'SELECT request_id, request_tags FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)
                ),
                lambda values: len(values) == 30,
                seconds=70,
            )
            assert len({row["request_id"] for row in landed}) == 30
            for row in landed:
                value: Final = row["request_tags"]
                assert (json.loads(value) if isinstance(value, str) else value) == EXPECTED


# C2: generic_api sink down mid burst; spend rows still land exactly once with the tags
def test_sink_outage_does_not_lose_spend_log_tags(gateway: Gateway, tmp_path: Path) -> None:
    stopped: Final = threading.Event()

    def stoppable_sink(request: Request) -> Reply:
        stopped.wait(timeout=30)
        return Reply(status=503)

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
            owned_proxy(
                gateway,
                tmp_path,
                {
                    **provider_env(wire.url),
                    "GENERIC_LOGGER_ENDPOINT": endpoint.url,
                    "GENERIC_LOGGER_HEADERS": "Authorization=Bearer synthetic-sink-secret",
                },
                config=config,
                workers=2,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(model=f"openai/{OPENAI_MODEL}", api_base=f"{wire.url}/v1")
            key: Final = scenario.key()

            def burst(index: int) -> tuple:
                return _tagged_requests(candidate, key, model, stream=False, index=index)

            with ThreadPoolExecutor(max_workers=6) as pool:
                first: Final = [response for group in pool.map(burst, range(6)) for response in group]
            stopped.set()  # sink goes down: the peer now returns 503 to every flush

            def second_burst(index: int) -> tuple:
                return _tagged_requests(candidate, key, model, stream=False, index=100 + index)

            with ThreadPoolExecutor(max_workers=6) as pool:
                second: Final = [response for group in pool.map(second_burst, range(6)) for response in group]
            responses: Final = [*first, *second]
            assert all(response.status_code == 200 for response in responses), [
                (response.status_code, response.text[:200]) for response in responses
            ]
            ids: Final = [_ids(response) for response in responses]
            assert len(set(ids)) == len(ids), "duplicate upstream id in burst"
            digest: Final = sha256(key.encode()).hexdigest()
            landed: Final = eventually(
                lambda: read_rows(
                    'SELECT request_id, request_tags FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)
                ),
                lambda values: len(values) == 36,
                seconds=70,
            )
            assert len({row["request_id"] for row in landed}) == 36
            for row in landed:
                value: Final = row["request_tags"]
                assert (json.loads(value) if isinstance(value, str) else value) == EXPECTED


# C3: killing one proxy worker mid burst loses no spend row
def test_worker_kill_mid_burst_loses_no_spend_rows(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(provider_reply) as wire:
        config: Final = write_config(tmp_path, {"litellm_settings": {"extra_spend_tag_headers": ["x-tenant-id"]}})
        with (
            owned_proxy_process(gateway, tmp_path, provider_env(wire.url), config=config, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            candidate: Final = owned.gateway
            model: Final = scenario.model(model=f"openai/{OPENAI_MODEL}", api_base=f"{wire.url}/v1")
            key: Final = scenario.key()

            def burst(index: int) -> tuple:
                return _tagged_requests(candidate, key, model, stream=False, index=index)

            with ThreadPoolExecutor(max_workers=6) as pool:
                first: Final = [response for group in pool.map(burst, range(6)) for response in group]

            workers: Final = [
                child
                for child in psutil.Process(owned.process.pid).children(recursive=True)
                if child.status() != psutil.STATUS_ZOMBIE
            ]
            assert len(workers) >= 2, f"expected two proxy workers, found {[w.pid for w in workers]}"
            workers[0].kill()

            def second_burst(index: int) -> tuple:
                return _tagged_requests(candidate, key, model, stream=False, index=100 + index)

            with ThreadPoolExecutor(max_workers=6) as pool:
                second: Final = [response for group in pool.map(second_burst, range(6)) for response in group]
            responses: Final = [*first, *second]
            ok: Final = [response for response in responses if response.status_code == 200]
            ids: Final = [_ids(response) for response in ok]
            assert len(set(ids)) == len(ids), "duplicate upstream id in burst"
            digest: Final = sha256(key.encode()).hexdigest()
            landed: Final = eventually(
                lambda: read_rows(
                    'SELECT request_id, request_tags FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)
                ),
                lambda values: len(values) == len(ids),
                seconds=70,
            )
            assert len({row["request_id"] for row in landed}) == len(ids)
            for row in landed:
                value: Final = row["request_tags"]
                assert (json.loads(value) if isinstance(value, str) else value) == EXPECTED
