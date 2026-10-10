from __future__ import annotations

import json
import os
import signal
import threading
import uuid
import zlib
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import group_members, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

BACKEND: Final = "gemini-embedding-001"
TARGET: Final = f"/models/{BACKEND}:batchEmbedContents"
END_OFFSETS: Final = {"fast": "3s", "slow": "5s", "malformed": "3s", "dropped": "9s"}
PLAN: Final = tuple(enumerate(("fast",) * 16 + ("slow",) * 8 + ("malformed",) * 8 + ("dropped",) * 8))
BURST: Final = 24
SPEND_SQL: Final = 'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s'


@dataclass(frozen=True, slots=True)
class Attempt:
    kind: str
    index: int
    status: int
    call_id: str


def source(kind: str, index: int) -> str:
    return f"gs://scripted-bucket/chaos/{kind}-{index}.mp4"


def body(model: str, kind: str, index: int) -> dict[str, JsonValue]:
    fps: Final[JsonValue] = "x" if kind == "malformed" else 1.0
    metadata: Final[dict[str, JsonValue]] = {"fps": fps, "start_offset": "0s", "end_offset": END_OFFSETS[kind]}
    return {
        "model": model,
        "input": [{"type": "file", "file": {"file_id": source(kind, index), "video_metadata": metadata}}],
    }


def first_part(request: Request) -> dict[str, JsonValue]:
    requests: Final = object_value(json.loads(request.body))["requests"]
    assert isinstance(requests, list) and len(requests) == 1, requests
    parts: Final = object_value(object_value(requests[0])["content"])["parts"]
    assert isinstance(parts, list) and len(parts) == 1, parts
    return object_value(parts[0])


def source_on_the_wire(request: Request) -> str:
    return string_value(object_value(first_part(request)["file_data"])["file_uri"])


def embed_reply(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.target.split("?")[0] == TARGET, request.target
    values: Final = [zlib.crc32(source_on_the_wire(request).encode()) / 2**32, 0.5]
    return Reply(body=json.dumps({"embeddings": [{"values": values}]}).encode())


def chaos_peer(healed: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        end_offset: Final = object_value(first_part(request)["video_metadata"])["endOffset"]
        if end_offset == END_OFFSETS["dropped"] and not healed.is_set():
            return Reply(drop_connection=True)
        reply: Final = embed_reply(request)
        if end_offset == END_OFFSETS["slow"]:
            return Reply(chunks=(reply.body[:8], reply.body[8:]), pause_between_chunks=1.5)
        return reply

    return respond


def wire_sources(wire: Wire) -> tuple[str, ...]:
    return tuple(source_on_the_wire(request) for request in wire.drain())


def spend_row_count(call_id: str) -> int:
    return len(read_rows(SPEND_SQL, (call_id,)))


def spend_row_counts(call_ids: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(spend_row_count(call_id) for call_id in call_ids)


def attempt(gateway: Gateway, model: str, index: int, kind: str) -> Attempt:
    response: Final = gateway.request("POST", "/v1/embeddings", body(model, kind, index))
    return Attempt(kind, index, response.status_code, response.headers.get("x-litellm-call-id", ""))


def of_kind(attempts: tuple[Attempt, ...], *kinds: str) -> tuple[Attempt, ...]:
    return tuple(item for item in attempts if item.kind in kinds)


@pytest.mark.timeout(240)
def test_mixed_burst_keeps_every_answer_and_spend_row_honest(gateway: Gateway) -> None:
    healed: Final = threading.Event()
    with wire_server(chaos_peer(healed)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"gemini/{BACKEND}", api_key="scripted-gemini-key", api_base=wire.url)
        with ThreadPoolExecutor(max_workers=len(PLAN)) as pool:
            attempts: Final = tuple(pool.map(lambda plan: attempt(gateway, model, plan[0], plan[1]), PLAN))
        served: Final = of_kind(attempts, "fast", "slow")
        assert all(item.status == 200 for item in served), attempts
        assert all(item.status == 400 for item in of_kind(attempts, "malformed")), attempts
        assert all(item.status >= 500 for item in of_kind(attempts, "dropped")), attempts
        reached: Final = wire_sources(wire)
        assert sorted(item for item in reached if "/malformed-" not in item and "/dropped-" not in item) == sorted(
            source(item.kind, item.index) for item in served
        ), reached
        assert not any("/malformed-" in item for item in reached), reached
        assert {item for item in reached if "/dropped-" in item} == {
            source(item.kind, item.index) for item in of_kind(attempts, "dropped")
        }, reached
        call_ids: Final = tuple(item.call_id for item in served)
        assert all(call_ids), attempts
        counts: Final = eventually(
            lambda: spend_row_counts(call_ids), lambda found: all(count >= 1 for count in found), seconds=90
        )
        assert counts == (1,) * len(call_ids), counts
        healed.set()
        with ThreadPoolExecutor(max_workers=len(PLAN)) as pool:
            resent: Final = tuple(
                pool.map(lambda item: attempt(gateway, model, item.index, item.kind), of_kind(attempts, "dropped"))
            )
        assert all(item.status == 200 for item in resent), resent
        assert sorted(wire_sources(wire)) == sorted(source(item.kind, item.index) for item in resent)


def write_config(directory: Path, url: str, name: str) -> Path:
    config: Final = directory / f"gemini_embeddings_{uuid.uuid4().hex}.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": name,
                        "litellm_params": {
                            "model": f"gemini/{BACKEND}",
                            "api_key": "scripted-gemini-key",
                            "api_base": url,
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                },
            }
        )
    )
    return config


def worker_pids(root_pid: int) -> frozenset[int]:
    def is_worker(process: psutil.Process) -> bool:
        try:
            return "spawn_main" in " ".join(process.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    return frozenset(process.pid for process in group_members(root_pid) if is_worker(process))


def outcome(candidate: Gateway, model: str, index: int) -> str:
    try:
        response: Final = candidate.request("POST", "/v1/embeddings", body(model, "fast", index))
    except httpx.TransportError as error:
        return f"transport:{type(error).__name__}"
    assert response.status_code == 200, response.text
    return f"ok:{response.headers['x-litellm-call-id']}"


@pytest.mark.timeout(300)
def test_block_embeddings_survive_a_worker_kill(gateway: Gateway, tmp_path: Path) -> None:
    name: Final = f"gemini-embeddings-{uuid.uuid4().hex}"
    with wire_server(embed_reply) as wire:
        config: Final = write_config(tmp_path, wire.url, name)
        with owned_proxy_process(gateway, tmp_path, {"STORE_MODEL_IN_DB": "False"}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            workers: Final = eventually(lambda: worker_pids(owned.process.pid), lambda pids: len(pids) == 2, seconds=30)
            victim: Final = min(workers)
            assert outcome(candidate, name, 0).startswith("ok:")

            def attempt_around_the_kill(index: int) -> str:
                if index == 2:
                    os.kill(victim, signal.SIGKILL)
                return outcome(candidate, name, index)

            with ThreadPoolExecutor(max_workers=BURST) as pool:
                outcomes: Final = tuple(pool.map(attempt_around_the_kill, range(1, BURST + 1)))
            assert outcomes.count("ok") == 0 and any(item.startswith("ok:") for item in outcomes), outcomes
            assert all(item.startswith("ok:") or item.startswith("transport:") for item in outcomes), outcomes
            respawned: Final = eventually(
                lambda: worker_pids(owned.process.pid),
                lambda pids: len(pids) == 2 and victim not in pids,
                seconds=60,
            )
            assert victim not in respawned, respawned

            def settled_burst() -> tuple[str, ...]:
                with ThreadPoolExecutor(max_workers=BURST) as pool:
                    return tuple(pool.map(lambda index: outcome(candidate, name, BURST + 1 + index), range(BURST)))

            final: Final = eventually(
                settled_burst, lambda values: all(v.startswith("ok:") for v in values), seconds=40
            )
            call_ids: Final = tuple(item.removeprefix("ok:") for item in (*outcomes, *final) if item.startswith("ok:"))
            counts: Final = eventually(
                lambda: spend_row_counts(call_ids), lambda found: all(count >= 1 for count in found), seconds=90
            )
            assert counts == (1,) * len(call_ids), counts
            assert len(wire.drain()) >= len(call_ids)
