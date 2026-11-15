from __future__ import annotations

import asyncio
import json
import socket
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Final, Literal
from urllib.parse import urlsplit

import pytest
import yaml
from integration._support.async_server import LoopbackServer
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.ports import allocate_port
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from pydantic import JsonValue, TypeAdapter

MODEL: Final = "rerank-held"
PER_QUERY_MODEL: Final = "rerank-per-query"
DEPLOYMENT_TIMEOUT_SECONDS: Final = 3.0
CANCEL_DEADLINE_SECONDS: Final = 1.0
INPUT_COST_PER_TOKEN: Final = 0.001
CELL_TIMEOUT: Final = 2 * graceful_stop_seconds() + 120
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])

Kind = Literal["arrived", "closed"]


@dataclass(frozen=True, slots=True)
class Event:
    marker: str
    kind: Kind
    at: float


@dataclass(frozen=True, slots=True)
class HeldUpstream:
    events: Queue[Event]

    async def hold(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head: Final = await reader.readuntil(b"\r\n\r\n")
        lengths: Final = tuple(
            int(line.split(b":", 1)[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:")
        )
        body: Final = await reader.readexactly(lengths[0] if lengths else 0)
        marker: Final = str(JSON_OBJECT.validate_json(body)["query"])
        self.events.put(Event(marker, "arrived", time.monotonic()))
        if marker.startswith("answer-"):
            reply: Final = json.dumps(
                {
                    "id": marker,
                    "results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}],
                    "meta": {"api_version": {"version": "2"}, "billed_units": {"search_units": 1}},
                }
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\n"
                + f"Content-Length: {len(reply)}\r\n\r\n".encode()
                + reply
            )
            await writer.drain()
            writer.close()
            return
        await reader.read()
        self.events.put(Event(marker, "closed", time.monotonic()))

    def times(self, marker: str, kind: Kind) -> tuple[float, ...]:
        with self.events.mutex:
            events: Final = tuple(self.events.queue)
        return tuple(event.at for event in events if event.marker == marker and event.kind == kind)


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: HeldUpstream

    def times(self, marker: str, kind: Kind) -> tuple[float, ...]:
        return self.upstream.times(marker, kind)


def rig_config(directory: Path, upstream_url: str, *, cancel_on_disconnect: bool) -> Path:
    base: Final = JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).parents[1] / "proxy_config.yaml").read_text())
    )
    general_settings: Final = base.get("general_settings", {})
    assert isinstance(general_settings, dict)
    config: Final[dict[str, JsonValue]] = {
        **base,
        "general_settings": {**general_settings, "cancel_on_disconnect": cancel_on_disconnect},
        "router_settings": {"num_retries": 1, "retry_after": 0},
        "model_list": [
            {
                "model_name": MODEL,
                "litellm_params": {
                    "model": "cohere/rerank-v5.0-pro",
                    "api_base": upstream_url,
                    "api_key": "synthetic-cohere-key",
                    "timeout": DEPLOYMENT_TIMEOUT_SECONDS,
                    "input_cost_per_token": INPUT_COST_PER_TOKEN,
                },
            },
            {
                "model_name": PER_QUERY_MODEL,
                "litellm_params": {
                    "model": "cohere/rerank-v3.5",
                    "api_base": upstream_url,
                    "api_key": "synthetic-cohere-key",
                    "timeout": DEPLOYMENT_TIMEOUT_SECONDS,
                },
            },
        ],
    }
    path: Final = directory / f"rerank-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def owned_rig(directory: Path, *, cancel_on_disconnect: bool) -> Iterator[Rig]:
    upstream: Final = HeldUpstream(Queue())
    port: Final = allocate_port()
    server: Final = LoopbackServer(port, upstream.hold)
    with ExitStack() as stack:
        server.start()
        stack.callback(server.stop)
        template: Final = stack.enter_context(gateway_from_environment())
        config: Final = rig_config(directory, f"http://127.0.0.1:{port}", cancel_on_disconnect=cancel_on_disconnect)
        owned: Final = stack.enter_context(owned_proxy_process(template, directory, {}, config=config))
        yield Rig(owned.gateway, upstream)


@pytest.fixture(scope="module")
def cancelling(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with owned_rig(tmp_path_factory.mktemp("rerank-cancelling"), cancel_on_disconnect=True) as rig:
        yield rig


@pytest.fixture(scope="module")
def holding(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with owned_rig(tmp_path_factory.mktemp("rerank-holding"), cancel_on_disconnect=False) as rig:
        yield rig


def rerank_request(gateway: Gateway, marker: str, call_id: str, model: str, documents: Sequence[str]) -> bytes:
    body: Final = json.dumps({"model": model, "query": marker, "documents": list(documents)}).encode()
    host: Final = urlsplit(str(gateway.client.base_url)).netloc
    return (
        f"POST /v1/rerank HTTP/1.1\r\nHost: {host}\r\nAuthorization: Bearer {gateway.key}\r\n"
        f"Content-Type: application/json\r\nx-litellm-call-id: {call_id}\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode() + body


def hang_up_once_upstream_holds(
    rig: Rig, marker: str, call_id: str, model: str = MODEL, documents: Sequence[str] = ("alpha", "beta")
) -> float:
    address: Final = urlsplit(str(rig.gateway.client.base_url))
    with socket.create_connection((address.hostname or "127.0.0.1", address.port or 80)) as client:
        client.sendall(rerank_request(rig.gateway, marker, call_id, model, documents))
        eventually(lambda: rig.times(marker, "arrived"), lambda arrived: len(arrived) == 1, seconds=30)
    return time.monotonic()


def logged_error_codes(rig: Rig, call_id: str) -> frozenset[str]:
    rows: Final = eventually(
        lambda: rig.gateway.rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return frozenset(error_code(row["metadata"]) for row in rows)


def spend_rows(rig: Rig, call_id: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: rig.gateway.rows(
            'SELECT spend, prompt_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )


def published_cost_per_query(gateway: Gateway, model_name: str) -> float:
    deployments: Final = gateway.get("/model/info")["data"]
    assert isinstance(deployments, list)
    rates: Final = tuple(
        JSON_OBJECT.validate_python(deployment["model_info"])["input_cost_per_query"]
        for deployment in map(JSON_OBJECT.validate_python, deployments)
        if deployment["model_name"] == model_name
    )
    assert len(rates) == 1, rates
    return float(str(rates[0]))


def error_code(metadata: JsonValue) -> str:
    error: Final = JSON_OBJECT.validate_python(metadata)["error_information"]
    return str(JSON_OBJECT.validate_python(error)["error_code"])


@pytest.mark.timeout(CELL_TIMEOUT)
def test_hang_up_cancels_the_upstream_rerank_without_a_retry_and_logs_499(cancelling: Rig) -> None:
    marker: Final = f"rerank-{uuid.uuid4().hex}"
    call_id: Final = str(uuid.uuid4())
    hung_up: Final = hang_up_once_upstream_holds(cancelling, marker, call_id)
    closed: Final = eventually(
        lambda: cancelling.times(marker, "closed"),
        lambda times: len(times) >= 1,
        seconds=2 * DEPLOYMENT_TIMEOUT_SECONDS,
    )
    assert closed[0] - hung_up < CANCEL_DEADLINE_SECONDS, f"upstream held {closed[0] - hung_up:.2f}s after hang-up"
    assert logged_error_codes(cancelling, call_id) == {"499"}
    assert len(cancelling.times(marker, "arrived")) == 1


@pytest.mark.timeout(CELL_TIMEOUT)
def test_hang_up_after_dispatch_bills_the_sent_query_at_the_deployment_input_rate(cancelling: Rig) -> None:
    marker: Final = f"rerank-{uuid.uuid4().hex}"
    call_id: Final = str(uuid.uuid4())
    hang_up_once_upstream_holds(cancelling, marker, call_id)
    rows: Final = spend_rows(cancelling, call_id)
    prompt_tokens: Final = int(str(rows[0]["prompt_tokens"]))
    assert prompt_tokens > 0, rows
    assert float(str(rows[0]["spend"])) == pytest.approx(prompt_tokens * INPUT_COST_PER_TOKEN)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_hang_up_on_a_per_query_model_bills_the_one_query_at_its_published_rate(cancelling: Rig) -> None:
    marker: Final = f"rerank-{uuid.uuid4().hex}"
    call_id: Final = str(uuid.uuid4())
    hang_up_once_upstream_holds(cancelling, marker, call_id, model=PER_QUERY_MODEL)
    rate: Final = published_cost_per_query(cancelling.gateway, PER_QUERY_MODEL)
    rows: Final = spend_rows(cancelling, call_id)
    assert rate > 0
    assert float(str(rows[0]["spend"])) == pytest.approx(rate), rows


@pytest.mark.timeout(CELL_TIMEOUT)
def test_hang_up_counts_the_query_and_the_documents_it_sent(cancelling: Rig) -> None:
    def logged_prompt_tokens(query_padding: str, documents: Sequence[str]) -> int:
        marker: Final = f"rerank-{uuid.uuid4().hex}{query_padding}"
        call_id: Final = str(uuid.uuid4())
        hang_up_once_upstream_holds(cancelling, marker, call_id, documents=documents)
        return int(str(spend_rows(cancelling, call_id)[0]["prompt_tokens"]))

    padding_words: Final = 800
    padding: Final = " word" * padding_words
    short_call: Final = logged_prompt_tokens("", ("alpha",))

    assert logged_prompt_tokens("", ("alpha", padding)) - short_call > padding_words // 2
    assert logged_prompt_tokens(padding, ("alpha",)) - short_call > padding_words // 2


@pytest.mark.timeout(CELL_TIMEOUT)
def test_hang_up_without_the_flag_logs_one_unbilled_row_as_before(holding: Rig) -> None:
    marker: Final = f"rerank-{uuid.uuid4().hex}"
    call_id: Final = str(uuid.uuid4())
    hang_up_once_upstream_holds(holding, marker, call_id)
    rows: Final = held(
        lambda: holding.gateway.rows(
            'SELECT spend, prompt_tokens FROM "LiteLLM_SpendLogs" WHERE request_id LIKE %s', (f"{call_id}%",)
        ),
        lambda found: len(found) == 1,
        holding=5,
        seconds=70,
    )
    assert (int(str(rows[0]["prompt_tokens"])), float(str(rows[0]["spend"]))) == (0, 0.0), rows


@pytest.mark.timeout(CELL_TIMEOUT)
def test_hang_up_without_the_flag_holds_the_upstream_until_the_deployment_timeout(holding: Rig) -> None:
    marker: Final = f"rerank-{uuid.uuid4().hex}"
    call_id: Final = str(uuid.uuid4())
    hang_up_once_upstream_holds(holding, marker, call_id)
    assert logged_error_codes(holding, call_id) == {"408"}
    arrived: Final = holding.times(marker, "arrived")
    closed: Final = holding.times(marker, "closed")
    assert len(arrived) == 2, arrived
    assert closed[0] - arrived[0] >= DEPLOYMENT_TIMEOUT_SECONDS - CANCEL_DEADLINE_SECONDS, (arrived, closed)


@pytest.mark.timeout(CELL_TIMEOUT)
def test_connected_client_still_times_out_and_retries_with_the_flag_on(cancelling: Rig) -> None:
    marker: Final = f"rerank-{uuid.uuid4().hex}"
    response: Final = cancelling.gateway.client.post(
        "/v1/rerank",
        json={"model": MODEL, "query": marker, "documents": ["alpha", "beta"]},
        headers={"Authorization": f"Bearer {cancelling.gateway.key}"},
        timeout=4 * DEPLOYMENT_TIMEOUT_SECONDS + 10,
    )
    assert response.status_code == 408, response.text
    assert len(cancelling.times(marker, "arrived")) == 2


@pytest.mark.timeout(CELL_TIMEOUT)
def test_connected_client_gets_the_rerank_answer_headers_and_a_success_row_with_the_flag_on(cancelling: Rig) -> None:
    marker: Final = f"answer-{uuid.uuid4().hex}"
    call_id: Final = str(uuid.uuid4())
    response: Final = cancelling.gateway.client.post(
        "/v1/rerank",
        json={"model": MODEL, "query": marker, "documents": ["alpha", "beta"]},
        headers={"Authorization": f"Bearer {cancelling.gateway.key}", "x-litellm-call-id": call_id},
        timeout=4 * DEPLOYMENT_TIMEOUT_SECONDS + 10,
    )
    assert response.status_code == 200, response.text
    assert [result["index"] for result in response.json()["results"]] == [1, 0]
    assert response.headers["x-litellm-call-id"] == call_id
    assert "x-litellm-response-cost" in response.headers
    rows: Final = eventually(
        lambda: cancelling.gateway.rows(
            'SELECT status, call_type FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (marker,)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    assert (rows[0]["status"], rows[0]["call_type"]) == ("success", "arerank")
    assert len(cancelling.times(marker, "arrived")) == 1
