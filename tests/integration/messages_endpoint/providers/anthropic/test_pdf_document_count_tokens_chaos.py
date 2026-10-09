import asyncio
import os
import re
import signal
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import httpx
import psutil
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.pdf_document import (
    COUNT_REFUSED,
    COUNT_TOKENS_TARGET,
    LETTER,
    messages_body,
    pdf_document,
    rendered_tokens,
)
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration.providers._cache_control_marks_support import anthropic_peer, owned_config
from pydantic import JsonValue, TypeAdapter

_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_MODEL: Final = "burst-claude"
_PROVIDER_KEY: Final = "synthetic-provider-key"
_ASK: Final = "Summarize the attached report in one sentence."
_COUNT_BURST: Final = 24
_MESSAGE_BURST: Final = 12
_PAGE_COUNTS: Final = (1, 3, 12)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class _Counted:
    pages: int
    status: int
    text: str
    input_tokens: int | None


@dataclass(frozen=True, slots=True)
class _Sent:
    marker: str
    status: int
    text: str
    call_id: str


def _peer(request: Request) -> Reply:
    if request.target == COUNT_TOKENS_TARGET:
        return COUNT_REFUSED
    return anthropic_peer(request)


def _held(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert release.wait(timeout=120), "Held peer was never released"
        return _peer(request)

    return respond


def _deployment(api_base: str) -> dict[str, JsonValue]:
    return {
        "model_name": _MODEL,
        "litellm_params": {"model": "anthropic/claude-opus-5-5", "api_base": api_base, "api_key": _PROVIDER_KEY},
    }


def _count_body(pages: int) -> dict[str, JsonValue]:
    blocks: Final[list[JsonValue]] = [pdf_document((LETTER,) * pages)] if pages else []
    return {"model": _MODEL, "messages": messages_body(_MODEL, blocks, _ASK)["messages"]}


def _input_tokens(response: httpx.Response) -> int | None:
    if response.status_code != 200:
        return None
    counted: Final = _JSON_OBJECT.validate_json(response.content).get("input_tokens")
    return counted if isinstance(counted, int) else None


async def _fire_counts(url: str, key: str, *, tolerate_transport_errors: bool = False) -> tuple[_Counted, ...]:
    async def one(client: httpx.AsyncClient, index: int) -> _Counted:
        pages: Final = _PAGE_COUNTS[index % len(_PAGE_COUNTS)]
        response: Final = await client.post(
            "/v1/messages/count_tokens", json=_count_body(pages), headers={"Authorization": f"Bearer {key}"}
        )
        return _Counted(pages, response.status_code, response.text, _input_tokens(response))

    async with httpx.AsyncClient(base_url=url, timeout=90, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(one(client, index) for index in range(_COUNT_BURST)), return_exceptions=tolerate_transport_errors
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Counted))


async def _fire_messages(url: str, key: str) -> tuple[_Sent, ...]:
    async def one(client: httpx.AsyncClient) -> _Sent:
        marker: Final = uuid.uuid4().hex
        body: Final = messages_body(_MODEL, [pdf_document((LETTER,))], f"{_ASK} marker-{marker}")
        response: Final = await client.post("/v1/messages", json=body, headers={"Authorization": f"Bearer {key}"})
        return _Sent(marker, response.status_code, response.text, response.headers.get("x-litellm-call-id", ""))

    async with httpx.AsyncClient(base_url=url, timeout=90, trust_env=False) as client:
        return tuple(await asyncio.gather(*(one(client) for _ in range(_MESSAGE_BURST))))


def _baseline(gateway: Gateway) -> int:
    response: Final = gateway.request("POST", "/v1/messages/count_tokens", _count_body(0))
    assert response.status_code == 200, response.text
    counted: Final = _input_tokens(response)
    assert counted is not None, response.text
    return counted


def _assert_exact(counts: tuple[_Counted, ...], baseline: int) -> None:
    for item in counts:
        assert item.status == 200, (item.pages, item.status, item.text)
        assert item.input_tokens is not None and item.input_tokens - baseline == rendered_tokens(
            (LETTER,) * item.pages
        ), (
            item.pages,
            item.input_tokens,
            baseline,
        )


def _single_spend_row(item: _Sent) -> None:
    assert item.status == 200, (item.status, item.text)
    assert item.call_id, item.text
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (item.call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert len(rows) == 1, item.call_id


def _held_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(2 * graceful_stop_seconds() + 180)
async def test_pdf_count_burst_across_two_workers_prices_every_page_and_logs_each_message_once(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_peer) as wire:
        config: Final = owned_config(tmp_path, [_deployment(wire.url)], litellm_settings={"cache": False})
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            eventually(
                lambda: _STARTED_WORKER.findall(owned.log.read_text()),
                lambda pids: len(pids) == 2,
                seconds=graceful_stop_seconds(),
            )
            owned_url: Final = str(owned.gateway.client.base_url)
            baseline: Final = _baseline(owned.gateway)
            wire.drain()
            counts, messages = await asyncio.gather(
                _fire_counts(owned_url, owned.gateway.key), _fire_messages(owned_url, owned.gateway.key)
            )
            received: Final = wire.drain()
    targets: Final = [request.target for request in received]
    assert targets.count(COUNT_TOKENS_TARGET) == _COUNT_BURST, targets
    assert targets.count("/v1/messages") == _MESSAGE_BURST, targets
    _assert_exact(counts, baseline)
    assert {item.marker for item in messages} == {
        _JSON_OBJECT.validate_json(item.text)["id"][len("msg_") :] for item in messages if item.status == 200
    }
    for item in messages:
        _single_spend_row(item)


@pytest.mark.timeout(2 * graceful_stop_seconds() + 180)
async def test_worker_sigkill_mid_pdf_count_burst_leaves_the_sibling_pricing_pages(
    gateway: Gateway, tmp_path: Path
) -> None:
    release: Final = threading.Event()
    with wire_server(_held(release)) as wire:
        config: Final = owned_config(tmp_path, [_deployment(wire.url)], litellm_settings={"cache": False})
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            workers: Final[tuple[int, ...]] = eventually(
                lambda: tuple(int(pid) for pid in _STARTED_WORKER.findall(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=graceful_stop_seconds(),
            )
            owned_url: Final = str(owned.gateway.client.base_url)
            burst: Final = asyncio.create_task(
                _fire_counts(owned_url, owned.gateway.key, tolerate_transport_errors=True)
            )
            try:
                await asyncio.to_thread(
                    eventually, lambda: wire.received.qsize(), lambda size: size >= _COUNT_BURST, 90
                )
                held_by: Final = MappingProxyType({pid: _held_upstream_connections(pid, wire.url) for pid in workers})
                victim: Final = max(workers, key=held_by.__getitem__)
                os.kill(victim, signal.SIGKILL)
            finally:
                release.set()
            served: Final = await burst
            wire.drain()
            baseline: Final = _baseline(owned.gateway)
            after: Final = await _fire_counts(owned_url, owned.gateway.key)
            after_received: Final = wire.drain()
    assert sum(held_by.values()) == _COUNT_BURST, held_by
    assert held_by[victim] > 0, held_by
    assert len(served) == _COUNT_BURST - held_by[victim], (len(served), held_by)
    assert len(after) == _COUNT_BURST, len(after)
    assert [request.target for request in after_received] == [COUNT_TOKENS_TARGET] * (_COUNT_BURST + 1)
    _assert_exact(served, baseline)
    _assert_exact(after, baseline)
