import os
import socket
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows, scratch_database
from integration._support.pdf_document import (
    COUNT_REFUSED,
    COUNT_TOKENS_TARGET,
    LETTER,
    chat_body,
    chat_file,
    messages_body,
    pdf_data_url,
    pdf_document,
    responses_body,
    responses_input_file,
)
from integration._support.process import OwnedProxy, graceful_stop_seconds, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers._cache_control_marks_support import anthropic_peer, marker_of, owned_config
from pydantic import JsonValue, TypeAdapter
from redis import Redis

_WINDOW: Final = "window-claude"
_ITPM: Final = "itpm-claude"
_BUDGET: Final = "budget-claude"
_AFFINITY: Final = "affinity-claude"
_LIMIT: Final = 5000
_PAGE_COUNT: Final = 12
_PAGES: Final = (LETTER,) * _PAGE_COUNT
_ASK: Final = "Summarize the attached report in one sentence."
_FOLLOW_UPS: Final = 9
_TRANSCRIPT: Final = " ".join(f"Line {number}: revenue, margins and headcount moved." for number in range(160))
_PIN_KEYS: Final = "*:prompt_caching"
_PROVIDER_KEY: Final = "synthetic-provider-key"
_SURFACES: Final = ("messages", "messages-stream", "chat-file", "responses-input-file")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    wire: Wire
    held_port: int


def _marker() -> str:
    return uuid.uuid4().hex


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return int(reserve.getsockname()[1])


def _deployment(
    name: str,
    backend: str,
    api_base: str,
    *,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": f"anthropic/{backend}", "api_base": api_base, "api_key": _PROVIDER_KEY, **params},
        "model_info": dict(model_info or {}),
    }


def _peer(request: Request) -> Reply:
    if request.target == COUNT_TOKENS_TARGET:
        return COUNT_REFUSED
    return anthropic_peer(request)


def _held(release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert release.wait(timeout=120), "Held peer was never released"
        return _peer(request)

    return respond


def _request(surface: str, model: str, pages: int, marker: str) -> tuple[str, dict[str, JsonValue]]:
    text: Final = f"{_ASK} marker-{marker}"
    letter_pages: Final = (LETTER,) * pages
    if surface == "messages":
        return "/v1/messages", messages_body(model, [pdf_document(letter_pages)], text)
    if surface == "messages-stream":
        return "/v1/messages", messages_body(model, [pdf_document(letter_pages)], text, stream=True)
    if surface == "chat-file":
        return "/v1/chat/completions", chat_body(model, [chat_file(pdf_data_url(letter_pages))], text)
    return "/v1/responses", responses_body(model, [responses_input_file(pdf_data_url(letter_pages))], text)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("pdf-pre-call")
    held_port: Final = _free_port()
    with gateway_from_environment() as gateway, wire_server(_peer) as wire:
        config: Final = owned_config(
            directory,
            [
                _deployment(_WINDOW, "claude-haiku-5-5", wire.url, model_info={"max_input_tokens": _LIMIT}),
                _deployment(_ITPM, "claude-sonnet-4-6", wire.url, itpm=_LIMIT),
                _deployment(_BUDGET, "claude-opus-4-8", f"http://127.0.0.1:{held_port}"),
            ],
            litellm_settings={"cache": False},
            router_settings={
                "enable_pre_call_checks": True,
                "optional_pre_call_checks": ["enforce_model_rate_limits"],
            },
        )
        with owned_proxy_process(gateway, directory, {}, config=config, workers=2) as owned:
            yield _Rig(owned, wire, held_port)


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT status FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


@pytest.mark.parametrize("surface", _SURFACES)
def test_context_window_check_rejects_a_pdf_whose_pages_exceed_max_input_tokens(rig: _Rig, surface: str) -> None:
    rig.wire.drain()
    path, body = _request(surface, _WINDOW, _PAGE_COUNT, _marker())
    response: Final = rig.owned.gateway.request("POST", path, body)
    assert response.status_code == 400, (response.status_code, response.text)
    assert "Context Window exceeded" in response.text, response.text
    assert [request.target for request in rig.wire.drain()] == []


def test_context_window_check_passes_a_one_page_pdf_and_forwards_its_bytes(rig: _Rig) -> None:
    rig.wire.drain()
    path, body = _request("messages", _WINDOW, 1, _marker())
    response: Final = rig.owned.gateway.request("POST", path, body)
    assert response.status_code == 200, response.text
    received: Final = rig.wire.drain()
    assert [request.target for request in received] == ["/v1/messages"]
    forwarded: Final = _JSON_OBJECT.validate_json(received[0].body)["messages"]
    assert isinstance(forwarded, list) and len(forwarded) == 1, forwarded
    blocks: Final = object_value(forwarded[0])["content"]
    assert isinstance(blocks, list) and pdf_document((LETTER,)) in blocks, blocks


def test_model_itpm_check_rejects_a_pdf_whose_pages_exceed_the_limit(rig: _Rig) -> None:
    rig.wire.drain()
    path, body = _request("messages", _ITPM, _PAGE_COUNT, _marker())
    response: Final = rig.owned.gateway.request("POST", path, body)
    assert response.status_code == 429, (response.status_code, response.text)
    assert '"error"' in response.text, response.text
    assert [request.target for request in rig.wire.drain()] == []


def test_key_budget_reservation_rejects_the_second_concurrent_pdf_request(rig: _Rig) -> None:
    release: Final = threading.Event()
    requests: Final = tuple(_request("messages", _BUDGET, _PAGE_COUNT, _marker()) for _ in range(2))
    with wire_server(_held(release), port=rig.held_port) as wire, rig.owned.gateway.scenario() as scenario:
        key: Final = scenario.key(max_budget=0.01)
        headers: Final = {"Authorization": f"Bearer {key}"}
        with (
            httpx.Client(base_url=str(rig.owned.gateway.client.base_url), timeout=90, trust_env=False) as client,
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            futures: Final = tuple(
                pool.submit(client.post, path, json=body, headers=headers) for path, body in requests
            )
            try:
                eventually(
                    lambda: wire.received.qsize() + sum(1 for future in futures if future.done()),
                    lambda settled: settled >= 2,
                    seconds=60,
                )
            finally:
                release.set()
            responses: Final = tuple(future.result(timeout=90) for future in futures)
        received: Final = wire.drain()
    assert sorted(response.status_code for response in responses) == [200, 422], [
        response.text for response in responses
    ]
    rejected: Final = next(response for response in responses if response.status_code != 200)
    assert "udget" in rejected.text, rejected.text
    assert [request.target for request in received] == ["/v1/messages"]


def test_a_pre_call_rejection_logs_one_failure_row_and_calls_no_upstream(rig: _Rig) -> None:
    path, body = _request("messages", _WINDOW, _PAGE_COUNT, _marker())
    response: Final = rig.owned.gateway.request("POST", path, body)
    assert response.status_code == 400, (response.status_code, response.text)
    assert _spend_row(response.headers["x-litellm-call-id"])["status"] == "failure"
    assert rig.wire.drain() == ()


def _first_turn(session: str, marker: str) -> list[JsonValue]:
    return [
        {"role": "system", "content": f"Answer from the attached report. session {session}"},
        *chat_body(
            _AFFINITY,
            [{"type": "text", "text": _TRANSCRIPT}, chat_file(pdf_data_url(_PAGES))],
            f"{_ASK} marker-{marker}",
        )["messages"],
    ]


def _follow_up(first_turn: list[JsonValue], marker: str) -> list[JsonValue]:
    return [
        *first_turn,
        {"role": "assistant", "content": "One sentence."},
        {"role": "user", "content": f"And the margins? marker-{marker}"},
    ]


def _served(wire: Wire) -> frozenset[str]:
    return frozenset(marker_of(request) for request in wire.drain())


def _pin_count(cache: Redis) -> int:
    return len(cache.keys(_PIN_KEYS))


@pytest.mark.timeout(2 * graceful_stop_seconds() + 180)
def test_prompt_caching_affinity_pins_follow_ups_of_a_cached_turn_that_carries_a_pdf(
    gateway: Gateway, tmp_path: Path
) -> None:
    session: Final = _marker()
    first_marker: Final = _marker()
    follow_up_markers: Final = tuple(_marker() for _ in range(_FOLLOW_UPS))
    first_turn: Final = _first_turn(session, first_marker)
    with scratch_database() as database_url, wire_server(_peer) as left, wire_server(_peer) as right:
        config: Final = owned_config(
            tmp_path,
            [
                _deployment(_AFFINITY, "claude-opus-5-5", left.url, model_info={"id": f"pdf-left-{session}"}),
                _deployment(_AFFINITY, "claude-opus-5-5", right.url, model_info={"id": f"pdf-right-{session}"}),
            ],
            litellm_settings={"cache": False},
            router_settings={
                "optional_pre_call_checks": ["prompt_caching"],
                "redis_host": os.environ["REDIS_HOST"],
                "redis_port": int(os.environ["REDIS_PORT"]),
            },
        )
        with (
            Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache,
            owned_proxy_process(
                gateway,
                tmp_path,
                {"DATABASE_URL": database_url},
                config=config,
                remove_environment=("DATABASE_URL_READ_REPLICA",),
                workers=2,
            ) as owned,
            owned.gateway.scenario() as scenario,
        ):
            key: Final = scenario.key(metadata={"enable_prompt_caching": True})
            pins_before: Final = _pin_count(cache)
            first: Final = owned.gateway.request(
                "POST", "/v1/chat/completions", {"model": _AFFINITY, "messages": first_turn, "max_tokens": 64}, key=key
            )
            assert first.status_code == 200, first.text
            eventually(lambda: _pin_count(cache), lambda count: count > pins_before, seconds=70)
            follow_ups: Final = tuple(
                owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": _AFFINITY, "messages": _follow_up(first_turn, marker), "max_tokens": 64},
                    key=key,
                )
                for marker in follow_up_markers
            )
        served: Final = {"left": _served(left), "right": _served(right)}
    assert [response.status_code for response in follow_ups] == [200] * _FOLLOW_UPS, [
        response.text for response in follow_ups
    ]
    assert served["left"] | served["right"] == {first_marker, *follow_up_markers}, served
    first_side: Final = next(side for side in served if first_marker in served[side])
    assert served[first_side] == {first_marker, *follow_up_markers}, served
