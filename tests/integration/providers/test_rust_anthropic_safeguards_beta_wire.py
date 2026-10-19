from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final

import anthropic
import httpx
import pytest
from integration._support.client import (
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
)
from integration._support.database import read_rows
from integration._support.process import (
    OwnedProxy,
    graceful_stop_seconds,
    owned_proxy_process,
)
from integration._support.wire import Wire, wire_server
from integration.providers._anthropic_safeguards_beta_support import (
    ANTHROPIC_DEPLOYMENT,
    ANTHROPIC_KEY,
    ANTHROPIC_TARGET,
    BACKEND,
    BASH_TOOL,
    CLIENT_BETAS,
    DANGEROUS_TOOL_USE,
    DEEPSEEK_DEPLOYMENT,
    DEEPSEEK_KEY,
    DEEPSEEK_TARGET,
    MERGED_BETAS,
    NOT_A_LIST_MESSAGE,
    OVERLOADED_MESSAGE,
    REPLY_TEXT,
    SAFEGUARD_RESULTS,
    SAFEGUARDS,
    VERTEX_DEPLOYMENT,
    VERTEX_STREAM_TARGET,
    VERTEX_TARGET,
    WEB_SEARCH,
    WEB_SEARCH_TOOL,
    anthropic_peer,
    chat_body,
    deliveries,
    json_object,
    messages_body,
    new_marker,
    one_delivery,
    outage_peer,
    owned_config,
    responses_body,
    rust_routed,
    safeguarded,
    started_workers,
)
from pydantic import JsonValue

pytestmark: Final = [pytest.mark.xdist_group("anthropic_safeguards_beta"), pytest.mark.shares_worker_state]

_CELL_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 120)
_BURST: Final = 24
_RECOVERY: Final = 6
_LARGE_ENTRY: Final[JsonValue] = [
    {
        "type": "dangerous_tool_use",
        "classifier_context": {"v": 1, "permission_mode": "auto", "note": "x" * 5120},
    }
]


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    wire: Wire
    outage: threading.Event

    @property
    def gateway(self) -> Gateway:
        return self.owned.gateway


@dataclass(frozen=True, slots=True)
class _Answer:
    path: str
    marker: str
    status: int
    text: str
    rust: bool


@pytest.fixture(scope="module")
def shared_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("anthropic-safeguards-beta")
    outage: Final = threading.Event()
    with (
        gateway_from_environment() as environment,
        wire_server(outage_peer(outage)) as wire,
    ):
        config: Final = owned_config(directory, environment, wire.url)
        with owned_proxy_process(environment, directory, {"LITELLM_RUST": "true"}, config=config, workers=2) as owned:
            eventually(
                lambda: started_workers(owned.log),
                lambda pids: len(pids) == 2,
                seconds=60,
            )
            wire.drain()
            yield _Rig(owned, wire, outage)


@pytest.fixture
def rig(shared_rig: _Rig) -> _Rig:
    shared_rig.outage.clear()
    shared_rig.wire.drain()
    return shared_rig


def _spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s',
        (call_id,),
    )


def _post(gateway: Gateway, body: Mapping[str, JsonValue], *, beta: str | None = None) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/messages",
        body,
        headers={} if beta is None else {"anthropic-beta": beta},
    )


@pytest.mark.timeout(_CELL_TIMEOUT)
def test_sdk_messages_with_safeguards_go_upstream_with_the_dangerous_tool_use_beta(
    rig: _Rig,
) -> None:
    marker: Final = new_marker()
    with httpx.Client(trust_env=False, timeout=60) as http:
        client: Final = anthropic.Anthropic(
            base_url=str(rig.gateway.client.base_url),
            api_key=rig.gateway.key,
            max_retries=0,
            http_client=http,
        )
        raw: Final = client.messages.with_raw_response.create(
            model=ANTHROPIC_DEPLOYMENT,
            max_tokens=64,
            messages=[{"role": "user", "content": marker}],
            extra_body={"safeguards": list(SAFEGUARDS), "tools": [dict(BASH_TOOL)]},
        )
    message: Final = raw.parse()
    assert rust_routed(raw.headers), dict(raw.headers)
    received: Final = json_object(message.model_dump_json())
    assert message.id == f"msg_{marker}", received
    assert received.get("safeguard_results") == list(SAFEGUARD_RESULTS), received
    sent: Final = one_delivery(rig.wire, marker)
    assert (sent.target, sent.api_key, sent.beta) == (
        ANTHROPIC_TARGET,
        ANTHROPIC_KEY,
        DANGEROUS_TOOL_USE,
    ), sent
    assert sent.safeguards == list(SAFEGUARDS), sent
    assert "x-litellm-call-id" in raw.headers, dict(raw.headers)
    call_id: Final = raw.headers["x-litellm-call-id"]
    rows: Final = eventually(lambda: _spend_rows(call_id), lambda found: len(found) == 1, seconds=70)
    assert len(rows) == 1, rows


@pytest.mark.timeout(_CELL_TIMEOUT)
async def test_sdk_streamed_messages_with_safeguards_go_upstream_with_the_beta(
    rig: _Rig,
) -> None:
    marker: Final = new_marker()
    async with httpx.AsyncClient(trust_env=False, timeout=60) as http:
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(rig.gateway.client.base_url),
            api_key=rig.gateway.key,
            max_retries=0,
            http_client=http,
        )
        async with client.messages.stream(
            model=ANTHROPIC_DEPLOYMENT,
            max_tokens=64,
            messages=[{"role": "user", "content": marker}],
            extra_body={"safeguards": list(SAFEGUARDS), "tools": [dict(BASH_TOOL)]},
        ) as stream:
            text: Final = "".join([piece async for piece in stream.text_stream])
            final: Final = await stream.get_final_message()
            assert rust_routed(stream.response.headers), dict(stream.response.headers)
    assert (final.id, text) == (f"msg_{marker}", REPLY_TEXT), (final, text)
    sent: Final = one_delivery(rig.wire, marker)
    assert (sent.target, sent.beta, sent.safeguards) == (
        ANTHROPIC_TARGET,
        DANGEROUS_TOOL_USE,
        list(SAFEGUARDS),
    ), sent


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.timeout(_CELL_TIMEOUT)
def test_raw_messages_merge_the_safeguards_beta_into_the_client_betas(rig: _Rig, stream: bool) -> None:
    marker: Final = new_marker()
    body: Final = safeguarded(messages_body(ANTHROPIC_DEPLOYMENT, marker, stream=stream), list(SAFEGUARDS))
    response: Final = _post(rig.gateway, body, beta=CLIENT_BETAS)
    assert response.status_code == 200, response.text
    assert rust_routed(response.headers), dict(response.headers)
    assert f"msg_{marker}" in response.text and "safeguard_results" in response.text, response.text
    sent: Final = one_delivery(rig.wire, marker)
    assert (sent.target, sent.beta) == (ANTHROPIC_TARGET, MERGED_BETAS), sent


@pytest.mark.parametrize("extra", [{"safeguards": None}, {}], ids=["null", "absent"])
@pytest.mark.timeout(_CELL_TIMEOUT)
def test_null_or_absent_safeguards_add_no_beta(rig: _Rig, extra: Mapping[str, JsonValue]) -> None:
    marker: Final = new_marker()
    response: Final = _post(rig.gateway, {**messages_body(ANTHROPIC_DEPLOYMENT, marker), **extra})
    assert response.status_code == 200, response.text
    assert rust_routed(response.headers), dict(response.headers)
    assert f"msg_{marker}" in response.text and "safeguard_results" not in response.text, response.text
    sent: Final = one_delivery(rig.wire, marker)
    assert (sent.beta, sent.safeguards) == (None, None), sent


@pytest.mark.parametrize(
    ("safeguards", "status", "fragment"),
    [
        (7, 400, NOT_A_LIST_MESSAGE),
        ("dangerous_tool_use", 400, NOT_A_LIST_MESSAGE),
        ({}, 400, NOT_A_LIST_MESSAGE),
        ([], 200, "safeguard_results"),
        (_LARGE_ENTRY, 200, "safeguard_results"),
    ],
    ids=["int", "string", "object", "empty_list", "5kb_entry"],
)
@pytest.mark.timeout(_CELL_TIMEOUT)
def test_malformed_safeguards_still_add_the_beta_and_relay_the_vendor_verdict(
    rig: _Rig, safeguards: JsonValue, status: int, fragment: str
) -> None:
    marker: Final = new_marker()
    response: Final = _post(
        rig.gateway,
        safeguarded(messages_body(ANTHROPIC_DEPLOYMENT, marker), safeguards),
    )
    assert (response.status_code, fragment in response.text) == (status, True), response.text
    sent: Final = one_delivery(rig.wire, marker)
    assert (sent.beta, sent.safeguards) == (DANGEROUS_TOOL_USE, safeguards), sent
    control: Final = new_marker()
    after: Final = _post(rig.gateway, messages_body(ANTHROPIC_DEPLOYMENT, control))
    assert after.status_code == 200, after.text
    assert rust_routed(after.headers), dict(after.headers)
    assert one_delivery(rig.wire, control).beta is None
    assert rig.gateway.request("GET", "/health/liveliness").status_code == 200


@pytest.mark.parametrize("last", ["null", "list"])
@pytest.mark.timeout(_CELL_TIMEOUT)
def test_duplicated_safeguards_keys_resolve_to_the_last_value(rig: _Rig, last: str) -> None:
    marker: Final = new_marker()
    opening: Final = json.dumps(messages_body(ANTHROPIC_DEPLOYMENT, marker))[:-1]
    null_field: Final = '"safeguards": null'
    list_field: Final = f'"safeguards": {json.dumps(list(SAFEGUARDS))}'
    ordered: Final = (null_field, list_field) if last == "list" else (list_field, null_field)
    raw: Final = f"{opening}, {ordered[0]}, {ordered[1]}}}".encode()
    response: Final = rig.gateway.client.post(
        "/v1/messages",
        content=raw,
        headers={
            "Authorization": f"Bearer {rig.gateway.key}",
            "content-type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    assert rust_routed(response.headers), dict(response.headers)
    assert ("safeguard_results" in response.text) == (last == "list"), response.text
    sent: Final = one_delivery(rig.wire, marker)
    assert sent.beta == (DANGEROUS_TOOL_USE if last == "list" else None), sent


@pytest.mark.timeout(_CELL_TIMEOUT)
def test_safeguards_request_with_a_bad_key_never_reaches_the_peer(rig: _Rig) -> None:
    body: Final = safeguarded(messages_body(ANTHROPIC_DEPLOYMENT, new_marker()), list(SAFEGUARDS))
    response: Final = rig.gateway.request("POST", "/v1/messages", body, key="sk-not-a-key")
    assert response.status_code == 401, response.text
    assert deliveries(rig.wire) == ()


@pytest.mark.parametrize("web_search", [False, True], ids=["bash_only", "with_web_search"])
@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
@pytest.mark.timeout(_CELL_TIMEOUT)
def test_vertex_messages_with_safeguards_keep_the_beta_from_the_shared_helper(
    rig: _Rig, stream: bool, web_search: bool
) -> None:
    marker: Final = new_marker()
    tools: Final[list[JsonValue]] = [
        dict(BASH_TOOL),
        *((dict(WEB_SEARCH_TOOL),) if web_search else ()),
    ]
    body: Final = safeguarded(messages_body(VERTEX_DEPLOYMENT, marker, stream=stream), list(SAFEGUARDS))
    response: Final = _post(rig.gateway, {**body, "tools": tools})
    assert response.status_code == 200, response.text
    assert rust_routed(response.headers), dict(response.headers)
    assert f"msg_{marker}" in response.text and "safeguard_results" in response.text, response.text
    sent: Final = one_delivery(rig.wire, marker)
    assert sent.target == (VERTEX_STREAM_TARGET if stream else VERTEX_TARGET), sent
    assert sent.authorization == "Bearer scripted-token", sent
    assert sent.beta == (f"{DANGEROUS_TOOL_USE},{WEB_SEARCH}" if web_search else DANGEROUS_TOOL_USE), sent


def _beta_set(header: str | None) -> frozenset[str]:
    return frozenset(part.strip() for part in (header or "").split(",") if part.strip())


@pytest.mark.parametrize("beta", [None, CLIENT_BETAS], ids=["no-client-betas", "client-betas"])
@pytest.mark.timeout(_CELL_TIMEOUT)
def test_deepseek_messages_forward_safeguards_without_adding_a_beta(rig: _Rig, beta: str | None) -> None:
    marker: Final = new_marker()
    response: Final = _post(
        rig.gateway,
        safeguarded(messages_body(DEEPSEEK_DEPLOYMENT, marker), list(SAFEGUARDS)),
        beta=beta,
    )
    assert response.status_code == 200, response.text
    assert rust_routed(response.headers), dict(response.headers)
    assert f"msg_{marker}" in response.text, response.text
    sent: Final = one_delivery(rig.wire, marker)
    assert (sent.target, sent.api_key, sent.safeguards) == (DEEPSEEK_TARGET, DEEPSEEK_KEY, list(SAFEGUARDS)), sent
    assert _beta_set(sent.beta) <= _beta_set(beta) - {DANGEROUS_TOOL_USE}, sent


def test_python_route_adds_the_beta_for_safeguards(gateway: Gateway, scenario: Scenario) -> None:
    with wire_server(anthropic_peer) as wire:
        deployment: Final = scenario.model(model=f"anthropic/{BACKEND}", api_base=wire.url, api_key=ANTHROPIC_KEY)
        marker: Final = new_marker()
        response: Final = _post(gateway, safeguarded(messages_body(deployment, marker), list(SAFEGUARDS)))
        assert response.status_code == 200, response.text
        assert "safeguard_results" in response.text, response.text
        (sent,) = deliveries(wire)
    assert (sent.marker, sent.safeguards) == (marker, list(SAFEGUARDS)), sent
    assert DANGEROUS_TOOL_USE in _beta_set(sent.beta), sent
    assert not rust_routed(response.headers), dict(response.headers)


def _burst_request(index: int, marker: str) -> tuple[str, dict[str, JsonValue]]:
    if index % 4 == 0:
        return "/v1/messages", safeguarded(messages_body(ANTHROPIC_DEPLOYMENT, marker), list(SAFEGUARDS))
    if index % 4 == 1:
        return "/v1/messages", safeguarded(messages_body(ANTHROPIC_DEPLOYMENT, marker, stream=True), list(SAFEGUARDS))
    if index % 4 == 2:
        return "/v1/chat/completions", chat_body(ANTHROPIC_DEPLOYMENT, marker)
    return "/v1/responses", responses_body(ANTHROPIC_DEPLOYMENT, marker)


def _answer(gateway: Gateway, path: str, marker: str, body: Mapping[str, JsonValue]) -> _Answer:
    response: Final = gateway.request("POST", path, body)
    return _Answer(path, marker, response.status_code, response.text, rust_routed(response.headers))


def _wave(pool: ThreadPoolExecutor, gateway: Gateway, markers: tuple[str, ...], offset: int) -> tuple[_Answer, ...]:
    planned: Final = tuple((marker, *_burst_request(offset + index, marker)) for index, marker in enumerate(markers))
    submitted: Final = tuple(pool.submit(_answer, gateway, path, marker, body) for marker, path, body in planned)
    return tuple(future.result(timeout=120) for future in submitted)


def _assert_served(answer: _Answer) -> None:
    assert answer.status == 200, answer
    assert REPLY_TEXT in answer.text, answer
    assert answer.rust == (answer.path == "/v1/messages"), answer
    if answer.path == "/v1/messages":
        assert f"msg_{answer.marker}" in answer.text and "safeguard_results" in answer.text, answer


@pytest.mark.timeout(_CELL_TIMEOUT)
def test_peer_outage_mid_safeguards_burst_relays_each_verdict_once_and_recovers(
    rig: _Rig,
) -> None:
    healthy: Final = tuple(new_marker() for _ in range(_BURST // 2))
    outage: Final = tuple(new_marker() for _ in range(_BURST // 2))
    recovery: Final = tuple(new_marker() for _ in range(_RECOVERY))
    with ThreadPoolExecutor(max_workers=8) as pool:
        before: Final = _wave(pool, rig.gateway, healthy, 0)
        rig.outage.set()
        assert rig.gateway.request("GET", "/health/liveliness").status_code == 200
        during: Final = _wave(pool, rig.gateway, outage, len(healthy))
        rig.outage.clear()
        after: Final = _wave(pool, rig.gateway, recovery, len(healthy) + len(outage))
    for answer in (*before, *after):
        _assert_served(answer)
    for answer in during:
        assert answer.status >= 500 and OVERLOADED_MESSAGE in answer.text, answer
    markers: Final = (*healthy, *outage, *recovery)
    safeguarded_markers: Final = frozenset(
        marker for index, marker in enumerate(markers) if _burst_request(index, marker)[0] == "/v1/messages"
    )
    delivered: Final = deliveries(rig.wire)
    seen: Final = tuple(sum(sent.marker == marker for sent in delivered) for marker in markers)
    assert seen == (1,) * len(seen), seen
    expected: Final = {
        marker: (True, DANGEROUS_TOOL_USE) if marker in safeguarded_markers else (False, None) for marker in markers
    }
    for sent in delivered:
        assert (sent.safeguards is not None, sent.beta) == expected[sent.marker], sent
