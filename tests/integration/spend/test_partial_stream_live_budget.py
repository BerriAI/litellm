import json
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, held
from integration._support.wire import Reply, Request, wire_server
from integration.spend._partial_stream import (
    BACKEND,
    DELIVERED_PIECES,
    JSON_OBJECT,
    PIECE,
    ROWS,
    StatusRow,
    assert_budget_refusal,
    broken_stream,
    chat,
    chats_sent,
    completion,
    failure_spend,
    peer,
    priced_model,
    recovered_cost,
    stream,
)
from pydantic import JsonValue

_CONCURRENT_STREAMS: Final = 6
_MESSAGES_BACKEND: Final = "claude-integration-partial-stream"
_UNCACHED: Final[dict[str, JsonValue]] = {"cache": {"no-cache": True}}


@pytest.mark.timeout(240)
def test_a_broken_stream_that_spends_past_the_budget_blocks_the_next_request(gateway: Gateway) -> None:
    with wire_server(peer(DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        cost: Final = recovered_cost(gateway, scenario, model)
        key: Final = scenario.key(models=[model], max_budget=cost / 2)
        chats_sent(wire)

        broken_stream(gateway, model, key)
        after: Final = chat(gateway, model, key)

        assert_budget_refusal(after)
        assert failure_spend(gateway, key, rows=2) == pytest.approx((cost, 0.0))
        assert chats_sent(wire) == 1


@pytest.mark.timeout(240)
def test_broken_stream_spend_counts_once_toward_the_budget(gateway: Gateway) -> None:
    with wire_server(peer(DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        cost: Final = recovered_cost(gateway, scenario, model)
        key: Final = scenario.key(models=[model], max_budget=cost * 1.5)

        broken_stream(gateway, model, key)
        broken_stream(gateway, model, key)
        after: Final = chat(gateway, model, key)

        assert_budget_refusal(after)
        assert failure_spend(gateway, key, rows=3) == pytest.approx((cost, cost, 0.0))


@pytest.mark.timeout(240)
def test_a_stream_that_breaks_before_any_chunk_leaves_the_budget_untouched(gateway: Gateway) -> None:
    with wire_server(peer(0)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        key: Final = scenario.key(models=[model], max_budget=0.05)

        refused: Final = stream(gateway, model, key)
        after: Final = chat(gateway, model, key)

        assert refused.status_code == 500, refused.text

        assert after.status_code == 200, after.text
        assert failure_spend(gateway, key) == (0.0,)


def _broken_stream_then_refusal(
    gateway: Gateway, model: str, spender: str, other: str, **fields: JsonValue
) -> httpx.Response:
    broken_stream(gateway, model, spender, **fields)
    return chat(gateway, model, other, **fields)


@pytest.mark.timeout(240)
def test_a_broken_stream_on_one_team_key_blocks_the_teams_other_keys(gateway: Gateway) -> None:
    with wire_server(peer(DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        cost: Final = recovered_cost(gateway, scenario, model)
        team: Final = scenario.team(models=[model], max_budget=cost / 2)

        after: Final = _broken_stream_then_refusal(
            gateway, model, scenario.key(team_id=team, models=[model]), scenario.key(team_id=team, models=[model])
        )

        assert_budget_refusal(after)


@pytest.mark.timeout(240)
def test_a_broken_stream_on_one_of_a_users_keys_blocks_their_other_keys(gateway: Gateway) -> None:
    with wire_server(peer(DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        cost: Final = recovered_cost(gateway, scenario, model)
        user: Final = scenario.user(max_budget=cost / 2)

        after: Final = _broken_stream_then_refusal(
            gateway, model, scenario.key(user_id=user, models=[model]), scenario.key(user_id=user, models=[model])
        )

        assert_budget_refusal(after)


@pytest.mark.timeout(240)
def test_a_broken_stream_for_an_end_user_blocks_their_next_request_on_any_key(gateway: Gateway) -> None:
    end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
    with wire_server(peer(DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        cost: Final = recovered_cost(gateway, scenario, model)
        gateway.post("/customer/new", {"user_id": end_user, "max_budget": cost / 2})
        scenario.cleanups.callback(gateway.post, "/customer/delete", {"user_ids": [end_user]})

        after: Final = _broken_stream_then_refusal(
            gateway, model, scenario.key(models=[model]), scenario.key(models=[model]), user=end_user
        )

        assert_budget_refusal(after)


@pytest.mark.timeout(240)
def test_a_broken_stream_under_a_tag_blocks_the_next_request_with_that_tag(gateway: Gateway) -> None:
    tag: Final = f"integration-partial-stream-{uuid.uuid4().hex}"
    with wire_server(peer(DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        cost: Final = recovered_cost(gateway, scenario, model)
        gateway.post("/tag/new", {"name": tag, "max_budget": cost / 2})
        scenario.cleanups.callback(gateway.post, "/tag/delete", {"name": tag})
        key: Final = scenario.key(models=[model])

        after: Final = _broken_stream_then_refusal(gateway, model, key, key, metadata={"tags": [tag]})
        untagged: Final = chat(gateway, model, key)

        assert_budget_refusal(after)
        assert untagged.status_code == 200, untagged.text


@pytest.mark.timeout(240)
def test_a_broken_stream_fills_the_keys_budget_window(gateway: Gateway) -> None:
    with wire_server(peer(DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        cost: Final = recovered_cost(gateway, scenario, model)
        key: Final = scenario.key(models=[model], budget_limits=[{"budget_duration": "1d", "max_budget": cost / 2}])

        after: Final = _broken_stream_then_refusal(gateway, model, key, key)

        assert_budget_refusal(after)


@pytest.mark.timeout(240)
def test_concurrent_broken_streams_each_count_once_toward_the_budget(gateway: Gateway) -> None:
    with wire_server(peer(DELIVERED_PIECES)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        cost: Final = recovered_cost(gateway, scenario, model)
        key: Final = scenario.key(models=[model], max_budget=cost * (_CONCURRENT_STREAMS + 0.5))

        with ThreadPoolExecutor(max_workers=_CONCURRENT_STREAMS) as pool:
            tuple(pool.map(lambda _: broken_stream(gateway, model, key), range(_CONCURRENT_STREAMS)))
        within_budget: Final = chat(gateway, model, key)
        broken_stream(gateway, model, key)
        over_budget: Final = chat(gateway, model, key)

        assert within_budget.status_code == 200, within_budget.text
        assert_budget_refusal(over_budget)
        assert sorted(failure_spend(gateway, key, rows=_CONCURRENT_STREAMS + 2)) == pytest.approx(
            [0.0] + [cost] * (_CONCURRENT_STREAMS + 1)
        )


@pytest.mark.timeout(240)
def test_completed_streams_count_once_toward_the_budget(gateway: Gateway) -> None:
    with wire_server(peer(None)) as wire, gateway.scenario() as scenario:
        model: Final = priced_model(scenario, wire)
        meter: Final = scenario.key(models=[model], max_budget=100)
        assert stream(gateway, model, meter, **_UNCACHED).status_code == 200
        (metered,) = eventually(
            lambda: gateway.rows(ROWS, (sha256(meter.encode()).hexdigest(),)),
            lambda found: len(found) == 1,
            seconds=70,
        )
        cost: Final = StatusRow.model_validate(metered).spend
        key: Final = scenario.key(models=[model], max_budget=cost * 1.5)

        first: Final = stream(gateway, model, key, **_UNCACHED)
        second: Final = stream(gateway, model, key, **_UNCACHED)
        assert (first.status_code, second.status_code) == (200, 200), (first.text, second.text)
        assert _rows(gateway, key, 2) == (StatusRow(status="success", spend=cost),) * 2
        third: Final = stream(gateway, model, key, **_UNCACHED)

        assert_budget_refusal(third)


def _hung_then_served(release: threading.Event, hung_model: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        if JSON_OBJECT.validate_json(request.body).get("model") == hung_model:
            return Reply(chunks=(completion(),), gate_before_first=release)
        return Reply(body=completion())

    return respond


def _timed_out_spend(gateway: Gateway, model: str, key: str) -> float:
    timed_out: Final = chat(gateway, model, key, messages=[{"role": "user", "content": "hung"}])
    assert timed_out.status_code in (408, 500), timed_out.text
    digest: Final = sha256(key.encode()).hexdigest()
    landed: Final = eventually(lambda: gateway.rows(ROWS, (digest,)), lambda found: len(found) >= 3, seconds=70)
    rows: Final = tuple(
        StatusRow.model_validate(row)
        for row in held(lambda: gateway.rows(ROWS, (digest,)), lambda latest: len(latest) == len(landed), holding=2)
    )
    assert {row.status for row in rows} == {"failure"}, rows
    return sum(row.spend for row in rows)


@pytest.mark.timeout(240)
def test_retries_the_provider_billed_before_a_timeout_count_toward_the_budget(gateway: Gateway) -> None:
    hung_backend: Final = f"{BACKEND}-hung"
    release: Final = threading.Event()
    with wire_server(_hung_then_served(release, hung_backend)) as wire, gateway.scenario() as scenario:
        scenario.cleanups.callback(release.set)
        hung: Final = scenario.model(
            model=f"openai/{hung_backend}",
            api_base=f"{wire.url}/v1",
            api_key="synthetic-openai-key",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            timeout=1,
            num_retries=2,
        )
        served: Final = priced_model(scenario, wire)
        billed: Final = _timed_out_spend(gateway, hung, scenario.key(models=[hung, served], max_budget=100))
        assert billed > 0
        key: Final = scenario.key(models=[hung, served], max_budget=billed / 2)

        assert _timed_out_spend(gateway, hung, key) == pytest.approx(billed)
        after: Final = chat(gateway, served, key)

        assert_budget_refusal(after)


def _messages_event(kind: str, fields: dict[str, JsonValue]) -> bytes:
    return f"event: {kind}\ndata: {json.dumps({'type': kind, **fields})}\n\n".encode()


def _messages_peer(request: Request) -> Reply:
    if request.method != "POST" or request.target != "/v1/messages":
        return Reply(status=404)
    message: Final[dict[str, JsonValue]] = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": _MESSAGES_BACKEND,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    if JSON_OBJECT.validate_json(request.body).get("stream") is not True:
        return Reply(
            body=json.dumps(
                {**message, "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"}
            ).encode()
        )
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _messages_event("message_start", {"message": {**message, "content": []}}),
            _messages_event("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            *(
                _messages_event("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": PIECE}})
                for _ in range(DELIVERED_PIECES)
            ),
            _messages_event("content_block_stop", {"index": 0}),
            _messages_event(
                "message_delta",
                {"delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 60}},
            ),
            _messages_event("message_stop", {}),
        ),
        abort_after=DELIVERED_PIECES + 2,
        pause_between_chunks=0.02,
    )


def _messages(gateway: Gateway, model: str, key: str, *, streamed: bool) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/messages",
        {
            "model": model,
            "max_tokens": 5,
            "stream": streamed,
            "messages": [{"role": "user", "content": f"partial stream {uuid.uuid4().hex}"}],
        },
        key=key,
    )


def _broken_messages_stream(gateway: Gateway, model: str, key: str) -> None:
    response: Final = _messages(gateway, model, key, streamed=True)
    assert response.status_code == 200, response.text
    assert response.text.count(PIECE) == DELIVERED_PIECES, response.text
    assert "message_stop" not in response.text, response.text


def _rows(gateway: Gateway, key: str, count: int) -> tuple[StatusRow, ...]:
    digest: Final = sha256(key.encode()).hexdigest()
    eventually(lambda: gateway.rows(ROWS, (digest,)), lambda found: len(found) >= count, seconds=70)
    return tuple(
        StatusRow.model_validate(row)
        for row in held(lambda: gateway.rows(ROWS, (digest,)), lambda latest: len(latest) == count, holding=2)
    )


@pytest.mark.timeout(240)
def test_a_broken_messages_stream_counts_toward_the_budget(gateway: Gateway) -> None:
    with wire_server(_messages_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MESSAGES_BACKEND}",
            api_base=wire.url,
            api_key="synthetic-anthropic-key",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        meter: Final = scenario.key(models=[model], max_budget=100)
        _broken_messages_stream(gateway, model, meter)
        (metered,) = _rows(gateway, meter, 1)
        assert metered.spend > 0, metered
        key: Final = scenario.key(models=[model], max_budget=metered.spend / 2)

        _broken_messages_stream(gateway, model, key)
        after: Final = _messages(gateway, model, key, streamed=False)

        assert_budget_refusal(after)
        assert _rows(gateway, key, 2) == (metered, StatusRow(status="failure", spend=0.0))
