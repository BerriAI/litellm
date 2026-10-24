import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue

_MODEL: Final = "claude-sonnet-4-5"
_FAST_MODE_MODEL: Final = "claude-opus-5-5"
_INPUT_RATE: Final = 1e-6
_OUTPUT_RATE: Final = 2e-6
_CACHE_READ_RATE: Final = 1e-7
_CACHE_WRITE_5M_RATE: Final = 1.25e-6
_CACHE_WRITE_1H_RATE: Final = 2e-6
_FAST_MULTIPLIER: Final = 6.0
_RATES: Final = {
    "input_cost_per_token": _INPUT_RATE,
    "output_cost_per_token": _OUTPUT_RATE,
    "cache_read_input_token_cost": _CACHE_READ_RATE,
    "cache_creation_input_token_cost": _CACHE_WRITE_5M_RATE,
    "cache_creation_input_token_cost_above_1hr": _CACHE_WRITE_1H_RATE,
}
_CONTENT: Final[tuple[dict[str, JsonValue], ...]] = ({"type": "text", "text": "PONG"},)
_USAGE: Final[dict[str, JsonValue]] = {
    "input_tokens": 100,
    "output_tokens": 50,
    "cache_read_input_tokens": 400,
    "cache_creation_input_tokens": 300,
    "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200},
}


class _BilledRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    spend: float
    prompt_tokens: int
    cache_read_cost: float
    cache_creation_cost: float


_STREAM_IDS: Final = (pytest.param(False, id="non-streamed"), pytest.param(True, id="streamed"))


def _reply(identity: str, model: str, usage: dict[str, JsonValue], stream: bool) -> Reply:
    if stream:
        return Reply(chunks=cc.message_stream(identity, model, _CONTENT, usage), content_type="text/event-stream")
    return Reply(body=cc.message_reply(identity, model, _CONTENT, usage))


def _billed_row(
    gateway: Gateway,
    model_name: str,
    usage: dict[str, JsonValue],
    stream: bool,
    request_fields: Mapping[str, JsonValue],
    model_info: Mapping[str, JsonValue] | None,
) -> _BilledRow:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return _reply(identity, model_name, usage, stream)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{model_name}",
            api_base=wire.url,
            api_key=cc.ANTHROPIC_API_KEY,
            model_info=model_info,
            **_RATES,
        )
        body: Final = {
            **cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"),
            **request_fields,
            "stream": stream,
            "model": model,
        }
        response: Final = gateway.request("POST", "/v1/messages", body)
        assert response.status_code == 200, response.text
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                "SELECT spend, prompt_tokens, "
                "(metadata->'cost_breakdown'->>'cache_read_cost')::float AS cache_read_cost, "
                "(metadata->'cost_breakdown'->>'cache_creation_cost')::float AS cache_creation_cost "
                'FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
    return _BilledRow.model_validate(rows[0])


@pytest.mark.parametrize("stream", _STREAM_IDS)
def test_cache_reads_and_5m_and_1h_cache_writes_are_each_billed_at_their_own_rate(
    gateway: Gateway, stream: bool
) -> None:
    row: Final = _billed_row(gateway, _MODEL, _USAGE, stream, {}, None)
    input_tokens, output_tokens, cache_read, cache_write_5m, cache_write_1h = 100, 50, 400, 100, 200
    assert row.prompt_tokens == input_tokens + cache_read + cache_write_5m + cache_write_1h, row
    assert row.cache_read_cost == pytest.approx(cache_read * _CACHE_READ_RATE), row
    assert row.cache_creation_cost == pytest.approx(
        cache_write_5m * _CACHE_WRITE_5M_RATE + cache_write_1h * _CACHE_WRITE_1H_RATE
    ), row
    assert row.spend == pytest.approx(
        input_tokens * _INPUT_RATE
        + output_tokens * _OUTPUT_RATE
        + cache_read * _CACHE_READ_RATE
        + cache_write_5m * _CACHE_WRITE_5M_RATE
        + cache_write_1h * _CACHE_WRITE_1H_RATE
    ), row


@pytest.mark.parametrize("stream", _STREAM_IDS)
def test_fast_mode_multiplies_cache_reads_and_writes_along_with_input_and_output(
    gateway: Gateway, stream: bool
) -> None:
    row: Final = _billed_row(
        gateway,
        _FAST_MODE_MODEL,
        {**_USAGE, "speed": "fast"},
        stream,
        {"speed": "fast"},
        {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
    )
    input_tokens, output_tokens, cache_read, cache_write_5m, cache_write_1h = 100, 50, 400, 100, 200
    assert row.spend == pytest.approx(
        _FAST_MULTIPLIER
        * (
            input_tokens * _INPUT_RATE
            + output_tokens * _OUTPUT_RATE
            + cache_read * _CACHE_READ_RATE
            + cache_write_5m * _CACHE_WRITE_5M_RATE
            + cache_write_1h * _CACHE_WRITE_1H_RATE
        )
    ), row


@pytest.mark.parametrize("stream", _STREAM_IDS)
def test_fast_mode_cost_breakdown_reports_the_multiplied_cache_costs(gateway: Gateway, stream: bool) -> None:
    pytest.skip("BUG: fast mode bills 6x cache costs but cost_breakdown.cache_read_cost/cache_creation_cost stay 1x")
    row: Final = _billed_row(
        gateway,
        _FAST_MODE_MODEL,
        {**_USAGE, "speed": "fast"},
        stream,
        {"speed": "fast"},
        {"provider_specific_entry": {"fast": _FAST_MULTIPLIER}},
    )
    cache_read, cache_write_5m, cache_write_1h = 400, 100, 200
    assert row.cache_read_cost == pytest.approx(_FAST_MULTIPLIER * cache_read * _CACHE_READ_RATE), row
    assert row.cache_creation_cost == pytest.approx(
        _FAST_MULTIPLIER * (cache_write_5m * _CACHE_WRITE_5M_RATE + cache_write_1h * _CACHE_WRITE_1H_RATE)
    ), row
