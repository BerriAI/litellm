import asyncio
import json
import os
import uuid
from hashlib import sha256
from typing import Final

import pytest
import websockets
from pydantic import BaseModel, ConfigDict, JsonValue

import litellm
from tests.integration._support.client import JSON_OBJECT, Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.upstream import delete_scenario, register_scenario
from tests.integration.cost_calculation.cost_tracking_case import RealtimeResponse


class RealtimeRates(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_cost_per_token: float
    input_cost_per_audio_token: float
    cache_read_input_token_cost: float
    cache_read_input_audio_token_cost: float | None = None
    output_cost_per_token: float
    output_cost_per_audio_token: float


MODEL: Final = "gpt-realtime-2"
RATES: Final = RealtimeRates.model_validate(litellm.get_model_info(MODEL, custom_llm_provider="openai"))
TEXT_RATE: Final = RATES.input_cost_per_token
AUDIO_RATE: Final = RATES.input_cost_per_audio_token
CACHED_TEXT_RATE: Final = RATES.cache_read_input_token_cost
CACHED_AUDIO_RATE: Final = (
    RATES.cache_read_input_token_cost
    if RATES.cache_read_input_audio_token_cost is None
    else RATES.cache_read_input_audio_token_cost
)
OUTPUT_TEXT_RATE: Final = RATES.output_cost_per_token
OUTPUT_AUDIO_RATE: Final = RATES.output_cost_per_audio_token
INPUT_TEXT_TOKENS: Final = 116
INPUT_AUDIO_TOKENS: Final = 167
CACHED_TEXT_TOKENS: Final = 64
CACHED_AUDIO_TOKENS: Final = 128
INPUT_TOKENS: Final = INPUT_TEXT_TOKENS + INPUT_AUDIO_TOKENS
OUTPUT_TEXT_TOKENS: Final = 8
OUTPUT_AUDIO_TOKENS: Final = 12
OUTPUT_TOKENS: Final = OUTPUT_TEXT_TOKENS + OUTPUT_AUDIO_TOKENS
EXPECTED_INPUT_COST: Final = (
    (INPUT_TEXT_TOKENS - CACHED_TEXT_TOKENS) * TEXT_RATE
    + CACHED_TEXT_TOKENS * CACHED_TEXT_RATE
    + (INPUT_AUDIO_TOKENS - CACHED_AUDIO_TOKENS) * AUDIO_RATE
    + CACHED_AUDIO_TOKENS * CACHED_AUDIO_RATE
)
EXPECTED_OUTPUT_COST: Final = OUTPUT_TEXT_TOKENS * OUTPUT_TEXT_RATE + OUTPUT_AUDIO_TOKENS * OUTPUT_AUDIO_RATE


def cached_audio_response_done() -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {
                "type": "response.done",
                "event_id": "evt_$REQUEST_ID",
                "response": {
                    "id": "resp_$REQUEST_ID",
                    "object": "realtime.response",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "total_tokens": INPUT_TOKENS + OUTPUT_TOKENS,
                        "input_tokens": INPUT_TOKENS,
                        "output_tokens": OUTPUT_TOKENS,
                        "input_token_details": {
                            "text_tokens": INPUT_TEXT_TOKENS,
                            "audio_tokens": INPUT_AUDIO_TOKENS,
                            "cached_tokens": CACHED_TEXT_TOKENS + CACHED_AUDIO_TOKENS,
                            "cached_tokens_details": {
                                "text_tokens": CACHED_TEXT_TOKENS,
                                "audio_tokens": CACHED_AUDIO_TOKENS,
                            },
                        },
                        "output_token_details": {
                            "text_tokens": OUTPUT_TEXT_TOKENS,
                            "audio_tokens": OUTPUT_AUDIO_TOKENS,
                        },
                    },
                },
            },
        ),
    )


async def _one_realtime_turn(proxy_url: str, key: str, model: str) -> dict[str, JsonValue]:
    async with websockets.connect(
        f"{proxy_url.replace('http://', 'ws://').replace('https://', 'wss://')}/v1/realtime?model={model}",
        additional_headers={"Authorization": f"Bearer {key}"},
    ) as websocket:
        session: Final = JSON_OBJECT.validate_json(await websocket.recv())
        await websocket.send(json.dumps({"type": "response.create"}))
        async for message in websocket:
            if JSON_OBJECT.validate_json(message).get("type") == "response.done":
                return session
        raise AssertionError(f"websocket closed before response.done for {model}")


@pytest.mark.covers("pricing.realtime.cached_audio_tokens_bill_at_audio_cache_read_rate")
def test_realtime_cached_audio_tokens_bill_at_audio_cache_read_rate_not_full_audio_rate(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"realtime-cached-audio-{uuid.uuid4().hex[:12]}"
        handle: Final = register_scenario(scenario_id, cached_audio_response_done())
        scenario.cleanups.callback(delete_scenario, handle)
        key: Final = scenario.key()
        model: Final = scenario.model(
            model=f"openai/{MODEL}", api_key=scenario_id, api_base=gateway.upstream_url.rstrip("/")
        )
        session: Final = asyncio.run(_one_realtime_turn(os.environ["INTEGRATION_PROXY_URL"].rstrip("/"), key, model))
        assert session.get("type") == "session.created", session
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, prompt_tokens, completion_tokens, call_type FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["call_type"] == "_arealtime", rows
        assert rows[0]["prompt_tokens"] == INPUT_TOKENS, rows
        assert rows[0]["completion_tokens"] == OUTPUT_TOKENS, rows
        assert float(str(rows[0]["spend"])) == pytest.approx(EXPECTED_INPUT_COST + EXPECTED_OUTPUT_COST, rel=1e-6), rows
