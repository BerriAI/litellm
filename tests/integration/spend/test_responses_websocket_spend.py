from __future__ import annotations

import asyncio
import json
import os
import uuid
from hashlib import sha256
from typing import Final

import pytest
import websockets
from integration._support.client import JSON_OBJECT, Gateway, eventually
from integration._support.database import read_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import ResponsesWebSocketResponse
from pydantic import JsonValue

_TERMINAL_TYPES: Final = frozenset({"response.completed", "response.incomplete"})


def _response_event(
    event_type: str, status: str, text: str, input_tokens: int, output_tokens: int
) -> dict[str, JsonValue]:
    return {
        "type": event_type,
        "response": {
            "id": "resp_$UNIQUE_ID",
            "object": "response",
            "status": status,
            **({"incomplete_details": {"reason": "max_output_tokens"}} if status == "incomplete" else {}),
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "id": "msg_$UNIQUE_ID",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        },
    }


async def _run_responses_websocket(url: str, key: str, model_name: str) -> tuple[dict[str, JsonValue], ...]:
    ws_url: Final = f"{url.replace('http://', 'ws://').replace('https://', 'wss://')}/v1/responses?model={model_name}"
    async with websockets.connect(ws_url, additional_headers={"Authorization": f"Bearer {key}"}) as websocket:
        terminal_events: list[dict[str, JsonValue]] = []
        for index in range(2):
            await websocket.send(json.dumps({"type": "response.create", "input": f"turn {index}"}))
            while True:
                event: Final = JSON_OBJECT.validate_json(await websocket.recv())
                if event.get("type") in _TERMINAL_TYPES:
                    terminal_events.append(event)
                    break
        return tuple(terminal_events)


@pytest.mark.covers("spend.responses_websocket.native_session_usage_is_billed")
def test_native_responses_websocket_session_records_summed_usage_and_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id: Final = f"responses-ws-{uuid.uuid4().hex[:12]}"
        completed_event: Final = _response_event("response.completed", "completed", "first turn", 10, 5)
        incomplete_event: Final = _response_event("response.incomplete", "incomplete", "second turn", 7, 4)
        handle: Final = register_scenario(
            scenario_id,
            ResponsesWebSocketResponse(
                content_type="application/x-responses-websocket",
                events=(completed_event, incomplete_event),
            ),
        )
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
            api_key=scenario_id,
            api_base=f"{gateway.upstream_url}/v1",
        )
        key: Final = scenario.key(models=[model])
        terminal_events: Final = asyncio.run(
            _run_responses_websocket(os.environ["INTEGRATION_PROXY_URL"].rstrip("/"), key, model)
        )
        totals: Final = tuple(
            JSON_OBJECT.validate_python(JSON_OBJECT.validate_python(event["response"])["usage"])["total_tokens"]
            for event in terminal_events
        )
        assert totals == (15, 11), terminal_events
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT prompt_tokens, completion_tokens, spend, call_type, status FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows == [
            {
                "prompt_tokens": 17,
                "completion_tokens": 9,
                "spend": pytest.approx(17 * 0.001 + 9 * 0.002),
                "call_type": "_aresponses_websocket",
                "status": "success",
            }
        ], rows
