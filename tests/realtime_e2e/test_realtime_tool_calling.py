"""Layer 1: full tool-calling round-trip through the proxy, per provider.

This is the highest-signal test: it exercises the part of each provider's
transform that the current suite does not cover -> emitting a normalized
function call AND round-tripping the tool result back into a follow-up
response. Gemini/Bedrock <-> OpenAI-GA mappings break here first.
"""

from __future__ import annotations

import json

import pytest

from .conftest import skip_if_creds_missing
from .providers import PROVIDER_IDS, PROVIDERS, RealtimeProvider
from .raw_ws_client import (
    connect_realtime,
    function_call_item,
    function_call_output_event,
    is_type,
    last_of_type,
    response_transcript,
    user_text_event,
)

pytestmark = [pytest.mark.realtime_e2e, pytest.mark.asyncio]

WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current temperature in Fahrenheit for a given city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}


@pytest.mark.parametrize("provider", PROVIDERS, ids=PROVIDER_IDS)
async def test_tool_call_round_trip(
    provider: RealtimeProvider, proxy_ws_url: str, proxy_api_key: str
) -> None:
    skip_if_creds_missing(provider)

    async with connect_realtime(
        proxy_ws_url=proxy_ws_url, api_key=proxy_api_key, model=provider.model
    ) as conn:
        await conn.collect_until(is_type("session.created"), timeout=20)

        await conn.send_event(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["text"],
                    "tools": [WEATHER_TOOL],
                    "tool_choice": "auto",
                    "instructions": (
                        "Use the get_weather tool whenever the user asks about weather. "
                        "After receiving the result, state the temperature."
                    ),
                },
            }
        )
        await conn.collect_until(is_type("session.updated"), timeout=20)

        await conn.send_event(user_text_event("What's the weather in Paris right now?"))
        await conn.send_event({"type": "response.create"})
        first = await conn.collect_until(is_type("response.done"), timeout=60)

        args_done = last_of_type(first, "response.function_call_arguments.done")
        assert args_done is not None, "model did not emit a function call"
        arguments = json.loads(args_done.raw["arguments"])
        assert "city" in arguments, f"tool args missing 'city': {arguments}"
        call_id = args_done.raw["call_id"]

        item = function_call_item(first)
        assert item is not None, "no completed function_call output item"
        assert item["name"] == "get_weather"
        assert item["call_id"] == call_id

        await conn.send_event(
            function_call_output_event(
                call_id, {"city": arguments["city"], "temperature_f": 72}
            )
        )
        await conn.send_event({"type": "response.create"})
        second = await conn.collect_until(is_type("response.done"), timeout=60)

        follow_up = response_transcript(second)
        assert "72" in follow_up, (
            f"follow-up response did not incorporate the tool result: {follow_up!r}"
        )
