import uuid
from typing import Final, cast

from integration._support.responses_replay import (
    ANSWER,
    MODEL,
    WEATHER_REPLAY,
    WEATHER_TOOL,
    WEATHER_TURN,
    posted_inputs,
    responses_upstream,
)
from integration._support.wire import wire_server

import litellm
from litellm.types.llms.openai import ChatCompletionToolMessage, ChatCompletionToolParam
from litellm.types.utils import ChatCompletionMessageToolCall


def _rain(call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
    return {"role": "tool", "tool_call_id": call.id, "content": "Rain, 18C"}


def test_run_tool_loop_replays_responses_items_in_the_order_openai_produced() -> None:
    prompt: Final = f"Pick a city and check whether it is raining there {uuid.uuid4().hex}"
    with wire_server(responses_upstream(WEATHER_TURN)) as wire:
        content: Final = litellm.run_tool_loop(
            model=f"openai/responses/{MODEL}",
            messages=[{"role": "user", "content": prompt}],
            tools=[cast(ChatCompletionToolParam, WEATHER_TOOL)],
            execute_tool=_rain,
            api_base=wire.url,
            api_key="synthetic-openai-key",
        )
        inputs: Final = posted_inputs(wire)
    assert content == ANSWER
    assert inputs[1] == (("message", "user", None, prompt), *WEATHER_REPLAY), inputs
