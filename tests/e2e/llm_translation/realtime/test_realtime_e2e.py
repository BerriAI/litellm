"""Live e2e for the proxy realtime websocket (/v1/realtime).

Each test opens a websocket through the proxy, speaks the OpenAI GA realtime
event schema, and asserts the proxy normalizes the provider's stream into that
schema: the session lifecycle, the canonical response event sequence with a
reconstructed transcript and usage, and a full tool-call round-trip (call ->
tool result -> a follow-up response that uses the result).

One GA-speaking client validates every provider; only the model alias changes. A
provider whose realtime alias is not configured on the proxy skips (skip on
environment); once it is configured, a protocol failure is a hard failure. See
REALTIME_COVERAGE_MATRIX.md.
"""

import pytest
from pydantic import BaseModel

from realtime_client import (
    PROVIDERS,
    ConversationItemCreate,
    FunctionCallArgumentsDone,
    FunctionCallOutputItem,
    FunctionTool,
    JsonSchema,
    JsonSchemaProperty,
    RealtimeClient,
    RealtimeProvider,
    ResponseCreate,
    ResponseDone,
    SessionConfig,
    SessionUpdate,
    function_call_item,
    parse_last,
    realtime_model,
    transcript,
    user_message,
)

pytestmark = pytest.mark.e2e

PROVIDER_PARAMS = [pytest.param(p, id=p.id) for p in PROVIDERS]

WEATHER_TOOL = FunctionTool(
    name="get_weather",
    description="Get the current temperature in Fahrenheit for a given city.",
    parameters=JsonSchema(
        properties={"city": JsonSchemaProperty(type="string")}, required=["city"]
    ),
)


class WeatherArgs(BaseModel):
    city: str


class WeatherResult(BaseModel):
    city: str
    temperature_f: int


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_text_conversation(
    client: RealtimeClient,
    scoped_key: str,
    realtime_models: dict[str, str],
    provider: RealtimeProvider,
) -> None:
    model = realtime_model(provider, realtime_models)

    with client.connect(key=scoped_key, model=model) as session:
        created = session.collect_until("session.created", timeout=20)
        assert created[-1].type == "session.created"

        session.send(
            SessionUpdate(
                session=SessionConfig(
                    instructions="You are a terse assistant. Reply in one short sentence."
                )
            )
        )
        session.collect_until("session.updated", timeout=20)

        session.send(user_message("Say the single word hello."))
        session.send(ResponseCreate())
        events = session.collect_until("response.done", timeout=60)

        types = {e.type for e in events}
        assert "response.created" in types
        assert "response.output_item.added" in types

        assert transcript(events).strip() != ""

        done = parse_last(events, "response.done", ResponseDone)
        assert done is not None
        assert done.response.usage is not None, "response.done missing normalized usage"


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_tool_call_round_trip(
    client: RealtimeClient,
    scoped_key: str,
    realtime_models: dict[str, str],
    provider: RealtimeProvider,
) -> None:
    model = realtime_model(provider, realtime_models)

    with client.connect(key=scoped_key, model=model) as session:
        session.collect_until("session.created", timeout=20)
        session.send(
            SessionUpdate(
                session=SessionConfig(
                    tools=[WEATHER_TOOL],
                    tool_choice="auto",
                    instructions=(
                        "Use the get_weather tool when asked about weather. "
                        "After receiving the result, state the temperature."
                    ),
                )
            )
        )
        session.collect_until("session.updated", timeout=20)

        session.send(user_message("What's the weather in Paris right now?"))
        session.send(ResponseCreate())
        first = session.collect_until("response.done", timeout=60)

        args_event = parse_last(
            first, "response.function_call_arguments.done", FunctionCallArgumentsDone
        )
        assert args_event is not None, "model did not emit a function call"
        args = WeatherArgs.model_validate_json(args_event.arguments)

        item = function_call_item(first)
        assert item is not None, "no completed function_call output item"
        assert item.name == "get_weather"
        assert item.call_id == args_event.call_id

        tool_result = WeatherResult(city=args.city, temperature_f=72)
        session.send(
            ConversationItemCreate(
                item=FunctionCallOutputItem(
                    call_id=args_event.call_id, output=tool_result.model_dump_json()
                )
            )
        )
        session.send(ResponseCreate())
        second = session.collect_until("response.done", timeout=60)

        assert "72" in transcript(second), "follow-up did not use the tool result"
