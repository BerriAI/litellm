"""Live e2e for the proxy realtime websocket (/v1/realtime).

Each test opens a websocket through the proxy, speaks the OpenAI GA realtime
event schema, and asserts the proxy normalizes the provider's stream into that
schema: the session lifecycle, the canonical response event sequence with a
reconstructed transcript and usage, and a full tool-call round-trip (call ->
tool result -> a follow-up response that uses the result).

One GA-speaking client validates every provider; only the model alias changes.
Every provider is provisioned at session start, so a missing realtime alias is a
hard failure, not a skip; once configured, a protocol failure is likewise a hard
failure. See REALTIME_COVERAGE_MATRIX.md.
"""

from typing import Final

import pytest
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
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
    RealtimeSession,
    ReceivedEvent,
    ResponseCreate,
    ResponseDone,
    ServerEnvelope,
    SessionConfig,
    SessionUpdate,
    as_text,
    parse_last,
    realtime_model,
    transcript,
    user_message,
)
from websockets.exceptions import ConnectionClosedError

pytestmark = pytest.mark.e2e

AZURE_REALTIME_MODEL: Final = "azure/gpt-realtime"

TEXT_PARAMS: Final = tuple(
    pytest.param(
        p,
        id=p.id,
        marks=meta(
            Subject(
                domain=Domain.LLM_TRANSLATION,
                route=Route.REALTIME,
                providers=(Provider(p.id),),
                models=(p.litellm_params.model,),
                mode=Mode.WEBSOCKET,
            )
        ),
    )
    for p in PROVIDERS
)

BEDROCK_TOOL_COVERS: Final = pytest.mark.covers(
    "llm.realtime.bedrock_converse.tool_use.stream.works", exercised_on=["realtime"]
)

TOOL_PARAMS: Final = tuple(
    pytest.param(
        p,
        id=p.id,
        marks=(
            meta(
                Subject(
                    domain=Domain.LLM_TRANSLATION,
                    route=Route.REALTIME,
                    providers=(Provider(p.id),),
                    models=(p.litellm_params.model,),
                    capabilities=(Capability.FUNCTION_CALLING,),
                    mode=Mode.WEBSOCKET,
                )
            ),
            *((BEDROCK_TOOL_COVERS,) if p.id == "bedrock" else ()),
        ),
    )
    for p in PROVIDERS
)

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


@pytest.mark.parametrize("provider", TEXT_PARAMS)
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


@pytest.mark.parametrize("provider", TOOL_PARAMS)
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
        final_answer = _tool_call_loop(session, max_rounds=4)

        assert "72" in transcript(final_answer), (
            "the answer after the tool results did not use them"
        )


def _tool_call_loop(session: RealtimeSession, *, max_rounds: int) -> tuple[ReceivedEvent, ...]:
    """Answer every function_call in each response.done until one arrives without any.

    Nova 2 Sonic emits two tool calls per question, each inside its own response.done,
    so a one-call round trip cannot wait for "the" follow-up: the second response.done
    is another call, not the answer. The loop sends one function_call_output per call
    then a single response.create, and returns the events of the first call-free round."""
    for round_index in range(max_rounds):
        events = session.collect_until("response.done", timeout=60)
        done = parse_last(events, "response.done", ResponseDone)
        assert done is not None
        calls = tuple(
            item for item in done.response.output or () if item.type == "function_call"
        )
        if not calls:
            assert round_index > 0, "model did not emit a function call"
            return events
        done_args = tuple(
            FunctionCallArgumentsDone.model_validate_json(event.payload)
            for event in events
            if event.type == "response.function_call_arguments.done"
        )
        for item in calls:
            assert item.name == "get_weather", item
            args_event = next(
                event for event in done_args if event.call_id == item.call_id
            )
            args = WeatherArgs.model_validate_json(args_event.arguments)
            tool_result = WeatherResult(city=args.city, temperature_f=72)
            session.send(
                ConversationItemCreate(
                    item=FunctionCallOutputItem(
                        call_id=args_event.call_id, output=tool_result.model_dump_json()
                    )
                )
            )
        session.send(ResponseCreate())
    raise AssertionError(f"still calling tools after {max_rounds} rounds")


_REFUSED_UPSTREAMS = (
    RealtimeProvider(
        "azure-bad-key",
        "azure-realtime-refused",
        LiteLLMParamsBody(
            model=AZURE_REALTIME_MODEL,
            api_key="invalid-e2e-key",
            api_version="2025-08-28",
            realtime_protocol="GA",
        ),
    ),
)


@pytest.mark.parametrize("provider", _REFUSED_UPSTREAMS, ids=[p.id for p in _REFUSED_UPSTREAMS])
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.REALTIME,
        providers=(Provider.AZURE,),
        models=(AZURE_REALTIME_MODEL,),
        mode=Mode.WEBSOCKET,
    )
)
def test_upstream_handshake_refusal_is_an_error_event_and_policy_close(
    client: RealtimeClient,
    resources: ResourceManager,
    scoped_key: str,
    provider: RealtimeProvider,
) -> None:
    model_name, model_id = client.provision(provider)
    resources.defer(lambda: client.proxy.delete_model(model_id))

    with client.connect(key=scoped_key, model=model_name) as session:
        first = ServerEnvelope.model_validate_json(
            as_text(session.connection.recv(timeout=15))
        )
        assert first.type == "error", first
        with pytest.raises(ConnectionClosedError) as closed:
            session.connection.recv(timeout=15)

    assert closed.value.rcvd is not None
    assert closed.value.rcvd.code == 1008, closed.value
