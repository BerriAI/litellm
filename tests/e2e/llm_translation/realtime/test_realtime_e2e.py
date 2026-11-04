"""Live e2e for the proxy realtime websocket (/v1/realtime).

Each test opens a websocket through the proxy, speaks the OpenAI GA realtime
event schema, and asserts the proxy normalizes the provider's stream into that
schema: the session lifecycle, the canonical response event sequence with a
reconstructed transcript and usage, and a full tool-call round-trip (call ->
tool result -> a follow-up response that uses the result). The audio tests
stream the spoken weather question into /v1/realtime/translations and into a
/v1/realtime?intent=transcription session, and assert the translated or
transcribed output and the session's spend row.

One GA-speaking client validates every provider; only the model alias changes.
Every provider is provisioned at session start, so a missing realtime alias is a
hard failure, not a skip; once configured, a protocol failure is likewise a hard
failure. See REALTIME_COVERAGE_MATRIX.md.
"""

import base64
import wave
from pathlib import Path
from types import MappingProxyType
from typing import Final

import pytest
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, SpendLogRow
from pydantic import BaseModel
from realtime_client import (
    PROVIDERS,
    TRANSLATIONS_ROUTE,
    ConversationItemCreate,
    DeltaEvent,
    FunctionCallArgumentsDone,
    FunctionCallOutputItem,
    FunctionTool,
    InputAudioBufferAppend,
    InputAudioBufferCommit,
    InputTranscription,
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
    SessionInputAudioBufferAppend,
    SessionUpdate,
    TranscriptionAudio,
    TranscriptionAudioInput,
    TranscriptionSessionConfig,
    TranscriptionSessionUpdate,
    TranslationAudio,
    TranslationOutput,
    TranslationSessionConfig,
    TranslationSessionUpdate,
    as_text,
    events_of_type,
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
                    modalities=list(provider.modalities),
                    instructions="You are a terse assistant. Reply in one short sentence.",
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
                    modalities=list(provider.modalities),
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


WEATHER_WAV: Final = Path(__file__).parent / "fixtures" / "weather_question_24k.wav"
PCM_RATE: Final = 24000
CHUNK_SECONDS: Final = 0.1
CHUNK_BYTES: Final = int(PCM_RATE * CHUNK_SECONDS) * 2
SILENCE_B64: Final = base64.b64encode(bytes(CHUNK_BYTES)).decode()
REGISTRY_ROUTES: Final = MappingProxyType({"openai": "openai", "azure": "azure_openai"})


def weather_pcm_chunks() -> tuple[str, ...]:
    with wave.open(str(WEATHER_WAV), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, PCM_RATE), (
            "fixture must be mono PCM16 at 24 kHz"
        )
        pcm: Final = wav.readframes(wav.getnframes())
    return tuple(base64.b64encode(pcm[i : i + CHUNK_BYTES]).decode() for i in range(0, len(pcm), CHUNK_BYTES))


def has_billed_row(rows: list[SpendLogRow]) -> bool:
    return any((row.spend or 0) > 0 for row in rows)


def audio_marks(provider: RealtimeProvider, capability: str) -> tuple[pytest.MarkDecorator, ...]:
    return (
        pytest.mark.covers(f"llm.realtime.{REGISTRY_ROUTES[provider.id]}.{capability}.stream.works"),
        meta(
            Subject(
                domain=Domain.LLM_TRANSLATION,
                route=Route.REALTIME,
                providers=(Provider(provider.id),),
                models=(provider.litellm_params.model,),
                capabilities=(Capability.AUDIO_INPUT,),
                mode=Mode.WEBSOCKET,
            )
        ),
    )


TRANSLATION_PROVIDERS: Final = (
    RealtimeProvider(
        "openai",
        "openai-translate",
        LiteLLMParamsBody(model="openai/gpt-realtime-translate", api_key="os.environ/OPENAI_API_KEY"),
    ),
    RealtimeProvider(
        "azure",
        "azure-translate",
        LiteLLMParamsBody(
            model="azure/gpt-realtime-translate",
            api_base="os.environ/AZURE_API_BASE",
            api_key="os.environ/AZURE_API_KEY",
        ),
    ),
)

LIVE_TRANSCRIPTION_PROVIDERS: Final = (
    RealtimeProvider(
        "openai",
        "openai-live-transcribe",
        LiteLLMParamsBody(model="openai/gpt-live-transcribe", api_key="os.environ/OPENAI_API_KEY"),
    ),
    RealtimeProvider(
        "azure",
        "azure-live-transcribe",
        LiteLLMParamsBody(
            model="azure/gpt-live-transcribe",
            api_base="os.environ/AZURE_API_BASE",
            api_key="os.environ/AZURE_API_KEY",
        ),
    ),
)

TRANSLATION_PARAMS: Final = tuple(
    pytest.param(p, id=p.id, marks=audio_marks(p, "translation")) for p in TRANSLATION_PROVIDERS
)
LIVE_TRANSCRIPTION_PARAMS: Final = tuple(
    pytest.param(p, id=p.id, marks=audio_marks(p, "transcription")) for p in LIVE_TRANSCRIPTION_PROVIDERS
)
AZURE_GA_PROVIDER: Final = next(p for p in PROVIDERS if p.id == "azure")


class TranscriptionCompleted(BaseModel):
    type: str
    transcript: str


class TestRealtimeTranslation:
    @pytest.mark.parametrize("provider", TRANSLATION_PARAMS)
    def test_spoken_english_comes_back_as_french_audio_and_transcript_and_is_billed(
        self,
        client: RealtimeClient,
        resources: ResourceManager,
        scoped_key: str,
        provider: RealtimeProvider,
    ) -> None:
        model, model_id = client.provision(provider)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        with client.connect(key=scoped_key, model=model, route=TRANSLATIONS_ROUTE) as session:
            session.send(
                TranslationSessionUpdate(
                    session=TranslationSessionConfig(audio=TranslationAudio(output=TranslationOutput(language="fr")))
                )
            )
            session.collect_until("session.updated", timeout=30)
            events: Final = session.stream_and_collect(
                tuple(SessionInputAudioBufferAppend(audio=chunk) for chunk in weather_pcm_chunks()),
                tail=SessionInputAudioBufferAppend(audio=SILENCE_B64),
                interval=CHUNK_SECONDS,
                idle=8,
                timeout=60,
            )

        types: Final = tuple(e.type for e in events)
        audio_bytes: Final = sum(
            len(base64.b64decode(DeltaEvent.model_validate_json(e.payload).delta))
            for e in events_of_type(events, "session.output_audio.delta")
        )
        assert audio_bytes > 0, f"translation returned no translated audio; types={types}"
        translated: Final = "".join(
            DeltaEvent.model_validate_json(e.payload).delta
            for e in events_of_type(events, "session.output_transcript.delta")
        ).lower()
        assert translated.strip(), f"translation returned no translated text; types={types}"
        assert "weather" not in translated, f"the spoken weather question came back untranslated: {translated!r}"

        rows: Final = client.proxy.poll_logs_for_key(scoped_key, predicate=has_billed_row)
        assert has_billed_row(rows), f"the translation session was not billed: {rows}"


class TestRealtimeLiveTranscription:
    @pytest.mark.parametrize("provider", LIVE_TRANSCRIPTION_PARAMS)
    def test_spoken_question_streams_transcript_deltas_that_match_the_completed_transcript(
        self,
        client: RealtimeClient,
        resources: ResourceManager,
        scoped_key: str,
        provider: RealtimeProvider,
    ) -> None:
        model, model_id = client.provision(provider)
        resources.defer(lambda: client.proxy.delete_model(model_id))

        with client.connect(key=scoped_key, model=model, intent="transcription") as session:
            session.send(
                TranscriptionSessionUpdate(
                    session=TranscriptionSessionConfig(
                        audio=TranscriptionAudio(
                            input=TranscriptionAudioInput(transcription=InputTranscription(model=model))
                        )
                    )
                )
            )
            session.collect_until("session.updated", timeout=30)
            for chunk in weather_pcm_chunks():
                session.send(InputAudioBufferAppend(audio=chunk))
            session.send(InputAudioBufferCommit())
            events: Final = session.collect_until("conversation.item.input_audio_transcription.completed", timeout=60)

        completed: Final = TranscriptionCompleted.model_validate_json(events[-1].payload)
        assert "weather" in completed.transcript.lower(), (
            f"transcript of a spoken weather question does not mention weather: {completed.transcript!r}"
        )
        streamed: Final = "".join(
            DeltaEvent.model_validate_json(e.payload).delta
            for e in events_of_type(events, "conversation.item.input_audio_transcription.delta")
        )
        assert streamed.strip() == completed.transcript.strip(), (
            f"streamed deltas {streamed!r} do not add up to the completed transcript {completed.transcript!r}"
        )

        rows: Final = client.proxy.poll_logs_for_key(scoped_key, predicate=has_billed_row)
        assert has_billed_row(rows), f"the live transcription session was not billed: {rows}"


@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.REALTIME,
        providers=(Provider.AZURE,),
        models=(AZURE_GA_PROVIDER.litellm_params.model,),
        mode=Mode.WEBSOCKET,
    )
)
def test_azure_ga_text_turn_is_answered_and_billed(
    client: RealtimeClient, resources: ResourceManager, scoped_key: str
) -> None:
    model, model_id = client.provision(AZURE_GA_PROVIDER)
    resources.defer(lambda: client.proxy.delete_model(model_id))

    with client.connect(key=scoped_key, model=model) as session:
        session.collect_until("session.created", timeout=20)
        session.send(SessionUpdate(session=SessionConfig(instructions="Reply with the single word hello.")))
        session.collect_until("session.updated", timeout=20)
        session.send(user_message("Say the single word hello."))
        session.send(ResponseCreate())
        events: Final = session.collect_until("response.done", timeout=60)

    assert "hello" in transcript(events).lower(), f"Azure GA realtime did not answer the turn: {transcript(events)!r}"
    rows: Final = client.proxy.poll_logs_for_key(scoped_key, predicate=has_billed_row)
    assert has_billed_row(rows), f"the Azure GA realtime turn was not billed: {rows}"
