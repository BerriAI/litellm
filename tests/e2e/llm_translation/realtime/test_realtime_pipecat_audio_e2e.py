"""Pipecat audio + server-VAD smoke tests for the proxy realtime websocket.

Exercises the bot.py LiteLLMRealtimeLLMService pattern (simplified):
  - custom _connect (no keepalive pings)
  - _create_response override (tools session.update sent before history)
  - _handle_evt_session_created (immediate session ready without waiting for
    session.updated echo)

Three test scenarios per provider:
  test_pipecat_server_vad        – session configured with server-VAD settings;
                                   bot receives a text prompt and produces a reply.
  test_pipecat_audio_output      – same pipeline, asserts at least one
                                   TTSAudioRawFrame with non-empty audio bytes.
  test_pipecat_server_vad_audio_input – streams a real PCM16 audio fixture through
                                   the pipeline without LLMRunFrame; server VAD
                                   detects end-of-speech and auto-creates a response.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportUntypedBaseClass=false, reportUnknownParameterType=false
# pyright: reportMissingParameterType=false

import asyncio
import wave
from pathlib import Path

import pytest

from realtime_client import (
    PROVIDERS,
    RealtimeProvider,
    _ws_base_url,
    realtime_model,
)

pytestmark = pytest.mark.e2e

pytest.importorskip("pipecat", reason="pipecat-ai not installed")

from pipecat.adapters.schemas.function_schema import FunctionSchema  # noqa: E402
from pipecat.adapters.schemas.tools_schema import ToolsSchema  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    Frame,
    InputAudioRawFrame,
    LLMRunFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)  # noqa: E402
from pipecat.services.llm_service import FunctionCallParams  # noqa: E402
from pipecat.services.openai.realtime import events as rt_events  # noqa: E402
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService  # noqa: E402

from pipecat_service import LiteLLMRealtimeLLMService  # noqa: E402

PROVIDER_PARAMS = [pytest.param(p, id=p.id) for p in PROVIDERS]

# PCM16 24 kHz mono WAV of "What is the weather in Paris?" (generated via macOS
# `say` and resampled with audioop). Used by the server-VAD audio-input test.
_FIXTURES_DIR = Path(__file__).parent / "fixtures"
WEATHER_WAV = _FIXTURES_DIR / "weather_question_24k.wav"

# How long to stream silence after the speech ends so server VAD has time to
# detect the end-of-turn and fire a response.
_VAD_TAIL_SILENCE_MS = 1500

WEATHER_TOOL = ToolsSchema(
    standard_tools=[
        FunctionSchema(
            name="get_weather",
            description="Get current temperature in Fahrenheit for a city.",
            properties={"city": {"type": "string", "description": "City name."}},
            required=["city"],
        )
    ]
)

# Server-VAD session properties matching bot.py defaults.
SERVER_VAD_SETTINGS = rt_events.SessionProperties(
    output_modalities=["audio"],
    audio=rt_events.AudioConfiguration(
        input=rt_events.AudioInput(
            noise_reduction=rt_events.InputAudioNoiseReduction(type="near_field"),
            turn_detection=rt_events.TurnDetection(
                type="server_vad",
                threshold=0.8,
                prefix_padding_ms=300,
                silence_duration_ms=700,
            ),
        )
    ),
)


# ---------------------------------------------------------------------------
# Helper frame-capture processor
# ---------------------------------------------------------------------------


class _CaptureFrames(FrameProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []
        self.audio_bytes: int = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSTextFrame):
            self.texts.append(frame.text)
        elif isinstance(frame, TTSAudioRawFrame):
            self.audio_bytes += len(frame.audio)
        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Shared pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(
    key: str,
    model: str,
    *,
    prompt: str = "What is the weather in Paris?",
    timeout: float = 45.0,
) -> tuple[bool, bool, int]:
    """Run a minimal pipecat pipeline and return (tool_called, got_text, audio_bytes)."""
    tool_called = asyncio.Event()

    async def get_weather(params: FunctionCallParams) -> None:
        tool_called.set()
        city = (params.arguments or {}).get("city", "Paris")
        await params.result_callback({"city": city, "temperature_f": 72})

    llm = LiteLLMRealtimeLLMService(
        api_key=key,
        base_url=f"{_ws_base_url()}/v1/realtime",
        settings=OpenAIRealtimeLLMService.Settings(
            model=model,
            system_instruction=(
                "You are a helpful assistant. "
                "When asked about the weather, always call the get_weather tool. "
                "Never guess temperatures."
            ),
            session_properties=SERVER_VAD_SETTINGS,
        ),
    )
    llm.register_function("get_weather", get_weather)

    context = LLMContext(tools=WEATHER_TOOL)
    aggregator = LLMContextAggregatorPair(context)
    capture = _CaptureFrames()
    task = PipelineTask(
        Pipeline([aggregator.user(), llm, capture, aggregator.assistant()])
    )

    await task.queue_frames(
        [
            TranscriptionFrame(prompt, user_id="e2e", timestamp=""),
            LLMRunFrame(),
        ]
    )
    try:
        await asyncio.wait_for(PipelineRunner().run(task), timeout=timeout)
    except asyncio.TimeoutError:
        await task.queue_frame(EndFrame())

    return tool_called.is_set(), bool(capture.texts), capture.audio_bytes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_pipecat_server_vad(
    scoped_key: str,
    realtime_models: dict[str, str],
    provider: RealtimeProvider,
) -> None:
    """Session is configured with server-VAD; bot must respond to a text prompt."""
    model = realtime_model(provider, realtime_models)

    tool_called, got_text, _ = asyncio.run(_run_pipeline(scoped_key, model))

    assert tool_called, "get_weather tool was not invoked"
    assert got_text, "no assistant text frames produced"


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_pipecat_audio_output(
    scoped_key: str,
    realtime_models: dict[str, str],
    provider: RealtimeProvider,
) -> None:
    """Bot must produce at least one non-empty TTS audio frame."""
    model = realtime_model(provider, realtime_models)

    _, got_text, audio_bytes = asyncio.run(
        _run_pipeline(
            scoped_key,
            model,
            prompt="Say hello in one short sentence.",
            timeout=30.0,
        )
    )

    assert got_text, "no assistant text frames produced"
    assert audio_bytes > 0, "no TTS audio bytes received"


# ---------------------------------------------------------------------------
# Audio-input pipeline: streams a WAV fixture, server VAD fires the response
# ---------------------------------------------------------------------------


def _load_wav_chunks(path: Path, chunk_ms: int = 20) -> tuple[list[bytes], int]:
    """Read a PCM16 mono WAV and split into ``chunk_ms``-sized byte chunks."""
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1, "fixture must be mono"
        assert wf.getsampwidth() == 2, "fixture must be 16-bit PCM"
        sample_rate = wf.getframerate()
        frames_per_chunk = sample_rate * chunk_ms // 1000
        chunks = []
        while True:
            data = wf.readframes(frames_per_chunk)
            if not data:
                break
            chunks.append(data)
        return chunks, sample_rate


async def _run_audio_input_pipeline(
    key: str,
    model: str,
    *,
    timeout: float = 60.0,
) -> tuple[bool, int]:
    """Stream a WAV fixture as InputAudioRawFrame; return (got_text, audio_bytes).

    No LLMRunFrame is sent — server VAD is expected to detect end-of-speech
    and auto-trigger a response.

    Audio is streamed at real-time pace (20 ms per chunk) after the session is
    ready.  Pre-queuing all frames at once floods the VAD buffer before the
    backend is even connected and prevents speech_stopped from firing.
    """
    chunks, sample_rate = _load_wav_chunks(WEATHER_WAV)
    chunk_duration_s = 0.020  # 20 ms per chunk

    # Append silence after speech so VAD has enough quiet to fire.
    silence_frames = sample_rate * _VAD_TAIL_SILENCE_MS // 1000
    silence_chunk = b"\x00" * silence_frames * 2  # 16-bit zero samples
    chunks.append(silence_chunk)

    llm = LiteLLMRealtimeLLMService(
        api_key=key,
        base_url=f"{_ws_base_url()}/v1/realtime",
        settings=OpenAIRealtimeLLMService.Settings(
            model=model,
            system_instruction=(
                "You are a helpful assistant. "
                "When asked about the weather, always call the get_weather tool. "
                "Never guess temperatures."
            ),
            session_properties=SERVER_VAD_SETTINGS,
        ),
    )

    tool_called = asyncio.Event()

    async def get_weather(params: FunctionCallParams) -> None:
        tool_called.set()
        city = (params.arguments or {}).get("city", "Paris")
        await params.result_callback({"city": city, "temperature_f": 72})

    llm.register_function("get_weather", get_weather)

    context = LLMContext(tools=WEATHER_TOOL)
    aggregator = LLMContextAggregatorPair(context)
    capture = _CaptureFrames()
    task = PipelineTask(
        Pipeline([aggregator.user(), llm, capture, aggregator.assistant()])
    )

    async def _stream_audio() -> None:
        # Wait for the LLM session to be ready before streaming so audio
        # doesn't arrive before the backend WebSocket is connected.
        for _ in range(100):
            if getattr(llm, "_api_session_ready", False):
                break
            await asyncio.sleep(0.1)

        for chunk in chunks:
            await task.queue_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=1)
            )
            await asyncio.sleep(chunk_duration_s)

    async def _run() -> None:
        await asyncio.gather(
            PipelineRunner().run(task),
            _stream_audio(),
        )

    try:
        await asyncio.wait_for(_run(), timeout=timeout)
    except asyncio.TimeoutError:
        await task.queue_frame(EndFrame())

    return bool(capture.texts), capture.audio_bytes


@pytest.mark.parametrize("provider", PROVIDER_PARAMS)
def test_pipecat_server_vad_audio_input(
    scoped_key: str,
    realtime_models: dict[str, str],
    provider: RealtimeProvider,
) -> None:
    """Stream a real PCM16 WAV fixture; server VAD must detect speech end and respond.

    This exercises the full audio path: InputAudioRawFrame → input_audio_buffer.append
    → server-VAD turn detection → response.create (auto) → assistant reply.
    No LLMRunFrame is sent — the response must be triggered entirely by VAD.
    """
    assert WEATHER_WAV.exists(), f"audio fixture not found: {WEATHER_WAV}"
    model = realtime_model(provider, realtime_models)

    got_text, audio_bytes = asyncio.run(
        _run_audio_input_pipeline(scoped_key, model)
    )

    assert got_text, "server VAD did not trigger a response (no assistant text)"
    assert audio_bytes > 0, "no TTS audio bytes received"
