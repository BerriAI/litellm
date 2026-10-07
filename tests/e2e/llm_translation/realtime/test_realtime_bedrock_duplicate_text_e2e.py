"""Live e2e: Bedrock Nova 2 Sonic realtime shows each assistant sentence once.

User flow: a voice client on /v1/realtime streams mic audio
through input_audio_buffer.append, keeps streaming after the user stops talking,
and renders every response.text.done as a chat bubble. Nova 2 Sonic sends each
assistant text block twice, a SPECULATIVE preview next to the audio and then the
FINAL transcript once the audio turn has ended, tagging each block with
generationStage in its contentStart. The test streams a spoken question the same
way, reads the whole turn including events after the first response.done, and
asserts no assistant sentence reaches the client in more than one
response.text.done.

NOVA_SONIC pins a vendor-owned model id; see test_realtime_bedrock_e2e.py for
how to re-check it before concluding litellm broke.
"""

from __future__ import annotations

import array
import base64
import wave
from collections import Counter
from pathlib import Path
from typing import Final

import pytest

from e2e_config import unique_marker
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from realtime_client import (
    InputAudioBufferAppend,
    RealtimeClient,
    SessionConfig,
    SessionUpdate,
    TextDone,
    events_of_type,
)

pytestmark = pytest.mark.e2e

NOVA_SONIC = "bedrock/amazon.nova-2-sonic-v1:0"
QUESTION_WAV = Path(__file__).parent / "fixtures" / "weather_question_24k.wav"
NOVA_INPUT_RATE = 16000
CHUNK_SECONDS = 0.1


def pcm16_at_nova_input_rate(path: Path) -> bytes:
    """Mono PCM16 samples from `path`, linearly resampled to Nova Sonic's 16 kHz input rate."""
    with wave.open(str(path), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth()) == (1, 2), "fixture must be mono PCM16"
        ratio: Final = wav.getframerate() / NOVA_INPUT_RATE
        source: Final = array.array("h", wav.readframes(wav.getnframes()))
    positions: Final = (i * ratio for i in range(int(len(source) / ratio) - 1))
    return array.array("h", (_interpolated_sample(source, position) for position in positions)).tobytes()


def _interpolated_sample(source: array.array[int], position: float) -> int:
    base: Final = int(position)
    fraction: Final = position - base
    return round(source[base] * (1 - fraction) + source[base + 1] * fraction)


def mic_chunks(pcm: bytes) -> list[InputAudioBufferAppend]:
    size = int(NOVA_INPUT_RATE * CHUNK_SECONDS) * 2
    return [
        InputAudioBufferAppend(audio=base64.b64encode(pcm[i : i + size]).decode()) for i in range(0, len(pcm), size)
    ]


class TestNovaSonicAssistantText:
    @pytest.mark.covers(
        "llm.realtime.bedrock_converse.basic.stream.works",
        exercised_on=["realtime"],
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.REALTIME,
            providers=(Provider.BEDROCK,),
            models=(NOVA_SONIC,),
            capabilities=(Capability.AUDIO_INPUT, Capability.AUDIO_OUTPUT),
            mode=Mode.WEBSOCKET,
        )
    )
    def test_nova_sonic_voice_turn_shows_each_sentence_once(
        self, client: RealtimeClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model: Final = f"e2e-nova-sonic-dupe-{unique_marker()}"
        model_id: Final = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=NOVA_SONIC,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
            mode="realtime",
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        silence: Final = InputAudioBufferAppend(
            audio=base64.b64encode(bytes(int(NOVA_INPUT_RATE * CHUNK_SECONDS) * 2)).decode()
        )

        with client.connect(key=scoped_key, model=model) as session:
            session.collect_until("session.created", timeout=30)
            session.send(
                SessionUpdate(
                    session=SessionConfig(
                        modalities=["text", "audio"],
                        instructions="You are a friendly assistant. Answer in two or three short sentences.",
                    )
                )
            )
            session.collect_until("session.updated", timeout=30)
            events: Final = session.stream_and_collect(
                mic_chunks(pcm16_at_nova_input_rate(QUESTION_WAV)),
                tail=silence,
                interval=CHUNK_SECONDS,
                idle=15,
                timeout=90,
            )

        types: Final = [e.type for e in events]
        assert "response.done" in types, f"Nova Sonic never finished a response; types={types}"
        spoken: Final = [
            text
            for e in events_of_type(events, "response.text.done")
            if (text := TextDone.model_validate_json(e.payload).text.strip())
        ]
        assert spoken, f"Nova Sonic produced no assistant text; types={types}"
        repeated: Final = {t: n for t, n in Counter(spoken).items() if n > 1}
        assert not repeated, f"assistant text delivered more than once: {repeated}; all text.done={spoken}"
