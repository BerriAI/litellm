#!/usr/bin/env python3

import argparse
import asyncio
import base64
import json
import os
import sys
import wave
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit

import websockets
from websockets.asyncio.client import ClientConnection


SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2
CHUNK_DURATION_SECONDS = 0.1
CHUNK_BYTES = int(SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH * CHUNK_DURATION_SECONDS)
OUTPUT_IDLE_TIMEOUT_SECONDS = 3.0
INITIAL_RESPONSE_TIMEOUT_SECONDS = 30.0
AUDIO_EVENT_TYPES = frozenset(
    {
        "session.output_audio.delta",
        "response.audio.delta",
        "response.output_audio.delta",
    }
)
TRANSCRIPT_EVENT_TYPES = frozenset(
    {
        "session.output_transcript.delta",
        "response.text.delta",
        "response.output_audio_transcript.delta",
    }
)


@dataclass(frozen=True, slots=True)
class Settings:
    input_wav: Path
    output_wav: Path
    base_url: str
    model: str
    target_language: str
    trailing_silence_seconds: float
    api_key: str


def write_stdout(message: str = "", *, end: str = "\n", flush: bool = False) -> None:
    sys.stdout.write(f"{message}{end}")
    if flush:
        sys.stdout.flush()


def write_stderr(message: str) -> None:
    sys.stderr.write(f"{message}\n")


def parse_args(argv: Sequence[str] | None = None) -> Settings | str:
    parser = argparse.ArgumentParser(
        description="Stream a 24 kHz PCM16 WAV through gpt-realtime-translate and save the translated audio",
    )
    parser.add_argument("input_wav", type=Path)
    parser.add_argument("--output", type=Path, default=Path("translated.wav"))
    parser.add_argument("--base-url", default=os.getenv("LITELLM_BASE_URL", "http://localhost:4000"))
    parser.add_argument("--model", default=os.getenv("REALTIME_TRANSLATE_MODEL", "gpt-realtime-translate"))
    parser.add_argument("--target-language", default="fr")
    parser.add_argument("--trailing-silence", type=float, default=1.5)
    parsed = parser.parse_args(argv)
    api_key = os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Set LITELLM_API_KEY or OPENAI_API_KEY before running the script"
    if parsed.trailing_silence < 0:
        return "--trailing-silence must be zero or greater"
    return Settings(
        input_wav=parsed.input_wav,
        output_wav=parsed.output,
        base_url=parsed.base_url,
        model=parsed.model,
        target_language=parsed.target_language,
        trailing_silence_seconds=parsed.trailing_silence,
        api_key=api_key,
    )


def translation_url(base_url: str, model: str) -> str | None:
    parsed = urlsplit(base_url.rstrip("/"))
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parsed.scheme)
    if not scheme or not parsed.netloc:
        return None
    base_path = parsed.path.rstrip("/")
    realtime_path = (
        f"{base_path}/realtime/translations" if base_path.endswith("/v1") else f"{base_path}/v1/realtime/translations"
    )
    return urlunsplit((scheme, parsed.netloc, realtime_path, urlencode({"model": model}), ""))


def read_pcm16_wav(path: Path) -> bytes | str:
    try:
        with wave.open(str(path), "rb") as source:
            actual_format = (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
                source.getcomptype(),
            )
            expected_format = (CHANNELS, SAMPLE_WIDTH, SAMPLE_RATE, "NONE")
            if actual_format != expected_format:
                return (
                    f"{path} must be mono, 16-bit PCM, 24 kHz WAV; received "
                    f"channels={actual_format[0]}, sample_width={actual_format[1]}, "
                    f"sample_rate={actual_format[2]}, compression={actual_format[3]}"
                )
            return source.readframes(source.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        return f"Unable to read {path}: {exc}"


def audio_chunks(audio: bytes) -> Iterator[bytes]:
    return (audio[offset : offset + CHUNK_BYTES] for offset in range(0, len(audio), CHUNK_BYTES))


def audio_message(audio: bytes) -> str:
    return json.dumps(
        {
            "type": "session.input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode("ascii"),
        }
    )


async def configure_session(connection: ClientConnection, target_language: str) -> str | None:
    await connection.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {"audio": {"output": {"language": target_language}}},
            }
        )
    )
    while True:
        raw_event = await asyncio.wait_for(connection.recv(), timeout=20)
        event = json.loads(raw_event)
        event_type = event.get("type")
        if event_type == "session.created":
            write_stdout(f"Session: {event.get('session', {}).get('id', 'created')}")
        if event_type == "session.updated":
            return None
        if event_type == "error":
            return f"Realtime API error: {json.dumps(event.get('error', event), ensure_ascii=False)}"


async def send_audio(
    connection: ClientConnection, pcm: bytes, trailing_silence_seconds: float, finished: asyncio.Event
) -> None:
    silence = bytes(round(trailing_silence_seconds * SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH))
    try:
        for chunk in audio_chunks(pcm + silence):
            await connection.send(audio_message(chunk))
            await asyncio.sleep(len(chunk) / (SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH))
    finally:
        finished.set()


async def receive_translation(
    connection: ClientConnection, output_path: Path, sender_finished: asyncio.Event
) -> str | None:
    audio_received = asyncio.Event()
    try:
        with wave.open(str(output_path), "wb") as output:
            output.setnchannels(CHANNELS)
            output.setsampwidth(SAMPLE_WIDTH)
            output.setframerate(SAMPLE_RATE)
            write_stdout("Translation: ", end="", flush=True)
            while True:
                timeout = OUTPUT_IDLE_TIMEOUT_SECONDS if sender_finished.is_set() else INITIAL_RESPONSE_TIMEOUT_SECONDS
                try:
                    raw_event = await asyncio.wait_for(connection.recv(), timeout=timeout)
                except TimeoutError:
                    if sender_finished.is_set() and audio_received.is_set():
                        write_stdout()
                        return None
                    return "The translation stream ended without translated audio"
                event = json.loads(raw_event)
                event_type = event.get("type")
                if event_type in AUDIO_EVENT_TYPES:
                    output.writeframes(base64.b64decode(event.get("delta", ""), validate=True))
                    audio_received.set()
                elif event_type in TRANSCRIPT_EVENT_TYPES:
                    write_stdout(event.get("delta", event.get("text", "")), end="", flush=True)
                elif event_type == "error":
                    return f"Realtime API error: {json.dumps(event.get('error', event), ensure_ascii=False)}"
    except (OSError, wave.Error) as exc:
        return f"Unable to write {output_path}: {exc}"


async def translate(settings: Settings, pcm: bytes) -> str | None:
    url = translation_url(settings.base_url, settings.model)
    if not url:
        return f"Invalid --base-url: {settings.base_url}"
    sender_finished = asyncio.Event()
    try:
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {settings.api_key}"},
            proxy=None,
            open_timeout=20,
            close_timeout=5,
        ) as connection:
            configuration_error = await configure_session(connection, settings.target_language)
            if configuration_error:
                return configuration_error
            async with asyncio.TaskGroup() as tasks:
                receiver = tasks.create_task(receive_translation(connection, settings.output_wav, sender_finished))
                tasks.create_task(send_audio(connection, pcm, settings.trailing_silence_seconds, sender_finished))
            return receiver.result()
    except Exception as exc:
        return f"Translation failed: {type(exc).__name__}: {exc}"


def main(argv: Sequence[str] | None = None) -> int:
    settings = parse_args(argv)
    if isinstance(settings, str):
        write_stderr(settings)
        return 2
    pcm = read_pcm16_wav(settings.input_wav)
    if isinstance(pcm, str):
        write_stderr(pcm)
        return 2
    error = asyncio.run(translate(settings, pcm))
    if error:
        write_stderr(error)
        return 1
    write_stdout(f"Translated audio: {settings.output_wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
