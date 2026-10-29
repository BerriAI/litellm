import asyncio
import base64
import json
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Final, cast

import pytest
from websockets.asyncio.client import ClientConnection

from cookbook import gpt_realtime_translate as translate


@pytest.mark.asyncio
async def test_short_upload_waits_for_first_translated_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(translate, "OUTPUT_IDLE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(translate, "INITIAL_RESPONSE_TIMEOUT_SECONDS", 0.1)
    audio: Final = bytes(480)
    events: Final = iter(
        (
            {"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode()},
            {"type": "error", "error": {"message": "session closed"}},
        )
    )

    async def recv() -> str:
        event: Final = next(events)
        if event["type"] == "session.output_audio.delta":
            await asyncio.sleep(0.03)
        return json.dumps(event)

    sender_finished: Final = asyncio.Event()
    sender_finished.set()
    output: Final = tmp_path / "translation.wav"

    result: Final = await translate.receive_translation(
        cast(ClientConnection, SimpleNamespace(recv=recv)), output, sender_finished
    )

    assert result == 'Realtime API error: {"message": "session closed"}'
    with wave.open(str(output), "rb") as rendered:
        assert rendered.readframes(240) == audio


@pytest.mark.asyncio
async def test_long_upload_keeps_waiting_for_first_translated_audio_while_still_sending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(translate, "OUTPUT_IDLE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(translate, "INITIAL_RESPONSE_TIMEOUT_SECONDS", 0.05)
    audio: Final = bytes(480)
    pending: Final[asyncio.Queue[str]] = asyncio.Queue()
    sender_finished: Final = asyncio.Event()

    async def speak_after_a_long_pause() -> None:
        await asyncio.sleep(0.2)
        await pending.put(json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode()}))
        await asyncio.sleep(0.02)
        sender_finished.set()

    output: Final = tmp_path / "translation.wav"
    speaker: Final = asyncio.create_task(speak_after_a_long_pause())

    result: Final = await translate.receive_translation(
        cast(ClientConnection, SimpleNamespace(recv=pending.get)), output, sender_finished
    )
    await speaker

    assert result is None
    with wave.open(str(output), "rb") as rendered:
        assert rendered.readframes(240) == audio


@pytest.mark.asyncio
async def test_translation_finishes_once_output_idles_after_the_sender_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(translate, "OUTPUT_IDLE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(translate, "INITIAL_RESPONSE_TIMEOUT_SECONDS", 5.0)
    audio: Final = bytes(480)
    events: Final = iter(
        (
            {"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode()},
            {"type": "error", "error": {"type": "server_error", "message": "idle session closed"}},
        )
    )

    async def recv() -> str:
        event: Final = next(events)
        if event["type"] == "error":
            await asyncio.sleep(1.0)
        return json.dumps(event)

    sender_finished: Final = asyncio.Event()

    async def finish_sending() -> None:
        await asyncio.sleep(0.01)
        sender_finished.set()

    output: Final = tmp_path / "translation.wav"
    finisher: Final = asyncio.create_task(finish_sending())

    result: Final = await translate.receive_translation(
        cast(ClientConnection, SimpleNamespace(recv=recv)), output, sender_finished
    )
    await finisher

    assert result is None
    with wave.open(str(output), "rb") as rendered:
        assert rendered.readframes(240) == audio
