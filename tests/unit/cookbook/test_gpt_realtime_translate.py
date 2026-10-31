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
    monkeypatch.setattr(translate, "RESPONSE_TIMEOUT_SECONDS", 0.1)
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
    monkeypatch.setattr(translate, "RESPONSE_TIMEOUT_SECONDS", 0.05)
    audio: Final = bytes(480)
    pending: Final[asyncio.Queue[str]] = asyncio.Queue()
    sender_finished: Final = asyncio.Event()

    async def speak_after_a_long_pause() -> None:
        await asyncio.sleep(0.2)
        await pending.put(json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode()}))
        await asyncio.sleep(0.02)
        sender_finished.set()
        await pending.put(json.dumps({"type": "session.closed"}))

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
async def test_a_quiet_gap_after_the_sender_stops_keeps_the_rest_of_the_translation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(translate, "RESPONSE_TIMEOUT_SECONDS", 1.0)
    first: Final = bytes(480)
    second: Final = bytes(range(240)) * 2
    events: Final = iter(
        (
            {"type": "session.output_audio.delta", "delta": base64.b64encode(first).decode()},
            {"type": "session.output_audio.delta", "delta": base64.b64encode(second).decode()},
            {"type": "session.closed"},
        )
    )

    async def recv() -> str:
        event: Final = next(events)
        if event.get("delta") == base64.b64encode(second).decode():
            await asyncio.sleep(0.3)
        return json.dumps(event)

    sender_finished: Final = asyncio.Event()
    sender_finished.set()
    output: Final = tmp_path / "translation.wav"

    result: Final = await translate.receive_translation(
        cast(ClientConnection, SimpleNamespace(recv=recv)), output, sender_finished
    )

    assert result is None
    with wave.open(str(output), "rb") as rendered:
        assert rendered.readframes(480) == first + second


@pytest.mark.asyncio
async def test_a_stream_that_never_closes_the_session_reports_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(translate, "RESPONSE_TIMEOUT_SECONDS", 0.05)
    audio: Final = bytes(480)
    pending: Final[asyncio.Queue[str]] = asyncio.Queue()
    await pending.put(json.dumps({"type": "session.output_audio.delta", "delta": base64.b64encode(audio).decode()}))
    sender_finished: Final = asyncio.Event()
    sender_finished.set()

    result: Final = await translate.receive_translation(
        cast(ClientConnection, SimpleNamespace(recv=pending.get)), tmp_path / "translation.wav", sender_finished
    )

    assert result == "The translation stream went quiet for 0.05 s without closing the session"


@pytest.mark.asyncio
async def test_the_sender_closes_the_session_after_the_last_audio_chunk() -> None:
    sent: Final[asyncio.Queue[str]] = asyncio.Queue()
    finished: Final = asyncio.Event()
    pcm: Final = bytes(translate.CHUNK_BYTES + 2)

    await translate.send_audio(cast(ClientConnection, SimpleNamespace(send=sent.put)), pcm, 0.0, finished)

    messages: Final = tuple(json.loads(sent.get_nowait()) for _ in range(sent.qsize()))
    assert [message["type"] for message in messages] == [
        "session.input_audio_buffer.append",
        "session.input_audio_buffer.append",
        "session.close",
    ]
    assert b"".join(base64.b64decode(message["audio"]) for message in messages[:-1]) == pcm
    assert finished.is_set()
