import asyncio
import base64
import json
import threading
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import pytest
import websockets
from integration._support.client import eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.ports import allocate_port
from integration._support.process import OwnedProxy, owned_proxy_process
from pydantic import JsonValue, TypeAdapter
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

pytestmark: Final = [pytest.mark.timeout(180), pytest.mark.shares_worker_state]

PROVIDER_KEY: Final = "synthetic-elevenlabs-key"
FLASH: Final = "eleven_flash_v2_5"
DEFAULT_MODEL: Final = "eleven_multilingual_v2"
AUDIO: Final = base64.b64encode(bytes(range(48))).decode()
VOICE_SETTINGS: Final = {"stability": 0.5, "similarity_boost": 0.8}
SENTENCES: Final = ("Hello from the gateway. ", "Hola, ¿qué tal?  ")
SPEND_SQL: Final = 'SELECT spend, model, custom_llm_provider FROM "LiteLLM_SpendLogs" WHERE api_key = %s'
COST_MAP_PATH: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
COST_MAP: Final = TypeAdapter(dict[str, dict[str, object]])


@dataclass(frozen=True, slots=True)
class Upgrade:
    target: str
    provider_key: str | None
    frames: tuple[dict[str, JsonValue], ...]


@dataclass(frozen=True, slots=True)
class Peer:
    url: str
    upgrades: SimpleQueue[Upgrade]


@dataclass(frozen=True, slots=True)
class Session:
    frames: tuple[dict[str, JsonValue], ...]
    close_code: int | None
    close_reason: str


def _per_character(model_id: str) -> float:
    row: Final = COST_MAP.validate_json(COST_MAP_PATH.read_bytes())[f"elevenlabs/{model_id}"]
    rate: Final = row["input_cost_per_character"]
    assert isinstance(rate, float) and rate > 0, rate
    return rate


async def _client_frames(connection: ServerConnection) -> AsyncIterator[dict[str, JsonValue]]:
    async for raw in connection:
        frame: Final = JSON_OBJECT.validate_json(raw)
        yield frame
        text: Final = frame.get("text")
        if text == "":
            await connection.send(json.dumps({"isFinal": True}))
            return
        if isinstance(text, str) and text.strip():
            await connection.send(
                json.dumps(
                    {
                        "audio": AUDIO,
                        "alignment": {"chars": list(text), "charStartTimesMs": [0], "charDurationsMs": [40]},
                        "isFinal": None,
                    }
                )
            )


async def _speak(connection: ServerConnection, upgrades: SimpleQueue[Upgrade]) -> None:
    request: Final = connection.request
    assert request is not None
    frames: Final = tuple([frame async for frame in _client_frames(connection)])
    upgrades.put(Upgrade(request.path, request.headers.get("xi-api-key"), frames))


async def _serve(port: int, upgrades: SimpleQueue[Upgrade], ready: SimpleQueue[bool], stop: asyncio.Event) -> None:
    async with serve(lambda connection: _speak(connection, upgrades), "127.0.0.1", port):
        ready.put(True)
        await stop.wait()


@contextmanager
def elevenlabs_peer() -> Iterator[Peer]:
    port: Final = allocate_port()
    loop: Final = asyncio.new_event_loop()
    stop: Final = asyncio.Event()
    upgrades: Final = SimpleQueue[Upgrade]()
    ready: Final = SimpleQueue[bool]()
    serving: Final = _serve(port, upgrades, ready, stop)
    thread: Final = threading.Thread(target=loop.run_until_complete, args=(serving,), daemon=True)
    thread.start()
    try:
        assert ready.get(timeout=10)
        yield Peer(f"http://127.0.0.1:{port}", upgrades)
    finally:
        loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=10)
        loop.close()


@pytest.fixture(scope="module")
def elevenlabs() -> Iterator[Peer]:
    with elevenlabs_peer() as peer:
        yield peer


@pytest.fixture(scope="module")
def owned(elevenlabs: Peer, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("elevenlabs-tts")
    overrides: Final = {"ELEVENLABS_API_KEY": PROVIDER_KEY, "ELEVENLABS_API_BASE": elevenlabs.url}
    with gateway_from_environment() as gateway:
        with owned_proxy_process(gateway, directory, overrides) as proxy:
            yield proxy


def _ws_url(proxy: OwnedProxy, voice_id: str, query: str) -> str:
    base: Final = str(proxy.gateway.client.base_url).rstrip("/").replace("http://", "ws://")
    return f"{base}/elevenlabs/v1/text-to-speech/{voice_id}/stream-input?{query}"


async def _received(connection: websockets.ClientConnection) -> AsyncIterator[dict[str, JsonValue]]:
    async for raw in connection:
        yield JSON_OBJECT.validate_json(raw)


async def _exchange(
    connection: websockets.ClientConnection, key: str, texts: tuple[str, ...]
) -> tuple[dict[str, JsonValue], ...]:
    try:
        await connection.send(json.dumps({"text": " ", "voice_settings": VOICE_SETTINGS, "xi_api_key": key}))
        for text in texts:
            await connection.send(json.dumps({"text": text}, ensure_ascii=False))
        await connection.send(json.dumps({"text": ""}))
        return tuple([frame async for frame in _received(connection)])
    except ConnectionClosed:
        return ()


async def _stream(url: str, key: str, texts: tuple[str, ...]) -> Session:
    async with websockets.connect(url, additional_headers={"xi-api-key": key}, open_timeout=30) as connection:
        frames: Final = await _exchange(connection, key, texts)
        return Session(frames, connection.close_code, connection.close_reason or "")


def _session(url: str, key: str, texts: tuple[str, ...]) -> Session:
    return asyncio.run(asyncio.wait_for(_stream(url, key, texts), 60))


def _upgrade_for(peer: Peer, voice_id: str) -> Upgrade:
    upgrade: Final = peer.upgrades.get(timeout=30)
    assert f"/{voice_id}/" in upgrade.target, upgrade.target
    return upgrade


def _spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),)),
        lambda rows: len(rows) == 1,
        seconds=70,
    )


def test_a_session_reaches_elevenlabs_with_the_proxy_key_and_is_billed_per_character(
    owned: OwnedProxy, elevenlabs: Peer
) -> None:
    voice_id: Final = f"voice{uuid.uuid4().hex[:12]}"
    with owned.gateway.scenario() as scenario:
        key: Final = scenario.key()
        query: Final = f"model_id={FLASH}&output_format=pcm_16000"
        session: Final = _session(_ws_url(owned, voice_id, query), key, SENTENCES)

        assert [frame.get("audio") for frame in session.frames] == [AUDIO, AUDIO, None]
        assert session.frames[-1] == {"isFinal": True}
        assert session.close_code == 1000
        upgrade: Final = _upgrade_for(elevenlabs, voice_id)
        assert upgrade.target == f"/v1/text-to-speech/{voice_id}/stream-input?{query}"
        assert upgrade.provider_key == PROVIDER_KEY
        assert upgrade.frames == (
            {"text": " ", "voice_settings": VOICE_SETTINGS},
            *({"text": text} for text in SENTENCES),
            {"text": ""},
        )
        rows: Final = _spend_rows(key)
        characters: Final = sum(len(text.rstrip()) for text in SENTENCES)
        assert float(rows[0]["spend"]) == pytest.approx(characters * _per_character(FLASH))
        assert rows[0]["model"] == FLASH
        assert rows[0]["custom_llm_provider"] == "elevenlabs"


def test_an_unpriced_model_is_refused_before_elevenlabs_is_contacted_and_the_default_model_is_billed(
    owned: OwnedProxy, elevenlabs: Peer
) -> None:
    refused_voice: Final = f"voice{uuid.uuid4().hex[:12]}"
    default_voice: Final = f"voice{uuid.uuid4().hex[:12]}"
    with owned.gateway.scenario() as scenario:
        key: Final = scenario.key()
        refused: Final = _session(_ws_url(owned, refused_voice, "model_id=eleven_not_in_the_cost_map"), key, SENTENCES)
        default: Final = _session(_ws_url(owned, default_voice, "output_format=pcm_16000"), key, SENTENCES[:1])

        assert refused.frames == ()
        assert refused.close_code == 1008
        assert "elevenlabs/eleven_not_in_the_cost_map" in refused.close_reason
        assert default.frames[-1] == {"isFinal": True}
        upgrade: Final = _upgrade_for(elevenlabs, default_voice)
        assert upgrade.target == (
            f"/v1/text-to-speech/{default_voice}/stream-input?output_format=pcm_16000&model_id={DEFAULT_MODEL}"
        )
        assert elevenlabs.upgrades.empty()
        rows: Final = _spend_rows(key)
        assert float(rows[0]["spend"]) == pytest.approx(len(SENTENCES[0].rstrip()) * _per_character(DEFAULT_MODEL))
        assert rows[0]["model"] == DEFAULT_MODEL
