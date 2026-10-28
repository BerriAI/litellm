from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import AsyncIterator, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import SimpleQueue
from threading import Thread
from typing import Final

import anthropic
import httpx
import openai
import websockets
import yaml
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import GEMINI_LIVE_PATH, ScenarioHandle, delete_scenario, register_scenario
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request
from integration.cost_calculation.cost_tracking_case import RealtimeResponse
from pydantic import JsonValue
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

BACKEND: Final = "gemini-3.8-flash-tts"
GEMINI_API_KEY: Final = "synthetic-gemini-key"
VERTEX_PROJECT: Final = "scripted-tts-project"
VERTEX_LOCATION: Final = "us-central1"
VERTEX_MODEL_PATH: Final = (
    f"/v1/projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/publishers/google/models/{BACKEND}"
)
GEMINI_PREBUILT_VOICES: Final = frozenset(
    {
        "Zephyr",
        "Puck",
        "Charon",
        "Kore",
        "Fenrir",
        "Leda",
        "Orus",
        "Aoede",
        "Callirrhoe",
        "Autonoe",
        "Enceladus",
        "Iapetus",
        "Umbriel",
        "Algieba",
        "Despina",
        "Erinome",
        "Algenib",
        "Rasalgethi",
        "Laomedeia",
        "Achernar",
        "Alnilam",
        "Schedar",
        "Gacrux",
        "Pulcherrima",
        "Achird",
        "Zubenelgenubi",
        "Vindemiatrix",
        "Sadachbia",
        "Sadaltager",
        "Sulafat",
    }
)
ACCEPTED_VOICES: Final = GEMINI_PREBUILT_VOICES | {"nova"}
REJECTED_PREFIX: Final = "No matching speaker voice found for name: "
NO_VOICE: Final = "<no voiceConfig>"
PCM: Final = bytes(range(256)) * 4
PCM_B64: Final = base64.b64encode(PCM).decode()
AUDIO_MIME: Final = "audio/L16;codec=pcm;rate=24000"
PROMPT_TOKENS: Final = 5
AUDIO_TOKENS: Final = 50
SPEND_BY_ID_SQL: Final = 'SELECT call_type, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
SPEND_BY_KEY_SQL: Final = 'SELECT call_type, request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s'


def marker() -> str:
    return f"tts-{uuid.uuid4().hex[:12]}"


def audio_param(voice: JsonValue) -> dict[str, JsonValue]:
    return {"voice": voice, "format": "pcm16"}


def chat_body(model: str, voice: JsonValue, text: str, *, stream: bool = False) -> dict[str, JsonValue]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "modalities": ["audio"],
        "audio": audio_param(voice),
        **({"stream": True} if stream else {}),
    }


def received_voice(request: Request) -> JsonValue:
    return voice_in(JSON_OBJECT.validate_json(request.body))


def text_in(body: Mapping[str, JsonValue]) -> str:
    contents: Final = body["contents"]
    assert isinstance(contents, list) and contents, body
    parts: Final = object_value(contents[0])["parts"]
    assert isinstance(parts, list) and parts, body
    return string_value(object_value(parts[0])["text"])


def received_text(request: Request) -> str:
    return text_in(JSON_OBJECT.validate_json(request.body))


def _candidate(text: str) -> dict[str, JsonValue]:
    return {
        "content": {
            "role": "model",
            "parts": [{"text": text}, {"inlineData": {"mimeType": AUDIO_MIME, "data": PCM_B64}}],
        },
        "finishReason": "STOP",
    }


def _usage() -> dict[str, JsonValue]:
    return {
        "promptTokenCount": PROMPT_TOKENS,
        "candidatesTokenCount": AUDIO_TOKENS,
        "totalTokenCount": PROMPT_TOKENS + AUDIO_TOKENS,
        "candidatesTokensDetails": [{"modality": "AUDIO", "tokenCount": AUDIO_TOKENS}],
    }


def _rejection(voice: JsonValue) -> Reply:
    error: Final = {"code": 400, "message": f"{REJECTED_PREFIX}{voice} and language: ", "status": "INVALID_ARGUMENT"}
    return Reply(status=400, body=json.dumps({"error": error}).encode())


def _accepted(voice: JsonValue) -> bool:
    return voice == NO_VOICE or (isinstance(voice, str) and voice in ACCEPTED_VOICES)


def gemini_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    voice: Final = received_voice(request)
    if not _accepted(voice):
        return _rejection(voice)
    text: Final = received_text(request)
    if "streamGenerateContent" in request.target:
        first: Final = json.dumps({"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]})
        last: Final = json.dumps({"candidates": [_candidate("")], "usageMetadata": _usage(), "modelVersion": BACKEND})
        return Reply(
            content_type="text/event-stream", chunks=(f"data: {first}\n\n".encode(), f"data: {last}\n\n".encode())
        )
    body: Final = {"candidates": [_candidate(text)], "usageMetadata": _usage(), "modelVersion": BACKEND}
    return Reply(body=json.dumps(body).encode())


def gemini_deployment(scenario: Scenario, wire_url: str, **extra: JsonValue) -> str:
    return scenario.model(model=f"gemini/{BACKEND}", api_base=wire_url, api_key=GEMINI_API_KEY, **extra)


def vertex_deployment(
    scenario: Scenario, wire_url: str, token_url: str, *, api_base: str | None = None, **extra: JsonValue
) -> str:
    return scenario.model(
        model=f"vertex_ai/{BACKEND}",
        api_base=api_base or f"{wire_url}{VERTEX_MODEL_PATH}",
        api_key=None,
        vertex_project=VERTEX_PROJECT,
        vertex_location=VERTEX_LOCATION,
        vertex_credentials=service_account_json(VERTEX_PROJECT, token_url.rstrip("/")),
        **extra,
    )


def openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        base_url=f"{gateway.client.base_url}/v1",
        api_key=gateway.key,
        max_retries=0,
        http_client=httpx.AsyncClient(trust_env=False, timeout=60),
    )


def anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(gateway.client.base_url),
        api_key=gateway.key,
        auth_token=gateway.key,
        max_retries=0,
        http_client=httpx.Client(trust_env=False, timeout=60),
    )


def spend_row(response_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(SPEND_BY_ID_SQL, (response_id,)), lambda values: len(values) == 1, seconds=70
    )
    return rows[0]


def spend_rows_for_key(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(SPEND_BY_KEY_SQL, (sha256(key.encode()).hexdigest(),)),
        lambda rows: len(rows) == count,
        seconds=70,
    )


def error_message(response: httpx.Response) -> str:
    error: Final = JSON_OBJECT.validate_json(response.content)["error"]
    return string_value(object_value(error)["message"]) if isinstance(error, dict) else string_value(error)


def audio_data(payload: Mapping[str, JsonValue]) -> str:
    choices: Final = payload["choices"]
    assert isinstance(choices, list) and len(choices) == 1, payload
    message: Final = object_value(object_value(choices[0])["message"])
    return string_value(object_value(message["audio"])["data"])


LIVE_BACKEND: Final = "gemini-3.8-live"
LIVE_INPUT_TOKENS: Final = 7
LIVE_OUTPUT_TOKENS: Final = 5
SPEND_BY_CALL_SQL: Final = 'SELECT call_type, completion_tokens FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s'


def voice_in(body: Mapping[str, JsonValue]) -> JsonValue:
    generation: Final = body.get("generationConfig")
    speech: Final = object_value(generation).get("speechConfig") if isinstance(generation, dict) else None
    voice_config: Final = object_value(speech).get("voiceConfig") if isinstance(speech, dict) else None
    if not isinstance(voice_config, dict):
        return NO_VOICE
    return object_value(voice_config["prebuiltVoiceConfig"])["voiceName"]


def spend_row_by_call(call_id: str) -> dict[str, JsonValue]:
    assert call_id, "the response carried no x-litellm-call-id"
    rows: Final = eventually(
        lambda: read_rows(SPEND_BY_CALL_SQL, (call_id,)), lambda values: len(values) == 1, seconds=70
    )
    return rows[0]


def live_turn_script() -> RealtimeResponse:
    return RealtimeResponse(
        content_type="application/x-realtime",
        events=(
            {"serverContent": {"modelTurn": {"parts": [{"text": "scripted $REQUEST_ID"}]}}},
            {
                "serverContent": {"turnComplete": True},
                "usageMetadata": {
                    "promptTokenCount": LIVE_INPUT_TOKENS,
                    "responseTokenCount": LIVE_OUTPUT_TOKENS,
                    "totalTokenCount": LIVE_INPUT_TOKENS + LIVE_OUTPUT_TOKENS,
                },
            },
        ),
    )


def live_scenario(scenario: Scenario) -> ScenarioHandle:
    handle: Final = register_scenario(f"tts-live-{uuid.uuid4().hex[:12]}", live_turn_script())
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def live_deployment(scenario: Scenario, project: str, upstream_url: str) -> str:
    return scenario.model(
        model=f"vertex_ai/{LIVE_BACKEND}",
        api_base=upstream_url.rstrip("/"),
        api_key=None,
        vertex_project=project,
        vertex_location=VERTEX_LOCATION,
        vertex_credentials=service_account_json(project, upstream_url.rstrip("/")),
        model_info={"mode": "realtime"},
    )


def _user_turn() -> str:
    item: Final = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "say the line"}]}
    return json.dumps({"type": "conversation.item.create", "item": item})


async def _live_frames(socket: ClientConnection, voice: str) -> AsyncIterator[dict[str, JsonValue]]:
    try:
        first: Final = JSON_OBJECT.validate_json(await socket.recv())
        yield first
        if first.get("type") != "session.created":
            return
        await socket.send(json.dumps({"type": "session.update", "session": {"voice": voice}}))
        await socket.send(_user_turn())
        async for message in socket:
            event: Final = JSON_OBJECT.validate_json(message)
            yield event
            if event.get("type") == "response.done":
                return
    except ConnectionClosed as closed:
        yield {"type": "closed", "code": 1006 if closed.rcvd is None else closed.rcvd.code}


async def _live_session(ws_base: str, model: str, key: str, voice: str) -> tuple[dict[str, JsonValue], ...]:
    headers: Final = {"Authorization": f"Bearer {key}"}
    async with websockets.connect(f"{ws_base}/v1/realtime?model={model}", additional_headers=headers) as socket:
        return tuple([frame async for frame in _live_frames(socket, voice)])


def live_turn(gateway: Gateway, model: str, key: str, voice: str) -> tuple[dict[str, JsonValue], ...]:
    ws_base: Final = str(gateway.client.base_url).rstrip("/").replace("http://", "ws://")
    return asyncio.run(asyncio.wait_for(_live_session(ws_base, model, key, voice), 60))


def frame_types(frames: tuple[dict[str, JsonValue], ...]) -> tuple[str, ...]:
    return tuple(string_value(frame["type"]) for frame in frames)


def live_setups(gateway: Gateway, project: str) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream:
        observed: Final = tuple(map(object_value, upstream.get("/__observations").json()["requests"]))
    frames: Final = tuple(
        object_value(request["body"])
        for request in observed
        if request["api_key"] == project and request["method"] == "WEBSOCKET_FRAME"
    )
    assert all(request["path"] == GEMINI_LIVE_PATH for request in observed if request["api_key"] == project)
    return tuple(object_value(frame["setup"]) for frame in frames if "setup" in frame)


HEALTH_DEFAULT: Final = "tts-health-default-voice"
HEALTH_NOVA: Final = "tts-health-nova"
HEALTH_FABLE: Final = "tts-health-fable"


def _health_model(name: str, **model_info: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": f"gemini/{BACKEND}", "api_key": GEMINI_API_KEY},
        "model_info": {"mode": "audio_speech", **model_info},
    }


def owned_config(directory: Path, model_list: Sequence[JsonValue]) -> Path:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    settings: Final = object_value(config["litellm_settings"])
    cache_params: Final = {**object_value(settings["cache_params"]), "namespace": f"tts-{uuid.uuid4().hex}"}
    merged: Final = {
        **config,
        "model_list": list(model_list),
        "litellm_settings": {**settings, "cache_params": cache_params},
        "router_settings": {**object_value(config["router_settings"]), "num_retries": 0},
    }
    path: Final = directory / f"gemini-tts-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path


def health_config(directory: Path) -> Path:
    return owned_config(
        directory,
        (
            _health_model(HEALTH_DEFAULT),
            _health_model(HEALTH_NOVA, health_check_voice="nova"),
            _health_model(HEALTH_FABLE, health_check_voice="fable"),
        ),
    )


def speech_deployment(scenario: Scenario, wire_url: str) -> str:
    return scenario.model(
        model=f"gemini/{BACKEND}", api_base=wire_url, api_key=GEMINI_API_KEY, model_info={"mode": "audio_speech"}
    )


def wav_payload(body: bytes) -> bytes:
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE", body[:12]
    return body[44:]


@dataclass(frozen=True, slots=True)
class ChunkedPeer:
    url: str
    received: SimpleQueue[Request]

    def drain(self) -> tuple[Request, ...]:
        return tuple(self.received.get_nowait() for _ in range(self.received.qsize()))


def _read_chunked(handler: BaseHTTPRequestHandler) -> bytes:
    def chunks() -> Generator[bytes, None, None]:
        while True:
            size: Final = int(handler.rfile.readline().split(b";", 1)[0].strip(), 16)
            if size == 0:
                handler.rfile.readline()
                return
            yield handler.rfile.read(size)
            handler.rfile.readline()

    return b"".join(chunks())


@contextmanager
def chunked_peer(respond: Callable[[Request], Reply]) -> Generator[ChunkedPeer, None, None]:
    """Owned TCP peer like ``wire_server`` that also reads a ``transfer-encoding: chunked`` upload body,
    the shape the Vertex files route streams a batch file to GCS in."""
    received: Final[SimpleQueue[Request]] = SimpleQueue()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            chunked: Final = self.headers.get("transfer-encoding", "").lower() == "chunked"
            body: Final = (
                _read_chunked(self) if chunked else self.rfile.read(int(self.headers.get("content-length", "0")))
            )
            request: Final = Request(
                self.command, self.path, {name.lower(): value for name, value in self.headers.items()}, body
            )
            received.put(request)
            reply: Final = respond(request)
            self.send_response(reply.status)
            self.send_header("content-type", reply.content_type)
            self.send_header("content-length", str(len(reply.body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(reply.body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield ChunkedPeer(f"http://127.0.0.1:{server.server_address[1]}", received)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
