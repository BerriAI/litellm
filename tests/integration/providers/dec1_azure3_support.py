"""Shared rig for the December 1 Azure contracts, round 3 (`test_dec1_azure3_*`).

Round 1 (`test_dec1_azure_*`) pinned Azure OpenAI and Foundry chat, Responses and embeddings; round 2
(`test_dec1_bigprov2_*`) added environment credentials, timeouts and the router relay. Round 3 covers what
both left in "Not covered": Azure realtime over TLS, image generation request shapes, audio speech and
transcription (Azure OpenAI and Azure AI Speech), Assistants, Foundry Agents, the Foundry Model Router,
Foundry serverless (MaaS) families, Responses background mode, files, vector stores, data zone pricing and
the `v1` / `latest` / `preview` api-version family.

Everything here builds on `dec1_azure_support.py` (imported, never changed). This module only adds:

- A websocket handshake driver and observation reader for the TLS scripted upstream (`owned_upstream(...,
  tls=True)`), because the Azure realtime handler always passes an SSL context to `websockets.connect`
  (`litellm/llms/azure/realtime/handler.py`, `get_shared_realtime_ssl_context`), so a plain `ws://` peer is
  never reached. The scripted upstream records every websocket upgrade as an observation with method
  `WEBSOCKET`, its path, its query pairs, the `Authorization` header and the `api-key` header
  (`tests/integration/_support/upstream.py`, `realtime`).
- Multipart parsing of what an owned TCP peer received, for files and audio transcription uploads.
- Small reply builders for Azure image, audio, Foundry Agents and Responses lifecycle answers.

Nothing here ever reaches a real Azure host: every `api_base` is a loopback peer.
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

import httpx
import websockets
from integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration._support.wire import Reply, Request, wire_server
from integration.cost_calculation.cost_tracking_case import RealtimeResponse
from integration.providers.dec1_azure_support import Rig, register_model
from pydantic import JsonValue
from websockets.exceptions import ConnectionClosed

PNG_B64: Final = base64.b64encode(b"\x89PNG\r\n\x1a\nsynthetic-azure-image").decode()
"""A synthetic base64 image payload. Not a valid PNG beyond its signature; nothing decodes it."""
SPEECH_BYTES: Final = b"ID3synthetic-azure-speech-bytes"
WAV_BYTES: Final = b"RIFF\x24\x00\x00\x00WAVEfmt synthetic-azure-audio"

REALTIME_DEFAULT_BETA_API_VERSION: Final = "2024-10-01-preview"
"""The api_version the realtime dispatcher uses when the deployment names none
(`litellm/realtime_api/main.py`, `api_version or litellm_params.api_version or "2024-10-01-preview"`)."""

AGENTS_DEFAULT_API_VERSION: Final = "2025-05-01"
"""`AzureAIAgentsConfig.DEFAULT_API_VERSION` (`litellm/llms/azure_ai/agents/transformation.py:71`)."""


@contextmanager
def full_peer(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    model: str,
    model_info: Mapping[str, JsonValue] | None = None,
    base_path: str = "",
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Like `dec1_azure_support.azure`, but every request (GET, DELETE included) reaches `respond`.

    `azure(...)` answers every non-POST with an empty list so model probes never fail; lifecycle routes
    (Foundry Agents run polls, Responses GET/cancel/DELETE, file reads) need their GETs scripted, so this rig
    hands them all to `respond`. Parameters are exactly `litellm_params` plus `api_base` (a `None` value drops
    the key); nothing is defaulted.
    """
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        parameters: Final = {"api_base": f"{wire.url}{base_path}", **litellm_params}
        cleaned: Final = {key: value for key, value in parameters.items() if value is not None}
        name: Final = register_model(scenario, model=model, model_info=model_info, **cleaned)
        yield Rig(gateway, scenario, wire, name)


def ws_base(http_url: str) -> str:
    return http_url.replace("https://", "wss://").replace("http://", "ws://")


def realtime_scenario(scenario: Scenario, control_url: str, *, transcript: str = "hello there") -> ScenarioHandle:
    """Register a realtime script on the (TLS) scripted upstream whose id doubles as the deployment credential.

    The upstream answers `session.created` on accept, then one transcript and one `response.done` per client
    turn. The id is removed when the scenario closes.
    """
    scenario_id: Final = f"dec1-azure3-rt-{uuid.uuid4().hex[:12]}"
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "evt_$UNIQUE_ID",
            "item_id": "item_$UNIQUE_ID",
            "content_index": 0,
            "transcript": transcript,
            "usage": {"type": "duration", "seconds": 1.5},
        },
        {
            "type": "response.done",
            "event_id": "evt_$UNIQUE_ID",
            "response": {
                "id": "resp_$UNIQUE_ID",
                "object": "realtime.response",
                "status": "completed",
                "output": [],
                "usage": {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
            },
        },
    )
    handle: Final = register_scenario(
        scenario_id, RealtimeResponse(content_type="application/x-realtime", events=events), control_url=control_url
    )
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


@dataclass(frozen=True, slots=True)
class Handshake:
    """What the client saw: the first event (or none) and the close code if the proxy closed the socket."""

    first_event: dict[str, JsonValue] | None
    events: tuple[dict[str, JsonValue], ...]
    close_code: int | None


async def _handshake(url: str, headers: Mapping[str, str], read_events: int) -> Handshake:
    events: list[dict[str, JsonValue]] = []
    close_code: int | None = None
    try:
        async with websockets.connect(url, additional_headers=dict(headers)) as socket:
            for _ in range(read_events):
                message = await asyncio.wait_for(socket.recv(), 60)
                events.append(JSON_OBJECT.validate_json(message))
    except ConnectionClosed as closed:
        close_code = 1006 if closed.rcvd is None else closed.rcvd.code
    except TimeoutError:
        pass
    return Handshake(events[0] if events else None, tuple(events), close_code)


def open_realtime(
    proxy_url: str,
    key: str,
    query: str,
    *,
    path: str = "/v1/realtime",
    headers: Mapping[str, str] | None = None,
    read_events: int = 1,
) -> Handshake:
    """Open a realtime websocket on the proxy with `Authorization: Bearer <key>`, read `read_events` frames, close."""
    url: Final = f"{ws_base(proxy_url)}{path}?{query}"
    return asyncio.run(_handshake(url, {"Authorization": f"Bearer {key}", **(headers or {})}, read_events))


def upstream_observations(upstream_url: str) -> tuple[dict[str, JsonValue], ...]:
    with httpx.Client(base_url=upstream_url, timeout=5, trust_env=False, verify=False) as client:
        observations: Final = JSON_OBJECT.validate_json(client.get("/__observations").content)
    requests: Final = observations["requests"]
    assert isinstance(requests, list), observations
    return tuple(object_value(request) for request in requests)


@dataclass(frozen=True, slots=True)
class Upgrade:
    """One websocket upgrade the TLS upstream accepted or refused."""

    path: str
    query: tuple[tuple[str, str], ...]
    authorization: str
    api_key: str


def upgrades_for(upstream_url: str, credential: str) -> tuple[Upgrade, ...]:
    """Every websocket upgrade that carried `credential` as `api-key` or as `Authorization: Bearer`."""
    found: list[Upgrade] = []
    for request in upstream_observations(upstream_url):
        if request.get("method") != "WEBSOCKET":
            continue
        authorization = str(request.get("authorization") or "")
        api_key = str(request.get("api_key") or "")
        if api_key != credential and authorization != f"Bearer {credential}":
            continue
        pairs = object_value(request["body"]).get("query") or []
        assert isinstance(pairs, list), request
        query = tuple((str(pair[0]), str(pair[1])) for pair in pairs if isinstance(pair, list))  # pyright: ignore[reportUnknownArgumentType]
        found.append(Upgrade(str(request["path"]), query, authorization, api_key))
    return tuple(found)


def multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers.get("content-type")
    return tuple(parsed.iter_parts())


def multipart_fields(request: Request) -> dict[str, str]:
    """The text form fields of a multipart request (file parts left out)."""
    fields: Final[dict[str, str]] = {}
    for part in multipart_parts(request):
        if part.get_filename() is None:
            payload = part.get_payload(decode=True)
            assert isinstance(payload, bytes)
            fields[str(part.get_param("name", header="content-disposition"))] = payload.decode()
    return fields


def multipart_file(request: Request, name: str = "file") -> bytes:
    for part in multipart_parts(request):
        if part.get_param("name", header="content-disposition") == name and part.get_filename() is not None:
            payload = part.get_payload(decode=True)
            assert isinstance(payload, bytes)
            return payload
    raise AssertionError(f"no file part named {name}")


def image_json(*, revised_prompt: str = "a synthetic lighthouse") -> dict[str, JsonValue]:
    """An Azure OpenAI image generation answer with one base64 image."""
    return {"created": 1, "data": [{"b64_json": PNG_B64, "revised_prompt": revised_prompt}]}


def json_bytes(payload: Mapping[str, JsonValue] | Sequence[JsonValue]) -> bytes:
    return json.dumps(payload).encode()


def binary_reply(body: bytes, content_type: str) -> Reply:
    return Reply(body=body, content_type=content_type)


def agent_thread(thread_id: str) -> dict[str, JsonValue]:
    return {"id": thread_id, "object": "thread", "created_at": 1, "metadata": {}}


def agent_run(run_id: str, thread_id: str, agent_id: str, status: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": run_id,
        "object": "thread.run",
        "thread_id": thread_id,
        "assistant_id": agent_id,
        "status": status,
        **extra,
    }


def agent_messages(text: str, *, annotations: Sequence[JsonValue] = ()) -> dict[str, JsonValue]:
    """A `GET /threads/{id}/messages` page: newest first, the assistant answer then the user turn."""
    return {
        "object": "list",
        "data": [
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "object": "thread.message",
                "role": "assistant",
                "content": [{"type": "text", "text": {"value": text, "annotations": list(annotations)}}],
            },
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "object": "thread.message",
                "role": "user",
                "content": [{"type": "text", "text": {"value": "earlier user turn", "annotations": []}}],
            },
        ],
        "first_id": None,
        "last_id": None,
        "has_more": False,
    }


def agent_sse(thread_id: str, texts: Sequence[str]) -> Reply:
    """A Foundry Agents create-thread-and-run SSE stream: `thread.created`, one `thread.message.delta` per text,
    then `[DONE]`. Events carry an `event:` line, which the handler reads to classify each `data:` line."""
    lines: list[bytes] = [f"event: thread.created\ndata: {json.dumps(agent_thread(thread_id))}\n\n".encode()]
    for text in texts:
        delta = {
            "id": "msg_1",
            "object": "thread.message.delta",
            "delta": {"content": [{"index": 0, "type": "text", "text": {"value": text}}]},
        }
        lines.append(f"event: thread.message.delta\ndata: {json.dumps(delta)}\n\n".encode())
    lines.append(b"event: done\ndata: [DONE]\n\n")
    return Reply(content_type="text/event-stream", chunks=tuple(lines))


def responses_object(response_id: str, status: str, *, text: str | None = None) -> dict[str, JsonValue]:
    """An Azure Responses object in any lifecycle status; `text` adds one completed output message and usage."""
    output: Final[list[JsonValue]] = (
        [
            {
                "id": f"msg_{response_id}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
        if text is not None
        else []
    )
    body: Final[dict[str, JsonValue]] = {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": status,
        "background": True,
        "model": "gpt-5.4-nano",
        "output": output,
    }
    if text is not None:
        body["usage"] = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}
    return body


def decoded_response_id(encoded: str) -> str:
    """Decode a proxy managed Responses id (`resp_` + base64 of
    `litellm:custom_llm_provider:{provider};model_id:{id};response_id:{raw}`, `litellm/types/utils.py:5308`)."""
    assert encoded.startswith("resp_"), encoded
    body: Final = encoded.removeprefix("resp_")
    # Run 44176682961 got an id without base64 padding; managed ids elsewhere use unpadded urlsafe base64
    # (`litellm/llms/base_llm/managed_resources/utils.py:328`), so accept both alphabets and restore the padding.
    return base64.urlsafe_b64decode(body.replace("+", "-").replace("/", "_") + "=" * (-len(body) % 4)).decode()
