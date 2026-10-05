from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import uuid
import zlib
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from typing import Final, cast

import httpx
import uvicorn
from _fake_openai_endpoint_server import chat_completions, completions, embeddings, health, moderations
from integration.cost_calculation.cost_tracking_case import (
    BinaryResponse,
    EventStreamEvent,
    EventStreamResponse,
    JsonResponse,
    RealtimeResponse,
    RoutedResponse,
    SseResponse,
    StoredResponse,
    TextResponse,
)
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
CASES_FILE: Final = Path(__file__).resolve().parents[1] / "cost_calculation" / "cost_tracking_cases.json"
INTERNAL_FIELDS: Final = frozenset(
    {
        "litellm_params",
        "litellm_logging_obj",
        "litellm_call_id",
        "litellm_metadata",
        "proxy_server_request",
        "rpm",
        "tpm",
        "timeout",
        "stream_chunk_size",
    }
)


def error_type(status: int) -> str:
    if status == 429:
        return "rate_limit_error"
    return "invalid_request_error" if status < 500 else "server_error"


def _form_observation_value(value: str | StarletteUploadFile) -> JsonValue:
    if isinstance(value, StarletteUploadFile):
        return {"filename": value.filename, "content_type": value.content_type}
    return value


@dataclass(frozen=True, slots=True)
class Observation:
    path: str
    authorization: str
    body: dict[str, JsonValue]
    method: str = "POST"
    api_key: str = ""


class InteractionState(BaseModel):
    """What the scripted Interactions API answers for one interaction id until a DELETE drops it."""

    model_config = ConfigDict(extra="forbid")

    status: str
    usage: dict[str, JsonValue] | None = None
    get_status: int = 200
    delay_seconds: float = 0


class _ScenarioRegistration(BaseModel):
    scenario_id: str
    response: StoredResponse


def _aws_str_header(name: str, value: str) -> bytes:
    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return (
        struct.pack("!B", len(name_bytes))
        + name_bytes
        + struct.pack("!B", 7)
        + struct.pack("!H", len(value_bytes))
        + value_bytes
    )


def _aws_int_header(name: str, value: int) -> bytes:
    name_bytes: Final = name.encode()
    return struct.pack("!B", len(name_bytes)) + name_bytes + struct.pack("!B", 4) + struct.pack("!i", value)


def aws_event_stream_frame(headers: Mapping[str, str | int], payload: bytes) -> bytes:
    """One AWS event-stream frame: a string header is wire type 7, an int header wire type 4 (int32)."""
    headers_bytes: Final = b"".join(
        _aws_str_header(name, value) if isinstance(value, str) else _aws_int_header(name, value)
        for name, value in headers.items()
    )
    total_length: Final = 12 + len(headers_bytes) + len(payload) + 4
    prelude: Final = struct.pack("!II", total_length, len(headers_bytes))
    prelude_crc: Final = struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message: Final = prelude + prelude_crc + headers_bytes + payload
    return message + struct.pack("!I", zlib.crc32(message) & 0xFFFFFFFF)


def _aws_event_frame(
    event_type: str,
    payload: Mapping[str, JsonValue],
    scenario_id: str,
    unique_id: str,
) -> bytes:
    payload_bytes: Final = (
        json.dumps(payload, separators=(",", ":"))
        .replace("$REQUEST_ID", scenario_id)
        .replace("$UNIQUE_ID", unique_id)
        .encode()
    )
    return aws_event_stream_frame(
        {":event-type": event_type, ":content-type": "application/json", ":message-type": "event"}, payload_bytes
    )


class ScenarioStore:
    def __init__(self) -> None:
        self._scenarios: dict[str, StoredResponse] = {}

    def put(self, scenario_id: str, response: StoredResponse) -> None:
        self._scenarios[scenario_id] = response

    def drop(self, scenario_id: str) -> bool:
        return self._scenarios.pop(scenario_id, None) is not None

    def get(self, scenario_id: str) -> StoredResponse | None:
        return self._scenarios.get(scenario_id)


@dataclass(frozen=True, slots=True)
class Provider:
    observations: SimpleQueue[Observation] = field(default_factory=SimpleQueue)
    scripts: dict[str, deque[int]] = field(default_factory=dict)
    scenario_store: ScenarioStore = field(default_factory=ScenarioStore)
    interactions: dict[str, InteractionState] = field(default_factory=dict)

    async def chat(self, request: Request) -> Response:
        body: Final = JSON_OBJECT.validate_json(await request.body())
        self.observations.put(
            Observation(request.url.path, request.headers.get("authorization", ""), body, request.method)
        )
        leaked: Final = tuple(sorted(INTERNAL_FIELDS.intersection(body)))
        if leaked:
            return JSONResponse({"error": {"message": f"Unexpected provider fields: {leaked}"}}, status_code=400)
        messages: Final = body.get("messages")
        if not isinstance(body.get("model"), str) or not isinstance(messages, list) or not messages:
            return JSONResponse({"error": {"message": "model and nonempty messages are required"}}, status_code=400)
        if any(
            not isinstance(message, dict)
            or message.get("role") not in {"system", "developer", "user", "assistant", "tool"}
            or "content" not in message
            for message in messages
        ):
            return JSONResponse({"error": {"message": "Invalid selected message contract"}}, status_code=400)
        script: Final = self.scripts.get(str(body["model"]))
        if script is not None:
            if not script:
                return JSONResponse({"error": {"message": "Script exhausted", "type": "api_error"}}, status_code=500)
            status: Final = script.popleft()
            if status != 200:
                return JSONResponse(
                    {
                        "error": {
                            "message": "Controlled provider failure",
                            "type": error_type(status),
                            "code": str(status),
                        }
                    },
                    status_code=status,
                )
        return await chat_completions(request)

    async def vector_store_search(self, request: Request) -> Response:
        body: Final = JSON_OBJECT.validate_json(await request.body())
        self.observations.put(
            Observation(request.url.path, request.headers.get("authorization", ""), body, request.method)
        )
        query: Final = body.get("query")
        if not isinstance(query, str) or not query:
            return JSONResponse({"error": {"message": "query is required"}}, status_code=400)
        vector_store_id: Final = cast(str, request.path_params["vector_store_id"])
        return JSONResponse(
            {
                "object": "vector_store.search_results.page",
                "search_query": query,
                "data": [
                    {
                        "file_id": f"file_{vector_store_id}",
                        "filename": "scripted.txt",
                        "score": 0.9,
                        "attributes": {},
                        "content": [{"type": "text", "text": f"scripted context for {query}"}],
                    }
                ],
                "has_more": False,
                "next_page": None,
            }
        )

    async def script(self, request: Request) -> Response:
        name: Final = cast(str, request.path_params["model"])
        if request.method in {"DELETE", "GET"} and name not in self.scripts:
            return JSONResponse({"error": "Script not found"}, status_code=404)
        if request.method == "GET":
            return JSONResponse({"remaining": list(self.scripts[name])})
        if request.method == "DELETE":
            remaining: Final = self.scripts.pop(name)
            return JSONResponse({"remaining": list(remaining)})
        body: Final = JSON_OBJECT.validate_json(await request.body())
        statuses: Final = body.get("statuses")
        if not isinstance(statuses, list) or not statuses or any(type(value) is not int for value in statuses):
            return JSONResponse({"error": "A nonempty list of HTTP status codes is required"}, status_code=400)
        self.scripts[name] = deque(int(str(value)) for value in statuses)
        return JSONResponse({"configured": len(statuses)})

    async def observed(self, request: Request) -> Response:
        values: Final = tuple(self.observations.get() for _ in range(self.observations.qsize()))
        return JSONResponse(
            {
                "requests": [
                    {
                        "path": value.path,
                        "authorization": value.authorization,
                        "body": value.body,
                        "method": value.method,
                        "api_key": value.api_key,
                    }
                    for value in values
                ]
            }
        )

    async def register_scenario(self, request: Request) -> Response:
        try:
            registration: Final = _ScenarioRegistration.model_validate_json(await request.body())
        except ValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        self.scenario_store.put(registration.scenario_id, registration.response)
        return JSONResponse({"scenario_id": registration.scenario_id})

    async def delete_scenario(self, request: Request) -> Response:
        scenario_id: Final = cast(str, request.path_params["scenario_id"])
        deleted: Final = self.scenario_store.drop(scenario_id)
        return JSONResponse({"deleted": deleted}, status_code=200 if deleted else 404)

    async def cost_map(self, _request: Request) -> Response:
        cases_file: Final = JSON_OBJECT.validate_json(CASES_FILE.read_bytes())
        return JSONResponse(cases_file["cost_map"])

    async def oauth_token(self, _request: Request) -> Response:
        return JSONResponse(
            {
                "access_token": "scripted-token",
                "token_type": "Bearer",
                "expires_in": 3600,
            }
        )

    async def scripted(self, request: Request) -> Response:
        segments: Final = tuple(segment for segment in cast(str, request.path_params["path"]).split("/") if segment)
        scenario_id: Final = (
            segments[0].split(":", 1)[0]
            if segments and self.scenario_store.get(segments[0].split(":", 1)[0]) is not None
            else request.headers.get("x-scripted-scenario", "")
        )
        response: Final = self.scenario_store.get(scenario_id)
        if response is None:
            return JSONResponse({"error": "Unknown scenario"}, status_code=404)
        content_type: Final = request.headers.get("content-type", "")
        if request.method == "POST" and "json" in content_type:
            raw_body: Final = await request.body()
            if raw_body:
                body: Final = JSON_OBJECT.validate_json(raw_body)
                if isinstance(body, dict):
                    self.observations.put(
                        Observation(
                            request.url.path,
                            request.headers.get("authorization", ""),
                            body,
                            request.method,
                            request.headers.get("x-goog-api-key", ""),
                        )
                    )
        elif request.method == "POST" and "multipart/form-data" in content_type:
            fields: Final = await request.form()
            body: Final = {name: _form_observation_value(value) for name, value in fields.items()}
            self.observations.put(
                Observation(request.url.path, request.headers.get("authorization", ""), body, request.method)
            )
        elif request.method == "GET":
            self.observations.put(
                Observation(request.url.path, request.headers.get("authorization", ""), {}, request.method)
            )
        if isinstance(response, RoutedResponse):
            route_key: Final = f"{request.method} /{'/'.join(segments[1:])}"
            route: Final = next(
                (
                    candidate
                    for key, candidate in response.routes.items()
                    if key.replace("$REQUEST_ID", scenario_id) == route_key
                ),
                None,
            )
            if route is None:
                return JSONResponse({"error": "Unknown scripted route"}, status_code=404)
            return self._response(route, scenario_id)
        return self._response(response, scenario_id)

    async def interaction_state(self, request: Request) -> Response:
        interaction_id: Final = cast(str, request.path_params["interaction_id"])
        if request.method == "DELETE":
            self.interactions.pop(interaction_id, None)
            return JSONResponse({"interaction_id": interaction_id, "registered": False})
        try:
            state: Final = InteractionState.model_validate_json(await request.body())
        except ValidationError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        self.interactions[interaction_id] = state
        return JSONResponse({"interaction_id": interaction_id, "registered": True})

    def _observe_interaction(self, request: Request) -> None:
        self.observations.put(
            Observation(
                request.url.path,
                request.headers.get("authorization", ""),
                {},
                method=request.method,
                api_key=request.headers.get("x-goog-api-key", ""),
            )
        )

    async def interaction(self, request: Request) -> Response:
        self._observe_interaction(request)
        interaction_id: Final = cast(str, request.path_params["interaction_id"])
        state: Final = self.interactions.get(interaction_id)
        if state is None:
            return JSONResponse(_interaction_not_found(interaction_id), status_code=404)
        if state.delay_seconds:
            await asyncio.sleep(state.delay_seconds)
        if request.method == "DELETE":
            if self.interactions.pop(interaction_id, None) is None:
                return JSONResponse(_interaction_not_found(interaction_id), status_code=404)
            return JSONResponse({})
        if state.get_status != 200:
            return JSONResponse(
                {"error": {"code": state.get_status, "message": "Scripted interaction fetch failure"}},
                status_code=state.get_status,
            )
        return JSONResponse(_interaction_body(interaction_id, state))

    async def cancel_interaction(self, request: Request) -> Response:
        self._observe_interaction(request)
        interaction_id: Final = cast(str, request.path_params["interaction_id"])
        state: Final = self.interactions.get(interaction_id)
        if state is None:
            return JSONResponse(_interaction_not_found(interaction_id), status_code=404)
        cancelled: Final = InteractionState(status="cancelled", usage=state.usage, get_status=state.get_status)
        self.interactions[interaction_id] = cancelled
        return JSONResponse(_interaction_body(interaction_id, cancelled))

    async def realtime(self, websocket: WebSocket) -> None:
        scenario_id: Final = websocket.headers.get("authorization", "").removeprefix("Bearer ")
        response: Final = self.scenario_store.get(scenario_id)
        if not isinstance(response, RealtimeResponse):
            await websocket.close(code=4404)
            return
        await websocket.accept()
        model: Final = websocket.query_params.get("model", "")
        await websocket.send_json(
            {
                "type": "session.created",
                "session": {
                    "id": f"sess_{scenario_id}",
                    "model": response.session_model if response.session_model is not None else model,
                },
            }
        )
        event_index: Final = iter(response.events)
        async for message in websocket.iter_json():
            payload: Final = JSON_OBJECT.validate_python(message)
            if payload.get("type") != "response.create":
                continue
            event: Final = next(event_index, None)
            if event is None:
                continue
            rendered: Final = JSON_OBJECT.validate_json(
                json.dumps(event, separators=(",", ":"))
                .replace("$REQUEST_ID", scenario_id)
                .replace("$UNIQUE_ID", f"{scenario_id}-{uuid.uuid4().hex[:8]}")
            )
            await websocket.send_json(rendered)

    @staticmethod
    def _response(response: StoredResponse, scenario_id: str) -> Response:
        unique_id: Final = f"{scenario_id}-{uuid.uuid4().hex[:8]}"
        match response:
            case JsonResponse():
                return Response(
                    content=json.dumps(response.body, separators=(",", ":"))
                    .replace("$REQUEST_ID", scenario_id)
                    .replace("$UNIQUE_ID", unique_id)
                    .encode(),
                    media_type=response.content_type,
                    status_code=response.status,
                )
            case BinaryResponse():
                return Response(
                    content=b"\x00" * response.length,
                    media_type=response.content_type,
                )
            case TextResponse():
                return Response(
                    content=response.body.replace("$REQUEST_ID", scenario_id).encode(),
                    media_type=response.content_type,
                    status_code=response.status,
                )
            case SseResponse():
                if response.frame_delay_ms > 0:

                    async def stream() -> AsyncIterator[bytes]:
                        for frame in response.frames:
                            yield (
                                f"{frame.replace('$REQUEST_ID', scenario_id).replace('$UNIQUE_ID', unique_id)}\n\n"
                            ).encode()
                            await asyncio.sleep(response.frame_delay_ms / 1000)

                    return StreamingResponse(stream(), media_type=response.content_type)
                stream_body: Final = (
                    ("\n\n".join(response.frames) + "\n\n")
                    .replace("$REQUEST_ID", scenario_id)
                    .replace("$UNIQUE_ID", unique_id)
                )
                return Response(content=stream_body.encode(), media_type=response.content_type)
            case EventStreamResponse():
                events: Final = (
                    tuple(
                        EventStreamEvent(
                            event_type="chunk",
                            payload={
                                "bytes": base64.b64encode(
                                    json.dumps(event.payload, separators=(",", ":"))
                                    .replace("$REQUEST_ID", scenario_id)
                                    .replace("$UNIQUE_ID", unique_id)
                                    .encode()
                                ).decode(),
                            },
                        )
                        for event in response.events
                    )
                    if response.framing == "invoke"
                    else response.events
                )
                event_body: Final = b"".join(
                    _aws_event_frame(event.event_type, event.payload, scenario_id, unique_id) for event in events
                )
                return Response(content=event_body, media_type=response.content_type)

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/health", health),
                Route("/__observations", self.observed),
                Route("/__scripts/{model}", self.script, methods=["POST", "DELETE", "GET"]),
                Route("/__scenarios", self.register_scenario, methods=["POST"]),
                Route("/__scenarios/{scenario_id}", self.delete_scenario, methods=["DELETE"]),
                Route("/_cost_map", self.cost_map, methods=["GET"]),
                Route("/_oauth/token", self.oauth_token, methods=["POST"]),
                Route("/v1/chat/completions", self.chat, methods=["POST"]),
                Route("/v1/completions", completions, methods=["POST"]),
                Route("/v1/embeddings", embeddings, methods=["POST"]),
                Route("/v1/moderations", moderations, methods=["POST"]),
                Route("/vector_stores/{vector_store_id}/search", self.vector_store_search, methods=["POST"]),
                Route("/__interactions/{interaction_id}", self.interaction_state, methods=["PUT", "DELETE"]),
                Route("/v1beta/interactions/{interaction_id}:cancel", self.cancel_interaction, methods=["POST"]),
                Route("/v1beta/interactions/{interaction_id}", self.interaction, methods=["GET", "DELETE"]),
                Route(
                    "/{prefix:path}/v1beta/interactions/{interaction_id}:cancel",
                    self.cancel_interaction,
                    methods=["POST"],
                ),
                Route(
                    "/{prefix:path}/v1beta/interactions/{interaction_id}",
                    self.interaction,
                    methods=["GET", "DELETE"],
                ),
                Route("/{path:path}", self.scripted, methods=["POST"]),
                Route("/{path:path}", self.scripted, methods=["GET"]),
                WebSocketRoute("/v1/realtime", self.realtime),
            ]
        )


CONTROL_URL: Final = os.environ.get("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:8190").rstrip("/")


def _interaction_not_found(interaction_id: str) -> dict[str, JsonValue]:
    return {"error": {"code": 404, "message": f"Interaction {interaction_id} not found", "status": "NOT_FOUND"}}


def _interaction_body(interaction_id: str, state: InteractionState) -> dict[str, JsonValue]:
    return {
        "id": interaction_id,
        "object": "interaction",
        "model": "gemini-3.8-flash",
        "status": state.status,
        "steps": [],
        "usage": state.usage,
    }


@dataclass(frozen=True, slots=True)
class ScenarioHandle:
    scenario_id: str
    control_url: str

    def api_base(self) -> str:
        return f"{self.control_url}/{self.scenario_id}"


def register_scenario(scenario_id: str, response: StoredResponse, *, control_url: str = CONTROL_URL) -> ScenarioHandle:
    http_response: Final = httpx.post(
        f"{control_url}/__scenarios",
        json={"scenario_id": scenario_id, "response": response.model_dump(mode="json")},
        trust_env=False,
        timeout=15,
    )
    http_response.raise_for_status()
    return ScenarioHandle(
        scenario_id=scenario_id,
        control_url=control_url,
    )


def delete_scenario(handle: ScenarioHandle) -> None:
    response: Final = httpx.delete(
        f"{handle.control_url}/__scenarios/{handle.scenario_id}",
        trust_env=False,
        timeout=15,
    )
    response.raise_for_status()


def set_interaction_state(control_url: str, interaction_id: str, state: InteractionState) -> None:
    response: Final = httpx.put(
        f"{control_url}/__interactions/{interaction_id}",
        content=state.model_dump_json(),
        headers={"content-type": "application/json"},
        trust_env=False,
        timeout=15,
    )
    response.raise_for_status()


def clear_interaction_state(control_url: str, interaction_id: str) -> None:
    response: Final = httpx.delete(f"{control_url}/__interactions/{interaction_id}", trust_env=False, timeout=15)
    response.raise_for_status()


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8190)
    arguments: Final = parser.parse_args()
    uvicorn.run(Provider().app(), host="127.0.0.1", port=cast(int, arguments.port), access_log=False)


if __name__ == "__main__":
    main()
