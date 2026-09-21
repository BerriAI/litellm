from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Mapping
import json
from dataclasses import dataclass, field
import os
from pathlib import Path
from queue import SimpleQueue
import struct
from typing import Final, cast
import zlib

import httpx
import uvicorn
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from _fake_openai_endpoint_server import chat_completions, completions, embeddings, health, moderations
from integration.cost_calculation.cost_tracking_case import (
    EventStreamResponse,
    JsonResponse,
    SseResponse,
    StoredResponse,
)

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


@dataclass(frozen=True, slots=True)
class Observation:
    path: str
    authorization: str
    body: dict[str, JsonValue]


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


def _aws_event_frame(event_type: str, payload: Mapping[str, JsonValue], scenario_id: str) -> bytes:
    payload_bytes: Final = json.dumps(payload, separators=(",", ":")).replace(
        "$REQUEST_ID", scenario_id
    ).encode()
    headers_bytes: Final = (
        _aws_str_header(":event-type", event_type)
        + _aws_str_header(":content-type", "application/json")
        + _aws_str_header(":message-type", "event")
    )
    total_length: Final = 12 + len(headers_bytes) + len(payload_bytes) + 4
    prelude: Final = struct.pack("!II", total_length, len(headers_bytes))
    prelude_crc: Final = struct.pack("!I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message: Final = prelude + prelude_crc + headers_bytes + payload_bytes
    return message + struct.pack("!I", zlib.crc32(message) & 0xFFFFFFFF)


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

    async def chat(self, request: Request) -> Response:
        body: Final = JSON_OBJECT.validate_json(await request.body())
        self.observations.put(Observation(request.url.path, request.headers.get("authorization", ""), body))
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
                    {"error": {"message": "Controlled provider failure", "type": error_type(status), "code": str(status)}},
                    status_code=status,
                )
        return await chat_completions(request)

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

    async def observed(self, _request: Request) -> Response:
        values: Final = tuple(self.observations.get() for _ in range(self.observations.qsize()))
        return JSONResponse(
            {
                "requests": [
                    {"path": value.path, "authorization": value.authorization, "body": value.body} for value in values
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
        if not segments:
            return JSONResponse({"error": "Unknown scenario"}, status_code=404)
        scenario_id: Final = segments[0].split(":", 1)[0]
        response: Final = self.scenario_store.get(scenario_id)
        if response is None:
            return JSONResponse({"error": "Unknown scenario"}, status_code=404)
        return self._response(response, scenario_id)

    @staticmethod
    def _response(response: StoredResponse, scenario_id: str) -> Response:
        match response:
            case JsonResponse():
                return Response(
                    content=json.dumps(response.body, separators=(",", ":")).replace(
                        "$REQUEST_ID", scenario_id
                    ).encode(),
                    media_type=response.content_type,
                    status_code=response.status,
                )
            case SseResponse():
                stream_body: Final = ("\n\n".join(response.frames) + "\n\n").replace(
                    "$REQUEST_ID", scenario_id
                )
                return Response(content=stream_body.encode(), media_type=response.content_type)
            case EventStreamResponse():
                event_body: Final = b"".join(
                    _aws_event_frame(event.event_type, event.payload, scenario_id) for event in response.events
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
                Route("/{path:path}", self.scripted, methods=["POST"]),
            ]
        )


CONTROL_URL: Final = os.environ.get("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:8190").rstrip("/")


@dataclass(frozen=True, slots=True)
class ScenarioHandle:
    scenario_id: str
    control_url: str

    def api_base(self) -> str:
        return f"{self.control_url}/{self.scenario_id}"


def register_scenario(scenario_id: str, response: StoredResponse) -> ScenarioHandle:
    http_response: Final = httpx.post(
        f"{CONTROL_URL}/__scenarios",
        json={"scenario_id": scenario_id, "response": response.model_dump(mode="json")},
        trust_env=False,
        timeout=15,
    )
    http_response.raise_for_status()
    return ScenarioHandle(
        scenario_id=scenario_id,
        control_url=CONTROL_URL,
    )


def delete_scenario(handle: ScenarioHandle) -> None:
    response: Final = httpx.delete(
        f"{CONTROL_URL}/__scenarios/{handle.scenario_id}",
        trust_env=False,
        timeout=15,
    )
    response.raise_for_status()


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8190)
    arguments: Final = parser.parse_args()
    uvicorn.run(Provider().app(), host="127.0.0.1", port=cast(int, arguments.port), access_log=False)


if __name__ == "__main__":
    main()
