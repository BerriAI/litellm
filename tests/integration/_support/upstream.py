from __future__ import annotations

import argparse
from collections import deque
import json
from dataclasses import dataclass, field
import os
from pathlib import Path
from queue import SimpleQueue
from typing import Final, cast

import httpx
import uvicorn
from pydantic import JsonValue, TypeAdapter, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from _fake_openai_endpoint_server import chat_completions, completions, embeddings, health, moderations
from integration._support.scripted_wires import (
    WIRE_MOUNTS,
    RenderedResponse,
    Scenario,
    ScenarioDeleted,
    ScenarioRegistered,
    ScenarioStore,
    Wire,
    render,
)

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
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
        name: Final = request.path_params["model"]
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
            scenario: Final = Scenario.model_validate_json(await request.body())
        except ValidationError as exc:
            return self._render(
                RenderedResponse(400, "application/json", json.dumps({"error": str(exc)}).encode("utf-8"))
            )
        self.scenario_store.put(scenario)
        return self._render(
            RenderedResponse(
                200,
                "application/json",
                json.dumps({"scenario_id": scenario.scenario_id}).encode("utf-8"),
            )
        )

    async def delete_scenario(self, request: Request) -> Response:
        scenario_id: Final = cast(str, request.path_params["scenario_id"])
        deleted: Final = self.scenario_store.drop(scenario_id)
        return self._render(
            RenderedResponse(
                200 if deleted else 404,
                "application/json",
                json.dumps({"deleted": deleted}).encode("utf-8"),
            )
        )

    async def cost_map(self, _request: Request) -> Response:
        return self._render(
            RenderedResponse(
                200,
                "application/json",
                (Path(__file__).resolve().parents[1] / "cost_calculation" / "cost_map.json").read_bytes(),
            )
        )

    async def oauth_token(self, _request: Request) -> Response:
        return self._render(
            RenderedResponse(
                200,
                "application/json",
                json.dumps(
                    {
                        "access_token": "scripted-token",
                        "token_type": "Bearer",
                        "expires_in": 3600,
                    }
                ).encode("utf-8"),
            )
        )

    async def scripted(self, request: Request) -> Response:
        rendered: Final = render(
            self.scenario_store,
            request.method,
            request.url.path,
            await request.body(),
        )
        return self._render(rendered)

    @staticmethod
    def _render(rendered: RenderedResponse) -> Response:
        return Response(
            content=rendered.body,
            status_code=rendered.status_code,
            media_type=rendered.content_type,
        )

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
                Route("/{scenario_id}/{tail:path}", self.scripted, methods=["POST"]),
            ]
        )


CONTROL_URL: Final = os.environ.get("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:8190").rstrip("/")


@dataclass(frozen=True, slots=True)
class ScenarioHandle:
    scenario_id: str
    wire: Wire
    control_url: str

    def api_base(self) -> str:
        return f"{self.control_url}/{self.scenario_id}/{self._mount()}"

    def _mount(self) -> str:
        return WIRE_MOUNTS[self.wire]


def register_scenario(scenario: Scenario) -> ScenarioHandle:
    response: Final = httpx.post(
        f"{CONTROL_URL}/__scenarios",
        json=scenario.model_dump(mode="json"),
        trust_env=False,
        timeout=15,
    )
    response.raise_for_status()
    result: Final = ScenarioRegistered.model_validate_json(response.content)
    return ScenarioHandle(
        scenario_id=result.scenario_id,
        wire=scenario.wire,
        control_url=CONTROL_URL,
    )


def delete_scenario(handle: ScenarioHandle) -> None:
    response: Final = httpx.delete(
        f"{CONTROL_URL}/__scenarios/{handle.scenario_id}",
        trust_env=False,
        timeout=15,
    )
    response.raise_for_status()
    ScenarioDeleted.model_validate_json(response.content)


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8190)
    arguments: Final = parser.parse_args()
    uvicorn.run(Provider().app(), host="127.0.0.1", port=cast(int, arguments.port), access_log=False)


if __name__ == "__main__":
    main()
