from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Final

import uvicorn
from _fake_openai_endpoint_server import (
    chat_completions,
    completions,
    health,
    moderations,
)
from _fake_openai_endpoint_server import (
    embeddings as fake_embeddings,
)
from pydantic import JsonValue, TypeAdapter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

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

    async def embeddings(self, request: Request) -> Response:
        body: Final = JSON_OBJECT.validate_json(await request.body())
        self.observations.put(Observation(request.url.path, request.headers.get("authorization", ""), body))
        return await fake_embeddings(request)

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

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/health", health),
                Route("/__observations", self.observed),
                Route("/__scripts/{model}", self.script, methods=["POST", "DELETE", "GET"]),
                Route("/v1/chat/completions", self.chat, methods=["POST"]),
                Route("/v1/completions", completions, methods=["POST"]),
                Route("/v1/embeddings", self.embeddings, methods=["POST"]),
                Route("/v1/moderations", moderations, methods=["POST"]),
            ]
        )


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8190)
    arguments: Final = parser.parse_args()
    uvicorn.run(Provider().app(), host="127.0.0.1", port=arguments.port, access_log=False)


if __name__ == "__main__":
    main()
