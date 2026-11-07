"""Scripted A2A agents shaped like the servers litellm proxies to, for the a2a_protocol_version cells.

The ``langgraph`` dialect reproduces langgraph-api 0.15.4 byte for byte where it matters: its card declares
``protocolBinding: "JSONRPC"`` with ``protocolVersion: "1.0"``, and every reply, whether the caller used the
0.3 ``message/send`` or the v1 ``SendMessage`` method, is in the 0.3 ``kind``-discriminated dialect (a v1
``SendMessage`` reply wraps the 0.3 task under ``result.task``, as langgraph-api does). ``v1`` is a genuine
A2A 1.0 agent that answers ``SendMessage`` and ``SendStreamingMessage`` in the proto JSON dialect and rejects
the 0.3 method names.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final, Literal

from integration._support.client import Gateway
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

Dialect = Literal["langgraph", "v1"]
LEGACY_METHODS: Final = frozenset({"message/send", "message/stream"})
V1_METHODS: Final = frozenset({"SendMessage", "SendStreamingMessage"})
HINT: Final = (
    "(the agent replied in the A2A 0.3 dialect while its card declares 1.0; "
    'set litellm_params.a2a_protocol_version: "0.3" on this agent)'
)
KIND_ERROR: Final = 'has no field named "kind"'


@dataclass(slots=True)
class Peer:
    """One scripted agent: ``card`` is what the card endpoint serves, ``dialect`` how the POSTs answer."""

    marker: str
    dialect: Dialect
    interfaces: list[dict[str, str]] | None = None
    card_url: str | None = None
    error: Mapping[str, JsonValue] | None = None
    post_status: int = 200
    card_status: int = 200
    wire: Wire | None = None
    extra_targets: frozenset[str] = field(default_factory=frozenset)

    @property
    def url(self) -> str:
        assert self.wire is not None
        return self.wire.url + "/a2a/" + self.marker

    def card(self) -> dict[str, JsonValue]:
        interfaces: Final = self.interfaces if self.interfaces is not None else [{"protocolBinding": "JSONRPC"}]
        return {
            "name": self.marker,
            "description": "Synthetic langgraph-api 0.15 echo assistant",
            "url": self.card_url or self.url,
            "supportedInterfaces": [
                {
                    "url": interface.get("url", self.card_url or self.url),
                    "protocolBinding": interface["protocolBinding"],
                    "protocolVersion": interface.get("protocolVersion", "1.0"),
                }
                for interface in interfaces
            ],
            "capabilities": {"streaming": True, "pushNotifications": False},
            "defaultInputModes": ["application/json", "text/plain"],
            "defaultOutputModes": ["application/json", "text/plain"],
            "skills": [],
            "version": "0.15.4",
        }

    def respond(self, request: Request) -> Reply:
        if request.method == "GET":
            if self.card_status != 200:
                return Reply(status=self.card_status, body=b'{"detail": "no card here"}')
            return Reply(body=json.dumps(self.card()).encode())
        if self.post_status != 200:
            return Reply(status=self.post_status, body=b'{"detail": "scripted failure"}')
        body: Final = json.loads(request.body)
        method: Final = body["method"]
        if self.error is not None:
            return _json({"jsonrpc": "2.0", "id": body["id"], "error": dict(self.error)})
        if self.dialect == "v1" and method not in V1_METHODS:
            return _json({"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": "Method not found"}})
        text: Final = _input_text(body["params"]["message"])
        reply: Final = "echo: " + text
        if method in {"message/stream", "SendStreamingMessage"}:
            frames: Final = (
                self._status_frame(body["id"], reply, final=False),
                self._artifact_frame(body["id"], reply),
                self._status_frame(body["id"], None, final=True),
            )
            return Reply(
                content_type="text/event-stream",
                chunks=tuple(f"event: message\ndata: {json.dumps(frame)}\n\n".encode() for frame in frames),
            )
        task: Final = self._task(reply, text, v1=self.dialect == "v1")
        result: Final = {"task": task} if method == "SendMessage" else task
        return _json({"jsonrpc": "2.0", "id": body["id"], "result": result})

    def _ids(self) -> tuple[str, str]:
        return self.marker + "-task", self.marker + "-context"

    def _task(self, reply: str, text: str, *, v1: bool) -> dict[str, JsonValue]:
        task_id, context_id = self._ids()
        if v1:
            return {
                "id": task_id,
                "contextId": context_id,
                "status": {"state": "TASK_STATE_COMPLETED"},
                "artifacts": [{"artifactId": self.marker + "-artifact", "parts": [{"text": reply}]}],
                "history": [{"role": "ROLE_USER", "messageId": self.marker + "-in", "parts": [{"text": text}]}],
            }
        return {
            "kind": "task",
            "id": task_id,
            "contextId": context_id,
            "status": {"state": "completed"},
            "artifacts": [{"artifactId": self.marker + "-artifact", "parts": [{"kind": "text", "text": reply}]}],
            "history": [
                {
                    "kind": "message",
                    "role": "user",
                    "messageId": self.marker + "-in",
                    "parts": [{"kind": "text", "text": text}],
                }
            ],
        }

    def _status_frame(self, request_id: JsonValue, reply: str | None, *, final: bool) -> dict[str, JsonValue]:
        task_id, context_id = self._ids()
        if self.dialect == "v1":
            status: dict[str, JsonValue] = {"state": "TASK_STATE_COMPLETED" if final else "TASK_STATE_WORKING"}
            if reply is not None:
                status["message"] = {
                    "role": "ROLE_AGENT",
                    "messageId": self.marker + "-out",
                    "parts": [{"text": reply}],
                }
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"statusUpdate": {"taskId": task_id, "contextId": context_id, "status": status}},
            }
        status = {"state": "completed" if final else "working"}
        if reply is not None:
            status["message"] = {
                "kind": "message",
                "role": "agent",
                "messageId": self.marker + "-out",
                "parts": [{"kind": "text", "text": reply}],
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "kind": "status-update",
                "taskId": task_id,
                "contextId": context_id,
                "status": status,
                "final": final,
            },
        }

    def _artifact_frame(self, request_id: JsonValue, reply: str) -> dict[str, JsonValue]:
        task_id, context_id = self._ids()
        if self.dialect == "v1":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "artifactUpdate": {
                        "taskId": task_id,
                        "contextId": context_id,
                        "artifact": {"artifactId": self.marker + "-artifact", "parts": [{"text": reply}]},
                        "lastChunk": True,
                    }
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "kind": "artifact-update",
                "taskId": task_id,
                "contextId": context_id,
                "artifact": {"artifactId": self.marker + "-artifact", "parts": [{"kind": "text", "text": reply}]},
                "lastChunk": True,
            },
        }


def _json(value: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(value).encode())


def _input_text(message: Mapping[str, JsonValue]) -> str:
    parts: Final = message["parts"]
    assert isinstance(parts, list) and parts, message
    first: Final = parts[0]
    assert isinstance(first, dict), message
    return str(first["text"])


@contextmanager
def peer(marker_prefix: str, dialect: Dialect, **fields: object) -> Iterator[Peer]:
    agent: Final = Peer(marker=marker_prefix + uuid.uuid4().hex, dialect=dialect, **fields)  # type: ignore[arg-type]
    with wire_server(agent.respond) as wire:
        agent.wire = wire
        yield agent


def register(
    gateway: Gateway,
    cleanups: Callable[..., object],
    name: str,
    url: str,
    litellm_params: Mapping[str, JsonValue] | None,
) -> str:
    body: dict[str, JsonValue] = {
        "agent_name": name,
        "agent_card_params": {
            "protocolVersion": "1.0",
            "name": name,
            "description": "Synthetic langgraph-api 0.15 echo assistant",
            "version": "0.15.4",
            "url": url,
            "capabilities": {"streaming": True},
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "skills": [],
        },
    }
    if litellm_params is not None:
        body["litellm_params"] = dict(litellm_params)
    created: Final = gateway.request("POST", "/v1/agents", body)
    assert created.status_code == 200, created.text
    identity: Final = str(created.json()["agent_id"])

    def delete() -> None:
        deleted: Final = gateway.request("DELETE", f"/v1/agents/{identity}")
        assert deleted.status_code == 200, deleted.text
        assert read_rows('SELECT agent_id FROM "LiteLLM_AgentsTable" WHERE agent_id=%s', (identity,)) == []

    cleanups(delete)
    return identity


def legacy_send(marker: str, text: str = "synthetic ping", method: str = "message/send") -> dict[str, JsonValue]:
    return {
        "jsonrpc": "2.0",
        "id": marker,
        "method": method,
        "params": {
            "message": {
                "kind": "message",
                "role": "user",
                "messageId": marker + "-in",
                "parts": [{"kind": "text", "text": text}],
            }
        },
    }


def posts(wire: Wire) -> tuple[dict[str, JsonValue], ...]:
    return tuple(json.loads(item.body) for item in wire.drain() if item.method == "POST")


def sse_frames(text: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line[6:].strip()
    )


def spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, model, call_type, status FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)
    )
