from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias, assert_never
from urllib.parse import urlsplit

import psutil
import psycopg
from integration._support import responses_vendor as rv
from integration._support.client import eventually, object_value, string_value
from integration._support.database import ROWS
from integration._support.openai_wire import answering_model_discovery, responses_reply
from integration._support.wire import Reply, Request, Wire
from psycopg.rows import DictRow, dict_row
from pydantic import JsonValue

MODEL: Final = "openai/responses/gpt-6.1-sol"
EXPLICIT: Final[Mapping[str, JsonValue]] = {"mode": "explicit"}
EXPLICIT_30M: Final[Mapping[str, JsonValue]] = {"mode": "explicit", "ttl": "30m"}
NO_CACHE: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
INJECTION: Final[Mapping[str, JsonValue]] = {
    "cache_control_injection_points": [{"location": "message", "role": "system"}],
    "prompt_cache_options": {"mode": "explicit"},
}
_SCRIPTED_FAILURE: Final = re.compile(r"fail-(\d{3})")
_MINTED_RESPONSE: Final = re.compile(r"^resp_([0-9a-f]{32})-[0-9a-f]{32}$")
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")

Kind: TypeAlias = Literal["text", "image_url", "file", "video_url"]
KINDS: Final[tuple[Kind, ...]] = ("text", "image_url", "file", "video_url")
WIRE_TYPE: Final[Mapping[Kind, str]] = {
    "text": "input_text",
    "image_url": "input_image",
    "file": "input_file",
    "video_url": "input_text",
}


def _scripted(request: Request) -> Reply:
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    text: Final = request.body.decode()
    marker: Final = rv.newest_marker(text)
    failure: Final = _SCRIPTED_FAILURE.search(text)
    if failure is not None:
        return rv.error(int(failure.group(1)), f"scripted {failure.group(1)} marker-{marker}", "scripted_failure")
    return responses_reply(
        f"resp_{marker or uuid.uuid4().hex}-{uuid.uuid4().hex}",
        string_value(body["model"]),
        rv.answer(marker),
        stream=body.get("stream") is True,
    )


respond: Final = answering_model_discovery(_scripted)


def response_marker(identity: str) -> str | None:
    minted: Final = tuple(
        found for candidate in rv.response_identities(identity) if (found := _MINTED_RESPONSE.match(candidate))
    )
    return minted[0].group(1) if minted else None


def answers(identity: str, marker: str) -> bool:
    return response_marker(identity) == marker


def prompt(marker: str) -> str:
    return f"Say marker-{marker}"


def text(value: str) -> dict[str, JsonValue]:
    return {"type": "text", "text": value}


def marked(block: Mapping[str, JsonValue], marker: JsonValue) -> dict[str, JsonValue]:
    return {**block, "prompt_cache_breakpoint": marker}


def block(kind: Kind, value: str) -> dict[str, JsonValue]:
    match kind:
        case "text":
            return text(value)
        case "image_url":
            return {"type": "image_url", "image_url": {"url": "https://example.com/breakpoint.png"}}
        case "file":
            return {"type": "file", "file": {"file_id": "file-breakpoint"}}
        case "video_url":
            return {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
        case _:
            assert_never(kind)


AUDIO_PAYLOAD: Final[Mapping[str, JsonValue]] = {"data": "Zm9v", "format": "wav"}


def audio() -> dict[str, JsonValue]:
    return {"type": "input_audio", "input_audio": dict(AUDIO_PAYLOAD)}


def drained_posts(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


def with_marker(posts: Sequence[Request], marker: str) -> tuple[Request, ...]:
    return tuple(request for request in posts if f"marker-{marker}" in request.body.decode())


def posted(wire: Wire, marker: str) -> Request:
    matching: Final = with_marker(drained_posts(wire), marker)
    assert len(matching) == 1, [request.body for request in matching]
    (request,) = matching
    assert request.target == "/v1/responses", request.target
    return request


def body_of(request: Request) -> dict[str, JsonValue]:
    return rv.JSON_OBJECT.validate_json(request.body)


def input_items(request: Request) -> list[dict[str, JsonValue]]:
    return rv.ITEMS.validate_python(body_of(request)["input"])


def content_of(items: Sequence[Mapping[str, JsonValue]], role: str) -> list[dict[str, JsonValue]]:
    messages: Final = tuple(item for item in items if item.get("type") == "message" and item.get("role") == role)
    assert len(messages) == 1, items
    return rv.ITEMS.validate_python(messages[0]["content"])


def single_block(items: Sequence[Mapping[str, JsonValue]], role: str) -> dict[str, JsonValue]:
    blocks: Final = content_of(items, role)
    assert len(blocks) == 1, blocks
    return blocks[0]


def instruction_block(items: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    messages: Final = tuple(
        item for item in items if item.get("type") == "message" and item.get("role") in ("system", "developer")
    )
    assert len(messages) == 1, items
    blocks: Final = rv.ITEMS.validate_python(messages[0]["content"])
    assert len(blocks) == 1, blocks
    return blocks[0]


def function_output(items: Sequence[Mapping[str, JsonValue]], call_id: str) -> list[dict[str, JsonValue]]:
    outputs: Final = tuple(
        item for item in items if item.get("type") == "function_call_output" and item.get("call_id") == call_id
    )
    assert len(outputs) == 1, items
    return rv.ITEMS.validate_python(outputs[0]["output"])


def assert_marker(block_on_wire: Mapping[str, JsonValue], expected: JsonValue) -> None:
    if expected is None:
        assert "prompt_cache_breakpoint" not in block_on_wire, block_on_wire
        return
    assert block_on_wire.get("prompt_cache_breakpoint") == expected, block_on_wire


@dataclass(frozen=True, slots=True)
class SpendLogs:
    connection: psycopg.Connection[DictRow]

    def rows_for(self, model: str) -> list[dict[str, JsonValue]]:
        cursor: Final = self.connection.execute(
            'SELECT litellm_call_id, request_id, status FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model,)
        )
        return ROWS.validate_python(cursor.fetchall())

    def landed(
        self, model: str, call_id: str, marker: str | None, *, status: str = "success", seconds: float = 70
    ) -> dict[str, JsonValue]:
        rows: Final = eventually(
            lambda: self.rows_for(model),
            lambda found: any(row["litellm_call_id"] == call_id for row in found),
            seconds=seconds,
        )
        matching: Final = tuple(row for row in rows if row["litellm_call_id"] == call_id)
        assert len(matching) == 1, rows
        (row,) = matching
        assert row["status"] == status, row
        assert marker is None or answers(string_value(row["request_id"]), marker), (row, marker)
        return row


@contextmanager
def spend_logs() -> Iterator[SpendLogs]:
    with psycopg.connect(os.environ["DATABASE_URL"], row_factory=dict_row, autocommit=True) as connection:
        connection.execute("SET default_transaction_read_only = on")
        yield SpendLogs(connection)


def model_id(entries: Sequence[JsonValue], model: str) -> str:
    matching: Final = tuple(entry for entry in entries if object_value(entry).get("model_name") == model)
    assert len(matching) == 1, entries
    return string_value(object_value(object_value(matching[0])["model_info"])["id"])


def started_worker_pids(log: Path) -> tuple[int, ...]:
    return tuple(int(found.group(1)) for found in _STARTED_WORKER.finditer(log.read_text()))


def open_upstream_connections(pid: int, upstream: str) -> int:
    port: Final = urlsplit(upstream).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )
