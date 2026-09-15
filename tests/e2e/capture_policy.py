from __future__ import annotations

import base64
import binascii
import hashlib
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from fixture_bundle import RecordedHttpResponse, RecordedResponse, RecordedStreamedResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
SCENARIO_BYTES: Final = 5 * 1024 * 1024
RUN_BYTES: Final = 20 * 1024 * 1024
SOFT_AGE_SECONDS: Final = 24 * 60 * 60
HARD_AGE_SECONDS: Final = 7 * SOFT_AGE_SECONDS


class RequestBudget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    model: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0, le=4096)
    max_request_bytes: int = Field(default=65536, gt=0, le=65536)

    def error(self, body: bytes | None) -> str | None:
        if body is None or len(body) > self.max_request_bytes:
            return "capture request exceeds its byte budget"
        try:
            payload: Final = JSON_OBJECT.validate_json(body)
        except ValidationError:
            return "capture request is not JSON"
        if payload.get("model") != self.model or payload.get("n", 1) != 1:
            return "capture request changes its approved model or completion count"
        limits: Final = tuple(payload[key] for key in ("max_tokens", "max_completion_tokens") if key in payload)
        if not limits or any(
            not isinstance(limit, int) or isinstance(limit, bool) or not 0 < limit <= self.max_output_tokens
            for limit in limits
        ):
            return "capture request exceeds its output-token budget"
        return None


def canonical_scenario_id(node: str) -> str:
    path, separator, test = node.partition("::")
    relative: Final = path.split("tests/e2e/", 1)[-1]
    if (
        not separator
        or not test
        or not relative.endswith(".py")
        or PurePosixPath(relative).is_absolute()
        or ".." in PurePosixPath(relative).parts
        or "\\" in path
        or "\x00" in node
    ):
        raise ValueError("scenario must name an E2E test node under tests/e2e")
    return f"tests/e2e/{relative}::{test}"


@dataclass(frozen=True, slots=True)
class ScenarioIdentity:
    node: str
    contract_sha256: str
    credential_profile: str
    matcher_version: str = "stateless_v1"

    def __post_init__(self) -> None:
        canonical_scenario_id(self.node)
        if len(self.contract_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.contract_sha256):
            raise ValueError("scenario contract must be a SHA-256 digest")
        if not self.credential_profile or self.matcher_version != "stateless_v1":
            raise ValueError("capture requires a credential profile and stateless_v1 matcher")

    @property
    def key(self) -> str:
        return hashlib.sha256(
            json.dumps(
                (
                    canonical_scenario_id(self.node),
                    self.contract_sha256,
                    self.credential_profile,
                    self.matcher_version,
                ),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ScenarioOutcome:
    setup: bool = False
    call: bool = False
    teardown: bool = False


def _invalid_count(value: JsonValue) -> bool:
    return not isinstance(value, int) or isinstance(value, bool) or value < 0


def _usage_error(value: JsonValue) -> str | None:
    if not isinstance(value, dict):
        return "invalid usage object"
    counts: Final = tuple(value.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    if any(key in value for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
        if not all(type(count) is int and count >= 0 for count in counts):
            return "invalid token usage"
        prompt, completion, total = counts
        if isinstance(prompt, int) and isinstance(completion, int) and total != prompt + completion:
            return "inconsistent token usage"
    if any(_invalid_count(value[key]) for key in ("input_tokens", "output_tokens") if key in value):
        return "invalid token usage"
    return None


def _json_error(body: bytes, *, streaming: bool = False) -> str | None:
    try:
        value: Final = JSON_OBJECT.validate_json(body)
    except ValidationError:
        return "invalid response JSON"
    if not value or "error" in value or value.get("type") == "error":
        return "provider response contains an error"
    if value.get("status") in ("incomplete", "failed", "cancelled"):
        return "provider response did not complete"
    if streaming and value.get("usage") is None:
        return None
    usage_error: Final = _usage_error(value["usage"]) if "usage" in value else None
    if usage_error is not None or streaming:
        return usage_error
    return _completion_error(value)


def _completion_error(value: dict[str, JsonValue]) -> str | None:
    if not all(isinstance(value.get(key), str) and value[key] for key in ("id", "model")):
        return "completion lacks provider identity"
    usage: Final = value.get("usage")
    if not isinstance(usage, dict):
        return "completion lacks token usage"
    if value.get("object") == "chat.completion":
        choices: Final = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            return "completion requires exactly one choice"
        choice: Final = choices[0]
        message: Final = choice.get("message")
        if (
            type(choice.get("index")) is not int
            or choice["index"] != 0
            or choice.get("finish_reason") != "stop"
            or not isinstance(message, dict)
            or message.get("role") != "assistant"
            or not isinstance(message.get("content"), str)
        ):
            return "completion lacks a finished assistant message"
        if not all(key in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens")):
            return "completion lacks token usage"
        return None
    if value.get("type") == "message":
        content: Final = value.get("content")
        if (
            value.get("role") != "assistant"
            or value.get("stop_reason") not in ("end_turn", "stop_sequence")
            or not isinstance(content, list)
            or not content
            or any(
                not isinstance(block, dict) or block.get("type") != "text" or not isinstance(block.get("text"), str)
                for block in content
            )
        ):
            return "completion lacks finished message content"
        if not all(key in usage for key in ("input_tokens", "output_tokens")):
            return "completion lacks token usage"
        return None
    return "unsupported completion response"


def _stream_error(response: RecordedStreamedResponse) -> str | None:
    if response.truncated is not None:
        return "stream was truncated"
    try:
        body: Final = b"".join(base64.b64decode(chunk, validate=True) for chunk in response.chunks_b64).decode()
    except (ValueError, UnicodeError, binascii.Error):
        return "invalid stream encoding"
    normalized: Final = body.replace("\r\n", "\n")
    if not normalized.endswith("\n\n"):
        return "stream lacks complete event framing"
    events: Final = tuple(
        data
        for event in normalized.split("\n\n")
        if (data := "\n".join(line[5:].lstrip(" ") for line in event.splitlines() if line.startswith("data:")))
    )
    if not events or any(not data for data in events):
        return "stream contains no data event"
    if "[DONE]" in events[:-1]:
        return "stream has events after completion"
    errors: Final = tuple(_json_error(data.encode(), streaming=True) for data in events if data != "[DONE]")
    if any(errors):
        return next(error for error in errors if error is not None)
    parsed: Final = tuple(JSON_OBJECT.validate_json(data) for data in events if data != "[DONE]")
    if not parsed:
        return "stream contains no completion payload"
    if events[-1] == "[DONE]":
        if any(
            not isinstance(entries := value.get("choices"), list)
            or any(not isinstance(entry, dict) for entry in entries)
            for value in parsed
        ):
            return "stream contains invalid choices"
        choices: Final = tuple(
            entry
            for value in parsed
            if isinstance(entries := value.get("choices"), list)
            for entry in entries
            if isinstance(entry, dict)
        )
        if not choices or choices[-1].get("finish_reason") not in ("stop", "tool_calls", "function_call"):
            return "stream lacks a successful completion reason"
        if any(choice.get("index", 0) != 0 for choice in choices):
            return "stream changes its approved completion count"
        if any(choice.get("finish_reason") is not None for choice in choices[:-1]):
            return "stream has choice events after completion"
        return None
    if parsed[-1].get("type") != "message_stop" or any(v.get("type") == "message_stop" for v in parsed[:-1]):
        return "stream lacks a final success terminator"
    if parsed[0].get("type") != "message_start":
        return "stream lacks message start"
    deltas: Final = tuple(value.get("delta") for value in parsed if value.get("type") == "message_delta")
    if not any(
        isinstance(delta, dict) and delta.get("stop_reason") in ("end_turn", "tool_use", "stop_sequence")
        for delta in deltas
    ):
        return "stream lacks a successful completion reason"
    return None


def _response_error(response: RecordedResponse) -> str | None:
    if not 200 <= response.status_code < 300:
        return "provider response was not successful"
    if isinstance(response, RecordedHttpResponse):
        try:
            return _json_error(base64.b64decode(response.body_b64, validate=True))
        except (ValueError, binascii.Error):
            return "invalid response encoding"
    return _stream_error(response)


def publication_error(outcome: ScenarioOutcome, responses: tuple[RecordedResponse, ...]) -> str | None:
    if not (outcome.setup and outcome.call and outcome.teardown):
        return "trusted scenario setup, call and teardown must all pass"
    if not responses:
        return "capture contains no interactions"
    if sum(len(response.model_dump_json().encode()) for response in responses) > SCENARIO_BYTES:
        return "scenario exceeds capture byte limit"
    return next((error for response in responses if (error := _response_error(response)) is not None), None)
