import base64
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Mapping

from pydantic import TypeAdapter, ValidationError

from litellm.proxy.experimental_codex_gateway.pipeline import PipelineResult
from litellm.proxy.experimental_codex_gateway.types import JsonValue

_SENSITIVE_KEYS = frozenset(
    {
        "account",
        "account_id",
        "api_key",
        "authorization",
        "chatgpt-account-id",
        "cookie",
        "local_key",
        "password",
        "secret",
        "token",
        "x-litellm-api-key",
    }
)
_TEXT_SECRET_PATTERNS = (
    re.compile(r"(?i)bearer\s+[^\s\"',]+"),
    re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}\b"),
    re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    re.compile(r"(?<!\w)(?:\+?\d[\d .()-]{7,}\d)(?!\w)"),
    re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"),
)
_BYTE_SECRET_PATTERNS = tuple(
    re.compile(pattern.pattern.encode(), pattern.flags & ~re.UNICODE) for pattern in _TEXT_SECRET_PATTERNS
)
_BYTE_SENSITIVE_FIELD = re.compile(
    rb'(?i)("(?:account|account_id|api_key|authorization|chatgpt-account-id|cookie|local_key|password|secret|token|x-litellm-api-key)"\s*:\s*")([^"\\]*)(")'
)
_TRACE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_TRACE_ADAPTER = TypeAdapter(dict[str, JsonValue])
_REQUEST_HEADER_ALLOWLIST = frozenset(
    {"accept", "content-type", "originator", "traceparent", "tracestate", "user-agent"}
)
_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "content-type",
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    }
)


def _masked_text(value: str, local_key: str) -> str:
    key_redacted = value.replace(local_key, "*" * len(local_key)) if local_key else value
    result = key_redacted
    for pattern in _TEXT_SECRET_PATTERNS:
        result = pattern.sub(lambda match: "*" * len(match.group(0)), result)
    return result


def _redacted_json(value: JsonValue, local_key: str) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key.lower() in _SENSITIVE_KEYS else _redacted_json(child, local_key)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_redacted_json(child, local_key) for child in value]
    if isinstance(value, str):
        return _masked_text(value, local_key)
    return value


def redact_json_bytes(body: bytes, local_key: str, limit: int) -> JsonValue:
    if len(body) > limit:
        return {"omitted": True, "truncated": True}
    bounded_body = body[:limit]
    try:
        parsed = _JSON_ADAPTER.validate_json(bounded_body)
    except ValidationError:
        return {
            "base64": base64.b64encode(redact_bytes(bounded_body, local_key)).decode(),
            "truncated": len(body) > limit,
        }
    return _redacted_json(parsed, local_key)


def redact_bytes(value: bytes, local_key: str) -> bytes:
    key_bytes = local_key.encode()
    result = value.replace(key_bytes, b"*" * len(key_bytes)) if key_bytes else value
    for pattern in _BYTE_SECRET_PATTERNS:
        result = pattern.sub(lambda match: b"*" * len(match.group(0)), result)
    return _BYTE_SENSITIVE_FIELD.sub(
        lambda match: match.group(1) + (b"*" * len(match.group(2))) + match.group(3),
        result,
    )


def _safe_headers(
    headers: tuple[tuple[bytes, bytes], ...], allowlist: frozenset[str], local_key: str
) -> dict[str, JsonValue]:
    return {
        key.decode("latin-1").lower(): _masked_text(value.decode("latin-1"), local_key)
        for key, value in headers
        if key.decode("latin-1").lower() in allowlist
    }


@dataclass(frozen=True, slots=True)
class ResponseChunk:
    offset_ms: int
    body: bytes


class TraceRecorder:
    def __init__(
        self,
        method: str,
        path: str,
        query_string: bytes,
        request_headers: tuple[tuple[bytes, bytes], ...],
        pipeline_result: PipelineResult,
        local_key: str,
        max_trace_bytes: int,
    ) -> None:
        self.trace_id = uuid.uuid4().hex
        self._started = time.monotonic()
        self._method = method
        self._path = path
        self._query_hash = hashlib.sha256(query_string).hexdigest()
        self._request_headers = request_headers
        self._pipeline_result = pipeline_result
        self._local_key = local_key
        self._capture_budget = max_trace_bytes // 3
        self._captured_bytes = 0
        self._response_status = 500
        self._response_headers: tuple[tuple[bytes, bytes], ...] = ()
        self._response_chunks: list[ResponseChunk] = []
        self._response_hash = hashlib.sha256()
        self._first_byte_seconds: float | None = None
        self._truncated = False

    def response_start(self, status: int, headers: tuple[tuple[bytes, bytes], ...]) -> None:
        self._response_status = status
        self._response_headers = headers

    def response_body(self, body: bytes) -> None:
        self._response_hash.update(body)
        elapsed = time.monotonic() - self._started
        if body and self._first_byte_seconds is None:
            self._first_byte_seconds = elapsed
        remaining = self._capture_budget - self._captured_bytes
        captured = body[: max(remaining, 0)]
        if captured:
            self._response_chunks.append(ResponseChunk(offset_ms=round(elapsed * 1000), body=captured))
            self._captured_bytes += len(captured)
        self._truncated = self._truncated or len(captured) != len(body)

    @property
    def first_byte_seconds(self) -> float | None:
        return self._first_byte_seconds

    def export(self) -> dict[str, JsonValue]:
        elapsed = time.monotonic() - self._started
        joined_response = b"".join(chunk.body for chunk in self._response_chunks)
        redacted_response = b"" if self._truncated else redact_bytes(joined_response, self._local_key)
        chunk_lengths = tuple(len(chunk.body) for chunk in self._response_chunks)
        boundaries = tuple(sum(chunk_lengths[:index]) for index in range(len(chunk_lengths) + 1))
        response_chunks: list[JsonValue] = (
            []
            if self._truncated
            else [
                {
                    "offset_ms": chunk.offset_ms,
                    "base64": base64.b64encode(redacted_response[boundaries[index] : boundaries[index + 1]]).decode(),
                }
                for index, chunk in enumerate(self._response_chunks)
            ]
        )
        request_limit = self._capture_budget
        pipeline = self._pipeline_result
        request: dict[str, JsonValue] = {
            "method": self._method,
            "path": _masked_text(self._path, self._local_key),
            "query_sha256": self._query_hash,
            "headers": _safe_headers(self._request_headers, _REQUEST_HEADER_ALLOWLIST, self._local_key),
            "body_sha256": hashlib.sha256(pipeline.original_body).hexdigest(),
            "body": redact_json_bytes(pipeline.original_body, self._local_key, request_limit),
            "forwarded_body_sha256": hashlib.sha256(pipeline.body).hexdigest(),
        }
        stage_audits: list[JsonValue] = [
            {
                "stage": _masked_text(audit.stage, self._local_key),
                "outcome": audit.outcome.value,
                "findings": [_masked_text(finding, self._local_key) for finding in audit.findings],
            }
            for audit in pipeline.audit
        ]
        pipeline_details: dict[str, JsonValue] = {
            "outcome": pipeline.outcome.value,
            "stages": stage_audits,
        }
        response: dict[str, JsonValue] = {
            "status": self._response_status,
            "headers": _safe_headers(self._response_headers, _RESPONSE_HEADER_ALLOWLIST, self._local_key),
            "body_sha256": self._response_hash.hexdigest(),
            "chunks": response_chunks,
            "truncated": self._truncated,
        }
        timing: dict[str, JsonValue] = {
            "first_byte_ms": None if self._first_byte_seconds is None else round(self._first_byte_seconds * 1000),
            "total_ms": round(elapsed * 1000),
        }
        return {
            "schema": "litellm-codex-gateway.trace.v1",
            "trace_id": self.trace_id,
            "request": request,
            "pipeline": pipeline_details,
            "response": response,
            "timing": timing,
        }


@dataclass(frozen=True, slots=True)
class TraceStore:
    directory: Path
    max_trace_bytes: int
    max_storage_bytes: int
    retention_seconds: int
    _lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def write(self, trace: Mapping[str, JsonValue]) -> bool:
        with self._lock:
            return self._write(trace)

    def _write(self, trace: Mapping[str, JsonValue]) -> bool:
        trace_id = trace.get("trace_id")
        if (
            trace.get("schema") != "litellm-codex-gateway.trace.v1"
            or not isinstance(trace_id, str)
            or _TRACE_ID_PATTERN.fullmatch(trace_id) is None
        ):
            return False
        encoded = json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        if len(encoded) > self.max_trace_bytes:
            return False
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)
        with tempfile.NamedTemporaryFile(mode="wb", dir=self.directory, delete=False) as temporary:
            temporary.write(encoded)
            temporary_path = Path(temporary.name)
        os.chmod(temporary_path, 0o600)
        temporary_path.replace(self.directory / f"{trace_id}.json")
        self._evict()
        return True

    def read(self, trace_id: str) -> bytes | None:
        with self._lock:
            return self._read(trace_id)

    def _read(self, trace_id: str) -> bytes | None:
        if _TRACE_ID_PATTERN.fullmatch(trace_id) is None:
            return None
        path = self.directory / f"{trace_id}.json"
        try:
            if path.stat().st_size > self.max_trace_bytes:
                return None
            encoded = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            trace = _TRACE_ADAPTER.validate_json(encoded)
        except ValidationError:
            return None
        if trace.get("schema") != "litellm-codex-gateway.trace.v1" or trace.get("trace_id") != trace_id:
            return None
        return encoded

    def evict(self) -> None:
        with self._lock:
            self._evict()

    def _evict(self) -> None:
        now = time.time()
        files = tuple(
            sorted(
                (path for path in self.directory.glob("*.json") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
            )
        )
        retained = tuple(path for path in files if now - path.stat().st_mtime <= self.retention_seconds)
        expired = tuple(path for path in files if path not in retained)
        for path in expired:
            path.unlink(missing_ok=True)
        total = sum(path.stat().st_size for path in retained)
        for path in retained:
            if total <= self.max_storage_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size


def _decode_response_chunk(value: JsonValue) -> bytes | None:
    if not isinstance(value, dict):
        return None
    encoded = value.get("base64")
    if not isinstance(encoded, str):
        return None
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError:
        return None


def replay_response_chunks(trace: bytes) -> tuple[bytes, ...] | None:
    try:
        payload = _TRACE_ADAPTER.validate_json(trace)
    except ValidationError:
        return None
    if payload.get("schema") != "litellm-codex-gateway.trace.v1":
        return None
    response = payload.get("response")
    if not isinstance(response, dict):
        return None
    chunks = response.get("chunks")
    if not isinstance(chunks, list):
        return None
    decoded = tuple(_decode_response_chunk(chunk) for chunk in chunks)
    if any(chunk is None for chunk in decoded):
        return None
    return tuple(chunk for chunk in decoded if chunk is not None)
