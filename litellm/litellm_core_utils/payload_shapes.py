from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias, TypeVar, cast

from pydantic import BaseModel

from litellm.rust_bridge import forwarding

Stage: TypeAlias = Literal[
    "litellm.request.received",
    "provider.request.transformed",
    "provider.request.sent",
    "provider.response.received",
    "litellm.response.normalized",
]
Outcome: TypeAlias = Literal["success", "failure", "cancelled"]


@dataclass(frozen=True, slots=True)
class ShapeLimits:
    nodes: int = 4096
    depth: int = 16
    paths: int = 256
    bytes: int = 16384


_DEFAULT_LIMITS: Final = ShapeLimits()


@dataclass(frozen=True, slots=True)
class PayloadShape:
    field_paths: tuple[str, ...] = ()
    truncated: bool = False

    def merge(self, other: PayloadShape, limits: ShapeLimits = _DEFAULT_LIMITS) -> PayloadShape:
        paths: Final = frozenset(self.field_paths) | frozenset(other.field_paths)
        if self.truncated or other.truncated or len(paths) > limits.paths:
            return PayloadShape(truncated=True)
        if sum(len(path.encode("utf-8")) for path in paths) > limits.bytes:
            return PayloadShape(truncated=True)
        return PayloadShape(tuple(sorted(paths)))


class _LimitReached(Exception):
    pass


def _fields(value: object) -> Iterator[tuple[str, object]]:
    if isinstance(value, Mapping):
        mapping: Final = cast(Mapping[object, object], value)  # cast-ok: guarded Mapping, fields inspected as objects
        for key, child in mapping.items():
            if isinstance(key, str):
                yield key, child


def extract_shape(value: object, limits: ShapeLimits = _DEFAULT_LIMITS) -> PayloadShape:
    native: Final = forwarding.payload_logger()
    if native is None:
        return PayloadShape(truncated=True)
    paths, truncated = native.extract_payload_shape(
        value, nodes=limits.nodes, depth=limits.depth, paths=limits.paths, bytes=limits.bytes
    )
    return PayloadShape(tuple(paths), truncated)


def _native() -> forwarding.NativeDiagnosticLogger | None:
    try:
        native: Final = forwarding.payload_logger()
        return native if native is not None and native.payload_shapes_enabled() else None
    except Exception:  # noqa: BLE001  # optional telemetry must not interrupt inference
        return None


def _payload(value: object, raw_json: bool) -> object:
    if not raw_json or not isinstance(value, (str, bytes)):
        return value
    if len(value) > 1_048_576:
        raise _LimitReached
    return cast(object, json.loads(value))  # cast-ok: JSON is traversed only after runtime container checks


def _send(
    native: ShapeEmitter,
    stage: Stage,
    shape: PayloadShape,
    capture_id: str,
    outcome: Outcome | None = None,
) -> None:
    event: Final = {
        "stage": stage,
        "shape": {"field_paths": shape.field_paths, "truncated": shape.truncated},
        "capture_id": capture_id,
        "outcome": outcome,
    }
    native.emit_payload_shape(json.dumps(event, ensure_ascii=False))


def _shape(value: object, raw_json: bool) -> PayloadShape:
    try:
        return extract_shape(_payload(value, raw_json))
    except Exception:  # noqa: BLE001  # optional telemetry must not interrupt inference
        return PayloadShape(truncated=True)


def record_shape(stage: Stage, value: object, capture_id: object, *, raw_json: bool = False) -> None:
    native: Final = _native()
    if native is None or not isinstance(capture_id, str):
        return
    try:
        _send(native, stage, _shape(value, raw_json), capture_id)
    except Exception:  # noqa: BLE001  # optional telemetry must not interrupt inference
        return


def record_sdk_request(value: Mapping[str, object], capture_id: object) -> None:
    native: Final = _native()
    if native is None or not isinstance(capture_id, str):
        return
    try:
        body: Final = {
            key: child
            for key, child in value.items()
            if key not in ("extra_body", "extra_headers", "extra_query", "timeout")
        }
        extra: Final = dict(_fields(value.get("extra_body")))
        _send(native, "provider.request.sent", extract_shape(body | extra), capture_id)
    except Exception:  # noqa: BLE001  # optional telemetry must not interrupt inference
        return


_Response: Final = TypeVar("_Response")


def capture_result(value: _Response, capture_id: object) -> _Response:
    if isinstance(value, (Mapping, BaseModel)):
        record_shape("litellm.response.normalized", value, capture_id)
    return value


def _call_fields(function: Callable[..., object], bound: inspect.BoundArguments) -> Iterator[tuple[str, object]]:
    arguments: Final = cast(Mapping[str, object], bound.arguments)  # cast-ok: bind_partial validates parameter names
    parameters: Final = inspect.signature(function).parameters
    for name, value in arguments.items():
        if parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
            for key, child in _fields(value):
                if key != "litellm_call_id":
                    yield key, child
        else:
            yield name, value


def record_call_shape(function: Callable[..., object], args: Sequence[object], kwargs: Mapping[str, object]) -> None:
    native: Final = _native()
    capture_id: Final = kwargs.get("litellm_call_id")
    if native is None or not isinstance(capture_id, str):
        return
    try:
        bound: Final = inspect.signature(function).bind_partial(*args, **kwargs)
        fields: Final = dict(_call_fields(function, bound))
        _send(native, "litellm.request.received", extract_shape(fields), capture_id)
    except Exception:  # noqa: BLE001  # optional telemetry must not interrupt inference
        return


def _stream_payload(value: object) -> object:
    if not isinstance(value, (str, bytes)):
        return value
    text: Final = value.decode("utf-8") if isinstance(value, bytes) else value
    if len(text) > 1_048_576:
        raise _LimitReached
    if text.strip() == "[DONE]" or text.strip() == "data: [DONE]":
        return None
    if not text.startswith(("data:", "event:", ":")):
        return text
    data: Final = "\n".join(line[5:].removeprefix(" ") for line in text.splitlines() if line.startswith("data:"))
    return None if not data or data == "[DONE]" else data


class ShapeEmitter(Protocol):
    def emit_payload_shape(self, event: str) -> None: ...


class StreamShapes:
    def __init__(self, capture_id: object, *, native: ShapeEmitter | None = None) -> None:
        self.capture_id: Final = capture_id
        self.native: Final = (native if native is not None else _native()) if isinstance(capture_id, str) else None
        self.received: PayloadShape = PayloadShape()
        self.normalized: PayloadShape = PayloadShape()
        self.finished: bool = False
        self.chunks: int = 0

    def add(self, stage: Stage, value: object) -> None:
        if self.native is None or self.finished:
            return
        try:
            if stage == "provider.response.received":
                self.chunks += 1
            shape: Final = (
                PayloadShape(truncated=True)
                if self.chunks > 4096
                else _shape(_stream_payload(value), isinstance(value, (str, bytes)))
            )
            if stage == "provider.response.received":
                self.received = self.received.merge(shape)
            else:
                self.normalized = self.normalized.merge(shape)
        except Exception:  # noqa: BLE001  # optional telemetry must not interrupt inference
            if stage == "provider.response.received":
                self.received = PayloadShape(truncated=True)
            else:
                self.normalized = PayloadShape(truncated=True)

    def finish(self, outcome: Outcome) -> None:
        if self.native is None or self.finished or not isinstance(self.capture_id, str):
            return
        self.finished = True
        try:
            _send(self.native, "provider.response.received", self.received, self.capture_id, outcome)
            _send(self.native, "litellm.response.normalized", self.normalized, self.capture_id, outcome)
        except Exception:  # noqa: BLE001  # optional telemetry must not interrupt inference
            return
