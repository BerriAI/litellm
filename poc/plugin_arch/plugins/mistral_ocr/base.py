"""Plugin interface and transform scaffolding.

The thin core only ever sees `LLMPlugin`. It calls `handle(request)` and gets
back a response (or an error in the interface's error shape). The core does
not know that internally a plugin may transform the request, call upstream,
and transform the response back. That is an implementation detail.

`TransformingLLMPlugin` is a base class for the common case
(transform_request -> call_upstream -> transform_response). Providers that
need fully custom control flow (multi-step auth, entangled signing/call,
etc.) implement `handle` directly and ignore the scaffolding. Both satisfy
the same `LLMPlugin` interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar


@dataclass(frozen=True, slots=True)
class PluginRequest:
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()
    content_type: str = "application/json"


@dataclass(frozen=True, slots=True)
class PluginResponse:
    body: bytes
    content_type: str = "application/json"
    status_code: int = 200


@dataclass(frozen=True, slots=True)
class PluginError:
    code: str
    message: str
    type: str
    http_status: int = 502


@dataclass(frozen=True, slots=True)
class Capabilities:
    models: tuple[str, ...] = ()
    endpoints: tuple[str, ...] = ()


N = TypeVar("N")


@dataclass(frozen=True, slots=True)
class Ok(Generic[N]):
    value: N
    tag: Literal["ok"] = "ok"


@dataclass(frozen=True, slots=True)
class Err:
    error: PluginError
    tag: Literal["err"] = "err"


class LLMPlugin(ABC):
    @abstractmethod
    def handle(self, request: PluginRequest) -> PluginResponse | PluginError: ...

    @abstractmethod
    def capabilities(self) -> Capabilities: ...


NativeRequest = TypeVar("NativeRequest")
NativeResponse = TypeVar("NativeResponse")


class TransformingLLMPlugin(LLMPlugin, Generic[NativeRequest, NativeResponse]):
    def handle(self, request: PluginRequest) -> PluginResponse | PluginError:
        native_req = self.transform_request(request)
        upstream = self.call_upstream(native_req)
        match upstream:
            case Ok(value=payload):
                return self.transform_response(payload)
            case Err(error=err):
                return err

    @abstractmethod
    def transform_request(self, request: PluginRequest) -> NativeRequest: ...

    @abstractmethod
    def call_upstream(self, native_request: NativeRequest) -> Ok[NativeResponse] | Err: ...

    @abstractmethod
    def transform_response(self, native_response: NativeResponse) -> PluginResponse: ...
