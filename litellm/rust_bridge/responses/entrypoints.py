from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge import execution
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.openai import ResponsesAPIResponse


class NativeResponses(Protocol):
    def __call__(
        self,
        request: NativeCall,
    ) -> ResponsesAPIResponse: ...


class NativeAresponses(Protocol):
    def __call__(
        self,
        request: NativeCall,
    ) -> Awaitable[ResponsesAPIResponse]: ...


def _responses_binding(value: object) -> NativeResponses | None:
    if not callable(value):
        return None
    native: Final = execution.synchronous(value)
    return cast("NativeResponses", native)  # cast-ok: the execution adapter returns the route's result


def _aresponses_binding(value: object) -> NativeAresponses | None:
    if not callable(value):
        return None
    native: Final = execution.asynchronous(value)
    return cast("NativeAresponses", native)  # cast-ok: the execution adapter returns the route's result


NATIVE_RESPONSES: Final = NativeBinding("responses", validate=_responses_binding)
NATIVE_ARESPONSES: Final = NativeBinding("aresponses", validate=_aresponses_binding)
