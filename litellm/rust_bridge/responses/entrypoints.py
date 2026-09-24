from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge.bindings import NativeBinding
from litellm.types.llms.openai import ResponsesAPIResponse


@dataclass(frozen=True, slots=True)
class LiteLLMResponsesRequest:
    model: str
    input: object
    stream: bool | None
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: Mapping[str, object] | None
    kwargs: Mapping[str, object]


class NativeResponses(Protocol):
    def __call__(
        self,
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ResponsesAPIResponse: ...


class NativeAresponses(Protocol):
    def __call__(
        self,
        request: LiteLLMResponsesRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Awaitable[ResponsesAPIResponse]: ...


def _responses_binding(value: object) -> NativeResponses | None:
    if not callable(value):
        return None
    return cast("NativeResponses", value)  # cast-ok: callable validated at the native binding boundary


def _aresponses_binding(value: object) -> NativeAresponses | None:
    if not callable(value):
        return None
    return cast("NativeAresponses", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_RESPONSES: Final = NativeBinding("responses", validate=_responses_binding)
NATIVE_ARESPONSES: Final = NativeBinding("aresponses", validate=_aresponses_binding)
