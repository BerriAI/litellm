from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge.bindings import NativeBinding
from litellm.types.utils import ModelResponse


@dataclass(frozen=True, slots=True)
class LiteLLMChatCompletionsRequest:
    model: str
    messages: Sequence[object]
    stream: bool | None
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: Mapping[str, object] | None
    kwargs: Mapping[str, object]


class NativeCompletion(Protocol):
    def __call__(
        self,
        request: LiteLLMChatCompletionsRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> ModelResponse: ...


class NativeAcompletion(Protocol):
    def __call__(
        self,
        request: LiteLLMChatCompletionsRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> Awaitable[ModelResponse]: ...


def _completion_binding(value: object) -> NativeCompletion | None:
    if not callable(value):
        return None
    return cast("NativeCompletion", value)  # cast-ok: callable validated at the native binding boundary


def _acompletion_binding(value: object) -> NativeAcompletion | None:
    if not callable(value):
        return None
    return cast("NativeAcompletion", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_COMPLETION: Final = NativeBinding("completion", validate=_completion_binding)
NATIVE_ACOMPLETION: Final = NativeBinding("acompletion", validate=_acompletion_binding)
