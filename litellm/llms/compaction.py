from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from pydantic import TypeAdapter

from litellm.types.llms.openai import AllMessageValues, ChatCompletionResponseMessage

if TYPE_CHECKING:
    from litellm.router import Router

CompactionProtocol: TypeAlias = Literal["chat", "messages"]
_MAPPING: Final = TypeAdapter(Mapping[str, object])
_MESSAGES: Final = TypeAdapter(list[AllMessageValues])


class NativeCompactionProvider(Protocol):
    def supports_native_compaction(self, params: Mapping[str, object]) -> bool: ...

    def compatible_defaults(self, payload: Mapping[str, object]) -> bool: ...

    def request_kwargs(self) -> Mapping[str, object]: ...

    def extract_summary(self, protocol: CompactionProtocol, response: Mapping[str, object]) -> str | None: ...


def get_native_compaction_provider(params: Mapping[str, object]) -> NativeCompactionProvider | None:
    from litellm.llms.anthropic import compaction

    return compaction if compaction.supports_native_compaction(params) else None


class ResponsesCompactionCodec(Protocol):
    """Provider side of carrying compaction through the Responses chat-completions bridge.

    The bridge only moves an opaque ``encrypted_content`` token between a Responses
    ``compaction`` item and a chat message; the provider owns what the token encodes, how
    it is replayed, and which of several accumulated tokens a request may carry.
    """

    def encode(self, provider_specific_fields: Mapping[str, object]) -> str | None: ...

    def is_streaming_compaction(self, provider_specific_fields: object) -> bool: ...

    def replay_message(self, encrypted_content: str) -> ChatCompletionResponseMessage | None: ...

    def inspectable_text(self, encrypted_content: str) -> str | None: ...

    def superseded_replay_indices(self, messages: Sequence[object]) -> frozenset[int]: ...


def get_responses_compaction_codec() -> ResponsesCompactionCodec:
    from litellm.llms.anthropic import responses_compaction

    return responses_compaction


async def dispatch(router: Router, protocol: CompactionProtocol, payload: Mapping[str, object]) -> Mapping[str, object]:
    if protocol == "messages":
        return _MAPPING.validate_python(
            await router.aanthropic_messages(custom_llm_provider=None, client=None, **payload)
        )
    response: Final = await router.acompletion(
        model=str(payload["model"]),
        messages=_MESSAGES.validate_python(payload["messages"]),
        stream=False,
        **MappingProxyType(
            {key: value for key, value in payload.items() if key not in ("model", "messages", "stream")}
        ),
    )
    return _MAPPING.validate_python(response.model_dump())
