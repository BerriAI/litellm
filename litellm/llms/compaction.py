from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

from pydantic import TypeAdapter

from litellm.types.llms.openai import AllMessageValues

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
