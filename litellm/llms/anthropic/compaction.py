from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn

from pydantic import TypeAdapter

from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES, AnthropicCompaction
from litellm.types.llms.openai import AllMessageValues

if TYPE_CHECKING:
    from litellm.router import Router

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_CHAT_MESSAGES: Final = TypeAdapter(list[AllMessageValues])
_SEQUENCE: Final = TypeAdapter(tuple[object, ...])
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
CLIENT_COMPACTION_PARAMS: Final = ("context_management", "compaction")


def request_kwargs() -> Mapping[str, object]:
    operation: Final[AnthropicCompaction] = {"type": "summarize"}
    return MappingProxyType(
        {
            "compaction": operation,
            "extra_headers": {"anthropic-beta": ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_09_04.value},
        }
    )


def supports_native_compaction(params: Mapping[str, object]) -> bool:
    from litellm.utils import get_model_info

    if params.get("custom_llm_provider") not in (None, "anthropic", "openai"):
        return False
    model: Final = str(params.get("model", "")).removeprefix("openai/").removeprefix("anthropic/")
    try:
        return get_model_info(model=model, custom_llm_provider="anthropic").get("supports_anthropic_compaction") is True
    except Exception:
        return False


def _reject(model: str, reason: str) -> NoReturn:
    from litellm.exceptions import BadRequestError

    raise BadRequestError(message=f"Context compaction: {reason}", model=model, llm_provider="")


def _objects(value: object) -> tuple[Mapping[str, object], ...]:
    return (
        tuple(
            _MAPPING.validate_python(block) for block in _SEQUENCE.validate_python(value) if isinstance(block, Mapping)
        )
        if isinstance(value, (list, tuple))
        else ()
    )


def validate_compactor_defaults(payload: Mapping[str, object]) -> None:
    if any(
        payload.get(key) is not None
        for key in ("context_management", "response_format", "stop", "stop_sequences", "tool_choice")
    ):
        _reject(str(payload.get("model", "")), "Compactor defaults conflict with on-demand compaction")


def _native_blocks(
    protocol: Literal["chat", "messages"], response: Mapping[str, object], model: str
) -> tuple[Mapping[str, object], ...]:
    if protocol == "messages":
        if response.get("stop_reason") != "compaction":
            _reject(model, "The native compactor did not finish compaction")
        return _objects(response.get("content"))
    choices: Final = _objects(response.get("choices"))
    choice: Final = choices[0] if len(choices) == 1 else _EMPTY
    message: Final = _MAPPING.validate_python(choice.get("message", _EMPTY))
    provider_fields: Final = _MAPPING.validate_python(message.get("provider_specific_fields", _EMPTY))
    return _objects(provider_fields.get("compaction_blocks"))


def extract_summary(protocol: Literal["chat", "messages"], response: Mapping[str, object], model: str) -> str:
    blocks: Final = _native_blocks(protocol, response, model)
    if len(blocks) != 1:
        _reject(model, "Expected exactly one native compaction block")
    block: Final = _MAPPING.validate_python(blocks[0])
    content: Final = block.get("content")
    if (
        block.get("type") != "compaction"
        or not block.get("signature")
        or not isinstance(content, str)
        or not content.strip()
    ):
        _reject(model, "The native compaction block is incomplete")
    return content


async def dispatch(
    router: Router, protocol: Literal["chat", "messages"], payload: Mapping[str, object]
) -> Mapping[str, object]:
    if protocol == "chat":
        chat_response: Final = await router.acompletion(
            model=str(payload["model"]),
            messages=_CHAT_MESSAGES.validate_python(payload["messages"]),
            stream=False,
            **MappingProxyType(
                {key: value for key, value in payload.items() if key not in ("model", "messages", "stream")}
            ),
        )
        return _MAPPING.validate_python(chat_response.model_dump())
    return _MAPPING.validate_python(await router.aanthropic_messages(custom_llm_provider=None, client=None, **payload))
