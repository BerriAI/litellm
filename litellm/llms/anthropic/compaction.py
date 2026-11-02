from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

from litellm.llms.compaction import CompactionProtocol
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES, AnthropicCompaction

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECTS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_HEADERS: Final = TypeAdapter(dict[str, str])
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_CONFLICTS: Final = ("context_management", "response_format", "stop", "stop_sequences", "tool_choice")


def supports_native_compaction(params: Mapping[str, object]) -> bool:
    from litellm.utils import get_model_info

    if params.get("custom_llm_provider") not in (None, "anthropic", "openai"):
        return False
    model: Final = str(params.get("model", "")).removeprefix("openai/").removeprefix("anthropic/")
    try:
        return get_model_info(model=model, custom_llm_provider="anthropic").get("supports_anthropic_compaction") is True
    except Exception:
        return False


def compatible_defaults(payload: Mapping[str, object]) -> bool:
    return all(payload.get(key) is None for key in _CONFLICTS)


def request_kwargs() -> Mapping[str, object]:
    operation: Final[AnthropicCompaction] = {"type": "summarize"}
    return MappingProxyType(
        {
            "compaction": operation,
            "extra_headers": _HEADERS.validate_python(
                MappingProxyType({"anthropic-beta": ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_09_04.value})
            ),
        }
    )


def _native_blocks(protocol: CompactionProtocol, response: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if protocol == "messages":
        return (
            _OBJECTS.validate_python(response.get("content", ())) if response.get("stop_reason") == "compaction" else ()
        )
    choices: Final = _OBJECTS.validate_python(response.get("choices", ()))
    choice: Final = choices[0] if len(choices) == 1 else _EMPTY
    message: Final = _MAPPING.validate_python(choice.get("message", _EMPTY))
    fields: Final = _MAPPING.validate_python(message.get("provider_specific_fields") or _EMPTY)
    return _OBJECTS.validate_python(fields.get("compaction_blocks", ()))


def extract_summary(protocol: CompactionProtocol, response: Mapping[str, object]) -> str | None:
    blocks: Final = _native_blocks(protocol, response)
    block: Final = blocks[0] if len(blocks) == 1 else _EMPTY
    content: Final = block.get("content")
    return (
        content
        if block.get("type") == "compaction"
        and isinstance(block.get("signature"), str)
        and block.get("signature")
        and isinstance(content, str)
        and content.strip()
        else None
    )
