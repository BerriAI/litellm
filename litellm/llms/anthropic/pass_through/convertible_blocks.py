from collections.abc import Iterable, Iterator, Mapping, Sequence
from itertools import chain, starmap
from types import MappingProxyType
from typing import Final, NoReturn, TypeVar

from pydantic import JsonValue, TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping, drops_unsupported_params
from litellm.litellm_core_utils.prompt_templates.common_utils import is_unrendered_block_without_text
from litellm.types.utils import TokenCountMessagesFormat

_CONVERTIBLE_BLOCK_TYPES: Final = MappingProxyType(
    {
        "user": frozenset({"text", "image", "document", "tool_result"}),
        "assistant": frozenset({"text", "tool_use", "thinking", "redacted_thinking"}),
    }
)
_SOURCED_BLOCK_TYPES: Final = frozenset({"image", "document"})
_TOOL_RESULT_BLOCK_TYPES: Final = frozenset({"text", "image", "document"})


_Block = TypeVar("_Block")  # rebind-ok: a TypeVar cannot be Final
_JSON_MESSAGES: Final = TypeAdapter(tuple[Mapping[str, JsonValue], ...])


def _json_messages(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, JsonValue], ...]:
    try:
        return _JSON_MESSAGES.validate_python(messages)
    except ValidationError:
        return ()


def _source_label(block_type: str, source: JsonValue) -> str | None:
    if not isinstance(source, dict):
        return f"{block_type} without a source"
    match source.get("type"):
        case "base64" if source.get("data"):
            return None
        case "url" if source.get("url"):
            return None
        case "base64" | "url" as source_type:
            return f"{block_type} with an empty {source_type} source"
        case str() as source_type:
            return f"{block_type} with a {source_type} source"
        case _:
            return f"{block_type} with an untyped source"


def _block_label(block: JsonValue, convertible: frozenset[str]) -> str | None:
    if not isinstance(block, dict):
        return None
    block_type: Final = block.get("type")
    if not isinstance(block_type, str):
        return "a block without a type"
    if is_unrendered_block_without_text(block):
        return None
    if block_type not in convertible:
        return block_type
    return _source_label(block_type, block.get("source")) if block_type in _SOURCED_BLOCK_TYPES else None


def _tool_result_labels(block: JsonValue) -> Iterator[str]:
    if not isinstance(block, dict) or block.get("type") != "tool_result":
        return iter(())
    inner: Final = block.get("content")
    if not isinstance(inner, list):
        return iter(())
    return (
        f"{label} inside a tool_result"
        for item in inner
        if (label := _block_label(item, _TOOL_RESULT_BLOCK_TYPES)) is not None
    )


def _content_labels(block: JsonValue, convertible: frozenset[str]) -> Iterator[str]:
    label: Final = _block_label(block, convertible)
    return chain(() if label is None else (label,), _tool_result_labels(block))


def _message_labels(message: Mapping[str, JsonValue]) -> Iterator[str]:
    role: Final = message.get("role")
    convertible: Final = _CONVERTIBLE_BLOCK_TYPES.get(role) if isinstance(role, str) else None
    content: Final = message.get("content")
    if convertible is None or not isinstance(content, list):
        return iter(())
    return chain.from_iterable(_content_labels(block, convertible) for block in content)


def unconvertible_anthropic_blocks(messages: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    from litellm.llms.anthropic.pass_through.messages.utils import normalize_native_messages

    normalized: Final = _json_messages(normalize_native_messages(messages))
    return tuple(sorted(set(chain.from_iterable(_message_labels(message) for message in normalized))))


def unconvertible_blocks_error(
    messages: Sequence[Mapping[str, object]] | None, messages_format: TokenCountMessagesFormat
) -> str | None:
    match messages_format:
        case "openai":
            return None
        case "anthropic":
            labels: Final = unconvertible_anthropic_blocks(messages or ())
            return (
                "count_tokens can't convert these content blocks for this model, so the count would leave them out: "
                + ", ".join(labels)
                if labels
                else None
            )
        case _:
            assert_never(messages_format)


_OBJECTS: Final = TypeAdapter(tuple[object, ...])


def _list_items(value: object) -> tuple[object, ...]:
    return _OBJECTS.validate_python(value) if isinstance(value, list) else ()


def _is_non_string_text(block: object) -> bool:
    match as_str_mapping(block):
        case {"type": "text", "text": str()}:
            return False
        case {"type": "text", "text": _}:
            return True
        case _:
            return False


def _user_content_blocks(message: object) -> tuple[object, ...]:
    match as_str_mapping(message):
        case {"role": "user", "content": content}:
            return _list_items(content)
        case _:
            return ()


def _message_text_errors(message_index: int, message: object) -> Iterator[str]:
    return (
        f"messages.{message_index}.content.{block_index}.text: Input should be a valid string"
        for block_index, block in enumerate(_user_content_blocks(message))
        if _is_non_string_text(block)
    )


def non_string_user_text_error(messages: Iterable[object]) -> str | None:
    return next(chain.from_iterable(starmap(_message_text_errors, enumerate(messages))), None)


def non_string_system_text_error(system: object) -> str | None:
    return next(
        (
            f"system.{index}.text: Input should be a valid string"
            for index, block in enumerate(_list_items(system))
            if _is_non_string_text(block)
        ),
        None,
    )


def _without_non_string_texts(
    message: dict[str, object],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
) -> dict[str, object] | None:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    match message:
        case {"role": "user", "content": list()}:
            blocks: Final = _user_content_blocks(message)
            kept: Final = [block for block in blocks if not _is_non_string_text(block)]
            return {**message, "content": kept} if kept or not blocks else None
        case _:
            return message


def drop_non_string_texts(
    messages: Sequence[dict[str, object]],  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    drop_params: object,
) -> Sequence[dict[str, object]]:  # mutable-ok: AnthropicMessagesRequest.messages is typed list[dict]
    if not drops_unsupported_params(drop_params) or non_string_user_text_error(messages) is None:
        return messages
    return [kept for message in messages if (kept := _without_non_string_texts(message)) is not None]


def without_non_string_system_texts(
    system: Sequence[_Block],
) -> list[_Block] | None:  # mutable-ok: AnthropicMessagesRequest.system is typed list[dict]
    return [block for block in system if not _is_non_string_text(block)] or None


def drop_non_string_system_texts(
    system: str | list[dict[str, JsonValue]] | None,  # mutable-ok: AnthropicMessagesRequest.system is typed list[dict]
    drop_params: object,
) -> str | list[dict[str, JsonValue]] | None:  # mutable-ok: AnthropicMessagesRequest.system is typed list[dict]
    if (
        not isinstance(system, list)
        or not drops_unsupported_params(drop_params)
        or non_string_system_text_error(system) is None
    ):
        return system
    return without_non_string_system_texts(system)


def emptied_messages_error(sent: Sequence[object], kept: Sequence[object]) -> str | None:
    if kept or non_string_user_text_error(sent) is None:
        return None
    return "messages: no message has content left after dropping empty or non-string text blocks"


def reject_non_string_user_text(messages: Iterable[object], drop_params: object, model: str, provider: object) -> None:
    if drops_unsupported_params(drop_params):
        return
    error: Final = non_string_user_text_error(messages)
    if error is None:
        return
    _raise_bad_request(error, model, provider)


def reject_non_string_system_text(system: object, drop_params: object, model: str, provider: object) -> None:
    error: Final = None if drops_unsupported_params(drop_params) else non_string_system_text_error(system)
    if error is None:
        return
    _raise_bad_request(error, model, provider)


def reject_emptied_messages(sent: Sequence[object], kept: Sequence[object], model: str, provider: str | None) -> None:
    error: Final = emptied_messages_error(sent, kept)
    if error is None:
        return
    _raise_bad_request(error, model, provider)


def _raise_bad_request(error: str, model: str, provider: object) -> NoReturn:
    raise litellm.BadRequestError(message=error, model=model, llm_provider=_provider_name(model, provider))


def _provider_name(model: str, provider: object) -> str:
    if isinstance(provider, str) and provider:
        return provider
    try:
        return litellm.get_llm_provider(model=model)[1]
    except litellm.BadRequestError:
        return ""
