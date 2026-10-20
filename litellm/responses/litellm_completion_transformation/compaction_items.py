import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from types import MappingProxyType
from typing import Final, TypeVar

from openai.types.responses.easy_input_message_param import EasyInputMessageParam
from openai.types.responses.response_compaction_item import ResponseCompactionItem
from openai.types.responses.response_create_params import ResponseInputParam
from pydantic import TypeAdapter, ValidationError

from litellm.responses.utils import ResponsesAPIRequestUtils

from .tool_search import chat_config_builds_anthropic_request, provider_chat_config

_JSON_OBJECT: Final = TypeAdapter(dict[str, object])
_OBJECT_LIST: Final = TypeAdapter(list[object])
_Item: Final = TypeVar("_Item")
_INSTRUCTION_ROLES: Final = frozenset(("system", "developer"))


def _json_object(value: object) -> Mapping[str, object] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


@dataclass(frozen=True, slots=True)
class CompactionBlock:
    summary: str
    wire_block: Mapping[str, object]


def _compaction_block(value: object) -> CompactionBlock | None:
    block: Final = _json_object(value)
    if block is None:
        return None
    summary: Final = block.get("content")
    if block.get("type") != "compaction" or not isinstance(summary, str) or not summary:
        return None
    return CompactionBlock(summary=summary, wire_block=block)


def _newest_compaction_block(compaction_blocks: object) -> CompactionBlock | None:
    if not isinstance(compaction_blocks, list):
        return None
    blocks: Final = _OBJECT_LIST.validate_python(compaction_blocks)
    return next((block for value in reversed(blocks) if (block := _compaction_block(value)) is not None), None)


def _provider_compaction_block(provider_specific_fields: object) -> CompactionBlock | None:
    fields: Final = _json_object(provider_specific_fields)
    return None if fields is None else _newest_compaction_block(fields.get("compaction_blocks"))


def provider_compaction_summary(provider_specific_fields: object) -> str | None:
    block: Final = _provider_compaction_block(provider_specific_fields)
    return None if block is None else block.summary


def _choice_provider_specific_fields(choice: object) -> object:
    choice_fields: Final = _json_object(choice)
    message: Final = None if choice_fields is None else _json_object(choice_fields.get("message"))
    return None if message is None else message.get("provider_specific_fields")


def response_holds_compaction_block(response: object) -> bool:
    fields: Final = _json_object(response)
    if fields is None:
        return False
    if fields.get("object") == "response":
        return holds_compaction_summary(fields.get("output"))
    try:
        choices: Final = _OBJECT_LIST.validate_python(fields.get("choices"))
    except ValidationError:
        return False
    return any(provider_compaction_summary(_choice_provider_specific_fields(choice)) is not None for choice in choices)


def compaction_output_item(provider_specific_fields: object) -> ResponseCompactionItem | None:
    block: Final = _provider_compaction_block(provider_specific_fields)
    if block is None:
        return None
    return ResponseCompactionItem(
        id=f"cmp_{uuid.uuid4()}",
        encrypted_content=json.dumps(dict(block.wire_block), separators=(",", ":")),
        type="compaction",
    )


def _entry_type(entry: object) -> object:
    fields: Final = _json_object(entry)
    return None if fields is None else fields.get("type")


def _has_entry_of_type(entries: object, entry_type: str) -> bool:
    try:
        rows: Final = _OBJECT_LIST.validate_python(entries)
    except ValidationError:
        return False
    return any(_entry_type(row) == entry_type for row in rows)


def requests_compaction(context_management: object) -> bool:
    if _has_entry_of_type(context_management, "compaction"):
        return True
    anthropic_form: Final = _json_object(context_management)
    return anthropic_form is not None and _has_entry_of_type(anthropic_form.get("edits"), "compact_20260112")


def provider_takes_compaction_blocks(model: str, custom_llm_provider: str | None) -> bool:
    from litellm.llms.bedrock.claude_platform.transformation import BedrockClaudePlatformConfig

    chat_config: Final = provider_chat_config(model, custom_llm_provider)
    return chat_config_builds_anthropic_request(chat_config) and not isinstance(
        chat_config, BedrockClaudePlatformConfig
    )


def _without_affinity_wrap(encrypted_content: str) -> str:
    _, original = ResponsesAPIRequestUtils.unwrap_encrypted_content_with_model_id(encrypted_content)
    return original


def decode_compaction_item(encrypted_content: object) -> CompactionBlock | None:
    if not isinstance(encrypted_content, str):
        return None
    try:
        decoded: Final = _JSON_OBJECT.validate_json(_without_affinity_wrap(encrypted_content))
    except ValidationError:
        return None
    return _compaction_block(decoded)


def compaction_summary(item: object) -> str | None:
    fields: Final = _json_object(item)
    if fields is None or fields.get("type") != "compaction":
        return None
    block: Final = decode_compaction_item(fields.get("encrypted_content"))
    return None if block is None else block.summary


def summary_only_compaction_content(summary: str) -> str:
    return json.dumps({"type": "compaction", "content": summary}, separators=(",", ":"))


def with_compaction_summary(encrypted_content: str, summary: str) -> str:
    original: Final = _without_affinity_wrap(encrypted_content)
    block: Final = decode_compaction_item(original)
    if block is None:
        return encrypted_content
    rewritten: Final = json.dumps({**block.wire_block, "content": summary}, separators=(",", ":"))
    return encrypted_content.removesuffix(original) + rewritten


def with_visited_compaction_summary(item: object, visit: Callable[[str], str]) -> Mapping[str, object] | None:
    fields: Final = _json_object(item)
    if fields is None:
        return None
    summary: Final = compaction_summary(fields)
    encrypted_content: Final = fields.get("encrypted_content")
    if summary is None or not isinstance(encrypted_content, str):
        return None
    return {**fields, "encrypted_content": with_compaction_summary(encrypted_content, visit(summary))}


def holds_compaction_summary(input: object) -> bool:
    try:
        items: Final = _OBJECT_LIST.validate_python(input)
    except ValidationError:
        return False
    return any(compaction_summary(item) is not None for item in items)


def _is_instruction_item(item: object) -> bool:
    fields: Final = _json_object(item)
    if fields is None or fields.get("type", "message") != "message":
        return False
    role: Final = fields.get("role")
    return isinstance(role, str) and role in _INSTRUCTION_ROLES


def input_items(input: object) -> tuple[object, ...]:
    try:
        return tuple(_OBJECT_LIST.validate_python(input))
    except ValidationError:
        return ()


def instruction_items(input: object) -> tuple[object, ...]:
    return tuple(item for item in input_items(input) if _is_instruction_item(item))


def _call_id(item: object) -> str | None:
    fields: Final = _json_object(item)
    call_id: Final = None if fields is None else fields.get("call_id")
    return call_id if isinstance(call_id, str) else None


def _with_its_earlier_call(item: _Item, earlier_calls: Mapping[str, _Item]) -> tuple[_Item, ...]:
    call_id: Final = _call_id(item)
    call: Final = None if call_id is None else earlier_calls.get(call_id)
    return (item,) if call is None else (call, item)


def _items_from_newest_compaction(input: Sequence[_Item]) -> tuple[_Item, ...]:
    newest: Final = next(
        (index for index in range(len(input) - 1, 0, -1) if compaction_summary(input[index]) is not None), 0
    )
    earlier_calls: Final = MappingProxyType(
        {call_id: item for item in input[:newest] if (call_id := _call_id(item)) is not None}
    )
    return (
        *(item for item in input[:newest] if _is_instruction_item(item)),
        *chain.from_iterable(_with_its_earlier_call(item, earlier_calls) for item in input[newest:]),
    )


def without_items_before_compaction(input: str | ResponseInputParam) -> str | ResponseInputParam:
    return list(_items_from_newest_compaction(input)) if isinstance(input, list) else input


def _summary_or_item(item: object) -> object:
    summary: Final = compaction_summary(item)
    return item if summary is None else summary


def billable_input_since_newest_compaction(input: object) -> object:
    try:
        items: Final = _OBJECT_LIST.validate_python(input)
    except ValidationError:
        return input
    return [_summary_or_item(item) for item in _items_from_newest_compaction(items)]


def _as_summary_message(item: _Item) -> _Item | EasyInputMessageParam:
    summary: Final = compaction_summary(item)
    return item if summary is None else EasyInputMessageParam(role="user", content=summary, type="message")


def with_compaction_items_as_summaries(input: str | ResponseInputParam) -> str | ResponseInputParam:
    if not isinstance(input, list) or not holds_compaction_summary(input):
        return input
    return [_as_summary_message(item) for item in _items_from_newest_compaction(input)]


def messages_since_newest_compaction(input: object) -> object:
    try:
        items: Final = _OBJECT_LIST.validate_python(input)
    except ValidationError:
        return input
    return [_as_summary_message(item) for item in _items_from_newest_compaction(items)]
