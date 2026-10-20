"""
Utility functions for base LLM classes.
"""

import copy
import itertools
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

from openai.lib import _parsing, _pydantic  # pyright: ignore[reportPrivateUsage]  # SDK parser internals
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypeIs  # noqa: TID251  # TypeIs lands in typing only on 3.13

from litellm._logging import verbose_logger
from litellm.constants import ANTHROPIC_BILLING_METADATA_PREFIX
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionToolCallChunk,
)
from litellm.types.utils import (
    CountContextManagement,
    Message,
    ProviderSpecificModelInfo,
    TokenCountMessagesFormat,
    TokenCountResponse,
)


class BaseTokenCounter(ABC):
    @abstractmethod
    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, Any]] | None,
        contents: list[dict[str, object]] | None,
        deployment: dict[str, object] | None = None,
        request_model: str = "",
        tools: list[dict[str, Any]] | None = None,
        system: Any | None = None,
        messages_format: TokenCountMessagesFormat = "openai",
        output_format: JsonValue = None,
        context_management: CountContextManagement | None = None,
    ) -> TokenCountResponse | None:
        pass

    @abstractmethod
    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """
        Returns True if we should the this API for token counting for the selected `custom_llm_provider`
        """
        return False


class BaseLLMModelInfo(ABC):
    def get_provider_info(
        self,
        model: str,
    ) -> ProviderSpecificModelInfo | None:
        """
        Default values all models of this provider support.
        """
        return None

    def get_model_cost_key(self, model: str) -> str | None:
        """
        Maps the model name a user sends to the key `litellm.model_cost` stores it under, when the two differ.
        `get_model_info` tries this key once the exact `model` and `provider/model` keys miss. The default None means
        the provider's user-facing names already match the cost map, so there is nothing extra to try.
        """
        return None

    @abstractmethod
    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        """
        Returns a list of models supported by this provider.
        """
        return []

    def discover_models(
        self, litellm_params: Mapping[str, object] | None = None
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        """
        Live model discovery for a configured deployment. Defaults to the api_key/api_base
        facade every provider already implements via ``get_models``; a provider whose
        discovery needs more of ``litellm_params`` (e.g. Anthropic's workload identity
        federation) overrides this instead of widening ``get_models`` for every provider.
        """
        api_key: Final = litellm_params.get("api_key") if litellm_params is not None else None
        api_base: Final = litellm_params.get("api_base") if litellm_params is not None else None
        return self.get_models(
            api_key=api_key if isinstance(api_key, str) else None,
            api_base=api_base if isinstance(api_base, str) else None,
        )

    @staticmethod
    @abstractmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        pass

    @staticmethod
    @abstractmethod
    def get_api_base(
        api_base: str | None = None,
    ) -> str | None:
        pass

    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def get_base_model(model: str) -> str | None:
        """
        Returns the base model name from the given model name.

        Some providers like bedrock - can receive model=`invoke/anthropic.claude-3-opus-20240229-v1:0` or `converse/anthropic.claude-3-opus-20240229-v1:0`
            This function will return `anthropic.claude-3-opus-20240229-v1:0`
        """

    def get_token_counter(self) -> BaseTokenCounter | None:
        """
        Factory method to create a token counter for this provider.

        Returns:
            Optional TokenCounterInterface implementation for this provider,
            or None if token counting is not supported.
        """
        return None


def convert_tool_response_to_message(
    tool_calls: list[ChatCompletionToolCallChunk],
) -> Message | None:
    """
    In JSON mode, Anthropic API returns JSON schema as a tool call, we need to convert it to a message to follow the OpenAI format

    """
    ## HANDLE JSON MODE - anthropic returns single function call
    json_mode_content_str: Final[str | None] = tool_calls[0]["function"].get("arguments")
    try:
        if json_mode_content_str is not None:
            args: Final = _DECODED_JSON.validate_python(json.loads(json_mode_content_str))
            args_object: Final = as_str_mapping(args)
            if args_object is not None and (values := args_object.get("values")) is not None:
                _message = Message(content=json.dumps(values))
                return _message
            else:
                # a lot of the times the `values` key is not present in the tool response
                # relevant issue: https://github.com/BerriAI/litellm/issues/6741
                _message = Message(content=json.dumps(args))
                return _message
    except json.JSONDecodeError:
        # json decode error does occur, return the original tool response str
        return Message(content=json_mode_content_str)
    return None


_convert_tool_response_to_message = convert_tool_response_to_message


@dataclass(frozen=True, slots=True)
class ResponseFormatSchema:
    declared_names: frozenset[str]
    required_names: frozenset[str] = frozenset()
    allows_undeclared_keys: bool = False

    @property
    def accepts_a_lone_undeclared_key(self) -> bool:
        return self.allows_undeclared_keys and not self.required_names


_DECODED_JSON: Final = TypeAdapter(object)
_JSON_OBJECT: Final = TypeAdapter(Mapping[str, JsonValue])
_NAMES: Final = TypeAdapter(frozenset[str])


def as_json_object(value: object) -> Mapping[str, JsonValue] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def _names(value: object) -> frozenset[str]:
    try:
        return _NAMES.validate_python(value)
    except ValidationError:
        return frozenset()


def _allows_undeclared_keys(document: Mapping[str, JsonValue]) -> bool:
    additional_properties: Final = document.get("additionalProperties")
    return (
        additional_properties is True
        or as_json_object(additional_properties) is not None
        or bool(as_json_object(document.get("patternProperties")))
    )


def response_format_schema(value: object) -> ResponseFormatSchema | None:
    document: Final = as_json_object(value)
    if document is None:
        return None
    properties: Final = as_json_object(document.get("properties", {}))
    if properties is None:
        return None
    required_names: Final = _names(document.get("required", ()))
    return ResponseFormatSchema(
        declared_names=frozenset(properties) | required_names,
        required_names=required_names,
        allows_undeclared_keys=_allows_undeclared_keys(document),
    )


RESPONSE_FORMAT_ENVELOPE_NAME_KEYS: Final = frozenset({"name", "tool_name"})


@dataclass(frozen=True, slots=True)
class _EnvelopeLayer:
    inner: Mapping[str, JsonValue]
    names_the_tool: bool


def _json_object_or_encoded_object(value: JsonValue) -> Mapping[str, JsonValue] | None:
    if not isinstance(value, str):
        return as_json_object(value)
    try:
        return as_json_object(_DECODED_JSON.validate_python(json.loads(value)))
    except (json.JSONDecodeError, RecursionError):
        return None


def _label_keys(payload_object: Mapping[str, JsonValue], schema: ResponseFormatSchema) -> frozenset[str]:
    if not schema.required_names:
        return frozenset()
    return frozenset(
        key for key, value in payload_object.items() if isinstance(value, str) and key not in schema.declared_names
    )


def _envelope_layer(
    payload_object: Mapping[str, JsonValue], schema: ResponseFormatSchema, tool_name: str
) -> _EnvelopeLayer | None:
    name_keys: Final = frozenset(
        key for key in RESPONSE_FORMAT_ENVELOPE_NAME_KEYS if payload_object.get(key) == tool_name
    )
    label_keys: Final = name_keys or _label_keys(payload_object, schema)
    wrapper_keys: Final = tuple(key for key in payload_object if key not in label_keys)
    if len(wrapper_keys) != 1 or wrapper_keys[0] in schema.declared_names:
        return None
    wrapped: Final = payload_object[wrapper_keys[0]]
    inner: Final = _json_object_or_encoded_object(wrapped) if name_keys else as_json_object(wrapped)
    return None if inner is None else _EnvelopeLayer(inner=inner, names_the_tool=bool(name_keys))


def _envelope_layers(
    payload_object: Mapping[str, JsonValue], schema: ResponseFormatSchema, tool_name: str
) -> Iterator[_EnvelopeLayer]:
    layer = _envelope_layer(payload_object, schema, tool_name)  # rebind-ok: the recursion gate needs a loop
    while layer is not None:
        yield layer
        layer = _envelope_layer(layer.inner, schema, tool_name)


def _schema_object_inside_envelope(
    payload_object: Mapping[str, JsonValue], schema: ResponseFormatSchema, tool_name: str
) -> Mapping[str, JsonValue] | None:
    walk: Final = _envelope_layers(payload_object, schema, tool_name)
    layers: Final = tuple(walk if schema.declared_names else itertools.islice(walk, 1))
    schema_object: Final = next((layer.inner for layer in layers if _shares_a_schema_key(layer.inner, schema)), None)
    if schema_object is not None:
        return schema_object
    return next((layer.inner for layer in layers if layer.names_the_tool), None)


def _shares_a_schema_key(candidate: Mapping[str, JsonValue], schema: ResponseFormatSchema) -> bool:
    return not schema.accepts_a_lone_undeclared_key and not schema.declared_names.isdisjoint(candidate)


def response_format_envelope_contents(payload: object, schema: ResponseFormatSchema, tool_name: str) -> object | None:
    payload_object: Final = as_json_object(payload)
    if payload_object is None:
        return None
    schema_object: Final = _schema_object_inside_envelope(payload_object, schema, tool_name)
    if schema_object is not None:
        return schema_object
    return None if "values" in schema.declared_names else payload_object.get("values")


_LONE_SURROGATE: Final = re.compile("[\ud800-\udfff]")


def response_format_answer_json(value: object) -> str:
    text: Final = json.dumps(value, ensure_ascii=False)
    return json.dumps(value) if _LONE_SURROGATE.search(text) else text


def unwrap_response_format_envelope(payload: object, schema: ResponseFormatSchema, tool_name: str) -> object:
    contents: Final = response_format_envelope_contents(payload, schema, tool_name)
    return payload if contents is None else contents


def _field_names(
    blocks: tuple[Mapping[str, JsonValue], ...], key: str, schema: ResponseFormatSchema
) -> tuple[str, ...] | None:
    values: Final = (block[key] for block in blocks)
    names: Final = tuple(value for value in values if isinstance(value, str) and value in schema.declared_names)
    return names if len(frozenset(names)) == len(blocks) else None


def _is_a_label_every_block_repeats(blocks: tuple[Mapping[str, JsonValue], ...], key: str) -> bool:
    return isinstance(blocks[0][key], str) and all(block[key] == blocks[0][key] for block in blocks)


def _value_key(blocks: tuple[Mapping[str, JsonValue], ...], keys: frozenset[str]) -> str | None:
    if len(keys) == 1:
        return next(iter(keys))
    varying: Final = tuple(key for key in keys if not _is_a_label_every_block_repeats(blocks, key))
    return varying[0] if len(varying) == 1 else None


def _split_answer_fields(
    blocks: tuple[Mapping[str, JsonValue], ...], schema: ResponseFormatSchema
) -> Mapping[str, JsonValue] | None:
    keys: Final = frozenset(blocks[0])
    if not keys.isdisjoint(schema.declared_names) or any(frozenset(block) != keys for block in blocks):
        return None
    named: Final = tuple((key, names) for key in keys if (names := _field_names(blocks, key, schema)) is not None)
    if len(named) != 1:
        return None
    ((name_key, field_names),) = named
    value_key: Final = _value_key(blocks, keys - {name_key})
    if value_key is None:
        return None
    return dict(zip(field_names, (block[value_key] for block in blocks), strict=True))


def _merged_partial_answers(
    blocks: tuple[Mapping[str, JsonValue], ...], schema: ResponseFormatSchema
) -> Mapping[str, JsonValue] | None:
    keys: Final = frozenset(itertools.chain.from_iterable(blocks))
    if not keys <= schema.declared_names or len(keys) != sum(len(block) for block in blocks):
        return None
    return dict(itertools.chain.from_iterable(block.items() for block in blocks))


def split_response_format_answer(
    encoded_blocks: Sequence[str], schema: ResponseFormatSchema
) -> Mapping[str, JsonValue] | None:
    decoded: Final = tuple(_json_object_or_encoded_object(encoded) for encoded in encoded_blocks)
    blocks: Final = tuple(block for block in decoded if block is not None)
    if len(blocks) < 2 or len(blocks) != len(decoded):
        return None
    named_fields: Final = _split_answer_fields(blocks, schema)
    return named_fields if named_fields is not None else _merged_partial_answers(blocks, schema)


def partial_response_format_answer(encoded_block: str, schema: ResponseFormatSchema) -> Mapping[str, JsonValue] | None:
    block: Final = _json_object_or_encoded_object(encoded_block)
    return block if block is not None and frozenset(block) <= schema.declared_names else None


def names_a_response_format_property(encoded_block: str, schema: ResponseFormatSchema) -> bool:
    block: Final = _json_object_or_encoded_object(encoded_block)
    if block is None:
        return False
    return any(isinstance(value, str) and value in schema.declared_names for value in block.values())


def _dict_to_response_format_helper(response_format: dict, ref_template: str | None = None) -> dict:
    if ref_template is not None and response_format.get("type") == "json_schema":
        # Deep copy to avoid modifying original
        modified_format: Final = copy.deepcopy(response_format)
        schema: Final = modified_format["json_schema"]["schema"]

        # Update all $ref values in the schema
        def update_refs(schema):
            stack: Final = [(schema, [])]
            visited: Final = set()

            while stack:
                obj, path = stack.pop()
                obj_id = id(obj)

                if obj_id in visited:
                    continue
                visited.add(obj_id)

                if isinstance(obj, dict):
                    if "$ref" in obj:
                        ref_path = obj["$ref"]
                        model_name = ref_path.split("/")[-1]
                        obj["$ref"] = ref_template.format(model=model_name)

                    for k, v in obj.items():
                        if isinstance(v, (dict, list)):
                            stack.append((v, path + [k]))

                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            stack.append((item, path + [i]))

        update_refs(schema)
        return modified_format
    return response_format


def type_to_response_format_param(
    response_format: type[BaseModel] | dict | None,
    ref_template: str | None = None,
) -> dict | None:
    """
    Re-implementation of openai's 'type_to_response_format_param' function

    Used for converting pydantic object to api schema.
    """
    if response_format is None:
        return None

    if isinstance(response_format, dict):
        return _dict_to_response_format_helper(response_format, ref_template)

    # type checkers don't narrow the negation of a `TypeGuard` as it isn't
    # a safe default behaviour but we know that at this point the `response_format`
    # can only be a `type`
    if not _parsing._completions.is_basemodel_type(response_format):  # pyright: ignore[reportPrivateUsage]  # SDK parser internals
        raise TypeError(f"Unsupported response_format type - {response_format}")

    if ref_template is not None:
        schema = response_format.model_json_schema(ref_template=ref_template)
    else:
        schema = _pydantic.to_strict_json_schema(response_format)

    return {
        "type": "json_schema",
        "json_schema": {
            "schema": schema,
            "name": response_format.__name__,
            "strict": True,
        },
    }


def map_developer_role_to_system_role(
    messages: Sequence[AllMessageValues],
) -> list[AllMessageValues]:
    """
    Translate `developer` role to `system` role for non-OpenAI providers.
    """
    new_messages: Final[list[AllMessageValues]] = []
    for m in messages:
        if m["role"] == "developer":
            verbose_logger.debug(
                "Translating developer role to system role for non-OpenAI providers."
            )  # ensure user knows what's happening with their input.
            new_messages.append({"role": "system", "content": m["content"]})
        else:
            new_messages.append(m)
    return new_messages


SystemMessageContent: TypeAlias = str | list[object]

_system_content_adapter: Final = TypeAdapter[SystemMessageContent | None](SystemMessageContent | None)


def _system_content(message: ChatCompletionSystemMessage) -> SystemMessageContent:
    if "content" not in message:
        return ""
    content: Final = _system_content_adapter.validate_python(message["content"])
    return "" if content is None else content


def _message_cache_control(message: ChatCompletionSystemMessage) -> ChatCompletionCachedContent | None:
    if "cache_control" not in message:
        return None
    return message["cache_control"]


def _with_message_cache_control(
    blocks: Sequence[object], cache_control: ChatCompletionCachedContent | None
) -> list[object]:  # mutable-ok: content block lists are the wire format callers isinstance-check
    last: Final = as_str_mapping(blocks[-1]) if blocks else None
    if last is None or cache_control is None or "cache_control" in last:
        return [*blocks]
    return [*blocks[:-1], {**last, "cache_control": cache_control}]


def _as_system_message(message: AllMessageValues) -> AllMessageValues:
    if message["role"] != "developer":
        return message
    translated: Final[ChatCompletionSystemMessage] = {**message, "role": "system"}
    return translated


def _text_blocks(
    message: ChatCompletionSystemMessage,
) -> list[object]:  # mutable-ok: content block lists are the wire format callers isinstance-check
    content: Final = _system_content(message)
    cache_control: Final = _message_cache_control(message)
    if not isinstance(content, str):
        return _with_message_cache_control(content, cache_control)
    if not content:
        return []
    if cache_control:
        cached_block: Final[ChatCompletionTextObject] = {
            "type": "text",
            "text": content,
            "cache_control": cache_control,
        }
        return [cached_block]
    text_block: Final[ChatCompletionTextObject] = {"type": "text", "text": content}
    return [text_block]


def _lone_text_block_text(blocks: list[object]) -> str | None:
    match blocks:
        case [{"type": "text", "text": str() as text}] if blocks == [{"type": "text", "text": text}]:
            return text
        case _:
            return None


def _plain_text(message: ChatCompletionSystemMessage) -> str | None:
    if _message_cache_control(message):
        return None
    content: Final = _system_content(message)
    text: Final = content if isinstance(content, str) else _lone_text_block_text(content)
    if text is None or text.startswith(ANTHROPIC_BILLING_METADATA_PREFIX):
        return None
    return text


def _is_system_message(
    message: AllMessageValues,
) -> TypeIs[ChatCompletionSystemMessage]:  # guard-ok: the role literal picks the TypedDict member
    return message["role"] == "system"


def _merged_system_content(run: Sequence[ChatCompletionSystemMessage]) -> SystemMessageContent:
    texts: Final = tuple(_plain_text(message) for message in run)
    if all(text is not None for text in texts):
        return "\n\n".join(text for text in texts if text)
    return list(itertools.chain.from_iterable(map(_text_blocks, run)))


def _merged_system_run(run: Sequence[ChatCompletionSystemMessage]) -> ChatCompletionSystemMessage:
    if len(run) == 1:
        return run[0]
    content: Final = _merged_system_content(run)
    named: Final = next((message for message in reversed(run) if "name" in message), None)
    if named is None:
        merged: Final[ChatCompletionSystemMessage] = {"role": "system", "content": content}
        return merged
    with_name: Final[ChatCompletionSystemMessage] = {"role": "system", "content": content, "name": named["name"]}
    return with_name


def _merged_system_runs(messages: Sequence[AllMessageValues]) -> Iterator[AllMessageValues]:
    for is_system, run in itertools.groupby(messages, key=_is_system_message):
        if is_system:
            yield _merged_system_run(tuple(message for message in run if _is_system_message(message)))
        else:
            yield from run


_INSTRUCTION_ROLES: Final = frozenset(("system", "developer"))


def _leading_instruction_block_length(roles: Sequence[str | None]) -> int:
    return next((index for index, role in enumerate(roles) if role not in _INSTRUCTION_ROLES), len(roles))


def _closing_instruction_block_start(roles: Sequence[str | None], leading_length: int) -> int:
    last_conversation_index: Final = next(
        (index for index in range(len(roles) - 1, leading_length - 1, -1) if roles[index] not in _INSTRUCTION_ROLES),
        None,
    )
    if last_conversation_index is None or roles[last_conversation_index] != "assistant":
        return len(roles)
    return last_conversation_index + 1


def hoisted_instruction_indices(roles: Sequence[str | None]) -> tuple[int, ...]:
    """
    Indices of the leading instruction block plus every developer item after it, except
    a developer block that closes the conversation right after an assistant turn, which
    stays in place so the request does not end on the assistant's turn. Items without a
    role (tool calls and their outputs) count as conversation.
    """
    leading_length: Final = _leading_instruction_block_length(roles)
    closing_start: Final = _closing_instruction_block_start(roles, leading_length)
    moved: Final = tuple(index for index in range(leading_length, closing_start) if roles[index] == "developer")
    return (*range(leading_length), *moved)


def _move_later_developer_messages_up(messages: Sequence[AllMessageValues]) -> tuple[AllMessageValues, ...]:
    hoisted: Final = hoisted_instruction_indices(tuple(message["role"] for message in messages))
    hoisted_set: Final = frozenset(hoisted)
    rest: Final = (index for index in range(len(messages)) if index not in hoisted_set)
    return tuple(messages[index] for index in (*hoisted, *rest))


def hoist_developer_messages_into_leading_system_message(
    messages: Sequence[AllMessageValues],
) -> Sequence[AllMessageValues]:
    """
    Translate `developer` role to `system` role for OpenAI-compatible backends whose
    chat template allows a single system message and only at the start: developer
    messages that arrive after the first user turn move into the leading system
    block, except a developer message that closes the conversation right after an
    assistant turn, which stays in place so the request does not end on the
    assistant's turn. Each run of consecutive system messages is then folded into
    one message in a single pass.
    """
    translated: Final = tuple(map(_as_system_message, _move_later_developer_messages_up(messages)))
    return tuple(_merged_system_runs(translated))
