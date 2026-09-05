"""
Utility functions for base LLM classes.
"""

import copy
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from itertools import groupby
from typing import Any, Final, TypeAlias

from openai.lib import _parsing, _pydantic
from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypeIs  # noqa: TID251  # TypeIs lands in typing only on 3.13

from litellm._logging import verbose_logger
from litellm.constants import ANTHROPIC_BILLING_METADATA_PREFIX
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionSystemMessage,
    ChatCompletionTextObject,
    ChatCompletionToolCallChunk,
)
from litellm.types.utils import Message, ProviderSpecificModelInfo, TokenCountResponse


class BaseTokenCounter(ABC):
    @abstractmethod
    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, Any]] | None,
        contents: list[dict[str, Any]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, Any]] | None = None,
        system: Any | None = None,
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

    @abstractmethod
    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        """
        Returns a list of models supported by this provider.
        """
        return []

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


def _convert_tool_response_to_message(
    tool_calls: list[ChatCompletionToolCallChunk],
) -> Message | None:
    """
    In JSON mode, Anthropic API returns JSON schema as a tool call, we need to convert it to a message to follow the OpenAI format

    """
    ## HANDLE JSON MODE - anthropic returns single function call
    json_mode_content_str: Final[str | None] = tool_calls[0]["function"].get("arguments")
    try:
        if json_mode_content_str is not None:
            args: Final = json.loads(json_mode_content_str)
            if isinstance(args, dict) and (values := args.get("values")) is not None:
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
    if not _parsing._completions.is_basemodel_type(response_format):
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


SystemMessageContent: TypeAlias = str | list[object]

_system_content_adapter: Final = TypeAdapter[SystemMessageContent](SystemMessageContent)


def _system_content(message: ChatCompletionSystemMessage) -> SystemMessageContent:
    return _system_content_adapter.validate_python(message["content"])


def _as_system_message(message: AllMessageValues) -> AllMessageValues:
    if message["role"] != "developer":
        return message
    verbose_logger.debug("Translating developer role to system role for non-OpenAI providers.")
    translated: Final[ChatCompletionSystemMessage] = {**message, "role": "system"}
    return translated


def map_developer_role_to_system_role(
    messages: Sequence[AllMessageValues],
) -> Sequence[AllMessageValues]:
    """
    Translate `developer` role to `system` role, keeping every message where the client put it.
    """
    return tuple(_as_system_message(message) for message in messages)


def _text_blocks(message: ChatCompletionSystemMessage) -> list[object]:  # mutable-ok: block lists are the wire format
    content: Final = _system_content(message)
    if not isinstance(content, str):
        return content
    if not content:
        return []  # mutable-ok: block lists are the wire format
    cache_control: Final = message.get("cache_control")
    if cache_control:
        cached_block: Final[ChatCompletionTextObject] = {
            "type": "text",
            "text": content,
            "cache_control": cache_control,
        }
        return [cached_block]  # mutable-ok: block lists are the wire format
    text_block: Final[ChatCompletionTextObject] = {"type": "text", "text": content}
    return [text_block]  # mutable-ok: block lists are the wire format


def _plain_text(message: ChatCompletionSystemMessage) -> str | None:
    content: Final = _system_content(message)
    if (
        not isinstance(content, str)
        or message.get("cache_control")
        or content.startswith(ANTHROPIC_BILLING_METADATA_PREFIX)
    ):
        return None
    return content


def _is_system_message(
    message: AllMessageValues,
) -> TypeIs[ChatCompletionSystemMessage]:  # guard-ok: the role literal picks the TypedDict member
    return message["role"] == "system"


def _merged_system_run(run: Sequence[ChatCompletionSystemMessage]) -> ChatCompletionSystemMessage:
    if len(run) == 1:
        return run[0]
    texts: Final = tuple(_plain_text(message) for message in run)
    content: Final[SystemMessageContent] = (
        "\n\n".join(text for text in texts if text)
        if all(text is not None for text in texts)
        else [
            block for message in run for block in _text_blocks(message)
        ]  # mutable-ok: block lists are the wire format
    )
    named: Final = next((message for message in reversed(run) if "name" in message), None)
    if named is None:
        merged: Final[ChatCompletionSystemMessage] = {"role": "system", "content": content}
        return merged
    with_name: Final[ChatCompletionSystemMessage] = {"role": "system", "content": content, "name": named["name"]}
    return with_name


def _merged_system_runs(messages: Sequence[AllMessageValues]) -> Iterator[AllMessageValues]:
    for is_system, run in groupby(messages, key=_is_system_message):
        if is_system:
            yield _merged_system_run(tuple(message for message in run if _is_system_message(message)))
        else:
            yield from run


def _leading_system_block_length(messages: Sequence[AllMessageValues]) -> int:
    return next(
        (index for index, message in enumerate(messages) if message["role"] not in ("system", "developer")),
        len(messages),
    )


def _move_later_developer_messages_up(messages: Sequence[AllMessageValues]) -> tuple[AllMessageValues, ...]:
    leading_length: Final = _leading_system_block_length(messages)
    later: Final = messages[leading_length:]
    hoisted: Final = tuple(message for message in later if message["role"] == "developer")
    if hoisted:
        verbose_logger.debug(
            "Hoisting %d developer message(s) into the leading system block for OpenAI-compatible backends.",
            len(hoisted),
        )
    return (
        *messages[:leading_length],
        *hoisted,
        *(message for message in later if message["role"] != "developer"),
    )


def hoist_developer_messages_into_leading_system_message(
    messages: Sequence[AllMessageValues],
) -> Sequence[AllMessageValues]:
    """
    Translate `developer` role to `system` role for OpenAI-compatible backends whose
    chat template allows a single system message and only at the start: developer
    messages that arrive after the first user turn move into the leading system
    block, and each run of consecutive system messages is folded into one message
    in a single pass.
    """
    translated: Final = tuple(map(_as_system_message, _move_later_developer_messages_up(messages)))
    return tuple(_merged_system_runs(translated))
