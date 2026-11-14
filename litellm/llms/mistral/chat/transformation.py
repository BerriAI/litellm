"""
Transformation logic from OpenAI /v1/chat/completion format to Mistral's /chat/completion format.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.mistral.ai/api/
"""

import reprlib
from collections.abc import AsyncIterator, Coroutine, Iterator, Mapping, Sequence
from itertools import accumulate, count
from typing import TYPE_CHECKING, Final, Literal, cast, get_type_hints, overload
from urllib.parse import urlparse

import httpx
from pydantic import JsonValue, TypeAdapter
from typing_extensions import TypeIs, assert_never  # noqa: TID251  # these are absent from typing on Python 3.10

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import allowed_openai_param_names, as_object_list, as_str_mapping
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import is_str_keyed_dict
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
    strip_litellm_internal_message_fields,
    strip_none_values_from_message,
)
from litellm.litellm_core_utils.reasoning_effort_capability import (
    REASONING_EFFORT_ADVERTISEMENT_ORDER,
    declared_reasoning_efforts_for_model,
    nearest_declared_reasoning_effort,
)
from litellm.litellm_core_utils.reasoning_effort_utils import reasoning_effort_from_thinking, warn_dropped_reasoning
from litellm.llms.mistral.common_utils import get_mistral_chat_credentials
from litellm.llms.openai.chat.gpt_transformation import (
    OpenAIChatCompletionStreamingHandler,
    OpenAIGPTConfig,
)
from litellm.types.llms.mistral import (
    MistralCacheKeyParams,
    MistralFilePart,
    MistralReasoningEffortParams,
    MistralRequestRouting,
    MistralTextBlock,
    MistralThinkingBlock,
    MistralToolCallMessage,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantMessage,
    ChatCompletionTextObject,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
)
from litellm.types.utils import ModelResponse, ModelResponseStream
from litellm.utils import convert_to_model_response_object, supports_reasoning

if TYPE_CHECKING:
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer
    from litellm.llms.base_llm.base_utils import BaseTokenCounter


def _accepted_reasoning_effort(model: str, requested: str, custom_llm_provider: str) -> str:
    declared: Final = declared_reasoning_efforts_for_model(model, custom_llm_provider)
    if declared is None:
        return requested
    accepted: Final = nearest_declared_reasoning_effort(requested, declared)
    if accepted != requested:
        verbose_logger.debug(
            "%s: %s takes reasoning_effort %s, sending %s in place of %s",
            custom_llm_provider,
            model,
            declared,
            accepted,
            requested,
        )
    return accepted


def _thinking_as_reasoning_effort(thinking: object, *, model: str, drop_params: bool) -> str | None:
    match thinking:
        case {"type": "disabled"}:
            return "none"
        case {"type": "adaptive"}:
            return "medium"
        case _:
            return reasoning_effort_from_thinking(thinking, model=model, drop_params=drop_params)


_NO_TOOL_CALL_TURN: Final[tuple[int, frozenset[str]]] = (-1, frozenset())
_MISTRAL_REQUEST_ROUTING: Final = TypeAdapter(MistralRequestRouting)
_MISTRAL_CACHE_KEY_PARAMS: Final = TypeAdapter(MistralCacheKeyParams)
_MISTRAL_REASONING_EFFORT_PARAMS: Final = TypeAdapter(MistralReasoningEffortParams)
_MISTRAL_FILE_PART: Final = TypeAdapter(MistralFilePart)
_DECODED_JSON: Final = TypeAdapter(object)
_REASONING_LEVELS: Final = frozenset(REASONING_EFFORT_ADVERTISEMENT_ORDER)
_PROVIDER_DEFAULT_EFFORT: Final = "default"
_NON_LEVEL_EFFORT_REASON: Final = "the Mistral API takes it only as a level such as 'high'."
_UNUSABLE_TOOL_CHOICE_REASON: Final = (
    "the Mistral API takes tool_choice only as 'auto', 'none', 'any', 'required', "
    "or {'type': 'function', 'function': {'name': <tool name>}}."
)


def _reasoning_level_and_summary(reasoning_effort: object) -> tuple[object, object]:
    requested: Final = as_str_mapping(reasoning_effort)
    if requested is None:
        return reasoning_effort, None
    return requested.get("effort"), requested.get("summary")


def _is_json_array(value: object) -> TypeIs[Sequence[object]]:  # guard-ok: a decoded JSON array is a list
    return isinstance(value, list)


def _public_reasoning_level(result: str | UnsupportedParamsError | None) -> str | None:
    match result:
        case UnsupportedParamsError():
            raise result
        case str() | None:
            return result
        case _:
            assert_never(result)


class MistralConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.mistral.ai/api/

    The class `MistralConfig` provides configuration for the Mistral's Chat API interface. Below are the parameters:

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2. API Default - 0.7.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling. API Default - 1.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion. API Default - null.

    - `tools` (list or null): A list of available tools for the model. Use this to specify functions for which the model can generate JSON inputs.

    - `tool_choice` (string - 'auto'/'any'/'none', an object naming one function such as {"type": "function", "function": {"name": "my_function"}}, or null): Specifies if/how functions are called. If set to none the model won't call a function and will generate a message instead. If set to auto the model can choose to either generate a message or call a function. If set to any the model is forced to call a function. Default - 'auto'.

    - `stop` (string or array of strings): Stop generation if this token is detected. Or if one of these tokens is detected when providing an array

    - `random_seed` (integer or null): The seed to use for random sampling. If set, different calls will generate deterministic results.

    - `safe_prompt` (boolean): Whether to inject a safety prompt before all conversations. API Default - 'false'.

    - `response_format` (object or null): An object specifying the format that the model must output. Setting to { "type": "json_object" } enables JSON mode, which guarantees the message the model generates is in JSON. When using JSON mode you MUST also instruct the model to produce JSON yourself with a system or a user message.
    """

    temperature: int | None = None
    top_p: int | None = None
    max_tokens: int | None = None
    tools: list | None = None
    tool_choice: Literal["auto", "any", "none"] | ChatCompletionToolChoiceObjectParam | None = None
    random_seed: int | None = None
    safe_prompt: bool | None = None
    response_format: dict | None = None
    stop: str | list | None = None

    def __init__(
        self,
        temperature: int | None = None,
        top_p: int | None = None,
        max_tokens: int | None = None,
        tools: list | None = None,
        tool_choice: Literal["auto", "any", "none"] | ChatCompletionToolChoiceObjectParam | None = None,
        random_seed: int | None = None,
        safe_prompt: bool | None = None,
        response_format: dict | None = None,
        stop: str | list | None = None,
    ) -> None:
        locals_: Final[Mapping[str, object]] = dict(locals())
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    @property
    def custom_llm_provider(self) -> str:
        return "mistral"

    def get_token_counter(self) -> "BaseTokenCounter | None":
        from litellm.llms.mistral.count_tokens.token_counter import MistralTokenCounter

        return MistralTokenCounter()

    def accepts_reasoning(self, model: str) -> bool:
        return "magistral" in model.lower() or supports_reasoning(
            model=model, custom_llm_provider=self.custom_llm_provider
        )

    def get_supported_openai_params(self, model: str) -> list[str]:
        is_magistral: Final = "magistral" in model.lower()
        accepts_reasoning_effort: Final = self.accepts_reasoning(model)
        accepts_prompt_cache_key: Final = self.custom_llm_provider == "mistral"
        return [
            "stream",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "seed",
            "stop",
            "response_format",
            "parallel_tool_calls",
            *(("thinking",) if is_magistral else ()),
            *(("reasoning_effort",) if accepts_reasoning_effort else ()),
            *(("prompt_cache_key",) if accepts_prompt_cache_key else ()),
        ]

    @staticmethod
    def map_tool_choice(tool_choice: object) -> str | ChatCompletionToolChoiceObjectParam | None:
        match tool_choice:
            case str(choice) if choice in ("auto", "none"):
                return choice
            case "required" | "any":
                return "any"
            case {"type": "function", "function": {"name": str(name)}} if name:
                return ChatCompletionToolChoiceObjectParam(
                    type="function", function=ChatCompletionToolChoiceFunctionParam(name=name)
                )
            case _:
                return None

    def _tool_choice_to_send(
        self, model: str, tool_choice: object, drop_params: bool
    ) -> str | ChatCompletionToolChoiceObjectParam | UnsupportedParamsError | None:
        mapped: Final = self.map_tool_choice(tool_choice)
        if mapped is not None:
            return mapped
        if drop_params or litellm.drop_params is True:
            warn_dropped_reasoning(
                "tool_choice", tool_choice, model, _UNUSABLE_TOOL_CHOICE_REASON, drop_params_gated=True
            )
            return None
        return UnsupportedParamsError(
            message=(
                f"Unsupported tool_choice={reprlib.repr(tool_choice)} for model={model}, since "
                f"{_UNUSABLE_TOOL_CHOICE_REASON} "
                "To drop it, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
            ),
            llm_provider=self.custom_llm_provider,
            model=model,
        )

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        requested: Final = _MISTRAL_REASONING_EFFORT_PARAMS.validate_python(non_default_params)
        reasoning_level: Final = (
            _public_reasoning_level(self._reasoning_level_to_send(model, requested, drop_params))
            if "reasoning_effort" in requested
            else None
        )
        leaves_level_to_mistral: Final = (
            _reasoning_level_and_summary(requested.get("reasoning_effort"))[0] == _PROVIDER_DEFAULT_EFFORT
        )
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "max_completion_tokens":  # max_completion_tokens should take priority
                optional_params["max_tokens"] = value
            if param == "tools":
                # Clean tools to remove problematic schema fields for Mistral API
                optional_params["tools"] = self._clean_tool_schema_for_mistral(value)
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "tool_choice" and "tools" in non_default_params and non_default_params["tools"]:
                match self._tool_choice_to_send(
                    model,
                    cast(object, value),  # cast-ok: untyped request params, the match narrows it
                    drop_params,
                ):
                    case UnsupportedParamsError() as unusable_tool_choice:
                        raise unusable_tool_choice
                    case None:
                        pass
                    case tool_choice:
                        optional_params["tool_choice"] = tool_choice
            if param == "seed":
                optional_params["extra_body"] = {"random_seed": value}
            if param == "reasoning_effort" and reasoning_level is not None:
                optional_params["reasoning_effort"] = reasoning_level
            if (
                param == "thinking"
                and reasoning_level is None
                and not leaves_level_to_mistral
                and (thinking_effort := _thinking_as_reasoning_effort(value, model=model, drop_params=drop_params))
                is not None
            ):
                optional_params["reasoning_effort"] = _accepted_reasoning_effort(
                    model, thinking_effort, self.custom_llm_provider
                )
            if param in ("temperature", "top_p", "stop", "response_format", "parallel_tool_calls", "prompt_cache_key"):
                optional_params[param] = value
        return optional_params

    def _reasoning_level_to_send(
        self, model: str, requested: MistralReasoningEffortParams, drop_params: bool
    ) -> str | UnsupportedParamsError | None:
        requested_effort: Final = requested.get("reasoning_effort")
        level, summary = _reasoning_level_and_summary(requested_effort)
        leaves_level_to_mistral: Final = level == _PROVIDER_DEFAULT_EFFORT
        sendable: Final = (
            _accepted_reasoning_effort(model, level, self.custom_llm_provider)
            if isinstance(level, str) and level in _REASONING_LEVELS
            else None
        )
        if summary is None and (sendable is not None or leaves_level_to_mistral):
            return sendable
        if drop_params or litellm.drop_params is True:
            if sendable is None and level is not None and not leaves_level_to_mistral:
                warn_dropped_reasoning(
                    "reasoning_effort", requested_effort, model, _NON_LEVEL_EFFORT_REASON, drop_params_gated=True
                )
            return sendable
        return UnsupportedParamsError(
            message=(
                f"{self.custom_llm_provider} does not support parameters: ['reasoning_effort'], for model={model}, "
                "since the Mistral API takes it only as a level such as 'high', with no reasoning summary. "
                "To drop it, set `litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
            ),
            llm_provider=self.custom_llm_provider,
            model=model,
        )

    def accepts_prompt_cache_key_at(self, api_base: str | None) -> bool:
        resolved_api_base, _ = self._get_openai_compatible_provider_info(api_base, None)
        hostname: Final = urlparse(resolved_api_base).hostname or ""
        return hostname == "mistral.ai" or hostname.endswith(".mistral.ai")

    @staticmethod
    def takes_prompt_cache_key_value(prompt_cache_key: object) -> bool:
        return isinstance(prompt_cache_key, str) and prompt_cache_key.isascii()

    def _prompt_cache_key_rejection(self, prompt_cache_key: object, api_base: str | None) -> str | None:
        if not self.accepts_prompt_cache_key_at(api_base):
            return "since only the Mistral API (*.mistral.ai) takes it"
        if not self.takes_prompt_cache_key_value(prompt_cache_key):
            return "since the Mistral API takes it only as a string of ASCII characters"
        return None

    def _prompt_cache_key_withheld(self, model: str, prompt_cache_key: object, routing: MistralRequestRouting) -> bool:
        requested_api_base: Final = routing.get("api_base")
        api_base: Final = requested_api_base if isinstance(requested_api_base, str) else None
        if "prompt_cache_key" in allowed_openai_param_names(routing.get("allowed_openai_params")):
            return False
        rejection: Final = self._prompt_cache_key_rejection(prompt_cache_key, api_base)
        if rejection is None:
            return False
        if litellm.drop_params is True or routing.get("drop_params") is True:
            return True
        requested_provider: Final = routing.get("custom_llm_provider")
        provider: Final = requested_provider if isinstance(requested_provider, str) else self.custom_llm_provider
        raise UnsupportedParamsError(
            message=(
                f"{provider} does not support parameters: ['prompt_cache_key'], for model={model}, {rejection}. "
                "To send it anyway, pass `allowed_openai_params=['prompt_cache_key']`. To drop it, set "
                "`litellm.drop_params=True` or for proxy:\n\n`litellm_settings:\n drop_params: true`\n"
            ),
            llm_provider=provider,
            model=model,
        )

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return get_mistral_chat_credentials(api_base=None, api_key=api_key)[1]

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return get_mistral_chat_credentials(api_base=api_base, api_key=None)[0]

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: matches BaseConfig contract
        from litellm.utils import bundled_provider_models

        return list(bundled_provider_models(self.custom_llm_provider))

    def _get_openai_compatible_provider_info(self, api_base: str | None, api_key: str | None) -> tuple[str, str | None]:
        return get_mistral_chat_credentials(api_base, api_key)

    # fmt: off

    @overload
    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[object, object, list[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: list[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> list[AllMessageValues]: 
        ...
    # fmt: on

    def _transform_messages(
        self, messages: list[AllMessageValues], model: str, is_async: bool = False
    ) -> list[AllMessageValues] | Coroutine[object, object, list[AllMessageValues]]:
        """
        - handles scenario where content is list and not string
        - content list is just text, and no images
        - if image passed in, then just return as is (user-intended)
        - if `name` is passed, then drop it for mistral API: https://github.com/BerriAI/litellm/issues/6696

        Motivation: mistral api doesn't support content as a list.
        The above statement is not valid now. Need to plan to remove all the #1,2,3
        Mistral API supports content as a list.
        """
        keeps_reasoning: Final = self.accepts_reasoning(model)
        replayed_messages: Final = MistralConfig._replayed_messages(messages, keeps_reasoning)

        ## 1. If 'image_url' or 'file' in content, then transform with base class and mistral-specific handling
        if MistralConfig._sends_content_as_is(replayed_messages):
            if is_async:
                return self._transform_messages_async(list(replayed_messages), model)
            else:
                return self._transform_messages_sync(list(replayed_messages), model)

        ## 2. If content is list, then convert to string
        ## 3. Handle name in message
        prepared_messages: Final = tuple(
            MistralConfig._handle_tool_call_message(m)
            for m in MistralConfig._flattened_messages(replayed_messages, keeps_reasoning)
        )
        new_messages: Final[list[AllMessageValues]] = [
            strip_none_values_from_message(MistralConfig._with_tool_reply_content(m))
            for m, is_sent in zip(prepared_messages, MistralConfig._sent_flags(prepared_messages), strict=True)
            if is_sent
        ]

        if is_async:
            return super()._transform_messages(new_messages, model, True)
        else:
            return super()._transform_messages(new_messages, model, False)

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    def transform_messages(
        self,
        messages: list[AllMessageValues],  # mutable-ok: mirrors override contract
        model: str,
        is_async: bool = False,
    ) -> (
        list[AllMessageValues]  # mutable-ok: mirrors override contract
        | Coroutine[object, object, list[AllMessageValues]]
    ):
        return self._transform_messages(messages, model, is_async)

    async def _transform_messages_async(self, messages: list[AllMessageValues], model: str) -> list[AllMessageValues]:
        """
        Handle modification of messages for Mistral API in an async context.
        """
        # Call parent async method to handle basic transformations
        # and then apply Mistral-specific handling for files
        messages = await super()._transform_messages(messages, model, True)
        messages = self._handle_message_with_file(messages)
        return messages

    def _transform_messages_sync(self, messages: list[AllMessageValues], model: str) -> list[AllMessageValues]:
        """Handle modification of messages for Mistral API in a sync context."""
        # Call parent sync method to handle basic transformations
        # and then apply Mistral-specific handling for files
        # This is the sync version of the async method above
        messages = super()._transform_messages(messages, model, False)
        messages = self._handle_message_with_file(messages)
        return messages

    @staticmethod
    def _with_mistral_file_id(block: Mapping[str, object]) -> Mapping[str, object]:
        file_part: Final = block.get("file") if block.get("type") == "file" else None
        file_id: Final = (
            _MISTRAL_FILE_PART.validate_python(file_part).get("file_id") if isinstance(file_part, Mapping) else None
        )
        if not file_id:
            return block
        return {**{key: value for key, value in block.items() if key != "file"}, "file_id": file_id}

    @classmethod
    def _with_mistral_file_ids(cls, message: AllMessageValues) -> AllMessageValues:
        content: Final = message.get("content")
        if not isinstance(content, list):
            return message
        blocks: Final[tuple[Mapping[str, object], ...]] = tuple(content)
        return cast(  # cast-ok: only file blocks change, to the file_id shape the Mistral API takes
            AllMessageValues, {**message, "content": [cls._with_mistral_file_id(block) for block in blocks]}
        )

    def _handle_message_with_file(self, messages: list[AllMessageValues]) -> list[AllMessageValues]:
        """
        Mistral API supports only 'file_id' in message content with type 'file'.
        """
        return [self._with_mistral_file_ids(m) for m in messages]

    @classmethod
    def _clean_tool_schema_for_mistral(cls, tools: list[JsonValue]) -> list[JsonValue]:
        """
        Clean tool schemas to remove fields that cause issues with Mistral API.

        Removes:
        - $id and $schema fields (cause grammar validation errors)
        - additionalProperties=False (causes OpenAI API schema errors)
        - strict field (not supported by Mistral)

        Args:
            tools: List of tool definitions
            max_depth: Maximum recursion depth for schema cleaning (default: 10)

        Returns:
            Cleaned tools list
        """
        if not tools:
            return tools

        import copy

        from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
        from litellm.utils import remove_json_schema_refs

        cleaned_tools = copy.deepcopy(tools)

        # Apply all cleaning functions with max_depth protection
        cleaned_tools = remove_json_schema_refs(cleaned_tools, max_depth=DEFAULT_MAX_RECURSE_DEPTH)

        return cleaned_tools

    @staticmethod
    def _is_mistral_think_chunk(part: object) -> bool:
        match part:
            case {"type": "thinking", "thinking": list()}:
                return True
            case _:
                return False

    @staticmethod
    def _text_item_text(item: object) -> str | None:
        match item:
            case {"type": "text", "text": str(text)}:
                return text
            case _:
                return None

    @staticmethod
    def think_chunk_texts(part: object) -> tuple[str, ...] | None:
        match part:
            case {"type": "thinking", "thinking": object() as items} if _is_json_array(items):
                return tuple(text for text in map(MistralConfig._text_item_text, items) if text is not None)
            case _:
                return None

    @staticmethod
    def _sendable_think_chunk(
        chunk: object, items: Sequence[object], may_stay_open: bool
    ) -> MistralThinkingBlock | None:
        text_items: Final = [
            MistralTextBlock(type="text", text=text)
            for text in map(MistralConfig._text_item_text, items)
            if text is not None
        ]
        if not text_items:
            return None
        match chunk:
            case {"closed": False} if not may_stay_open:
                return MistralThinkingBlock(type="thinking", thinking=text_items)
            case {"closed": bool(closed)}:
                return MistralThinkingBlock(type="thinking", thinking=text_items, closed=closed)
            case _:
                return MistralThinkingBlock(type="thinking", thinking=text_items)

    @staticmethod
    def _sendable_part(part: object, may_stay_open: bool) -> object | None:
        match part:
            case {"type": "thinking", "thinking": object() as items} if _is_json_array(items):
                return MistralConfig._sendable_think_chunk(part, items, may_stay_open)
            case _:
                return part

    @staticmethod
    def _with_sendable_think_chunks(
        message: AllMessageValues, content_parts: tuple[object, ...], may_end_open: bool
    ) -> AllMessageValues:
        last_position: Final = len(content_parts) - 1
        sendable_parts: Final = [
            sendable
            for position, part in enumerate(content_parts)
            if (sendable := MistralConfig._sendable_part(part, may_end_open and position == last_position)) is not None
        ]
        return cast(  # cast-ok: Mistral's ThinkChunk is outside the OpenAI assistant content union
            AllMessageValues, {**message, "content": sendable_parts or None}
        )

    @staticmethod
    def _thinking_block_text(block: object) -> str:
        match block:
            case {"type": "thinking", "thinking": str(thinking)}:
                return thinking
            case _:
                return ""

    @staticmethod
    def replayed_reasoning_text(message: Mapping[str, object]) -> str | None:
        reasoning_content: Final = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return reasoning_content
        thinking_blocks: Final = as_object_list(message.get("thinking_blocks")) or ()
        thinking_text: Final = "".join(map(MistralConfig._thinking_block_text, thinking_blocks))
        return thinking_text if thinking_text.strip() else None

    @staticmethod
    def _assistant_content_parts(message: ChatCompletionAssistantMessage) -> tuple[object, ...] | None:
        match message.get("content"):
            case None | "":
                return ()
            case str(text):
                return (ChatCompletionTextObject(type="text", text=text),)
            case list(parts) | tuple(parts):
                return tuple(parts)
            case _:
                return None

    @staticmethod
    def _is_prefix_turn(message: object) -> bool:
        match message:
            case {"prefix": True}:
                return True
            case _:
                return False

    def replayed_turns(self, messages: Sequence[Mapping[str, object]], model: str) -> tuple[AllMessageValues, ...]:
        chat_messages: Final = cast(  # cast-ok: a count holds the chat turns as plain mappings
            list[AllMessageValues], [dict(message) for message in messages]
        )
        return MistralConfig._replayed_messages(chat_messages, self.accepts_reasoning(model))

    def sent_turns(self, messages: Sequence[Mapping[str, object]], model: str) -> tuple[bool, ...]:
        replayed_messages: Final = self.replayed_turns(messages, model)
        if MistralConfig._sends_content_as_is(replayed_messages):
            return tuple(True for _ in replayed_messages)
        return MistralConfig._sent_flags(
            tuple(
                map(
                    MistralConfig._handle_tool_call_message,
                    MistralConfig._flattened_messages(replayed_messages, self.accepts_reasoning(model)),
                )
            )
        )

    @staticmethod
    def _replayed_messages(messages: Sequence[AllMessageValues], keeps_reasoning: bool) -> tuple[AllMessageValues, ...]:
        final_position: Final = len(messages) - 1
        return tuple(
            MistralConfig._with_replayed_reasoning(m, keeps_reasoning, position == final_position)
            for position, m in enumerate(messages)
        )

    @staticmethod
    def _sends_content_as_is(messages: Sequence[Mapping[str, object]]) -> bool:
        return any(map(MistralConfig._has_image_or_file_part, messages))

    @staticmethod
    def _has_image_or_file_part(message: Mapping[str, object]) -> bool:
        content: Final = message.get("content")
        return _is_json_array(content) and any(map(MistralConfig._is_image_or_file_part, content))

    @staticmethod
    def _is_image_or_file_part(part: object) -> bool:
        match part:
            case {"type": "image_url" | "file"}:
                return True
            case _:
                return False

    @staticmethod
    def _flattened_messages(
        messages: Sequence[AllMessageValues], keeps_reasoning: bool
    ) -> tuple[AllMessageValues, ...]:
        return tuple(
            MistralConfig._handle_name_in_message(MistralConfig._with_flattened_content(m, keeps_reasoning))
            for m in messages
        )

    @staticmethod
    def _sent_flags(messages: Sequence[AllMessageValues]) -> tuple[bool, ...]:
        empty_flags: Final = tuple(map(MistralConfig._is_empty_assistant_message, messages))
        tool_call_turns: Final = accumulate(
            zip(count(), messages, empty_flags, strict=False),
            MistralConfig._tool_call_turn,
            initial=_NO_TOOL_CALL_TURN,
        )
        tool_reply_keys: Final = tuple(map(MistralConfig._tool_reply_key, messages, tool_call_turns))
        first_reply_positions: Final = {
            key: position for position, key in reversed(tuple(enumerate(tool_reply_keys))) if key is not None
        }
        keys_answered_with_content: Final = frozenset(
            key for key, is_empty in zip(tool_reply_keys, empty_flags, strict=True) if not is_empty
        )
        answering_positions: Final = frozenset(
            position for key, position in first_reply_positions.items() if key not in keys_answered_with_content
        )
        return tuple(not is_empty or position in answering_positions for position, is_empty in enumerate(empty_flags))

    @staticmethod
    def _with_replayed_reasoning(
        message: AllMessageValues, keeps_reasoning: bool, is_final_turn: bool
    ) -> AllMessageValues:
        if message["role"] != "assistant":
            return message
        stripped_message: Final = strip_litellm_internal_message_fields(message)
        content_parts: Final = MistralConfig._assistant_content_parts(message)
        if not keeps_reasoning:
            return MistralConfig._without_think_chunks(stripped_message, content_parts)
        if content_parts is not None and any(map(MistralConfig._is_mistral_think_chunk, content_parts)):
            return MistralConfig._with_sendable_think_chunks(
                stripped_message, content_parts, is_final_turn and MistralConfig._is_prefix_turn(message)
            )
        reasoning_text: Final = MistralConfig.replayed_reasoning_text(message)
        has_turn_output: Final = bool(content_parts) or bool(message.get("tool_calls"))
        if reasoning_text is None or content_parts is None or not has_turn_output:
            return stripped_message
        think_chunk: Final = MistralThinkingBlock(
            type="thinking", thinking=[MistralTextBlock(type="text", text=reasoning_text)], closed=True
        )
        return cast(  # cast-ok: Mistral's ThinkChunk is outside the OpenAI assistant content union
            AllMessageValues, {**stripped_message, "content": [think_chunk, *content_parts]}
        )

    @staticmethod
    def _without_think_chunks(message: AllMessageValues, content_parts: tuple[object, ...] | None) -> AllMessageValues:
        if content_parts is None or not any(map(MistralConfig._is_mistral_think_chunk, content_parts)):
            return message
        kept_parts: Final = [part for part in content_parts if not MistralConfig._is_mistral_think_chunk(part)]
        return cast(  # cast-ok: an assistant turn takes a content list or no content
            AllMessageValues, {**message, "content": kept_parts or None}
        )

    @staticmethod
    def _with_flattened_content(message: AllMessageValues, keeps_reasoning: bool) -> AllMessageValues:
        flattened_text: Final = convert_content_list_to_str(message)
        match message:
            case {"role": "assistant", "content": list(content)} if keeps_reasoning and any(
                map(MistralConfig._is_mistral_think_chunk, content)
            ):
                text_parts: Final = (
                    (ChatCompletionTextObject(type="text", text=flattened_text),) if flattened_text else ()
                )
                think_chunks: Final = tuple(filter(MistralConfig._is_mistral_think_chunk, content))
                sent_chunks: Final = tuple(
                    sendable
                    for chunk in think_chunks
                    if (sendable := MistralConfig._sendable_part(chunk, not text_parts)) is not None
                )
                return cast(  # cast-ok: Mistral's ThinkChunk is outside the OpenAI assistant content union
                    AllMessageValues, {**message, "content": [*sent_chunks, *text_parts]}
                )
            case _ if flattened_text:
                return cast(  # cast-ok: every message role takes str content
                    AllMessageValues, {**message, "content": flattened_text}
                )
            case _:
                return message

    @classmethod
    def _handle_name_in_message(cls, message: AllMessageValues) -> AllMessageValues:
        """
        Mistral API only supports `name` in tool messages

        If role == tool, then we keep `name` if it's not an empty string
        Otherwise, we drop `name`
        """
        _name: Final = message.get("name")

        if _name is not None:
            # Remove name if not a tool message
            if message["role"] != "tool" or isinstance(_name, str) and len(_name.strip()) == 0:
                message_without_name: Final = {k: v for k, v in message.items() if k != "name"}
                return cast(AllMessageValues, message_without_name)  # cast-ok: the same message minus name

        return message

    @classmethod
    def _handle_tool_call_message(cls, message: AllMessageValues) -> AllMessageValues:
        """
        Mistral API only supports tool_calls in Messages in `MistralToolCallMessage` spec
        """
        _tool_calls: Final = message.get("tool_calls")
        mistral_tool_calls: Final[list[MistralToolCallMessage]] = []
        if _tool_calls is not None and isinstance(_tool_calls, list):
            for _tool in _tool_calls:
                _tool_call_message = MistralToolCallMessage(
                    id=_tool.get("id"),
                    type="function",
                    function=_tool.get("function"),
                )
                mistral_tool_calls.append(_tool_call_message)
            message["tool_calls"] = mistral_tool_calls
        return message

    @staticmethod
    def _tool_call_id(tool_call: object) -> str | None:
        match tool_call:
            case {"id": str(tool_call_id)}:
                return tool_call_id
            case _:
                return None

    @staticmethod
    def _tool_call_turn(
        turn: tuple[int, frozenset[str]], message_entry: tuple[int, AllMessageValues, bool]
    ) -> tuple[int, frozenset[str]]:
        match message_entry:
            case (position, {"role": "assistant", "tool_calls": list(tool_calls)}, _):
                return position, frozenset(filter(None, map(MistralConfig._tool_call_id, tool_calls)))
            case (_, {"role": "tool"}, _) | (_, _, True):
                return turn
            case _:
                return _NO_TOOL_CALL_TURN

    @staticmethod
    def _tool_reply_key(message: AllMessageValues, turn: tuple[int, frozenset[str]]) -> tuple[int, str] | None:
        match message, turn:
            case {"role": "tool", "tool_call_id": str(tool_call_id)}, (position, tool_call_ids) if (
                tool_call_id in tool_call_ids
            ):
                return position, tool_call_id
            case _:
                return None

    @staticmethod
    def _with_tool_reply_content(message: AllMessageValues) -> AllMessageValues:
        match message:
            case {"role": "tool"} if not message.get("content"):
                return {**message, "content": ""}
            case _:
                return message

    @classmethod
    def _is_empty_assistant_message(cls, message: AllMessageValues) -> bool:
        """
        Mistral API does not support empty string in assistant content.
        """
        from litellm.types.llms.openai import ChatCompletionAssistantMessage

        set_keys: Final = get_type_hints(ChatCompletionAssistantMessage).keys()

        all_expected_values_are_empty = True
        for key in set_keys:
            if key != "role" and message.get(key) is not None:
                if key == "content" and message.get(key) == "":
                    continue
                else:
                    all_expected_values_are_empty = False
                    break
        return all_expected_values_are_empty

    @staticmethod
    def _handle_empty_content_response(response_data: dict) -> dict:
        """
        Handle Mistral-specific behavior where empty string content should be converted to None.

        Mistral API sometimes returns empty string content ('') instead of null,
        which can cause issues with downstream processing.

        Args:
            response_data: The raw response data from Mistral API

        Returns:
            dict: The response data with empty string content converted to None
        """
        if response_data.get("choices") and len(response_data["choices"]) > 0:
            for choice in response_data["choices"]:
                if choice.get("message") and choice["message"].get("content") == "":
                    choice["message"]["content"] = None
        return response_data

    @staticmethod
    def _convert_thinking_block_to_reasoning_content(
        thinking_blocks: MistralThinkingBlock,
    ) -> str:
        """
        Convert Mistral thinking blocks to reasoning content.
        """
        return "\n".join([block.get("text", "") for block in thinking_blocks["thinking"]])

    @staticmethod
    def _handle_content_list_to_str_conversion(response_data: dict) -> dict:
        """
        Handle Mistral's content list format and extract thinking content.

        Map mistral's content list to string and extract thinking blocks:
        - Thinking block -> reasoning_content field
        - Text block -> content field
        """

        if response_data.get("choices") and len(response_data["choices"]) > 0:
            for choice in response_data["choices"]:
                if choice.get("message") and choice["message"].get("content"):
                    content = choice["message"]["content"]

                    # Only process if content is a list
                    if isinstance(content, list):
                        thinking_content = ""
                        text_content = ""

                        # Process each content block
                        for block in content:
                            if block.get("type") == "thinking":
                                thinking_blocks = block.get("thinking", [])
                                thinking_texts = []
                                for thinking_block in thinking_blocks:
                                    if thinking_block.get("type") == "text":
                                        thinking_texts.append(thinking_block.get("text", ""))
                                thinking_content = "\n".join(thinking_texts)
                            elif block.get("type") == "text":
                                text_content = block.get("text", "")

                        # Set the extracted content
                        choice["message"]["content"] = text_content
                        if thinking_content:
                            choice["message"]["reasoning_content"] = thinking_content

        return response_data

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.

        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        cache_key_params: Final = _MISTRAL_CACHE_KEY_PARAMS.validate_python(optional_params)
        withheld_params: Final = (
            ("client_metadata", "prompt_cache_key")
            if "prompt_cache_key" in cache_key_params
            and self._prompt_cache_key_withheld(
                model, cache_key_params["prompt_cache_key"], _MISTRAL_REQUEST_ROUTING.validate_python(litellm_params)
            )
            else ("client_metadata",)
        )
        upstream_params: Final = {key: value for key, value in optional_params.items() if key not in withheld_params}

        # Call parent transform_request which handles _transform_messages
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=upstream_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Transform the raw response from Mistral API.
        Handles Mistral-specific behavior like converting empty string content to None
        and extracting thinking content from content lists.
        """
        logging_obj.post_call(original_response=raw_response.text)
        logging_obj.model_call_details["response_headers"] = raw_response.headers

        # Handle Mistral-specific response transformations
        decoded_response: Final = _DECODED_JSON.validate_python(raw_response.json())
        if not is_str_keyed_dict(decoded_response):
            missing_get: Final = f"{type(decoded_response).__name__!r} object has no attribute 'get'"
            raise AttributeError(missing_get, name="get", obj=decoded_response)
        response_data: Final = self._handle_content_list_to_str_conversion(
            self._handle_empty_content_response(decoded_response)
        )

        final_response_obj: Final = cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=response_data,
                model_response_object=model_response,
                hidden_params={"headers": raw_response.headers},
                _response_headers=dict(raw_response.headers),
            ),
        )

        return final_response_obj

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ):
        return MistralChatResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class MistralChatResponseIterator(OpenAIChatCompletionStreamingHandler):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                content = delta.get("content")
                if isinstance(content, list):
                    (
                        normalized_text,
                        thinking_blocks,
                        reasoning_content,
                    ) = self._normalize_content_blocks(content)
                    delta["content"] = normalized_text
                    if thinking_blocks:
                        delta["thinking_blocks"] = thinking_blocks
                        delta["reasoning_content"] = reasoning_content
                    else:
                        delta.pop("thinking_blocks", None)
                        delta.pop("reasoning_content", None)
        except Exception:
            # Fall back to default parsing if custom handling fails
            return super().chunk_parser(chunk)

        return super().chunk_parser(chunk)

    @staticmethod
    def _normalize_content_blocks(
        content_blocks: list[dict],
    ) -> tuple[str | None, list[dict], str | None]:
        """
        Convert Mistral magistral content blocks into OpenAI-compatible content + thinking_blocks.
        """
        text_segments: Final[list[str]] = []
        thinking_blocks: Final[list[dict]] = []
        reasoning_segments: Final[list[str]] = []

        for block in content_blocks:
            block_type = block.get("type")
            if block_type == "thinking":
                mistral_thinking = block.get("thinking", [])
                thinking_text_parts: list[str] = []
                for thinking_block in mistral_thinking:
                    if thinking_block.get("type") == "text":
                        thinking_text_parts.append(thinking_block.get("text", ""))
                thinking_text = "".join(thinking_text_parts)
                if thinking_text:
                    reasoning_segments.append(thinking_text)
                    thinking_blocks.append(
                        {
                            "type": "thinking",
                            "thinking": thinking_text,
                            "signature": "mistral",
                        }
                    )
            elif block_type == "text":
                text_segments.append(block.get("text", ""))

        normalized_text: Final = "".join(text_segments) if text_segments else None
        reasoning_content: Final = "\n".join(reasoning_segments) if reasoning_segments else None
        return normalized_text, thinking_blocks, reasoning_content
