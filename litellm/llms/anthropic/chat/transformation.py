import types
from typing import List, Literal, Optional, Tuple, Union

import litellm
from litellm.llms.prompt_templates.factory import anthropic_messages_pt
from litellm.types.llms.anthropic import (
    AnthropicMessageRequestBase,
    AnthropicMessagesRequest,
    AnthropicMessagesTool,
    AnthropicMessagesToolChoice,
    AnthropicSystemMessageContent,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionSystemMessage,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.utils import add_dummy_tool, has_tool_call_blocks

from ..common_utils import AnthropicError


class AnthropicConfig:
    """
    Reference: https://docs.anthropic.com/claude/reference/messages_post

    to pass metadata to anthropic, it's {"user_id": "any-relevant-information"}
    """

    max_tokens: Optional[int] = (
        4096  # anthropic requires a default value (Opus, Sonnet, and Haiku have the same default)
    )
    stop_sequences: Optional[list] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    metadata: Optional[dict] = None
    system: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[
            int
        ] = 4096,  # You can pass in a value yourself or use the default value 4096
        stop_sequences: Optional[list] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        metadata: Optional[dict] = None,
        system: Optional[str] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self):
        return [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "extra_headers",
        ]

    def get_cache_control_headers(self) -> dict:
        return {
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        messages: Optional[List[AllMessageValues]] = None,
    ):
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "tool_choice":
                _tool_choice: Optional[AnthropicMessagesToolChoice] = None
                if value == "auto":
                    _tool_choice = {"type": "auto"}
                elif value == "required":
                    _tool_choice = {"type": "any"}
                elif isinstance(value, dict):
                    _tool_choice = {"type": "tool", "name": value["function"]["name"]}

                if _tool_choice is not None:
                    optional_params["tool_choice"] = _tool_choice
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "stop":
                if isinstance(value, str):
                    if (
                        value == "\n"
                    ) and litellm.drop_params is True:  # anthropic doesn't allow whitespace characters as stop-sequences
                        continue
                    value = [value]
                elif isinstance(value, list):
                    new_v = []
                    for v in value:
                        if (
                            v == "\n"
                        ) and litellm.drop_params is True:  # anthropic doesn't allow whitespace characters as stop-sequences
                            continue
                        new_v.append(v)
                    if len(new_v) > 0:
                        value = new_v
                    else:
                        continue
                optional_params["stop_sequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value

        ## VALIDATE REQUEST
        """
        Anthropic doesn't support tool calling without `tools=` param specified.
        """
        if (
            "tools" not in non_default_params
            and messages is not None
            and has_tool_call_blocks(messages)
        ):
            if litellm.modify_params:
                optional_params["tools"] = add_dummy_tool(
                    custom_llm_provider="bedrock_converse"
                )
            else:
                raise litellm.UnsupportedParamsError(
                    message="Anthropic doesn't support tool calling without `tools=` param specified. Pass `tools=` param OR set `litellm.modify_params = True` // `litellm_settings::modify_params: True` to add dummy tool to the request.",
                    model="",
                    llm_provider="anthropic",
                )

        return optional_params

    def is_cache_control_set(self, messages: List[AllMessageValues]) -> bool:
        """
        Return if {"cache_control": ..} in message content block

        Used to check if anthropic prompt caching headers need to be set.
        """
        for message in messages:
            _message_content = message.get("content")
            if _message_content is not None and isinstance(_message_content, list):
                for content in _message_content:
                    if "cache_control" in content:
                        return True

        return False

    def translate_system_message(
        self, messages: List[AllMessageValues]
    ) -> List[AnthropicSystemMessageContent]:
        """
        Translate system message to anthropic format.

        Removes system message from the original list and returns a new list of anthropic system message content.
        """
        system_prompt_indices = []
        anthropic_system_message_list: List[AnthropicSystemMessageContent] = []
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                valid_content: bool = False
                system_message_block = ChatCompletionSystemMessage(**message)
                if isinstance(system_message_block["content"], str):
                    anthropic_system_message_content = AnthropicSystemMessageContent(
                        type="text",
                        text=system_message_block["content"],
                    )
                    if "cache_control" in system_message_block:
                        anthropic_system_message_content["cache_control"] = (
                            system_message_block["cache_control"]
                        )
                    anthropic_system_message_list.append(
                        anthropic_system_message_content
                    )
                    valid_content = True
                elif isinstance(message["content"], list):
                    for _content in message["content"]:
                        anthropic_system_message_content = (
                            AnthropicSystemMessageContent(
                                type=_content.get("type"),
                                text=_content.get("text"),
                            )
                        )
                        if "cache_control" in _content:
                            anthropic_system_message_content["cache_control"] = (
                                _content["cache_control"]
                            )

                        anthropic_system_message_list.append(
                            anthropic_system_message_content
                        )
                    valid_content = True

                if valid_content:
                    system_prompt_indices.append(idx)
        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)

        return anthropic_system_message_list

    def _transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        headers: dict,
        _is_function_call: bool,
        is_vertex_request: bool,
    ) -> dict:
        """
        Translate messages to anthropic format.
        """
        # Separate system prompt from rest of message
        anthropic_system_message_list = self.translate_system_message(messages=messages)
        # Handling anthropic API Prompt Caching
        if len(anthropic_system_message_list) > 0:
            optional_params["system"] = anthropic_system_message_list
        # Format rest of message according to anthropic guidelines
        try:
            anthropic_messages = anthropic_messages_pt(
                model=model,
                messages=messages,
                llm_provider="anthropic",
            )
        except Exception as e:
            raise AnthropicError(
                status_code=400,
                message="{}\nReceived Messages={}".format(str(e), messages),
            )  # don't use verbose_logger.exception, if exception is raised

        ## Load Config
        config = litellm.AnthropicConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        ## Handle Tool Calling
        if "tools" in optional_params:
            _is_function_call = True
            anthropic_tools = []
            for tool in optional_params["tools"]:
                if "input_schema" in tool:  # assume in anthropic format
                    anthropic_tools.append(tool)
                else:  # assume openai tool call
                    new_tool = tool["function"]
                    parameters = new_tool.pop(
                        "parameters",
                        {
                            "type": "object",
                            "properties": {},
                        },
                    )
                    new_tool["input_schema"] = parameters  # rename key
                    if "cache_control" in tool:
                        new_tool["cache_control"] = tool["cache_control"]
                    anthropic_tools.append(new_tool)
            optional_params["tools"] = anthropic_tools

        data = {
            "messages": anthropic_messages,
            **optional_params,
        }
        if not is_vertex_request:
            data["model"] = model
        return data
