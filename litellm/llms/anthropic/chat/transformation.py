import types
from typing import List, Literal, Optional, Tuple, Union

import litellm
from litellm.llms.prompt_templates.factory import anthropic_messages_pt
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthropicComputerTool,
    AnthropicHostedTools,
    AnthropicMessageRequestBase,
    AnthropicMessagesRequest,
    AnthropicMessagesTool,
    AnthropicMessagesToolChoice,
    AnthropicSystemMessageContent,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionCachedContent,
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
            "parallel_tool_calls",
            "response_format",
        ]

    def get_cache_control_headers(self) -> dict:
        return {
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }

    def get_anthropic_headers(
        self,
        api_key: str,
        anthropic_version: Optional[str] = None,
        computer_tool_used: bool = False,
        prompt_caching_set: bool = False,
    ) -> dict:
        import json

        betas = []
        if prompt_caching_set:
            betas.append("prompt-caching-2024-07-31")
        if computer_tool_used:
            betas.append("computer-use-2024-10-22")
        headers = {
            "anthropic-version": anthropic_version or "2023-06-01",
            "x-api-key": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }
        if len(betas) > 0:
            headers["anthropic-beta"] = ",".join(betas)
        return headers

    def _map_tool_choice(
        self, tool_choice: Optional[str], disable_parallel_tool_use: Optional[bool]
    ) -> Optional[AnthropicMessagesToolChoice]:
        _tool_choice: Optional[AnthropicMessagesToolChoice] = None
        if tool_choice == "auto":
            _tool_choice = AnthropicMessagesToolChoice(
                type="auto",
            )
        elif tool_choice == "required":
            _tool_choice = AnthropicMessagesToolChoice(type="any")
        elif isinstance(tool_choice, dict):
            _tool_name = tool_choice.get("function", {}).get("name")
            _tool_choice = AnthropicMessagesToolChoice(type="tool")
            if _tool_name is not None:
                _tool_choice["name"] = _tool_name

        if disable_parallel_tool_use is not None:
            if _tool_choice is not None:
                _tool_choice["disable_parallel_tool_use"] = disable_parallel_tool_use
            else:  # use anthropic defaults and make sure to send the disable_parallel_tool_use flag
                _tool_choice = AnthropicMessagesToolChoice(
                    type="auto",
                    disable_parallel_tool_use=disable_parallel_tool_use,
                )
        return _tool_choice

    def _map_tool_helper(
        self, tool: ChatCompletionToolParam
    ) -> AllAnthropicToolsValues:
        returned_tool: Optional[AllAnthropicToolsValues] = None

        if tool["type"] == "function" or tool["type"] == "custom":
            _tool = AnthropicMessagesTool(
                name=tool["function"]["name"],
                input_schema=tool["function"].get(
                    "parameters",
                    {
                        "type": "object",
                        "properties": {},
                    },
                ),
            )

            _description = tool["function"].get("description")
            if _description is not None:
                _tool["description"] = _description

            returned_tool = _tool

        elif tool["type"].startswith("computer_"):
            ## check if all required 'display_' params are given
            if "parameters" not in tool["function"]:
                raise ValueError("Missing required parameter: parameters")

            _display_width_px: Optional[int] = tool["function"]["parameters"].get(
                "display_width_px"
            )
            _display_height_px: Optional[int] = tool["function"]["parameters"].get(
                "display_height_px"
            )
            if _display_width_px is None or _display_height_px is None:
                raise ValueError(
                    "Missing required parameter: display_width_px or display_height_px"
                )

            _computer_tool = AnthropicComputerTool(
                type=tool["type"],
                name=tool["function"].get("name", "computer"),
                display_width_px=_display_width_px,
                display_height_px=_display_height_px,
            )

            _display_number = tool["function"]["parameters"].get("display_number")
            if _display_number is not None:
                _computer_tool["display_number"] = _display_number

            returned_tool = _computer_tool
        elif tool["type"].startswith("bash_") or tool["type"].startswith(
            "text_editor_"
        ):
            function_name = tool["function"].get("name")
            if function_name is None:
                raise ValueError("Missing required parameter: name")

            returned_tool = AnthropicHostedTools(
                type=tool["type"],
                name=function_name,
            )
        if returned_tool is None:
            raise ValueError(f"Unsupported tool type: {tool['type']}")

        ## check if cache_control is set in the tool
        _cache_control = tool.get("cache_control", None)
        _cache_control_function = tool.get("function", {}).get("cache_control", None)
        if _cache_control is not None:
            returned_tool["cache_control"] = _cache_control
        elif _cache_control_function is not None and isinstance(
            _cache_control_function, dict
        ):
            returned_tool["cache_control"] = ChatCompletionCachedContent(
                **_cache_control_function  # type: ignore
            )

        return returned_tool

    def _map_tools(self, tools: List) -> List[AllAnthropicToolsValues]:
        anthropic_tools = []
        for tool in tools:
            if "input_schema" in tool:  # assume in anthropic format
                anthropic_tools.append(tool)
            else:  # assume openai tool call
                new_tool = self._map_tool_helper(tool)

                anthropic_tools.append(new_tool)
        return anthropic_tools

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
                optional_params["tools"] = self._map_tools(value)
            if param == "tool_choice" or param == "parallel_tool_calls":
                _tool_choice: Optional[AnthropicMessagesToolChoice] = (
                    self._map_tool_choice(
                        tool_choice=non_default_params.get("tool_choice"),
                        disable_parallel_tool_use=non_default_params.get(
                            "parallel_tool_calls"
                        ),
                    )
                )

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
            if param == "response_format" and isinstance(value, dict):
                json_schema: Optional[dict] = None
                if "response_schema" in value:
                    json_schema = value["response_schema"]
                elif "json_schema" in value:
                    json_schema = value["json_schema"]["schema"]
                """
                When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
                - You usually want to provide a single tool
                - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
                - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.
                """
                _tool_choice = None
                _tool_choice = {"name": "json_tool_call", "type": "tool"}

                _tool = AnthropicMessagesTool(
                    name="json_tool_call",
                    input_schema={
                        "type": "object",
                        "properties": {"values": json_schema},  # type: ignore
                    },
                )

                optional_params["tools"] = [_tool]
                optional_params["tool_choice"] = _tool_choice
                optional_params["json_mode"] = True

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
                optional_params["tools"] = self._map_tools(
                    add_dummy_tool(custom_llm_provider="anthropic")
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
            if message.get("cache_control", None) is not None:
                return True
            _message_content = message.get("content")
            if _message_content is not None and isinstance(_message_content, list):
                for content in _message_content:
                    if "cache_control" in content:
                        return True

        return False

    def is_computer_tool_used(
        self, tools: Optional[List[AllAnthropicToolsValues]]
    ) -> bool:
        if tools is None:
            return False
        for tool in tools:
            if "type" in tool and tool["type"].startswith("computer_"):
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

        data = {
            "messages": anthropic_messages,
            **optional_params,
        }
        if not is_vertex_request:
            data["model"] = model
        return data
