import json
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import httpx

import litellm
from litellm.constants import (
    ANTHROPIC_WEB_SEARCH_TOOL_MAX_USES,
    DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    RESPONSE_FORMAT_TOOL_NAME,
)
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.llms.base_llm.base_utils import type_to_response_format_param
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.anthropic import (
    AllAnthropicMessageValues,
    AllAnthropicToolsValues,
    AnthropicCodeExecutionTool,
    AnthropicComputerTool,
    AnthropicHostedTools,
    AnthropicInputSchema,
    AnthropicMcpServerTool,
    AnthropicMessagesTool,
    AnthropicMessagesToolChoice,
    AnthropicSystemMessageContent,
    AnthropicThinkingParam,
    AnthropicWebSearchTool,
    AnthropicWebSearchUserLocation,
)
from litellm.types.llms.openai import (
    REASONING_EFFORT,
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionSystemMessage,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParam,
    OpenAIMcpServerTool,
    OpenAIWebSearchOptions,
)
from litellm.types.utils import CompletionTokensDetailsWrapper
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import PromptTokensDetailsWrapper, ServerToolUse
from litellm.utils import (
    ModelResponse,
    Usage,
    add_dummy_tool,
    has_tool_call_blocks,
    supports_reasoning,
    token_counter,
)

from ..common_utils import AnthropicError, AnthropicModelInfo, process_anthropic_headers

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


ANTHROPIC_HOSTED_TOOLS = ["web_search", "bash", "text_editor", "code_execution"]


class AnthropicConfig(AnthropicModelInfo, BaseConfig):
    """
    Reference: https://docs.anthropic.com/claude/reference/messages_post

    to pass metadata to anthropic, it's {"user_id": "any-relevant-information"}
    """

    max_tokens: Optional[int] = (
        DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS  # anthropic requires a default value (Opus, Sonnet, and Haiku have the same default)
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
        ] = DEFAULT_ANTHROPIC_CHAT_MAX_TOKENS,  # You can pass in a value yourself or use the default value 4096
        stop_sequences: Optional[list] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        metadata: Optional[dict] = None,
        system: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "anthropic"

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):

        params = [
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
            "user",
            "reasoning_effort",
            "web_search_options",
        ]

        if "claude-3-7-sonnet" in model or supports_reasoning(
            model=model,
            custom_llm_provider=self.custom_llm_provider,
        ):
            params.append("thinking")

        return params

    def get_json_schema_from_pydantic_object(
        self, response_format: Union[Any, Dict, None]
    ) -> Optional[dict]:
        return type_to_response_format_param(
            response_format, ref_template="/$defs/{model}"
        )  # Relevant issue: https://github.com/BerriAI/litellm/issues/7755

    def get_cache_control_headers(self) -> dict:
        return {
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "prompt-caching-2024-07-31",
        }

    def _map_tool_choice(
        self, tool_choice: Optional[str], parallel_tool_use: Optional[bool]
    ) -> Optional[AnthropicMessagesToolChoice]:
        _tool_choice: Optional[AnthropicMessagesToolChoice] = None
        if tool_choice == "auto":
            _tool_choice = AnthropicMessagesToolChoice(
                type="auto",
            )
        elif tool_choice == "required":
            _tool_choice = AnthropicMessagesToolChoice(type="any")
        elif tool_choice == "none":
            _tool_choice = AnthropicMessagesToolChoice(type="none")
        elif isinstance(tool_choice, dict):
            _tool_name = tool_choice.get("function", {}).get("name")
            _tool_choice = AnthropicMessagesToolChoice(type="tool")
            if _tool_name is not None:
                _tool_choice["name"] = _tool_name

        if parallel_tool_use is not None:
            # Anthropic uses 'disable_parallel_tool_use' flag to determine if parallel tool use is allowed
            # this is the inverse of the openai flag.
            if tool_choice == "none":
                pass
            elif _tool_choice is not None:
                _tool_choice["disable_parallel_tool_use"] = not parallel_tool_use
            else:  # use anthropic defaults and make sure to send the disable_parallel_tool_use flag
                _tool_choice = AnthropicMessagesToolChoice(
                    type="auto",
                    disable_parallel_tool_use=not parallel_tool_use,
                )
        return _tool_choice

    def _map_tool_helper(
        self, tool: ChatCompletionToolParam
    ) -> Tuple[Optional[AllAnthropicToolsValues], Optional[AnthropicMcpServerTool]]:
        returned_tool: Optional[AllAnthropicToolsValues] = None
        mcp_server: Optional[AnthropicMcpServerTool] = None

        if tool["type"] == "function" or tool["type"] == "custom":
            _input_schema: dict = tool["function"].get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                },
            )
            input_schema: AnthropicInputSchema = AnthropicInputSchema(**_input_schema)
            _tool = AnthropicMessagesTool(
                name=tool["function"]["name"],
                input_schema=input_schema,
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
        elif any(tool["type"].startswith(t) for t in ANTHROPIC_HOSTED_TOOLS):
            function_name = tool.get("name", tool.get("function", {}).get("name"))
            if function_name is None or not isinstance(function_name, str):
                raise ValueError("Missing required parameter: name")

            additional_tool_params = {}
            for k, v in tool.items():
                if k != "type" and k != "name":
                    additional_tool_params[k] = v

            returned_tool = AnthropicHostedTools(
                type=tool["type"], name=function_name, **additional_tool_params  # type: ignore
            )
        elif tool["type"] == "url":  # mcp server tool
            mcp_server = AnthropicMcpServerTool(**tool)  # type: ignore
        elif tool["type"] == "mcp":
            mcp_server = self._map_openai_mcp_server_tool(
                cast(OpenAIMcpServerTool, tool)
            )
        if returned_tool is None and mcp_server is None:
            raise ValueError(f"Unsupported tool type: {tool['type']}")

        ## check if cache_control is set in the tool
        _cache_control = tool.get("cache_control", None)
        _cache_control_function = tool.get("function", {}).get("cache_control", None)
        if returned_tool is not None:
            if _cache_control is not None:
                returned_tool["cache_control"] = _cache_control
            elif _cache_control_function is not None and isinstance(
                _cache_control_function, dict
            ):
                returned_tool["cache_control"] = ChatCompletionCachedContent(
                    **_cache_control_function  # type: ignore
                )

        return returned_tool, mcp_server

    def _map_openai_mcp_server_tool(
        self, tool: OpenAIMcpServerTool
    ) -> AnthropicMcpServerTool:
        from litellm.types.llms.anthropic import AnthropicMcpServerToolConfiguration

        allowed_tools = tool.get("allowed_tools", None)
        tool_configuration: Optional[AnthropicMcpServerToolConfiguration] = None
        if allowed_tools is not None:
            tool_configuration = AnthropicMcpServerToolConfiguration(
                allowed_tools=tool.get("allowed_tools", None),
            )

        headers = tool.get("headers", {})
        authorization_token: Optional[str] = None
        if headers is not None:
            bearer_token = headers.get("Authorization", None)
            if bearer_token is not None:
                authorization_token = bearer_token.replace("Bearer ", "")

        initial_tool = AnthropicMcpServerTool(
            type="url",
            url=tool["server_url"],
            name=tool["server_label"],
        )

        if tool_configuration is not None:
            initial_tool["tool_configuration"] = tool_configuration
        if authorization_token is not None:
            initial_tool["authorization_token"] = authorization_token
        return initial_tool

    def _map_tools(
        self, tools: List
    ) -> Tuple[List[AllAnthropicToolsValues], List[AnthropicMcpServerTool]]:
        anthropic_tools = []
        mcp_servers = []
        for tool in tools:
            if "input_schema" in tool:  # assume in anthropic format
                anthropic_tools.append(tool)
            else:  # assume openai tool call
                new_tool, mcp_server_tool = self._map_tool_helper(tool)

                if new_tool is not None:
                    anthropic_tools.append(new_tool)
                if mcp_server_tool is not None:
                    mcp_servers.append(mcp_server_tool)
        return anthropic_tools, mcp_servers

    def _map_stop_sequences(
        self, stop: Optional[Union[str, List[str]]]
    ) -> Optional[List[str]]:
        new_stop: Optional[List[str]] = None
        if isinstance(stop, str):
            if (
                stop.isspace() and litellm.drop_params is True
            ):  # anthropic doesn't allow whitespace characters as stop-sequences
                return new_stop
            new_stop = [stop]
        elif isinstance(stop, list):
            new_v = []
            for v in stop:
                if (
                    v.isspace() and litellm.drop_params is True
                ):  # anthropic doesn't allow whitespace characters as stop-sequences
                    continue
                new_v.append(v)
            if len(new_v) > 0:
                new_stop = new_v
        return new_stop

    @staticmethod
    def _map_reasoning_effort(
        reasoning_effort: Optional[Union[REASONING_EFFORT, str]],
    ) -> Optional[AnthropicThinkingParam]:
        if reasoning_effort is None:
            return None
        elif reasoning_effort == "low":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
            )
        elif reasoning_effort == "medium":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
            )
        elif reasoning_effort == "high":
            return AnthropicThinkingParam(
                type="enabled",
                budget_tokens=DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
            )
        else:
            raise ValueError(f"Unmapped reasoning effort: {reasoning_effort}")

    def map_response_format_to_anthropic_tool(
        self, value: Optional[dict], optional_params: dict, is_thinking_enabled: bool
    ) -> Optional[AnthropicMessagesTool]:
        ignore_response_format_types = ["text"]
        if (
            value is None or value["type"] in ignore_response_format_types
        ):  # value is a no-op
            return None

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

        _tool = self._create_json_tool_call_for_response_format(
            json_schema=json_schema,
        )

        return _tool

    def map_web_search_tool(
        self,
        value: OpenAIWebSearchOptions,
    ) -> AnthropicWebSearchTool:
        value_typed = cast(OpenAIWebSearchOptions, value)
        hosted_web_search_tool = AnthropicWebSearchTool(
            type="web_search_20250305",
            name="web_search",
        )
        user_location = value_typed.get("user_location")
        if user_location is not None:
            anthropic_user_location = AnthropicWebSearchUserLocation(type="approximate")
            anthropic_user_location_keys = (
                AnthropicWebSearchUserLocation.__annotations__.keys()
            )
            user_location_approximate = user_location.get("approximate")
            if user_location_approximate is not None:
                for key, user_location_value in user_location_approximate.items():
                    if key in anthropic_user_location_keys and key != "type":
                        anthropic_user_location[key] = user_location_value  # type: ignore
                hosted_web_search_tool["user_location"] = anthropic_user_location

        ## MAP SEARCH CONTEXT SIZE
        search_context_size = value_typed.get("search_context_size")
        if search_context_size is not None:
            hosted_web_search_tool["max_uses"] = ANTHROPIC_WEB_SEARCH_TOOL_MAX_USES[
                search_context_size
            ]

        return hosted_web_search_tool

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        is_thinking_enabled = self.is_thinking_enabled(
            non_default_params=non_default_params
        )

        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                # check if optional params already has tools
                anthropic_tools, mcp_servers = self._map_tools(value)
                optional_params = self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=anthropic_tools
                )
                if mcp_servers:
                    optional_params["mcp_servers"] = mcp_servers
            if param == "tool_choice" or param == "parallel_tool_calls":
                _tool_choice: Optional[AnthropicMessagesToolChoice] = (
                    self._map_tool_choice(
                        tool_choice=non_default_params.get("tool_choice"),
                        parallel_tool_use=non_default_params.get("parallel_tool_calls"),
                    )
                )

                if _tool_choice is not None:
                    optional_params["tool_choice"] = _tool_choice
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "stop" and (isinstance(value, str) or isinstance(value, list)):
                _value = self._map_stop_sequences(value)
                if _value is not None:
                    optional_params["stop_sequences"] = _value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "response_format" and isinstance(value, dict):
                _tool = self.map_response_format_to_anthropic_tool(
                    value, optional_params, is_thinking_enabled
                )
                if _tool is None:
                    continue
                if not is_thinking_enabled:
                    _tool_choice = {"name": RESPONSE_FORMAT_TOOL_NAME, "type": "tool"}
                    optional_params["tool_choice"] = _tool_choice
                optional_params["json_mode"] = True
                optional_params = self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=[_tool]
                )
            if (
                param == "user"
                and value is not None
                and isinstance(value, str)
                and _valid_user_id(value)  # anthropic fails on emails
            ):
                optional_params["metadata"] = {"user_id": value}
            if param == "thinking":
                optional_params["thinking"] = value
            elif param == "reasoning_effort" and isinstance(value, str):
                optional_params["thinking"] = AnthropicConfig._map_reasoning_effort(
                    value
                )
            elif param == "web_search_options" and isinstance(value, dict):
                hosted_web_search_tool = self.map_web_search_tool(
                    cast(OpenAIWebSearchOptions, value)
                )
                self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=[hosted_web_search_tool]
                )

        ## handle thinking tokens
        self.update_optional_params_with_thinking_tokens(
            non_default_params=non_default_params, optional_params=optional_params
        )

        return optional_params

    def _create_json_tool_call_for_response_format(
        self,
        json_schema: Optional[dict] = None,
    ) -> AnthropicMessagesTool:
        """
        Handles creating a tool call for getting responses in JSON format.

        Args:
            json_schema (Optional[dict]): The JSON schema the response should be in

        Returns:
            AnthropicMessagesTool: The tool call to send to Anthropic API to get responses in JSON format
        """
        _input_schema: AnthropicInputSchema = AnthropicInputSchema(
            type="object",
        )

        if json_schema is None:
            # Anthropic raises a 400 BadRequest error if properties is passed as None
            # see usage with additionalProperties (Example 5) https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb
            _input_schema["additionalProperties"] = True
            _input_schema["properties"] = {}
        else:
            _input_schema.update(cast(AnthropicInputSchema, json_schema))

        _tool = AnthropicMessagesTool(
            name=RESPONSE_FORMAT_TOOL_NAME, input_schema=_input_schema
        )
        return _tool

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

    def add_code_execution_tool(
        self,
        messages: List[AllAnthropicMessageValues],
        tools: List[Union[AllAnthropicToolsValues, Dict]],
    ) -> List[Union[AllAnthropicToolsValues, Dict]]:
        """if 'container_upload' in messages, add code_execution tool"""
        add_code_execution_tool = False
        for message in messages:
            message_content = message.get("content", None)
            if message_content and isinstance(message_content, list):
                for content in message_content:
                    content_type = content.get("type", None)
                    if content_type == "container_upload":
                        add_code_execution_tool = True
                        break

        if add_code_execution_tool:
            ## check if code_execution tool is already in tools
            for tool in tools:
                tool_type = tool.get("type", None)
                if (
                    tool_type
                    and isinstance(tool_type, str)
                    and tool_type.startswith("code_execution")
                ):
                    return tools
            tools.append(
                AnthropicCodeExecutionTool(
                    name="code_execution",
                    type="code_execution_20250522",
                )
            )
        return tools

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Translate messages to anthropic format.
        """
        ## VALIDATE REQUEST
        """
        Anthropic doesn't support tool calling without `tools=` param specified.
        """
        from litellm.litellm_core_utils.prompt_templates.factory import (
            anthropic_messages_pt,
        )

        if (
            "tools" not in optional_params
            and messages is not None
            and has_tool_call_blocks(messages)
        ):
            if litellm.modify_params:
                optional_params["tools"], _ = self._map_tools(
                    add_dummy_tool(custom_llm_provider="anthropic")
                )
            else:
                raise litellm.UnsupportedParamsError(
                    message="Anthropic doesn't support tool calling without `tools=` param specified. Pass `tools=` param OR set `litellm.modify_params = True` // `litellm_settings::modify_params: True` to add dummy tool to the request.",
                    model="",
                    llm_provider="anthropic",
                )

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

        ## Add code_execution tool if container_upload is in messages
        _tools = (
            cast(
                Optional[List[Union[AllAnthropicToolsValues, Dict]]],
                optional_params.get("tools"),
            )
            or []
        )
        tools = self.add_code_execution_tool(messages=anthropic_messages, tools=_tools)
        if len(tools) > 1:
            optional_params["tools"] = tools

        ## Load Config
        config = litellm.AnthropicConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        ## Handle user_id in metadata
        _litellm_metadata = litellm_params.get("metadata", None)
        if (
            _litellm_metadata
            and isinstance(_litellm_metadata, dict)
            and "user_id" in _litellm_metadata
            and not _valid_user_id(_litellm_metadata.get("user_id", None))
        ):
            optional_params["metadata"] = {"user_id": _litellm_metadata["user_id"]}

        data = {
            "model": model,
            "messages": anthropic_messages,
            **optional_params,
        }

        return data

    def _transform_response_for_json_mode(
        self,
        json_mode: Optional[bool],
        tool_calls: List[ChatCompletionToolCallChunk],
    ) -> Optional[LitellmMessage]:
        _message: Optional[LitellmMessage] = None
        if json_mode is True and len(tool_calls) == 1:
            # check if tool name is the default tool name
            json_mode_content_str: Optional[str] = None
            if (
                "name" in tool_calls[0]["function"]
                and tool_calls[0]["function"]["name"] == RESPONSE_FORMAT_TOOL_NAME
            ):
                json_mode_content_str = tool_calls[0]["function"].get("arguments")
            if json_mode_content_str is not None:
                _message = AnthropicConfig._convert_tool_response_to_message(
                    tool_calls=tool_calls,
                )
        return _message

    def extract_response_content(self, completion_response: dict) -> Tuple[
        str,
        Optional[List[Any]],
        Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ],
        Optional[str],
        List[ChatCompletionToolCallChunk],
    ]:
        text_content = ""
        citations: Optional[List[Any]] = None
        thinking_blocks: Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ] = None
        reasoning_content: Optional[str] = None
        tool_calls: List[ChatCompletionToolCallChunk] = []
        for idx, content in enumerate(completion_response["content"]):
            if content["type"] == "text":
                text_content += content["text"]
            ## TOOL CALLING
            elif content["type"] == "tool_use":
                tool_calls.append(
                    ChatCompletionToolCallChunk(
                        id=content["id"],
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=content["name"],
                            arguments=json.dumps(content["input"]),
                        ),
                        index=idx,
                    )
                )

            elif content.get("thinking", None) is not None:
                if thinking_blocks is None:
                    thinking_blocks = []
                thinking_blocks.append(cast(ChatCompletionThinkingBlock, content))
            elif content["type"] == "redacted_thinking":
                if thinking_blocks is None:
                    thinking_blocks = []
                thinking_blocks.append(
                    cast(ChatCompletionRedactedThinkingBlock, content)
                )

            ## CITATIONS
            if content.get("citations") is not None:
                if citations is None:
                    citations = []
                citations.append(content["citations"])
        if thinking_blocks is not None:
            reasoning_content = ""
            for block in thinking_blocks:
                thinking_content = cast(Optional[str], block.get("thinking"))
                if thinking_content is not None:
                    reasoning_content += thinking_content

        return text_content, citations, thinking_blocks, reasoning_content, tool_calls

    def calculate_usage(
        self, usage_object: dict, reasoning_content: Optional[str]
    ) -> Usage:
        prompt_tokens = usage_object.get("input_tokens", 0)
        completion_tokens = usage_object.get("output_tokens", 0)
        _usage = usage_object
        cache_creation_input_tokens: int = 0
        cache_read_input_tokens: int = 0
        web_search_requests: Optional[int] = None
        if "cache_creation_input_tokens" in _usage:
            cache_creation_input_tokens = _usage["cache_creation_input_tokens"]
        if "cache_read_input_tokens" in _usage:
            cache_read_input_tokens = _usage["cache_read_input_tokens"]
            prompt_tokens += cache_read_input_tokens
        if "server_tool_use" in _usage:
            if "web_search_requests" in _usage["server_tool_use"]:
                web_search_requests = cast(
                    int, _usage["server_tool_use"]["web_search_requests"]
                )

        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=cache_read_input_tokens,
        )
        completion_token_details = (
            CompletionTokensDetailsWrapper(
                reasoning_tokens=token_counter(
                    text=reasoning_content, count_response_tokens=True
                )
            )
            if reasoning_content
            else None
        )
        total_tokens = prompt_tokens + completion_tokens

        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            completion_tokens_details=completion_token_details,
            server_tool_use=(
                ServerToolUse(web_search_requests=web_search_requests)
                if web_search_requests is not None
                else None
            ),
        )
        return usage

    def transform_parsed_response(
        self,
        completion_response: dict,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        json_mode: Optional[bool] = None,
        prefix_prompt: Optional[str] = None,
    ):
        _hidden_params: Dict = {}
        _hidden_params["additional_headers"] = process_anthropic_headers(
            dict(raw_response.headers)
        )
        if "error" in completion_response:
            response_headers = getattr(raw_response, "headers", None)
            raise AnthropicError(
                message=str(completion_response["error"]),
                status_code=raw_response.status_code,
                headers=response_headers,
            )
        else:
            text_content = ""
            citations: Optional[List[Any]] = None
            thinking_blocks: Optional[
                List[
                    Union[
                        ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock
                    ]
                ]
            ] = None
            reasoning_content: Optional[str] = None
            tool_calls: List[ChatCompletionToolCallChunk] = []

            (
                text_content,
                citations,
                thinking_blocks,
                reasoning_content,
                tool_calls,
            ) = self.extract_response_content(completion_response=completion_response)

            if (
                prefix_prompt is not None
                and not text_content.startswith(prefix_prompt)
                and not litellm.disable_add_prefix_to_prompt
            ):
                text_content = prefix_prompt + text_content

            _message = litellm.Message(
                tool_calls=tool_calls,
                content=text_content or None,
                provider_specific_fields={
                    "citations": citations,
                    "thinking_blocks": thinking_blocks,
                },
                thinking_blocks=thinking_blocks,
                reasoning_content=reasoning_content,
            )

            ## HANDLE JSON MODE - anthropic returns single function call
            json_mode_message = self._transform_response_for_json_mode(
                json_mode=json_mode,
                tool_calls=tool_calls,
            )
            if json_mode_message is not None:
                completion_response["stop_reason"] = "stop"
                _message = json_mode_message

            model_response.choices[0].message = _message  # type: ignore
            model_response._hidden_params["original_response"] = completion_response[
                "content"
            ]  # allow user to access raw anthropic tool calling response

            model_response.choices[0].finish_reason = map_finish_reason(
                completion_response["stop_reason"]
            )

        ## CALCULATING USAGE
        usage = self.calculate_usage(
            usage_object=completion_response["usage"],
            reasoning_content=reasoning_content,
        )
        setattr(model_response, "usage", usage)  # type: ignore

        model_response.created = int(time.time())
        model_response.model = completion_response["model"]

        model_response._hidden_params = _hidden_params

        return model_response

    def get_prefix_prompt(self, messages: List[AllMessageValues]) -> Optional[str]:
        """
        Get the prefix prompt from the messages.

        Check last message
        - if it's assistant message, with 'prefix': true, return the content

        E.g. :    {"role": "assistant", "content": "Argentina", "prefix": True}
        """
        if len(messages) == 0:
            return None

        message = messages[-1]
        message_content = message.get("content")
        if (
            message["role"] == "assistant"
            and message.get("prefix", False)
            and isinstance(message_content, str)
        ):
            return message_content

        return None

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise AnthropicError(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), raw_response.text
                ),
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        prefix_prompt = self.get_prefix_prompt(messages=messages)

        model_response = self.transform_parsed_response(
            completion_response=completion_response,
            raw_response=raw_response,
            model_response=model_response,
            json_mode=json_mode,
            prefix_prompt=prefix_prompt,
        )
        return model_response

    @staticmethod
    def _convert_tool_response_to_message(
        tool_calls: List[ChatCompletionToolCallChunk],
    ) -> Optional[LitellmMessage]:
        """
        In JSON mode, Anthropic API returns JSON schema as a tool call, we need to convert it to a message to follow the OpenAI format

        """
        ## HANDLE JSON MODE - anthropic returns single function call
        json_mode_content_str: Optional[str] = tool_calls[0]["function"].get(
            "arguments"
        )
        try:
            if json_mode_content_str is not None:
                args = json.loads(json_mode_content_str)
                if (
                    isinstance(args, dict)
                    and (values := args.get("values")) is not None
                ):
                    _message = litellm.Message(content=json.dumps(values))
                    return _message
                else:
                    # a lot of the times the `values` key is not present in the tool response
                    # relevant issue: https://github.com/BerriAI/litellm/issues/6741
                    _message = litellm.Message(content=json.dumps(args))
                    return _message
        except json.JSONDecodeError:
            # json decode error does occur, return the original tool response str
            return litellm.Message(content=json_mode_content_str)
        return None

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, httpx.Headers]
    ) -> BaseLLMException:
        return AnthropicError(
            status_code=status_code,
            message=error_message,
            headers=cast(httpx.Headers, headers),
        )


def _valid_user_id(user_id: str) -> bool:
    """
    Validate that user_id is not an email or phone number.
    Returns: bool: True if valid (not email or phone), False otherwise
    """
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    phone_pattern = r"^\+?[\d\s\(\)-]{7,}$"

    if re.match(email_pattern, user_id):
        return False
    if re.match(phone_pattern, user_id):
        return False

    return True
