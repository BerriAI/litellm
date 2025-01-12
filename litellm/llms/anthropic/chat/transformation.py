import json
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import httpx

import litellm
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthropicComputerTool,
    AnthropicHostedTools,
    AnthropicInputSchema,
    AnthropicMessagesTool,
    AnthropicMessagesToolChoice,
    AnthropicSystemMessageContent,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionCachedContent,
    ChatCompletionSystemMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParam,
)
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import PromptTokensDetailsWrapper
from litellm.utils import ModelResponse, Usage, add_dummy_tool, has_tool_call_blocks

from ..common_utils import AnthropicError, process_anthropic_headers

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class AnthropicConfig(BaseConfig):
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
        return super().get_config()

    def get_supported_openai_params(self, model: str):
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
            "user",
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
        pdf_used: bool = False,
        is_vertex_request: bool = False,
    ) -> dict:

        betas = []
        if prompt_caching_set:
            betas.append("prompt-caching-2024-07-31")
        if computer_tool_used:
            betas.append("computer-use-2024-10-22")
        if pdf_used:
            betas.append("pdfs-2024-09-25")
        headers = {
            "anthropic-version": anthropic_version or "2023-06-01",
            "x-api-key": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        }

        # Don't send any beta headers to Vertex, Vertex has failed requests when they are sent
        if is_vertex_request is True:
            pass
        elif len(betas) > 0:
            headers["anthropic-beta"] = ",".join(betas)

        return headers

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
        elif isinstance(tool_choice, dict):
            _tool_name = tool_choice.get("function", {}).get("name")
            _tool_choice = AnthropicMessagesToolChoice(type="tool")
            if _tool_name is not None:
                _tool_choice["name"] = _tool_name

        if parallel_tool_use is not None:
            # Anthropic uses 'disable_parallel_tool_use' flag to determine if parallel tool use is allowed
            # this is the inverse of the openai flag.
            if _tool_choice is not None:
                _tool_choice["disable_parallel_tool_use"] = not parallel_tool_use
            else:  # use anthropic defaults and make sure to send the disable_parallel_tool_use flag
                _tool_choice = AnthropicMessagesToolChoice(
                    type="auto",
                    disable_parallel_tool_use=not parallel_tool_use,
                )
        return _tool_choice

    def _map_tool_helper(
        self, tool: ChatCompletionToolParam
    ) -> AllAnthropicToolsValues:
        returned_tool: Optional[AllAnthropicToolsValues] = None

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

    def _map_stop_sequences(
        self, stop: Optional[Union[str, List[str]]]
    ) -> Optional[List[str]]:
        new_stop: Optional[List[str]] = None
        if isinstance(stop, str):
            if stop.isspace() and litellm.drop_params is True:  # anthropic doesn't allow whitespace characters as stop-sequences
                return new_stop
            new_stop = [stop]
        elif isinstance(stop, list):
            new_v = []
            for v in stop:
                if v.isspace() and litellm.drop_params is True:  # anthropic doesn't allow whitespace characters as stop-sequences
                    continue
                new_v.append(v)
            if len(new_v) > 0:
                new_stop = new_v
        return new_stop

    def _add_tools_to_optional_params(
        self, optional_params: dict, tools: List[AllAnthropicToolsValues]
    ) -> dict:
        if "tools" not in optional_params:
            optional_params["tools"] = tools
        else:
            optional_params["tools"] = [
                *optional_params["tools"],
                *tools,
            ]
        return optional_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                # check if optional params already has tools
                tool_value = self._map_tools(value)
                optional_params = self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=tool_value
                )
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

                _tool_choice = {"name": RESPONSE_FORMAT_TOOL_NAME, "type": "tool"}
                _tool = self._create_json_tool_call_for_response_format(
                    json_schema=json_schema,
                )
                optional_params = self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=[_tool]
                )
                optional_params["tool_choice"] = _tool_choice
                optional_params["json_mode"] = True
            if param == "user":
                optional_params["metadata"] = {"user_id": value}

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
            _input_schema["properties"] = {"values": json_schema}

        _tool = AnthropicMessagesTool(
            name=RESPONSE_FORMAT_TOOL_NAME, input_schema=_input_schema
        )
        return _tool

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

    def is_pdf_used(self, messages: List[AllMessageValues]) -> bool:
        """
        Set to true if media passed into messages.

        """
        for message in messages:
            if (
                "content" in message
                and message["content"] is not None
                and isinstance(message["content"], list)
            ):
                for content in message["content"]:
                    if "type" in content and content["type"] != "text":
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
        if (
            "tools" not in optional_params
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

        ## Handle user_id in metadata
        _litellm_metadata = litellm_params.get("metadata", None)
        if (
            _litellm_metadata
            and isinstance(_litellm_metadata, dict)
            and "user_id" in _litellm_metadata
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
        _hidden_params: Dict = {}
        _hidden_params["additional_headers"] = process_anthropic_headers(
            dict(raw_response.headers)
        )
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
        if "error" in completion_response:
            response_headers = getattr(raw_response, "headers", None)
            raise AnthropicError(
                message=str(completion_response["error"]),
                status_code=raw_response.status_code,
                headers=response_headers,
            )
        else:
            text_content = ""
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

            _message = litellm.Message(
                tool_calls=tool_calls,
                content=text_content or None,
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
        prompt_tokens = completion_response["usage"]["input_tokens"]
        completion_tokens = completion_response["usage"]["output_tokens"]
        _usage = completion_response["usage"]
        cache_creation_input_tokens: int = 0
        cache_read_input_tokens: int = 0

        model_response.created = int(time.time())
        model_response.model = model
        if "cache_creation_input_tokens" in _usage:
            cache_creation_input_tokens = _usage["cache_creation_input_tokens"]
            prompt_tokens += cache_creation_input_tokens
        if "cache_read_input_tokens" in _usage:
            cache_read_input_tokens = _usage["cache_read_input_tokens"]
            prompt_tokens += cache_read_input_tokens

        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=cache_read_input_tokens
        )
        total_tokens = prompt_tokens + completion_tokens
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )

        setattr(model_response, "usage", usage)  # type: ignore

        model_response._hidden_params = _hidden_params
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

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        if api_key is None:
            raise litellm.AuthenticationError(
                message="Missing Anthropic API Key - A call is being made to anthropic but no key is set either in the environment variables or via params. Please set `ANTHROPIC_API_KEY` in your environment vars",
                llm_provider="anthropic",
                model=model,
            )

        tools = optional_params.get("tools")
        prompt_caching_set = self.is_cache_control_set(messages=messages)
        computer_tool_used = self.is_computer_tool_used(tools=tools)
        pdf_used = self.is_pdf_used(messages=messages)
        anthropic_headers = self.get_anthropic_headers(
            computer_tool_used=computer_tool_used,
            prompt_caching_set=prompt_caching_set,
            pdf_used=pdf_used,
            api_key=api_key,
            is_vertex_request=False,
        )

        headers = {**headers, **anthropic_headers}
        return headers
