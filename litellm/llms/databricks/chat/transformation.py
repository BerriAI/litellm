"""
Translates from OpenAI's `/v1/chat/completions` to Databricks' `/chat/completions`
"""

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Coroutine,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

import httpx
from pydantic import BaseModel

from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _handle_invalid_parallel_tool_calls,
    _should_convert_tool_call_to_json_mode,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
    strip_name_from_messages,
)
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.anthropic import AllAnthropicToolsValues
from litellm.types.llms.databricks import (
    AllDatabricksContentValues,
    DatabricksChoice,
    DatabricksFunction,
    DatabricksResponse,
    DatabricksTool,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    ChatCompletionToolChoiceFunctionParam,
    ChatCompletionToolChoiceObjectParam,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Message,
    ModelResponse,
    ModelResponseStream,
    ProviderField,
    Usage,
)

from ...anthropic.chat.transformation import AnthropicConfig
from ...openai_like.chat.transformation import OpenAILikeChatConfig
from ..common_utils import DatabricksBase, DatabricksException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class DatabricksConfig(DatabricksBase, OpenAILikeChatConfig, AnthropicConfig):
    """
    Reference: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chat-request
    """

    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    stop: Optional[Union[List[str], str]] = None
    n: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        stop: Optional[Union[List[str], str]] = None,
        n: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_required_params(self) -> List[ProviderField]:
        """For a given provider, return it's required fields with a description"""
        return [
            ProviderField(
                field_name="api_key",
                field_type="string",
                field_description="Your Databricks API Key.",
                field_value="dapi...",
            ),
            ProviderField(
                field_name="api_base",
                field_type="string",
                field_description="Your Databricks API Base.",
                field_value="https://adb-..",
            ),
        ]

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        api_base, headers = self.databricks_validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="chat_completions",
            custom_endpoint=False,
            headers=headers,
        )
        # Ensure Content-Type header is set
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = self._get_api_base(api_base)
        complete_url = f"{api_base}/chat/completions"
        return complete_url

    def get_supported_openai_params(self, model: Optional[str] = None) -> list:
        return [
            "stream",
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "response_format",
            "tools",
            "tool_choice",
            "reasoning_effort",
            "thinking",
        ]

    def convert_anthropic_tool_to_databricks_tool(
        self, tool: Optional[AllAnthropicToolsValues]
    ) -> Optional[DatabricksTool]:
        if tool is None:
            return None

        return DatabricksTool(
            type="function",
            function=DatabricksFunction(
                name=tool["name"],
                parameters=cast(dict, tool.get("input_schema") or {}),
            ),
        )

    def _map_openai_to_dbrx_tool(self, model: str, tools: List) -> List[DatabricksTool]:
        # if not claude, send as is
        if "claude" not in model:
            return tools

        # if claude, convert to anthropic tool and then to databricks tool
        anthropic_tools, _ = self._map_tools(
            tools=tools
        )  # unclear how mcp tool calling on databricks works
        databricks_tools = [
            cast(DatabricksTool, self.convert_anthropic_tool_to_databricks_tool(tool))
            for tool in anthropic_tools
        ]
        return databricks_tools

    def map_response_format_to_databricks_tool(
        self,
        model: str,
        value: Optional[dict],
        optional_params: dict,
        is_thinking_enabled: bool,
    ) -> Optional[DatabricksTool]:
        if value is None:
            return None

        tool = self.map_response_format_to_anthropic_tool(
            value, optional_params, is_thinking_enabled
        )

        databricks_tool = self.convert_anthropic_tool_to_databricks_tool(tool)
        return databricks_tool

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:
        is_thinking_enabled = self.is_thinking_enabled(non_default_params)
        mapped_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        if "tools" in mapped_params:
            mapped_params["tools"] = self._map_openai_to_dbrx_tool(
                model=model, tools=mapped_params["tools"]
            )
        if (
            "max_completion_tokens" in non_default_params
            and replace_max_completion_tokens_with_max_tokens
        ):
            mapped_params["max_tokens"] = non_default_params[
                "max_completion_tokens"
            ]  # most openai-compatible providers support 'max_tokens' not 'max_completion_tokens'
            mapped_params.pop("max_completion_tokens", None)

        if "response_format" in non_default_params and "claude" in model:
            _tool = self.map_response_format_to_databricks_tool(
                model,
                non_default_params["response_format"],
                mapped_params,
                is_thinking_enabled,
            )

            if _tool is not None:
                self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=[_tool]
                )
                optional_params["json_mode"] = True
                if not is_thinking_enabled:
                    _tool_choice = ChatCompletionToolChoiceObjectParam(
                        type="function",
                        function=ChatCompletionToolChoiceFunctionParam(
                            name=RESPONSE_FORMAT_TOOL_NAME
                        ),
                    )
                    optional_params["tool_choice"] = _tool_choice
            optional_params.pop(
                "response_format", None
            )  # unsupported for claude models - if json_schema -> convert to tool call

        if "reasoning_effort" in non_default_params and "claude" in model:
            optional_params["thinking"] = AnthropicConfig._map_reasoning_effort(
                non_default_params.get("reasoning_effort")
            )
            optional_params.pop("reasoning_effort", None)
        ## handle thinking tokens
        self.update_optional_params_with_thinking_tokens(
            non_default_params=non_default_params, optional_params=mapped_params
        )

        return mapped_params

    def _should_fake_stream(self, optional_params: dict) -> bool:
        """
        Databricks doesn't support 'response_format' while streaming
        """
        if optional_params.get("response_format") is not None:
            return True

        return False

    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Databricks does not support:
        - content in list format.
        - 'name' in user message.
        """
        new_messages = []
        for idx, message in enumerate(messages):
            if isinstance(message, BaseModel):
                _message = message.model_dump(exclude_none=True)
            else:
                _message = message
            new_messages.append(_message)
        new_messages = handle_messages_with_content_list_to_str_conversion(new_messages)
        new_messages = strip_name_from_messages(new_messages)

        if is_async:
            return super()._transform_messages(
                messages=new_messages, model=model, is_async=cast(Literal[True], True)
            )
        else:
            return super()._transform_messages(
                messages=new_messages, model=model, is_async=cast(Literal[False], False)
            )

    @staticmethod
    def extract_content_str(
        content: Optional[AllDatabricksContentValues],
    ) -> Optional[str]:
        if content is None:
            return None
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            content_str = ""
            for item in content:
                if item["type"] == "text":
                    content_str += item["text"]
            return content_str
        else:
            raise Exception(f"Unsupported content type: {type(content)}")

    @staticmethod
    def extract_reasoning_content(
        content: Optional[AllDatabricksContentValues],
    ) -> Tuple[
        Optional[str],
        Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ],
    ]:
        """
        Extract and return the reasoning content and thinking blocks
        """
        if content is None:
            return None, None
        thinking_blocks: Optional[
            List[
                Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
            ]
        ] = None
        reasoning_content: Optional[str] = None
        if isinstance(content, list):
            for item in content:
                if item["type"] == "reasoning":
                    for sum in item["summary"]:
                        if reasoning_content is None:
                            reasoning_content = ""
                        reasoning_content += sum["text"]
                        thinking_block = ChatCompletionThinkingBlock(
                            type="thinking",
                            thinking=sum.get("text", ""),
                            signature=sum.get("signature", ""),
                        )
                        if thinking_blocks is None:
                            thinking_blocks = []
                        thinking_blocks.append(thinking_block)
        return reasoning_content, thinking_blocks

    def _transform_dbrx_choices(
        self, choices: List[DatabricksChoice], json_mode: Optional[bool] = None
    ) -> List[Choices]:
        transformed_choices = []

        for choice in choices:
            ## HANDLE JSON MODE - anthropic returns single function call]
            tool_calls = choice["message"].get("tool_calls", None)
            if tool_calls is not None:
                _openai_tool_calls = []
                for _tc in tool_calls:
                    _openai_tc = ChatCompletionMessageToolCall(**_tc)  # type: ignore
                    _openai_tool_calls.append(_openai_tc)
                fixed_tool_calls = _handle_invalid_parallel_tool_calls(
                    _openai_tool_calls
                )

                if fixed_tool_calls is not None:
                    tool_calls = fixed_tool_calls

            translated_message: Optional[Message] = None
            finish_reason: Optional[str] = None
            if tool_calls and _should_convert_tool_call_to_json_mode(
                tool_calls=tool_calls,
                convert_tool_call_to_json_mode=json_mode,
            ):
                # to support response_format on claude models
                json_mode_content_str: Optional[str] = (
                    str(tool_calls[0]["function"].get("arguments", "")) or None
                )
                if json_mode_content_str is not None:
                    translated_message = Message(content=json_mode_content_str)
                    finish_reason = "stop"

            if translated_message is None:
                ## get the content str
                content_str = DatabricksConfig.extract_content_str(
                    choice["message"]["content"]
                )

                ## get the reasoning content
                (
                    reasoning_content,
                    thinking_blocks,
                ) = DatabricksConfig.extract_reasoning_content(
                    choice["message"].get("content")
                )

                translated_message = Message(
                    role="assistant",
                    content=content_str,
                    reasoning_content=reasoning_content,
                    thinking_blocks=thinking_blocks,
                    tool_calls=choice["message"].get("tool_calls"),
                )

            if finish_reason is None:
                finish_reason = choice["finish_reason"]

            translated_choice = Choices(
                finish_reason=finish_reason,
                index=choice["index"],
                message=translated_message,
                logprobs=None,
                enhancements=None,
            )

            transformed_choices.append(translated_choice)

        return transformed_choices

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
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
            completion_response = DatabricksResponse(**raw_response.json())  # type: ignore
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise DatabricksException(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), raw_response.text
                ),
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        model_response.model = completion_response["model"]
        model_response.id = completion_response["id"]
        model_response.created = completion_response["created"]
        setattr(model_response, "usage", Usage(**completion_response["usage"]))

        model_response.choices = self._transform_dbrx_choices(  # type: ignore
            choices=completion_response["choices"],
            json_mode=json_mode,
        )

        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return DatabricksChatResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )


class DatabricksChatResponseIterator(BaseModelResponseIterator):
    def __init__(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        super().__init__(streaming_response, sync_stream)

        self.json_mode = json_mode
        self._last_function_name = None  # Track the last seen function name

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            translated_choices = []
            for choice in chunk["choices"]:
                tool_calls = choice["delta"].get("tool_calls")
                if tool_calls and self.json_mode:
                    # 1. Check if the function name is set and == RESPONSE_FORMAT_TOOL_NAME
                    # 2. If no function name, just args -> check last function name (saved via state variable)
                    # 3. Convert args to json
                    # 4. Convert json to message
                    # 5. Set content to message.content
                    # 6. Set tool_calls to None
                    from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
                    from litellm.llms.base_llm.base_utils import (
                        _convert_tool_response_to_message,
                    )

                    # Check if this chunk has a function name
                    function_name = tool_calls[0].get("function", {}).get("name")
                    if function_name is not None:
                        self._last_function_name = function_name

                    # If we have a saved function name that matches RESPONSE_FORMAT_TOOL_NAME
                    # or this chunk has the matching function name
                    if (
                        self._last_function_name == RESPONSE_FORMAT_TOOL_NAME
                        or function_name == RESPONSE_FORMAT_TOOL_NAME
                    ):
                        # Convert tool calls to message format
                        message = _convert_tool_response_to_message(tool_calls)
                        if message is not None:
                            if message.content == "{}":  # empty json
                                message.content = ""
                            choice["delta"]["content"] = message.content
                            choice["delta"]["tool_calls"] = None
                elif tool_calls:
                    for _tc in tool_calls:
                        if _tc.get("function", {}).get("arguments") == "{}":
                            _tc["function"]["arguments"] = ""  # avoid invalid json
                # extract the content str
                content_str = DatabricksConfig.extract_content_str(
                    choice["delta"].get("content")
                )

                # extract the reasoning content
                (
                    reasoning_content,
                    thinking_blocks,
                ) = DatabricksConfig.extract_reasoning_content(
                    choice["delta"].get("content")
                )

                choice["delta"]["content"] = content_str
                choice["delta"]["reasoning_content"] = reasoning_content
                choice["delta"]["thinking_blocks"] = thinking_blocks
                translated_choices.append(choice)
            return ModelResponseStream(
                id=chunk["id"],
                object="chat.completion.chunk",
                created=chunk["created"],
                model=chunk["model"],
                choices=translated_choices,
            )
        except KeyError as e:
            raise DatabricksException(
                message=f"KeyError: {e}, Got unexpected response from Databricks: {chunk}",
                status_code=400,
            )
        except Exception as e:
            raise e
