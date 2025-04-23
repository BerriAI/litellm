"""
Translating between OpenAI's `/chat/completion` format and Amazon's `/converse` format
"""

import copy
import time
import types
from typing import List, Literal, Optional, Tuple, Union, cast, overload

import httpx

import litellm
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.prompt_templates.factory import (
    BedrockConverseMessagesProcessor,
    _bedrock_converse_messages_pt,
    _bedrock_tools_pt,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.bedrock import *
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionResponseMessage,
    ChatCompletionSystemMessage,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
    OpenAIChatCompletionToolParam,
    OpenAIMessageContentListBlock,
)
from litellm.types.utils import ModelResponse, PromptTokensDetailsWrapper, Usage
from litellm.utils import add_dummy_tool, has_tool_call_blocks

from ..common_utils import BedrockError, BedrockModelInfo, get_bedrock_tool_name


class AmazonConverseConfig(BaseConfig):
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    #2 - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features
    """

    maxTokens: Optional[int]
    stopSequences: Optional[List[str]]
    temperature: Optional[int]
    topP: Optional[int]
    topK: Optional[int]

    def __init__(
        self,
        maxTokens: Optional[int] = None,
        stopSequences: Optional[List[str]] = None,
        temperature: Optional[int] = None,
        topP: Optional[int] = None,
        topK: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "bedrock_converse"

    @classmethod
    def get_config_blocks(cls) -> dict:
        return {
            "guardrailConfig": GuardrailConfigBlock,
            "performanceConfig": PerformanceConfigBlock,
        }

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

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = [
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stream_options",
            "stop",
            "temperature",
            "top_p",
            "extra_headers",
            "response_format",
        ]

        ## Filter out 'cross-region' from model name
        base_model = BedrockModelInfo.get_base_model(model)

        if (
            base_model.startswith("anthropic")
            or base_model.startswith("mistral")
            or base_model.startswith("cohere")
            or base_model.startswith("meta.llama3-1")
            or base_model.startswith("meta.llama3-2")
            or base_model.startswith("meta.llama3-3")
            or base_model.startswith("amazon.nova")
        ):
            supported_params.append("tools")

        if litellm.utils.supports_tool_choice(
            model=model, custom_llm_provider=self.custom_llm_provider
        ):
            # only anthropic and mistral support tool choice config. otherwise (E.g. cohere) will fail the call - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            supported_params.append("tool_choice")

        if (
            "claude-3-7" in model
        ):  # [TODO]: move to a 'supports_reasoning_content' param from model cost map
            supported_params.append("thinking")
            supported_params.append("reasoning_effort")
        return supported_params

    def map_tool_choice_values(
        self, model: str, tool_choice: Union[str, dict], drop_params: bool
    ) -> Optional[ToolChoiceValuesBlock]:
        if tool_choice == "none":
            if litellm.drop_params is True or drop_params is True:
                return None
            else:
                raise litellm.utils.UnsupportedParamsError(
                    message="Bedrock doesn't support tool_choice={}. To drop it from the call, set `litellm.drop_params = True.".format(
                        tool_choice
                    ),
                    status_code=400,
                )
        elif tool_choice == "required":
            return ToolChoiceValuesBlock(any={})
        elif tool_choice == "auto":
            return ToolChoiceValuesBlock(auto={})
        elif isinstance(tool_choice, dict):
            # only supported for anthropic + mistral models - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            specific_tool = SpecificToolChoiceBlock(
                name=tool_choice.get("function", {}).get("name", "")
            )
            return ToolChoiceValuesBlock(tool=specific_tool)
        else:
            raise litellm.utils.UnsupportedParamsError(
                message="Bedrock doesn't support tool_choice={}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.".format(
                    tool_choice
                ),
                status_code=400,
            )

    def get_supported_image_types(self) -> List[str]:
        return ["png", "jpeg", "gif", "webp"]

    def get_supported_document_types(self) -> List[str]:
        return ["pdf", "csv", "doc", "docx", "xls", "xlsx", "html", "txt", "md"]

    def get_all_supported_content_types(self) -> List[str]:
        return self.get_supported_image_types() + self.get_supported_document_types()

    def _create_json_tool_call_for_response_format(
        self,
        json_schema: Optional[dict] = None,
        schema_name: str = "json_tool_call",
        description: Optional[str] = None,
    ) -> ChatCompletionToolParam:
        """
        Handles creating a tool call for getting responses in JSON format.

        Args:
            json_schema (Optional[dict]): The JSON schema the response should be in

        Returns:
            AnthropicMessagesTool: The tool call to send to Anthropic API to get responses in JSON format
        """

        if json_schema is None:
            # Anthropic raises a 400 BadRequest error if properties is passed as None
            # see usage with additionalProperties (Example 5) https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb
            _input_schema = {
                "type": "object",
                "additionalProperties": True,
                "properties": {},
            }
        else:
            _input_schema = json_schema

        tool_param_function_chunk = ChatCompletionToolParamFunctionChunk(
            name=schema_name, parameters=_input_schema
        )
        if description:
            tool_param_function_chunk["description"] = description

        _tool = ChatCompletionToolParam(
            type="function",
            function=tool_param_function_chunk,
        )
        return _tool

    def _apply_tool_call_transformation(
        self,
        tools: List[OpenAIChatCompletionToolParam],
        model: str,
        non_default_params: dict,
        optional_params: dict,
    ):
        optional_params = self._add_tools_to_optional_params(
            optional_params=optional_params, tools=tools
        )

        if (
            "meta.llama3-3-70b-instruct-v1:0" in model
            and non_default_params.get("stream", False) is True
        ):
            optional_params["fake_stream"] = True

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        is_thinking_enabled = self.is_thinking_enabled(non_default_params)

        for param, value in non_default_params.items():
            if param == "response_format" and isinstance(value, dict):
                ignore_response_format_types = ["text"]
                if value["type"] in ignore_response_format_types:  # value is a no-op
                    continue

                json_schema: Optional[dict] = None
                schema_name: str = ""
                description: Optional[str] = None
                if "response_schema" in value:
                    json_schema = value["response_schema"]
                    schema_name = "json_tool_call"
                elif "json_schema" in value:
                    json_schema = value["json_schema"]["schema"]
                    schema_name = value["json_schema"]["name"]
                    description = value["json_schema"].get("description")

                if "type" in value and value["type"] == "text":
                    continue

                """
                Follow similar approach to anthropic - translate to a single tool call. 

                When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
                - You usually want to provide a single tool
                - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
                - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.
                """
                _tool = self._create_json_tool_call_for_response_format(
                    json_schema=json_schema,
                    schema_name=schema_name if schema_name != "" else "json_tool_call",
                    description=description,
                )
                optional_params = self._add_tools_to_optional_params(
                    optional_params=optional_params, tools=[_tool]
                )
                if (
                    litellm.utils.supports_tool_choice(
                        model=model, custom_llm_provider=self.custom_llm_provider
                    )
                    and not is_thinking_enabled
                ):
                    optional_params["tool_choice"] = ToolChoiceValuesBlock(
                        tool=SpecificToolChoiceBlock(
                            name=schema_name if schema_name != "" else "json_tool_call"
                        )
                    )
                optional_params["json_mode"] = True
                if non_default_params.get("stream", False) is True:
                    optional_params["fake_stream"] = True
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["maxTokens"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                if isinstance(value, str):
                    if len(value) == 0:  # converse raises error for empty strings
                        continue
                    value = [value]
                optional_params["stopSequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["topP"] = value
            if param == "tools" and isinstance(value, list):
                self._apply_tool_call_transformation(
                    tools=cast(List[OpenAIChatCompletionToolParam], value),
                    model=model,
                    non_default_params=non_default_params,
                    optional_params=optional_params,
                )
            if param == "tool_choice":
                _tool_choice_value = self.map_tool_choice_values(
                    model=model, tool_choice=value, drop_params=drop_params  # type: ignore
                )
                if _tool_choice_value is not None:
                    optional_params["tool_choice"] = _tool_choice_value
            if param == "thinking":
                optional_params["thinking"] = value
            elif param == "reasoning_effort" and isinstance(value, str):
                optional_params["thinking"] = AnthropicConfig._map_reasoning_effort(
                    value
                )

        self.update_optional_params_with_thinking_tokens(
            non_default_params=non_default_params, optional_params=optional_params
        )

        return optional_params

    @overload
    def _get_cache_point_block(
        self,
        message_block: Union[
            OpenAIMessageContentListBlock,
            ChatCompletionUserMessage,
            ChatCompletionSystemMessage,
        ],
        block_type: Literal["system"],
    ) -> Optional[SystemContentBlock]:
        pass

    @overload
    def _get_cache_point_block(
        self,
        message_block: Union[
            OpenAIMessageContentListBlock,
            ChatCompletionUserMessage,
            ChatCompletionSystemMessage,
        ],
        block_type: Literal["content_block"],
    ) -> Optional[ContentBlock]:
        pass

    def _get_cache_point_block(
        self,
        message_block: Union[
            OpenAIMessageContentListBlock,
            ChatCompletionUserMessage,
            ChatCompletionSystemMessage,
        ],
        block_type: Literal["system", "content_block"],
    ) -> Optional[Union[SystemContentBlock, ContentBlock]]:
        if message_block.get("cache_control", None) is None:
            return None
        if block_type == "system":
            return SystemContentBlock(cachePoint=CachePointBlock(type="default"))
        else:
            return ContentBlock(cachePoint=CachePointBlock(type="default"))

    def _transform_system_message(
        self, messages: List[AllMessageValues]
    ) -> Tuple[List[AllMessageValues], List[SystemContentBlock]]:
        system_prompt_indices = []
        system_content_blocks: List[SystemContentBlock] = []
        for idx, message in enumerate(messages):
            if message["role"] == "system":
                system_prompt_indices.append(idx)
                if isinstance(message["content"], str) and message["content"]:
                    system_content_blocks.append(
                        SystemContentBlock(text=message["content"])
                    )
                    cache_block = self._get_cache_point_block(
                        message, block_type="system"
                    )
                    if cache_block:
                        system_content_blocks.append(cache_block)
                elif isinstance(message["content"], list):
                    for m in message["content"]:
                        if m.get("type") == "text" and m.get("text"):
                            system_content_blocks.append(
                                SystemContentBlock(text=m["text"])
                            )
                            cache_block = self._get_cache_point_block(
                                m, block_type="system"
                            )
                            if cache_block:
                                system_content_blocks.append(cache_block)
        if len(system_prompt_indices) > 0:
            for idx in reversed(system_prompt_indices):
                messages.pop(idx)
        return messages, system_content_blocks

    def _transform_inference_params(self, inference_params: dict) -> InferenceConfig:
        if "top_k" in inference_params:
            inference_params["topK"] = inference_params.pop("top_k")
        return InferenceConfig(**inference_params)

    def _handle_top_k_value(self, model: str, inference_params: dict) -> dict:
        base_model = BedrockModelInfo.get_base_model(model)

        val_top_k = None
        if "topK" in inference_params:
            val_top_k = inference_params.pop("topK")
        elif "top_k" in inference_params:
            val_top_k = inference_params.pop("top_k")

        if val_top_k:
            if base_model.startswith("anthropic"):
                return {"top_k": val_top_k}
            if base_model.startswith("amazon.nova"):
                return {"inferenceConfig": {"topK": val_top_k}}

        return {}

    def _transform_request_helper(
        self,
        model: str,
        system_content_blocks: List[SystemContentBlock],
        optional_params: dict,
        messages: Optional[List[AllMessageValues]] = None,
    ) -> CommonRequestObject:
        ## VALIDATE REQUEST
        """
        Bedrock doesn't support tool calling without `tools=` param specified.
        """
        if (
            "tools" not in optional_params
            and messages is not None
            and has_tool_call_blocks(messages)
        ):
            if litellm.modify_params:
                optional_params["tools"] = add_dummy_tool(
                    custom_llm_provider="bedrock_converse"
                )
            else:
                raise litellm.UnsupportedParamsError(
                    message="Bedrock doesn't support tool calling without `tools=` param specified. Pass `tools=` param OR set `litellm.modify_params = True` // `litellm_settings::modify_params: True` to add dummy tool to the request.",
                    model="",
                    llm_provider="bedrock",
                )

        inference_params = copy.deepcopy(optional_params)
        supported_converse_params = list(
            AmazonConverseConfig.__annotations__.keys()
        ) + ["top_k"]
        supported_tool_call_params = ["tools", "tool_choice"]
        supported_config_params = list(self.get_config_blocks().keys())
        total_supported_params = (
            supported_converse_params
            + supported_tool_call_params
            + supported_config_params
        )
        inference_params.pop("json_mode", None)  # used for handling json_schema

        # keep supported params in 'inference_params', and set all model-specific params in 'additional_request_params'
        additional_request_params = {
            k: v for k, v in inference_params.items() if k not in total_supported_params
        }
        inference_params = {
            k: v for k, v in inference_params.items() if k in total_supported_params
        }

        # Only set the topK value in for models that support it
        additional_request_params.update(
            self._handle_top_k_value(model, inference_params)
        )

        bedrock_tools: List[ToolBlock] = _bedrock_tools_pt(
            inference_params.pop("tools", [])
        )
        bedrock_tool_config: Optional[ToolConfigBlock] = None
        if len(bedrock_tools) > 0:
            tool_choice_values: ToolChoiceValuesBlock = inference_params.pop(
                "tool_choice", None
            )
            bedrock_tool_config = ToolConfigBlock(
                tools=bedrock_tools,
            )
            if tool_choice_values is not None:
                bedrock_tool_config["toolChoice"] = tool_choice_values

        data: CommonRequestObject = {
            "additionalModelRequestFields": additional_request_params,
            "system": system_content_blocks,
            "inferenceConfig": self._transform_inference_params(
                inference_params=inference_params
            ),
        }

        # Handle all config blocks
        for config_name, config_class in self.get_config_blocks().items():
            config_value = inference_params.pop(config_name, None)
            if config_value is not None:
                data[config_name] = config_class(**config_value)  # type: ignore

        # Tool Config
        if bedrock_tool_config is not None:
            data["toolConfig"] = bedrock_tool_config

        return data

    async def _async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
    ) -> RequestObject:
        messages, system_content_blocks = self._transform_system_message(messages)
        ## TRANSFORMATION ##

        _data: CommonRequestObject = self._transform_request_helper(
            model=model,
            system_content_blocks=system_content_blocks,
            optional_params=optional_params,
            messages=messages,
        )

        bedrock_messages = (
            await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
                messages=messages,
                model=model,
                llm_provider="bedrock_converse",
                user_continue_message=litellm_params.pop("user_continue_message", None),
            )
        )

        data: RequestObject = {"messages": bedrock_messages, **_data}

        return data

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return cast(
            dict,
            self._transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
            ),
        )

    def _transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
    ) -> RequestObject:
        messages, system_content_blocks = self._transform_system_message(messages)

        _data: CommonRequestObject = self._transform_request_helper(
            model=model,
            system_content_blocks=system_content_blocks,
            optional_params=optional_params,
            messages=messages,
        )

        ## TRANSFORMATION ##
        bedrock_messages: List[MessageBlock] = _bedrock_converse_messages_pt(
            messages=messages,
            model=model,
            llm_provider="bedrock_converse",
            user_continue_message=litellm_params.pop("user_continue_message", None),
        )

        data: RequestObject = {"messages": bedrock_messages, **_data}

        return data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Logging,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        return self._transform_response(
            model=model,
            response=raw_response,
            model_response=model_response,
            stream=optional_params.get("stream", False),
            logging_obj=logging_obj,
            optional_params=optional_params,
            api_key=api_key,
            data=request_data,
            messages=messages,
            encoding=encoding,
        )

    def _transform_reasoning_content(
        self, reasoning_content_blocks: List[BedrockConverseReasoningContentBlock]
    ) -> str:
        """
        Extract the reasoning text from the reasoning content blocks

        Ensures deepseek reasoning content compatible output.
        """
        reasoning_content_str = ""
        for block in reasoning_content_blocks:
            if "reasoningText" in block:
                reasoning_content_str += block["reasoningText"]["text"]
        return reasoning_content_str

    def _transform_thinking_blocks(
        self, thinking_blocks: List[BedrockConverseReasoningContentBlock]
    ) -> List[Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]]:
        """Return a consistent format for thinking blocks between Anthropic and Bedrock."""
        thinking_blocks_list: List[
            Union[ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock]
        ] = []
        for block in thinking_blocks:
            if "reasoningText" in block:
                _thinking_block = ChatCompletionThinkingBlock(type="thinking")
                _text = block["reasoningText"].get("text")
                _signature = block["reasoningText"].get("signature")
                if _text is not None:
                    _thinking_block["thinking"] = _text
                if _signature is not None:
                    _thinking_block["signature"] = _signature
                thinking_blocks_list.append(_thinking_block)
            elif "redactedContent" in block:
                _redacted_block = ChatCompletionRedactedThinkingBlock(
                    type="redacted_thinking", data=block["redactedContent"]
                )
                thinking_blocks_list.append(_redacted_block)
        return thinking_blocks_list

    def _transform_usage(self, usage: ConverseTokenUsageBlock) -> Usage:
        input_tokens = usage["inputTokens"]
        output_tokens = usage["outputTokens"]
        total_tokens = usage["totalTokens"]
        cache_creation_input_tokens: int = 0
        cache_read_input_tokens: int = 0

        if "cacheReadInputTokens" in usage:
            cache_read_input_tokens = usage["cacheReadInputTokens"]
            input_tokens += cache_read_input_tokens
        if "cacheWriteInputTokens" in usage:
            """
            Do not increment prompt_tokens with cacheWriteInputTokens
            """
            cache_creation_input_tokens = usage["cacheWriteInputTokens"]

        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=cache_read_input_tokens
        )
        openai_usage = Usage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=total_tokens,
            prompt_tokens_details=prompt_tokens_details,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        return openai_usage

    def _transform_response(
        self,
        model: str,
        response: httpx.Response,
        model_response: ModelResponse,
        stream: bool,
        logging_obj: Optional[Logging],
        optional_params: dict,
        api_key: Optional[str],
        data: Union[dict, str],
        messages: List,
        encoding,
    ) -> ModelResponse:
        ## LOGGING
        if logging_obj is not None:
            logging_obj.post_call(
                input=messages,
                api_key=api_key,
                original_response=response.text,
                additional_args={"complete_input_dict": data},
            )

        json_mode: Optional[bool] = optional_params.pop("json_mode", None)
        ## RESPONSE OBJECT
        try:
            completion_response = ConverseResponseBlock(**response.json())  # type: ignore
        except Exception as e:
            raise BedrockError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    response.text, str(e)
                ),
                status_code=422,
            )

        """
        Bedrock Response Object has optional message block 

        completion_response["output"].get("message", None)

        A message block looks like this (Example 1): 
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "text": "Is there anything else you'd like to talk about? Perhaps I can help with some economic questions or provide some information about economic concepts?"
                    }
                ]
            }
        },
        (Example 2):
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_hbTgdi0CSLq_hM4P8csZJA",
                            "name": "top_song",
                            "input": {
                                "sign": "WZPZ"
                            }
                        }
                    }
                ]
            }
        }

        """
        message: Optional[MessageBlock] = completion_response["output"]["message"]
        chat_completion_message: ChatCompletionResponseMessage = {"role": "assistant"}
        content_str = ""
        tools: List[ChatCompletionToolCallChunk] = []
        reasoningContentBlocks: Optional[
            List[BedrockConverseReasoningContentBlock]
        ] = None

        if message is not None:
            for idx, content in enumerate(message["content"]):
                """
                - Content is either a tool response or text
                """
                if "text" in content:
                    content_str += content["text"]
                if "toolUse" in content:
                    ## check tool name was formatted by litellm
                    _response_tool_name = content["toolUse"]["name"]
                    response_tool_name = get_bedrock_tool_name(
                        response_tool_name=_response_tool_name
                    )
                    _function_chunk = ChatCompletionToolCallFunctionChunk(
                        name=response_tool_name,
                        arguments=json.dumps(content["toolUse"]["input"]),
                    )

                    _tool_response_chunk = ChatCompletionToolCallChunk(
                        id=content["toolUse"]["toolUseId"],
                        type="function",
                        function=_function_chunk,
                        index=idx,
                    )
                    tools.append(_tool_response_chunk)
                if "reasoningContent" in content:
                    if reasoningContentBlocks is None:
                        reasoningContentBlocks = []
                    reasoningContentBlocks.append(content["reasoningContent"])

        if reasoningContentBlocks is not None:
            chat_completion_message["provider_specific_fields"] = {
                "reasoningContentBlocks": reasoningContentBlocks,
            }
            chat_completion_message[
                "reasoning_content"
            ] = self._transform_reasoning_content(reasoningContentBlocks)
            chat_completion_message[
                "thinking_blocks"
            ] = self._transform_thinking_blocks(reasoningContentBlocks)
        chat_completion_message["content"] = content_str
        if json_mode is True and tools is not None and len(tools) == 1:
            # to support 'json_schema' logic on bedrock models
            json_mode_content_str: Optional[str] = tools[0]["function"].get("arguments")
            if json_mode_content_str is not None:
                chat_completion_message["content"] = json_mode_content_str
        else:
            chat_completion_message["tool_calls"] = tools

        ## CALCULATING USAGE - bedrock returns usage in the headers
        usage = self._transform_usage(completion_response["usage"])

        model_response.choices = [
            litellm.Choices(
                finish_reason=map_finish_reason(completion_response["stopReason"]),
                index=0,
                message=litellm.Message(**chat_completion_message),
            )
        ]
        model_response.created = int(time.time())
        model_response.model = model

        setattr(model_response, "usage", usage)

        # Add "trace" from Bedrock guardrails - if user has opted in to returning it
        if "trace" in completion_response:
            setattr(model_response, "trace", completion_response["trace"])

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BedrockError(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

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
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers
