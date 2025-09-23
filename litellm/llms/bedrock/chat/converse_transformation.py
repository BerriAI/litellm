"""
Translating between OpenAI's `/chat/completion` format and Amazon's `/converse` format
"""

import copy
import time
import types
from typing import List, Literal, Optional, Tuple, Union, cast, overload

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    _parse_content_for_reasoning,
)
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
    ChatCompletionAssistantMessage,
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
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage,
)
from litellm.utils import add_dummy_tool, has_tool_call_blocks, supports_reasoning

from ..common_utils import (
    BedrockError,
    BedrockModelInfo,
    get_anthropic_beta_from_headers,
    get_bedrock_tool_name,
)

# Computer use tool prefixes supported by Bedrock
BEDROCK_COMPUTER_USE_TOOLS = [
    "computer_use_preview",
    "computer_",
    "bash_",
    "text_editor_",
]


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

    @staticmethod
    def _convert_consecutive_user_messages_to_guarded_text(
        messages: List[AllMessageValues], optional_params: dict
    ) -> List[AllMessageValues]:
        """
        Convert consecutive user messages at the end to guarded_text type if guardrailConfig is present
        and no guarded_text is already present in those messages.
        """
        # Check if guardrailConfig is present
        if "guardrailConfig" not in optional_params:
            return messages

        # Find all consecutive user messages at the end
        consecutive_user_message_indices = []
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                consecutive_user_message_indices.append(i)
            else:
                break

        if not consecutive_user_message_indices:
            return messages

        # Process each consecutive user message
        messages_copy = copy.deepcopy(messages)
        for user_message_index in consecutive_user_message_indices:
            user_message = messages_copy[user_message_index]
            content = user_message.get("content", [])

            if isinstance(content, list):
                has_guarded_text = any(
                    isinstance(item, dict) and item.get("type") == "guarded_text"
                    for item in content
                )
                if has_guarded_text:
                    continue  # Skip this message if it already has guarded_text

                # Convert text elements to guarded_text
                new_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        new_item = {"type": "guarded_text", "text": item["text"]}  # type: ignore
                        new_content.append(new_item)
                    else:
                        new_content.append(item)

                messages_copy[user_message_index]["content"] = new_content  # type: ignore
            elif isinstance(content, str):
                # If content is a string, convert it to guarded_text
                messages_copy[user_message_index]["content"] = [  # type: ignore
                    {"type": "guarded_text", "text": content}  # type: ignore
                ]

        return messages_copy

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

    def _validate_request_metadata(self, metadata: dict) -> None:
        """
        Validate requestMetadata according to AWS Bedrock Converse API constraints.

        Constraints:
        - Maximum of 16 items
        - Keys: 1-256 characters, pattern [a-zA-Z0-9\\s:_@$#=/+,-.]{1,256}
        - Values: 0-256 characters, pattern [a-zA-Z0-9\\s:_@$#=/+,-.]{0,256}
        """
        import re

        if not isinstance(metadata, dict):
            raise litellm.exceptions.BadRequestError(
                message="requestMetadata must be a dictionary",
                model="bedrock",
                llm_provider="bedrock",
            )

        if len(metadata) > 16:
            raise litellm.exceptions.BadRequestError(
                message="requestMetadata can contain a maximum of 16 items",
                model="bedrock",
                llm_provider="bedrock",
            )

        key_pattern = re.compile(r"^[a-zA-Z0-9\s:_@$#=/+,.-]{1,256}$")
        value_pattern = re.compile(r"^[a-zA-Z0-9\s:_@$#=/+,.-]{0,256}$")

        for key, value in metadata.items():
            if not isinstance(key, str):
                raise litellm.exceptions.BadRequestError(
                    message="requestMetadata keys must be strings",
                    model="bedrock",
                    llm_provider="bedrock",
                )

            if not isinstance(value, str):
                raise litellm.exceptions.BadRequestError(
                    message="requestMetadata values must be strings",
                    model="bedrock",
                    llm_provider="bedrock",
                )

            if len(key) == 0 or len(key) > 256:
                raise litellm.exceptions.BadRequestError(
                    message="requestMetadata key length must be 1-256 characters",
                    model="bedrock",
                    llm_provider="bedrock",
                )

            if len(value) > 256:
                raise litellm.exceptions.BadRequestError(
                    message="requestMetadata value length must be 0-256 characters",
                    model="bedrock",
                    llm_provider="bedrock",
                )

            if not key_pattern.match(key):
                raise litellm.exceptions.BadRequestError(
                    message=f"requestMetadata key '{key}' contains invalid characters. Allowed: [a-zA-Z0-9\\s:_@$#=/+,.-]",
                    model="bedrock",
                    llm_provider="bedrock",
                )

            if not value_pattern.match(value):
                raise litellm.exceptions.BadRequestError(
                    message=f"requestMetadata value '{value}' contains invalid characters. Allowed: [a-zA-Z0-9\\s:_@$#=/+,.-]",
                    model="bedrock",
                    llm_provider="bedrock",
                )

    def get_supported_openai_params(self, model: str) -> List[str]:
        from litellm.utils import supports_function_calling

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
            "requestMetadata",
        ]

        if (
            "arn" in model
        ):  # we can't infer the model from the arn, so just add all params
            supported_params.append("tools")
            supported_params.append("tool_choice")
            supported_params.append("thinking")
            supported_params.append("reasoning_effort")
            return supported_params

        ## Filter out 'cross-region' from model name
        base_model = BedrockModelInfo.get_base_model(model)

        if (
            base_model.startswith("anthropic")
            or base_model.startswith("mistral")
            or base_model.startswith("cohere")
            or base_model.startswith("meta.llama3-1")
            or base_model.startswith("meta.llama3-2")
            or base_model.startswith("meta.llama3-3")
            or base_model.startswith("meta.llama4")
            or base_model.startswith("amazon.nova")
            or supports_function_calling(
                model=model, custom_llm_provider=self.custom_llm_provider
            )
        ):
            supported_params.append("tools")

        if litellm.utils.supports_tool_choice(
            model=model, custom_llm_provider=self.custom_llm_provider
        ) or litellm.utils.supports_tool_choice(
            model=base_model, custom_llm_provider=self.custom_llm_provider
        ):
            # only anthropic and mistral support tool choice config. otherwise (E.g. cohere) will fail the call - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            supported_params.append("tool_choice")

        if "gpt-oss" in model:
            supported_params.append("reasoning_effort")
        elif (
            "claude-3-7" in model
            or "claude-sonnet-4" in model
            or "claude-opus-4" in model
            or "deepseek.r1" in model
            or supports_reasoning(
                model=model,
                custom_llm_provider=self.custom_llm_provider,
            )
            or supports_reasoning(
                model=base_model, custom_llm_provider=self.custom_llm_provider
            )
        ):
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

    def get_supported_video_types(self) -> List[str]:
        return ["mp4", "mov", "mkv", "webm", "flv", "mpeg", "mpg", "wmv", "3gp"]

    def get_all_supported_content_types(self) -> List[str]:
        return (
            self.get_supported_image_types()
            + self.get_supported_document_types()
            + self.get_supported_video_types()
        )

    def is_computer_use_tool_used(
        self, tools: Optional[List[OpenAIChatCompletionToolParam]], model: str
    ) -> bool:
        """Check if computer use tools are being used in the request."""
        if tools is None:
            return False

        for tool in tools:
            if "type" in tool:
                tool_type = tool["type"]
                for computer_use_prefix in BEDROCK_COMPUTER_USE_TOOLS:
                    if tool_type.startswith(computer_use_prefix):
                        return True
        return False

    def _transform_computer_use_tools(
        self, computer_use_tools: List[OpenAIChatCompletionToolParam]
    ) -> List[dict]:
        """Transform computer use tools to Bedrock format."""
        transformed_tools: List[dict] = []

        for tool in computer_use_tools:
            tool_type = tool.get("type", "")

            # Check if this is a computer use tool with the startswith method
            is_computer_use_tool = False
            for computer_use_prefix in BEDROCK_COMPUTER_USE_TOOLS:
                if tool_type.startswith(computer_use_prefix):
                    is_computer_use_tool = True
                    break

            transformed_tool: dict = {}
            if is_computer_use_tool:
                if tool_type.startswith("computer_") and "function" in tool:
                    # Computer use tool with function format
                    func = tool["function"]
                    transformed_tool = {
                        "type": tool_type,
                        "name": func.get("name", "computer"),
                        **func.get("parameters", {}),
                    }
                else:
                    # Direct tools - just need to ensure name is present
                    transformed_tool = dict(tool)
                    if "name" not in transformed_tool:
                        if tool_type.startswith("bash_"):
                            transformed_tool["name"] = "bash"
                        elif tool_type.startswith("text_editor_"):
                            transformed_tool["name"] = "str_replace_editor"
            else:
                # Pass through other tools as-is
                transformed_tool = dict(tool)

            transformed_tools.append(transformed_tool)

        return transformed_tools

    def _separate_computer_use_tools(
        self, tools: List[OpenAIChatCompletionToolParam], model: str
    ) -> Tuple[
        List[OpenAIChatCompletionToolParam], List[OpenAIChatCompletionToolParam]
    ]:
        """
        Separate computer use tools from regular function tools.

        Args:
            tools: List of tools to separate
            model: The model name to check if it supports computer use

        Returns:
            Tuple of (computer_use_tools, regular_tools)
        """
        computer_use_tools = []
        regular_tools = []

        for tool in tools:
            if "type" in tool:
                tool_type = tool["type"]
                is_computer_use_tool = False
                for computer_use_prefix in BEDROCK_COMPUTER_USE_TOOLS:
                    if tool_type.startswith(computer_use_prefix):
                        is_computer_use_tool = True
                        break
                if is_computer_use_tool:
                    computer_use_tools.append(tool)
                else:
                    regular_tools.append(tool)
            else:
                regular_tools.append(tool)

        return computer_use_tools, regular_tools

    def _create_json_tool_call_for_response_format(
        self,
        json_schema: Optional[dict] = None,
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
            # Use the schema as-is for Bedrock
            # Bedrock requires the tool schema to be of type "object" and doesn't need unwrapping
            _input_schema = json_schema

        tool_param_function_chunk = ChatCompletionToolParamFunctionChunk(
            name=RESPONSE_FORMAT_TOOL_NAME, parameters=_input_schema
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
                optional_params = self._translate_response_format_param(
                    value=value,
                    model=model,
                    optional_params=optional_params,
                    non_default_params=non_default_params,
                    is_thinking_enabled=is_thinking_enabled,
                )
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
                if "gpt-oss" in model:
                    # GPT-OSS models: keep reasoning_effort as-is
                    # It will be passed through to additionalModelRequestFields
                    optional_params["reasoning_effort"] = value
                else:
                    # Anthropic and other models: convert to thinking parameter
                    optional_params["thinking"] = AnthropicConfig._map_reasoning_effort(
                        value
                    )
            if param == "requestMetadata":
                if value is not None and isinstance(value, dict):
                    self._validate_request_metadata(value)  # type: ignore
                    optional_params["requestMetadata"] = value

        # Only update thinking tokens for non-GPT-OSS models
        if "gpt-oss" not in model:
            self.update_optional_params_with_thinking_tokens(
                non_default_params=non_default_params, optional_params=optional_params
            )

        return optional_params

    def _translate_response_format_param(
        self,
        value: dict,
        model: str,
        optional_params: dict,
        non_default_params: dict,
        is_thinking_enabled: bool,
    ) -> dict:
        """
        Handles translation of response_format parameter to Bedrock format.

        Returns `optional_params` with the translated response_format parameter.
        """
        ignore_response_format_types = ["text"]
        if value["type"] in ignore_response_format_types:  # value is a no-op
            return optional_params

        json_schema: Optional[dict] = None
        description: Optional[str] = None
        if "response_schema" in value:
            json_schema = value["response_schema"]
        elif "json_schema" in value:
            json_schema = value["json_schema"]["schema"]
            description = value["json_schema"].get("description")

        if "type" in value and value["type"] == "text":
            return optional_params

        """
        Follow similar approach to anthropic - translate to a single tool call. 

        When using tools in this way: - https://docs.anthropic.com/en/docs/build-with-claude/tool-use#json-mode
        - You usually want to provide a single tool
        - You should set tool_choice (see Forcing tool use) to instruct the model to explicitly use that tool
        - Remember that the model will pass the input to the tool, so the name of the tool and description should be from the model’s perspective.
        """
        _tool = self._create_json_tool_call_for_response_format(
            json_schema=json_schema,
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
                tool=SpecificToolChoiceBlock(name=RESPONSE_FORMAT_TOOL_NAME)
            )
        optional_params["json_mode"] = True
        if non_default_params.get("stream", False) is True:
            optional_params["fake_stream"] = True

        return optional_params

    def update_optional_params_with_thinking_tokens(
        self, non_default_params: dict, optional_params: dict
    ):
        """
        Handles scenario where max tokens is not specified. For anthropic models (anthropic api/bedrock/vertex ai), this requires having the max tokens being set and being greater than the thinking token budget.

        Checks 'non_default_params' for 'thinking' and 'max_tokens'

        if 'thinking' is enabled and 'max_tokens' is not specified, set 'max_tokens' to the thinking token budget + DEFAULT_MAX_TOKENS
        """
        from litellm.constants import DEFAULT_MAX_TOKENS

        is_thinking_enabled = self.is_thinking_enabled(optional_params)
        is_max_tokens_in_request = self.is_max_tokens_in_request(non_default_params)
        if is_thinking_enabled and not is_max_tokens_in_request:
            thinking_token_budget = cast(dict, optional_params["thinking"]).get(
                "budget_tokens", None
            )
            if thinking_token_budget is not None:
                optional_params["maxTokens"] = (
                    thinking_token_budget + DEFAULT_MAX_TOKENS
                )

    @overload
    def _get_cache_point_block(
        self,
        message_block: Union[
            OpenAIMessageContentListBlock,
            ChatCompletionUserMessage,
            ChatCompletionSystemMessage,
            ChatCompletionAssistantMessage,
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
            ChatCompletionAssistantMessage,
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
            ChatCompletionAssistantMessage,
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

    def _prepare_request_params(
        self, optional_params: dict, model: str
    ) -> Tuple[dict, dict, dict]:
        """Prepare and separate request parameters."""
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

        # Extract requestMetadata before processing other parameters
        request_metadata = inference_params.pop("requestMetadata", None)
        if request_metadata is not None:
            self._validate_request_metadata(request_metadata)

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

        return inference_params, additional_request_params, request_metadata

    def _process_tools_and_beta(
        self,
        original_tools: list,
        model: str,
        headers: Optional[dict],
        additional_request_params: dict,
    ) -> Tuple[List[ToolBlock], list]:
        """Process tools and collect anthropic_beta values."""
        bedrock_tools: List[ToolBlock] = []

        # Collect anthropic_beta values from user headers
        anthropic_beta_list = []
        if headers:
            user_betas = get_anthropic_beta_from_headers(headers)
            anthropic_beta_list.extend(user_betas)

        # Only separate tools if computer use tools are actually present
        if original_tools and self.is_computer_use_tool_used(original_tools, model):
            # Separate computer use tools from regular function tools
            computer_use_tools, regular_tools = self._separate_computer_use_tools(
                original_tools, model
            )

            # Process regular function tools using existing logic
            bedrock_tools = _bedrock_tools_pt(regular_tools)

            # Add computer use tools and anthropic_beta if needed (only when computer use tools are present)
            if computer_use_tools:
                anthropic_beta_list.append("computer-use-2024-10-22")
                # Transform computer use tools to proper Bedrock format
                transformed_computer_tools = self._transform_computer_use_tools(
                    computer_use_tools
                )
                additional_request_params["tools"] = transformed_computer_tools
        else:
            # No computer use tools, process all tools as regular tools
            bedrock_tools = _bedrock_tools_pt(original_tools)

        # Set anthropic_beta in additional_request_params if we have any beta features
        if anthropic_beta_list:
            # Remove duplicates while preserving order
            unique_betas = []
            seen = set()
            for beta in anthropic_beta_list:
                if beta not in seen:
                    unique_betas.append(beta)
                    seen.add(beta)
            additional_request_params["anthropic_beta"] = unique_betas

        return bedrock_tools, anthropic_beta_list

    def _transform_request_helper(
        self,
        model: str,
        system_content_blocks: List[SystemContentBlock],
        optional_params: dict,
        messages: Optional[List[AllMessageValues]] = None,
        headers: Optional[dict] = None,
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

        # Prepare and separate parameters
        inference_params, additional_request_params, request_metadata = (
            self._prepare_request_params(optional_params, model)
        )

        original_tools = inference_params.pop("tools", [])

        # Process tools and collect beta values
        bedrock_tools, anthropic_beta_list = self._process_tools_and_beta(
            original_tools, model, headers, additional_request_params
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

        # Request Metadata (top-level field)
        if request_metadata is not None:
            data["requestMetadata"] = request_metadata

        return data

    async def _async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: Optional[dict] = None,
    ) -> RequestObject:
        messages, system_content_blocks = self._transform_system_message(messages)

        # Convert last user message to guarded_text if guardrailConfig is present
        messages = self._convert_consecutive_user_messages_to_guarded_text(
            messages, optional_params
        )
        ## TRANSFORMATION ##

        _data: CommonRequestObject = self._transform_request_helper(
            model=model,
            system_content_blocks=system_content_blocks,
            optional_params=optional_params,
            messages=messages,
            headers=headers,
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
                headers=headers,
            ),
        )

    def _transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: Optional[dict] = None,
    ) -> RequestObject:
        messages, system_content_blocks = self._transform_system_message(messages)

        # Convert last user message to guarded_text if guardrailConfig is present
        messages = self._convert_consecutive_user_messages_to_guarded_text(
            messages, optional_params
        )

        _data: CommonRequestObject = self._transform_request_helper(
            model=model,
            system_content_blocks=system_content_blocks,
            optional_params=optional_params,
            messages=messages,
            headers=headers,
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

    def get_tool_call_names(
        self,
        tools: Optional[
            Union[List[ToolBlock], List[OpenAIChatCompletionToolParam]]
        ] = None,
    ) -> List[str]:
        if tools is None:
            return []
        tool_set: set[str] = set()
        for tool in tools:
            tool_spec = tool.get("toolSpec")
            function = tool.get("function")
            if tool_spec is not None:
                _name = cast(dict, tool_spec).get("name")
                if _name is not None and isinstance(_name, str):
                    tool_set.add(_name)
            if function is not None:
                _name = cast(dict, function).get("name")
                if _name is not None and isinstance(_name, str):
                    tool_set.add(_name)
        return list(tool_set)

    def apply_tool_call_transformation_if_needed(
        self,
        message: Message,
        tools: Optional[List[ToolBlock]] = None,
        initial_finish_reason: Optional[str] = None,
    ) -> Tuple[Message, Optional[str]]:
        """
        Apply tool call transformation to a message.

        LLM providers (e.g. Bedrock, Vertex AI) sometimes return tool call in the response content.

        If the response content is a JSON object, we can parse it and return the tool call in the tool_calls field.
        """
        returned_finish_reason = initial_finish_reason
        if tools is None:
            return message, returned_finish_reason

        if message.content is not None:
            try:
                tool_call_names = self.get_tool_call_names(tools)
                json_content = json.loads(message.content)
                if (
                    json_content.get("type") == "function"
                    and json_content.get("name") in tool_call_names
                ):
                    tool_calls = [
                        ChatCompletionMessageToolCall(function=Function(**json_content))
                    ]

                    message.tool_calls = tool_calls
                    message.content = None
                    returned_finish_reason = "tool_calls"
            except Exception:
                pass

        return message, returned_finish_reason

    def _translate_message_content(self, content_blocks: List[ContentBlock]) -> Tuple[
        str,
        List[ChatCompletionToolCallChunk],
        Optional[List[BedrockConverseReasoningContentBlock]],
    ]:
        """
        Translate the message content to a string and a list of tool calls and reasoning content blocks

        Returns:
            content_str: str
            tools: List[ChatCompletionToolCallChunk]
            reasoningContentBlocks: Optional[List[BedrockConverseReasoningContentBlock]]
        """
        content_str = ""
        tools: List[ChatCompletionToolCallChunk] = []
        reasoningContentBlocks: Optional[List[BedrockConverseReasoningContentBlock]] = (
            None
        )
        for idx, content in enumerate(content_blocks):
            """
            - Content is either a tool response or text
            """
            extracted_reasoning_content_str: Optional[str] = None
            if "text" in content:
                (
                    extracted_reasoning_content_str,
                    _content_str,
                ) = _parse_content_for_reasoning(content["text"])
                if _content_str is not None:
                    content_str += _content_str
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
            if extracted_reasoning_content_str is not None:
                if reasoningContentBlocks is None:
                    reasoningContentBlocks = []
                reasoningContentBlocks.append(
                    BedrockConverseReasoningContentBlock(
                        reasoningText=BedrockConverseReasoningTextBlock(
                            text=extracted_reasoning_content_str,
                        )
                    )
                )
            if "reasoningContent" in content:
                if reasoningContentBlocks is None:
                    reasoningContentBlocks = []
                reasoningContentBlocks.append(content["reasoningContent"])

        return content_str, tools, reasoningContentBlocks

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
        reasoningContentBlocks: Optional[List[BedrockConverseReasoningContentBlock]] = (
            None
        )

        if message is not None:
            (
                content_str,
                tools,
                reasoningContentBlocks,
            ) = self._translate_message_content(message["content"])

        if reasoningContentBlocks is not None:
            chat_completion_message["provider_specific_fields"] = {
                "reasoningContentBlocks": reasoningContentBlocks,
            }
            chat_completion_message["reasoning_content"] = (
                self._transform_reasoning_content(reasoningContentBlocks)
            )
            chat_completion_message["thinking_blocks"] = (
                self._transform_thinking_blocks(reasoningContentBlocks)
            )
        chat_completion_message["content"] = content_str
        if (
            json_mode is True
            and tools is not None
            and len(tools) == 1
            and tools[0]["function"].get("name") == RESPONSE_FORMAT_TOOL_NAME
        ):
            verbose_logger.debug(
                "Processing JSON tool call response for response_format"
            )
            json_mode_content_str: Optional[str] = tools[0]["function"].get("arguments")
            if json_mode_content_str is not None:
                import json

                # Bedrock returns the response wrapped in a "properties" object
                # We need to extract the actual content from this wrapper
                try:
                    response_data = json.loads(json_mode_content_str)

                    # If Bedrock wrapped the response in "properties", extract the content
                    if (
                        isinstance(response_data, dict)
                        and "properties" in response_data
                        and len(response_data) == 1
                    ):
                        response_data = response_data["properties"]
                        json_mode_content_str = json.dumps(response_data)
                except json.JSONDecodeError:
                    # If parsing fails, use the original response
                    pass

                chat_completion_message["content"] = json_mode_content_str
        else:
            chat_completion_message["tool_calls"] = tools

        ## CALCULATING USAGE - bedrock returns usage in the headers
        usage = self._transform_usage(completion_response["usage"])

        ## HANDLE TOOL CALLS
        _message = Message(**chat_completion_message)
        initial_finish_reason = map_finish_reason(completion_response["stopReason"])

        (
            returned_message,
            returned_finish_reason,
        ) = self.apply_tool_call_transformation_if_needed(
            message=_message,
            tools=optional_params.get("tools"),
            initial_finish_reason=initial_finish_reason,
        )
        model_response.choices = [
            litellm.Choices(
                finish_reason=returned_finish_reason,
                index=0,
                message=returned_message,
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

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
        fake_stream: Optional[bool] = None,
    ) -> bool:
        """
        Returns True if the model/provider should fake stream
        """
        ###################################################################
        # If an upstream method already set fake_stream to True, return True
        ###################################################################
        if fake_stream is True:
            return True

        ###################################################################
        # Bedrock Converse Specific Logic
        ###################################################################
        if stream is True:
            if model is not None:
                ###################################################################
                # GPT-OSS models do not support streaming
                ###################################################################
                if "gpt-oss" in model:
                    return True
                ###################################################################
                # AI21 models do not support streaming
                ###################################################################
                if "ai21" in model:
                    return True
        return False
