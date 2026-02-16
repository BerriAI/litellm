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
from litellm.constants import (
    BEDROCK_MIN_THINKING_BUDGET_TOKENS,
    RESPONSE_FORMAT_TOOL_NAME,
)
from litellm.litellm_core_utils.core_helpers import (
    filter_exceptions_from_params,
    filter_internal_params,
    map_finish_reason,
    safe_deep_copy,
)
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
from litellm.utils import (
    add_dummy_tool,
    any_assistant_message_has_thinking_blocks,
    has_tool_call_blocks,
    last_assistant_with_tool_calls_has_no_thinking_blocks,
    supports_reasoning,
)

from ..common_utils import (
    BedrockError,
    BedrockModelInfo,
    get_anthropic_beta_from_headers,
    get_bedrock_tool_name,
    is_claude_4_5_on_bedrock,
)

# Computer use tool prefixes supported by Bedrock
BEDROCK_COMPUTER_USE_TOOLS = [
    "computer_use_preview",
    "computer_",
    "bash_",
    "text_editor_",
]

# Beta header patterns that are not supported by Bedrock Converse API
# These will be filtered out to prevent errors
UNSUPPORTED_BEDROCK_CONVERSE_BETA_PATTERNS = [
    "advanced-tool-use",  # Bedrock Converse doesn't support advanced-tool-use beta headers
    "prompt-caching",  # Prompt caching not supported in Converse API
    "compact-2026-01-12", # The compact beta feature is not currently supported on the Converse and ConverseStream APIs
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
            "serviceTier": ServiceTierBlock,
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

    def _is_nova_lite_2_model(self, model: str) -> bool:
        """
        Check if the model is a Nova Lite 2 model that supports reasoningConfig.

        Nova Lite 2 models use a different reasoning configuration structure compared to
        Anthropic's thinking parameter and GPT-OSS's reasoning_effort parameter.

        Supported models:
        - amazon.nova-2-lite-v1:0
        - us.amazon.nova-2-lite-v1:0
        - eu.amazon.nova-2-lite-v1:0
        - apac.amazon.nova-2-lite-v1:0

        Args:
            model: The model identifier

        Returns:
            True if the model is a Nova Lite 2 model, False otherwise

        Examples:
            >>> config = AmazonConverseConfig()
            >>> config._is_nova_lite_2_model("amazon.nova-2-lite-v1:0")
            True
            >>> config._is_nova_lite_2_model("us.amazon.nova-2-lite-v1:0")
            True
            >>> config._is_nova_lite_2_model("amazon.nova-pro-1-5-v1:0")
            False
            >>> config._is_nova_lite_2_model("amazon.nova-pro-v1:0")
            False
        """
        # Remove regional prefix if present (us., eu., apac.)
        model_without_region = model
        for prefix in ["us.", "eu.", "apac."]:
            if model.startswith(prefix):
                model_without_region = model[len(prefix) :]
                break

        # Check if the model is specifically Nova Lite 2
        return "nova-2-lite" in model_without_region

    def _map_web_search_options(
        self, web_search_options: dict, model: str
    ) -> Optional[BedrockToolBlock]:
        """
        Map web_search_options to Nova grounding systemTool.

        Nova grounding (web search) is only supported on Amazon Nova models.
        Returns None for non-Nova models.

        Args:
            web_search_options: The web_search_options dict from the request
            model: The model identifier string

        Returns:
            BedrockToolBlock with systemTool for Nova models, None otherwise

        Reference: https://docs.aws.amazon.com/nova/latest/userguide/grounding.html
        """
        # Only Nova models support nova_grounding
        # Model strings can be like: "amazon.nova-pro-v1:0", "us.amazon.nova-pro-v1:0", etc.
        if "nova" not in model.lower():
            verbose_logger.debug(
                f"web_search_options passed but model {model} is not a Nova model. "
                "Nova grounding is only supported on Amazon Nova models."
            )
            return None

        # Nova doesn't support search_context_size or user_location params
        # (unlike Anthropic), so we just enable grounding with no options
        return BedrockToolBlock(systemTool={"name": "nova_grounding"})

    def _transform_reasoning_effort_to_reasoning_config(
        self, reasoning_effort: str
    ) -> dict:
        """
        Transform reasoning_effort parameter to Nova 2 reasoningConfig structure.

        Nova 2 models use a reasoningConfig structure in additionalModelRequestFields
        that differs from both Anthropic's thinking parameter and GPT-OSS's reasoning_effort.

        Args:
            reasoning_effort: The reasoning effort level, must be "low" or "high"

        Returns:
            dict: A dictionary containing the reasoningConfig structure:
                {
                    "reasoningConfig": {
                        "type": "enabled",
                        "maxReasoningEffort": "low" | "medium" |"high"
                    }
                }

        Raises:
            BadRequestError: If reasoning_effort is not "low", "medium" or "high"

        Examples:
            >>> config = AmazonConverseConfig()
            >>> config._transform_reasoning_effort_to_reasoning_config("high")
            {'reasoningConfig': {'type': 'enabled', 'maxReasoningEffort': 'high'}}
            >>> config._transform_reasoning_effort_to_reasoning_config("low")
            {'reasoningConfig': {'type': 'enabled', 'maxReasoningEffort': 'low'}}
        """
        valid_values = ["low", "medium", "high"]
        if reasoning_effort not in valid_values:
            raise litellm.exceptions.BadRequestError(
                message=f"Invalid reasoning_effort value '{reasoning_effort}' for Nova 2 models. "
                f"Supported values: {valid_values}",
                model="amazon.nova-2-lite-v1:0",
                llm_provider="bedrock_converse",
            )

        return {
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": reasoning_effort,
            }
        }

    def _handle_reasoning_effort_parameter(
        self, model: str, reasoning_effort: str, optional_params: dict
    ) -> None:
        """
        Handle the reasoning_effort parameter based on the model type.

        Different model families handle reasoning effort differently:
        - GPT-OSS models: Keep reasoning_effort as-is (passed to additionalModelRequestFields)
        - Nova Lite 2 models: Transform to reasoningConfig structure
        - Other models (Anthropic, etc.): Convert to thinking parameter

        Args:
            model: The model identifier
            reasoning_effort: The reasoning effort value
            optional_params: Dictionary of optional parameters to update in-place

        Examples:
            >>> config = AmazonConverseConfig()
            >>> params = {}
            >>> config._handle_reasoning_effort_parameter("gpt-oss-model", "high", params)
            >>> params
            {'reasoning_effort': 'high'}

            >>> params = {}
            >>> config._handle_reasoning_effort_parameter("amazon.nova-2-lite-v1:0", "high", params)
            >>> params
            {'reasoningConfig': {'type': 'enabled', 'maxReasoningEffort': 'high'}}

            >>> params = {}
            >>> config._handle_reasoning_effort_parameter("anthropic.claude-3", "high", params)
            >>> params
            {'thinking': {'type': 'enabled', 'budget_tokens': 10000}}
        """
        if "gpt-oss" in model:
            # GPT-OSS models: keep reasoning_effort as-is
            # It will be passed through to additionalModelRequestFields
            optional_params["reasoning_effort"] = reasoning_effort
        elif self._is_nova_lite_2_model(model):
            # Nova Lite 2 models: transform to reasoningConfig
            reasoning_config = self._transform_reasoning_effort_to_reasoning_config(
                reasoning_effort
            )
            optional_params.update(reasoning_config)
        else:
            # Anthropic and other models: convert to thinking parameter
            optional_params["thinking"] = AnthropicConfig._map_reasoning_effort(
                reasoning_effort=reasoning_effort, model=model
            )

    @staticmethod
    def _clamp_thinking_budget_tokens(optional_params: dict) -> None:
        """
        Clamp thinking.budget_tokens to the Bedrock minimum (1024).

        Bedrock returns a 400 error if budget_tokens < 1024.
        """
        thinking = optional_params.get("thinking")
        if isinstance(thinking, dict):
            budget = thinking.get("budget_tokens")
            if isinstance(budget, int) and budget < BEDROCK_MIN_THINKING_BUDGET_TOKENS:
                verbose_logger.debug(
                    "Bedrock requires thinking.budget_tokens >= %d, got %d. "
                    "Clamping to minimum.",
                    BEDROCK_MIN_THINKING_BUDGET_TOKENS,
                    budget,
                )
                thinking["budget_tokens"] = BEDROCK_MIN_THINKING_BUDGET_TOKENS

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
            "service_tier",
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

        # Nova models support web_search_options (mapped to nova_grounding systemTool)
        if base_model.startswith("amazon.nova"):
            supported_params.append("web_search_options")

        if litellm.utils.supports_tool_choice(
            model=model, custom_llm_provider=self.custom_llm_provider
        ) or litellm.utils.supports_tool_choice(
            model=base_model, custom_llm_provider=self.custom_llm_provider
        ):
            # only anthropic and mistral support tool choice config. otherwise (E.g. cohere) will fail the call - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            supported_params.append("tool_choice")

        if "gpt-oss" in model:
            supported_params.append("reasoning_effort")
        elif self._is_nova_lite_2_model(model):
            # Nova Lite 2 models support reasoning_effort (transformed to reasoningConfig)
            # These models use a different reasoning structure than Anthropic's thinking parameter
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
                self._handle_reasoning_effort_parameter(
                    model=model, reasoning_effort=value, optional_params=optional_params
                )
            if param == "requestMetadata":
                if value is not None and isinstance(value, dict):
                    self._validate_request_metadata(value)  # type: ignore
                    optional_params["requestMetadata"] = value
            if param == "service_tier" and isinstance(value, str):
                # Map OpenAI service_tier (string) to Bedrock serviceTier (object)
                # OpenAI values: "auto", "default", "flex", "priority"
                # Bedrock values: "default", "flex", "priority" (no "auto")
                bedrock_tier = value
                if value == "auto":
                    bedrock_tier = "default"  # Bedrock doesn't support "auto"
                if bedrock_tier in ("default", "flex", "priority"):
                    optional_params["serviceTier"] = {"type": bedrock_tier}

            if param == "web_search_options" and isinstance(value, dict):
                # Note: we use `isinstance(value, dict)` instead of `value and isinstance(value, dict)`
                # because empty dict {} is falsy but is a valid way to enable Nova grounding
                grounding_tool = self._map_web_search_options(value, model)
                if grounding_tool is not None:
                    optional_params = self._add_tools_to_optional_params(
                        optional_params=optional_params, tools=[grounding_tool]
                    )

        # Only update thinking tokens for non-GPT-OSS models and non-Nova-Lite-2 models
        # Nova Lite 2 handles token budgeting differently through reasoningConfig
        if "gpt-oss" not in model and not self._is_nova_lite_2_model(model):
            self.update_optional_params_with_thinking_tokens(
                non_default_params=non_default_params, optional_params=optional_params
            )

        final_is_thinking_enabled = self.is_thinking_enabled(optional_params)
        if final_is_thinking_enabled and "tool_choice" in optional_params:
            tool_choice_block = optional_params["tool_choice"]
            if isinstance(tool_choice_block, dict):
                if "any" in tool_choice_block or "tool" in tool_choice_block:
                    verbose_logger.info(
                        f"{model} does not support forced tool use (tool_choice='required' or specific tool) "
                        f"when reasoning is enabled. Changing tool_choice to 'auto'."
                    )
                    optional_params["tool_choice"] = ToolChoiceValuesBlock(auto={})

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

        Also clamps thinking.budget_tokens to the Bedrock minimum (1024) to
        prevent 400 errors from the Bedrock API.
        """
        from litellm.constants import DEFAULT_MAX_TOKENS

        self._clamp_thinking_budget_tokens(optional_params)

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
        model: Optional[str] = None,
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
        model: Optional[str] = None,
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
        model: Optional[str] = None,
    ) -> Optional[Union[SystemContentBlock, ContentBlock]]:
        cache_control = message_block.get("cache_control", None)
        if cache_control is None:
            return None

        cache_point = CachePointBlock(type="default")
        if isinstance(cache_control, dict) and "ttl" in cache_control:
            ttl = cache_control["ttl"]
            if ttl in ["5m", "1h"] and model is not None:
                if is_claude_4_5_on_bedrock(model):
                    cache_point["ttl"] = ttl

        if block_type == "system":
            return SystemContentBlock(cachePoint=cache_point)
        else:
            return ContentBlock(cachePoint=cache_point)

    def _transform_system_message(
        self, messages: List[AllMessageValues], model: Optional[str] = None
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
                        message, block_type="system", model=model
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
                                m, block_type="system", model=model
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
        # Filter out exception objects before deepcopy to prevent deepcopy failures
        # Exceptions should not be stored in optional_params (this is a defensive fix)
        cleaned_params = filter_exceptions_from_params(optional_params)
        inference_params = safe_deep_copy(cleaned_params)
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

        # Filter out internal/MCP-related parameters that shouldn't be sent to the API
        # These are LiteLLM internal parameters, not API parameters
        additional_request_params = filter_internal_params(additional_request_params)

        # Filter out non-serializable objects (exceptions, callables, logging objects, etc.)
        # from additional_request_params to prevent JSON serialization errors
        # This filters: Exception objects, callable objects (functions), Logging objects, etc.
        additional_request_params = filter_exceptions_from_params(
            additional_request_params
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

        # Separate pre-formatted Bedrock tools (e.g. systemTool from web_search_options)
        # from OpenAI-format tools that need transformation via _bedrock_tools_pt
        filtered_tools = []
        pre_formatted_tools: List[ToolBlock] = []
        if original_tools:
            for tool in original_tools:
                # Already-formatted Bedrock tools (e.g. systemTool for Nova grounding)
                if "systemTool" in tool:
                    pre_formatted_tools.append(tool)
                    continue
                tool_type = tool.get("type", "")
                if tool_type in (
                    "tool_search_tool_regex_20251119",
                    "tool_search_tool_bm25_20251119",
                ):
                    # Tool search not supported in Converse API - skip it
                    continue
                filtered_tools.append(tool)

        # Only separate tools if computer use tools are actually present
        if filtered_tools and self.is_computer_use_tool_used(filtered_tools, model):
            # Separate computer use tools from regular function tools
            computer_use_tools, regular_tools = self._separate_computer_use_tools(
                filtered_tools, model
            )

            # Process regular function tools using existing logic
            bedrock_tools = _bedrock_tools_pt(regular_tools)

            # Add computer use tools and anthropic_beta if needed (only when computer use tools are present)
            if computer_use_tools:
                # Determine the correct computer-use beta header based on model
                # "computer-use-2025-11-24" for Claude Opus 4.6, Claude Opus 4.5
                # "computer-use-2025-01-24" for Claude Sonnet 4.5, Haiku 4.5, Opus 4.1, Sonnet 4, Opus 4, and Sonnet 3.7
                # "computer-use-2024-10-22" for older models
                model_lower = model.lower()
                if "opus-4.6" in model_lower or "opus_4.6" in model_lower or "opus-4-6" in model_lower or "opus_4_6" in model_lower:
                    computer_use_header = "computer-use-2025-11-24"
                elif "opus-4.5" in model_lower or "opus_4.5" in model_lower or "opus-4-5" in model_lower or "opus_4_5" in model_lower:
                    computer_use_header = "computer-use-2025-11-24"
                elif any(pattern in model_lower for pattern in [
                    "sonnet-4.5", "sonnet_4.5", "sonnet-4-5", "sonnet_4_5",
                    "haiku-4.5", "haiku_4.5", "haiku-4-5", "haiku_4_5",
                    "opus-4.1", "opus_4.1", "opus-4-1", "opus_4_1",
                    "sonnet-4", "sonnet_4",
                    "opus-4", "opus_4",
                    "sonnet-3.7", "sonnet_3.7", "sonnet-3-7", "sonnet_3_7"
                ]):
                    computer_use_header = "computer-use-2025-01-24"
                else:
                    computer_use_header = "computer-use-2024-10-22"
                
                anthropic_beta_list.append(computer_use_header)
                # Transform computer use tools to proper Bedrock format
                transformed_computer_tools = self._transform_computer_use_tools(
                    computer_use_tools
                )
                additional_request_params["tools"] = transformed_computer_tools
        else:
            # No computer use tools, process all tools as regular tools
            bedrock_tools = _bedrock_tools_pt(filtered_tools)

        # Append pre-formatted tools (systemTool etc.) after transformation
        bedrock_tools.extend(pre_formatted_tools)

        # Set anthropic_beta in additional_request_params if we have any beta features
        # ONLY apply to Anthropic/Claude models - other models (e.g., Qwen, Llama) don't support this field
        base_model = BedrockModelInfo.get_base_model(model)
        if anthropic_beta_list and base_model.startswith("anthropic"):
            additional_request_params["anthropic_beta"] = anthropic_beta_list

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

        # Drop thinking param if thinking is enabled but thinking_blocks are missing
        # This prevents the error: "Expected thinking or redacted_thinking, but found tool_use"
        #
        # IMPORTANT: Only drop thinking if NO assistant messages have thinking_blocks.
        # If any message has thinking_blocks, we must keep thinking enabled, otherwise
        # Related issues: https://github.com/BerriAI/litellm/issues/14194
        if (
            optional_params.get("thinking") is not None
            and messages is not None
            and last_assistant_with_tool_calls_has_no_thinking_blocks(messages)
            and not any_assistant_message_has_thinking_blocks(messages)
        ):
            if litellm.modify_params:
                optional_params.pop("thinking", None)
                litellm.verbose_logger.warning(
                    "Dropping 'thinking' param because the last assistant message with tool_calls "
                    "has no thinking_blocks. The model won't use extended thinking for this turn."
                )

        # Prepare and separate parameters
        (
            inference_params,
            additional_request_params,
            request_metadata,
        ) = self._prepare_request_params(optional_params, model)

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
        messages, system_content_blocks = self._transform_system_message(
            messages, model=model
        )

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
        messages, system_content_blocks = self._transform_system_message(
            messages, model=model
        )

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
            cache_creation_input_tokens = usage["cacheWriteInputTokens"]
            input_tokens += cache_creation_input_tokens

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

    def _translate_message_content(
        self, content_blocks: List[ContentBlock]
    ) -> Tuple[
        str,
        List[ChatCompletionToolCallChunk],
        Optional[List[BedrockConverseReasoningContentBlock]],
        Optional[List[CitationsContentBlock]],
    ]:
        """
        Translate the message content to a string and a list of tool calls, reasoning content blocks, and citations.

        Returns:
            content_str: str
            tools: List[ChatCompletionToolCallChunk]
            reasoningContentBlocks: Optional[List[BedrockConverseReasoningContentBlock]]
            citationsContentBlocks: Optional[List[CitationsContentBlock]] - Citations from Nova grounding
        """
        content_str = ""
        tools: List[ChatCompletionToolCallChunk] = []
        reasoningContentBlocks: Optional[
            List[BedrockConverseReasoningContentBlock]
        ] = None
        citationsContentBlocks: Optional[List[CitationsContentBlock]] = None
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
            # Handle Nova grounding citations content
            if "citationsContent" in content:
                if citationsContentBlocks is None:
                    citationsContentBlocks = []
                citationsContentBlocks.append(content["citationsContent"])

        return content_str, tools, reasoningContentBlocks, citationsContentBlocks

    def _transform_response(  # noqa: PLR0915
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
        citationsContentBlocks: Optional[List[CitationsContentBlock]] = None

        if message is not None:
            (
                content_str,
                tools,
                reasoningContentBlocks,
                citationsContentBlocks,
            ) = self._translate_message_content(message["content"])

        # Initialize provider_specific_fields if we have any special content blocks
        provider_specific_fields: dict = {}
        if reasoningContentBlocks is not None:
            provider_specific_fields["reasoningContentBlocks"] = reasoningContentBlocks
        if citationsContentBlocks is not None:
            provider_specific_fields["citationsContent"] = citationsContentBlocks

        if provider_specific_fields:
            chat_completion_message[
                "provider_specific_fields"
            ] = provider_specific_fields

        if reasoningContentBlocks is not None:
            chat_completion_message[
                "reasoning_content"
            ] = self._transform_reasoning_content(reasoningContentBlocks)
            chat_completion_message[
                "thinking_blocks"
            ] = self._transform_thinking_blocks(reasoningContentBlocks)
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

        # Add service_tier if present in Bedrock response
        # Map Bedrock serviceTier (object) to OpenAI service_tier (string)
        if "serviceTier" in completion_response:
            service_tier_block = completion_response["serviceTier"]
            if isinstance(service_tier_block, dict) and "type" in service_tier_block:
                setattr(model_response, "service_tier", service_tier_block["type"])

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
                # AI21 models do not support streaming
                ###################################################################
                if "ai21" in model:
                    return True
        return False
