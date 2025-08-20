"""
Transformation logic from OpenAI /v1/chat/completion format to Mistral's /chat/completion format.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.mistral.ai/api/
"""

from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, cast, overload

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
    strip_none_values_from_message,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.mistral import MistralToolCallMessage
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import convert_to_model_response_object


class MistralConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.mistral.ai/api/

    The class `MistralConfig` provides configuration for the Mistral's Chat API interface. Below are the parameters:

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2. API Default - 0.7.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling. API Default - 1.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion. API Default - null.

    - `tools` (list or null): A list of available tools for the model. Use this to specify functions for which the model can generate JSON inputs.

    - `tool_choice` (string - 'auto'/'any'/'none' or null): Specifies if/how functions are called. If set to none the model won't call a function and will generate a message instead. If set to auto the model can choose to either generate a message or call a function. If set to any the model is forced to call a function. Default - 'auto'.

    - `stop` (string or array of strings): Stop generation if this token is detected. Or if one of these tokens is detected when providing an array

    - `random_seed` (integer or null): The seed to use for random sampling. If set, different calls will generate deterministic results.

    - `safe_prompt` (boolean): Whether to inject a safety prompt before all conversations. API Default - 'false'.

    - `response_format` (object or null): An object specifying the format that the model must output. Setting to { "type": "json_object" } enables JSON mode, which guarantees the message the model generates is in JSON. When using JSON mode you MUST also instruct the model to produce JSON yourself with a system or a user message.
    """

    temperature: Optional[int] = None
    top_p: Optional[int] = None
    max_tokens: Optional[int] = None
    tools: Optional[list] = None
    tool_choice: Optional[Literal["auto", "any", "none"]] = None
    random_seed: Optional[int] = None
    safe_prompt: Optional[bool] = None
    response_format: Optional[dict] = None
    stop: Optional[Union[str, list]] = None

    def __init__(
        self,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Literal["auto", "any", "none"]] = None,
        random_seed: Optional[int] = None,
        safe_prompt: Optional[bool] = None,
        response_format: Optional[dict] = None,
        stop: Optional[Union[str, list]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = [
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
        ]

        # Add reasoning support for magistral models
        if "magistral" in model.lower():
            supported_params.extend(["thinking", "reasoning_effort"])

        return supported_params

    def _map_tool_choice(self, tool_choice: str) -> str:
        if tool_choice == "auto" or tool_choice == "none":
            return tool_choice
        elif tool_choice == "required":
            return "any"
        else:  # openai 'tool_choice' object param not supported by Mistral API
            return "any"

    @staticmethod
    def _get_mistral_reasoning_system_prompt() -> str:
        """
        Returns the system prompt for Mistral reasoning models.
        Based on Mistral's documentation: https://huggingface.co/mistralai/Magistral-Small-2506

        Mistral recommends the following system prompt for reasoning:
        """
        return """
        <s>[SYSTEM_PROMPT]system_prompt
        A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.

        Your thinking process must follow the template below:
        <think>
        Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
        </think>

        Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.

        Problem:

        [/SYSTEM_PROMPT][INST]user_message[/INST]<think>
        reasoning_traces
        </think>
        assistant_response</s>[INST]user_message[/INST]
        """

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
            if param == "max_completion_tokens":  # max_completion_tokens should take priority
                optional_params["max_tokens"] = value
            if param == "tools":
                # Clean tools to remove problematic schema fields for Mistral API
                optional_params["tools"] = self._clean_tool_schema_for_mistral(value)
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "tool_choice" and isinstance(value, str):
                optional_params["tool_choice"] = self._map_tool_choice(tool_choice=value)
            if param == "seed":
                optional_params["extra_body"] = {"random_seed": value}
            if param == "response_format":
                optional_params["response_format"] = value
            if param == "reasoning_effort" and "magistral" in model.lower():
                # Flag that we need to add reasoning system prompt
                optional_params["_add_reasoning_prompt"] = True
            if param == "thinking" and "magistral" in model.lower():
                # Flag that we need to add reasoning system prompt
                optional_params["_add_reasoning_prompt"] = True
            if param == "parallel_tool_calls":
                optional_params["parallel_tool_calls"] = value
        return optional_params

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        # mistral is openai compatible, we just need to set this to custom_openai and have the api_base be https://api.mistral.ai
        api_base = (
            api_base
            or get_secret_str("MISTRAL_AZURE_API_BASE")  # for Azure AI Mistral
            or "https://api.mistral.ai/v1"
        )  # type: ignore

        # if api_base does not end with /v1 we add it
        if api_base is not None and not api_base.endswith("/v1"):  # Mistral always needs a /v1 at the end
            api_base = api_base + "/v1"
        dynamic_api_key = (
            api_key
            or get_secret_str("MISTRAL_AZURE_API_KEY")  # for Azure AI Mistral
            or get_secret_str("MISTRAL_API_KEY")
        )
        return api_base, dynamic_api_key

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
        - handles scenario where content is list and not string
        - content list is just text, and no images
        - if image passed in, then just return as is (user-intended)
        - if `name` is passed, then drop it for mistral API: https://github.com/BerriAI/litellm/issues/6696

        Motivation: mistral api doesn't support content as a list
        """
        ## 1. If 'image_url' in content, then return as is
        for m in messages:
            _content_block = m.get("content")
            if _content_block and isinstance(_content_block, list):
                for c in _content_block:
                    if c.get("type") == "image_url":
                        if is_async:
                            return super()._transform_messages(messages, model, True)
                        else:
                            return super()._transform_messages(messages, model, False)

        ## 2. If content is list, then convert to string
        messages = handle_messages_with_content_list_to_str_conversion(messages)

        ## 3. Handle name in message
        new_messages: List[AllMessageValues] = []
        for m in messages:
            m = MistralConfig._handle_name_in_message(m)
            m = MistralConfig._handle_tool_call_message(m)
            m = strip_none_values_from_message(m)  # prevents 'extra_forbidden' error
            new_messages.append(m)

        if is_async:
            return super()._transform_messages(new_messages, model, True)
        else:
            return super()._transform_messages(new_messages, model, False)

    def _add_reasoning_system_prompt_if_needed(
        self, messages: List[AllMessageValues], optional_params: dict
    ) -> List[AllMessageValues]:
        """
        Add reasoning system prompt for Mistral magistral models when reasoning_effort is specified.
        """
        if not optional_params.get("_add_reasoning_prompt", False):
            return messages

        # Check if there's already a system message
        has_system_message = any(msg.get("role") == "system" for msg in messages)

        if has_system_message:
            # Prepend reasoning instructions to existing system message
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    existing_content = msg.get("content", "")
                    reasoning_prompt = self._get_mistral_reasoning_system_prompt()

                    # Handle both string and list content, preserving original format
                    if isinstance(existing_content, str):
                        # String content - prepend reasoning prompt
                        new_content: Union[str, list] = f"{reasoning_prompt}\n\n{existing_content}"
                    elif isinstance(existing_content, list):
                        # List content - prepend reasoning prompt as text block
                        new_content = [{"type": "text", "text": reasoning_prompt + "\n\n"}] + existing_content
                    else:
                        # Fallback for any other type - convert to string
                        new_content = f"{reasoning_prompt}\n\n{str(existing_content)}"

                    messages[i] = cast(AllMessageValues, {**msg, "content": new_content})
                    break
        else:
            # Add new system message with reasoning instructions
            reasoning_message: AllMessageValues = cast(
                AllMessageValues, {"role": "system", "content": self._get_mistral_reasoning_system_prompt()}
            )
            messages = [reasoning_message] + messages

        # Remove the internal flag
        optional_params.pop("_add_reasoning_prompt", None)
        return messages

    @classmethod
    def _clean_tool_schema_for_mistral(cls, tools: list) -> list:
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
        from litellm.utils import _remove_json_schema_refs

        cleaned_tools = copy.deepcopy(tools)
        
        # Apply all cleaning functions with max_depth protection
        cleaned_tools = _remove_json_schema_refs(cleaned_tools, max_depth=DEFAULT_MAX_RECURSE_DEPTH)
        
        return cleaned_tools

    @classmethod
    def _handle_name_in_message(cls, message: AllMessageValues) -> AllMessageValues:
        """
        Mistral API only supports `name` in tool messages

        If role == tool, then we keep `name` if it's not an empty string
        Otherwise, we drop `name`
        """
        _name = message.get("name")  # type: ignore

        if _name is not None:
            # Remove name if not a tool message
            if message["role"] != "tool":
                message.pop("name", None)  # type: ignore
            # For tool messages, remove name if it's an empty string
            elif isinstance(_name, str) and len(_name.strip()) == 0:
                message.pop("name", None)  # type: ignore

        return message

    @classmethod
    def _handle_tool_call_message(cls, message: AllMessageValues) -> AllMessageValues:
        """
        Mistral API only supports tool_calls in Messages in `MistralToolCallMessage` spec
        """
        _tool_calls = message.get("tool_calls")
        mistral_tool_calls: List[MistralToolCallMessage] = []
        if _tool_calls is not None and isinstance(_tool_calls, list):
            for _tool in _tool_calls:
                _tool_call_message = MistralToolCallMessage(
                    id=_tool.get("id"),
                    type="function",
                    function=_tool.get("function"),  # type: ignore
                )
                mistral_tool_calls.append(_tool_call_message)
            message["tool_calls"] = mistral_tool_calls  # type: ignore
        return message

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

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the overall request to be sent to the API.
        For magistral models, adds reasoning system prompt when reasoning_effort is specified.

        Returns:
            dict: The transformed request. Sent as the body of the API call.
        """
        # Add reasoning system prompt if needed (for magistral models)
        if "magistral" in model.lower() and optional_params.get("_add_reasoning_prompt", False):
            messages = self._add_reasoning_system_prompt_if_needed(messages, optional_params)

        # Call parent transform_request which handles _transform_messages
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
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
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the raw response from Mistral API.
        Handles Mistral-specific behavior like converting empty string content to None.
        """
        logging_obj.post_call(original_response=raw_response.text)
        logging_obj.model_call_details["response_headers"] = raw_response.headers

        # Handle Mistral-specific empty string content conversion to None
        response_data = raw_response.json()
        response_data = self._handle_empty_content_response(response_data)

        final_response_obj = cast(
            ModelResponse,
            convert_to_model_response_object(
                response_object=response_data,
                model_response_object=model_response,
                hidden_params={"headers": raw_response.headers},
                _response_headers=dict(raw_response.headers),
            ),
        )

        return final_response_obj
