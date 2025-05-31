# What is this?
## httpx client for vertex ai calls
## Initial implementation - covers gemini + image gen calls
import json
import time
import uuid
from copy import deepcopy
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx  # type: ignore

import litellm
import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm import verbose_logger
from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.llms.anthropic import AnthropicThinkingParam
from litellm.types.llms.gemini import BidiGenerateContentServerMessage
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionResponseMessage,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUsageBlock,
    OpenAIChatCompletionFinishReason,
)
from litellm.types.llms.vertex_ai import (
    VERTEX_CREDENTIALS_TYPES,
    Candidates,
    ContentType,
    FunctionCallingConfig,
    FunctionDeclaration,
    GeminiThinkingConfig,
    GenerateContentResponseBody,
    HttpxPartType,
    LogprobsResult,
    ToolConfig,
    Tools,
    UsageMetadata,
)
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    CompletionTokensDetailsWrapper,
    GenericStreamingChunk,
    PromptTokensDetailsWrapper,
    TopLogprob,
    Usage,
)
from litellm.utils import CustomStreamWrapper, ModelResponse, is_base64_encoded, supports_reasoning

from ....utils import _remove_additional_properties, _remove_strict_from_schema
from ..common_utils import VertexAIError, _build_vertex_schema
from ..vertex_llm_base import VertexBase
from .transformation import (
    _gemini_convert_messages_with_history,
    async_transform_request_body,
    sync_transform_request_body,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class VertexAIBaseConfig:
    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ]

    def get_us_regions(self) -> List[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "us-central1",
            "us-east1",
            "us-east4",
            "us-east5",
            "us-south1",
            "us-west1",
            "us-west4",
            "us-west5",
        ]


class VertexGeminiConfig(VertexAIBaseConfig, BaseConfig):
    """
    Reference: https://cloud.google.com/vertex-ai/docs/generative-ai/chat/test-chat-prompts
    Reference: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

    The class `VertexAIConfig` provides configuration for the VertexAI's API interface. Below are the parameters:

    - `temperature` (float): This controls the degree of randomness in token selection.

    - `max_output_tokens` (integer): This sets the limitation for the maximum amount of token in the text output. In this case, the default value is 256.

    - `top_p` (float): The tokens are selected from the most probable to the least probable until the sum of their probabilities equals the `top_p` value. Default is 0.95.

    - `top_k` (integer): The value of `top_k` determines how many of the most probable tokens are considered in the selection. For example, a `top_k` of 1 means the selected token is the most probable among all tokens. The default value is 40.

    - `response_mime_type` (str): The MIME type of the response. The default value is 'text/plain'.

    - `candidate_count` (int): Number of generated responses to return.

    - `stop_sequences` (List[str]): The set of character sequences (up to 5) that will stop output generation. If specified, the API will stop at the first appearance of a stop sequence. The stop sequence will not be included as part of the response.

    - `frequency_penalty` (float): This parameter is used to penalize the model from repeating the same output. The default value is 0.0.

    - `presence_penalty` (float): This parameter is used to penalize the model from generating the same output as the input. The default value is 0.0.

    - `seed` (int): The seed value is used to help generate the same output for the same input. The default value is None.

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    response_mime_type: Optional[str] = None
    candidate_count: Optional[int] = None
    stop_sequences: Optional[list] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
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
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "tools",
            "functions",
            "tool_choice",
            "response_format",
            "n",
            "stop",
            "frequency_penalty",
            "presence_penalty",
            "extra_headers",
            "seed",
            "logprobs",
            "top_logprobs",
            "modalities",
            "parallel_tool_calls",
            "web_search_options",
        ]
        if supports_reasoning(model):
            supported_params.append("reasoning_effort")
            supported_params.append("thinking")
        return supported_params

    def map_tool_choice_values(
        self, model: str, tool_choice: Union[str, dict]
    ) -> Optional[ToolConfig]:
        if tool_choice == "none":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="NONE"))
        elif tool_choice == "required":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="ANY"))
        elif tool_choice == "auto":
            return ToolConfig(functionCallingConfig=FunctionCallingConfig(mode="AUTO"))
        elif isinstance(tool_choice, dict):
            # only supported for anthropic + mistral models - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
            name = tool_choice.get("function", {}).get("name", "")
            return ToolConfig(
                functionCallingConfig=FunctionCallingConfig(
                    mode="ANY", allowed_function_names=[name]
                )
            )
        else:
            raise litellm.utils.UnsupportedParamsError(
                message="VertexAI doesn't support tool_choice={}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.".format(
                    tool_choice
                ),
                status_code=400,
            )

    def _map_web_search_options(self, value: dict) -> Tools:
        """
        Base Case: empty dict

        Google doesn't support user_location or search_context_size params
        """
        return Tools(googleSearch={})

    def _map_function(self, value: List[dict]) -> List[Tools]:
        gtool_func_declarations = []
        googleSearch: Optional[dict] = None
        googleSearchRetrieval: Optional[dict] = None
        enterpriseWebSearch: Optional[dict] = None
        code_execution: Optional[dict] = None
        # remove 'additionalProperties' from tools
        value = _remove_additional_properties(value)
        # remove 'strict' from tools
        value = _remove_strict_from_schema(value)

        def get_tool_value(tool: dict, tool_name: str) -> Optional[dict]:
            """
            Helper function to get tool value handling both camelCase and underscore_case variants

            Args:
                tool (dict): The tool dictionary
                tool_name (str): The base tool name (e.g. "codeExecution")

            Returns:
                Optional[dict]: The tool value if found, None otherwise
            """
            # Convert camelCase to underscore_case
            underscore_name = "".join(
                ["_" + c.lower() if c.isupper() else c for c in tool_name]
            ).lstrip("_")
            # Try both camelCase and underscore_case variants

            if tool.get(tool_name) is not None:
                return tool.get(tool_name)
            elif tool.get(underscore_name) is not None:
                return tool.get(underscore_name)
            else:
                return None

        for tool in value:
            openai_function_object: Optional[
                ChatCompletionToolParamFunctionChunk
            ] = None
            if "function" in tool:  # tools list
                _openai_function_object = ChatCompletionToolParamFunctionChunk(  # type: ignore
                    **tool["function"]
                )

                if (
                    "parameters" in _openai_function_object
                    and _openai_function_object["parameters"] is not None
                ):  # OPENAI accepts JSON Schema, Google accepts OpenAPI schema.
                    _openai_function_object["parameters"] = _build_vertex_schema(
                        _openai_function_object["parameters"]
                    )

                openai_function_object = _openai_function_object

            elif "name" in tool:  # functions list
                openai_function_object = ChatCompletionToolParamFunctionChunk(**tool)  # type: ignore

            tool_name = list(tool.keys())[0] if len(tool.keys()) == 1 else None
            if tool_name and (
                tool_name == "codeExecution" or tool_name == "code_execution"
            ):  # code_execution maintained for backwards compatibility
                code_execution = get_tool_value(tool, "codeExecution")
            elif tool_name and tool_name == "googleSearch":
                googleSearch = get_tool_value(tool, "googleSearch")
            elif tool_name and tool_name == "googleSearchRetrieval":
                googleSearchRetrieval = get_tool_value(tool, "googleSearchRetrieval")
            elif tool_name and tool_name == "enterpriseWebSearch":
                enterpriseWebSearch = get_tool_value(tool, "enterpriseWebSearch")
            elif openai_function_object is not None:
                gtool_func_declaration = FunctionDeclaration(
                    name=openai_function_object["name"],
                )
                _description = openai_function_object.get("description", None)
                _parameters = openai_function_object.get("parameters", None)
                if _description is not None:
                    gtool_func_declaration["description"] = _description
                if _parameters is not None:
                    gtool_func_declaration["parameters"] = _parameters
                gtool_func_declarations.append(gtool_func_declaration)
            else:
                # assume it's a provider-specific param
                verbose_logger.warning(
                    "Invalid tool={}. Use `litellm.set_verbose` or `litellm --detailed_debug` to see raw request."
                )

        _tools = Tools(
            function_declarations=gtool_func_declarations,
        )
        if googleSearch is not None:
            _tools["googleSearch"] = googleSearch
        if googleSearchRetrieval is not None:
            _tools["googleSearchRetrieval"] = googleSearchRetrieval
        if enterpriseWebSearch is not None:
            _tools["enterpriseWebSearch"] = enterpriseWebSearch
        if code_execution is not None:
            _tools["code_execution"] = code_execution
        return [_tools]

    def _map_response_schema(self, value: dict) -> dict:
        old_schema = deepcopy(value)
        if isinstance(old_schema, list):
            for item in old_schema:
                if isinstance(item, dict):
                    item = _build_vertex_schema(
                        parameters=item, add_property_ordering=True
                    )

        elif isinstance(old_schema, dict):
            old_schema = _build_vertex_schema(
                parameters=old_schema, add_property_ordering=True
            )
        return old_schema

    def apply_response_schema_transformation(self, value: dict, optional_params: dict):
        new_value = deepcopy(value)
        # remove 'additionalProperties' from json schema
        new_value = _remove_additional_properties(new_value)
        # remove 'strict' from json schema
        new_value = _remove_strict_from_schema(new_value)
        if new_value["type"] == "json_object":
            optional_params["response_mime_type"] = "application/json"
        elif new_value["type"] == "text":
            optional_params["response_mime_type"] = "text/plain"
        if "response_schema" in new_value:
            optional_params["response_mime_type"] = "application/json"
            optional_params["response_schema"] = new_value["response_schema"]
        elif new_value["type"] == "json_schema":  # type: ignore
            if "json_schema" in new_value and "schema" in new_value["json_schema"]:  # type: ignore
                optional_params["response_mime_type"] = "application/json"
                optional_params["response_schema"] = new_value["json_schema"]["schema"]  # type: ignore

        if "response_schema" in optional_params and isinstance(
            optional_params["response_schema"], dict
        ):
            optional_params["response_schema"] = self._map_response_schema(
                value=optional_params["response_schema"]
            )

    @staticmethod
    def _map_reasoning_effort_to_thinking_budget(
        reasoning_effort: str,
    ) -> GeminiThinkingConfig:
        if reasoning_effort == "low":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET,
                "includeThoughts": True,
            }
        elif reasoning_effort == "medium":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
                "includeThoughts": True,
            }
        elif reasoning_effort == "high":
            return {
                "thinkingBudget": DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
                "includeThoughts": True,
            }
        else:
            raise ValueError(f"Invalid reasoning effort: {reasoning_effort}")

    @staticmethod
    def _is_thinking_budget_zero(thinking_budget: Optional[int]) -> bool:
        return thinking_budget is not None and thinking_budget == 0

    @staticmethod
    def _map_thinking_param(
        thinking_param: AnthropicThinkingParam,
    ) -> GeminiThinkingConfig:
        thinking_enabled = thinking_param.get("type") == "enabled"
        thinking_budget = thinking_param.get("budget_tokens")

        params: GeminiThinkingConfig = {}
        if thinking_enabled and not VertexGeminiConfig._is_thinking_budget_zero(
            thinking_budget
        ):
            params["includeThoughts"] = True
        if thinking_budget is not None and isinstance(thinking_budget, int):
            params["thinkingBudget"] = thinking_budget

        return params

    def map_response_modalities(self, value: list) -> list:
        response_modalities = []
        for modality in value:
            if modality == "text":
                response_modalities.append("TEXT")
            elif modality == "image":
                response_modalities.append("IMAGE")
            elif modality == "audio":
                response_modalities.append("AUDIO")
            else:
                response_modalities.append("MODALITY_UNSPECIFIED")
        return response_modalities

    def validate_parallel_tool_calls(self, value: bool, non_default_params: dict):
        tools = non_default_params.get("tools", non_default_params.get("functions"))
        num_function_declarations = len(tools) if isinstance(tools, list) else 0
        if num_function_declarations > 1:
            raise litellm.utils.UnsupportedParamsError(
                message=(
                    "`parallel_tool_calls=False` is not supported by Gemini when multiple tools are "
                    "provided. Specify a single tool, or set "
                    "`parallel_tool_calls=True`. If you want to drop this param, set `litellm.drop_params = True` or pass in `(.., drop_params=True)` in the requst - https://docs.litellm.ai/docs/completion/drop_params"
                ),
                status_code=400,
            )

    def map_openai_params(
        self,
        non_default_params: Dict,
        optional_params: Dict,
        model: str,
        drop_params: bool,
    ) -> Dict:
        for param, value in non_default_params.items():
            if param == "temperature":
                optional_params["temperature"] = value
            elif param == "top_p":
                optional_params["top_p"] = value
            elif (
                param == "stream" and value is True
            ):  # sending stream = False, can cause it to get passed unchecked and raise issues
                optional_params["stream"] = value
            elif param == "n":
                optional_params["candidate_count"] = value
            elif param == "stop":
                if isinstance(value, str):
                    optional_params["stop_sequences"] = [value]
                elif isinstance(value, list):
                    optional_params["stop_sequences"] = value
            elif param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_output_tokens"] = value
            elif param == "response_format" and isinstance(value, dict):  # type: ignore
                self.apply_response_schema_transformation(
                    value=value, optional_params=optional_params
                )
            elif param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            elif param == "presence_penalty":
                optional_params["presence_penalty"] = value
            elif param == "logprobs":
                optional_params["responseLogprobs"] = value
            elif param == "top_logprobs":
                optional_params["logprobs"] = value
            elif (
                (param == "tools" or param == "functions")
                and isinstance(value, list)
                and value
            ):
                optional_params = self._add_tools_to_optional_params(
                    optional_params, self._map_function(value=value)
                )
            elif param == "tool_choice" and (
                isinstance(value, str) or isinstance(value, dict)
            ):
                _tool_choice_value = self.map_tool_choice_values(
                    model=model, tool_choice=value  # type: ignore
                )
                if _tool_choice_value is not None:
                    optional_params["tool_choice"] = _tool_choice_value
            elif param == "parallel_tool_calls":
                if value is False and not (
                    drop_params or litellm.drop_params
                ):  # if drop params is True, then we should just ignore this
                    self.validate_parallel_tool_calls(value, non_default_params)
                else:
                    optional_params["parallel_tool_calls"] = value
            elif param == "seed":
                optional_params["seed"] = value
            elif param == "reasoning_effort" and isinstance(value, str):
                optional_params[
                    "thinkingConfig"
                ] = VertexGeminiConfig._map_reasoning_effort_to_thinking_budget(value)
            elif param == "thinking":
                optional_params[
                    "thinkingConfig"
                ] = VertexGeminiConfig._map_thinking_param(
                    cast(AnthropicThinkingParam, value)
                )
            elif param == "modalities" and isinstance(value, list):
                response_modalities = self.map_response_modalities(value)
                optional_params["responseModalities"] = response_modalities
            elif param == "web_search_options" and value and isinstance(value, dict):
                _tools = self._map_web_search_options(value)
                optional_params = self._add_tools_to_optional_params(
                    optional_params, [_tools]
                )
        if litellm.vertex_ai_safety_settings is not None:
            optional_params["safety_settings"] = litellm.vertex_ai_safety_settings
        return optional_params

    def get_mapped_special_auth_params(self) -> dict:
        """
        Common auth params across bedrock/vertex_ai/azure/watsonx
        """
        return {"project": "vertex_project", "region_name": "vertex_location"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()

        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions
        """
        return [
            "europe-central2",
            "europe-north1",
            "europe-southwest1",
            "europe-west1",
            "europe-west2",
            "europe-west3",
            "europe-west4",
            "europe-west6",
            "europe-west8",
            "europe-west9",
        ]

    @staticmethod
    def get_model_for_vertex_ai_url(model: str) -> str:
        """
        Returns the model name to use in the request to Vertex AI

        Handles 2 cases:
        1. User passed `model="vertex_ai/gemini/ft-uuid"`, we need to return `ft-uuid` for the request to Vertex AI
        2. User passed `model="vertex_ai/gemini-2.0-flash-001"`, we need to return `gemini-2.0-flash-001` for the request to Vertex AI

        Args:
            model (str): The model name to use in the request to Vertex AI

        Returns:
            str: The model name to use in the request to Vertex AI
        """
        if VertexGeminiConfig._is_model_gemini_spec_model(model):
            return VertexGeminiConfig._get_model_name_from_gemini_spec_model(model)
        return model

    @staticmethod
    def _is_model_gemini_spec_model(model: Optional[str]) -> bool:
        """
        Returns true if user is trying to call custom model in `/gemini` request/response format
        """
        if model is None:
            return False
        if "gemini/" in model:
            return True
        return False

    @staticmethod
    def _get_model_name_from_gemini_spec_model(model: str) -> str:
        """
        Returns the model name if model="vertex_ai/gemini/<unique_id>"

        Example:
        - model = "gemini/1234567890"
        - returns "1234567890"
        """
        if "gemini/" in model:
            return model.split("/")[-1]
        return model

    def get_flagged_finish_reasons(self) -> Dict[str, str]:
        """
        Return Dictionary of finish reasons which indicate response was flagged

        and what it means
        """
        return {
            "SAFETY": "The token generation was stopped as the response was flagged for safety reasons. NOTE: When streaming the Candidate.content will be empty if content filters blocked the output.",
            "RECITATION": "The token generation was stopped as the response was flagged for unauthorized citations.",
            "BLOCKLIST": "The token generation was stopped as the response was flagged for the terms which are included from the terminology blocklist.",
            "PROHIBITED_CONTENT": "The token generation was stopped as the response was flagged for the prohibited contents.",
            "SPII": "The token generation was stopped as the response was flagged for Sensitive Personally Identifiable Information (SPII) contents.",
            "IMAGE_SAFETY": "The token generation was stopped as the response was flagged for image safety reasons.",
        }

    def get_finish_reason_mapping(self) -> Dict[str, OpenAIChatCompletionFinishReason]:
        """
        Return Dictionary of finish reasons which indicate response was flagged

        and what it means
        """
        return {
            "FINISH_REASON_UNSPECIFIED": "stop",  # openai doesn't have a way of representing this
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "LANGUAGE": "content_filter",
            "OTHER": "content_filter",
            "BLOCKLIST": "content_filter",
            "PROHIBITED_CONTENT": "content_filter",
            "SPII": "content_filter",
            "MALFORMED_FUNCTION_CALL": "stop",  # openai doesn't have a way of representing this
            "IMAGE_SAFETY": "content_filter",
        }

    def translate_exception_str(self, exception_string: str):
        if (
            "GenerateContentRequest.tools[0].function_declarations[0].parameters.properties: should be non-empty for OBJECT type"
            in exception_string
        ):
            return "'properties' field in tools[0]['function']['parameters'] cannot be empty if 'type' == 'object'. Received error from provider - {}".format(
                exception_string
            )
        return exception_string

    def get_assistant_content_message(
        self, parts: List[HttpxPartType]
    ) -> Tuple[Optional[str], Optional[str]]:
        content_str: Optional[str] = None
        reasoning_content_str: Optional[str] = None

        for part in parts:
            _content_str = ""
            if "text" in part:
                text_content = part["text"]
                # Check if text content is audio data URI - if so, exclude from text content
                if text_content.startswith("data:audio") and ";base64," in text_content:
                    try:
                        if is_base64_encoded(text_content):
                            media_type, _ = text_content.split("data:")[1].split(";base64,")
                            if media_type.startswith("audio/"):
                                continue
                    except (ValueError, IndexError):
                        # If parsing fails, treat as regular text
                        pass
                _content_str += text_content
            elif "inlineData" in part:
                mime_type = part["inlineData"]["mimeType"]
                data = part["inlineData"]["data"]
                # Check if inline data is audio - if so, exclude from text content
                if mime_type.startswith("audio/"):
                    continue
                _content_str += "data:{};base64,{}".format(mime_type, data)

            if len(_content_str) > 0:
                if part.get("thought") is True:
                    if reasoning_content_str is None:
                        reasoning_content_str = ""
                    reasoning_content_str += _content_str
                else:
                    if content_str is None:
                        content_str = ""
                    content_str += _content_str

        return content_str, reasoning_content_str

    def _extract_audio_response_from_parts(
        self, parts: List[HttpxPartType]
    ) -> Optional[ChatCompletionAudioResponse]:
        """Extract audio response from parts if present"""
        for part in parts:
            if "text" in part:
                text_content = part["text"]
                # Check if text content contains audio data URI
                if text_content.startswith("data:audio") and ";base64," in text_content:
                    try:
                        if is_base64_encoded(text_content):
                            media_type, audio_data = text_content.split("data:")[1].split(";base64,")

                            if media_type.startswith("audio/"):
                                expires_at = int(time.time()) + (24 * 60 * 60)
                                transcript = ""  # Gemini doesn't provide transcript

                                return ChatCompletionAudioResponse(
                                    data=audio_data,
                                    expires_at=expires_at,
                                    transcript=transcript
                                )
                    except (ValueError, IndexError):
                        pass

            elif "inlineData" in part:
                mime_type = part["inlineData"]["mimeType"]
                data = part["inlineData"]["data"]

                if mime_type.startswith("audio/"):
                    expires_at = int(time.time()) + (24 * 60 * 60)
                    transcript = ""  # Gemini doesn't provide transcript

                    return ChatCompletionAudioResponse(
                        data=data,
                        expires_at=expires_at,
                        transcript=transcript
                    )

        return None

    def _transform_parts(
        self,
        parts: List[HttpxPartType],
        index: int,
        is_function_call: Optional[bool],
    ) -> Tuple[
        Optional[ChatCompletionToolCallFunctionChunk],
        Optional[List[ChatCompletionToolCallChunk]],
    ]:
        function: Optional[ChatCompletionToolCallFunctionChunk] = None
        _tools: List[ChatCompletionToolCallChunk] = []
        for part in parts:
            if "functionCall" in part:
                _function_chunk = ChatCompletionToolCallFunctionChunk(
                    name=part["functionCall"]["name"],
                    arguments=json.dumps(part["functionCall"]["args"]),
                )
                if is_function_call is True:
                    function = _function_chunk
                else:
                    _tool_response_chunk = ChatCompletionToolCallChunk(
                        id=f"call_{str(uuid.uuid4())}",
                        type="function",
                        function=_function_chunk,
                        index=index,
                    )
                    _tools.append(_tool_response_chunk)
        if len(_tools) == 0:
            tools: Optional[List[ChatCompletionToolCallChunk]] = None
        else:
            tools = _tools
        return function, tools

    def _transform_logprobs(
        self, logprobs_result: Optional[LogprobsResult]
    ) -> Optional[ChoiceLogprobs]:
        if logprobs_result is None:
            return None
        if "chosenCandidates" not in logprobs_result:
            return None
        logprobs_list: List[ChatCompletionTokenLogprob] = []
        for index, candidate in enumerate(logprobs_result["chosenCandidates"]):
            top_logprobs: List[TopLogprob] = []
            if "topCandidates" in logprobs_result and index < len(
                logprobs_result["topCandidates"]
            ):
                top_candidates_for_index = logprobs_result["topCandidates"][index][
                    "candidates"
                ]

                for options in top_candidates_for_index:
                    top_logprobs.append(
                        TopLogprob(
                            token=options["token"], logprob=options["logProbability"]
                        )
                    )
            logprobs_list.append(
                ChatCompletionTokenLogprob(
                    token=candidate["token"],
                    logprob=candidate["logProbability"],
                    top_logprobs=top_logprobs,
                )
            )
        return ChoiceLogprobs(content=logprobs_list)

    def _handle_blocked_response(
        self,
        model_response: ModelResponse,
        completion_response: GenerateContentResponseBody,
    ) -> ModelResponse:
        # If set, the prompt was blocked and no candidates are returned. Rephrase your prompt
        model_response.choices[0].finish_reason = "content_filter"

        chat_completion_message: ChatCompletionResponseMessage = {
            "role": "assistant",
            "content": None,
        }

        choice = litellm.Choices(
            finish_reason="content_filter",
            index=0,
            message=chat_completion_message,  # type: ignore
            logprobs=None,
            enhancements=None,
        )

        model_response.choices = [choice]

        ## GET USAGE ##
        usage = Usage(
            prompt_tokens=completion_response["usageMetadata"].get(
                "promptTokenCount", 0
            ),
            completion_tokens=completion_response["usageMetadata"].get(
                "candidatesTokenCount", 0
            ),
            total_tokens=completion_response["usageMetadata"].get("totalTokenCount", 0),
        )

        setattr(model_response, "usage", usage)

        return model_response

    def _handle_content_policy_violation(
        self,
        model_response: ModelResponse,
        completion_response: GenerateContentResponseBody,
    ) -> ModelResponse:
        ## CONTENT POLICY VIOLATION ERROR
        model_response.choices[0].finish_reason = "content_filter"

        _chat_completion_message = {
            "role": "assistant",
            "content": None,
        }

        choice = litellm.Choices(
            finish_reason="content_filter",
            index=0,
            message=_chat_completion_message,
            logprobs=None,
            enhancements=None,
        )

        model_response.choices = [choice]

        ## GET USAGE ##
        usage = Usage(
            prompt_tokens=completion_response["usageMetadata"].get(
                "promptTokenCount", 0
            ),
            completion_tokens=completion_response["usageMetadata"].get(
                "candidatesTokenCount", 0
            ),
            total_tokens=completion_response["usageMetadata"].get("totalTokenCount", 0),
        )

        setattr(model_response, "usage", usage)

        return model_response

    def is_candidate_token_count_inclusive(self, usage_metadata: UsageMetadata) -> bool:
        """
        Check if the candidate token count is inclusive of the thinking token count

        if prompttokencount + candidatesTokenCount == totalTokenCount, then the candidate token count is inclusive of the thinking token count

        else the candidate token count is exclusive of the thinking token count

        Addresses - https://github.com/BerriAI/litellm/pull/10141#discussion_r2052272035
        """
        if usage_metadata.get("promptTokenCount", 0) + usage_metadata.get(
            "candidatesTokenCount", 0
        ) == usage_metadata.get("totalTokenCount", 0):
            return True
        else:
            return False

    def _calculate_usage(
        self,
        completion_response: Union[
            GenerateContentResponseBody, BidiGenerateContentServerMessage
        ],
    ) -> Usage:
        if "usageMetadata" not in completion_response:
            raise ValueError(
                f"usageMetadata not found in completion_response. Got={completion_response}"
            )
        cached_tokens: Optional[int] = None
        audio_tokens: Optional[int] = None
        text_tokens: Optional[int] = None
        prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
        reasoning_tokens: Optional[int] = None
        response_tokens: Optional[int] = None
        response_tokens_details: Optional[CompletionTokensDetailsWrapper] = None
        if "cachedContentTokenCount" in completion_response["usageMetadata"]:
            cached_tokens = completion_response["usageMetadata"][
                "cachedContentTokenCount"
            ]

        ## GEMINI LIVE API ONLY PARAMS ##
        if "responseTokenCount" in completion_response["usageMetadata"]:
            response_tokens = completion_response["usageMetadata"]["responseTokenCount"]
        if "responseTokensDetails" in completion_response["usageMetadata"]:
            response_tokens_details = CompletionTokensDetailsWrapper()
            for detail in completion_response["usageMetadata"]["responseTokensDetails"]:
                if detail["modality"] == "TEXT":
                    response_tokens_details.text_tokens = detail["tokenCount"]
                elif detail["modality"] == "AUDIO":
                    response_tokens_details.audio_tokens = detail["tokenCount"]
        #########################################################

        if "promptTokensDetails" in completion_response["usageMetadata"]:
            for detail in completion_response["usageMetadata"]["promptTokensDetails"]:
                if detail["modality"] == "AUDIO":
                    audio_tokens = detail["tokenCount"]
                elif detail["modality"] == "TEXT":
                    text_tokens = detail["tokenCount"]
        if "thoughtsTokenCount" in completion_response["usageMetadata"]:
            reasoning_tokens = completion_response["usageMetadata"][
                "thoughtsTokenCount"
            ]
        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=cached_tokens,
            audio_tokens=audio_tokens,
            text_tokens=text_tokens,
        )

        completion_tokens = response_tokens or completion_response["usageMetadata"].get(
            "candidatesTokenCount", 0
        )
        if (
            not self.is_candidate_token_count_inclusive(
                completion_response["usageMetadata"]
            )
            and reasoning_tokens
        ):
            completion_tokens = reasoning_tokens + completion_tokens
        ## GET USAGE ##
        usage = Usage(
            prompt_tokens=completion_response["usageMetadata"].get(
                "promptTokenCount", 0
            ),
            completion_tokens=completion_tokens,
            total_tokens=completion_response["usageMetadata"].get("totalTokenCount", 0),
            prompt_tokens_details=prompt_tokens_details,
            reasoning_tokens=reasoning_tokens,
            completion_tokens_details=response_tokens_details,
        )

        return usage

    def _check_finish_reason(
        self,
        chat_completion_message: Optional[ChatCompletionResponseMessage],
        finish_reason: Optional[str],
    ) -> OpenAIChatCompletionFinishReason:
        mapped_finish_reason = self.get_finish_reason_mapping()
        if chat_completion_message and chat_completion_message.get("function_call"):
            return "function_call"
        elif chat_completion_message and chat_completion_message.get("tool_calls"):
            return "tool_calls"
        elif (
            finish_reason and finish_reason in mapped_finish_reason.keys()
        ):  # vertex ai
            return mapped_finish_reason[finish_reason]
        else:
            return "stop"

    def _process_candidates(
        self, _candidates, model_response, standard_optional_params: dict
    ):
        """Helper method to process candidates and extract metadata"""
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            is_function_call,
        )

        grounding_metadata: List[dict] = []
        safety_ratings: List = []
        citation_metadata: List = []
        chat_completion_message: ChatCompletionResponseMessage = {"role": "assistant"}
        chat_completion_logprobs: Optional[ChoiceLogprobs] = None
        tools: Optional[List[ChatCompletionToolCallChunk]] = []
        functions: Optional[ChatCompletionToolCallFunctionChunk] = None

        for idx, candidate in enumerate(_candidates):
            if "content" not in candidate:
                continue

            if "groundingMetadata" in candidate:
                grounding_metadata.append(candidate["groundingMetadata"])  # type: ignore

            if "safetyRatings" in candidate:
                safety_ratings.append(candidate["safetyRatings"])

            if "citationMetadata" in candidate:
                citation_metadata.append(candidate["citationMetadata"])

            if "parts" in candidate["content"]:
                (
                    content,
                    reasoning_content,
                ) = VertexGeminiConfig().get_assistant_content_message(
                    parts=candidate["content"]["parts"]
                )

                audio_response = VertexGeminiConfig()._extract_audio_response_from_parts(
                    parts=candidate["content"]["parts"]
                )

                if audio_response is not None:
                    cast(Dict[str, Any], chat_completion_message)["audio"] = audio_response
                    chat_completion_message["content"] = None  # OpenAI spec
                elif content is not None:
                    chat_completion_message["content"] = content

                if reasoning_content is not None:
                    chat_completion_message["reasoning_content"] = reasoning_content

                functions, tools = self._transform_parts(
                    parts=candidate["content"]["parts"],
                    index=candidate.get("index", idx),
                    is_function_call=is_function_call(standard_optional_params),
                )

            if "logprobsResult" in candidate:
                chat_completion_logprobs = self._transform_logprobs(
                    logprobs_result=candidate["logprobsResult"]
                )

            if tools:
                chat_completion_message["tool_calls"] = tools

            if functions is not None:
                chat_completion_message["function_call"] = functions

            choice = litellm.Choices(
                finish_reason=self._check_finish_reason(
                    chat_completion_message, candidate.get("finishReason")
                ),
                index=candidate.get("index", idx),
                message=chat_completion_message,  # type: ignore
                logprobs=chat_completion_logprobs,
                enhancements=None,
            )

            model_response.choices.append(choice)

        return grounding_metadata, safety_ratings, citation_metadata

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Dict,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = GenerateContentResponseBody(**raw_response.json())  # type: ignore
        except Exception as e:
            raise VertexAIError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    raw_response.text, str(e)
                ),
                status_code=422,
                headers=raw_response.headers,
            )

        ## GET MODEL ##
        model_response.model = model

        ## CHECK IF RESPONSE FLAGGED
        if (
            "promptFeedback" in completion_response
            and "blockReason" in completion_response["promptFeedback"]
        ):
            return self._handle_blocked_response(
                model_response=model_response,
                completion_response=completion_response,
            )

        _candidates = completion_response.get("candidates")
        if _candidates and len(_candidates) > 0:
            content_policy_violations = (
                VertexGeminiConfig().get_flagged_finish_reasons()
            )
            if (
                "finishReason" in _candidates[0]
                and _candidates[0]["finishReason"] in content_policy_violations.keys()
            ):
                return self._handle_content_policy_violation(
                    model_response=model_response,
                    completion_response=completion_response,
                )

        model_response.choices = []

        try:
            grounding_metadata, safety_ratings, citation_metadata = [], [], []
            if _candidates:
                (
                    grounding_metadata,
                    safety_ratings,
                    citation_metadata,
                ) = self._process_candidates(
                    _candidates, model_response, logging_obj.optional_params
                )

            usage = self._calculate_usage(completion_response=completion_response)
            setattr(model_response, "usage", usage)

            ## ADD METADATA TO RESPONSE ##
            setattr(model_response, "vertex_ai_grounding_metadata", grounding_metadata)
            model_response._hidden_params[
                "vertex_ai_grounding_metadata"
            ] = grounding_metadata

            setattr(model_response, "vertex_ai_safety_results", safety_ratings)
            model_response._hidden_params[
                "vertex_ai_safety_results"
            ] = safety_ratings  # older approach - maintaining to prevent regressions

            ## ADD CITATION METADATA ##
            setattr(model_response, "vertex_ai_citation_metadata", citation_metadata)
            model_response._hidden_params[
                "vertex_ai_citation_metadata"
            ] = citation_metadata  # older approach - maintaining to prevent regressions

        except Exception as e:
            raise VertexAIError(
                message="Received={}, Error converting to valid response block={}. File an issue if litellm error - https://github.com/BerriAI/litellm/issues".format(
                    completion_response, str(e)
                ),
                status_code=422,
                headers=raw_response.headers,
            )

        return model_response

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[ContentType]:
        return _gemini_convert_messages_with_history(messages=messages)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, httpx.Headers]
    ) -> BaseLLMException:
        return VertexAIError(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        headers: Dict,
    ) -> Dict:
        raise NotImplementedError(
            "Vertex AI has a custom implementation of transform_request. Needs sync + async."
        )

    def validate_environment(
        self,
        headers: Optional[Dict],
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        default_headers = {
            "Content-Type": "application/json",
        }
        if api_key is not None:
            default_headers["Authorization"] = f"Bearer {api_key}"
        if headers is not None:
            default_headers.update(headers)

        return default_headers


async def make_call(
    client: Optional[AsyncHTTPHandler],
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
):
    if client is None:
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
        )

    try:
        response = await client.post(api_base, headers=headers, data=data, stream=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        exception_string = str(await e.response.aread())
        raise VertexAIError(
            status_code=e.response.status_code,
            message=VertexGeminiConfig().translate_exception_str(exception_string),
            headers=e.response.headers,
        )
    if response.status_code != 200 and response.status_code != 201:
        raise VertexAIError(
            status_code=response.status_code,
            message=response.text,
            headers=response.headers,
        )

    completion_stream = ModelResponseIterator(
        streaming_response=response.aiter_lines(), sync_stream=False
    )
    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


def make_sync_call(
    client: Optional[HTTPHandler],  # module-level client
    gemini_client: Optional[HTTPHandler],  # if passed by user
    api_base: str,
    headers: dict,
    data: str,
    model: str,
    messages: list,
    logging_obj,
):
    if gemini_client is not None:
        client = gemini_client
    if client is None:
        client = HTTPHandler()  # Create a new client if none provided

    response = client.post(api_base, headers=headers, data=data, stream=True)

    if response.status_code != 200 and response.status_code != 201:
        raise VertexAIError(
            status_code=response.status_code,
            message=str(response.read()),
            headers=response.headers,
        )

    completion_stream = ModelResponseIterator(
        streaming_response=response.iter_lines(), sync_stream=True
    )

    # LOGGING
    logging_obj.post_call(
        input=messages,
        api_key="",
        original_response="first stream response received",
        additional_args={"complete_input_dict": data},
    )

    return completion_stream


class VertexLLM(VertexBase):
    def __init__(self) -> None:
        super().__init__()

    async def async_streaming(
        self,
        model: str,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: dict,
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params: dict,
        logger_fn=None,
        api_base: Optional[str] = None,
        client: Optional[AsyncHTTPHandler] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        gemini_api_key: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ) -> CustomStreamWrapper:
        request_body = await async_transform_request_body(**data)  # type: ignore

        should_use_v1beta1_features = self.is_using_v1beta1_features(
            optional_params=optional_params
        )

        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=should_use_v1beta1_features,
        )

        headers = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        request_body_str = json.dumps(request_body)
        streaming_response = CustomStreamWrapper(
            completion_stream=None,
            make_call=partial(
                make_call,
                client=client,
                api_base=api_base,
                headers=headers,
                data=request_body_str,
                model=model,
                messages=messages,
                logging_obj=logging_obj,
            ),
            model=model,
            custom_llm_provider="vertex_ai_beta",
            logging_obj=logging_obj,
        )
        return streaming_response

    async def async_completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        data: dict,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        timeout: Optional[Union[float, httpx.Timeout]],
        encoding,
        logging_obj,
        stream,
        optional_params: dict,
        litellm_params: dict,
        logger_fn=None,
        api_base: Optional[str] = None,
        client: Optional[AsyncHTTPHandler] = None,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        gemini_api_key: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        should_use_v1beta1_features = self.is_using_v1beta1_features(
            optional_params=optional_params
        )

        _auth_header, vertex_project = await self._ensure_access_token_async(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, api_base = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=should_use_v1beta1_features,
        )

        headers = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        request_body = await async_transform_request_body(**data)  # type: ignore
        _async_client_params = {}
        if timeout:
            _async_client_params["timeout"] = timeout
        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(
                params=_async_client_params, llm_provider=litellm.LlmProviders.VERTEX_AI
            )
        else:
            client = client  # type: ignore
        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": request_body,
                "api_base": api_base,
                "headers": headers,
            },
        )

        try:
            response = await client.post(
                api_base, headers=headers, json=cast(dict, request_body)
            )  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(
                status_code=error_code,
                message=err.response.text,
                headers=err.response.headers,
            )
        except httpx.TimeoutException:
            raise VertexAIError(
                status_code=408,
                message="Timeout error occurred.",
                headers=None,
            )

        return VertexGeminiConfig().transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key="",
            request_data=cast(dict, request_body),
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
        )

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: Callable,
        custom_llm_provider: Literal[
            "vertex_ai", "vertex_ai_beta", "gemini"
        ],  # if it's vertex_ai or gemini (google ai studio)
        encoding,
        logging_obj,
        optional_params: dict,
        acompletion: bool,
        timeout: Optional[Union[float, httpx.Timeout]],
        vertex_project: Optional[str],
        vertex_location: Optional[str],
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES],
        gemini_api_key: Optional[str],
        litellm_params: dict,
        logger_fn=None,
        extra_headers: Optional[dict] = None,
        client: Optional[Union[AsyncHTTPHandler, HTTPHandler]] = None,
        api_base: Optional[str] = None,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        stream: Optional[bool] = optional_params.pop("stream", None)  # type: ignore

        transform_request_params = {
            "gemini_api_key": gemini_api_key,
            "messages": messages,
            "api_base": api_base,
            "model": model,
            "client": client,
            "timeout": timeout,
            "extra_headers": extra_headers,
            "optional_params": optional_params,
            "logging_obj": logging_obj,
            "custom_llm_provider": custom_llm_provider,
            "litellm_params": litellm_params,
        }

        ### ROUTING (ASYNC, STREAMING, SYNC)
        if acompletion:
            ### ASYNC STREAMING
            if stream is True:
                return self.async_streaming(
                    model=model,
                    messages=messages,
                    api_base=api_base,
                    model_response=model_response,
                    print_verbose=print_verbose,
                    encoding=encoding,
                    logging_obj=logging_obj,
                    optional_params=optional_params,
                    stream=stream,
                    litellm_params=litellm_params,
                    logger_fn=logger_fn,
                    timeout=timeout,
                    client=client,  # type: ignore
                    data=transform_request_params,
                    vertex_project=vertex_project,
                    vertex_location=vertex_location,
                    vertex_credentials=vertex_credentials,
                    gemini_api_key=gemini_api_key,
                    custom_llm_provider=custom_llm_provider,
                    extra_headers=extra_headers,
                )
            ### ASYNC COMPLETION
            return self.async_completion(
                model=model,
                messages=messages,
                data=transform_request_params,  # type: ignore
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                stream=stream,
                litellm_params=litellm_params,
                logger_fn=logger_fn,
                timeout=timeout,
                client=client,  # type: ignore
                vertex_project=vertex_project,
                vertex_location=vertex_location,
                vertex_credentials=vertex_credentials,
                gemini_api_key=gemini_api_key,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
            )

        should_use_v1beta1_features = self.is_using_v1beta1_features(
            optional_params=optional_params
        )

        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider=custom_llm_provider,
        )

        auth_header, url = self._get_token_and_url(
            model=model,
            gemini_api_key=gemini_api_key,
            auth_header=_auth_header,
            vertex_project=vertex_project,
            vertex_location=vertex_location,
            vertex_credentials=vertex_credentials,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            should_use_v1beta1_features=should_use_v1beta1_features,
        )
        headers = VertexGeminiConfig().validate_environment(
            api_key=auth_header,
            headers=extra_headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        ## TRANSFORMATION ##
        data = sync_transform_request_body(**transform_request_params)

        ## LOGGING
        logging_obj.pre_call(
            input=messages,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": url,
                "headers": headers,
            },
        )

        ## SYNC STREAMING CALL ##
        if stream is True:
            request_data_str = json.dumps(data)
            streaming_response = CustomStreamWrapper(
                completion_stream=None,
                make_call=partial(
                    make_sync_call,
                    gemini_client=(
                        client
                        if client is not None and isinstance(client, HTTPHandler)
                        else None
                    ),
                    api_base=url,
                    data=request_data_str,
                    model=model,
                    messages=messages,
                    logging_obj=logging_obj,
                    headers=headers,
                ),
                model=model,
                custom_llm_provider="vertex_ai_beta",
                logging_obj=logging_obj,
            )

            return streaming_response
        ## COMPLETION CALL ##

        if client is None or isinstance(client, AsyncHTTPHandler):
            _params = {}
            if timeout is not None:
                if isinstance(timeout, float) or isinstance(timeout, int):
                    timeout = httpx.Timeout(timeout)
                _params["timeout"] = timeout
            client = HTTPHandler(**_params)  # type: ignore
        else:
            client = client

        try:
            response = client.post(url=url, headers=headers, json=data)  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise VertexAIError(
                status_code=error_code,
                message=err.response.text,
                headers=err.response.headers,
            )
        except httpx.TimeoutException:
            raise VertexAIError(
                status_code=408,
                message="Timeout error occurred.",
                headers=None,
            )

        return VertexGeminiConfig().transform_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key="",
            request_data=data,  # type: ignore
            messages=messages,
            encoding=encoding,
        )


class ModelResponseIterator:
    def __init__(self, streaming_response, sync_stream: bool):
        self.streaming_response = streaming_response
        self.chunk_type: Literal["valid_json", "accumulated_json"] = "valid_json"
        self.accumulated_json = ""
        self.sent_first_chunk = False

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            processed_chunk = GenerateContentResponseBody(**chunk)  # type: ignore

            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            _candidates: Optional[List[Candidates]] = processed_chunk.get("candidates")
            gemini_chunk: Optional[Candidates] = None
            if _candidates and len(_candidates) > 0:
                gemini_chunk = _candidates[0]

            if (
                gemini_chunk
                and "content" in gemini_chunk
                and "parts" in gemini_chunk["content"]
            ):
                if "text" in gemini_chunk["content"]["parts"][0]:
                    text = gemini_chunk["content"]["parts"][0]["text"]
                elif "functionCall" in gemini_chunk["content"]["parts"][0]:
                    function_call = ChatCompletionToolCallFunctionChunk(
                        name=gemini_chunk["content"]["parts"][0]["functionCall"][
                            "name"
                        ],
                        arguments=json.dumps(
                            gemini_chunk["content"]["parts"][0]["functionCall"]["args"]
                        ),
                    )
                    tool_use = ChatCompletionToolCallChunk(
                        id=str(uuid.uuid4()),
                        type="function",
                        function=function_call,
                        index=0,
                    )

            if gemini_chunk and "finishReason" in gemini_chunk:
                finish_reason = VertexGeminiConfig()._check_finish_reason(
                    chat_completion_message=None,
                    finish_reason=gemini_chunk["finishReason"],
                )
                ## DO NOT SET 'is_finished' = True
                ## GEMINI SETS FINISHREASON ON EVERY CHUNK!

            if "usageMetadata" in processed_chunk:
                usage = ChatCompletionUsageBlock(
                    prompt_tokens=processed_chunk["usageMetadata"].get(
                        "promptTokenCount", 0
                    ),
                    completion_tokens=processed_chunk["usageMetadata"].get(
                        "candidatesTokenCount", 0
                    ),
                    total_tokens=processed_chunk["usageMetadata"].get(
                        "totalTokenCount", 0
                    ),
                    completion_tokens_details={
                        "reasoning_tokens": processed_chunk["usageMetadata"].get(
                            "thoughtsTokenCount", 0
                        )
                    },
                )

            returned_chunk = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=False,
                finish_reason=finish_reason,
                usage=usage,
                index=0,
            )
            return returned_chunk
        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        self.response_iterator = self.streaming_response
        return self

    def handle_valid_json_chunk(self, chunk: str) -> GenericStreamingChunk:
        chunk = chunk.strip()
        try:
            json_chunk = json.loads(chunk)

        except json.JSONDecodeError as e:
            if (
                self.sent_first_chunk is False
            ):  # only check for accumulated json, on first chunk, else raise error. Prevent real errors from being masked.
                self.chunk_type = "accumulated_json"
                return self.handle_accumulated_json_chunk(chunk=chunk)
            raise e

        if self.sent_first_chunk is False:
            self.sent_first_chunk = True

        return self.chunk_parser(chunk=json_chunk)

    def handle_accumulated_json_chunk(self, chunk: str) -> GenericStreamingChunk:
        chunk = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
        message = chunk.replace("\n\n", "")

        # Accumulate JSON data
        self.accumulated_json += message

        # Try to parse the accumulated JSON
        try:
            _data = json.loads(self.accumulated_json)
            self.accumulated_json = ""  # reset after successful parsing
            return self.chunk_parser(chunk=_data)
        except json.JSONDecodeError:
            # If it's not valid JSON yet, continue to the next event
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

    def _common_chunk_parsing_logic(self, chunk: str) -> GenericStreamingChunk:
        try:
            chunk = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(chunk) or ""
            if len(chunk) > 0:
                """
                Check if initial chunk valid json
                - if partial json -> enter accumulated json logic
                - if valid - continue
                """
                if self.chunk_type == "valid_json":
                    return self.handle_valid_json_chunk(chunk=chunk)
                elif self.chunk_type == "accumulated_json":
                    return self.handle_accumulated_json_chunk(chunk=chunk)

            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )
        except Exception:
            raise

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            if self.chunk_type == "accumulated_json" and self.accumulated_json:
                return self.handle_accumulated_json_chunk(chunk="")
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self._common_chunk_parsing_logic(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            if self.chunk_type == "accumulated_json" and self.accumulated_json:
                return self.handle_accumulated_json_chunk(chunk="")
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self._common_chunk_parsing_logic(chunk=chunk)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")
