"""
Transformation logic for Perplexity Agentic Research API (Responses API)

This module handles the translation between OpenAI's Responses API format
and Perplexity's Responses API format, which supports:
- Third-party model access (OpenAI, Anthropic, Google, xAI, etc.)
- Presets for optimized configurations
- Web search and URL fetching tools
- Reasoning effort control
- Instructions parameter for system-level guidance
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class PerplexityResponsesConfig(OpenAIResponsesAPIConfig):
    """
    Configuration for Perplexity Agentic Research API (Responses API)

    
    Reference: https://docs.perplexity.ai/agentic-research/quickstart
    """

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.PERPLEXITY

    def get_supported_openai_params(self, model: str) -> list:
        """
        Perplexity Responses API supports a different set of parameters
        
        Ref: https://docs.perplexity.ai/api-reference/responses-post
        """
        return [
            "max_output_tokens",
            "stream",
            "temperature",
            "top_p",
            "tools",
            "reasoning",
            "preset",
            "instructions",
            "models",  # Model fallback support
        ]

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """Validate environment and set up headers"""
        # Get API key from environment
        api_key = (
            get_secret_str("PERPLEXITYAI_API_KEY")
            or get_secret_str("PERPLEXITY_API_KEY")
        )
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        headers["Content-Type"] = "application/json"
        
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """Get the complete URL for the Perplexity Responses API"""
        if api_base is None:
            api_base = get_secret_str("PERPLEXITY_API_BASE") or "https://api.perplexity.ai"
        
        # Ensure api_base doesn't end with a slash
        api_base = api_base.rstrip("/")
        
        # Add the responses endpoint
        return f"{api_base}/v1/responses"

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI Responses API parameters to Perplexity format
        
        Key differences:
        - Supports 'preset' parameter for predefined configurations
        - Supports 'instructions' parameter for system-level guidance
        - Tools are specified differently (web_search, fetch_url)
        """
        mapped_params: Dict[str, Any] = {}
        
        # Map standard parameters
        if response_api_optional_params.get("max_output_tokens"):
            mapped_params["max_output_tokens"] = response_api_optional_params["max_output_tokens"]
        
        if response_api_optional_params.get("temperature"):
            mapped_params["temperature"] = response_api_optional_params["temperature"]
        
        if response_api_optional_params.get("top_p"):
            mapped_params["top_p"] = response_api_optional_params["top_p"]
        
        if response_api_optional_params.get("stream"):
            mapped_params["stream"] = response_api_optional_params["stream"]
        
        if response_api_optional_params.get("stream_options"):
            mapped_params["stream_options"] = response_api_optional_params["stream_options"]
        
        # Map Perplexity-specific parameters (using .get() with Any dict access)
        preset = response_api_optional_params.get("preset")  # type: ignore
        if preset:
            mapped_params["preset"] = preset
        
        instructions = response_api_optional_params.get("instructions")  # type: ignore
        if instructions:
            mapped_params["instructions"] = instructions
        
        if response_api_optional_params.get("reasoning"):
            mapped_params["reasoning"] = response_api_optional_params["reasoning"]
        
        tools = response_api_optional_params.get("tools")
        if tools:
            # Convert tools to list of dicts for transformation
            tools_list = [dict(tool) if hasattr(tool, '__dict__') else tool for tool in tools]  # type: ignore
            mapped_params["tools"] = self._transform_tools(tools_list)  # type: ignore
        
        return mapped_params

    def _transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform tools to Perplexity format
        
        Perplexity supports:
        - web_search: Performs web searches
        - fetch_url: Fetches content from URLs
        """
        perplexity_tools = []
        
        for tool in tools:
            if isinstance(tool, dict):
                tool_type = tool.get("type")
                
                # Direct Perplexity tool format
                if tool_type in ["web_search", "fetch_url"]:
                    perplexity_tools.append(tool)
                
                # OpenAI function format - try to map to Perplexity tools
                elif tool_type == "function":
                    function = tool.get("function", {})
                    function_name = function.get("name", "")
                    
                    if function_name == "web_search" or "search" in function_name.lower():
                        perplexity_tools.append({"type": "web_search"})
                    elif function_name == "fetch_url" or "fetch" in function_name.lower():
                        perplexity_tools.append({"type": "fetch_url"})
        
        return perplexity_tools

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Transform request to Perplexity Responses API format
        """
        # Check if the model is a preset (format: preset/preset-name)
        if model.startswith("preset/"):
            preset_name = model.replace("preset/", "")
            data = {
                "preset": preset_name,
                "input": self._format_input(input),
            }
        # Check if preset is explicitly provided in params
        elif response_api_optional_request_params.get("preset"):
            data = {
                "preset": response_api_optional_request_params.pop("preset"),
                "input": self._format_input(input),
            }
        else:
            # Full request format for third-party models
            data = {
                "model": model,
                "input": self._format_input(input),
            }
        
        # Add all optional parameters
        for key, value in response_api_optional_request_params.items():
            data[key] = value
        
        return data

    def _format_input(self, input: Union[str, ResponseInputParam]) -> Union[str, List[Dict[str, Any]]]:
        """
        Format input for Perplexity Responses API
        
        The API accepts either:
        - A simple string for single-turn queries
        - An array of message objects for multi-turn conversations
        """
        if isinstance(input, str):
            return input
        
        # Handle ResponseInputParam format
        if isinstance(input, list):
            formatted_messages = []
            for item in input:
                if isinstance(item, dict):
                    formatted_message = {
                        "type": "message",
                        "role": item.get("role"),
                        "content": item.get("content", ""),
                    }
                    formatted_messages.append(formatted_message)
            return formatted_messages
        
        return str(input)

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """
        Transform Perplexity Responses API response to OpenAI Responses API format
        """
        try:
            raw_response_json = raw_response.json()
        except Exception as e:
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=f"Failed to parse response: {str(e)}",
            )

        # Check for error status
        status = raw_response_json.get("status")
        if status == "failed":
            error = raw_response_json.get("error", {})
            error_message = error.get("message", "Unknown error")
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=error_message,
            )

        # Transform usage to handle Perplexity's cost structure
        usage_data = raw_response_json.get("usage", {})
        transformed_usage_dict = self._transform_usage(usage_data)
        
        # Convert usage dict to ResponseAPIUsage object
        usage_obj = ResponseAPIUsage(**transformed_usage_dict) if transformed_usage_dict else None
        
        # Map Perplexity response to OpenAI Responses API format
        response = ResponsesAPIResponse(
            id=raw_response_json.get("id", ""),
            object="response",
            created_at=raw_response_json.get("created_at", 0),
            status=raw_response_json.get("status", "completed"),
            model=raw_response_json.get("model", model),
            output=raw_response_json.get("output", []),
            usage=usage_obj,
        )

        return response
    
    def _transform_usage(self, usage_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform Perplexity usage data to OpenAI format
        
        Perplexity returns:
        {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "cost": {
                "currency": "USD",
                "input_cost": 0.0001,
                "output_cost": 0.0002,
                "total_cost": 0.0003
            }
        }
        
        OpenAI expects:
        {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "cost": 0.0003
        }
        """
        transformed = {
            "input_tokens": usage_data.get("input_tokens", 0),
            "output_tokens": usage_data.get("output_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }
        
        # Transform cost from Perplexity format (dict) to OpenAI format (float)
        cost_obj = usage_data.get("cost")
        if isinstance(cost_obj, dict) and "total_cost" in cost_obj:
            transformed["cost"] = cost_obj["total_cost"]
            verbose_logger.debug(
                "Transformed Perplexity cost object to float: %s -> %s",
                cost_obj,
                cost_obj["total_cost"]
            )
        elif cost_obj is not None:
            # If cost is already a float/number, use it as-is
            transformed["cost"] = cost_obj
        
        # Add input_tokens_details if present
        if "input_tokens_details" in usage_data:
            transformed["input_tokens_details"] = usage_data["input_tokens_details"]
        
        # Add output_tokens_details if present
        if "output_tokens_details" in usage_data:
            transformed["output_tokens_details"] = usage_data["output_tokens_details"]
        
        return transformed

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        """
        Transform a parsed streaming response chunk into a ResponsesAPIStreamingResponse
        """
        # Get the event type from the chunk
        verbose_logger.debug("Raw Perplexity Chunk=%s", parsed_chunk)
        event_type = str(parsed_chunk.get("type"))
        event_pydantic_model = PerplexityResponsesConfig.get_event_model_class(
            event_type=event_type
        )
        
        # Transform Perplexity-specific fields to OpenAI format
        parsed_chunk = self._transform_perplexity_chunk(parsed_chunk)
        
        # Defensive: Handle error.code being null (similar to OpenAI implementation)
        try:
            error_obj = parsed_chunk.get("error")
            if isinstance(error_obj, dict) and error_obj.get("code") is None:
                # Preserve other fields, but ensure `code` is a non-null string
                parsed_chunk = dict(parsed_chunk)
                parsed_chunk["error"] = dict(error_obj)
                parsed_chunk["error"]["code"] = "unknown_error"
        except Exception:
            # If anything unexpected happens here, fall back to attempting
            # instantiation and let higher-level handlers manage errors.
            verbose_logger.debug("Failed to coalesce error.code in parsed_chunk")

        return event_pydantic_model(**parsed_chunk)

    def _transform_perplexity_chunk(self, chunk: dict) -> dict:
        """
        Transform Perplexity-specific fields in a streaming chunk to OpenAI format.
        
        This handles:
        - Converting Perplexity's cost object to a simple float
        """
        # Make a copy to avoid modifying the original
        chunk = dict(chunk)
        
        # Transform usage.cost from Perplexity format to OpenAI format
        # Perplexity: {"currency": "USD", "input_cost": 0.0001, "output_cost": 0.0002, "total_cost": 0.0003}
        # OpenAI: 0.0003 (just the total_cost as a float)
        try:
            response_obj = chunk.get("response")
            if isinstance(response_obj, dict):
                usage_obj = response_obj.get("usage")
                if isinstance(usage_obj, dict):
                    cost_obj = usage_obj.get("cost")
                    if isinstance(cost_obj, dict) and "total_cost" in cost_obj:
                        # Replace the cost object with just the total_cost value
                        chunk = dict(chunk)
                        chunk["response"] = dict(response_obj)
                        chunk["response"]["usage"] = dict(usage_obj)
                        chunk["response"]["usage"]["cost"] = cost_obj["total_cost"]
                        verbose_logger.debug(
                            "Transformed Perplexity cost object to float: %s -> %s",
                            cost_obj,
                            cost_obj["total_cost"]
                        )
        except Exception as e:
            # If transformation fails, log and continue with original chunk
            verbose_logger.debug("Failed to transform Perplexity cost object: %s", e)
        
        return chunk
