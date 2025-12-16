"""
Support for Snowflake REST API
"""

import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ChatCompletionMessageToolCall, Function, ModelResponse, ModelResponseStream

from ...openai_like.chat.transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SnowflakeStreamingHandler(BaseModelResponseIterator):
    """
    Custom streaming handler for Snowflake that handles missing fields in chunk responses.
    Snowflake's streaming responses may not include all OpenAI-expected fields like 'created'.
    """

    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        """
        Parse Snowflake streaming chunks, providing defaults for missing fields.
        Also transforms Claude-format tool_use to OpenAI-format tool_calls.

        Args:
            chunk: Streaming chunk from Snowflake

        Returns:
            ModelResponseStream with all required fields
        """
        from litellm._logging import verbose_logger
        import json

        # TRACE 4: Log each chunk from Snowflake
        verbose_logger.debug("=" * 80)
        verbose_logger.debug("TRACE 4: SNOWFLAKE API RESPONSE CHUNK")
        verbose_logger.debug(f"Chunk: {json.dumps(chunk, indent=2, default=str)}")
        verbose_logger.debug("=" * 80)

        try:
            # Snowflake may not include 'created' timestamp, use current time as default
            created = chunk.get("created", int(time.time()))

            # Transform choices to convert tool_use (Claude format) to tool_calls (OpenAI format)
            choices = chunk.get("choices", [])
            transformed_choices = []

            for choice in choices:
                delta = choice.get("delta", {})

                # Check if this is a tool_use block (Claude format)
                if delta.get("type") == "tool_use":
                    # Convert to OpenAI tool_calls format
                    from litellm.types.utils import ChatCompletionDeltaToolCall, Function

                    tool_call = ChatCompletionDeltaToolCall(
                        id=delta.get("tool_use_id") or delta.get("id"),
                        type="function",
                        function=Function(
                            name=delta.get("name"),
                            arguments=delta.get("input", "")
                        ),
                        index=choice.get("index", 0)
                    )

                    # Update delta to include tool_calls
                    delta["tool_calls"] = [tool_call]
                    # Remove Claude-specific fields
                    delta.pop("type", None)
                    delta.pop("tool_use_id", None)
                    delta.pop("content_list", None)

                transformed_choices.append(choice)

            return ModelResponseStream(
                id=chunk.get("id", ""),
                object="chat.completion.chunk",
                created=created,
                model=chunk.get("model", ""),
                choices=transformed_choices,
                usage=chunk.get("usage"),
            )
        except Exception as e:
            raise e


class SnowflakeConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api

    Snowflake Cortex LLM REST API supports function calling with specific models (e.g., Claude 3.5 Sonnet).
    This config handles transformation between OpenAI format and Snowflake's tool_spec format.
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "temperature",
            "max_tokens",
            "top_p",
            "response_format",
            "tools",
            "tool_choice",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        If any supported_openai_params are in non_default_params, add them to optional_params, so they are used in API call

        Args:
            non_default_params (dict): Non-default parameters to filter.
            optional_params (dict): Optional parameters to update.
            model (str): Model name for parameter support check.

        Returns:
            dict: Updated optional_params with supported non-default parameters.
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _transform_tool_calls_from_snowflake_to_openai(
        self, content_list: List[Dict[str, Any]]
    ) -> Tuple[str, Optional[List[ChatCompletionMessageToolCall]]]:
        """
        Transform Snowflake tool calls to OpenAI format.

        Args:
            content_list: Snowflake's content_list array containing text and tool_use items

        Returns:
            Tuple of (text_content, tool_calls)

        Snowflake format in content_list:
        {
          "type": "tool_use",
          "tool_use": {
            "tool_use_id": "tooluse_...",
            "name": "get_weather",
            "input": {"location": "Paris"}
          }
        }

        OpenAI format (returned tool_calls):
        ChatCompletionMessageToolCall(
            id="tooluse_...",
            type="function",
            function=Function(name="get_weather", arguments='{"location": "Paris"}')
        )
        """
        text_content = ""
        tool_calls: List[ChatCompletionMessageToolCall] = []

        for idx, content_item in enumerate(content_list):
            if content_item.get("type") == "text":
                text_content += content_item.get("text", "")

            ## TOOL CALLING
            elif content_item.get("type") == "tool_use":
                tool_use_data = content_item.get("tool_use", {})
                tool_call = ChatCompletionMessageToolCall(
                    id=tool_use_data.get("tool_use_id", ""),
                    type="function",
                    function=Function(
                        name=tool_use_data.get("name", ""),
                        arguments=json.dumps(tool_use_data.get("input", {})),
                    ),
                )
                tool_calls.append(tool_call)

        return text_content, tool_calls if tool_calls else None

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
        from litellm._logging import verbose_logger

        response_json = raw_response.json()

        verbose_logger.debug(f"Snowflake response: status={raw_response.status_code}, has_choices={bool(response_json.get('choices'))}")
        if response_json.get("choices"):
            first_choice = response_json["choices"][0]
            verbose_logger.debug(f"Snowflake first choice: has_message={bool(first_choice.get('message'))}, finish_reason={first_choice.get('finish_reason')}")
            if first_choice.get("message"):
                msg = first_choice["message"]
                verbose_logger.debug(f"Snowflake message: has_content={bool(msg.get('content'))}, has_content_list={bool(msg.get('content_list'))}, has_tool_calls={bool(msg.get('tool_calls'))}")

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE TRANSFORMATION
        # Snowflake returns content_list (not content) with tool_use objects
        # We need to transform this to OpenAI's format with content + tool_calls
        if "choices" in response_json and len(response_json["choices"]) > 0:
            choice = response_json["choices"][0]
            if "message" in choice and "content_list" in choice["message"]:
                content_list = choice["message"]["content_list"]
                (
                    text_content,
                    tool_calls,
                ) = self._transform_tool_calls_from_snowflake_to_openai(content_list)

                # Update the choice message with OpenAI format
                choice["message"]["content"] = text_content
                if tool_calls:
                    choice["message"]["tool_calls"] = tool_calls

                # Remove Snowflake-specific content_list
                del choice["message"]["content_list"]

        returned_response = ModelResponse(**response_json)

        returned_response.model = "snowflake/" + (returned_response.model or "")

        if model is not None:
            returned_response._hidden_params["model"] = model
        return returned_response

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
        """
        Return headers to use for Snowflake completion request

        Snowflake REST API Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#api-reference
        Expected headers:
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + <JWT or PAT>,
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT" or "PROGRAMMATIC_ACCESS_TOKEN"
        }
        """

        if api_key is None:
            raise ValueError("Missing Snowflake JWT or PAT key")

        # Detect if using PAT token (prefixed with "pat/")
        token_type = "KEYPAIR_JWT"
        token = api_key

        if api_key.startswith("pat/"):
            token_type = "PROGRAMMATIC_ACCESS_TOKEN"
            token = api_key[4:]  # Strip "pat/" prefix

        headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + token,
                "X-Snowflake-Authorization-Token-Type": token_type,
            }
        )
        return headers

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or f"""https://{get_secret_str("SNOWFLAKE_ACCOUNT_ID")}.snowflakecomputing.com/api/v2/cortex/inference:complete"""
            or get_secret_str("SNOWFLAKE_API_BASE")
        )
        dynamic_api_key = api_key or get_secret_str("SNOWFLAKE_JWT")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default Snowflake Cortex endpoint.
        """
        if not api_base:
            api_base = f"""https://{get_secret_str("SNOWFLAKE_ACCOUNT_ID")}.snowflakecomputing.com/api/v2/cortex/inference:complete"""

        return api_base

    def get_model_response_iterator(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """
        Return custom streaming handler for Snowflake that handles missing fields.
        """
        return SnowflakeStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def _transform_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform OpenAI tool format to Snowflake tool format.

        Args:
            tools: List of tools in OpenAI format

        Returns:
            List of tools in Snowflake format

        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "...",
                "parameters": {...}
            }
        }

        Snowflake format:
        {
            "tool_spec": {
                "type": "generic",
                "name": "get_weather",
                "description": "...",
                "input_schema": {...}
            }
        }
        """
        snowflake_tools: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                snowflake_tool: Dict[str, Any] = {
                    "tool_spec": {
                        "type": "generic",
                        "name": function.get("name"),
                        "input_schema": function.get(
                            "parameters",
                            {"type": "object", "properties": {}},
                        ),
                    }
                }
                # Add description if present
                if "description" in function:
                    snowflake_tool["tool_spec"]["description"] = function[
                        "description"
                    ]

                snowflake_tools.append(snowflake_tool)

        return snowflake_tools

    def _transform_tool_choice(
        self, tool_choice: Union[str, Dict[str, Any]], tool_names: List[str] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Transform OpenAI tool_choice format to Snowflake format.

        Args:
            tool_choice: Tool choice in OpenAI format (str or dict)
            tool_names: List of available tool names

        Returns:
            Tool choice in Snowflake format

        OpenAI format (string):
        "auto", "required", "none"

        Snowflake format (object):
        {"type": "auto", "name": ["tool1", "tool2"]}, {"type": "required", "name": [...]}, {"type": "none"}

        OpenAI format (dict):
        {"type": "function", "function": {"name": "get_weather"}}

        Snowflake format (dict):
        {"type": "tool", "name": ["get_weather"]}
        """
        if isinstance(tool_choice, str):
            # Convert string values to Snowflake's object format
            # OpenAI: "auto" -> Snowflake: {"type": "auto"}
            # OpenAI: "required" -> Snowflake: {"type": "required", "name": [...]}
            result = {"type": tool_choice}
            # Only add tool names for "required" type (not for "auto" or "none")
            # "auto" means let the model decide from all available tools
            if tool_names and tool_choice == "required":
                result["name"] = tool_names
            return result

        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                function_name = tool_choice.get("function", {}).get("name")
                if function_name:
                    return {
                        "type": "tool",
                        "name": [function_name],  # Snowflake expects array
                    }

        return tool_choice

    def _transform_messages(
        self, messages: List[AllMessageValues]
    ) -> List[Dict[str, Any]]:
        """
        Transform OpenAI message format to Snowflake format.

        Handles tool result messages by converting from OpenAI's role="tool" format
        to Snowflake's content_list with tool_results format.

        Also handles assistant messages with tool_calls by converting them to Snowflake's
        content_list with tool_use format.

        OpenAI tool result format:
        {
            "role": "tool",
            "tool_call_id": "call_xyz",
            "content": "result text",
            "name": "get_weather"  # Optional in OpenAI format
        }

        Snowflake tool result format:
        {
            "role": "user",
            "content_list": [{
                "type": "tool_results",
                "tool_results": {
                    "tool_use_id": "call_xyz",
                    "name": "get_weather",  # Optional but recommended
                    "content": [{"type": "text", "text": "result text"}]
                }
            }]
        }

        OpenAI assistant with tool_calls format:
        {
            "role": "assistant",
            "content": "text",
            "tool_calls": [{
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{...}"}
            }]
        }

        Snowflake assistant with tool_use format:
        {
            "role": "assistant",
            "content_list": [
                {"type": "text", "text": "text"},
                {
                    "type": "tool_use",
                    "tool_use": {
                        "tool_use_id": "call_xyz",
                        "name": "get_weather",
                        "input": {...}
                    }
                }
            ]
        }
        """
        transformed_messages = []
        # Track tool call IDs to names for matching tool results
        tool_call_map = {}

        for message in messages:
            # Convert to dict for easier access, but preserve original if not transforming
            if isinstance(message, dict):
                msg_dict = message
            else:
                # Convert Pydantic models or other objects to dict
                msg_dict = dict(message) if hasattr(message, '__dict__') else message

            # Handle tool result messages (role="tool")
            if msg_dict.get("role") == "tool":
                tool_call_id = msg_dict.get("tool_call_id")
                content = msg_dict.get("content", "")
                tool_name = msg_dict.get("name")  # OpenAI optionally includes this

                # If name not in message, try to find it from our tracking map
                if not tool_name and tool_call_id in tool_call_map:
                    tool_name = tool_call_map[tool_call_id]

                # Build tool_results object
                tool_results = {
                    "tool_use_id": tool_call_id,
                    "content": [{"type": "text", "text": str(content)}],
                }

                # Add name if available
                if tool_name:
                    tool_results["name"] = tool_name

                # Transform to Snowflake format with content_list
                # Snowflake requires both 'content' and 'content_list' fields
                transformed_message = {
                    "role": "user",
                    "content": "",  # Required even when using content_list
                    "content_list": [
                        {
                            "type": "tool_results",
                            "tool_results": tool_results,
                        }
                    ],
                }
                transformed_messages.append(transformed_message)

            # Handle assistant messages with tool_calls
            elif msg_dict.get("role") == "assistant" and msg_dict.get("tool_calls"):
                content_list = []

                # Transform tool_calls to tool_use format
                for tool_call in msg_dict.get("tool_calls", []):
                    if tool_call.get("type") == "function":
                        function_data = tool_call.get("function", {})
                        tool_call_id = tool_call.get("id")
                        tool_name = function_data.get("name")
                        arguments_str = function_data.get("arguments", "{}")

                        # Track tool call ID to name mapping for later tool result messages
                        if tool_call_id and tool_name:
                            tool_call_map[tool_call_id] = tool_name

                        # Parse arguments if it's a string
                        try:
                            arguments = (
                                json.loads(arguments_str)
                                if isinstance(arguments_str, str)
                                else arguments_str
                            )
                        except json.JSONDecodeError:
                            arguments = {}

                        content_list.append(
                            {
                                "type": "tool_use",
                                "tool_use": {
                                    "tool_use_id": tool_call_id,
                                    "name": tool_name,
                                    "input": arguments,
                                },
                            }
                        )

                # Snowflake requires both 'content' and 'content_list' fields
                # Text content goes in 'content', tool_use goes in 'content_list'
                content_text = msg_dict.get("content") or ""

                transformed_message = {
                    "role": "assistant",
                    "content": content_text,
                    "content_list": content_list,
                }
                transformed_messages.append(transformed_message)

            else:
                # Pass through other messages, but ensure they have at least a content field
                # Snowflake requires either 'content' or 'content_list' to be present
                if isinstance(message, dict):
                    msg_to_append = message.copy()
                else:
                    # Convert Pydantic models or other objects to dict
                    msg_to_append = dict(message) if hasattr(message, '__dict__') else dict(message)

                # Ensure content field exists (Snowflake requires content OR content_list)
                # Snowflake rejects messages with None content and no content_list
                content_value = msg_to_append.get("content")
                has_content_list = "content_list" in msg_to_append and msg_to_append.get("content_list")

                # If content is None or missing AND there's no content_list, set empty string
                if content_value is None and not has_content_list:
                    msg_to_append["content"] = ""  # Empty string satisfies Snowflake's requirement

                transformed_messages.append(msg_to_append)

        return transformed_messages

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        from litellm._logging import verbose_logger

        stream: bool = optional_params.pop("stream", None) or False
        extra_body = optional_params.pop("extra_body", {})

        ## MESSAGE TRANSFORMATION
        # Transform messages to handle tool results and assistant tool_calls
        transformed_messages = self._transform_messages(messages)

        # TRACE 2.5: Log transformed messages to diagnose content/contentList issue
        verbose_logger.debug("=" * 80)
        verbose_logger.debug("TRACE 2.5: SNOWFLAKE MESSAGE TRANSFORMATION")
        verbose_logger.debug(f"Snowflake: Transformed {len(transformed_messages)} messages")
        for i, msg in enumerate(transformed_messages):
            verbose_logger.debug(f"Message {i}: role={msg.get('role')}, has_content={'content' in msg}, content_value={repr(msg.get('content'))}, has_content_list={'content_list' in msg}")
        verbose_logger.debug("=" * 80)

        ## TOOL CALLING
        # Transform tools from OpenAI format to Snowflake's tool_spec format
        tools = optional_params.pop("tools", None)
        if tools:
            verbose_logger.debug(f"Snowflake: Received {len(tools)} tools in OpenAI format")
            transformed_tools = self._transform_tools(tools)
            optional_params["tools"] = transformed_tools
            verbose_logger.debug(f"Snowflake: Transformed tools to Snowflake format")
        else:
            verbose_logger.debug("Snowflake: No tools in request")

        # Transform tool_choice from OpenAI format to Snowflake's tool name array format
        tool_choice = optional_params.pop("tool_choice", None)
        if tool_choice:
            verbose_logger.debug(f"Snowflake: tool_choice before transform: {tool_choice}")
            # Pass transformed_tools to get tool names for Snowflake format
            tool_names = []
            if tools:
                tool_names = [t.get("tool_spec", {}).get("name") for t in transformed_tools if t.get("tool_spec", {}).get("name")]
            transformed_tool_choice = self._transform_tool_choice(tool_choice, tool_names)
            optional_params["tool_choice"] = transformed_tool_choice
            verbose_logger.debug(f"Snowflake: tool_choice after transform: {transformed_tool_choice}")
        else:
            verbose_logger.debug("Snowflake: No tool_choice in request")

        request_data = {
            "model": model,
            "messages": transformed_messages,
            "stream": stream,
            **optional_params,
            **extra_body,
        }

        # TRACE 3: Log final request to Snowflake API
        import json
        verbose_logger.debug("=" * 80)
        verbose_logger.debug("TRACE 3: FINAL REQUEST TO SNOWFLAKE API")
        verbose_logger.debug(f"Messages: {len(transformed_messages)}, tools={'tools' in request_data}")
        verbose_logger.debug(f"Full request body:\n{json.dumps(request_data, indent=2, default=str)}")
        verbose_logger.debug("=" * 80)

        return request_data

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """
        Return custom streaming handler for Snowflake that handles missing 'created' field
        and transforms Claude-format tool_use to OpenAI-format tool_calls.

        Some Snowflake models (like claude-sonnet-4-5) may not include the 'created' field
        in their streaming responses, and return tool calls in Claude's format.
        """
        return SnowflakeStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
