"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""

import json
import os
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

from openai.types.responses.tool_param import FunctionToolParam
from pydantic import BaseModel

import litellm
from litellm import ModelResponse
from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.bridges.completion_transformation import (
    CompletionTransformationBridge,
)
from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionToolParamFunctionChunk,
    Reasoning,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

if TYPE_CHECKING:
    from openai.types.responses import ResponseInputImageParam
    from pydantic import BaseModel

    from litellm import LiteLLMLoggingObj, ModelResponse
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
    from litellm.types.llms.openai import (
        ALL_RESPONSES_API_TOOL_PARAMS,
        AllMessageValues,
        ChatCompletionImageObject,
        ChatCompletionThinkingBlock,
        OpenAIMessageContentListBlock,
    )


class LiteLLMResponsesTransformationHandler(CompletionTransformationBridge):
    """
    Handler for transforming /chat/completions api requests to litellm.responses requests
    """

    def __init__(self):
        pass

    def _handle_raw_dict_response_item(
        self, item: Dict[str, Any], index: int
    ) -> Tuple[Optional[Any], int]:
        """
        Handle raw dict response items from Responses API (e.g., GPT-5 Codex format).

        Args:
            item: Raw dict response item with 'type' field
            index: Current choice index

        Returns:
            Tuple of (Choice object or None, updated index)
        """
        from litellm.types.utils import Choices, Message

        item_type = item.get("type")

        # Ignore reasoning items for now
        if item_type == "reasoning":
            return None, index

        # Handle message items with output_text content
        if item_type == "message":
            content_list = item.get("content", [])
            for content_item in content_list:
                if isinstance(content_item, dict):
                    content_type = content_item.get("type")
                    if content_type == "output_text":
                        response_text = content_item.get("text", "")
                        # Extract annotations from content if present
                        annotations = LiteLLMResponsesTransformationHandler._convert_annotations_to_chat_format(
                            content_item.get("annotations", None)
                        )
                        msg = Message(
                            role=item.get("role", "assistant"),
                            content=response_text if response_text else "",
                            annotations=annotations,
                        )
                        choice = Choices(message=msg, finish_reason="stop", index=index)
                        return choice, index + 1

        # Handle function_call items (e.g., from GPT-5 Codex format)
        if item_type == "function_call":
            # Extract provider_specific_fields if present and pass through as-is
            provider_specific_fields = item.get("provider_specific_fields")
            if provider_specific_fields and not isinstance(
                provider_specific_fields, dict
            ):
                provider_specific_fields = (
                    dict(provider_specific_fields)
                    if hasattr(provider_specific_fields, "__dict__")
                    else {}
                )

            tool_call_dict = {
                "id": item.get("call_id") or item.get("id", ""),
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", ""),
                },
                "type": "function",
            }

            # Pass through provider_specific_fields as-is if present
            if provider_specific_fields:
                tool_call_dict["provider_specific_fields"] = provider_specific_fields
                # Also add to function's provider_specific_fields for consistency
                tool_call_dict["function"][
                    "provider_specific_fields"
                ] = provider_specific_fields

            msg = Message(
                content=None,
                tool_calls=[tool_call_dict],
            )
            choice = Choices(message=msg, finish_reason="tool_calls", index=index)
            return choice, index + 1

        # Unknown or unsupported type
        return None, index

    def convert_chat_completion_messages_to_responses_api(
        self, messages: List["AllMessageValues"]
    ) -> Tuple[List[Any], Optional[str]]:
        input_items: List[Any] = []
        instructions: Optional[str] = None

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "system":
                # Extract system message as instructions
                if isinstance(content, str):
                    if instructions:
                        # Concatenate multiple system prompts with a space
                        instructions = f"{instructions} {content}"
                    else:
                        instructions = content
                else:
                    input_items.append(
                        {
                            "type": "message",
                            "role": role,
                            "content": self._convert_content_to_responses_format(
                                content, role  # type: ignore
                            ),
                        }
                    )
            elif role == "tool":
                # Convert tool message to function call output format
                # The Responses API expects 'output' to be a list with input_text/input_image types
                # Using list format for consistency across text and multimodal content
                tool_output: List[Dict[str, Any]]
                if content is None:
                    tool_output = []
                elif isinstance(content, str):
                    # Convert string to list with input_text
                    tool_output = [{"type": "input_text", "text": content}]
                elif isinstance(content, list):
                    # Transform list content to Responses API format
                    tool_output = self._convert_content_to_responses_format(
                        content, "user"  # Use "user" role to get input_* types
                    )
                else:
                    # Fallback: convert unexpected types to input_text
                    tool_output = [{"type": "input_text", "text": str(content)}]
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": tool_output,
                    }
                )
            elif role == "assistant" and tool_calls and isinstance(tool_calls, list):
                for tool_call in tool_calls:
                    function = tool_call.get("function")
                    if function:
                        input_tool_call = {
                            "type": "function_call",
                            "call_id": tool_call["id"],
                        }
                        if "name" in function:
                            input_tool_call["name"] = function["name"]
                        if "arguments" in function:
                            input_tool_call["arguments"] = function["arguments"]
                        input_items.append(input_tool_call)
                    else:
                        raise ValueError(f"tool call not supported: {tool_call}")
            elif content is not None:
                # Regular user/assistant message
                input_items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": self._convert_content_to_responses_format(
                            content, cast(str, role)
                        ),
                    }
                )

        return input_items, instructions

    def _map_optional_params_to_responses_api_request(
        self,
        optional_params: dict,
        responses_api_request: "ResponsesAPIOptionalRequestParams",
    ) -> None:
        """Map optional_params into responses_api_request (mutates in place)."""
        for key, value in optional_params.items():
            if value is None:
                continue
            if key in ("max_tokens", "max_completion_tokens"):
                responses_api_request["max_output_tokens"] = value
            elif key == "tools" and value is not None:
                responses_api_request["tools"] = (
                    self._convert_tools_to_responses_format(
                        cast(List[Dict[str, Any]], value)
                    )
                )
            elif key == "response_format":
                text_format = self._transform_response_format_to_text_format(value)
                if text_format:
                    responses_api_request["text"] = text_format  # type: ignore
            elif key in ResponsesAPIOptionalRequestParams.__annotations__.keys():
                responses_api_request[key] = value  # type: ignore
            elif key == "previous_response_id":
                responses_api_request["previous_response_id"] = value
            elif key == "reasoning_effort":
                responses_api_request["reasoning"] = self._map_reasoning_effort(value)
            elif key == "web_search_options":
                self._add_web_search_tool(responses_api_request, value)

    def _build_sanitized_litellm_params(
        self, litellm_params: dict
    ) -> Dict[str, Any]:
        """Build sanitized litellm_params with merged metadata."""
        responses_optional_param_keys = set(
            ResponsesAPIOptionalRequestParams.__annotations__.keys()
        )
        sanitized: Dict[str, Any] = {
            key: value
            for key, value in litellm_params.items()
            if key not in responses_optional_param_keys
        }
        legacy_metadata = litellm_params.get("metadata")
        existing_litellm_metadata = litellm_params.get("litellm_metadata")
        merged_litellm_metadata: Dict[str, Any] = {}
        if isinstance(legacy_metadata, dict):
            merged_litellm_metadata.update(legacy_metadata)
        if isinstance(existing_litellm_metadata, dict):
            merged_litellm_metadata.update(existing_litellm_metadata)
        if merged_litellm_metadata:
            sanitized["litellm_metadata"] = merged_litellm_metadata
        else:
            sanitized.pop("litellm_metadata", None)
        return sanitized

    def _merge_responses_api_request_into_request_data(
        self,
        request_data: Dict[str, Any],
        responses_api_request: "ResponsesAPIOptionalRequestParams",
        instructions: Optional[str],
    ) -> None:
        """Add non-None values from responses_api_request into request_data."""
        for key, value in responses_api_request.items():
            if value is None:
                continue
            if key == "instructions" and instructions:
                request_data["instructions"] = instructions
            elif key == "stream_options" and isinstance(value, dict):
                request_data["stream_options"] = value.get("include_obfuscation")
            elif key == "user" and isinstance(value, str):
                # OpenAI API requires user param to be max 64 chars - truncate if longer
                if len(value) <= 64:
                    request_data["user"] = value
                else:
                    request_data["user"] = value[:64]
            else:
                request_data[key] = value

    def transform_request(
        self,
        model: str,
        messages: List["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
        litellm_logging_obj: "LiteLLMLoggingObj",
        client: Optional[Any] = None,
    ) -> dict:
        (
            input_items,
            instructions,
        ) = self.convert_chat_completion_messages_to_responses_api(messages)

        optional_params = self._extract_extra_body_params(optional_params)

        # Build responses API request using the reverse transformation logic
        responses_api_request = ResponsesAPIOptionalRequestParams()

        # Set instructions if we found a system message
        if instructions:
            responses_api_request["instructions"] = instructions

        self._map_optional_params_to_responses_api_request(
            optional_params, responses_api_request
        )

        stream = optional_params.get("stream") or litellm_params.get("stream", False)
        verbose_logger.debug(f"Chat provider: Stream parameter: {stream}")

        # Ensure stream is properly set in the request
        if stream:
            responses_api_request["stream"] = True

        # Handle session management if previous_response_id is provided
        previous_response_id = optional_params.get("previous_response_id")
        if previous_response_id:
            # Use the existing session handler for responses API
            verbose_logger.debug(
                f"Chat provider: Warning ignoring previous response ID: {previous_response_id}"
            )

        # Convert back to responses API format for the actual request

        api_model = model

        from litellm.types.utils import CallTypes

        setattr(litellm_logging_obj, "call_type", CallTypes.responses.value)

        sanitized_litellm_params = self._build_sanitized_litellm_params(
            litellm_params
        )

        request_data = {
            "model": api_model,
            "input": input_items,
            "litellm_logging_obj": litellm_logging_obj,
            **sanitized_litellm_params,
            "client": client,
        }

        verbose_logger.debug(
            f"Chat provider: Final request model={api_model}, input_items={len(input_items)}"
        )

        self._merge_responses_api_request_into_request_data(
            request_data, responses_api_request, instructions
        )

        if headers:
            request_data["extra_headers"] = headers

        return request_data

    @staticmethod
    def _convert_response_output_to_choices(
        output_items: List[Any],
        handle_raw_dict_callback: Optional[Callable] = None,
    ) -> List[Any]:
        """
        Convert Responses API output items to chat completion choices.

        Args:
            output_items: List of items from ResponsesAPIResponse.output
            handle_raw_dict_callback: Optional callback for handling raw dict items

        Returns:
            List of Choices objects
        """
        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseOutputMessage,
            ResponseReasoningItem,
        )

        from litellm.types.utils import Choices, Message

        choices: List[Choices] = []
        index = 0
        reasoning_content: Optional[str] = None

        # Collect all tool calls to put them in a single choice
        # (Chat Completions API expects all tool calls in one message)
        accumulated_tool_calls: List[Dict[str, Any]] = []
        tool_call_index = 0

        for item in output_items:
            if isinstance(item, ResponseReasoningItem):
                for summary_item in item.summary:
                    response_text = getattr(summary_item, "text", "")
                    reasoning_content = response_text if response_text else ""

            elif isinstance(item, ResponseOutputMessage):
                for content in item.content:
                    response_text = getattr(content, "text", "")
                    # Extract annotations from content if present
                    raw_annotations = getattr(content, "annotations", None)
                    annotations = LiteLLMResponsesTransformationHandler._convert_annotations_to_chat_format(
                        raw_annotations
                    )
                    msg = Message(
                        role=item.role,
                        content=response_text if response_text else "",
                        reasoning_content=reasoning_content,
                        annotations=annotations,
                    )

                    choices.append(
                        Choices(
                            message=msg,
                            finish_reason="stop",
                            index=index,
                        )
                    )

                    reasoning_content = None  # flush reasoning content
                    index += 1

            elif isinstance(item, ResponseFunctionToolCall):
                from litellm.responses.litellm_completion_transformation.transformation import (
                    LiteLLMCompletionResponsesConfig,
                )

                tool_call_dict = LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call(
                    tool_call_item=item,
                    index=tool_call_index,
                )
                accumulated_tool_calls.append(tool_call_dict)
                tool_call_index += 1

            elif isinstance(item, dict) and handle_raw_dict_callback is not None:
                # Handle raw dict responses (e.g., from GPT-5 Codex)
                choice, index = handle_raw_dict_callback(item=item, index=index)
                if choice is not None:
                    choices.append(choice)
            else:
                pass  # don't fail request if item in list is not supported

        # If we accumulated tool calls, create a single choice with all of them
        if accumulated_tool_calls:
            msg = Message(
                content=None,
                tool_calls=accumulated_tool_calls,
                reasoning_content=reasoning_content,
            )
            choices.append(
                Choices(message=msg, finish_reason="tool_calls", index=index)
            )
            reasoning_content = None

        return choices

    def transform_response(  # noqa: PLR0915
        self,
        model: str,
        raw_response: "BaseModel",
        model_response: "ModelResponse",
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict,
        messages: List["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> "ModelResponse":
        """Transform Responses API response to chat completion response"""
        from litellm.responses.utils import ResponseAPILoggingUtils
        from litellm.types.llms.openai import ResponsesAPIResponse

        if not isinstance(raw_response, ResponsesAPIResponse):
            raise ValueError(f"Unexpected response type: {type(raw_response)}")

        if raw_response.error is not None:
            raise ValueError(f"Error in response: {raw_response.error}")

        # Convert response output to choices using the static helper
        choices = self._convert_response_output_to_choices(
            output_items=raw_response.output,
            handle_raw_dict_callback=self._handle_raw_dict_response_item,
        )

        if len(choices) == 0:
            if (
                raw_response.incomplete_details is not None
                and raw_response.incomplete_details.reason is not None
            ):
                raise ValueError(
                    f"{model} unable to complete request: {raw_response.incomplete_details.reason}"
                )
            else:
                raise ValueError(
                    f"Unknown items in responses API response: {raw_response.output}"
                )

        setattr(model_response, "choices", choices)

        model_response.model = model

        setattr(
            model_response,
            "usage",
            ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                raw_response.usage
            ),
        )
        
        # Preserve hidden params from the ResponsesAPIResponse, especially the headers
        # which contain important provider information like x-request-id
        raw_response_hidden_params = getattr(raw_response, "_hidden_params", {})
        if raw_response_hidden_params:
            if not hasattr(model_response, "_hidden_params") or model_response._hidden_params is None:
                model_response._hidden_params = {}
            # Merge the raw_response hidden params with model_response hidden params
            # Preserve existing keys in model_response but add/override with raw_response params
            for key, value in raw_response_hidden_params.items():
                if key == "additional_headers" and key in model_response._hidden_params:
                    # Merge additional_headers to preserve both sets
                    existing_additional_headers = model_response._hidden_params.get("additional_headers", {})
                    merged_headers = {**value, **existing_additional_headers}
                    model_response._hidden_params[key] = merged_headers
                else:
                    model_response._hidden_params[key] = value
        
        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[
            Iterator[str], AsyncIterator[str], "ModelResponse", "BaseModel"
        ],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> BaseModelResponseIterator:
        return OpenAiResponsesToChatCompletionStreamIterator(
            streaming_response, sync_stream, json_mode
        )

    def _convert_content_str_to_input_text(
        self, content: str, role: str
    ) -> Dict[str, Any]:
        if role == "user" or role == "system" or role == "tool":
            return {"type": "input_text", "text": content}
        else:
            return {"type": "output_text", "text": content}

    def _convert_content_to_responses_format_image(
        self, content: "ChatCompletionImageObject", role: str
    ) -> "ResponseInputImageParam":
        from openai.types.responses import ResponseInputImageParam

        content_image_url = content.get("image_url")
        actual_image_url: Optional[str] = None
        detail: Optional[Literal["low", "high", "auto"]] = None

        if isinstance(content_image_url, str):
            actual_image_url = content_image_url
        elif isinstance(content_image_url, dict):
            actual_image_url = content_image_url.get("url")
            detail = cast(
                Optional[Literal["low", "high", "auto"]],
                content_image_url.get("detail"),
            )

        if actual_image_url is None:
            raise ValueError(f"Invalid image URL: {content_image_url}")

        image_param = ResponseInputImageParam(
            image_url=actual_image_url, detail="auto", type="input_image"
        )

        if detail:
            image_param["detail"] = detail

        return image_param

    def _convert_content_to_responses_format(
        self,
        content: Union[
            str,
            Iterable[
                Union["OpenAIMessageContentListBlock", "ChatCompletionThinkingBlock"]
            ],
        ],
        role: str,
    ) -> List[Dict[str, Any]]:
        """Convert chat completion content to responses API format"""
        from litellm.types.llms.openai import ChatCompletionImageObject

        verbose_logger.debug(
            f"Chat provider: Converting content to responses format - input type: {type(content)}"
        )

        if isinstance(content, str):
            result = [self._convert_content_str_to_input_text(content, role)]
            verbose_logger.debug(f"Chat provider: String content -> {result}")
            return result
        elif isinstance(content, list):
            result = []
            for i, item in enumerate(content):
                verbose_logger.debug(
                    f"Chat provider: Processing content item {i}: {type(item)} = {item}"
                )
                if isinstance(item, str):
                    converted = self._convert_content_str_to_input_text(item, role)
                    result.append(converted)
                    verbose_logger.debug(f"Chat provider:   -> {converted}")
                elif isinstance(item, dict):
                    # Handle multimodal content
                    original_type = item.get("type")
                    if original_type == "text":
                        converted = self._convert_content_str_to_input_text(
                            item.get("text", ""), role
                        )
                        result.append(converted)
                        verbose_logger.debug(f"Chat provider:   text -> {converted}")
                    elif original_type == "image_url":
                        # Map to responses API image format
                        converted = cast(
                            dict,
                            self._convert_content_to_responses_format_image(
                                cast(ChatCompletionImageObject, item), role
                            ),
                        )
                        result.append(converted)
                        verbose_logger.debug(
                            f"Chat provider:   image_url -> {converted}"
                        )
                    else:
                        # Try to map other types to responses API format
                        item_type = original_type or "input_text"
                        if item_type == "image":
                            converted = {"type": "input_image", **item}
                            result.append(converted)
                            verbose_logger.debug(
                                f"Chat provider:   image -> {converted}"
                            )
                        elif item_type in [
                            "input_text",
                            "input_image",
                            "output_text",
                            "refusal",
                            "input_file",
                            "computer_screenshot",
                            "summary_text",
                        ]:
                            # Already in responses API format
                            result.append(item)
                            verbose_logger.debug(
                                f"Chat provider:   passthrough -> {item}"
                            )
                        else:
                            # Default to input_text for unknown types
                            converted = self._convert_content_str_to_input_text(
                                str(item.get("text", item)), role
                            )
                            result.append(converted)
                            verbose_logger.debug(
                                f"Chat provider:   unknown({original_type}) -> {converted}"
                            )
            verbose_logger.debug(f"Chat provider: Final converted content: {result}")
            return result
        else:
            result = [self._convert_content_str_to_input_text(str(content), role)]
            verbose_logger.debug(f"Chat provider: Other content type -> {result}")
            return result

    def _convert_tools_to_responses_format(
        self, tools: List[Dict[str, Any]]
    ) -> List["ALL_RESPONSES_API_TOOL_PARAMS"]:
        """Convert chat completion tools to responses API tools format"""
        responses_tools: List["ALL_RESPONSES_API_TOOL_PARAMS"] = []
        for tool in tools:
            # convert function tool from chat completion to responses API format
            if tool.get("type") == "function":
                function_tool = cast(
                    ChatCompletionToolParamFunctionChunk, tool.get("function")
                )
                responses_tools.append(
                    FunctionToolParam(
                        name=function_tool["name"],
                        parameters=function_tool.get("parameters"),
                        strict=function_tool.get("strict"),
                        type="function",
                        description=function_tool.get("description"),
                    )
                )
            else:
                responses_tools.append(tool)  # type: ignore

        return cast(List["ALL_RESPONSES_API_TOOL_PARAMS"], responses_tools)

    def _extract_extra_body_params(self, optional_params: dict):
        """
        Extract extra_body from optional_params and separate supported Responses API params
        from unsupported ones. Supported params are moved to top-level optional_params,
        unsupported params remain in extra_body.
        """
        # Extract extra_body and separate supported params from unsupported ones
        extra_body = optional_params.pop("extra_body", None) or {}
        if not extra_body:
            return optional_params

        supported_responses_api_params = set(
            ResponsesAPIOptionalRequestParams.__annotations__.keys()
        )
        # Also include params we handle specially
        supported_responses_api_params.update(
            {
                "previous_response_id",
                "reasoning_effort",  # We map this to "reasoning"
            }
        )

        # Extract supported params from extra_body and merge into optional_params
        extra_body_copy = extra_body.copy()
        for key, value in extra_body_copy.items():
            if key in supported_responses_api_params:
                # Prefer extra_body value if it exists (may have more complete info like summary in reasoning_effort)
                optional_params[key] = extra_body.pop(key)

        return optional_params

    def _map_reasoning_effort(
        self, reasoning_effort: Union[str, Dict[str, Any]]
    ) -> Optional[Reasoning]:
        # If dict is passed, convert it directly to Reasoning object
        if isinstance(reasoning_effort, dict):
            return Reasoning(**reasoning_effort)  # type: ignore[typeddict-item]

        # Check if auto-summary is enabled via flag or environment variable
        # Priority: litellm.reasoning_auto_summary flag > LITELLM_REASONING_AUTO_SUMMARY env var
        auto_summary_enabled = (
            litellm.reasoning_auto_summary
            or os.getenv("LITELLM_REASONING_AUTO_SUMMARY", "false").lower() == "true"
        )

        # If string is passed, map with optional summary based on flag/env var
        if reasoning_effort == "none":
            return Reasoning(effort="none", summary="detailed") if auto_summary_enabled else Reasoning(effort="none")  # type: ignore
        elif reasoning_effort == "high":
            return Reasoning(effort="high", summary="detailed") if auto_summary_enabled else Reasoning(effort="high")
        elif reasoning_effort == "xhigh":
            return Reasoning(effort="xhigh", summary="detailed") if auto_summary_enabled else Reasoning(effort="xhigh")  # type: ignore[typeddict-item]
        elif reasoning_effort == "medium":
            return Reasoning(effort="medium", summary="detailed") if auto_summary_enabled else Reasoning(effort="medium")
        elif reasoning_effort == "low":
            return Reasoning(effort="low", summary="detailed") if auto_summary_enabled else Reasoning(effort="low")
        elif reasoning_effort == "minimal":
            return Reasoning(effort="minimal", summary="detailed") if auto_summary_enabled else Reasoning(effort="minimal")
        return None

    def _add_web_search_tool(
        self,
        responses_api_request: ResponsesAPIOptionalRequestParams,
        web_search_options: Any,
    ) -> None:
        """
        Add web search tool to responses API request.

        Args:
            responses_api_request: The responses API request dict to modify
            web_search_options: Web search configuration (dict or other value)
        """
        if "tools" not in responses_api_request or responses_api_request["tools"] is None:
            responses_api_request["tools"] = []

        # Get the tools list with proper type narrowing
        tools = responses_api_request["tools"]
        if tools is None:
            tools = []
            responses_api_request["tools"] = tools

        web_search_tool: Dict[str, Any] = {"type": "web_search"}
        if isinstance(web_search_options, dict):
            web_search_tool.update(web_search_options)

        # Cast to Any to match the expected union type for tools list items
        tools.append(cast(Any, web_search_tool))

    def _transform_response_format_to_text_format(
        self, response_format: Union[Dict[str, Any], Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Transform Chat Completion response_format parameter to Responses API text.format parameter.

        Chat Completion response_format structure:
        {
            "type": "json_schema",
            "json_schema": {
                "name": "schema_name",
                "schema": {...},
                "strict": True
            }
        }

        Responses API text parameter structure:
        {
            "format": {
                "type": "json_schema",
                "name": "schema_name",
                "schema": {...},
                "strict": True
            }
        }
        """
        if not response_format:
            return None

        if isinstance(response_format, dict):
            format_type = response_format.get("type")

            if format_type == "json_schema":
                json_schema = response_format.get("json_schema", {})
                return {
                    "format": {
                        "type": "json_schema",
                        "name": json_schema.get("name", "response_schema"),
                        "schema": json_schema.get("schema", {}),
                        "strict": json_schema.get("strict", False),
                    }
                }
            elif format_type == "json_object":
                return {"format": {"type": "json_object"}}
            elif format_type == "text":
                return {"format": {"type": "text"}}

        return None
    
    @staticmethod
    def _convert_annotations_to_chat_format(
        annotations: Optional[List[Any]],
    ) -> Optional[List[ChatCompletionAnnotation]]:
        """
        Convert annotations from Responses API to Chat Completions format.

        Annotations are already in compatible format between both APIs,
        so we just need to convert Pydantic models to dicts.
        """
        if not annotations:
            return None

        result: List[ChatCompletionAnnotation] = []
        for annotation in annotations:
            try:
                # Convert Pydantic models to dicts (handles both v1 and v2)
                if hasattr(annotation, "model_dump"):
                    annotation_dict = annotation.model_dump()
                elif hasattr(annotation, "dict"):
                    annotation_dict = annotation.dict()
                elif isinstance(annotation, dict):
                    annotation_dict = annotation
                else:
                    # Skip unsupported annotation types
                    verbose_logger.debug(f"Skipping unsupported annotation type: {type(annotation)}")
                    continue

                result.append(annotation_dict)  # type: ignore
            except Exception as e:
                # Skip malformed annotations
                verbose_logger.debug(f"Skipping malformed annotation: {annotation}, error: {e}")
                continue

        return result if result else None

    def _map_responses_status_to_finish_reason(self, status: Optional[str]) -> str:
        """Map responses API status to chat completion finish_reason"""
        if not status:
            return "stop"

        status_mapping = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "stop",
            "cancelled": "stop",
        }

        return status_mapping.get(status, "stop")


class OpenAiResponsesToChatCompletionStreamIterator(BaseModelResponseIterator):
    def __init__(
        self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        super().__init__(streaming_response, sync_stream, json_mode)

    def _handle_string_chunk(
        self, str_line: Union[str, "BaseModel"]
    ) -> Union["GenericStreamingChunk", "ModelResponseStream"]:
        from pydantic import BaseModel

        if isinstance(str_line, BaseModel):
            return self.chunk_parser(str_line.model_dump())

        if not str_line or str_line.startswith("event:"):
            # ignore.
            return GenericStreamingChunk(
                text="", tool_use=None, is_finished=False, finish_reason="", usage=None
            )
        index = str_line.find("data:")
        if index != -1:
            str_line = str_line[index + 5 :]

        return self.chunk_parser(json.loads(str_line))

    @staticmethod
    def translate_responses_chunk_to_openai_stream(  # noqa: PLR0915
        parsed_chunk: Union[dict, BaseModel],
    ) -> "ModelResponseStream":
        """
        Translate a Responses API streaming chunk to OpenAI chat completion streaming format.

        Args:
            parsed_chunk: Dict containing the Responses API event chunk

        Returns:
            ModelResponseStream: OpenAI-formatted streaming chunk

        Raises:
            ValueError: If chunk is invalid or missing required fields
        """
        from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
        from litellm.types.utils import (
            ChatCompletionToolCallChunk,
            Delta,
            ModelResponseStream,
            StreamingChoices,
        )

        if not parsed_chunk:
            raise ValueError("Chat provider: Empty parsed_chunk")

        if isinstance(parsed_chunk, BaseModel):
            parsed_chunk = parsed_chunk.model_dump()
        if not isinstance(parsed_chunk, dict):
            raise ValueError(f"Chat provider: Invalid chunk type {type(parsed_chunk)}")

        # Handle different event types from responses API
        event_type = parsed_chunk.get("type")
        if isinstance(event_type, ResponsesAPIStreamEvents):
            event_type = event_type.value
        verbose_logger.debug(f"Chat provider: Processing event type: {event_type}")

        if event_type == "response.created":
            # Initial response creation event
            verbose_logger.debug(f"Chat provider: response.created -> {parsed_chunk}")
            return ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=""),
                        finish_reason=None,
                    )
                ]
            )
        elif event_type == "response.output_item.added":
            # New output item added
            output_item = parsed_chunk.get("item", {})
            if output_item.get("type") == "function_call":
                # Extract provider_specific_fields if present
                provider_specific_fields = output_item.get("provider_specific_fields")
                if provider_specific_fields and not isinstance(
                    provider_specific_fields, dict
                ):
                    provider_specific_fields = (
                        dict(provider_specific_fields)
                        if hasattr(provider_specific_fields, "__dict__")
                        else {}
                    )

                function_chunk = ChatCompletionToolCallFunctionChunk(
                    name=output_item.get("name", None),
                    arguments=parsed_chunk.get("arguments", ""),
                )

                if provider_specific_fields:
                    function_chunk["provider_specific_fields"] = (
                        provider_specific_fields
                    )

                tool_call_chunk = ChatCompletionToolCallChunk(
                    id=output_item.get("call_id"),
                    index=0,
                    type="function",
                    function=function_chunk,
                )

                # Add provider_specific_fields if present
                if provider_specific_fields:
                    tool_call_chunk.provider_specific_fields = provider_specific_fields  # type: ignore

                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(tool_calls=[tool_call_chunk]),
                            finish_reason=None,
                        )
                    ]
                )
        elif event_type == "response.function_call_arguments.delta":
            content_part: Optional[str] = parsed_chunk.get("delta", None)
            if content_part:
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(
                                tool_calls=[
                                    ChatCompletionToolCallChunk(
                                        id=None,
                                        index=0,
                                        type="function",
                                        function=ChatCompletionToolCallFunctionChunk(
                                            name=None, arguments=content_part
                                        ),
                                    )
                                ]
                            ),
                            finish_reason=None,
                        )
                    ]
                )
            else:
                raise ValueError(
                    f"Chat provider: Invalid function argument delta {parsed_chunk}"
                )
        elif event_type == "response.output_item.done":
            # New output item added
            output_item = parsed_chunk.get("item", {})
            if output_item.get("type") == "function_call":
                # Extract provider_specific_fields if present
                provider_specific_fields = output_item.get("provider_specific_fields")
                if provider_specific_fields and not isinstance(
                    provider_specific_fields, dict
                ):
                    provider_specific_fields = (
                        dict(provider_specific_fields)
                        if hasattr(provider_specific_fields, "__dict__")
                        else {}
                    )

                function_chunk = ChatCompletionToolCallFunctionChunk(
                    name=output_item.get("name", None),
                    arguments="",  # responses API sends everything again, we don't
                )

                # Add provider_specific_fields to function if present
                if provider_specific_fields:
                    function_chunk["provider_specific_fields"] = (
                        provider_specific_fields
                    )

                tool_call_chunk = ChatCompletionToolCallChunk(
                    id=output_item.get("call_id"),
                    index=0,
                    type="function",
                    function=function_chunk,
                )

                # Add provider_specific_fields if present
                if provider_specific_fields:
                    tool_call_chunk.provider_specific_fields = provider_specific_fields  # type: ignore

                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(tool_calls=[tool_call_chunk]),
                            finish_reason="tool_calls",
                        )
                    ]
                )
            elif output_item.get("type") == "message":
                # Message completion should NOT emit finish_reason
                # This is the fix for issue #17246 - don't end stream prematurely
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content=""),
                            finish_reason=None,
                        )
                    ]
                )

        elif event_type == "response.output_text.delta":
            # Content part added to output
            content_part = parsed_chunk.get("delta", None)
            if content_part is not None:
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(content=content_part),
                            finish_reason=None,
                        )
                    ]
                )
            else:
                raise ValueError(f"Chat provider: Invalid text delta {parsed_chunk}")
        elif event_type == "response.reasoning_summary_text.delta":
            content_part = parsed_chunk.get("delta", None)
            if content_part:
                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=cast(int, parsed_chunk.get("summary_index")),
                            delta=Delta(reasoning_content=content_part),
                        )
                    ]
                )
        elif event_type == "response.completed":
            # Response is fully complete - now we can signal is_finished=True
            # This ensures we don't prematurely end the stream before tool_calls arrive
            return ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(content=""),
                        finish_reason="stop",
                    )
                ]
            )
        else:
            pass
        # For any unhandled event types, create a minimal valid chunk or skip
        verbose_logger.debug(
            f"Chat provider: Unhandled event type '{event_type}', creating empty chunk"
        )

        # Return a minimal valid chunk for unknown events
        return ModelResponseStream(
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=""),
                    finish_reason=None,
                )
            ]
        )

    def chunk_parser(self, chunk: dict) -> "ModelResponseStream":
        """
        Parse a Responses API streaming chunk and convert to OpenAI format.

        Args:
            chunk: Dict containing the Responses API event chunk

        Returns:
            ModelResponseStream: OpenAI-formatted streaming chunk
        """
        verbose_logger.debug(
            f"Chat provider: transform_streaming_response called with chunk: {chunk}"
        )
        return OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
            chunk
        )
