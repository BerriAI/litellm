"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Union,
    cast,
)

import httpx

from litellm import ModelResponse
from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.bridges.completion_transformation import (
    CompletionTransformationBridge,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

    from litellm import LiteLLMLoggingObj, ModelResponse
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
    from litellm.types.llms.openai import (
        ALL_RESPONSES_API_TOOL_PARAMS,
        AllMessageValues,
        ChatCompletionThinkingBlock,
        OpenAIMessageContentListBlock,
    )


class LiteLLMResponsesTransformationHandler(CompletionTransformationBridge):
    """
    Handler for transforming /chat/completions api requests to litellm.responses requests
    """

    def __init__(self):
        pass

    def transform_request(
        self,
        model: str,
        messages: List["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

        input_items: List[Any] = []

        # Process messages to extract system instructions and convert to responses format
        instructions: Optional[str] = None
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            tool_call_id = msg.get("tool_call_id")

            if role == "system":
                # Extract system message as instructions
                if isinstance(content, str):
                    instructions = content
                else:
                    raise ValueError(f"System message must be a string: {content}")
            elif role == "tool":
                # Convert tool message to function call output format
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": content,
                    }
                )
            elif role == "assistant" and tool_calls:
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
                        "content": self._convert_content_to_responses_format(content),
                    }
                )

        # Build responses API request using the reverse transformation logic
        responses_api_request = ResponsesAPIOptionalRequestParams()

        # Set instructions if we found a system message
        if instructions:
            responses_api_request["instructions"] = instructions

        # Map optional parameters
        for key, value in optional_params.items():
            if value is None:
                continue
            if key in ("max_tokens", "max_completion_tokens"):
                responses_api_request["max_output_tokens"] = value
            elif key == "tools" and value is not None:
                # Convert chat completion tools to responses API tools format
                responses_api_request[
                    "tools"
                ] = self._convert_tools_to_responses_format(
                    cast(List[Dict[str, Any]], value)
                )
            elif key in [
                "temperature",
                "top_p",
                "tool_choice",
                "parallel_tool_calls",
                "user",
                "stream",
            ]:
                responses_api_request[key] = value
            elif key == "metadata":
                responses_api_request["metadata"] = value
            elif key == "previous_response_id":
                # Support for responses API session management
                responses_api_request["previous_response_id"] = value

        # Get stream parameter from litellm_params if not in optional_params
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

        request_data = {
            "model": api_model,
            "input": input_items,
        }

        verbose_logger.debug(
            f"Chat provider: Final request model={api_model}, input_items={len(input_items)}"
        )

        # Add non-None values from responses_api_request
        for key, value in responses_api_request.items():
            if value is not None:
                if key == "instructions" and instructions:
                    request_data["instructions"] = instructions
                else:
                    request_data[key] = value

        return request_data

    def transform_response(
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
        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseOutputMessage,
            ResponseReasoningItem,
        )

        from litellm.responses.utils import ResponseAPILoggingUtils
        from litellm.types.llms.openai import ResponsesAPIResponse
        from litellm.types.responses.main import (
            GenericResponseOutputItem,
            OutputFunctionToolCall,
        )
        from litellm.types.utils import Choices, Message

        if not isinstance(raw_response, ResponsesAPIResponse):
            raise ValueError(f"Unexpected response type: {type(raw_response)}")

        choices: List[Choices] = []
        index = 0
        for item in raw_response.output:
            if isinstance(item, ResponseReasoningItem):
                pass  # ignore for now.
            elif isinstance(item, ResponseOutputMessage):
                for content in item.content:
                    response_text = getattr(content, "text", "")
                    msg = Message(
                        role=item.role, content=response_text if response_text else ""
                    )

                    choices.append(
                        Choices(message=msg, finish_reason="stop", index=index)
                    )
                    index += 1
            elif isinstance(item, ResponseFunctionToolCall):
                msg = Message(
                    content=None,
                    tool_calls=[
                        {
                            "id": item.call_id,
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                            "type": "function",
                        }
                    ],
                )

                choices.append(
                    Choices(message=msg, finish_reason="tool_calls", index=index)
                )
                index += 1
            elif isinstance(item, GenericResponseOutputItem):
                raise ValueError("GenericResponseOutputItem not supported")
            elif isinstance(item, OutputFunctionToolCall):
                # function/tool calls pass through as-is
                raise ValueError("Function calling not supported yet.")
            else:
                raise ValueError(f"Unknown item type: {item}")

        setattr(model_response, "choices", choices)

        setattr(
            model_response,
            "usage",
            ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                raw_response.usage
            ),
        )
        return model_response

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], "ModelResponse"],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> BaseModelResponseIterator:
        return super().get_model_response_iterator(
            streaming_response, sync_stream, json_mode
        )

    def _convert_content_to_responses_format(
        self,
        content: Union[
            str,
            Iterable[
                Union["OpenAIMessageContentListBlock", "ChatCompletionThinkingBlock"]
            ],
        ],
    ) -> List[Dict[str, Any]]:
        """Convert chat completion content to responses API format"""
        verbose_logger.debug(
            f"Chat provider: Converting content to responses format - input type: {type(content)}"
        )

        if isinstance(content, str):
            result = [{"type": "input_text", "text": content}]
            verbose_logger.debug(f"Chat provider: String content -> {result}")
            return result
        elif isinstance(content, list):
            result = []
            for i, item in enumerate(content):
                verbose_logger.debug(
                    f"Chat provider: Processing content item {i}: {type(item)} = {item}"
                )
                if isinstance(item, str):
                    converted = {"type": "input_text", "text": item}
                    result.append(converted)
                    verbose_logger.debug(f"Chat provider:   -> {converted}")
                elif isinstance(item, dict):
                    # Handle multimodal content
                    original_type = item.get("type")
                    if original_type == "text":
                        converted = {"type": "input_text", "text": item.get("text", "")}
                        result.append(converted)
                        verbose_logger.debug(f"Chat provider:   text -> {converted}")
                    elif original_type == "image_url":
                        # Map to responses API image format
                        converted = {
                            "type": "input_image",
                            "image_url": item.get("image_url", {}),
                        }
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
                            converted = {
                                "type": "input_text",
                                "text": str(item.get("text", item)),
                            }
                            result.append(converted)
                            verbose_logger.debug(
                                f"Chat provider:   unknown({original_type}) -> {converted}"
                            )
            verbose_logger.debug(f"Chat provider: Final converted content: {result}")
            return result
        else:
            result = [{"type": "input_text", "text": str(content)}]
            verbose_logger.debug(f"Chat provider: Other content type -> {result}")
            return result

    def _convert_tools_to_responses_format(
        self, tools: List[Dict[str, Any]]
    ) -> List["ALL_RESPONSES_API_TOOL_PARAMS"]:
        """Convert chat completion tools to responses API tools format"""
        responses_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                function = tool.get("function", {})
                responses_tools.append(
                    {
                        "type": "function",
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {}),
                        "strict": function.get("strict", False),
                    }
                )
        return cast(List["ALL_RESPONSES_API_TOOL_PARAMS"], responses_tools)

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
