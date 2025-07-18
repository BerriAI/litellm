"""
Handler for transforming /chat/completions api requests to litellm.responses requests
"""

import json
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
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

from litellm import ModelResponse
from litellm._logging import verbose_logger
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.llms.base_llm.bridges.completion_transformation import (
    CompletionTransformationBridge,
)
from litellm.types.llms.openai import Reasoning

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
    from litellm.types.utils import GenericStreamingChunk, ModelResponseStream


class LiteLLMResponsesTransformationHandler(CompletionTransformationBridge):
    """
    Handler for transforming /chat/completions api requests to litellm.responses requests
    """

    def __init__(self):
        pass

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
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": content,
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
        from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams

        (
            input_items,
            instructions,
        ) = self.convert_chat_completion_messages_to_responses_api(messages)

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
                responses_api_request["tools"] = (
                    self._convert_tools_to_responses_format(
                        cast(List[Dict[str, Any]], value)
                    )
                )
            elif key in ResponsesAPIOptionalRequestParams.__annotations__.keys():
                responses_api_request[key] = value  # type: ignore
            elif key in ("metadata"):
                responses_api_request["metadata"] = value
            elif key in ("previous_response_id"):
                responses_api_request["previous_response_id"] = value
            elif key == "reasoning_effort":
                responses_api_request["reasoning"] = self._map_reasoning_effort(value)

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

        from litellm.types.utils import CallTypes

        setattr(litellm_logging_obj, "call_type", CallTypes.responses.value)

        request_data = {
            "model": api_model,
            "input": input_items,
            "litellm_logging_obj": litellm_logging_obj,
            **litellm_params,
            "client": client,
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
        from litellm.types.utils import Choices, Message

        if not isinstance(raw_response, ResponsesAPIResponse):
            raise ValueError(f"Unexpected response type: {type(raw_response)}")

        if raw_response.error is not None:
            raise ValueError(f"Error in response: {raw_response.error}")

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
            else:
                pass  # don't fail request if item in list is not supported

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
        if role == "user" or role == "system":
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
        responses_tools = []
        for tool in tools:
            responses_tools.append(tool)
        return cast(List["ALL_RESPONSES_API_TOOL_PARAMS"], responses_tools)

    def _map_reasoning_effort(self, reasoning_effort: str) -> Optional[Reasoning]:
        if reasoning_effort == "high":
            return Reasoning(effort="high", summary="detailed")
        elif reasoning_effort == "medium":
            # docs say "summary": "concise" is also an option, but it was rejected in practice, so defaulting "auto"
            return Reasoning(effort="medium", summary="auto")
        elif reasoning_effort == "low":
            return Reasoning(effort="low", summary="auto")
        return None

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

    def chunk_parser(
        self, chunk: dict
    ) -> Union["GenericStreamingChunk", "ModelResponseStream"]:
        # Transform responses API streaming chunk to chat completion format
        from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
        from litellm.types.utils import (
            ChatCompletionToolCallChunk,
            GenericStreamingChunk,
        )

        verbose_logger.debug(
            f"Chat provider: transform_streaming_response called with chunk: {chunk}"
        )
        parsed_chunk = chunk

        if not parsed_chunk:
            raise ValueError("Chat provider: Empty parsed_chunk")

        if not isinstance(parsed_chunk, dict):
            raise ValueError(f"Chat provider: Invalid chunk type {type(parsed_chunk)}")

        # Handle different event types from responses API
        event_type = parsed_chunk.get("type")
        verbose_logger.debug(f"Chat provider: Processing event type: {event_type}")

        if event_type == "response.created":
            # Initial response creation event
            verbose_logger.debug(f"Chat provider: response.created -> {chunk}")
            return GenericStreamingChunk(
                text="", tool_use=None, is_finished=False, finish_reason="", usage=None
            )
        elif event_type == "response.output_item.added":
            # New output item added
            output_item = parsed_chunk.get("item", {})
            if output_item.get("type") == "function_call":
                return GenericStreamingChunk(
                    text="",
                    tool_use=ChatCompletionToolCallChunk(
                        id=output_item.get("call_id"),
                        index=0,
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=parsed_chunk.get("name", None),
                            arguments=parsed_chunk.get("arguments", ""),
                        ),
                    ),
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                )
            elif output_item.get("type") == "message":
                pass
            elif output_item.get("type") == "reasoning":
                pass
            else:
                raise ValueError(f"Chat provider: Invalid output_item  {output_item}")
        elif event_type == "response.function_call_arguments.delta":
            content_part: Optional[str] = parsed_chunk.get("delta", None)
            if content_part:
                return GenericStreamingChunk(
                    text="",
                    tool_use=ChatCompletionToolCallChunk(
                        id=None,
                        index=0,
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=None, arguments=content_part
                        ),
                    ),
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                )
            else:
                raise ValueError(
                    f"Chat provider: Invalid function argument delta {parsed_chunk}"
                )
        elif event_type == "response.output_item.done":
            # New output item added
            output_item = parsed_chunk.get("item", {})
            if output_item.get("type") == "function_call":
                return GenericStreamingChunk(
                    text="",
                    tool_use=ChatCompletionToolCallChunk(
                        id=output_item.get("call_id"),
                        index=0,
                        type="function",
                        function=ChatCompletionToolCallFunctionChunk(
                            name=parsed_chunk.get("name", None),
                            arguments="",  # responses API sends everything again, we don't
                        ),
                    ),
                    is_finished=True,
                    finish_reason="tool_calls",
                    usage=None,
                )
            elif output_item.get("type") == "message":
                return GenericStreamingChunk(
                    finish_reason="stop", is_finished=True, usage=None, text=""
                )
            elif output_item.get("type") == "reasoning":
                pass
            else:
                raise ValueError(f"Chat provider: Invalid output_item  {output_item}")

        elif event_type == "response.output_text.delta":
            # Content part added to output
            content_part = parsed_chunk.get("delta", None)
            if content_part is not None:
                return GenericStreamingChunk(
                    text=content_part,
                    tool_use=None,
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                )
            else:
                raise ValueError(f"Chat provider: Invalid text delta {parsed_chunk}")
        elif event_type == "response.reasoning_summary_text.delta":
            content_part = parsed_chunk.get("delta", None)
            if content_part:
                from litellm.types.utils import (
                    Delta,
                    ModelResponseStream,
                    StreamingChoices,
                )

                return ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=cast(int, parsed_chunk.get("summary_index")),
                            delta=Delta(reasoning_content=content_part),
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
        return GenericStreamingChunk(
            text="", tool_use=None, is_finished=False, finish_reason="", usage=None
        )
