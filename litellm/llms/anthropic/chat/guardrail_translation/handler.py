"""
Anthropic Message Handler for Unified Guardrails

This module provides a class-based handler for Anthropic-format messages.
The class methods can be overridden for custom behavior.

Pattern Overview:
-----------------
1. Extract text content from messages/responses (both string and list formats)
2. Create async tasks to apply guardrails to each text segment
3. Track mappings to know where each response belongs
4. Apply guardrail responses back to the original structure
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.core_helpers import remove_items_at_indices
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.llms.base_llm.guardrail_translation.base_translation import (
    GUARDRAIL_DELETED_KEY,
    BaseTranslation,
)
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from litellm.types.llms.anthropic import (
    AllAnthropicToolsValues,
    AnthropicMessagesRequest,
)
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolParam,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    GenericGuardrailAPIInputs,
    ModelResponse,
)

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.anthropic_messages.anthropic_response import (
        AnthropicMessagesResponse,
    )


class AnthropicMessagesHandler(BaseTranslation):
    """
    Handler for processing Anthropic messages with guardrails.

    This class provides methods to:
    1. Process input messages (pre-call hook)
    2. Process output responses (post-call hook)

    Methods can be overridden to customize behavior for different message formats.
    """

    def __init__(self):
        super().__init__()
        self.adapter = LiteLLMAnthropicMessagesAdapter()

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Any:
        """
        Process input messages by applying guardrails to text content.
        """
        messages = data.get("messages")
        if messages is None:
            return data

        (
            chat_completion_compatible_request,
            _tool_name_mapping,
        ) = LiteLLMAnthropicMessagesAdapter().translate_anthropic_to_openai(
            # Use a shallow copy to avoid mutating request data (pop on litellm_metadata).
            anthropic_message_request=cast(AnthropicMessagesRequest, data.copy())
        )

        structured_messages = chat_completion_compatible_request.get("messages", [])

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tools_to_check: List[
            ChatCompletionToolParam
        ] = chat_completion_compatible_request.get("tools", [])
        task_mappings: List[Tuple[int, Optional[int]]] = []
        # Track (message_index, content_index) for each text
        # content_index is None for string content, int for list content

        # Step 1: Extract all text content and images
        for msg_idx, message in enumerate(messages):
            self._extract_input_text_and_images(
                message=message,
                msg_idx=msg_idx,
                texts_to_check=texts_to_check,
                images_to_check=images_to_check,
                task_mappings=task_mappings,
            )

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check:
            inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            if tools_to_check:
                inputs["tools"] = tools_to_check
            if structured_messages:
                inputs["structured_messages"] = structured_messages
            # Include model information if available
            model = data.get("model")
            if model:
                inputs["model"] = model
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts = guardrailed_inputs.get("texts", [])
            guardrailed_tools = guardrailed_inputs.get("tools")
            if guardrailed_tools is not None:
                # Convert tools back from OpenAI format to Anthropic format
                anthropic_config = AnthropicConfig()
                anthropic_tools: List[AllAnthropicToolsValues] = []
                for tool in guardrailed_tools:
                    converted_tool, mcp_server = anthropic_config._map_tool_helper(tool)
                    if converted_tool is not None:
                        anthropic_tools.append(converted_tool)
                    # Note: MCP servers are handled separately in the main transformation
                data["tools"] = anthropic_tools

            # Step 3: Map guardrail responses back to original message structure
            await self._apply_guardrail_responses_to_input(
                messages=messages,
                responses=guardrailed_texts,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug(
            "Anthropic Messages: Processed input messages: %s", messages
        )

        return data

    def extract_request_tool_names(self, data: dict) -> List[str]:
        """Extract tool names from Anthropic messages request (tools[].name)."""
        names: List[str] = []
        for tool in data.get("tools") or []:
            if isinstance(tool, dict) and tool.get("name"):
                names.append(str(tool["name"]))
        return names

    def _extract_input_text_and_images(
        self,
        message: Dict[str, Any],
        msg_idx: int,
        texts_to_check: List[str],
        images_to_check: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Extract text content and images from a message.

        Override this method to customize text/image extraction logic.
        """
        content = message.get("content", None)
        tools = message.get("tools", None)
        if content is None and tools is None:
            return

        ## CHECK FOR TEXT + IMAGES
        if content is not None and isinstance(content, str):
            # Simple string content
            texts_to_check.append(content)
            task_mappings.append((msg_idx, None))

        elif content is not None and isinstance(content, list):
            # List content (e.g., multimodal with text and images)
            for content_idx, content_item in enumerate(content):
                # Extract text
                text_str = content_item.get("text", None)
                if text_str is not None:
                    texts_to_check.append(text_str)
                    task_mappings.append((msg_idx, int(content_idx)))

                # Extract images
                if content_item.get("type") == "image":
                    source = content_item.get("source", {})
                    if isinstance(source, dict):
                        # Could be base64 or url
                        data = source.get("data")
                        if data:
                            images_to_check.append(data)

    def _extract_input_tools(
        self,
        tools: List[Dict[str, Any]],
        tools_to_check: List[ChatCompletionToolParam],
    ) -> None:
        """
        Extract tools from a message.
        """
        ## CHECK FOR TOOLS
        if tools is not None and isinstance(tools, list):
            # TRANSFORM ANTHROPIC TOOLS TO OPENAI TOOLS
            openai_tools = self.adapter.translate_anthropic_tools_to_openai(
                tools=cast(List[AllAnthropicToolsValues], tools)
            )
            tools_to_check.extend(openai_tools)  # type: ignore

    async def _apply_guardrail_responses_to_input(
        self,
        messages: List[Dict[str, Any]],
        responses: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail responses back to input messages.

        Override this method to customize how responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            msg_idx = cast(int, mapping[0])
            content_idx_optional = cast(Optional[int], mapping[1])

            content = messages[msg_idx].get("content", None)
            if content is None:
                continue

            if isinstance(content, str) and content_idx_optional is None:
                # Replace string content with guardrail response
                messages[msg_idx]["content"] = guardrail_response

            elif isinstance(content, list) and content_idx_optional is not None:
                # Replace specific text item in list content
                messages[msg_idx]["content"][content_idx_optional][
                    "text"
                ] = guardrail_response

    async def process_output_response(
        self,
        response: "AnthropicMessagesResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process output response by applying guardrails to text content and tool calls.

        Args:
            response: Anthropic MessagesResponse object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - List content: response.content = [
                {"type": "text", "text": "text here"},
                {"type": "tool_use", "id": "...", "name": "...", "input": {...}},
                ...
            ]
        """
        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[ChatCompletionToolCallChunk] = []
        task_mappings: List[
            Tuple[int, Optional[int]]
        ] = []  # (content_idx, None) for each text
        tool_call_task_mappings: List[int] = []  # content_idx for each tool call

        response_content = self._get_response_content(response) or []
        if not response_content:
            return response

        # Step 1: Extract all text content and tool calls from response
        for content_idx, content_block in enumerate(response_content):
            block_dict = self._content_block_to_dict(content_block)
            if block_dict is None:
                continue

            block_type = block_dict.get("type")
            if block_type in ["text", "tool_use"]:
                prev_tool_call_count = len(tool_calls_to_check)
                self._extract_output_text_and_images(
                    content_block=block_dict,
                    content_idx=content_idx,
                    texts_to_check=texts_to_check,
                    images_to_check=images_to_check,
                    task_mappings=task_mappings,
                    tool_calls_to_check=tool_calls_to_check,
                )
                # Track content_idx for any newly added tool calls
                for _ in range(len(tool_calls_to_check) - prev_tool_call_count):
                    tool_call_task_mappings.append(content_idx)

        # Gate tool calls behind the guardrail_handles_tool_calls flag
        handles_tool_calls = guardrail_to_apply.guardrail_handles_tool_calls
        effective_tool_calls = tool_calls_to_check if handles_tool_calls else []
        if tool_calls_to_check and not handles_tool_calls:
            verbose_proxy_logger.debug(
                "Anthropic Messages: Skipping %d tool call(s) — "
                "guardrail_handles_tool_calls is False for '%s'",
                len(tool_calls_to_check),
                guardrail_to_apply.guardrail_name,
            )

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check or effective_tool_calls:
            # Create a request_data dict with response info and user API key metadata
            request_data: dict = {"response": response}

            # Add user API key metadata with prefixed keys
            user_metadata = self.transform_user_api_key_dict_to_metadata(
                user_api_key_dict
            )
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata

            inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            if effective_tool_calls:
                inputs["tool_calls"] = effective_tool_calls
            # Include model information from the response if available
            response_model = None
            if isinstance(response, dict):
                response_model = response.get("model")
            elif hasattr(response, "model"):
                response_model = getattr(response, "model", None)
            if response_model:
                inputs["model"] = response_model

            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts = guardrailed_inputs.get("texts", [])

            # Step 3: Map guardrail responses back to original response structure
            if guardrailed_texts:
                await self._apply_guardrail_responses_to_output(
                    response=response,
                    responses=guardrailed_texts,
                    task_mappings=task_mappings,
                )

            # Step 4: Apply guardrailed tool calls back to response
            # Use guardrailed results if returned, otherwise fall back to originals.
            # Note: `is not None` (not `or`) so an empty list from the guardrail
            # correctly signals "all tool calls deleted" rather than falling back.
            guardrailed_tool_calls = guardrailed_inputs.get("tool_calls")
            if effective_tool_calls:
                resolved = (
                    guardrailed_tool_calls
                    if guardrailed_tool_calls is not None
                    else effective_tool_calls
                )
                self._apply_guardrail_responses_to_output_tool_calls(
                    response=response,
                    tool_calls=resolved,
                    task_mappings=tool_call_task_mappings,
                )

        verbose_proxy_logger.debug(
            "Anthropic Messages: Processed output response: %s", response
        )

        return response

    async def process_output_streaming_response(
        self,
        responses_so_far: List[Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> List[Any]:
        """
        Process output streaming response by applying guardrails to text content.

        Get the string so far, check the apply guardrail to the string so far, and return the list of responses so far.
        """
        has_ended = self._check_streaming_has_ended(responses_so_far)
        if has_ended:
            # build the model response from the responses_so_far
            built_response = (
                AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
                    all_chunks=responses_so_far,
                    litellm_logging_obj=cast("LiteLLMLoggingObj", litellm_logging_obj),
                    model="",
                )
            )

            # Check if model_response is valid and has choices before accessing
            if (
                built_response is not None
                and hasattr(built_response, "choices")
                and built_response.choices
            ):
                model_response = cast(ModelResponse, built_response)
                first_choice = cast(Choices, model_response.choices[0])
                tool_calls_list = cast(
                    Optional[List[ChatCompletionMessageToolCall]],
                    first_choice.message.tool_calls,
                )
                string_so_far = first_choice.message.content
                guardrail_inputs = GenericGuardrailAPIInputs()
                if string_so_far:
                    guardrail_inputs["texts"] = [string_so_far]
                if tool_calls_list:
                    guardrail_inputs["tool_calls"] = tool_calls_list

                _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                    # allow rejecting the response, if invalid
                    inputs=guardrail_inputs,
                    request_data={},
                    input_type="response",
                    logging_obj=litellm_logging_obj,
                )
            else:
                verbose_proxy_logger.debug(
                    "Skipping output guardrail - model response has no choices"
                )
            return responses_so_far

        string_so_far = self.get_streaming_string_so_far(responses_so_far)
        _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(  # allow rejecting the response, if invalid
            inputs={"texts": [string_so_far]},
            request_data={},
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        return responses_so_far

    def get_streaming_string_so_far(self, responses_so_far: List[Any]) -> str:
        """
        Parse streaming responses and extract accumulated text content.

        Handles two formats:
        1. Raw bytes in SSE (Server-Sent Events) format from Anthropic API
        2. Parsed dict objects (for backwards compatibility)

        SSE format example:
            b'event: content_block_delta\\ndata: {"type":"content_block_delta","index":0,"delta":{
            "type":"text_delta","text":" curious"}}\\n\\n'

        Dict format example:
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "text_delta",
                    "text": " curious"
                }
            }
        """
        text_so_far = ""
        for response in responses_so_far:
            # Handle raw bytes in SSE format
            if isinstance(response, bytes):
                text_so_far += self._extract_text_from_sse(response)
            # Handle already-parsed dict format
            elif isinstance(response, dict):
                delta = response.get("delta") if response.get("delta") else None
                if delta and delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        text_so_far += text
        return text_so_far

    def _extract_text_from_sse(self, sse_bytes: bytes) -> str:
        """
        Extract text content from Server-Sent Events (SSE) format.

        Args:
            sse_bytes: Raw bytes in SSE format

        Returns:
            Accumulated text from all content_block_delta events
        """
        text = ""
        try:
            # Decode bytes to string
            sse_string = sse_bytes.decode("utf-8")

            # Split by double newline to get individual events
            events = sse_string.split("\n\n")

            for event in events:
                if not event.strip():
                    continue

                # Parse event lines
                lines = event.strip().split("\n")
                event_type = None
                data_line = None

                for line in lines:
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_line = line[5:].strip()

                # Only process content_block_delta events
                if event_type == "content_block_delta" and data_line:
                    try:
                        data = json.loads(data_line)
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text += delta.get("text", "")
                    except json.JSONDecodeError:
                        verbose_proxy_logger.warning(
                            f"Failed to parse JSON from SSE data: {data_line}"
                        )

        except Exception as e:
            verbose_proxy_logger.error(f"Error extracting text from SSE: {e}")

        return text

    def _check_streaming_has_ended(self, responses_so_far: List[Any]) -> bool:
        """
        Check if streaming response has ended by looking for non-null stop_reason.

        Handles two formats:
        1. Raw bytes in SSE (Server-Sent Events) format from Anthropic API
        2. Parsed dict objects (for backwards compatibility)

        SSE format example:
            b'event: message_delta\\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use",
            "stop_sequence":null},...}\\n\\n'

        Dict format example:
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use",
                    "stop_sequence": null
                }
            }

        Returns:
            True if stop_reason is set to a non-null value, indicating stream has ended
        """
        for response in responses_so_far:
            # Handle raw bytes in SSE format
            if isinstance(response, bytes):
                try:
                    # Decode bytes to string
                    sse_string = response.decode("utf-8")

                    # Split by double newline to get individual events
                    events = sse_string.split("\n\n")

                    for event in events:
                        if not event.strip():
                            continue

                        # Parse event lines
                        lines = event.strip().split("\n")
                        event_type = None
                        data_line = None

                        for line in lines:
                            if line.startswith("event:"):
                                event_type = line[6:].strip()
                            elif line.startswith("data:"):
                                data_line = line[5:].strip()

                        # Check for message_delta event with stop_reason
                        if event_type == "message_delta" and data_line:
                            try:
                                data = json.loads(data_line)
                                delta = data.get("delta", {})
                                stop_reason = delta.get("stop_reason")
                                if stop_reason is not None:
                                    return True
                            except json.JSONDecodeError:
                                verbose_proxy_logger.warning(
                                    f"Failed to parse JSON from SSE data: {data_line}"
                                )

                except Exception as e:
                    verbose_proxy_logger.error(
                        f"Error checking streaming end in SSE: {e}"
                    )

            # Handle already-parsed dict format
            elif isinstance(response, dict):
                if response.get("type") == "message_delta":
                    delta = response.get("delta", {})
                    stop_reason = delta.get("stop_reason")
                    if stop_reason is not None:
                        return True

        return False

    def _has_text_content(self, response: "AnthropicMessagesResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        if isinstance(response, dict):
            response_content = response.get("content", [])
        else:
            response_content = getattr(response, "content", None) or []

        if not response_content:
            return False
        for content_block in response_content:
            # Check if this is a text block by checking the 'type' field
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                content_text = content_block.get("text")
                if content_text and isinstance(content_text, str):
                    return True
        return False

    def _extract_output_text_and_images(
        self,
        content_block: Dict[str, Any],
        content_idx: int,
        texts_to_check: List[str],
        images_to_check: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
        tool_calls_to_check: Optional[List[ChatCompletionToolCallChunk]] = None,
    ) -> None:
        """
        Extract text content, images, and tool calls from a response content block.

        Override this method to customize text/image/tool extraction logic.
        """
        content_type = content_block.get("type")

        # Extract text content
        if content_type == "text":
            content_text = content_block.get("text")
            if content_text and isinstance(content_text, str):
                # Simple string content
                texts_to_check.append(content_text)
                task_mappings.append((content_idx, None))

        # Extract tool calls
        elif content_type == "tool_use":
            tool_call = AnthropicConfig.convert_tool_use_to_openai_format(
                anthropic_tool_content=content_block,
                index=content_idx,
            )
            if tool_calls_to_check is None:
                tool_calls_to_check = []
            tool_calls_to_check.append(tool_call)

    @staticmethod
    def _content_block_to_dict(content_block: Any) -> Optional[Dict[str, Any]]:
        """Convert a content block (dict or Pydantic object) to a dict, or None."""
        if isinstance(content_block, dict):
            return cast(Dict[str, Any], content_block)
        elif hasattr(content_block, "type"):
            if hasattr(content_block, "model_dump"):
                return content_block.model_dump()
            return {
                "type": getattr(content_block, "type", None),
                "text": getattr(content_block, "text", None),
            }
        return None

    @staticmethod
    def _get_response_content(
        response: "AnthropicMessagesResponse",
    ) -> Optional[List[Any]]:
        """Extract the content list from a dict or Pydantic response, or None if unavailable."""
        if isinstance(response, dict):
            return response.get("content", []) or []
        elif hasattr(response, "content"):
            return getattr(response, "content", None) or []
        return None

    async def _apply_guardrail_responses_to_output(
        self,
        response: "AnthropicMessagesResponse",
        responses: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail responses back to output response.

        Mapped texts replace existing content blocks. Extra texts beyond
        task_mappings length are appended as new text content blocks — this
        allows guardrails to inject replacement text into tool-call-only
        responses that originally had no text.
        """
        response_content = self._get_response_content(response)
        if response_content is None:
            return

        # Apply mapped texts back to their original locations
        for task_idx in range(min(len(responses), len(task_mappings))):
            guardrail_response = responses[task_idx]
            mapping = task_mappings[task_idx]
            content_idx = cast(int, mapping[0])

            if not response_content or content_idx >= len(response_content):
                continue

            content_block = response_content[content_idx]

            # Verify it's a text block and update the text field
            if isinstance(content_block, dict):
                if content_block.get("type") == "text":
                    content_block["text"] = guardrail_response
            elif (
                hasattr(content_block, "type")
                and getattr(content_block, "type", None) == "text"
            ):
                if hasattr(content_block, "text"):
                    content_block.text = guardrail_response

        # Append extra texts as new content blocks
        if len(responses) > len(task_mappings):
            for extra_text in responses[len(task_mappings) :]:
                response_content.append({"type": "text", "text": extra_text})

    @staticmethod
    def _apply_tool_call_to_content_block(
        content_block: Any,
        guardrailed_tool_call: Any,
    ) -> None:
        """Update a single content block's input/name from a guardrailed tool call."""
        func = None
        if (
            isinstance(guardrailed_tool_call, dict)
            and "function" in guardrailed_tool_call
        ):
            func = guardrailed_tool_call["function"]
        elif hasattr(guardrailed_tool_call, "function"):
            func = getattr(guardrailed_tool_call, "function", None)

        if func is None:
            return

        new_arguments = (
            func.get("arguments")
            if isinstance(func, dict)
            else getattr(func, "arguments", None)
        )
        new_name = (
            func.get("name") if isinstance(func, dict) else getattr(func, "name", None)
        )

        if isinstance(content_block, dict):
            if new_arguments is not None:
                try:
                    content_block["input"] = (
                        json.loads(new_arguments)
                        if isinstance(new_arguments, str)
                        else new_arguments
                    )
                except json.JSONDecodeError:
                    content_block["input"] = new_arguments
            if new_name is not None:
                content_block["name"] = new_name
        else:
            if new_arguments is not None:
                try:
                    parsed = (
                        json.loads(new_arguments)
                        if isinstance(new_arguments, str)
                        else new_arguments
                    )
                except json.JSONDecodeError:
                    parsed = new_arguments
                if hasattr(content_block, "input"):
                    content_block.input = parsed
            if new_name is not None and hasattr(content_block, "name"):
                content_block.name = new_name

    def _apply_guardrail_responses_to_output_tool_calls(
        self,
        response: "AnthropicMessagesResponse",
        tool_calls: List[Any],
        task_mappings: List[int],
    ) -> None:
        """
        Apply guardrailed tool calls back to output response content blocks.

        Two-pass approach similar to the OpenAI Chat pattern:
        1. Modify pass: update tool_use content blocks with modified arguments/name
        2. Delete pass: remove content blocks marked with guardrail_deleted
        """
        response_content = self._get_response_content(response)
        if not response_content:
            return

        # Pass 1: Modify non-deleted tool calls
        indices_to_delete: List[int] = []
        for task_idx, content_idx in enumerate(task_mappings):
            if task_idx >= len(tool_calls):
                continue

            guardrailed_tool_call = tool_calls[task_idx]

            # Check for deletion flag
            is_deleted = False
            if isinstance(guardrailed_tool_call, dict):
                is_deleted = guardrailed_tool_call.get(GUARDRAIL_DELETED_KEY) is True
            elif hasattr(guardrailed_tool_call, GUARDRAIL_DELETED_KEY):
                is_deleted = (
                    getattr(guardrailed_tool_call, GUARDRAIL_DELETED_KEY) is True
                )

            if is_deleted:
                indices_to_delete.append(content_idx)
                continue

            # Strip internal metadata before field-level writeback
            if isinstance(guardrailed_tool_call, dict):
                guardrailed_tool_call.pop(GUARDRAIL_DELETED_KEY, None)

            if content_idx < len(response_content):
                self._apply_tool_call_to_content_block(
                    response_content[content_idx], guardrailed_tool_call
                )

        # Pass 2: Delete marked content blocks
        if indices_to_delete:
            remove_items_at_indices(response_content, indices_to_delete)

            # Update stop_reason if no tool_use blocks remain
            has_tool_use = any(
                (b.get("type") if isinstance(b, dict) else getattr(b, "type", None))
                == "tool_use"
                for b in response_content
            )
            if not has_tool_use:
                if isinstance(response, dict):
                    if response.get("stop_reason") == "tool_use":
                        response["stop_reason"] = "end_turn"
                elif hasattr(response, "stop_reason"):
                    if getattr(response, "stop_reason", None) == "tool_use":
                        response.stop_reason = "end_turn"
