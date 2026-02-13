"""
OpenAI Chat Completions Message Handler for Unified Guardrails

This module provides a class-based handler for OpenAI-format chat completions.
The class methods can be overridden for custom behavior.

Pattern Overview:
-----------------
1. Extract text content from messages/responses (both string and list formats)
2. Create async tasks to apply guardrails to each text segment
3. Track mappings to know where each response belongs
4. Apply guardrail responses back to the original structure

This pattern can be replicated for other message formats (e.g., Anthropic).
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.main import stream_chunk_builder
from litellm.types.llms.openai import ChatCompletionToolParam
from litellm.types.utils import (
    Choices,
    GenericGuardrailAPIInputs,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail


class OpenAIChatCompletionsHandler(BaseTranslation):
    """
    Handler for processing OpenAI chat completions messages with guardrails.

    This class provides methods to:
    1. Process input messages (pre-call hook)
    2. Process output responses (post-call hook)

    Methods can be overridden to customize behavior for different message formats.
    """

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

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[ChatCompletionToolParam] = []
        text_task_mappings: List[Tuple[int, Optional[int]]] = []
        tool_call_task_mappings: List[Tuple[int, int]] = []
        # text_task_mappings: Track (message_index, content_index) for each text
        # content_index is None for string content, int for list content
        # tool_call_task_mappings: Track (message_index, tool_call_index) for each tool call

        # Step 1: Extract all text content, images, and tool calls
        for msg_idx, message in enumerate(messages):
            self._extract_inputs(
                message=message,
                msg_idx=msg_idx,
                texts_to_check=texts_to_check,
                images_to_check=images_to_check,
                tool_calls_to_check=tool_calls_to_check,
                text_task_mappings=text_task_mappings,
                tool_call_task_mappings=tool_call_task_mappings,
            )

        # Step 2: Apply guardrail to all texts and tool calls in batch
        if texts_to_check or tool_calls_to_check:
            inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            if tool_calls_to_check:
                inputs["tool_calls"] = tool_calls_to_check  # type: ignore
            if messages:
                inputs[
                    "structured_messages"
                ] = messages  # pass the openai /chat/completions messages to the guardrail, as-is
            # Pass tools (function definitions) to the guardrail
            tools = data.get("tools")
            if tools:
                inputs["tools"] = tools
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
            guardrailed_tool_calls = guardrailed_inputs.get("tool_calls", [])

            # Step 3: Map guardrail responses back to original message structure
            if guardrailed_texts and texts_to_check:
                await self._apply_guardrail_responses_to_input_texts(
                    messages=messages,
                    responses=guardrailed_texts,
                    task_mappings=text_task_mappings,
                )

            # Step 4: Apply guardrailed tool calls back to messages
            if guardrailed_tool_calls:
                # Note: The guardrail may modify tool_calls_to_check in place
                # or we may need to handle returned tool calls differently
                await self._apply_guardrail_responses_to_input_tool_calls(
                    messages=messages,
                    tool_calls=guardrailed_tool_calls,  # type: ignore
                    task_mappings=tool_call_task_mappings,
                )

        verbose_proxy_logger.debug(
            "OpenAI Chat Completions: Processed input messages: %s", messages
        )

        return data

    def _extract_inputs(
        self,
        message: Dict[str, Any],
        msg_idx: int,
        texts_to_check: List[str],
        images_to_check: List[str],
        tool_calls_to_check: List[ChatCompletionToolParam],
        text_task_mappings: List[Tuple[int, Optional[int]]],
        tool_call_task_mappings: List[Tuple[int, int]],
    ) -> None:
        """
        Extract text content, images, and tool calls from a message.

        Override this method to customize text/image/tool call extraction logic.
        """
        content = message.get("content", None)
        if content is not None:
            if isinstance(content, str):
                # Simple string content
                texts_to_check.append(content)
                text_task_mappings.append((msg_idx, None))

            elif isinstance(content, list):
                # List content (e.g., multimodal with text and images)
                for content_idx, content_item in enumerate(content):
                    # Extract text
                    text_str = content_item.get("text", None)
                    if text_str is not None:
                        texts_to_check.append(text_str)
                        text_task_mappings.append((msg_idx, int(content_idx)))

                    # Extract images (image_url)
                    if content_item.get("type") == "image_url":
                        image_url = content_item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url")
                            if url:
                                images_to_check.append(url)
                        elif isinstance(image_url, str):
                            images_to_check.append(image_url)

        # Extract tool calls (typically in assistant messages)
        tool_calls = message.get("tool_calls", None)
        if tool_calls is not None and isinstance(tool_calls, list):
            for tool_call_idx, tool_call in enumerate(tool_calls):
                if isinstance(tool_call, dict):
                    # Add the full tool call object to the list
                    tool_calls_to_check.append(cast(ChatCompletionToolParam, tool_call))
                    tool_call_task_mappings.append((msg_idx, int(tool_call_idx)))

    async def _apply_guardrail_responses_to_input_texts(
        self,
        messages: List[Dict[str, Any]],
        responses: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail responses back to input message text content.

        Override this method to customize how text responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            msg_idx = cast(int, mapping[0])
            content_idx_optional = cast(Optional[int], mapping[1])

            # Handle content
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

    async def _apply_guardrail_responses_to_input_tool_calls(
        self,
        messages: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        task_mappings: List[Tuple[int, int]],
    ) -> None:
        """
        Apply guardrailed tool calls back to input messages.

        The guardrail may have modified the tool_calls list in place,
        so we apply the modified tool calls back to the original messages.

        Override this method to customize how tool call responses are applied.
        """
        for task_idx, (msg_idx, tool_call_idx) in enumerate(task_mappings):
            if task_idx < len(tool_calls):
                guardrailed_tool_call = tool_calls[task_idx]
                message_tool_calls = messages[msg_idx].get("tool_calls", None)
                if message_tool_calls is not None and isinstance(
                    message_tool_calls, list
                ):
                    if tool_call_idx < len(message_tool_calls):
                        # Replace the tool call with the guardrailed version
                        message_tool_calls[tool_call_idx] = guardrailed_tool_call

    async def process_output_response(
        self,
        response: "ModelResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process output response by applying guardrails to text content.

        Args:
            response: LiteLLM ModelResponse object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - String content: choice.message.content = "text here"
            - List content: choice.message.content = [{"type": "text", "text": "text here"}, ...]
        """

        # Step 0: Check if response has any text content to process
        if not self._has_text_content(response):
            verbose_proxy_logger.warning(
                "OpenAI Chat Completions: No text content in response, skipping guardrail"
            )
            return response

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[Dict[str, Any]] = []
        text_task_mappings: List[Tuple[int, Optional[int]]] = []
        tool_call_task_mappings: List[Tuple[int, int]] = []
        # text_task_mappings: Track (choice_index, content_index) for each text
        # content_index is None for string content, int for list content
        # tool_call_task_mappings: Track (choice_index, tool_call_index) for each tool call

        # Step 1: Extract all text content, images, and tool calls from response choices
        for choice_idx, choice in enumerate(response.choices):
            self._extract_output_text_images_and_tool_calls(
                choice=choice,
                choice_idx=choice_idx,
                texts_to_check=texts_to_check,
                images_to_check=images_to_check,
                tool_calls_to_check=tool_calls_to_check,
                text_task_mappings=text_task_mappings,
                tool_call_task_mappings=tool_call_task_mappings,
            )

        # Step 2: Apply guardrail to all texts and tool calls in batch
        if texts_to_check or tool_calls_to_check:
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
            if tool_calls_to_check:
                inputs["tool_calls"] = tool_calls_to_check  # type: ignore
            # Include model information from the response if available
            if hasattr(response, "model") and response.model:
                inputs["model"] = response.model

            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts = guardrailed_inputs.get("texts", [])

            # Step 3: Map guardrail responses back to original response structure
            if guardrailed_texts and texts_to_check:
                await self._apply_guardrail_responses_to_output_texts(
                    response=response,
                    responses=guardrailed_texts,
                    task_mappings=text_task_mappings,
                )

            # Step 4: Apply guardrailed tool calls back to response
            if tool_calls_to_check:
                await self._apply_guardrail_responses_to_output_tool_calls(
                    response=response,
                    tool_calls=tool_calls_to_check,
                    task_mappings=tool_call_task_mappings,
                )

        verbose_proxy_logger.debug(
            "OpenAI Chat Completions: Processed output response: %s", response
        )

        return response

    async def process_output_streaming_response(
        self,
        responses_so_far: List["ModelResponseStream"],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> List["ModelResponseStream"]:
        """
        Process output streaming responses by applying guardrails to text content.

        Args:
            responses_so_far: List of LiteLLM ModelResponseStream objects
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified list of responses with guardrail applied to content

        Response Format Support:
            - String content: choice.message.content = "text here"
            - List content: choice.message.content = [{"type": "text", "text": "text here"}, ...]
        """
        # check if the stream has ended
        has_stream_ended = False
        for chunk in responses_so_far:
            if chunk.choices and chunk.choices[0].finish_reason is not None:
                has_stream_ended = True
                break

        if has_stream_ended:
            # convert to model response
            model_response = cast(
                ModelResponse,
                stream_chunk_builder(
                    chunks=responses_so_far, logging_obj=litellm_logging_obj
                ),
            )
            # run process_output_response
            await self.process_output_response(
                response=model_response,
                guardrail_to_apply=guardrail_to_apply,
                litellm_logging_obj=litellm_logging_obj,
                user_api_key_dict=user_api_key_dict,
            )

            return responses_so_far

        # Step 0: Check if any response has text content to process
        has_any_text_content = False
        for response in responses_so_far:
            if self._has_text_content(response):
                has_any_text_content = True
                break

        if not has_any_text_content:
            verbose_proxy_logger.warning(
                "OpenAI Chat Completions: No text content in streaming responses, skipping guardrail"
            )
            return responses_so_far

        # Step 1: Combine all streaming chunks into complete text per choice
        # For streaming, we need to concatenate all delta.content across all chunks
        # Key: (choice_idx, content_idx), Value: combined text
        combined_texts = self._combine_streaming_texts(responses_so_far)

        # Step 2: Create lists for guardrail processing
        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        task_mappings: List[Tuple[int, Optional[int]]] = []
        # Track (choice_index, content_index) for each combined text

        for (map_choice_idx, map_content_idx), combined_text in combined_texts.items():
            texts_to_check.append(combined_text)
            task_mappings.append((map_choice_idx, map_content_idx))

        # Step 3: Apply guardrail to all combined texts in batch
        if texts_to_check:
            # Create a request_data dict with response info and user API key metadata
            request_data: dict = {"responses": responses_so_far}

            # Add user API key metadata with prefixed keys
            user_metadata = self.transform_user_api_key_dict_to_metadata(
                user_api_key_dict
            )
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata

            inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            # Include model information from the first response if available
            if (
                responses_so_far
                and hasattr(responses_so_far[0], "model")
                and responses_so_far[0].model
            ):
                inputs["model"] = responses_so_far[0].model
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts = guardrailed_inputs.get("texts", [])

            # Step 4: Apply guardrailed text back to all streaming chunks
            # For each choice, replace the combined text across all chunks
            await self._apply_guardrail_responses_to_output_streaming(
                responses=responses_so_far,
                guardrailed_texts=guardrailed_texts,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug(
            "OpenAI Chat Completions: Processed output streaming responses: %s",
            responses_so_far,
        )

        return responses_so_far

    def _combine_streaming_texts(
        self, responses_so_far: List["ModelResponseStream"]
    ) -> Dict[Tuple[int, Optional[int]], str]:
        """
        Combine all streaming chunks into complete text per choice.

        For streaming, we need to concatenate all delta.content across all chunks.

        Args:
            responses_so_far: List of LiteLLM ModelResponseStream objects

        Returns:
            Dict mapping (choice_idx, content_idx) to combined text string
        """
        combined_texts: Dict[Tuple[int, Optional[int]], str] = {}

        for response_idx, response in enumerate(responses_so_far):
            for choice_idx, choice in enumerate(response.choices):
                if isinstance(choice, litellm.StreamingChoices):
                    content = choice.delta.content
                elif isinstance(choice, litellm.Choices):
                    content = choice.message.content
                else:
                    continue

                if content is None:
                    continue

                if isinstance(content, str):
                    # String content - accumulate for this choice
                    str_key: Tuple[int, Optional[int]] = (choice_idx, None)
                    if str_key not in combined_texts:
                        combined_texts[str_key] = ""
                    combined_texts[str_key] += content

                elif isinstance(content, list):
                    # List content - accumulate for each content item
                    for content_idx, content_item in enumerate(content):
                        text_str = content_item.get("text")
                        if text_str:
                            list_key: Tuple[int, Optional[int]] = (
                                choice_idx,
                                content_idx,
                            )
                            if list_key not in combined_texts:
                                combined_texts[list_key] = ""
                            combined_texts[list_key] += text_str

        return combined_texts

    def _has_text_content(
        self, response: Union["ModelResponse", "ModelResponseStream"]
    ) -> bool:
        """
        Check if response has any text content or tool calls to process.

        Override this method to customize text content detection.
        """
        from litellm.types.utils import ModelResponse, ModelResponseStream

        if isinstance(response, ModelResponse):
            for choice in response.choices:
                if isinstance(choice, litellm.Choices):
                    # Check for text content
                    if choice.message.content and isinstance(
                        choice.message.content, str
                    ):
                        return True
                    # Check for tool calls
                    if choice.message.tool_calls and isinstance(
                        choice.message.tool_calls, list
                    ):
                        if len(choice.message.tool_calls) > 0:
                            return True
        elif isinstance(response, ModelResponseStream):
            for choice in response.choices:
                if isinstance(choice, litellm.StreamingChoices):
                    # Check for text content
                    if choice.delta.content and isinstance(choice.delta.content, str):
                        return True
                    # Check for tool calls
                    if choice.delta.tool_calls and isinstance(
                        choice.delta.tool_calls, list
                    ):
                        if len(choice.delta.tool_calls) > 0:
                            return True
        return False

    def _extract_output_text_images_and_tool_calls(
        self,
        choice: Union[Choices, StreamingChoices],
        choice_idx: int,
        texts_to_check: List[str],
        images_to_check: List[str],
        tool_calls_to_check: List[Dict[str, Any]],
        text_task_mappings: List[Tuple[int, Optional[int]]],
        tool_call_task_mappings: List[Tuple[int, int]],
    ) -> None:
        """
        Extract text content, images, and tool calls from a response choice.

        Override this method to customize text/image/tool call extraction logic.
        """
        verbose_proxy_logger.debug(
            "OpenAI Chat Completions: Processing choice: %s", choice
        )

        # Determine content source and tool calls based on choice type
        content = None
        tool_calls: Optional[List[Any]] = None
        if isinstance(choice, litellm.Choices):
            content = choice.message.content
            tool_calls = choice.message.tool_calls
        elif isinstance(choice, litellm.StreamingChoices):
            content = choice.delta.content
            tool_calls = choice.delta.tool_calls
        else:
            # Unknown choice type, skip processing
            return

        # Process content if it exists
        if content and isinstance(content, str):
            # Simple string content
            texts_to_check.append(content)
            text_task_mappings.append((choice_idx, None))

        elif content and isinstance(content, list):
            # List content (e.g., multimodal response)
            for content_idx, content_item in enumerate(content):
                # Extract text
                content_text = content_item.get("text")
                if content_text:
                    texts_to_check.append(content_text)
                    text_task_mappings.append((choice_idx, int(content_idx)))

                # Extract images
                if content_item.get("type") == "image_url":
                    image_url = content_item.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url")
                        if url:
                            images_to_check.append(url)

        # Process tool calls if they exist
        if tool_calls is not None and isinstance(tool_calls, list):
            for tool_call_idx, tool_call in enumerate(tool_calls):
                # Convert tool call to dict format for guardrail processing
                tool_call_dict = self._convert_tool_call_to_dict(tool_call)
                if tool_call_dict:
                    tool_calls_to_check.append(tool_call_dict)
                    tool_call_task_mappings.append((choice_idx, int(tool_call_idx)))

    def _convert_tool_call_to_dict(
        self, tool_call: Union[Dict[str, Any], Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert a tool call object to dictionary format.

        Tool calls can be either dict or object depending on the type.
        """
        if isinstance(tool_call, dict):
            return tool_call
        elif hasattr(tool_call, "id") and hasattr(tool_call, "function"):
            # Convert object to dict
            function = tool_call.function
            function_dict = {}
            if hasattr(function, "name"):
                function_dict["name"] = function.name
            if hasattr(function, "arguments"):
                function_dict["arguments"] = function.arguments

            tool_call_dict = {
                "id": tool_call.id if hasattr(tool_call, "id") else None,
                "type": tool_call.type if hasattr(tool_call, "type") else "function",
                "function": function_dict,
            }
            return tool_call_dict
        return None

    async def _apply_guardrail_responses_to_output_texts(
        self,
        response: "ModelResponse",
        responses: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail text responses back to output response.

        Override this method to customize how text responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            choice_idx = cast(int, mapping[0])
            content_idx_optional = cast(Optional[int], mapping[1])

            choice = cast(Choices, response.choices[choice_idx])

            # Handle content
            content = choice.message.content
            if content is None:
                continue

            if isinstance(content, str) and content_idx_optional is None:
                # Replace string content with guardrail response
                choice.message.content = guardrail_response

            elif isinstance(content, list) and content_idx_optional is not None:
                # Replace specific text item in list content
                choice.message.content[content_idx_optional]["text"] = guardrail_response  # type: ignore

    async def _apply_guardrail_responses_to_output_tool_calls(
        self,
        response: "ModelResponse",
        tool_calls: List[Dict[str, Any]],
        task_mappings: List[Tuple[int, int]],
    ) -> None:
        """
        Apply guardrailed tool calls back to output response.

        The guardrail may have modified the tool_calls list in place,
        so we apply the modified tool calls back to the original response.

        Override this method to customize how tool call responses are applied.
        """
        for task_idx, (choice_idx, tool_call_idx) in enumerate(task_mappings):
            if task_idx < len(tool_calls):
                guardrailed_tool_call = tool_calls[task_idx]
                choice = cast(Choices, response.choices[choice_idx])
                choice_tool_calls = choice.message.tool_calls

                if choice_tool_calls is not None and isinstance(
                    choice_tool_calls, list
                ):
                    if tool_call_idx < len(choice_tool_calls):
                        # Update the tool call with guardrailed version
                        existing_tool_call = choice_tool_calls[tool_call_idx]
                        # Update object attributes (output responses always have typed objects)
                        if "function" in guardrailed_tool_call:
                            func_dict = guardrailed_tool_call["function"]
                            if "arguments" in func_dict:
                                existing_tool_call.function.arguments = func_dict[
                                    "arguments"
                                ]
                            if "name" in func_dict:
                                existing_tool_call.function.name = func_dict["name"]

    async def _apply_guardrail_responses_to_output_streaming(
        self,
        responses: List["ModelResponseStream"],
        guardrailed_texts: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail responses back to output streaming responses.

        For streaming responses, the guardrailed text (which is the combined text from all chunks)
        is placed in the first chunk, and subsequent chunks are cleared.

        Args:
            responses: List of ModelResponseStream objects to modify
            guardrailed_texts: List of guardrailed text responses (combined from all chunks)
            task_mappings: List of tuples (choice_idx, content_idx)

        Override this method to customize how responses are applied to streaming responses.
        """
        # Build a mapping of what guardrailed text to use for each (choice_idx, content_idx)
        guardrail_map: Dict[Tuple[int, Optional[int]], str] = {}
        for task_idx, guardrail_response in enumerate(guardrailed_texts):
            mapping = task_mappings[task_idx]
            choice_idx = cast(int, mapping[0])
            content_idx_optional = cast(Optional[int], mapping[1])
            guardrail_map[(choice_idx, content_idx_optional)] = guardrail_response

        # Track which choices we've already set the guardrailed text for
        # Key: (choice_idx, content_idx), Value: boolean (True if already set)
        already_set: Dict[Tuple[int, Optional[int]], bool] = {}

        # Iterate through all responses and update content
        for response_idx, response in enumerate(responses):
            for choice_idx_in_response, choice in enumerate(response.choices):
                if isinstance(choice, litellm.StreamingChoices):
                    content = choice.delta.content
                elif isinstance(choice, litellm.Choices):
                    content = choice.message.content
                else:
                    continue

                if content is None:
                    continue

                if isinstance(content, str):
                    # String content
                    str_key: Tuple[int, Optional[int]] = (choice_idx_in_response, None)
                    if str_key in guardrail_map:
                        if str_key not in already_set:
                            # First chunk - set the complete guardrailed text
                            if isinstance(choice, litellm.StreamingChoices):
                                choice.delta.content = guardrail_map[str_key]
                            elif isinstance(choice, litellm.Choices):
                                choice.message.content = guardrail_map[str_key]
                            already_set[str_key] = True
                        else:
                            # Subsequent chunks - clear the content
                            if isinstance(choice, litellm.StreamingChoices):
                                choice.delta.content = ""
                            elif isinstance(choice, litellm.Choices):
                                choice.message.content = ""

                elif isinstance(content, list):
                    # List content - handle each content item
                    for content_idx, content_item in enumerate(content):
                        if "text" in content_item:
                            list_key: Tuple[int, Optional[int]] = (
                                choice_idx_in_response,
                                content_idx,
                            )
                            if list_key in guardrail_map:
                                if list_key not in already_set:
                                    # First chunk - set the complete guardrailed text
                                    content_item["text"] = guardrail_map[list_key]
                                    already_set[list_key] = True
                                else:
                                    # Subsequent chunks - clear the text
                                    content_item["text"] = ""
