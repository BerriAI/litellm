"""
OpenAI Responses API Handler for Unified Guardrails

This module provides a class-based handler for OpenAI Responses API format.
The class methods can be overridden for custom behavior.

Pattern Overview:
-----------------
1. Extract text content from input/output (both string and list formats)
2. Create async tasks to apply guardrails to each text segment
3. Track mappings to know where each response belongs
4. Apply guardrail responses back to the original structure

Responses API Format:
---------------------
Input: Union[str, List[Dict]] where each dict has:
  - role: str
  - content: Union[str, List[Dict]] (can have text items)
  - type: str (e.g., "message")

Output: response.output is List[GenericResponseOutputItem] where each has:
  - type: str (e.g., "message")
  - id: str
  - status: str
  - role: str
  - content: List[OutputText] where OutputText has:
    - type: str (e.g., "output_text")
    - text: str
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from openai.types.responses import ResponseFunctionToolCall
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
    OpenAiResponsesToChatCompletionStreamIterator,
)
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolParam,
)
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    OutputFunctionToolCall,
    OutputText,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.llms.openai import ResponseInputParam
    from litellm.types.utils import ResponsesAPIResponse


class OpenAIResponsesHandler(BaseTranslation):
    """
    Handler for processing OpenAI Responses API with guardrails.

    This class provides methods to:
    1. Process input (pre-call hook)
    2. Process output response (post-call hook)

    Methods can be overridden to customize behavior for different message formats.
    """

    async def process_input_messages(
        self,
        data: dict,
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Any:
        """
        Process input by applying guardrails to text content.

        Handles both string input and list of message objects.
        """
        input_data: Optional[Union[str, "ResponseInputParam"]] = data.get("input")
        tools_to_check: List[ChatCompletionToolParam] = []
        if input_data is None:
            return data

        structured_messages = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
                input=input_data,
                responses_api_request=data,
            )
        )

        # Handle simple string input
        if isinstance(input_data, str):
            inputs = GenericGuardrailAPIInputs(texts=[input_data])

            # Extract and transform tools if present

            if "tools" in data and data["tools"]:
                self._extract_and_transform_tools(data["tools"], tools_to_check)
                if tools_to_check:
                    inputs["tools"] = tools_to_check
            if structured_messages:
                inputs["structured_messages"] = structured_messages  # type: ignore

            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )
            guardrailed_texts = guardrailed_inputs.get("texts", [])
            data["input"] = guardrailed_texts[0] if guardrailed_texts else input_data
            verbose_proxy_logger.debug("OpenAI Responses API: Processed string input")
            return data

        # Handle list input (ResponseInputParam)
        if not isinstance(input_data, list):
            return data

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        task_mappings: List[Tuple[int, Optional[int]]] = []
        # Track (message_index, content_index) for each text
        # content_index is None for string content, int for list content

        # Step 1: Extract all text content, images, and tools
        for msg_idx, message in enumerate(input_data):
            self._extract_input_text_and_images(
                message=message,
                msg_idx=msg_idx,
                texts_to_check=texts_to_check,
                images_to_check=images_to_check,
                task_mappings=task_mappings,
            )

        # Extract and transform tools if present
        if "tools" in data and data["tools"]:
            self._extract_and_transform_tools(data["tools"], tools_to_check)

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check:
            inputs = GenericGuardrailAPIInputs(texts=texts_to_check)
            if images_to_check:
                inputs["images"] = images_to_check
            if tools_to_check:
                inputs["tools"] = tools_to_check
            if structured_messages:
                inputs["structured_messages"] = structured_messages  # type: ignore
            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=data,
                input_type="request",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts = guardrailed_inputs.get("texts", [])

            # Step 3: Map guardrail responses back to original input structure
            await self._apply_guardrail_responses_to_input(
                messages=input_data,
                responses=guardrailed_texts,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug(
            "OpenAI Responses API: Processed input messages: %s", input_data
        )

        return data

    def _extract_and_transform_tools(
        self,
        tools: List[Dict[str, Any]],
        tools_to_check: List[ChatCompletionToolParam],
    ) -> None:
        """
        Extract and transform tools from Responses API format to Chat Completion format.

        Uses the LiteLLM transformation function to convert Responses API tools
        to Chat Completion tools that can be passed to guardrails.
        """
        if tools is not None and isinstance(tools, list):
            # Transform Responses API tools to Chat Completion tools
            (
                transformed_tools,
                _,
            ) = LiteLLMCompletionResponsesConfig.transform_responses_api_tools_to_chat_completion_tools(
                tools  # type: ignore
            )
            tools_to_check.extend(
                cast(List[ChatCompletionToolParam], transformed_tools)
            )

    def _extract_input_text_and_images(
        self,
        message: Any,  # Can be Dict[str, Any] or ResponseInputParam
        msg_idx: int,
        texts_to_check: List[str],
        images_to_check: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Extract text content and images from an input message.

        Override this method to customize text/image extraction logic.
        """
        content = message.get("content", None)
        if content is None:
            return

        if isinstance(content, str):
            # Simple string content
            texts_to_check.append(content)
            task_mappings.append((msg_idx, None))

        elif isinstance(content, list):
            # List content (e.g., multimodal with text and images)
            for content_idx, content_item in enumerate(content):
                if isinstance(content_item, dict):
                    # Extract text
                    text_str = content_item.get("text", None)
                    if text_str is not None:
                        texts_to_check.append(text_str)
                        task_mappings.append((msg_idx, int(content_idx)))

                    # Extract images
                    if content_item.get("type") == "image_url":
                        image_url = content_item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url")
                            if url:
                                images_to_check.append(url)

    async def _apply_guardrail_responses_to_input(
        self,
        messages: Any,  # Can be List[Dict[str, Any]] or ResponseInputParam
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
                if isinstance(messages[msg_idx]["content"][content_idx_optional], dict):
                    messages[msg_idx]["content"][content_idx_optional][
                        "text"
                    ] = guardrail_response

    async def process_output_response(
        self,
        response: "ResponsesAPIResponse",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
    ) -> Any:
        """
        Process output response by applying guardrails to text content and tool calls.

        Args:
            response: LiteLLM ResponsesAPIResponse object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - response.output is a list of output items
            - Each output item can be:
              * GenericResponseOutputItem with a content list of OutputText objects
              * ResponseFunctionToolCall with tool call data
            - Each OutputText object has a text field
        """

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        tool_calls_to_check: List[ChatCompletionToolCallChunk] = []
        task_mappings: List[Tuple[int, int]] = []
        # Track (output_item_index, content_index) for each text

        # Step 1: Extract all text content and tool calls from response output
        for output_idx, output_item in enumerate(response.output):
            self._extract_output_text_and_images(
                output_item=output_item,
                output_idx=output_idx,
                texts_to_check=texts_to_check,
                images_to_check=images_to_check,
                task_mappings=task_mappings,
                tool_calls_to_check=tool_calls_to_check,
            )

        # Step 2: Apply guardrail to all texts in batch
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
                inputs["tool_calls"] = tool_calls_to_check

            guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                inputs=inputs,
                request_data=request_data,
                input_type="response",
                logging_obj=litellm_logging_obj,
            )

            guardrailed_texts = guardrailed_inputs.get("texts", [])

            # Step 3: Map guardrail responses back to original response structure
            await self._apply_guardrail_responses_to_output(
                response=response,
                responses=guardrailed_texts,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug(
            "OpenAI Responses API: Processed output response: %s", response
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
        """

        final_chunk = responses_so_far[-1]

        if final_chunk.get("type") == "response.output_item.done":
            # convert openai response to model response
            model_response_stream = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(
                final_chunk
            )

            tool_calls = model_response_stream.choices[0].delta.tool_calls
            if tool_calls:
                _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                    inputs={
                        "tool_calls": cast(
                            List[ChatCompletionToolCallChunk], tool_calls
                        )
                    },
                    request_data={},
                    input_type="response",
                    logging_obj=litellm_logging_obj,
                )
                return responses_so_far
        elif final_chunk.get("type") == "response.completed":
            # convert openai response to model response
            outputs = final_chunk.get("response", {}).get("output", [])

            model_response_choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
                output_items=outputs,
                handle_raw_dict_callback=None,
            )

            tool_calls = model_response_choices[0].message.tool_calls
            text = model_response_choices[0].message.content
            guardrail_inputs = GenericGuardrailAPIInputs()
            if text:
                guardrail_inputs["texts"] = [text]
            if tool_calls:
                guardrail_inputs["tool_calls"] = cast(
                    List[ChatCompletionToolCallChunk], tool_calls
                )
            if tool_calls:
                _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
                    inputs=guardrail_inputs,
                    request_data={},
                    input_type="response",
                    logging_obj=litellm_logging_obj,
                )
                return responses_so_far
        # model_response_stream = OpenAiResponsesToChatCompletionStreamIterator.translate_responses_chunk_to_openai_stream(final_chunk)
        # tool_calls = model_response_stream.choices[0].tool_calls
        # convert openai response to model response
        string_so_far = self.get_streaming_string_so_far(responses_so_far)
        _guardrailed_inputs = await guardrail_to_apply.apply_guardrail(
            inputs={"texts": [string_so_far]},
            request_data={},
            input_type="response",
            logging_obj=litellm_logging_obj,
        )
        return responses_so_far

    def _check_streaming_has_ended(self, responses_so_far: List[Any]) -> bool:
        """
        Check if the streaming has ended.
        """
        return all(
            response.choices[0].finish_reason is not None
            for response in responses_so_far
        )

    def get_streaming_string_so_far(self, responses_so_far: List[Any]) -> str:
        """
        Get the string so far from the responses so far.
        """
        return "".join([response.get("text", "") for response in responses_so_far])

    def _has_text_content(self, response: "ResponsesAPIResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        if not hasattr(response, "output") or response.output is None:
            return False

        for output_item in response.output:
            if isinstance(output_item, BaseModel):
                try:
                    generic_response_output_item = (
                        GenericResponseOutputItem.model_validate(
                            output_item.model_dump()
                        )
                    )
                    if generic_response_output_item.content:
                        output_item = generic_response_output_item
                except Exception:
                    continue
            if isinstance(output_item, (GenericResponseOutputItem, dict)):
                content = (
                    output_item.content
                    if isinstance(output_item, GenericResponseOutputItem)
                    else output_item.get("content", [])
                )
                if content:
                    for content_item in content:
                        # Check if it's an OutputText with text
                        if isinstance(content_item, OutputText):
                            if content_item.text:

                                return True
                        elif isinstance(content_item, dict):
                            if content_item.get("text"):

                                return True
        return False

    def _extract_output_text_and_images(
        self,
        output_item: Any,
        output_idx: int,
        texts_to_check: List[str],
        images_to_check: List[str],
        task_mappings: List[Tuple[int, int]],
        tool_calls_to_check: Optional[List[ChatCompletionToolCallChunk]] = None,
    ) -> None:
        """
        Extract text content, images, and tool calls from a response output item.

        Override this method to customize text/image/tool extraction logic.
        """

        # Check if this is a tool call (OutputFunctionToolCall)
        if isinstance(output_item, OutputFunctionToolCall):
            if tool_calls_to_check is not None:
                tool_call_dict = LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call(
                    tool_call_item=output_item,
                    index=output_idx,
                )
                tool_calls_to_check.append(
                    cast(ChatCompletionToolCallChunk, tool_call_dict)
                )
            return
        elif (
            isinstance(output_item, BaseModel)
            and hasattr(output_item, "type")
            and getattr(output_item, "type") == "function_call"
        ):
            if tool_calls_to_check is not None:
                tool_call_dict = LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call(
                    tool_call_item=output_item,
                    index=output_idx,
                )
                tool_calls_to_check.append(
                    cast(ChatCompletionToolCallChunk, tool_call_dict)
                )
            return
        elif (
            isinstance(output_item, dict) and output_item.get("type") == "function_call"
        ):
            # Handle dict representation of tool call
            if tool_calls_to_check is not None:
                # Convert dict to ResponseFunctionToolCall for processing
                try:
                    tool_call_obj = ResponseFunctionToolCall(**output_item)
                    tool_call_dict = LiteLLMCompletionResponsesConfig.convert_response_function_tool_call_to_chat_completion_tool_call(
                        tool_call_item=tool_call_obj,
                        index=output_idx,
                    )
                    tool_calls_to_check.append(
                        cast(ChatCompletionToolCallChunk, tool_call_dict)
                    )
                except Exception:
                    pass
            return

        # Handle both GenericResponseOutputItem and dict
        content: Optional[Union[List[OutputText], List[dict]]] = None
        if isinstance(output_item, BaseModel):
            try:
                generic_response_output_item = GenericResponseOutputItem.model_validate(
                    output_item.model_dump()
                )
                if generic_response_output_item.content:
                    content = generic_response_output_item.content
            except Exception:
                return
        elif isinstance(output_item, dict):
            content = output_item.get("content", [])
        else:
            return

        if not content:
            return

        verbose_proxy_logger.debug(
            "OpenAI Responses API: Processing output item: %s", output_item
        )

        # Iterate through content items (list of OutputText objects)
        for content_idx, content_item in enumerate(content):
            # Handle both OutputText objects and dicts
            if isinstance(content_item, OutputText):
                text_content = content_item.text
            elif isinstance(content_item, dict):
                text_content = content_item.get("text")
            else:
                continue

            if text_content:
                texts_to_check.append(text_content)
                task_mappings.append((output_idx, int(content_idx)))

    async def _apply_guardrail_responses_to_output(
        self,
        response: "ResponsesAPIResponse",
        responses: List[str],
        task_mappings: List[Tuple[int, int]],
    ) -> None:
        """
        Apply guardrail responses back to output response.

        Override this method to customize how responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            output_idx = cast(int, mapping[0])
            content_idx = cast(int, mapping[1])

            output_item = response.output[output_idx]

            # Handle both GenericResponseOutputItem and dict
            if isinstance(output_item, GenericResponseOutputItem):
                content_item = output_item.content[content_idx]
                if isinstance(content_item, OutputText):
                    content_item.text = guardrail_response
                elif isinstance(content_item, dict):
                    content_item["text"] = guardrail_response
            elif isinstance(output_item, dict):
                content = output_item.get("content", [])
                if content and content_idx < len(content):
                    if isinstance(content[content_idx], dict):
                        content[content_idx]["text"] = guardrail_response
