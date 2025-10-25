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

import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, List, Optional, Tuple, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.responses.main import GenericResponseOutputItem, OutputText

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
    ) -> Any:
        """
        Process input by applying guardrails to text content.

        Handles both string input and list of message objects.
        """
        input_data: Optional[Union[str, "ResponseInputParam"]] = data.get("input")
        if input_data is None:
            return data

        # Handle simple string input
        if isinstance(input_data, str):
            guardrail_response = await guardrail_to_apply.apply_guardrail(
                text=input_data
            )
            data["input"] = guardrail_response
            verbose_proxy_logger.debug("OpenAI Responses API: Processed string input")
            return data

        # Handle list input (ResponseInputParam)
        if not isinstance(input_data, list):
            return data

        tasks: List[Coroutine[Any, Any, str]] = []
        task_mappings: List[Tuple[int, Optional[int]]] = []
        # Track (message_index, content_index) for each task
        # content_index is None for string content, int for list content

        # Step 1: Extract all text content and create guardrail tasks
        for msg_idx, message in enumerate(input_data):
            await self._extract_input_text_and_create_tasks(
                message=message,
                msg_idx=msg_idx,
                tasks=tasks,
                task_mappings=task_mappings,
                guardrail_to_apply=guardrail_to_apply,
            )

        # Step 2: Run all guardrail tasks in parallel
        if tasks:
            responses = await asyncio.gather(*tasks)

            # Step 3: Map guardrail responses back to original input structure
            await self._apply_guardrail_responses_to_input(
                messages=input_data,
                responses=responses,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug(
            "OpenAI Responses API: Processed input messages: %s", input_data
        )

        return data

    async def _extract_input_text_and_create_tasks(
        self,
        message: Any,  # Can be Dict[str, Any] or ResponseInputParam
        msg_idx: int,
        tasks: List[Coroutine[Any, Any, str]],
        task_mappings: List[Tuple[int, Optional[int]]],
        guardrail_to_apply: "CustomGuardrail",
    ) -> None:
        """
        Extract text content from an input message and create guardrail tasks.

        Override this method to customize text extraction logic.
        """
        content = message.get("content", None)
        if content is None:
            return

        if isinstance(content, str):
            # Simple string content
            tasks.append(guardrail_to_apply.apply_guardrail(text=content))
            task_mappings.append((msg_idx, None))

        elif isinstance(content, list):
            # List content (e.g., multimodal with text and images)
            for content_idx, content_item in enumerate(content):
                if isinstance(content_item, dict):
                    text_str = content_item.get("text", None)
                    if text_str is not None:
                        tasks.append(guardrail_to_apply.apply_guardrail(text=text_str))
                        task_mappings.append((msg_idx, int(content_idx)))

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
    ) -> Any:
        """
        Process output response by applying guardrails to text content.

        Args:
            response: LiteLLM ResponsesAPIResponse object
            guardrail_to_apply: The guardrail instance to apply

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - response.output is a list of output items
            - Each output item has a content list with OutputText objects
            - Each OutputText object has a text field
        """
        # Step 0: Check if response has any text content to process
        if not self._has_text_content(response):
            verbose_proxy_logger.warning(
                "OpenAI Responses API: No text content in response, skipping guardrail"
            )
            return response

        tasks: List[Coroutine[Any, Any, str]] = []
        task_mappings: List[Tuple[int, int]] = []
        # Track (output_item_index, content_index) for each task

        # Step 1: Extract all text content from response output
        for output_idx, output_item in enumerate(response.output):
            await self._extract_output_text_and_create_tasks(
                output_item=output_item,
                output_idx=output_idx,
                tasks=tasks,
                task_mappings=task_mappings,
                guardrail_to_apply=guardrail_to_apply,
            )

        # Step 2: Run all guardrail tasks in parallel
        if tasks:
            responses = await asyncio.gather(*tasks)

            # Step 3: Map guardrail responses back to original response structure
            await self._apply_guardrail_responses_to_output(
                response=response,
                responses=responses,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug(
            "OpenAI Responses API: Processed output response: %s", response
        )

        return response

    def _has_text_content(self, response: "ResponsesAPIResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        if not hasattr(response, "output") or response.output is None:
            return False

        for output_item in response.output:
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

    async def _extract_output_text_and_create_tasks(
        self,
        output_item: Any,
        output_idx: int,
        tasks: List,
        task_mappings: List[Tuple[int, int]],
        guardrail_to_apply: "CustomGuardrail",
    ) -> None:
        """
        Extract text content from a response output item and create guardrail tasks.

        Override this method to customize text extraction logic.
        """
        # Handle both GenericResponseOutputItem and dict
        if isinstance(output_item, GenericResponseOutputItem):
            content = output_item.content
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
                tasks.append(guardrail_to_apply.apply_guardrail(text=text_content))
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
