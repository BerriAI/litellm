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

import asyncio
from typing import TYPE_CHECKING, Any, Coroutine, Dict, List, Optional, Tuple, cast

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.utils import Choices

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.utils import ModelResponse


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
    ) -> Any:
        """
        Process input messages by applying guardrails to text content.
        """
        messages = data.get("messages")
        if messages is None:
            return data

        tasks: List[Coroutine[Any, Any, str]] = []
        task_mappings: List[Tuple[int, Optional[int]]] = []
        # Track (message_index, content_index) for each task
        # content_index is None for string content, int for list content

        # Step 1: Extract all text content and create guardrail tasks
        for msg_idx, message in enumerate(messages):
            await self._extract_input_text_and_create_tasks(
                message=message,
                msg_idx=msg_idx,
                tasks=tasks,
                task_mappings=task_mappings,
                guardrail_to_apply=guardrail_to_apply,
                request_data=data,
            )

        # Step 2: Run all guardrail tasks in parallel
        responses = await asyncio.gather(*tasks)

        # Step 3: Map guardrail responses back to original message structure
        await self._apply_guardrail_responses_to_input(
            messages=messages,
            responses=responses,
            task_mappings=task_mappings,
        )

        verbose_proxy_logger.debug(
            "OpenAI Chat Completions: Processed input messages: %s", messages
        )

        return data

    async def _extract_input_text_and_create_tasks(
        self,
        message: Dict[str, Any],
        msg_idx: int,
        tasks: List,
        task_mappings: List[Tuple[int, Optional[int]]],
        guardrail_to_apply: "CustomGuardrail",
        request_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Extract text content from a message and create guardrail tasks.

        Override this method to customize text extraction logic.
        """
        content = message.get("content", None)
        if content is None:
            return

        if isinstance(content, str):
            # Simple string content
            tasks.append(guardrail_to_apply.apply_guardrail(text=content, request_data=request_data))
            task_mappings.append((msg_idx, None))

        elif isinstance(content, list):
            # List content (e.g., multimodal with text and images)
            for content_idx, content_item in enumerate(content):
                text_str = content_item.get("text", None)
                if text_str is None:
                    continue
                tasks.append(guardrail_to_apply.apply_guardrail(text=text_str, request_data=request_data))
                task_mappings.append((msg_idx, int(content_idx)))

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
        response: "ModelResponse",
        guardrail_to_apply: "CustomGuardrail",
    ) -> Any:
        """
        Process output response by applying guardrails to text content.

        Args:
            response: LiteLLM ModelResponse object
            guardrail_to_apply: The guardrail instance to apply

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

        tasks: List[Coroutine[Any, Any, str]] = []
        task_mappings: List[Tuple[int, Optional[int]]] = []
        # Track (choice_index, content_index) for each task

        # Step 1: Extract all text content from response choices
        for choice_idx, choice in enumerate(response.choices):
            await self._extract_output_text_and_create_tasks(
                choice=choice,
                choice_idx=choice_idx,
                tasks=tasks,
                task_mappings=task_mappings,
                guardrail_to_apply=guardrail_to_apply,
            )

        # Step 2: Run all guardrail tasks in parallel
        responses = await asyncio.gather(*tasks)

        # Step 3: Map guardrail responses back to original response structure
        await self._apply_guardrail_responses_to_output(
            response=response,
            responses=responses,
            task_mappings=task_mappings,
        )

        verbose_proxy_logger.debug(
            "OpenAI Chat Completions: Processed output response: %s", response
        )

        return response

    def _has_text_content(self, response: "ModelResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        for choice in response.choices:
            if isinstance(choice, litellm.Choices):
                if choice.message.content and isinstance(choice.message.content, str):
                    return True
        return False

    async def _extract_output_text_and_create_tasks(
        self,
        choice: Any,
        choice_idx: int,
        tasks: List,
        task_mappings: List[Tuple[int, Optional[int]]],
        guardrail_to_apply: "CustomGuardrail",
        request_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Extract text content from a response choice and create guardrail tasks.

        Override this method to customize text extraction logic.
        """
        if not isinstance(choice, litellm.Choices):
            return

        verbose_proxy_logger.debug(
            "OpenAI Chat Completions: Processing choice: %s", choice
        )

        if choice.message.content and isinstance(choice.message.content, str):
            # Simple string content
            tasks.append(
                guardrail_to_apply.apply_guardrail(text=choice.message.content, request_data=request_data)
            )
            task_mappings.append((choice_idx, None))

        elif choice.message.content and isinstance(choice.message.content, list):
            # List content (e.g., multimodal response)
            for content_idx, content_item in enumerate(choice.message.content):
                content_text = content_item.get("text")
                if content_text:
                    tasks.append(guardrail_to_apply.apply_guardrail(text=content_text, request_data=request_data))
                    task_mappings.append((choice_idx, int(content_idx)))

    async def _apply_guardrail_responses_to_output(
        self,
        response: "ModelResponse",
        responses: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail responses back to output response.

        Override this method to customize how responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            choice_idx = cast(int, mapping[0])
            content_idx_optional = cast(Optional[int], mapping[1])

            content = cast(Choices, response.choices[choice_idx]).message.content
            if content is None:
                continue

            if isinstance(content, str) and content_idx_optional is None:
                # Replace string content with guardrail response
                cast(Choices, response.choices[choice_idx]).message.content = (
                    guardrail_response
                )

            elif isinstance(content, list) and content_idx_optional is not None:
                # Replace specific text item in list content
                cast(Choices, response.choices[choice_idx]).message.content[  # type: ignore
                    content_idx_optional
                ][
                    "text"
                ] = guardrail_response
