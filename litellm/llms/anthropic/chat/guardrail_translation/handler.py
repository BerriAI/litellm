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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation

if TYPE_CHECKING:
    from litellm.integrations.custom_guardrail import CustomGuardrail
    from litellm.types.llms.anthropic_messages.anthropic_response import (
        AnthropicMessagesResponse,
        AnthropicResponseTextBlock,
    )


class AnthropicMessagesHandler(BaseTranslation):
    """
    Handler for processing Anthropic messages with guardrails.

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
            guardrailed_texts, guardrailed_images = (
                await guardrail_to_apply.apply_guardrail(
                    texts=texts_to_check,
                    request_data=data,
                    input_type="request",
                    images=images_to_check if images_to_check else None,
                    logging_obj=litellm_logging_obj,
                )
            )

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
        if content is None:
            return

        if isinstance(content, str):
            # Simple string content
            texts_to_check.append(content)
            task_mappings.append((msg_idx, None))

        elif isinstance(content, list):
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
        Process output response by applying guardrails to text content.

        Args:
            response: Anthropic MessagesResponse object
            guardrail_to_apply: The guardrail instance to apply
            litellm_logging_obj: Optional logging object
            user_api_key_dict: User API key metadata to pass to guardrails

        Returns:
            Modified response with guardrail applied to content

        Response Format Support:
            - List content: response.content = [{"type": "text", "text": "text here"}, ...]
        """
        # Step 0: Check if response has any text content to process
        if not self._has_text_content(response):
            verbose_proxy_logger.warning(
                "Anthropic Messages: No text content in response, skipping guardrail"
            )
            return response

        texts_to_check: List[str] = []
        images_to_check: List[str] = []
        task_mappings: List[Tuple[int, Optional[int]]] = []
        # Track (content_index, None) for each text

        response_content = response.get("content", [])
        if not response_content:
            return response

        # Step 1: Extract all text content from response
        for content_idx, content_block in enumerate(response_content):
            # Check if this is a text block by checking the 'type' field
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                # Cast to dict to handle the union type properly
                self._extract_output_text_and_images(
                    content_block=cast(Dict[str, Any], content_block),
                    content_idx=content_idx,
                    texts_to_check=texts_to_check,
                    images_to_check=images_to_check,
                    task_mappings=task_mappings,
                )

        # Step 2: Apply guardrail to all texts in batch
        if texts_to_check:
            # Create a request_data dict with response info and user API key metadata
            request_data: dict = {"response": response}

            # Add user API key metadata with prefixed keys
            user_metadata = self.transform_user_api_key_dict_to_metadata(
                user_api_key_dict
            )
            if user_metadata:
                request_data["litellm_metadata"] = user_metadata

            guardrailed_texts, guardrailed_images = (
                await guardrail_to_apply.apply_guardrail(
                    texts=texts_to_check,
                    request_data=request_data,
                    input_type="response",
                    images=images_to_check if images_to_check else None,
                    logging_obj=litellm_logging_obj,
                )
            )

            # Step 3: Map guardrail responses back to original response structure
            await self._apply_guardrail_responses_to_output(
                response=response,
                responses=guardrailed_texts,
                task_mappings=task_mappings,
            )

        verbose_proxy_logger.debug(
            "Anthropic Messages: Processed output response: %s", response
        )

        return response

    def _has_text_content(self, response: "AnthropicMessagesResponse") -> bool:
        """
        Check if response has any text content to process.

        Override this method to customize text content detection.
        """
        response_content = response.get("content", [])
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
    ) -> None:
        """
        Extract text content and images from a response content block.

        Override this method to customize text/image extraction logic.
        """
        content_text = content_block.get("text")
        if content_text and isinstance(content_text, str):
            # Simple string content
            texts_to_check.append(content_text)
            task_mappings.append((content_idx, None))

    async def _apply_guardrail_responses_to_output(
        self,
        response: "AnthropicMessagesResponse",
        responses: List[str],
        task_mappings: List[Tuple[int, Optional[int]]],
    ) -> None:
        """
        Apply guardrail responses back to output response.

        Override this method to customize how responses are applied.
        """
        for task_idx, guardrail_response in enumerate(responses):
            mapping = task_mappings[task_idx]
            content_idx = cast(int, mapping[0])

            response_content = response.get("content", [])
            if not response_content:
                continue

            # Get the content block at the index
            if content_idx >= len(response_content):
                continue

            content_block = response_content[content_idx]

            # Verify it's a text block and update the text field
            if isinstance(content_block, dict) and content_block.get("type") == "text":
                # Cast to dict to handle the union type properly for assignment
                content_block = cast("AnthropicResponseTextBlock", content_block)
                content_block["text"] = guardrail_response
