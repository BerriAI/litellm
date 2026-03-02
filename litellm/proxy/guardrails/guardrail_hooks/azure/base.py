from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues

# Azure Content Safety APIs have a 10,000 character limit per request.
AZURE_CONTENT_SAFETY_MAX_TEXT_LENGTH = 10000


class AzureGuardrailBase:
    """
    Base class for Azure guardrails.
    """

    @staticmethod
    def split_text_by_words(text: str, max_length: int) -> List[str]:
        """
        Split text into chunks at word boundaries without breaking words.

        Always returns at least one chunk.  Short text (≤ max_length) is
        returned as a single-element list so callers can use a uniform
        loop without branching on length.

        Args:
            text: The text to split
            max_length: Maximum character length of each chunk

        Returns:
            List of text chunks, each not exceeding max_length
        """
        if len(text) <= max_length:
            return [text]

        chunks: List[str] = []
        current_chunk = ""
        words = text.split()

        for word in words:
            test_chunk = current_chunk + (" " if current_chunk else "") + word

            if len(test_chunk) <= max_length:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = word
                else:
                    # Single word longer than max_length – force-split it
                    while len(word) > max_length:
                        chunks.append(word[:max_length])
                        word = word[max_length:]
                    current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_user_prompt(self, messages: List["AllMessageValues"]) -> Optional[str]:
        """
        Get the last consecutive block of messages from the user.

        Example:
        messages = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm good, thank you!"},
            {"role": "user", "content": "What is the weather in Tokyo?"},
        ]
        get_user_prompt(messages) -> "What is the weather in Tokyo?"
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            convert_content_list_to_str,
        )

        if not messages:
            return None

        # Iterate from the end to find the last consecutive block of user messages
        user_messages = []
        for message in reversed(messages):
            if message.get("role") == "user":
                user_messages.append(message)
            else:
                # Stop when we hit a non-user message
                break

        if not user_messages:
            return None

        # Reverse to get the messages in chronological order
        user_messages.reverse()

        user_prompt = ""
        for message in user_messages:
            text_content = convert_content_list_to_str(message)
            user_prompt += text_content + "\n"

        result = user_prompt.strip()
        return result if result else None
