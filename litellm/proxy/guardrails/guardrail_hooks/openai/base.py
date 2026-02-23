from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues


class OpenAIGuardrailBase:
    """
    Base class for OpenAI guardrails.
    """

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