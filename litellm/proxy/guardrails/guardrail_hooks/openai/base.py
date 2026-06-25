from typing import TYPE_CHECKING, List, Optional

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_last_user_message,
)

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
        return get_last_user_message(messages)
