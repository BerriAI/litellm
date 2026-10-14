from typing import Any

from litellm.types.llms.base import LiteLLMBaseModel


class TestPromptRequest(LiteLLMBaseModel):
    dotprompt_content: str
    prompt_variables: dict[str, Any] | None = None
    conversation_history: list[dict[str, str]] | None = None
