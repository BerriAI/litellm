from typing import Any

from pydantic import BaseModel


class TestPromptRequest(BaseModel):
    dotprompt_content: str
    prompt_variables: dict[str, Any] | None = None
    conversation_history: list[dict[str, str]] | None = None
