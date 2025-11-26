from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class TestPromptRequest(BaseModel):
    dotprompt_content: str
    prompt_variables: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None

