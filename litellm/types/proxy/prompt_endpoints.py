from typing import Any, Dict, Optional

from pydantic import BaseModel


class TestPromptRequest(BaseModel):
    dotprompt_content: str
    prompt_variables: Optional[Dict[str, Any]] = None

