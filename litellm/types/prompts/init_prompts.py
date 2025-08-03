from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import Required, TypedDict


class SupportedPromptIntegrations(str, Enum):
    DOT_PROMPT = "dotprompt"
    LANGFUSE = "langfuse"
    CUSTOM = "custom"


class PromptLiteLLMParams(BaseModel):
    prompt_id: str
    prompt_integration: str

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class PromptSpec(TypedDict, total=False):
    prompt_id: Required[str]
    litellm_params: Required[PromptLiteLLMParams]
    prompt_info: Optional[Dict]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class ListPromptsResponse(BaseModel):
    prompts: List[PromptSpec]
