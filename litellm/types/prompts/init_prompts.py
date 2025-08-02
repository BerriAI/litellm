from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Required, TypedDict

from pydantic import BaseModel, ConfigDict


class SupportedPromptIntegrations(str, Enum):
    DOT_PROMPT = "dotprompt"
    LANGFUSE = "langfuse"
    CUSTOM = "custom"


class PromptLiteLLMParams(BaseModel):
    prompt_id: str

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class PromptSpec(TypedDict, total=False):
    prompt_id: Optional[str]
    prompt_name: Required[str]
    litellm_params: Required[PromptLiteLLMParams]
    prompt_info: Optional[Dict]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
