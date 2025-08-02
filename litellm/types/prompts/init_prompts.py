from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict
from typing_extensions import Required, TypedDict


class SupportedPromptIntegrations(str, Enum):
    DOT_PROMPT = "dotprompt"
    LANGFUSE = "langfuse"
    CUSTOM = "custom"


class PromptInfo(BaseModel):
    prompt_type: Literal["config", "db"]

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class PromptLiteLLMParams(BaseModel):
    prompt_id: str
    prompt_integration: str

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class PromptSpec(BaseModel):
    prompt_id: str
    litellm_params: PromptLiteLLMParams
    prompt_info: PromptInfo
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ListPromptsResponse(BaseModel):
    prompts: List[PromptSpec]
