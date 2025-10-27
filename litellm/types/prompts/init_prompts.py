from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict


class SupportedPromptIntegrations(str, Enum):
    DOT_PROMPT = "dotprompt"
    LANGFUSE = "langfuse"
    CUSTOM = "custom"
    BITBUCKET = "bitbucket"


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

    def __init__(self, **data):
        if "prompt_info" not in data:
            data["prompt_info"] = PromptInfo(prompt_type="config")
        elif "prompt_info" in data:
            if (
                isinstance(data["prompt_info"], dict)
                and data["prompt_info"].get("prompt_type") is None
            ):
                data["prompt_info"]["prompt_type"] = "config"
        super().__init__(**data)


class PromptTemplateBase(BaseModel):
    litellm_prompt_id: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class PromptInfoResponse(BaseModel):
    prompt_spec: PromptSpec
    raw_prompt_template: Optional[PromptTemplateBase] = None


class ListPromptsResponse(BaseModel):
    prompts: List[PromptSpec]
