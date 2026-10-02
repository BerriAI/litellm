from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class SupportedPromptIntegrations(str, Enum):
    DOT_PROMPT = "dotprompt"
    LANGFUSE = "langfuse"
    CUSTOM = "custom"
    BITBUCKET = "bitbucket"
    GITLAB = "gitlab"
    GENERIC_PROMPT_MANAGEMENT = "generic_prompt_management"
    ARIZE_PHOENIX = "arize_phoenix"


class PromptInfo(BaseModel):
    prompt_type: Literal["config", "db"]
    environment: str | None = "development"

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class PromptLiteLLMParams(BaseModel):
    prompt_id: str | None = None
    prompt_integration: str

    api_base: str | None = None
    api_key: str | None = None

    provider_specific_query_params: dict[str, Any] | None = None

    ignore_prompt_manager_model: bool | None = False
    ignore_prompt_manager_optional_params: bool | None = False

    dotprompt_content: str | None = None
    """
    allows saving the dotprompt file content
    """

    model_config = ConfigDict(extra="allow", protected_namespaces=())


class PromptSpec(BaseModel):
    prompt_id: str
    litellm_params: PromptLiteLLMParams
    prompt_info: PromptInfo
    created_at: datetime | None = None
    updated_at: datetime | None = None
    version: int | None = None  # Version number for version history
    environment: str | None = "development"
    created_by: str | None = None

    def __init__(self, **data) -> None:
        if "prompt_info" not in data:
            data["prompt_info"] = PromptInfo(prompt_type="config")
        elif "prompt_info" in data:
            if isinstance(data["prompt_info"], dict) and data["prompt_info"].get("prompt_type") is None:
                data["prompt_info"]["prompt_type"] = "config"
        super().__init__(**data)


class PromptTemplateBase(BaseModel):
    litellm_prompt_id: str
    content: str
    metadata: dict[str, Any] | None = None


class PromptInfoResponse(BaseModel):
    prompt_spec: PromptSpec
    raw_prompt_template: PromptTemplateBase | None = None
    environments: list[str] | None = None  # All environments this prompt is deployed to


class ListPromptsResponse(BaseModel):
    prompts: list[PromptSpec]
