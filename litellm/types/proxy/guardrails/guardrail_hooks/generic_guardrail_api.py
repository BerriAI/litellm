from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import TYPE_CHECKING, TypedDict

from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionToolCallChunk,
    ChatCompletionToolParam,
)
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.utils import ChatCompletionMessageToolCall


class GenericGuardrailAPIMetadata(TypedDict, total=False):
    user_api_key_hash: Optional[str]
    user_api_key_alias: Optional[str]
    user_api_key_user_id: Optional[str]
    user_api_key_user_email: Optional[str]
    user_api_key_team_id: Optional[str]
    user_api_key_team_alias: Optional[str]
    user_api_key_end_user_id: Optional[str]
    user_api_key_org_id: Optional[str]


class GenericGuardrailAPIOptionalParams(BaseModel):
    """Optional parameters for the Generic Guardrail API"""

    additional_provider_specific_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional provider-specific parameters to send with the guardrail request",
    )


class GenericGuardrailAPIConfigModel(
    GuardrailConfigModel[GenericGuardrailAPIOptionalParams],
):
    """Configuration parameters for the Generic Guardrail API guardrail"""

    optional_params: Optional[GenericGuardrailAPIOptionalParams] = Field(
        default_factory=GenericGuardrailAPIOptionalParams,
        description="Optional parameters for the Generic Guardrail API guardrail",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Generic Guardrail API"


class GenericGuardrailAPIRequest(BaseModel):
    """Request model for the Generic Guardrail API"""

    input_type: Literal["request", "response"]
    litellm_call_id: Optional[str] = None  # the call id of the individual LLM call
    litellm_trace_id: Optional[
        str
    ] = None  # the trace id of the LLM call - useful if there are multiple LLM calls for the same conversation
    structured_messages: Optional[List[AllMessageValues]] = None
    images: Optional[List[str]] = None
    tools: Optional[List[ChatCompletionToolParam]] = None
    texts: Optional[List[str]] = None
    request_data: GenericGuardrailAPIMetadata
    request_headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Sanitized inbound request headers from the original proxy request.",
    )
    litellm_version: Optional[str] = Field(
        default=None,
        description="LiteLLM library version running this proxy.",
    )
    additional_provider_specific_params: Optional[Dict[str, Any]] = None
    tool_calls: Optional[
        Union[List[ChatCompletionToolCallChunk], List[ChatCompletionMessageToolCall]]
    ] = None
    model: Optional[str] = None  # the model being used for the LLM call


class GenericGuardrailAPIResponse:
    """Response model for the Generic Guardrail API"""

    texts: Optional[List[str]]
    images: Optional[List[str]]
    tools: Optional[List[ChatCompletionToolParam]]
    action: str
    blocked_reason: Optional[str]

    def __init__(
        self,
        action: str,
        texts: Optional[List[str]] = None,
        blocked_reason: Optional[str] = None,
        images: Optional[List[str]] = None,
        tools: Optional[List[ChatCompletionToolParam]] = None,
    ):
        self.action = action
        self.blocked_reason = blocked_reason
        self.texts = texts
        self.images = images
        self.tools = tools

    @classmethod
    def from_dict(cls, data: dict) -> "GenericGuardrailAPIResponse":
        return cls(
            action=data.get("action", "NONE"),
            blocked_reason=data.get("blocked_reason"),
            texts=data.get("texts"),
            images=data.get("images"),
            tools=data.get("tools"),
        )
