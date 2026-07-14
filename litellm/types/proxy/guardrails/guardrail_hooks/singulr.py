"""
Author: Madan Singhal
Date: 23/06/26

"""

from typing import List, Optional, Union

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel, Field

from litellm.types.llms.openai import ChatCompletionToolParam, ChatCompletionToolCallChunk

from .base import GuardrailConfigModel


class SingulrGuardrailRequest(BaseModel):
    model: Optional[str] = None
    prompts: Optional[dict[str, list[str]]] = None
    completions: Optional[list[str]] = None
    tools: Optional[List[ChatCompletionToolParam]] = None
    tool_calls: Optional[
        List[
            Union[
                ChatCompletionToolCallChunk,
                ChatCompletionMessageToolCall,
            ]
        ]
    ] = None


class SingulrGuardrailPayload(BaseModel):
    """Payload sent to the Singulr guardrail API."""

    request: SingulrGuardrailRequest
    input_type: str


class SingulrGuardrailResponse(BaseModel):
    """Response returned by the Singulr guardrail API."""

    should_block: bool = False
    blocking_due_to: Optional[str] = None


class SingulrGuardrailConfigModel(GuardrailConfigModel):
    singulr_api_key: Optional[str] = Field(
        default=None,
        description="The Singulr API key. Generate API key from Singulr Platform.",
    )

    singulr_api_base: Optional[str] = Field(
        default=None,
        description="The Singulr API base URL. Get base URL from Singulr Platform.",
    )

    singulr_application_id: Optional[str] = Field(
        default=None,
        description="The Singulr application ID. Get application ID from Singulr Platform.",
    )

    singulr_guardrail_id: Optional[str] = Field(
        default=None,
        description="The Singulr Guardrail ID. Get guardrail ID from Singulr Platform.",
    )

    block_on_error: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to block requests when the Singulr Guardrails API is unavailable "
            "or returns an error. If enabled, requests fail closed. "
            "If disabled, requests continue without guardrail enforcement (fail open)."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Singulr"
