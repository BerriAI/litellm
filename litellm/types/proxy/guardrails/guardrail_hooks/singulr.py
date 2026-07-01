"""
Author: Madan Singhal
Date: 23/06/26

"""

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from .base import GuardrailConfigModel


class SingulrGuardrailRequest(BaseModel):
    """OpenAI-compatible request fields forwarded to Singulr.

    Only fields explicitly declared here are sent. Unknown fields --
    including LiteLLM-internal ones the proxy attaches to the request dict
    (auth state, http sessions, logging objects, ...) -- are dropped by
    Pydantic rather than forwarded to this third-party API.
    """

    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    messages: Optional[List[AllMessageValues]] = None
    tools: Optional[List[ChatCompletionToolParam]] = None
    functions: Optional[List[dict[str, Any]]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    response_format: Optional[dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: Optional[bool] = None


class SingulrGuardrailPayload(BaseModel):
    """Payload sent to the Singulr guardrail API."""

    request: SingulrGuardrailRequest
    input_type: str


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
