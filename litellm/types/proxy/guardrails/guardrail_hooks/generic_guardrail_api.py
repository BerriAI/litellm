from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


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


class GenericGuardrailAPIRequest:
    """Request model for the Generic Guardrail API"""

    input_type: Literal["request", "response"]

    def __init__(
        self,
        texts: List[str],
        request_data: GenericGuardrailAPIMetadata,
        input_type: Literal["request", "response"],
        additional_provider_specific_params: Optional[Dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ):
        self.texts = texts
        self.request_data = request_data
        self.additional_provider_specific_params = (
            additional_provider_specific_params or {}
        )
        self.images = images
        self.input_type = input_type

    def to_dict(self) -> dict:
        return {
            "texts": self.texts,
            "request_data": self.request_data,
            "images": self.images,
            "additional_provider_specific_params": self.additional_provider_specific_params,
            "input_type": self.input_type,
        }


class GenericGuardrailAPIResponse:
    """Response model for the Generic Guardrail API"""

    texts: Optional[List[str]]
    images: Optional[List[str]]
    action: str
    blocked_reason: Optional[str]

    def __init__(
        self,
        action: str,
        texts: Optional[List[str]] = None,
        blocked_reason: Optional[str] = None,
        images: Optional[List[str]] = None,
    ):
        self.action = action
        self.blocked_reason = blocked_reason
        self.texts = texts
        self.images = images

    @classmethod
    def from_dict(cls, data: dict) -> "GenericGuardrailAPIResponse":
        return cls(
            action=data.get("action", "NONE"),
            blocked_reason=data.get("blocked_reason"),
            texts=data.get("texts"),
            images=data.get("images"),
        )
