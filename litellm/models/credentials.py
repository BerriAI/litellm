"""
Credential table models.

These are the canonical credential types for the proxy. They live in the model
layer; ``litellm.types.utils`` re-exports them for backwards compatibility.
"""

from typing import Optional

from pydantic import BaseModel, model_validator


class CredentialBase(BaseModel):
    credential_name: str
    credential_info: dict


class CredentialItem(CredentialBase):
    credential_values: dict


class CreateCredentialItem(CredentialBase):
    credential_values: Optional[dict] = None
    model_id: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def check_credential_params(cls, values):
        if not values.get("credential_values") and not values.get("model_id"):
            raise ValueError("Either credential_values or model_id must be set")
        return values
