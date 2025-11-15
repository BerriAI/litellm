from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class PublicModelHubInfo(BaseModel):
    docs_title: str
    custom_docs_description: Optional[str]
    litellm_version: str
    useful_links: Optional[Dict[str, str]]


class ProviderCredentialField(BaseModel):
    key: str
    label: str
    placeholder: Optional[str] = None
    tooltip: Optional[str] = None
    required: bool = False
    field_type: Literal["text", "password", "select", "upload"] = "text"
    options: Optional[List[str]] = None
    default_value: Optional[str] = None


class ProviderCreateInfo(BaseModel):
    provider: str
    provider_display_name: str
    litellm_provider: str
    credential_fields: List[ProviderCredentialField]
    default_model_placeholder: Optional[str] = None
