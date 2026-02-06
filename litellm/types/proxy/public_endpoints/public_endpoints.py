from typing import Dict, List, Literal, Optional, Union, Any

from pydantic import BaseModel


class PublicModelHubInfo(BaseModel):
    docs_title: str
    custom_docs_description: Optional[str]
    litellm_version: str
    # Supports both old format (Dict[str, str]) and new format (Dict[str, Dict[str, Any]])
    # New format: { "displayName": { "url": "...", "index": 0 } }
    # Old format: { "displayName": "url" } (for backward compatibility)
    useful_links: Optional[Dict[str, Union[str, Dict[str, Any]]]]


class ProviderCredentialField(BaseModel):
    key: str
    label: str
    placeholder: Optional[str] = None
    tooltip: Optional[str] = None
    required: bool = False
    field_type: Literal["text", "password", "select", "upload", "textarea"] = "text"
    options: Optional[List[str]] = None
    default_value: Optional[str] = None


class ProviderCreateInfo(BaseModel):
    provider: str
    provider_display_name: str
    litellm_provider: str
    credential_fields: List[ProviderCredentialField]
    default_model_placeholder: Optional[str] = None


class AgentCredentialField(BaseModel):
    key: str
    label: str
    placeholder: Optional[str] = None
    tooltip: Optional[str] = None
    required: bool = False
    field_type: Literal["text", "password", "select", "upload", "textarea"] = "text"
    options: Optional[List[str]] = None
    default_value: Optional[str] = None
    include_in_litellm_params: Optional[bool] = None


class AgentCreateInfo(BaseModel):
    agent_type: str
    agent_type_display_name: str
    description: Optional[str] = None
    logo_url: Optional[str] = None
    credential_fields: List[AgentCredentialField]
    litellm_params_template: Optional[Dict[str, str]] = None
    model_template: Optional[str] = None
