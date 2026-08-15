from typing import Any, Literal

from pydantic import BaseModel


class PublicModelHubInfo(BaseModel):
    docs_title: str
    custom_docs_description: str | None
    litellm_version: str
    # Supports both old format (Dict[str, str]) and new format (Dict[str, Dict[str, Any]])
    # New format: { "displayName": { "url": "...", "index": 0 } }
    # Old format: { "displayName": "url" } (for backward compatibility)
    useful_links: dict[str, str | dict[str, Any]] | None


class ProviderCredentialField(BaseModel):
    key: str
    label: str
    placeholder: str | None = None
    tooltip: str | None = None
    required: bool = False
    field_type: Literal["text", "password", "select", "upload", "textarea"] = "text"
    options: list[str] | None = None
    default_value: str | None = None


class ProviderCreateInfo(BaseModel):
    provider: str
    provider_display_name: str
    litellm_provider: str
    credential_fields: list[ProviderCredentialField]
    default_model_placeholder: str | None = None


class AgentCredentialField(BaseModel):
    key: str
    label: str
    placeholder: str | None = None
    tooltip: str | None = None
    required: bool = False
    field_type: Literal["text", "password", "select", "upload", "textarea"] = "text"
    options: list[str] | None = None
    default_value: str | None = None
    include_in_litellm_params: bool | None = None


class AgentCreateInfo(BaseModel):
    agent_type: str
    agent_type_display_name: str
    description: str | None = None
    logo_url: str | None = None
    credential_fields: list[AgentCredentialField]
    litellm_params_template: dict[str, str] | None = None
    model_template: str | None = None


class EndpointProvider(BaseModel):
    slug: str
    display_name: str


class SupportedEndpoint(BaseModel):
    key: str
    label: str
    endpoint: str
    providers: list[EndpointProvider]


class SupportedEndpointsResponse(BaseModel):
    endpoints: list[SupportedEndpoint]
