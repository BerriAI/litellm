from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel
from typing_extensions import Required, TypedDict


# AgentProvider
class AgentProvider(TypedDict, total=False):
    """Represents the service provider of an agent."""

    organization: str  # required
    url: str  # required


# AgentExtension
class AgentExtension(TypedDict, total=False):
    """A declaration of a protocol extension supported by an Agent."""

    uri: str  # required
    description: Optional[str]
    required: Optional[bool]
    params: Optional[Dict[str, Any]]


# AgentCapabilities
class AgentCapabilities(TypedDict, total=False):
    """Defines optional capabilities supported by an agent."""

    streaming: Optional[bool]
    pushNotifications: Optional[bool]
    stateTransitionHistory: Optional[bool]
    extensions: Optional[List[AgentExtension]]


# SecurityScheme types
class SecuritySchemeBase(TypedDict, total=False):
    """Base properties shared by all security scheme objects."""

    description: Optional[str]


class APIKeySecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using an API key."""

    type: Literal["apiKey"]
    in_: Literal["query", "header", "cookie"]  # using in_ to avoid Python keyword
    name: str


class HTTPAuthSecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using HTTP authentication."""

    type: Literal["http"]
    scheme: str
    bearerFormat: Optional[str]


class MutualTLSSecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using mTLS authentication."""

    type: Literal["mutualTLS"]


class OAuthFlows(TypedDict, total=False):
    """Defines the configuration for the supported OAuth 2.0 flows."""

    authorizationCode: Optional[Dict[str, Any]]
    clientCredentials: Optional[Dict[str, Any]]
    implicit: Optional[Dict[str, Any]]
    password: Optional[Dict[str, Any]]


class OAuth2SecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using OAuth 2.0."""

    type: Literal["oauth2"]
    flows: OAuthFlows
    oauth2MetadataUrl: Optional[str]


class OpenIdConnectSecurityScheme(SecuritySchemeBase):
    """Defines a security scheme using OpenID Connect."""

    type: Literal["openIdConnect"]
    openIdConnectUrl: str


# Union of all security schemes
SecurityScheme = Union[
    APIKeySecurityScheme,
    HTTPAuthSecurityScheme,
    OAuth2SecurityScheme,
    OpenIdConnectSecurityScheme,
    MutualTLSSecurityScheme,
]


# AgentSkill
class AgentSkill(TypedDict, total=False):
    """Represents a distinct capability or function that an agent can perform."""

    id: str  # required
    name: str  # required
    description: str  # required
    tags: List[str]  # required
    examples: Optional[List[str]]
    inputModes: Optional[List[str]]
    outputModes: Optional[List[str]]
    security: Optional[List[Dict[str, List[str]]]]


# AgentInterface
class AgentInterface(TypedDict, total=False):
    """Declares a combination of a target URL and a transport protocol."""

    url: str  # required
    transport: str  # required (TransportProtocol | string)


# AgentCardSignature
class AgentCardSignature(TypedDict, total=False):
    """Represents a JWS signature of an AgentCard."""

    protected: str  # required
    signature: str  # required
    header: Optional[Dict[str, Any]]


# AgentCard
class AgentCard(TypedDict, total=False):
    """
    The AgentCard is a self-describing manifest for an agent.
    It provides essential metadata including the agent's identity, capabilities,
    skills, supported communication methods, and security requirements.
    """

    # Required fields
    protocolVersion: str
    name: str
    description: str
    url: str
    version: str
    capabilities: AgentCapabilities
    defaultInputModes: List[str]
    defaultOutputModes: List[str]
    skills: List[AgentSkill]

    # Optional fields
    preferredTransport: Optional[str]
    additionalInterfaces: Optional[List[AgentInterface]]
    iconUrl: Optional[str]
    provider: Optional[AgentProvider]
    documentationUrl: Optional[str]
    securitySchemes: Optional[Dict[str, SecurityScheme]]
    security: Optional[List[Dict[str, List[str]]]]
    supportsAuthenticatedExtendedCard: Optional[bool]
    signatures: Optional[List[AgentCardSignature]]


class AugmentedAgentCard(AgentCard):
    is_public: bool


class AgentConfig(TypedDict, total=False):
    agent_name: Required[str]
    agent_card_params: Required[AgentCard]
    litellm_params: Dict[str, Any]  # allow for any future litellm params


class PatchAgentRequest(TypedDict, total=False):
    agent_name: str
    agent_card_params: AgentCard
    litellm_params: Dict[str, Any]


# Request/Response models for CRUD endpoints


class AgentResponse(BaseModel):
    agent_id: str
    agent_name: str
    litellm_params: Optional[Dict[str, Any]] = None
    agent_card_params: Dict[str, Any]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_by: Optional[str] = None


class ListAgentsResponse(BaseModel):
    agents: List[AgentResponse]


class AgentMakePublicResponse(BaseModel):
    message: str
    public_agent_groups: List[str]
    updated_by: str


class MakeAgentsPublicRequest(BaseModel):
    agent_ids: List[str]
