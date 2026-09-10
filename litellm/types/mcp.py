import enum
import re
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from litellm.types.llms.base import HiddenParams

if TYPE_CHECKING:
    from mcp.types import EmbeddedResource as MCPEmbeddedResource
    from mcp.types import ImageContent as MCPImageContent
    from mcp.types import TextContent as MCPTextContent
else:
    MCPEmbeddedResource = Any
    MCPImageContent = Any
    MCPTextContent = Any


class MCPTransport(str, enum.Enum):
    sse = "sse"
    http = "http"
    stdio = "stdio"


class MCPSpecVersion(str, enum.Enum):
    nov_2024 = "2024-11-05"
    mar_2025 = "2025-03-26"
    jun_2025 = "2025-06-18"


class MCPAuth(str, enum.Enum):
    none = "none"
    api_key = "api_key"
    bearer_token = "bearer_token"
    basic = "basic"
    authorization = "authorization"
    oauth2 = "oauth2"
    aws_sigv4 = "aws_sigv4"
    token = "token"
    oauth2_token_exchange = "oauth2_token_exchange"
    oauth2_id_jag = "oauth2_id_jag"
    true_passthrough = "true_passthrough"
    oauth_delegate = "oauth_delegate"


# RFC 8693 default subject_token_type. A NULL column / omitted config key means
# "use this default"; it is applied at every egress build site via this single
# constant rather than a DB-level DEFAULT (Prisma writes explicit values on
# insert, so a column default would rarely apply anyway).
DEFAULT_SUBJECT_TOKEN_TYPE: Final = "urn:ietf:params:oauth:token-type:access_token"

# MCP Literals
MCPTransportType = Literal[MCPTransport.sse, MCPTransport.http, MCPTransport.stdio]
MCPSpecVersionType = Literal[MCPSpecVersion.nov_2024, MCPSpecVersion.mar_2025, MCPSpecVersion.jun_2025]
MCPAuthType = (
    Literal[
        MCPAuth.none,
        MCPAuth.api_key,
        MCPAuth.bearer_token,
        MCPAuth.basic,
        MCPAuth.authorization,
        MCPAuth.oauth2,
        MCPAuth.aws_sigv4,
        MCPAuth.token,
        MCPAuth.oauth2_token_exchange,
        MCPAuth.oauth2_id_jag,
        MCPAuth.true_passthrough,
        MCPAuth.oauth_delegate,
    ]
    | None
)


class MCPPublicServer(BaseModel):
    """
    Safe params for public MCP servers
    """

    server_id: str
    name: str
    alias: str | None = None
    server_name: str | None = None
    transport: MCPTransportType
    spec_path: str | None = None
    auth_type: MCPAuthType | None = None
    mcp_info: dict[str, Any] | None = None


class MCPToolSearchSettings(BaseModel):
    """`litellm_settings.mcp_tool_search`: how the native `mcp_tool_search` virtual tool ranks the caller's tools."""

    model_config = ConfigDict(frozen=True)

    embedding_model: str | None = Field(
        default=None,
        description="Embedding model from model_list used to rank tools by meaning. Unset keeps keyword matching.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Most ranked tools a search returns. A smaller top_k in the tool call wins. Core tools do not count.",
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Lowest cosine similarity a tool needs to appear in semantic results (0.0 = no cutoff).",
    )
    core_tools: tuple[str, ...] = Field(
        default=(),
        description="Tool names always returned first when the caller can access them, e.g. `my_server-get_rates`.",
    )


# OAuth 2.0 token-endpoint client authentication method (RFC 6749 section 2.3.1).
MCPTokenEndpointAuthMethod = Literal["client_secret_basic", "client_secret_post"]


class MCPCredentials(TypedDict, total=False):
    auth_value: str | None
    """
    Authentication value
    """

    client_id: str | None
    """
    OAuth 2.0 client identifier used when auth_type is oauth2
    """

    client_secret: str | None
    """
    OAuth 2.0 client secret used when auth_type is oauth2
    """

    scopes: list[str] | None
    """
    OAuth 2.0 scopes to request when exchanging the client credentials
    """

    # AWS SigV4 fields
    aws_access_key_id: str | None
    """AWS access key ID for SigV4 signing. Optional — falls back to boto3 credential chain."""

    aws_secret_access_key: str | None
    """AWS secret access key for SigV4 signing. Optional — falls back to boto3 credential chain."""

    aws_session_token: str | None
    """AWS session token for temporary STS credentials. Optional."""

    aws_region_name: str | None
    """AWS region for SigV4 signing (e.g., 'us-east-1'). Not a secret — stored unencrypted."""

    aws_service_name: str | None
    """AWS service name for SigV4 signing (e.g., 'bedrock-agentcore'). Not a secret — stored unencrypted."""

    aws_role_name: str | None
    """IAM role ARN for STS AssumeRole (e.g., 'arn:aws:iam::123456789012:role/MyRole'). Not a secret — stored unencrypted."""

    aws_session_name: str | None
    """Session name for STS AssumeRole (used in CloudTrail). Not a secret — stored unencrypted."""

    audience: str | None
    """
    Target audience for OAuth 2.0 Token Exchange (RFC 8693).

    Legacy input shape: this setting has a dedicated ``audience`` column, which is
    authoritative. A value sent here is accepted for back-compat (the pre-column
    REST shape, released since 2026-05), lifted into the column on write, and
    stripped from the stored blob. Prefer the top-level request field.
    """

    token_exchange_endpoint: str | None
    """
    IDP token endpoint for OAuth 2.0 Token Exchange (RFC 8693).

    Legacy input shape: lifted into the dedicated ``token_exchange_endpoint``
    column on write and stripped from the stored blob; the column is
    authoritative. Prefer the top-level request field.
    """

    subject_token_type: str | None
    """
    Subject token type for OAuth 2.0 Token Exchange (RFC 8693).
    Default: DEFAULT_SUBJECT_TOKEN_TYPE (urn:ietf:params:oauth:token-type:access_token).

    Legacy input shape: lifted into the dedicated ``subject_token_type`` column on
    write and stripped from the stored blob; the column is authoritative. Prefer
    the top-level request field.
    """

    id_jag_resource_token_endpoint: str | None
    """
    Resource authorization server JWT-bearer (RFC 7523) endpoint for ID-JAG leg 2
    """

    id_jag_resource: str | None
    """
    Optional RFC 8707 resource indicator sent on ID-JAG leg 1
    """

    upstream_resource: str | None
    """
    Optional RFC 8707 resource indicator sent on the upstream oauth2 legs (authorize, both token
    grants, and the client_credentials fetch). Omitted when unset, which is the default; "auto"
    derives the canonical URI from the server's url; any other value is sent verbatim.
    Distinct from ``id_jag_resource``, which is the same parameter on the ID-JAG exchange, and from
    ``audience``, which is the RFC 8693 token-exchange parameter.
    """

    upstream_token_header: str | None  # writable-ok: pydantic warns it cannot honour ReadOnly here
    """
    Which upstream header carries the credential LiteLLM resolves for this server. Omitted when
    unset, which keeps RFC 6750's default of ``Authorization``. Set it when the upstream expects the
    gateway's token somewhere else (an ESB terminating its own credential on e.g. ``esb-oauth``), so
    a separate operator-configured ``Authorization`` reaches the origin untouched. Non-secret, so it
    is stored in plaintext and returned on admin reads.
    """

    client_private_key: str | None
    """
    PEM private key used to sign the private-key-JWT client_assertion (RFC 7523)
    """

    client_private_key_id: str | None
    """
    Key id (kid) advertised in the client_assertion JWT header
    """

    client_assertion_signing_alg: str | None
    """
    Signing algorithm for the client_assertion JWT. Default: RS256
    """

    token_endpoint_auth_method: MCPTokenEndpointAuthMethod | None
    """
    How the gateway authenticates to the upstream token endpoint. "client_secret_basic"
    sends HTTP Basic; defaults to "client_secret_post" when unset.
    """

    redirect_uris: list[str] | None
    """
    The redirect URIs a dynamically registered (RFC 7591) OAuth client was bound to at
    registration time. Lets a later registration detect that the proxy's public origin no
    longer matches the registered callback and re-register instead of reusing a client the
    IdP will reject. Absent for admin-configured clients and for clients registered before
    this field existed. Not a secret; stored unencrypted.
    """

    token_exchange_profile: str | None
    """
    Token exchange wire dialect: "rfc8693" (default, the standard token-exchange grant) or
    "entra_obo" (Microsoft Entra On-Behalf-Of, the RFC 7523 jwt-bearer grant + requested_token_use
    extension). Not a secret; stored unencrypted.

    Legacy input shape: lifted into the dedicated ``token_exchange_profile`` column on
    write and stripped from the stored blob; the column is authoritative. Prefer the
    top-level request field.
    """


DEFAULT_CREDENTIAL_HEADER: Final = "Authorization"

_HEADER_NAME_TOKEN: Final = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")


def normalize_upstream_header_name(raw: str) -> str | None:
    """The trimmed header name if it is a usable RFC 7230 ``token``, else None.

    One owner for the grammar; each caller picks its own failure shape (a config-load raise, an
    API 400, a typed CredError). An operator-supplied name reaches egress verbatim, so a value
    carrying CR/LF, spaces or separators must never get that far.
    """
    stripped: Final = raw.strip()
    return stripped if stripped and _HEADER_NAME_TOKEN.match(stripped) else None


def same_header(name: str, other: str) -> bool:
    """Whether two HTTP header names are the same one. They are case-insensitive (RFC 7230 3.2)."""
    return name.lower() == other.lower()


def has_header(headers: Mapping[str, str] | None, name: str) -> bool:
    """Whether ``headers`` carries ``name`` under any casing."""
    return bool(headers) and any(same_header(key, name) for key in headers or {})


def without_header(headers: Mapping[str, str] | None, name: str) -> dict[str, str] | None:
    """A copy of ``headers`` with every casing of ``name`` removed, or None if nothing remains.

    The one owner of "drop this credential's header". Both MCP stacks and the upstream-credential
    resolver share it so a slot can never be dropped case-sensitively in one place and
    case-insensitively in another, which is how an injected header came to shadow a resolved
    credential on the v1 path.
    """
    if not headers:
        return None
    filtered: Final = {key: value for key, value in headers.items() if not same_header(key, name)}
    return filtered or None


_DEFAULT_PORTS: Final[Mapping[str, int]] = MappingProxyType({"http": 80, "https": 443})


def crosses_origin(configured: str, target: str) -> bool:
    """Whether ``target`` leaves ``configured``'s origin, by the rule HTTP clients use.

    Origin is scheme, host and port, not host alone, so a same-host HTTPS downgrade or a port change
    counts as crossing it. A plain http -> https upgrade of the same host is exempt, matching what
    httpx exempts when it decides whether to keep ``Authorization`` across a redirect.
    """
    a: Final = urlsplit(configured)
    b: Final = urlsplit(target)
    port_a: Final = a.port or _DEFAULT_PORTS.get(a.scheme)
    port_b: Final = b.port or _DEFAULT_PORTS.get(b.scheme)
    if a.scheme == b.scheme and a.hostname == b.hostname and port_a == port_b:
        return False
    return not (
        a.hostname == b.hostname and a.scheme == "http" and port_a == 80 and b.scheme == "https" and port_b == 443
    )


def custom_credential_slot(headers: Mapping[str, str] | None) -> str | None:
    """The first header carrying a credential somewhere other than ``Authorization``, if any."""
    return next((name for name in headers or {} if not same_header(name, DEFAULT_CREDENTIAL_HEADER)), None)


def credential_redirect_hook(
    configured_url: str, slot: str | None
) -> Callable[[httpx.Request], Awaitable[None]] | None:
    """An httpx request hook dropping ``slot`` once a redirect leaves ``configured_url``'s origin.

    None when no guard is needed, so callers do not each repeat the exemption: HTTP clients already
    strip ``Authorization`` across origins, but forward every other header, so only a credential an
    operator moved to its own slot can be replayed to whatever host the upstream redirects to.
    """
    if not configured_url or not slot or same_header(slot, DEFAULT_CREDENTIAL_HEADER):
        return None

    async def guard(request: httpx.Request) -> None:
        if slot in request.headers and crosses_origin(configured_url, str(request.url)):
            del request.headers[slot]

    return guard


MCP_ADMIN_CONFIG_CREDENTIAL_KEYS: Final[tuple[str, ...]] = ("upstream_resource", "upstream_token_header")
"""Non-secret credential keys returned on read so the admin form can show and clear them. Mirrors
``ADMIN_CONFIG_CREDENTIAL_KEYS`` in ``ui/litellm-dashboard/src/components/mcp_tools/types.tsx``."""


class MCPServerCostInfo(TypedDict, total=False):
    default_cost_per_query: float | None
    """
    Default cost per query for the MCP server tool call
    """

    tool_name_to_cost_per_query: dict[str, float] | None
    """
    Granular, set a custom cost for each tool in the MCP server
    """


class MCPStdioConfig(TypedDict, total=False):
    command: str
    """
    Command to run the MCP server (e.g., 'npx', 'python', 'node')
    """

    args: list[str]
    """
    Arguments to pass to the command
    """

    env: dict[str, str] | None
    """
    Environment variables to set when running the command
    """


class MCPPreCallRequestObject(BaseModel):
    """
    Pydantic object used for MCP pre_call_hook request validation and modification
    """

    tool_name: str
    arguments: dict[str, Any]
    server_name: str | None = None
    user_api_key_auth: dict[str, Any] | None = None
    hidden_params: HiddenParams = HiddenParams()


class MCPPreCallResponseObject(BaseModel):
    """
    Pydantic object used for MCP pre_call_hook response
    """

    should_proceed: bool = True
    modified_arguments: dict[str, Any] | None = None
    error_message: str | None = None
    hidden_params: HiddenParams = HiddenParams()


class MCPDuringCallRequestObject(BaseModel):
    """
    Pydantic object used for MCP during_call_hook request
    """

    tool_name: str
    arguments: dict[str, Any]
    server_name: str | None = None
    start_time: float | None = None
    hidden_params: HiddenParams = HiddenParams()


class MCPDuringCallResponseObject(BaseModel):
    """
    Pydantic object used for MCP during_call_hook response
    """

    should_continue: bool = True
    error_message: str | None = None
    hidden_params: HiddenParams = HiddenParams()


class MCPPostCallResponseObject(BaseModel):
    """
    Pydantic object used for MCP post_call_hook response
    """

    mcp_tool_call_response: list[MCPTextContent | MCPImageContent | MCPEmbeddedResource]
    hidden_params: HiddenParams
