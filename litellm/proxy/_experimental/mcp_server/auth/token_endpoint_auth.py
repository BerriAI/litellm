"""Client authentication for OAuth 2.0 token-endpoint requests (RFC 6749 section 2.3.1).

A confidential MCP upstream may require ``client_secret_basic`` (HTTP Basic, the OIDC
default) or ``client_secret_post`` (credentials in the form body). Every token-endpoint
POST in the MCP gateway builds its client authentication here so the two methods are
applied identically across the inbound exchange, the refresh grants, the M2M
client_credentials fetch, and RFC 8693 token exchange. The default is
``client_secret_post`` so servers that never set ``token_endpoint_auth_method`` keep
their current behavior.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from urllib.parse import quote_plus

from litellm.types.mcp_server.mcp_server_manager import MCPTokenEndpointAuthMethod


@dataclass(frozen=True, slots=True)
class TokenEndpointClientAuth:
    headers: dict[str, str]
    body: dict[str, str]


class TokenEndpointAuthConfigError(ValueError):
    """``client_secret_basic`` is configured but the client credentials needed for it are missing.

    Subclasses ``ValueError`` so existing call sites that already guard missing credentials with
    ``except ValueError`` / ``except Exception`` keep mapping it to their own failure contract.
    """


def normalize_token_endpoint_auth_method(
    value: object,
) -> MCPTokenEndpointAuthMethod | None:
    """Narrow an untyped (DB/JSON-sourced) value to the auth-method literal, else ``None``."""
    if value == "client_secret_basic":
        return "client_secret_basic"
    if value == "client_secret_post":
        return "client_secret_post"
    return None


def build_token_endpoint_client_auth(
    *,
    auth_method: MCPTokenEndpointAuthMethod | None,
    client_id: str | None,
    client_secret: str | None,
) -> TokenEndpointClientAuth:
    """Return the headers and body fields that authenticate the client to the token endpoint.

    ``client_secret_basic`` is a confidential-client method, so it requires both ``client_id`` and
    ``client_secret`` and raises ``TokenEndpointAuthConfigError`` when either is missing rather than
    silently degrading to a weaker request (RFC 6749 section 2.3.1; matches the "absent credential
    must surface, never fall sideways" rule). It sends an HTTP Basic ``Authorization`` header and
    keeps the credentials out of the body. Any other method (including ``None``, the default) is the
    ``client_secret_post`` path: it places whichever of ``client_id`` / ``client_secret`` are present
    into the body, so a secretless client_id (a public client authenticating with PKCE) stays valid.
    """
    if auth_method == "client_secret_basic":
        if not client_id or not client_secret:
            raise TokenEndpointAuthConfigError(
                "token_endpoint_auth_method=client_secret_basic requires both client_id and client_secret"
            )
        # RFC 6749 section 2.3.1: form-urlencode each value before joining with ':' so a
        # client_id/secret containing reserved characters (':', '+', '%', ...) is transmitted intact.
        userpass = f"{quote_plus(client_id)}:{quote_plus(client_secret)}"
        encoded = base64.b64encode(userpass.encode()).decode()
        return TokenEndpointClientAuth(headers={"Authorization": f"Basic {encoded}"}, body={})
    return TokenEndpointClientAuth(
        headers={},
        body={
            **({"client_id": client_id} if client_id else {}),
            **({"client_secret": client_secret} if client_secret else {}),
        },
    )
