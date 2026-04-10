"""
Persistence and runtime resolution for MCP OAuth2 ``oauth2_flow`` (M2M vs interactive).

Mirrors the inference used in ``rest_endpoints._execute_with_mcp_client`` so DB-backed
servers created from the UI (which omits ``oauth2_flow``) still persist and load as M2M
when ``client_id``, ``client_secret``, and ``token_url`` are present.

Interactive-only servers that reuse the same three fields (e.g. some GitHub Enterprise setups)
should set ``oauth2_flow`` to ``authorization_code`` explicitly via API or config.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Union

from litellm.types.mcp import MCPAuth

OAuth2FlowLiteral = Literal["client_credentials", "authorization_code"]


def _auth_type_str(auth_type: Optional[Any]) -> Optional[str]:
    if auth_type is None:
        return None
    return auth_type.value if hasattr(auth_type, "value") else str(auth_type)


def infer_oauth2_flow_for_storage(
    *,
    auth_type: Optional[Any],
    oauth2_flow: Optional[str],
    token_url: Optional[str],
    credentials_plain: Optional[Dict[str, Any]],
) -> Optional[OAuth2FlowLiteral]:
    """
    Resolve ``oauth2_flow`` to store on ``LiteLLM_MCPServerTable``.

    - Honors explicit ``client_credentials`` or ``authorization_code``.
    - If unset and auth is OAuth2, infers ``client_credentials`` when
      ``client_id``, ``client_secret``, and ``token_url`` are all present
      (same rule as ``_execute_with_mcp_client``).
    """
    if oauth2_flow in ("client_credentials", "authorization_code"):
        return oauth2_flow  # type: ignore[return-value]
    if oauth2_flow is not None and str(oauth2_flow).strip() != "":
        return None

    if _auth_type_str(auth_type) != MCPAuth.oauth2.value:
        return None

    creds = credentials_plain or {}
    client_id = creds.get("client_id")
    client_secret = creds.get("client_secret")
    if client_id and client_secret and token_url:
        return "client_credentials"
    return None


def resolve_oauth2_flow_for_runtime(
    *,
    auth_type: Optional[Any],
    stored_oauth2_flow: Optional[str],
    token_url: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
) -> Optional[OAuth2FlowLiteral]:
    """
    Effective ``oauth2_flow`` when constructing ``MCPServer`` from a DB row.

    Uses the stored column when set; otherwise applies the same inference as
    :func:`infer_oauth2_flow_for_storage` using decrypted credential fields.
    """
    if stored_oauth2_flow in ("client_credentials", "authorization_code"):
        return stored_oauth2_flow  # type: ignore[return-value]
    return infer_oauth2_flow_for_storage(
        auth_type=auth_type,
        oauth2_flow=None,
        token_url=token_url,
        credentials_plain=(
            {"client_id": client_id, "client_secret": client_secret}
            if (client_id or client_secret)
            else None
        ),
    )
