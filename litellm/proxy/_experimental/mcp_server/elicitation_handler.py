"""
MCP Elicitation Handler
Handles `elicitation/create` requests from upstream MCP servers by either:
1. Relaying them to the connected downstream MCP client (if it supports elicitation)
2. Returning a decline/error response (if no downstream client or unsupported)
Supports both Form mode (structured data collection) and URL mode (external URL
navigation for sensitive interactions like OAuth).
MCP Spec Reference:
    https://modelcontextprotocol.io/specification/2025-11-25/client/elicitation
"""

from typing import Any, Optional, Union
from litellm._logging import verbose_logger

# Guard imports that require the mcp package
try:
    from mcp.types import (
        ElicitRequestFormParams,
        ElicitRequestParams,
        ElicitRequestURLParams,
        ElicitResult,
        ErrorData,
    )

    MCP_ELICITATION_AVAILABLE = True
except ImportError:
    MCP_ELICITATION_AVAILABLE = False


async def handle_elicitation_request(
    context: Any,
    params: "ElicitRequestParams",
    downstream_session: Optional[Any] = None,
    downstream_capabilities: Optional[Any] = None,
) -> Union["ElicitResult", "ErrorData"]:
    """
    Handle an MCP elicitation/create request from an upstream MCP server.
    In Gateway mode (Mode A), we relay the elicitation request to the
    connected downstream client if they declared elicitation capabilities.
    In Tool Bridge mode (Mode B), there's no persistent downstream MCP
    client, so we return a decline response.
    Args:
        context: MCP RequestContext from the upstream server connection.
        params: The ElicitRequestParams (either form or URL mode).
        downstream_session: The ServerSession to the downstream client,
            if available (for relaying).
        downstream_capabilities: The downstream client's declared
            capabilities, used to check elicitation support.
    Returns:
        ElicitResult with the user's response, or ErrorData on failure.
    """
    if not MCP_ELICITATION_AVAILABLE:
        return ErrorData(
            code=-1,
            message="MCP elicitation is not available (mcp package not installed)",
        )
    try:
        mode = getattr(params, "mode", "form")
        verbose_logger.info(
            "MCP elicitation: received request mode=%s, message=%s",
            mode,
            getattr(params, "message", ""),
        )
        # Check if we have a downstream session to relay to
        if downstream_session is not None:
            return await _relay_elicitation_to_downstream(
                params=params,
                downstream_session=downstream_session,
                downstream_capabilities=downstream_capabilities,
            )
        # No downstream session — we're in Tool Bridge mode
        # or the client doesn't support elicitation
        verbose_logger.info(
            "MCP elicitation: no downstream session available, declining"
        )
        return ElicitResult(
            action="decline",
        )
    except Exception as e:
        verbose_logger.exception("MCP elicitation handler failed: %s", e)
        return ErrorData(
            code=-1,
            message=f"Elicitation failed: {str(e)}",
        )


async def _relay_elicitation_to_downstream(
    params: "ElicitRequestParams",
    downstream_session: Any,
    downstream_capabilities: Optional[Any] = None,
) -> Union["ElicitResult", "ErrorData"]:
    """
    Relay an elicitation request to the downstream MCP client.
    Uses the ServerSession's elicit_form() or elicit_url() methods to
    send the elicitation request back to the connected client.
    Args:
        params: The elicitation request parameters.
        downstream_session: The ServerSession connected to the downstream client.
        downstream_capabilities: Client capabilities to check support.
    Returns:
        ElicitResult from the downstream client.
    """
    mode = getattr(params, "mode", "form")
    # Check if the downstream client supports the requested mode
    if downstream_capabilities is not None:
        elicit_caps = getattr(downstream_capabilities, "elicitation", None)
        if elicit_caps is None:
            verbose_logger.info(
                "MCP elicitation: downstream client does not support elicitation"
            )
            return ElicitResult(action="decline")
        if mode == "url":
            url_cap = getattr(elicit_caps, "url", None)
            if url_cap is None:
                verbose_logger.info(
                    "MCP elicitation: downstream client does not support URL mode"
                )
                return ElicitResult(action="decline")
        if mode == "form":
            form_cap = getattr(elicit_caps, "form", None)
            if form_cap is None:
                verbose_logger.info(
                    "MCP elicitation: downstream client does not support form mode"
                )
                return ElicitResult(action="decline")
    try:
        if mode == "url" and isinstance(params, ElicitRequestURLParams):
            # URL mode: relay URL to client for external navigation
            verbose_logger.info(
                "MCP elicitation: relaying URL mode to downstream, url=%s",
                getattr(params, "url", ""),
            )
            result = await downstream_session.elicit_url(
                message=params.message,
                url=params.url,
                elicitation_id=getattr(params, "elicitationId", None),
            )
        elif isinstance(params, ElicitRequestFormParams):
            # Form mode: relay structured form to client
            verbose_logger.info("MCP elicitation: relaying form mode to downstream")
            result = await downstream_session.elicit_form(
                message=params.message,
                requestedSchema=getattr(params, "requestedSchema", None),
            )
        else:
            # Fallback for generic ElicitRequestParams
            verbose_logger.info(
                "MCP elicitation: relaying generic elicitation to downstream"
            )
            result = await downstream_session.elicit(
                message=getattr(params, "message", ""),
            )
        verbose_logger.info(
            "MCP elicitation: downstream responded with action=%s",
            getattr(result, "action", "unknown"),
        )
        return result
    except Exception as e:
        verbose_logger.warning("MCP elicitation: failed to relay to downstream: %s", e)
        # If relay fails, decline gracefully
        return ElicitResult(action="decline")
