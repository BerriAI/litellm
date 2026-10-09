"""Relay upstream elicitation through the initiating downstream MCP request."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_logger
from litellm.constants import MCP_CLIENT_TIMEOUT

if TYPE_CHECKING:
    from mcp.types import (
        INTERNAL_ERROR,
        INVALID_REQUEST,
        REQUEST_TIMEOUT,
        ClientCapabilities,
        ElicitRequestedSchema,
        ElicitRequestFormParams,
        ElicitRequestParams,
        ElicitRequestURLParams,
        ElicitResult,
        ErrorData,
        RequestId,
    )

try:
    from mcp.types import (
        INTERNAL_ERROR,
        INVALID_REQUEST,
        REQUEST_TIMEOUT,
        ClientCapabilities,
        ElicitRequestedSchema,
        ElicitRequestFormParams,
        ElicitRequestParams,
        ElicitRequestURLParams,
        ElicitResult,
        ErrorData,
    )

    MCP_ELICITATION_AVAILABLE = True
except ImportError:
    MCP_ELICITATION_AVAILABLE = False


class _DownstreamElicitSession(Protocol):
    async def elicit_url(
        self, message: str, url: str, elicitation_id: str, related_request_id: RequestId | None = None
    ) -> ElicitResult: ...

    async def elicit_form(
        self, message: str, requested_schema: ElicitRequestedSchema, related_request_id: RequestId | None = None
    ) -> ElicitResult: ...


async def handle_elicitation_request(
    context: object,
    params: ElicitRequestParams,
    downstream_session: _DownstreamElicitSession | None = None,
    downstream_capabilities: ClientCapabilities | None = None,
    related_request_id: RequestId | None = None,
    timeout: float = MCP_CLIENT_TIMEOUT,
) -> ElicitResult | ErrorData:
    if not MCP_ELICITATION_AVAILABLE:
        return ErrorData(code=INTERNAL_ERROR, message="MCP elicitation is not available")
    if downstream_session is None:
        return ErrorData(code=INVALID_REQUEST, message="MCP elicitation requires a connected downstream MCP client")
    try:
        return await asyncio.wait_for(
            _relay_elicitation_to_downstream(params, downstream_session, downstream_capabilities, related_request_id),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return ErrorData(code=REQUEST_TIMEOUT, message="MCP elicitation timed out waiting for the downstream client")
    except Exception:
        verbose_logger.warning("MCP elicitation: downstream relay failed")
        return ErrorData(
            code=INTERNAL_ERROR, message="MCP elicitation failed while communicating with the downstream client"
        )


async def _relay_elicitation_to_downstream(
    params: ElicitRequestParams,
    downstream_session: _DownstreamElicitSession,
    downstream_capabilities: ClientCapabilities | None = None,
    related_request_id: RequestId | None = None,
) -> ElicitResult | ErrorData:
    capabilities: Final = downstream_capabilities.elicitation if downstream_capabilities is not None else None
    if capabilities is None:
        return ErrorData(code=INVALID_REQUEST, message="Downstream client has not advertised elicitation support")
    if isinstance(params, ElicitRequestURLParams):
        if capabilities.url is None:
            return ErrorData(code=INVALID_REQUEST, message="Downstream client does not support URL elicitation")
        return await downstream_session.elicit_url(
            message=params.message,
            url=params.url,
            elicitation_id=params.elicitation_id,
            related_request_id=related_request_id,
        )
    if capabilities.form is None and capabilities.url is not None:
        return ErrorData(code=INVALID_REQUEST, message="Downstream client does not support form elicitation")
    if not isinstance(params, ElicitRequestFormParams):
        return ErrorData(code=INVALID_REQUEST, message="Unsupported MCP elicitation parameters")
    return await downstream_session.elicit_form(
        message=params.message,
        requested_schema=params.requested_schema,
        related_request_id=related_request_id,
    )
