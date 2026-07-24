"""Typed fault values for upstream OAuth/DCR failures (phase 1 of the MCP error-handling framework).

The invariant this package exists to enforce: an upstream failure is classified ONCE into a single
fault value, and the response status, wire error code, and prose are all derived from that value.
Deriving all three from one classification makes contradictory pairings (a caller-fault error code on
a server-fault status) unrepresentable, and gives the trust-boundary rule one enforcement point:
spec-defined machine fields may cross to callers, upstream prose and raw bodies go to server logs.
"""

from litellm.proxy._experimental.mcp_server.faults.classify import (
    classify_upstream_dcr_rejection,
    classify_upstream_token_rejection,
)
from litellm.proxy._experimental.mcp_server.faults.render_oauth import (
    dcr_fault_detail,
    render_token_fault,
)
from litellm.proxy._experimental.mcp_server.faults.traversal import iter_exception_tree
from litellm.proxy._experimental.mcp_server.faults.types import (
    CallerRejected,
    CredentialSource,
    GatewayRejected,
    UpstreamOAuthFault,
    UpstreamProtocolFault,
    UpstreamReportedFault,
)

__all__ = [
    "CallerRejected",
    "CredentialSource",
    "GatewayRejected",
    "UpstreamOAuthFault",
    "UpstreamProtocolFault",
    "UpstreamReportedFault",
    "classify_upstream_dcr_rejection",
    "classify_upstream_token_rejection",
    "dcr_fault_detail",
    "iter_exception_tree",
    "render_token_fault",
]
