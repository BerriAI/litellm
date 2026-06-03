"""
Protocol-level routing for LiteLLM.

Provides strict mode to enforce protocol compatibility without automatic conversion.
"""

from litellm.protocol_routing._types import (
    SupportedProtocol,
    ProtocolRoutingMode,
    ProtocolMismatchError,
    get_protocol_routing_mode,
    set_protocol_routing_mode,
)
from litellm.protocol_routing._mapping import (
    PROVIDER_DEFAULT_PROTOCOLS,
    infer_protocols,
)
from litellm.protocol_routing._filter import (
    filter_deployments_by_protocol,
    check_strict_protocol_for_provider,
)

__all__ = [
    # Types
    "SupportedProtocol",
    "ProtocolRoutingMode",
    "ProtocolMismatchError",
    # Configuration
    "get_protocol_routing_mode",
    "set_protocol_routing_mode",
    # Provider mapping
    "PROVIDER_DEFAULT_PROTOCOLS",
    "infer_protocols",
    # Filtering
    "filter_deployments_by_protocol",
    "check_strict_protocol_for_provider",
]
