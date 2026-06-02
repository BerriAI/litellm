"""Hydrate transports — how the payload reaches the warm VM.

We chose **SSM RunCommand push** based on B0's 1700ms median warm-attach
latency (LIT-2888). The long-poll alternative is intentionally not shipped
in this PR — it can be added later by implementing the same
``HydrateTransport`` protocol.

Public surface:
- ``HydrateTransport`` — protocol every transport implements
- ``SSMHydrateTransport`` — the production transport
"""

from litellm.proxy.agent_session_endpoints.warm_pool.transports.ssm import (
    HydrateTransport,
    HydrateTransportError,
    SSMHydrateTransport,
)

__all__ = [
    "HydrateTransport",
    "HydrateTransportError",
    "SSMHydrateTransport",
]
