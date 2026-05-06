"""Warm pool subsystem for `agent_session_endpoints` (LIT-2890).

Pre-provisions a small pool of EC2 (or other provider) VMs per team so
``POST /v2/sessions`` can attach a fully-booted VM in <3s instead of the
30-90s cold-boot path.

Public surface:
- ``WarmPoolManager`` — async maintenance loop (start it on proxy boot)
- ``attach_warm_vm`` — race-safe attach + hydrate, called from session create
- ``HydratePayload`` — wire format the daemon receives
"""

from litellm.proxy.agent_session_endpoints.warm_pool.attach import (
    AttachResult,
    attach_warm_vm,
)
from litellm.proxy.agent_session_endpoints.warm_pool.hydrate import (
    build_hydrate_payload,
)
from litellm.proxy.agent_session_endpoints.warm_pool.manager import WarmPoolManager
from litellm.proxy.agent_session_endpoints.warm_pool.types import (
    AgentConfig,
    HydratePayload,
    NetworkAccess,
    RepoSpec,
)

__all__ = [
    "AgentConfig",
    "AttachResult",
    "HydratePayload",
    "NetworkAccess",
    "RepoSpec",
    "WarmPoolManager",
    "attach_warm_vm",
    "build_hydrate_payload",
]
