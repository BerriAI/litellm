"""Sandbox adapters for managed agents v2.

Each adapter translates the sandbox-agnostic public API
(`/v2/sessions/:id/messages`, `/v2/sessions/:id/events`, etc.) into the
provider-specific HTTP calls described in contract §7. Today only
`opencode` ships — additional providers register here without touching
the public API surface.

Public exports:
  - `SandboxAdapter` — Protocol all adapters implement.
  - `SandboxUnreachableError` — raised on connect/timeout failures
    (handler maps to 504).
  - `SandboxBadGatewayError` — raised on malformed sandbox responses
    (handler maps to 502).
  - `OpencodeAdapter` — the shipped adapter for `sandbox.type="opencode"`.
  - `get_adapter` — registry lookup; raises `ValueError` for unknown
    sandbox types.
  - `normalize_opencode_message` / `normalize_opencode_event` — pure
    helpers, exported for unit tests.
"""

from litellm.managed_agents.adapters.base import (
    SandboxAdapter,
    SandboxBadGatewayError,
    SandboxUnreachableError,
)
from litellm.managed_agents.adapters.normalization import (
    event_matches_session,
    normalize_opencode_event,
    normalize_opencode_message,
)
from litellm.managed_agents.adapters.opencode import OpencodeAdapter
from litellm.managed_agents.adapters.registry import get_adapter

__all__ = [
    "SandboxAdapter",
    "SandboxUnreachableError",
    "SandboxBadGatewayError",
    "OpencodeAdapter",
    "get_adapter",
    "normalize_opencode_message",
    "normalize_opencode_event",
    "event_matches_session",
]
