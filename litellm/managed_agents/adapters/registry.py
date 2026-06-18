"""Adapter registry — selects the right adapter for a sandbox type.

Today ships only `"opencode"`. Adding a new sandbox provider means adding
a new module under `litellm/managed_agents/adapters/` and registering it
here — no other code change is required (the public API stays the same;
the `sandbox.type` field selects the adapter).
"""

from litellm.managed_agents.adapters.base import SandboxAdapter
from litellm.managed_agents.adapters.opencode import OpencodeAdapter

# Single shared instance — adapters are stateless, so this is safe and
# avoids per-request allocation.
_OPENCODE = OpencodeAdapter()


def get_adapter(sandbox_type: str) -> SandboxAdapter:
    """Return the adapter for a given `sandbox.type`.

    Raises `ValueError` for unknown sandbox types so the endpoint handler
    can translate to a 422 response.
    """
    if sandbox_type == "opencode":
        return _OPENCODE
    raise ValueError(f"Unknown sandbox type: {sandbox_type!r}")
