"""Wire format for the hydrate protocol (LIT-2890).

The proxy assembles a ``HydratePayload`` per session-attach and pushes it
into the warm VM via SSM RunCommand (or long-poll, depending on the
configured transport). The daemon writes:

  ``env_vars``        -> ``/etc/litellm-agent/env``               (mode 0644)
  ``secrets``         -> ``/etc/litellm-agent/secrets.env``       (mode 0600)
  ``network_access``  -> ``iptables`` rules applied BEFORE user code runs
  ``agent_config``    -> daemon's runtime config

Schema is stable across daemon and proxy. Bumping a field requires a
coordinated daemon release — keep additions backward-compatible (use
``Optional`` + default ``None`` for new fields).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RepoSpec(BaseModel):
    """Single git repo the daemon clones at hydrate time."""

    url: str
    ref: Optional[str] = Field(
        default=None, description="branch / tag / sha; daemon defaults to main"
    )


class NetworkAccess(BaseModel):
    """Egress firewall posture applied to the VM before user code runs."""

    mode: Literal["allow_all", "allowlist"] = "allow_all"
    allowlist: Optional[List[str]] = Field(
        default=None,
        description=(
            "When mode=='allowlist', daemon installs iptables rules permitting "
            "only these hostnames. Ignored when mode=='allow_all'."
        ),
    )


class AgentConfig(BaseModel):
    """Per-session daemon runtime config (model, prompt, PR auto-create flag)."""

    model: str
    system_prompt: str = ""
    auto_create_pr: bool = False


class HydratePayload(BaseModel):
    """Full hydrate payload pushed to the warm VM at session-attach.

    Locked schema per LIT-2890. The daemon reads this and transitions from
    ``LITELLM_AGENT_MODE=warm`` to ``LITELLM_AGENT_MODE=session``.
    """

    session_id: str
    agent_id: str
    jwt: str = Field(description="Session-scoped JWT, replaces the per-VM pool JWT.")
    jwt_expires_at: str = Field(description="ISO-8601 UTC expiry timestamp.")
    repos: List[RepoSpec] = Field(default_factory=list)
    env_vars: Dict[str, str] = Field(default_factory=dict)
    secrets: Dict[str, str] = Field(
        default_factory=dict,
        description="Sensitive values written to disk mode 0600. Never logged.",
    )
    network_access: NetworkAccess = Field(default_factory=NetworkAccess)
    agent_config: AgentConfig
