"""
ID generation helpers for the agent_session_endpoints module.

Each ID is ``<prefix>_<32 hex chars>`` (16 random bytes). The prefixes line
up with the model namespaces so logs and DB rows are self-describing:

  agent_<...>   -> LiteLLM_Agent
  sess_<...>    -> LiteLLM_AgentSession
  run_<...>     -> LiteLLM_AgentRun

Implemented as a single ``new_id`` helper rather than three near-duplicate
functions to keep the convention enforced in one place.
"""

import secrets

from litellm.proxy.agent_session_endpoints.constants import (
    AGENT_ID_PREFIX,
    RUN_ID_PREFIX,
    SESSION_ID_PREFIX,
)


def _new_id(prefix: str) -> str:
    """Return ``f"{prefix}{16 random bytes hex}"``."""
    return f"{prefix}{secrets.token_hex(16)}"


def new_agent_id() -> str:
    return _new_id(AGENT_ID_PREFIX)


def new_session_id() -> str:
    return _new_id(SESSION_ID_PREFIX)


def new_run_id() -> str:
    return _new_id(RUN_ID_PREFIX)
