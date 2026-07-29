"""
A2A registration helpers for the LiteLLM proxy.

- ``discovery``: fetches the upstream agent's well-known card so the UI can
  display its skills/capabilities for the user to pick from.
- ``agent_card``: pure merge logic that builds the LiteLLM-fronted agent card
  from the upstream card + the values the user set in the UI.
- ``endpoints``: FastAPI routes that wire the above into the proxy.
"""

from litellm.proxy.a2a.agent_card import (
    LITELLM_A2A_PROTOCOL_VERSION,
    LITELLM_SECURITY_SCHEMES,
    LITELLM_SECURITY_REQUIREMENTS,
    merge_agent_card,
)
from litellm.proxy.a2a.discovery import (
    AGENT_CARD_WELL_KNOWN_PATHS,
    fetch_well_known_card,
)

__all__ = [
    "AGENT_CARD_WELL_KNOWN_PATHS",
    "LITELLM_A2A_PROTOCOL_VERSION",
    "LITELLM_SECURITY_REQUIREMENTS",
    "LITELLM_SECURITY_SCHEMES",
    "fetch_well_known_card",
    "merge_agent_card",
]
