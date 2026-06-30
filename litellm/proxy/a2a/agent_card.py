"""
Pure logic for merging an upstream A2A agent card with LiteLLM-specific overrides.

The merge produces the card that LiteLLM exposes to A2A clients at
``/a2a/{agent_id}/.well-known/agent-card.json``. The upstream card is taken as
the base; specific fields are replaced so all traffic flows through the proxy
and uses LiteLLM auth.
"""

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

# Protocol version LiteLLM speaks. Bump when the proxy's A2A surface changes.
LITELLM_A2A_PROTOCOL_VERSION = "1.0"

# Security scheme exposed by the LiteLLM-fronted agent card. Always replaces
# whatever upstream advertised — the client must authenticate to the proxy,
# not the upstream agent.
LITELLM_SECURITY_SCHEMES: Dict[str, Dict[str, Any]] = {
    "LiteLLMKey": {
        "type": "http",
        "scheme": "bearer",
        "description": "LiteLLM virtual key",
    },
}

LITELLM_SECURITY_REQUIREMENTS: List[Dict[str, List[str]]] = [{"LiteLLMKey": []}]

# Capabilities LiteLLM can faithfully proxy today. Anything not in this set is
# dropped during merge so we don't advertise behavior the proxy can't deliver.
#
# TODO: re-enable ``streaming`` once the A2A streaming endpoint at
#   ``POST /a2a/{agent_id}/message/stream`` is exercised end-to-end with
#   cost tracking + guardrails. It's wired in ``a2a_endpoints.py`` but not
#   yet covered by tests, so we keep it gated on the upstream advertising it.
# TODO: ``pushNotifications`` — proxy has no webhook plumbing yet.
# TODO: ``extendedAgentCard`` — no separate authenticated-extended-card
#   endpoint exposed by the proxy.
# TODO: ``extensions`` — protocol extensions aren't validated/forwarded yet.
_ALLOWED_CAPABILITY_KEYS = {"streaming"}

# v1.0 AgentCard top-level fields. Anything else is stripped from the merged
# card as a defense against upstream drift. ``supportedInterfaces`` is kept
# verbatim per product spec even though it is not in the v1.0 schema — clients
# that expect it will find it; clients that don't will ignore it.
#
# ``additionalInterfaces`` is deliberately excluded: it advertises alternate
# upstream URLs (HTTP/JSONRPC/gRPC backends) that, if persisted and served,
# would let authenticated agent callers reach the backend directly and bypass
# the proxy's auth/budget/logging. The proxy publishes its own entrypoint via
# ``supportedInterfaces`` instead.
_ALLOWED_TOP_LEVEL_KEYS = {
    "protocolVersion",
    "name",
    "description",
    "version",
    "capabilities",
    "defaultInputModes",
    "defaultOutputModes",
    "skills",
    "preferredTransport",
    "supportedInterfaces",
    "iconUrl",
    "provider",
    "documentationUrl",
    "securitySchemes",
    "security",
    "supportsAuthenticatedExtendedCard",
    "signatures",
    # ``url`` is retained on the stored card because the runtime A2A invocation
    # path (``a2a_endpoints.py``) reads ``agent.agent_card_params['url']`` to
    # locate the upstream backend. The public ``/.well-known/agent-card.json``
    # endpoint rewrites this field to the proxy URL before serving it to
    # clients, so retaining it here does not leak the upstream to A2A callers.
    "url",
}

_DEFAULT_SKILLS: List[Dict[str, Any]] = [
    {
        "id": "chat",
        "name": "Chat",
        "description": "Conversational interaction with the agent.",
        "tags": ["chat"],
    }
]

_DEFAULT_MODES: List[str] = ["text"]

# Fallback ``version`` when the upstream card omits the field. The A2A v1.0
# schema requires ``version`` on every card, so without this default the
# merged card would fail validation on clients that ``model_validate`` it.
_DEFAULT_AGENT_VERSION = "1.0.0"


def _filter_capabilities(upstream_capabilities: Any) -> Dict[str, Any]:
    """Return a capabilities dict containing only allowlisted, truthy keys."""
    if not isinstance(upstream_capabilities, dict):
        return {}
    return {
        key: value for key, value in upstream_capabilities.items() if key in _ALLOWED_CAPABILITY_KEYS and bool(value)
    }


def _default_litellm_provider(proxy_base_url: str) -> Dict[str, str]:
    return {"organization": "LiteLLM Proxy", "url": proxy_base_url}


def merge_agent_card(
    upstream_card: Optional[Mapping[str, Any]],
    *,
    proxy_url: str,
    proxy_base_url: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the LiteLLM-fronted agent card.

    Args:
        upstream_card: Card returned by the upstream agent's well-known endpoint.
            May be ``None``/empty when the upstream did not expose one.
        proxy_url: Full URL clients should hit to invoke this agent through
            the proxy, e.g. ``https://proxy.example.com/a2a/<agent_id>``.
        proxy_base_url: Root URL of the LiteLLM proxy, used as a fallback when
            we synthesize a provider record.
        name: User-supplied agent name from the LiteLLM UI. Takes precedence
            over the upstream card's ``name``.
        description: User-supplied description from the LiteLLM UI. Takes
            precedence over the upstream card's ``description``.

    Returns:
        A dict suitable for serving as the proxy's agent card. Only keys in
        the v1.0 AgentCard schema (plus ``supportedInterfaces``) are emitted.
    """
    base: Dict[str, Any] = deepcopy(dict(upstream_card)) if upstream_card else {}

    # Keep the upstream ``url`` on the stored card: the runtime A2A
    # invocation path reads it from ``agent_card_params`` to know where to
    # proxy requests. The public well-known endpoint rewrites this field
    # to the proxy URL before exposing the card to clients.

    base["protocolVersion"] = LITELLM_A2A_PROTOCOL_VERSION

    if name:
        base["name"] = name
    if description:
        base["description"] = description

    if not base.get("version"):
        base["version"] = _DEFAULT_AGENT_VERSION

    base["capabilities"] = _filter_capabilities(base.get("capabilities"))

    if not base.get("skills"):
        base["skills"] = deepcopy(_DEFAULT_SKILLS)
    if not base.get("defaultInputModes"):
        base["defaultInputModes"] = list(_DEFAULT_MODES)
    if not base.get("defaultOutputModes"):
        base["defaultOutputModes"] = list(_DEFAULT_MODES)

    if not base.get("provider"):
        base["provider"] = _default_litellm_provider(proxy_base_url)

    base["supportedInterfaces"] = [
        {
            "url": proxy_url,
            "protocolBinding": "JSONRPC",
            "protocolVersion": LITELLM_A2A_PROTOCOL_VERSION,
        }
    ]

    base["securitySchemes"] = deepcopy(LITELLM_SECURITY_SCHEMES)
    # Use the standard A2A/OpenAPI ``security`` field for requirements, not
    # the non-standard ``securityRequirements`` alias. The upstream's own
    # ``security`` selector is overwritten here because the proxy enforces its
    # own scheme regardless of what upstream required.
    base["security"] = deepcopy(LITELLM_SECURITY_REQUIREMENTS)

    return {key: value for key, value in base.items() if key in _ALLOWED_TOP_LEVEL_KEYS}
