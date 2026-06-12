"""
FastAPI routes for the A2A registration flow.

Today this exposes a single endpoint, ``POST /v1/a2a/discover``, used by the
LiteLLM UI when an admin registers a new A2A agent: the UI hands us the
upstream agent's base URL, we fetch its well-known card, and we return the
raw card so the UI can render the agent's skills/capabilities and let the
admin pick which ones to expose through the proxy. The actual merge into a
LiteLLM-fronted card happens when the agent is saved via ``POST /v1/agents``.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.a2a.discovery import (
    AGENT_CARD_WELL_KNOWN_PATHS,
    AgentCardDiscoveryError,
    DiscoveryMode,
    fetch_well_known_card,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


class DiscoverAgentRequest(BaseModel):
    url: str = Field(
        ...,
        description=(
            "Base URL of the upstream agent. Behavior depends on "
            "``discovery_mode``: ``well_known_fallback`` (default) tries "
            f"{', '.join(AGENT_CARD_WELL_KNOWN_PATHS)} under this URL in "
            "order; ``langgraph_platform`` hits "
            "``/.well-known/agent-card.json?assistant_id=<id>`` instead."
        ),
    )
    discovery_mode: DiscoveryMode = Field(
        default=DiscoveryMode.WELL_KNOWN_FALLBACK,
        description=(
            "How to locate the upstream card. "
            "``well_known_fallback`` for pure A2A agents (try standard paths); "
            "``langgraph_platform`` for LangGraph Platform deployments where "
            "the card is shared across assistants and disambiguated by a "
            "query parameter."
        ),
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Mode-specific parameters. ``langgraph_platform`` requires "
            "``{'assistant_id': <id>}``. ``well_known_fallback`` ignores this."
        ),
    )


class DiscoverAgentResponse(BaseModel):
    url: str
    agent_card: Dict[str, Any]


@router.post(
    "/v1/a2a/discover",
    tags=["[beta] A2A Agents"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=DiscoverAgentResponse,
)
async def discover_agent_card(
    request: DiscoverAgentRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> JSONResponse:
    """
    Fetch the upstream agent's well-known card so the UI can show the admin
    which skills/capabilities the agent exposes.

    Only proxy admins can call this — the UI uses it during agent registration,
    and we don't want arbitrary keys probing internal URLs.

    Example:
    ```bash
    curl -X POST "http://localhost:4000/v1/a2a/discover" \\
        -H "Authorization: Bearer <admin_key>" \\
        -H "Content-Type: application/json" \\
        -d '{"url": "https://upstream-agent.example.com"}'
    ```
    """
    if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(
            status_code=403,
            detail=(
                "Only proxy admins can discover agent cards. "
                f"Your role={user_api_key_dict.user_role}"
            ),
        )

    try:
        card = await fetch_well_known_card(
            request.url,
            discovery_mode=request.discovery_mode,
            params=request.params,
        )
    except AgentCardDiscoveryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        verbose_proxy_logger.exception("Unexpected error during A2A discovery: %s", exc)
        raise HTTPException(status_code=500, detail=f"Discovery failed: {exc!s}")

    return JSONResponse(
        content={"url": request.url, "agent_card": card},
        media_type="application/json",
    )
