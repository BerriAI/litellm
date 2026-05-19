"""
Capability discovery endpoints.

GET /v1/capabilities
    Returns a single JSON document describing what the authenticated caller
    can use across model / agent / mcp / skill, plus access_group bundles.
    Shape: ``litellm.types.capabilities.CapabilitiesResponse``.

GET /.well-known/xct-capabilities
    Public (unauthenticated) variant; only public-flagged entities, with
    server URLs and credential metadata stripped.

S1-02 scope: endpoint skeleton — returns the full registry without permission
filtering. S1-03 wires the 6-level filter; S1-04 adds caching; S1-05 implements
the public variant; S1-06 enriches model entries with capability flags.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.capabilities import (
    AccessGroupSummary,
    AgentSummary,
    CapabilitiesResponse,
    CapabilityCaller,
    McpSummary,
    ModelCapabilityFlags,
    ModelSummary,
    SkillSummary,
)

router = APIRouter()


def _is_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    role = user_api_key_dict.user_role
    return (
        role == LitellmUserRoles.PROXY_ADMIN
        or role == LitellmUserRoles.PROXY_ADMIN.value
    )


def _build_caller(user_api_key_dict: UserAPIKeyAuth) -> CapabilityCaller:
    return CapabilityCaller(
        key_id=user_api_key_dict.api_key,
        team_id=user_api_key_dict.team_id,
        user_id=user_api_key_dict.user_id,
        org_id=getattr(user_api_key_dict, "org_id", None),
        app_id=getattr(user_api_key_dict, "app_id", None),
        is_admin=_is_admin(user_api_key_dict),
    )


def _collect_models() -> List[ModelSummary]:
    """Snapshot of all configured models. S1-03 narrows by caller scope."""
    summaries: List[ModelSummary] = []
    seen: set = set()
    model_costs: Dict[str, Any] = getattr(litellm, "model_cost", {}) or {}
    for model_name, info in model_costs.items():
        if model_name in seen or not isinstance(info, dict):
            continue
        seen.add(model_name)
        summaries.append(
            ModelSummary(
                id=model_name,
                provider=info.get("litellm_provider"),
                mode=info.get("mode"),
                context_window=info.get("max_input_tokens") or info.get("max_tokens"),
                max_output_tokens=info.get("max_output_tokens"),
                input_cost_per_token=info.get("input_cost_per_token"),
                output_cost_per_token=info.get("output_cost_per_token"),
                # capability flags filled in S1-06
                capabilities=ModelCapabilityFlags(),
            )
        )
    return summaries


def _collect_agents() -> List[AgentSummary]:
    """Snapshot of agents from the registry. S1-03 narrows by caller scope."""
    summaries: List[AgentSummary] = []
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        for agent in global_agent_registry.get_agent_list():
            card = agent.agent_card_params or {}
            summaries.append(
                AgentSummary(
                    agent_id=agent.agent_id,
                    agent_name=agent.agent_name,
                    description=card.get("description"),
                    version=card.get("version"),
                    is_public=bool(
                        litellm.public_agent_groups
                        and agent.agent_id in litellm.public_agent_groups
                    ),
                    supports_streaming=bool(
                        (card.get("capabilities") or {}).get("streaming", False)
                    ),
                    agent_card_url=f"/a2a/{agent.agent_id}/.well-known/agent-card.json",
                )
            )
    except Exception as e:
        verbose_proxy_logger.debug("capabilities: skipped agents (%s)", e)
    return summaries


async def _collect_mcps() -> List[McpSummary]:
    """Snapshot of MCP servers from the manager. S1-03 narrows by caller scope."""
    summaries: List[McpSummary] = []
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        for server in global_mcp_server_manager.get_registered_mcp_servers():
            summaries.append(
                McpSummary(
                    server_id=server.server_id,
                    server_name=server.name,
                    alias=getattr(server, "alias", None),
                    transport=getattr(server, "transport", None),
                    access_groups=getattr(server, "mcp_access_groups", []) or [],
                    auth_type=getattr(server, "auth_type", None),
                    needs_oauth=getattr(server, "auth_type", None) == "oauth2",
                )
            )
    except Exception as e:
        verbose_proxy_logger.debug("capabilities: skipped mcps (%s)", e)
    return summaries


async def _collect_skills() -> List[SkillSummary]:
    """Placeholder until S2-04 wires native skill table reads."""
    return []


async def _collect_access_groups() -> List[AccessGroupSummary]:
    """Placeholder until access-group enumeration is added."""
    return []


@router.get(
    "/v1/capabilities",
    response_model=CapabilitiesResponse,
    tags=["[beta] Capabilities"],
)
async def get_capabilities(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> CapabilitiesResponse:
    """
    Return the capabilities visible to the caller.

    S1-02: skeleton — no scoping yet, admin sees everything, non-admin
    currently also sees everything (will be tightened in S1-03).
    """
    return CapabilitiesResponse(
        caller=_build_caller(user_api_key_dict),
        models=_collect_models(),
        agents=_collect_agents(),
        mcps=await _collect_mcps(),
        skills=await _collect_skills(),
        access_groups=await _collect_access_groups(),
    )
