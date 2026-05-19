"""
Capability discovery endpoints.

GET /v1/capabilities
    Returns a single JSON document describing what the authenticated caller
    can use across model / agent / mcp / skill, plus access_group bundles.
    Shape: ``litellm.types.capabilities.CapabilitiesResponse``.

GET /.well-known/xct-capabilities
    Public (unauthenticated) variant; only public-flagged entities, with
    server URLs and credential metadata stripped.

S1-03 status: 6-level permission filter wired up. Admin still sees
everything. Non-admin gets:
  - models: intersection of key.models / team.models / proxy_model_list
            via auth.model_checks.get_complete_model_list — the same
            function /v1/models uses.
  - agents: explicit_grants ∪ public ∪ owned (the visibility set from
            the S3-01 hotfix).
  - mcps:   MCPRequestHandler.get_allowed_mcp_servers (the existing
            6-level handler; empty = no explicit restriction = visible).
  - skills: still stub until S2-04.
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


def _build_model_summary(model_name: str, info: Dict[str, Any]) -> ModelSummary:
    return ModelSummary(
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


def _collect_models(
    user_api_key_dict: UserAPIKeyAuth,
    *,
    admin: bool,
) -> List[ModelSummary]:
    """
    Models visible to the caller.

    Admin: every entry in ``litellm.model_cost``.
    Non-admin: intersection of key.models / team.models / proxy_model_list
    via ``auth.model_checks.get_complete_model_list`` — same filter
    ``/v1/models`` uses, so the discovery list matches what the caller
    can actually invoke through ``/v1/chat/completions``.
    """
    model_costs: Dict[str, Any] = getattr(litellm, "model_cost", {}) or {}

    if admin:
        return [
            _build_model_summary(name, info)
            for name, info in model_costs.items()
            if isinstance(info, dict)
        ]

    try:
        from litellm.proxy import proxy_server as _proxy_server
        from litellm.proxy.auth.model_checks import get_complete_model_list

        proxy_models = [
            (m.get("model_name") if isinstance(m, dict) else None)
            for m in (_proxy_server.llm_model_list or [])
        ]
        proxy_models = [m for m in proxy_models if m]

        allowed = get_complete_model_list(
            key_models=user_api_key_dict.models or [],
            team_models=getattr(user_api_key_dict, "team_models", []) or [],
            proxy_model_list=proxy_models,
            user_model=None,
            infer_model_from_keys=False,
        )
    except Exception as e:
        verbose_proxy_logger.debug("capabilities: model filter fell back to []: %s", e)
        allowed = []

    allowed_set = set(allowed)
    return [
        _build_model_summary(name, info)
        for name, info in model_costs.items()
        if isinstance(info, dict) and name in allowed_set
    ]


def _build_agent_summary(agent) -> AgentSummary:
    card = agent.agent_card_params or {}
    return AgentSummary(
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


async def _collect_agents(
    user_api_key_dict: UserAPIKeyAuth,
    *,
    admin: bool,
) -> List[AgentSummary]:
    """
    Agents visible to the caller.

    Admin: every agent in the registry.
    Non-admin: explicit_grants ∪ public ∪ owned, via the same
    ``_resolve_visible_agent_ids`` helper used by ``GET /v1/agents``
    (S3-01 hotfix). Empty grants do NOT fall back to "see everything".
    """
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        all_agents = global_agent_registry.get_agent_list()
    except Exception as e:
        verbose_proxy_logger.debug("capabilities: skipped agents (%s)", e)
        return []

    if admin:
        return [_build_agent_summary(a) for a in all_agents]

    try:
        from litellm.proxy.agent_endpoints.endpoints import (
            _resolve_visible_agent_ids,
        )

        visible_ids = await _resolve_visible_agent_ids(user_api_key_dict)
    except Exception as e:
        verbose_proxy_logger.debug("capabilities: agent filter fell back to []: %s", e)
        return []

    return [_build_agent_summary(a) for a in all_agents if a.agent_id in visible_ids]


def _build_mcp_summary(server) -> McpSummary:
    return McpSummary(
        server_id=server.server_id,
        server_name=server.name,
        alias=getattr(server, "alias", None),
        transport=getattr(server, "transport", None),
        access_groups=getattr(server, "mcp_access_groups", []) or [],
        auth_type=getattr(server, "auth_type", None),
        needs_oauth=getattr(server, "auth_type", None) == "oauth2",
    )


async def _collect_mcps(
    user_api_key_dict: UserAPIKeyAuth,
    *,
    admin: bool,
) -> List[McpSummary]:
    """
    MCP servers visible to the caller.

    Admin: every registered server.
    Non-admin: the set returned by ``MCPRequestHandler.get_allowed_mcp_servers``
    (the existing 6-level handler). When that returns ``[]`` the proxy's
    historical contract is "no explicit restriction = nothing scoped to me";
    we follow that convention here so discovery matches what
    ``/v1/mcp/tools`` already exposes.
    """
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        servers = global_mcp_server_manager.get_registered_mcp_servers()
    except Exception as e:
        verbose_proxy_logger.debug("capabilities: skipped mcps (%s)", e)
        return []

    if admin:
        return [_build_mcp_summary(s) for s in servers]

    try:
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )

        allowed = set(
            await MCPRequestHandler.get_allowed_mcp_servers(
                user_api_key_auth=user_api_key_dict
            )
        )
    except Exception as e:
        verbose_proxy_logger.debug("capabilities: mcp filter fell back to []: %s", e)
        return []

    return [_build_mcp_summary(s) for s in servers if s.server_id in allowed]


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
    """Return capabilities visible to the caller (admin: full; else: scoped)."""
    admin = _is_admin(user_api_key_dict)
    return CapabilitiesResponse(
        caller=_build_caller(user_api_key_dict),
        models=_collect_models(user_api_key_dict, admin=admin),
        agents=await _collect_agents(user_api_key_dict, admin=admin),
        mcps=await _collect_mcps(user_api_key_dict, admin=admin),
        skills=await _collect_skills(),
        access_groups=await _collect_access_groups(),
    )
