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

import hashlib
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
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
    PublicCapabilitiesResponse,
    SkillSummary,
)

router = APIRouter()

# 60s TTL is short enough that permission changes propagate quickly without
# hammering the registry/DB on every UI render. Override via env in tests.
CAPABILITIES_CACHE_TTL = int(os.environ.get("LITELLM_CAPABILITIES_CACHE_TTL", "60"))
# Singleton DualCache; populated lazily so test code can swap it.
_capabilities_cache: Optional[DualCache] = None


def _get_capabilities_cache() -> DualCache:
    global _capabilities_cache
    if _capabilities_cache is None:
        _capabilities_cache = DualCache(
            default_in_memory_ttl=CAPABILITIES_CACHE_TTL,
            default_redis_ttl=CAPABILITIES_CACHE_TTL,
        )
    return _capabilities_cache


def _hash_token(token: Optional[str]) -> str:
    """Stable, non-reversible identifier for cache keys."""
    return hashlib.sha256((token or "anon").encode("utf-8")).hexdigest()[:16]


def _capabilities_cache_key(uak: UserAPIKeyAuth) -> str:
    """Cache key = (hashed token, app_id). Different callers never share entries."""
    return "capabilities:{token}:{app}".format(
        token=_hash_token(uak.api_key),
        app=getattr(uak, "app_id", None) or "none",
    )


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


def _compute_model_capability_flags(
    model_name: str, info: Dict[str, Any]
) -> ModelCapabilityFlags:
    """Snap booleans from litellm.utils.supports_* + the model_cost row.

    Each ``supports_*`` helper takes (model, provider) and returns False if the
    model isn't recognized — perfect default for unknown providers. We also
    cross-check the ``info`` dict so unit tests that supply model_cost without
    populating ``model_prices`` still get sensible values.
    """
    try:
        from litellm.utils import (
            supports_audio_input,
            supports_audio_output,
            supports_function_calling,
            supports_pdf_input,
            supports_prompt_caching,
            supports_response_schema,
            supports_vision,
            supports_web_search,
        )
    except Exception:
        return ModelCapabilityFlags()

    provider = info.get("litellm_provider")

    def _safe(fn, *args) -> bool:
        try:
            return bool(fn(*args))
        except Exception:
            return False

    return ModelCapabilityFlags(
        vision=_safe(supports_vision, model_name, provider)
        or bool(info.get("supports_vision")),
        function_calling=_safe(supports_function_calling, model_name, provider)
        or bool(info.get("supports_function_calling")),
        structured_output=_safe(supports_response_schema, model_name, provider)
        or bool(info.get("supports_response_schema")),
        prompt_caching=_safe(supports_prompt_caching, model_name, provider)
        or bool(info.get("supports_prompt_caching")),
        pdf_input=_safe(supports_pdf_input, model_name, provider)
        or bool(info.get("supports_pdf_input")),
        web_search=_safe(supports_web_search, model_name, provider)
        or bool(info.get("supports_web_search")),
        audio_input=_safe(supports_audio_input, model_name, provider)
        or bool(info.get("supports_audio_input")),
        audio_output=_safe(supports_audio_output, model_name, provider)
        or bool(info.get("supports_audio_output")),
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
        capabilities=_compute_model_capability_flags(model_name, info),
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


async def _collect_skills(
    user_api_key_dict: UserAPIKeyAuth,
    *,
    admin: bool,
) -> List[SkillSummary]:
    """xct-native skills visible to the caller.

    Same scoping shape as ``/v1/xct-skills`` list: admin sees every
    ``source='custom'`` row; non-admin sees ``is_public OR user_id=me
    OR team_id=mine``. We re-implement the filter inline instead of
    importing from skill_endpoints to avoid a circular import.
    """
    try:
        from litellm.proxy.proxy_server import prisma_client
    except Exception:
        return []
    if prisma_client is None:
        return []

    where: Dict[str, Any] = {"source": "custom"}
    if not admin:
        caller_clauses: List[Dict[str, Any]] = [{"is_public": True}]
        if user_api_key_dict.user_id:
            caller_clauses.append({"user_id": user_api_key_dict.user_id})
        if user_api_key_dict.team_id:
            caller_clauses.append({"team_id": user_api_key_dict.team_id})
        where = {"AND": [where, {"OR": caller_clauses}]}

    try:
        rows = await prisma_client.db.litellm_skillstable.find_many(where=where)
    except Exception as e:
        verbose_proxy_logger.debug("capabilities: skill filter fell back to []: %s", e)
        return []

    summaries: List[SkillSummary] = []
    for row in rows:
        summaries.append(
            SkillSummary(
                skill_id=row.skill_id,
                display_title=getattr(row, "display_title", None),
                description=getattr(row, "description", None),
                version=getattr(row, "version", None),
                source=getattr(row, "source", "custom"),
                category=(getattr(row, "xct_metadata", None) or {}).get("category"),
                is_public=bool(getattr(row, "is_public", False)),
            )
        )
    return summaries


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
    """Return capabilities visible to the caller (admin: full; else: scoped).

    Result is cached for ``CAPABILITIES_CACHE_TTL`` seconds keyed on the
    caller's hashed token + app_id. Permission/grant changes propagate at
    most that fast; ``invalidate_capabilities_cache_for_caller`` lets us
    punch the entry sooner from key/team mutation handlers (wired in S4).
    """
    cache = _get_capabilities_cache()
    cache_key = _capabilities_cache_key(user_api_key_dict)

    hit = await cache.async_get_cache(cache_key)
    if hit is not None:
        _record_cache_metric(hit=True)
        # DualCache stores either a Pydantic model or its dict form depending
        # on serializer. Normalize back to the response model.
        if isinstance(hit, CapabilitiesResponse):
            return hit
        if isinstance(hit, dict):
            return CapabilitiesResponse(**hit)

    _record_cache_metric(hit=False)

    admin = _is_admin(user_api_key_dict)
    response = CapabilitiesResponse(
        caller=_build_caller(user_api_key_dict),
        models=_collect_models(user_api_key_dict, admin=admin),
        agents=await _collect_agents(user_api_key_dict, admin=admin),
        mcps=await _collect_mcps(user_api_key_dict, admin=admin),
        skills=await _collect_skills(user_api_key_dict, admin=admin),
        access_groups=await _collect_access_groups(),
    )

    # S4-08: app-tenancy intersect. If the caller carries an app_id AND
    # that app declares a capability_scope_id (= access group), the
    # caller's visible models/agents/mcps are FURTHER NARROWED to the
    # intersection with that access group. Empty intersection → empty
    # list (don't fall back to "all").
    if not admin:
        response = await _intersect_with_app_capability_scope(
            response, user_api_key_dict
        )

    await cache.async_set_cache(cache_key, response, ttl=CAPABILITIES_CACHE_TTL)
    return response


async def _intersect_with_app_capability_scope(
    response: CapabilitiesResponse,
    user_api_key_dict: UserAPIKeyAuth,
) -> CapabilitiesResponse:
    """If the caller's app has a capability_scope_id set, narrow each list."""
    app_id = getattr(user_api_key_dict, "app_id", None)
    if not app_id:
        return response
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return response
        # Look up the app and resolve capability_scope_id → access group row.
        # find_unique on app_id is O(1) thanks to the PK; we already cache the
        # whole capability response so this DB hit is once per cache window.
        app_row = await prisma_client.db.litellm_xctapptable.find_unique(
            where={"app_id": app_id}
        )
        if app_row is None:
            return response
        scope_id = getattr(app_row, "capability_scope_id", None)
        if not scope_id:
            return response
        group = await prisma_client.db.litellm_accessgrouptable.find_unique(
            where={"access_group_id": scope_id}
        )
        if group is None:
            return response

        allowed_models = set(group.access_model_names or [])
        allowed_agents = set(group.access_agent_ids or [])
        allowed_mcps = set(group.access_mcp_server_ids or [])

        response.models = [m for m in response.models if m.id in allowed_models]
        response.agents = [a for a in response.agents if a.agent_id in allowed_agents]
        response.mcps = [m for m in response.mcps if m.server_id in allowed_mcps]
        # skills + access_groups untouched — access groups don't constrain those today
    except Exception as e:
        verbose_proxy_logger.debug(
            "capability_scope intersect failed for app=%s: %s", app_id, e
        )
    return response


def _record_cache_metric(*, hit: bool) -> None:
    """Increment Prometheus counters if Prometheus is enabled. No-op otherwise."""
    try:
        from prometheus_client import Counter  # type: ignore

        global _capabilities_cache_hit, _capabilities_cache_miss
        if "_capabilities_cache_hit" not in globals():
            _capabilities_cache_hit = Counter(
                "litellm_capability_cache_hit_total",
                "Number of /v1/capabilities responses served from cache.",
            )
            _capabilities_cache_miss = Counter(
                "litellm_capability_cache_miss_total",
                "Number of /v1/capabilities responses computed (cache miss).",
            )
        (_capabilities_cache_hit if hit else _capabilities_cache_miss).inc()
    except Exception:  # pragma: no cover — metrics are best-effort
        pass


async def invalidate_capabilities_cache_for_caller(
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """Punch a specific caller's cache entry.

    Wired from key/team mutation paths once those handlers grow the
    capability-aware logic; safe to call from anywhere — silently no-ops
    if the cache hasn't been initialized yet.
    """
    cache = _get_capabilities_cache()
    try:
        await cache.async_delete_cache(_capabilities_cache_key(user_api_key_dict))
    except AttributeError:
        # Older DualCache without async_delete_cache; fall back to set with
        # a 1-second TTL, expiring within the next polling window.
        await cache.async_set_cache(
            _capabilities_cache_key(user_api_key_dict), None, ttl=1
        )


# ============================================================================
# S1-05: public capabilities discovery (unauthenticated)
# ============================================================================

# Wider TTL — public data changes much less often than per-caller scopes.
PUBLIC_CAPABILITIES_CACHE_TTL = int(
    os.environ.get("LITELLM_PUBLIC_CAPABILITIES_CACHE_TTL", "300")
)
_PUBLIC_CACHE_KEY = "capabilities:public"


def _public_model_allowlist() -> set:
    """Public-model allowlist source — same convention as public_agent_groups.

    Mirrors ``litellm.public_agent_groups``: a list configured at startup
    (via config.yaml or env) of model names that are safe to advertise
    anonymously. Empty/unset == nothing public.
    """
    return set(getattr(litellm, "public_model_groups", None) or [])


def _public_mcp_allowlist() -> set:
    """Public-mcp allowlist source. Empty == nothing public."""
    return set(getattr(litellm, "public_mcp_groups", None) or [])


async def _build_public_response() -> PublicCapabilitiesResponse:
    """Snapshot only public-flagged entities. Credential / server URL omitted."""
    model_costs: Dict[str, Any] = getattr(litellm, "model_cost", {}) or {}
    public_models = _public_model_allowlist()
    public_mcps = _public_mcp_allowlist()
    public_agents = set(getattr(litellm, "public_agent_groups", None) or [])

    models = [
        _build_model_summary(name, info)
        for name, info in model_costs.items()
        if isinstance(info, dict) and name in public_models
    ]

    agents: List[AgentSummary] = []
    try:
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        for agent in global_agent_registry.get_agent_list():
            if agent.agent_id in public_agents:
                agents.append(_build_agent_summary(agent))
    except Exception as e:
        verbose_proxy_logger.debug("public capabilities: skipped agents (%s)", e)

    mcps: List[McpSummary] = []
    try:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        for server in global_mcp_server_manager.get_registered_mcp_servers():
            if server.server_id not in public_mcps:
                continue
            # The public summary intentionally omits transport / auth_type /
            # access_groups — those leak deployment topology. Keep only the
            # bare minimum a marketing page would need.
            mcps.append(
                McpSummary(
                    server_id=server.server_id,
                    server_name=server.name,
                    needs_oauth=getattr(server, "auth_type", None) == "oauth2",
                )
            )
    except Exception as e:
        verbose_proxy_logger.debug("public capabilities: skipped mcps (%s)", e)

    skills: List[SkillSummary] = []
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is not None:
            rows = await prisma_client.db.litellm_skillstable.find_many(
                where={"AND": [{"source": "custom"}, {"is_public": True}]}
            )
            for row in rows:
                skills.append(
                    SkillSummary(
                        skill_id=row.skill_id,
                        display_title=getattr(row, "display_title", None),
                        description=getattr(row, "description", None),
                        version=getattr(row, "version", None),
                        source=getattr(row, "source", "custom"),
                        category=(getattr(row, "xct_metadata", None) or {}).get(
                            "category"
                        ),
                        is_public=True,
                    )
                )
    except Exception as e:
        verbose_proxy_logger.debug("public capabilities: skipped skills (%s)", e)

    return PublicCapabilitiesResponse(
        models=models, agents=agents, mcps=mcps, skills=skills
    )


@router.get(
    "/.well-known/xct-capabilities",
    response_model=PublicCapabilitiesResponse,
    tags=["[beta] Capabilities"],
)
async def get_public_capabilities() -> PublicCapabilitiesResponse:
    """Anonymous discovery — only public-flagged entities. 5-min cache."""
    cache = _get_capabilities_cache()
    hit = await cache.async_get_cache(_PUBLIC_CACHE_KEY)
    if hit is not None:
        if isinstance(hit, PublicCapabilitiesResponse):
            return hit
        if isinstance(hit, dict):
            return PublicCapabilitiesResponse(**hit)

    response = await _build_public_response()
    await cache.async_set_cache(
        _PUBLIC_CACHE_KEY, response, ttl=PUBLIC_CAPABILITIES_CACHE_TTL
    )
    return response
