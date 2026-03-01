"""
Routing Group management endpoints.

Provides CRUD operations for routing groups — named, persisted routing pipelines
that combine model deployments with a routing strategy.
"""

import uuid
from collections import defaultdict
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.utils import get_prisma_client_or_throw
from litellm.types.router import (
    FailureInjectionConfig,
    RoutingGroupConfig,
    RoutingGroupDeployment,
    RoutingGroupListResponse,
    RoutingGroupSimulationResult,
    RoutingGroupTestResult,
)

router = APIRouter(
    tags=["routing group management"],
)


def _get_llm_router():
    """Get the LLM router from the proxy server."""
    from litellm.proxy.proxy_server import llm_router

    return llm_router


def _record_to_config(record) -> RoutingGroupConfig:
    """Convert a Prisma routing group record to a RoutingGroupConfig."""
    return RoutingGroupConfig(
        routing_group_id=record.routing_group_id,
        routing_group_name=record.routing_group_name,
        description=record.description,
        routing_strategy=record.routing_strategy,
        deployments=[
            RoutingGroupDeployment(**d) for d in (record.deployments or [])
        ],
        fallback_config=record.fallback_config,
        retry_config=record.retry_config,
        cooldown_config=record.cooldown_config,
        settings=record.settings,
        assigned_team_ids=list(record.assigned_team_ids or []),
        assigned_key_ids=list(record.assigned_key_ids or []),
        is_active=record.is_active,
    )


VALID_ROUTING_STRATEGIES = frozenset(
    {
        "priority-failover",
        "simple-shuffle",
        "least-busy",
        "latency-based-routing",
        "cost-based-routing",
        "usage-based-routing-v2",
        "weighted",
    }
)


async def _sync_routing_group_to_router(config: RoutingGroupConfig) -> None:
    """
    Translate a RoutingGroupConfig into live Router configuration.

    For priority-failover: creates tiered model groups + fallback chain.
    The fallback chain implements the per-group ordering independently of
    the shared router's routing_strategy — this strategy is fully isolated
    per group.

    For all other strategies: deployments are added to a single model group
    under the routing_group_name.  The shared router's routing_strategy is
    NOT changed here — that is the proxy-level concern set in the config YAML.
    Each group's deployments are correctly scoped to their own model group name;
    the router picks between them using whatever strategy the proxy was started
    with.  For "weighted" groups, per-deployment weights are stored in
    litellm_params so simple_shuffle (if active) will honour them
    automatically.
    """
    llm_router = _get_llm_router()
    if llm_router is None:
        verbose_proxy_logger.warning(
            "No LLM router available, routing group will be applied on next startup"
        )
        return

    strategy = config.routing_strategy
    group_name = config.routing_group_name

    if strategy == "priority-failover":
        priority_groups: dict = defaultdict(list)
        for dep in config.deployments:
            p = dep.priority if dep.priority is not None else 999
            priority_groups[p].append(dep)

        sorted_priorities = sorted(priority_groups.keys())
        group_names_by_priority = []

        for i, priority in enumerate(sorted_priorities):
            if i == 0:
                # Primary group uses the routing_group_name directly
                tier_group_name = group_name
            else:
                tier_group_name = f"{group_name}__fallback_p{priority}"
            group_names_by_priority.append(tier_group_name)

            for dep in priority_groups[priority]:
                try:
                    deployment_dict = {
                        "model_name": tier_group_name,
                        "litellm_params": {
                            "model": dep.model_name,
                        },
                        "model_info": {
                            "id": dep.model_id,
                        },
                    }
                    llm_router.add_deployment(
                        deployment=litellm.types.router.Deployment(**deployment_dict)
                    )
                except Exception as e:
                    verbose_proxy_logger.debug(
                        f"Could not add deployment {dep.model_id} to router: {e}"
                    )

        # Wire fallback chain
        if len(group_names_by_priority) > 1:
            primary = group_names_by_priority[0]
            fallbacks = group_names_by_priority[1:]
            existing_fallbacks = llm_router.fallbacks or []
            # Remove old entry for this group if it exists
            existing_fallbacks = [f for f in existing_fallbacks if primary not in f]
            existing_fallbacks.append({primary: fallbacks})
            llm_router.fallbacks = existing_fallbacks

    else:
        # All strategies except priority-failover: single model group.
        for dep in config.deployments:
            try:
                litellm_params: dict = {"model": dep.model_name}
                if strategy == "weighted" and dep.weight is not None:
                    # simple_shuffle reads `weight` from litellm_params and
                    # does a proportional random pick — this is how "weighted"
                    # is actually honoured at call time.
                    litellm_params["weight"] = dep.weight

                deployment_dict = {
                    "model_name": group_name,
                    "litellm_params": litellm_params,
                    "model_info": {
                        "id": dep.model_id,
                    },
                }
                llm_router.add_deployment(
                    deployment=litellm.types.router.Deployment(**deployment_dict)
                )
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Could not add deployment {dep.model_id} to router: {e}"
                )



# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/v1/routing_group",
    tags=["routing_group"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RoutingGroupConfig,
    status_code=status.HTTP_201_CREATED,
    summary="Create a routing group",
)
async def create_routing_group(
    data: RoutingGroupConfig,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> RoutingGroupConfig:
    """Create a new routing group."""
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    if data.routing_strategy not in VALID_ROUTING_STRATEGIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid routing_strategy '{data.routing_strategy}'. Must be one of: {sorted(VALID_ROUTING_STRATEGIES)}",
        )

    if not data.deployments:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="deployments must contain at least one deployment",
        )

    caller = user_api_key_dict.user_id or "unknown"
    routing_group_id = str(uuid.uuid4())

    try:
        created = await prisma_client.db.litellm_routinggrouptable.create(
            data={
                "routing_group_id": routing_group_id,
                "routing_group_name": data.routing_group_name,
                "description": data.description,
                "routing_strategy": data.routing_strategy,
                "deployments": [d.model_dump() for d in data.deployments],
                "fallback_config": data.fallback_config or {},
                "retry_config": data.retry_config or {},
                "cooldown_config": data.cooldown_config or {},
                "settings": data.settings or {},
                "assigned_team_ids": data.assigned_team_ids or [],
                "assigned_key_ids": data.assigned_key_ids or [],
                "is_active": data.is_active,
                "created_by": caller,
                "updated_by": caller,
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create routing group: {str(e)}",
        )

    result = _record_to_config(created)
    await _sync_routing_group_to_router(result)
    return result


@router.get(
    "/v1/routing_group",
    tags=["routing_group"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RoutingGroupListResponse,
    summary="List routing groups",
)
async def list_routing_groups(
    page: int = 1,
    size: int = 50,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> RoutingGroupListResponse:
    """List all routing groups."""
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    skip = (page - 1) * size
    rows = await prisma_client.db.litellm_routinggrouptable.find_many(
        skip=skip,
        take=size,
        order={"created_at": "desc"},
    )
    total = await prisma_client.db.litellm_routinggrouptable.count()

    groups = [_record_to_config(r) for r in rows]
    return RoutingGroupListResponse(
        routing_groups=groups, total=total, page=page, size=size
    )


@router.get(
    "/v1/routing_group/{routing_group_id}",
    tags=["routing_group"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RoutingGroupConfig,
    summary="Get a routing group",
)
async def get_routing_group(
    routing_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> RoutingGroupConfig:
    """Get a specific routing group by ID."""
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    row = await prisma_client.db.litellm_routinggrouptable.find_unique(
        where={"routing_group_id": routing_group_id}
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing group '{routing_group_id}' not found",
        )

    return _record_to_config(row)


@router.put(
    "/v1/routing_group/{routing_group_id}",
    tags=["routing_group"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RoutingGroupConfig,
    summary="Update a routing group",
)
async def update_routing_group(
    routing_group_id: str,
    data: RoutingGroupConfig,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> RoutingGroupConfig:
    """Update an existing routing group."""
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    caller = user_api_key_dict.user_id or "unknown"

    try:
        updated = await prisma_client.db.litellm_routinggrouptable.update(
            where={"routing_group_id": routing_group_id},
            data={
                "routing_group_name": data.routing_group_name,
                "description": data.description,
                "routing_strategy": data.routing_strategy,
                "deployments": [d.model_dump() for d in data.deployments],
                "fallback_config": data.fallback_config or {},
                "retry_config": data.retry_config or {},
                "cooldown_config": data.cooldown_config or {},
                "settings": data.settings or {},
                "assigned_team_ids": data.assigned_team_ids or [],
                "assigned_key_ids": data.assigned_key_ids or [],
                "is_active": data.is_active,
                "updated_by": caller,
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing group not found or update failed: {str(e)}",
        )

    result = _record_to_config(updated)
    await _sync_routing_group_to_router(result)
    return result


@router.delete(
    "/v1/routing_group/{routing_group_id}",
    tags=["routing_group"],
    dependencies=[Depends(user_api_key_auth)],
    summary="Delete a routing group",
)
async def delete_routing_group(
    routing_group_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Delete a routing group."""
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    try:
        await prisma_client.db.litellm_routinggrouptable.delete(
            where={"routing_group_id": routing_group_id}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing group not found: {str(e)}",
        )

    return {"routing_group_id": routing_group_id, "deleted": True}


@router.post(
    "/v1/routing_group/{routing_group_id}/test",
    tags=["routing_group"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RoutingGroupTestResult,
    summary="Test a routing group with a single request",
)
async def test_routing_group(
    routing_group_id: str,
    messages: Optional[List] = None,
    mock: bool = False,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> RoutingGroupTestResult:
    """Send a single test request through a routing group and return the routing trace."""
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    row = await prisma_client.db.litellm_routinggrouptable.find_unique(
        where={"routing_group_id": routing_group_id}
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing group '{routing_group_id}' not found",
        )

    llm_router = _get_llm_router()
    if llm_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM router not initialized",
        )

    from litellm.proxy.routing_group_utils.test_engine import RoutingGroupTestEngine

    config = _record_to_config(row)
    engine = RoutingGroupTestEngine()
    return await engine.test_single_request(
        routing_group_name=config.routing_group_name,
        router=llm_router,
        messages=messages,
        mock=mock,
    )


@router.post(
    "/v1/routing_group/{routing_group_id}/simulate",
    tags=["routing_group"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=RoutingGroupSimulationResult,
    summary="Simulate traffic through a routing group",
)
async def simulate_routing_group(
    routing_group_id: str,
    num_requests: int = 100,
    concurrency: int = 10,
    mock: bool = True,
    failure_injection: Optional[FailureInjectionConfig] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> RoutingGroupSimulationResult:
    """Simulate N requests through a routing group and return traffic distribution statistics."""
    prisma_client = get_prisma_client_or_throw(
        CommonProxyErrors.db_not_connected_error.value
    )

    row = await prisma_client.db.litellm_routinggrouptable.find_unique(
        where={"routing_group_id": routing_group_id}
    )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Routing group '{routing_group_id}' not found",
        )

    llm_router = _get_llm_router()
    if llm_router is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM router not initialized",
        )

    from litellm.proxy.routing_group_utils.test_engine import RoutingGroupTestEngine

    config = _record_to_config(row)
    engine = RoutingGroupTestEngine()
    return await engine.simulate_traffic(
        routing_group_name=config.routing_group_name,
        router=llm_router,
        routing_group_config=config,
        num_requests=num_requests,
        concurrency=concurrency,
        mock=mock,
        failure_injection=failure_injection,
    )
