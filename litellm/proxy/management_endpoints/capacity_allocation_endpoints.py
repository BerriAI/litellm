"""
Capacity allocation management.

This v0 endpoint lets admins append fixed-period capacity costs into the
daily team spend reporting table without changing per-request spend logs.
"""

from datetime import date
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from litellm._uuid import uuid
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
from litellm.proxy.utils import PrismaClient


router = APIRouter()

CAPACITY_ALLOCATION_PROVIDER = "capacity_allocation"
CAPACITY_ALLOCATION_ENDPOINT = "capacity_allocation"
CAPACITY_ALLOCATION_API_KEY_PREFIX = "capacity_allocation"


class CapacityAllocationSplitRule(str, Enum):
    EQUAL = "equal"
    USAGE_WEIGHTED = "usage_weighted"


class CapacityAllocationCostSegment(BaseModel):
    start_date: date
    end_date: date
    ptu_count: float = Field(gt=0)
    cost_per_ptu: Optional[float] = Field(default=None, ge=0)
    total_cost: Optional[float] = Field(default=None, ge=0)


class CapacityAllocationRunRequest(BaseModel):
    model: str = Field(min_length=1)
    model_group: Optional[str] = None
    start_date: date
    end_date: date
    team_ids: List[str] = Field(min_length=1)
    split_rule: CapacityAllocationSplitRule = CapacityAllocationSplitRule.EQUAL
    cost_segments: List[CapacityAllocationCostSegment] = Field(min_length=1)
    allocation_id: Optional[str] = None
    description: Optional[str] = None


class CapacityAllocationTeamResult(BaseModel):
    team_id: str
    spend: float
    weight: float


class CapacityAllocationRunResponse(BaseModel):
    allocation_id: str
    model: str
    model_group: Optional[str]
    start_date: date
    end_date: date
    split_rule: CapacityAllocationSplitRule
    total_cost: float
    inserted_rows: int
    allocations: List[CapacityAllocationTeamResult]


def _get_segment_cost(segment: CapacityAllocationCostSegment) -> float:
    if segment.end_date < segment.start_date:
        raise ValueError("segment end_date must be on or after start_date")
    if segment.total_cost is not None:
        return segment.total_cost
    if segment.cost_per_ptu is None:
        raise ValueError("cost_per_ptu is required when total_cost is not set")
    return segment.ptu_count * segment.cost_per_ptu


def calculate_total_capacity_cost(
    segments: List[CapacityAllocationCostSegment],
) -> float:
    return sum(_get_segment_cost(segment) for segment in segments)


def calculate_capacity_allocation_amounts(
    *,
    team_ids: List[str],
    total_cost: float,
    split_rule: CapacityAllocationSplitRule,
    usage_spend_by_team: Optional[Dict[str, float]] = None,
) -> List[CapacityAllocationTeamResult]:
    if not team_ids:
        raise ValueError("team_ids must contain at least one team")

    if split_rule == CapacityAllocationSplitRule.EQUAL:
        weights = {team_id: 1 / len(team_ids) for team_id in team_ids}
    else:
        usage_spend_by_team = usage_spend_by_team or {}
        total_usage_spend = sum(
            max(usage_spend_by_team.get(team_id, 0.0), 0.0) for team_id in team_ids
        )
        if total_usage_spend <= 0:
            weights = {team_id: 1 / len(team_ids) for team_id in team_ids}
        else:
            weights = {
                team_id: max(usage_spend_by_team.get(team_id, 0.0), 0.0)
                / total_usage_spend
                for team_id in team_ids
            }

    allocations: List[CapacityAllocationTeamResult] = []
    remaining_cost = total_cost
    for index, team_id in enumerate(team_ids):
        if index == len(team_ids) - 1:
            spend = round(remaining_cost, 6)
        else:
            spend = round(total_cost * weights[team_id], 6)
            remaining_cost -= spend
        allocations.append(
            CapacityAllocationTeamResult(
                team_id=team_id,
                spend=spend,
                weight=weights[team_id],
            )
        )
    return allocations


async def _get_usage_spend_by_team(
    *,
    prisma_client: PrismaClient,
    team_ids: List[str],
    start_date: date,
    end_date: date,
    model: str,
) -> Dict[str, float]:
    spend_rows = await prisma_client.db.litellm_dailyteamspend.find_many(
        where={
            "team_id": {"in": team_ids},
            "date": {"gte": str(start_date), "lte": str(end_date)},
            "model": model,
        },
        select={"team_id": True, "spend": True, "custom_llm_provider": True},
    )

    spend_by_team = {team_id: 0.0 for team_id in team_ids}
    for row in spend_rows:
        if row.custom_llm_provider == CAPACITY_ALLOCATION_PROVIDER:
            continue
        if row.team_id in spend_by_team:
            spend_by_team[row.team_id] += row.spend
    return spend_by_team


def _build_daily_team_spend_rows(
    *,
    request_data: CapacityAllocationRunRequest,
    allocation_id: str,
    allocations: List[CapacityAllocationTeamResult],
) -> List[Dict[str, Any]]:
    allocation_date = str(request_data.end_date)
    return [
        {
            "team_id": allocation.team_id,
            "date": allocation_date,
            "api_key": (
                f"{CAPACITY_ALLOCATION_API_KEY_PREFIX}:"
                f"{allocation_id}:{allocation.team_id}"
            ),
            "model": request_data.model,
            "model_group": request_data.model_group,
            "custom_llm_provider": CAPACITY_ALLOCATION_PROVIDER,
            "endpoint": CAPACITY_ALLOCATION_ENDPOINT,
            "spend": allocation.spend,
            "api_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        for allocation in allocations
    ]


@router.post(
    "/team/capacity_allocation/run",
    tags=["team management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=CapacityAllocationRunResponse,
)
async def run_team_capacity_allocation(
    data: CapacityAllocationRunRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    from litellm.proxy.proxy_server import prisma_client

    if not _user_has_admin_view(user_api_key_dict):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error": "Only proxy admins can run capacity allocations."},
        )

    if prisma_client is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    if data.end_date < data.start_date:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "end_date must be on or after start_date"},
        )

    try:
        total_cost = calculate_total_capacity_cost(data.cost_segments)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(exc)},
        ) from exc

    usage_spend_by_team: Optional[Dict[str, float]] = None
    if data.split_rule == CapacityAllocationSplitRule.USAGE_WEIGHTED:
        usage_spend_by_team = await _get_usage_spend_by_team(
            prisma_client=prisma_client,
            team_ids=data.team_ids,
            start_date=data.start_date,
            end_date=data.end_date,
            model=data.model,
        )

    allocations = calculate_capacity_allocation_amounts(
        team_ids=data.team_ids,
        total_cost=total_cost,
        split_rule=data.split_rule,
        usage_spend_by_team=usage_spend_by_team,
    )
    allocation_id = data.allocation_id or str(uuid.uuid4())
    rows = _build_daily_team_spend_rows(
        request_data=data,
        allocation_id=allocation_id,
        allocations=allocations,
    )

    create_result = await prisma_client.db.litellm_dailyteamspend.create_many(data=rows)
    inserted_rows = getattr(create_result, "count", len(rows))

    return CapacityAllocationRunResponse(
        allocation_id=allocation_id,
        model=data.model,
        model_group=data.model_group,
        start_date=data.start_date,
        end_date=data.end_date,
        split_rule=data.split_rule,
        total_cost=total_cost,
        inserted_rows=inserted_rows,
        allocations=allocations,
    )
