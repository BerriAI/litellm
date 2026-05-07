from datetime import date

from litellm.proxy.management_endpoints.capacity_allocation_endpoints import (
    CAPACITY_ALLOCATION_ENDPOINT,
    CAPACITY_ALLOCATION_PROVIDER,
    CapacityAllocationCostSegment,
    CapacityAllocationRunRequest,
    CapacityAllocationSplitRule,
    _build_daily_team_spend_rows,
    calculate_capacity_allocation_amounts,
    calculate_total_capacity_cost,
)


def test_calculate_total_capacity_cost_from_segments_with_mid_period_change():
    total = calculate_total_capacity_cost(
        [
            CapacityAllocationCostSegment(
                start_date=date(2026, 3, 1),
                end_date=date(2026, 3, 15),
                ptu_count=1,
                cost_per_ptu=200,
            ),
            CapacityAllocationCostSegment(
                start_date=date(2026, 3, 15),
                end_date=date(2026, 3, 31),
                ptu_count=10,
                cost_per_ptu=200,
            ),
        ]
    )

    assert total == 2200


def test_calculate_total_capacity_cost_uses_explicit_segment_total():
    total = calculate_total_capacity_cost(
        [
            CapacityAllocationCostSegment(
                start_date=date(2026, 3, 1),
                end_date=date(2026, 3, 31),
                ptu_count=10,
                total_cost=733.33,
            )
        ]
    )

    assert total == 733.33


def test_calculate_capacity_allocation_amounts_equal_split():
    allocations = calculate_capacity_allocation_amounts(
        team_ids=["team-a", "team-b", "team-c"],
        total_cost=2200,
        split_rule=CapacityAllocationSplitRule.EQUAL,
    )

    assert [allocation.team_id for allocation in allocations] == [
        "team-a",
        "team-b",
        "team-c",
    ]
    assert [allocation.spend for allocation in allocations] == [
        733.333333,
        733.333333,
        733.333334,
    ]
    assert sum(allocation.spend for allocation in allocations) == 2200


def test_calculate_capacity_allocation_amounts_usage_weighted():
    allocations = calculate_capacity_allocation_amounts(
        team_ids=["team-a", "team-b", "team-c"],
        total_cost=1000,
        split_rule=CapacityAllocationSplitRule.USAGE_WEIGHTED,
        usage_spend_by_team={"team-a": 100, "team-b": 300, "team-c": 600},
    )

    assert [allocation.spend for allocation in allocations] == [100, 300, 600]
    assert [allocation.weight for allocation in allocations] == [0.1, 0.3, 0.6]


def test_calculate_capacity_allocation_amounts_usage_weighted_falls_back_to_equal():
    allocations = calculate_capacity_allocation_amounts(
        team_ids=["team-a", "team-b"],
        total_cost=100,
        split_rule=CapacityAllocationSplitRule.USAGE_WEIGHTED,
        usage_spend_by_team={"team-a": 0, "team-b": 0},
    )

    assert [allocation.spend for allocation in allocations] == [50, 50]
    assert [allocation.weight for allocation in allocations] == [0.5, 0.5]


def test_build_daily_team_spend_rows_labels_allocations_without_request_usage():
    request_data = CapacityAllocationRunRequest(
        model="azure/gpt-4",
        model_group="ptu-pool",
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 31),
        team_ids=["team-a", "team-b"],
        split_rule=CapacityAllocationSplitRule.EQUAL,
        cost_segments=[
            CapacityAllocationCostSegment(
                start_date=date(2026, 3, 1),
                end_date=date(2026, 3, 31),
                ptu_count=2,
                cost_per_ptu=100,
            )
        ],
    )
    allocations = calculate_capacity_allocation_amounts(
        team_ids=request_data.team_ids,
        total_cost=200,
        split_rule=CapacityAllocationSplitRule.EQUAL,
    )

    rows = _build_daily_team_spend_rows(
        request_data=request_data,
        allocation_id="alloc-123",
        allocations=allocations,
    )

    assert len(rows) == 2
    assert rows[0]["api_key"] == "capacity_allocation:alloc-123:team-a"
    assert rows[0]["custom_llm_provider"] == CAPACITY_ALLOCATION_PROVIDER
    assert rows[0]["endpoint"] == CAPACITY_ALLOCATION_ENDPOINT
    assert rows[0]["spend"] == 100
    assert rows[0]["api_requests"] == 0
    assert rows[0]["prompt_tokens"] == 0
