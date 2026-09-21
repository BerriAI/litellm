from __future__ import annotations

import httpx
from integration.cost_calculation.conftest import (
    CostBreakdown,
    CostRow,
    approx_equal,
    assert_total_is_sum_of_components,
)
from integration.cost_calculation.cost_tracking_case import ExactExpected, RecountExpected


def assert_breakdown(
    case_name: str,
    response_content_type: str,
    expected: ExactExpected,
    breakdown: CostBreakdown,
    response: httpx.Response | None,
) -> None:
    if response is None:
        assert not expected.cost_header, f"{case_name}: cost headers require an HTTP response"
    assert breakdown.input_cost is not None and approx_equal(breakdown.input_cost, expected.input_cost), (
        f"{case_name}: input_cost {breakdown.input_cost} != expected {expected.input_cost}"
    )
    assert breakdown.output_cost is not None and approx_equal(breakdown.output_cost, expected.output_cost), (
        f"{case_name}: output_cost {breakdown.output_cost} != expected {expected.output_cost}"
    )
    for field, header_name, actual_component, expected_component in (
        (
            "cache_read_cost",
            "x-litellm-response-cost-cache-read",
            breakdown.cache_read_cost,
            expected.cache_read_cost,
        ),
        (
            "cache_creation_cost",
            "x-litellm-response-cost-cache-creation",
            breakdown.cache_creation_cost,
            expected.cache_creation_cost,
        ),
        (
            "reasoning_cost",
            "x-litellm-response-cost-reasoning",
            breakdown.reasoning_cost,
            expected.reasoning_cost,
        ),
        (
            "tool_usage_cost",
            "x-litellm-response-cost-tool-usage",
            breakdown.tool_usage_cost,
            expected.tool_usage_cost,
        ),
    ):
        if expected_component is None:
            continue
        omitted_component_allowed: bool = expected_component == 0.0
        assert (actual_component is None and omitted_component_allowed) or (
            actual_component is not None and approx_equal(actual_component, expected_component)
        ), f"{case_name}: {field} {actual_component} != expected {expected_component}"
        if response is not None and expected.cost_header and response_content_type == "application/json":
            header: str | None = response.headers.get(header_name)
            assert (header is None and omitted_component_allowed) or (
                header is not None and approx_equal(float(header), expected_component)
            ), f"{case_name}: {header_name} {header} != expected {expected_component}"
    if response is not None and expected.cost_header and response_content_type == "application/json" and any(
        component is not None
        for component in (
            expected.cache_read_cost,
            expected.cache_creation_cost,
            expected.reasoning_cost,
            expected.tool_usage_cost,
        )
    ):
        input_header: str | None = response.headers.get("x-litellm-response-cost-input")
        output_header: str | None = response.headers.get("x-litellm-response-cost-output")
        expected_input_header: float = expected.input_cost - (
            expected.cache_read_cost or 0.0
        ) - (expected.cache_creation_cost or 0.0)
        assert input_header is not None and approx_equal(float(input_header), expected_input_header), (
            f"{case_name}: x-litellm-response-cost-input {input_header} != expected {expected_input_header}"
        )
        assert output_header is not None and approx_equal(float(output_header), expected.output_cost), (
            f"{case_name}: x-litellm-response-cost-output {output_header} != expected {expected.output_cost}"
        )


def assert_exact(
    case_name: str,
    response_content_type: str,
    expected: ExactExpected,
    row: CostRow,
    response: httpx.Response | None,
) -> None:
    assert row.spend is not None and approx_equal(row.spend, expected.spend), (
        f"{case_name}: spend {row.spend} != expected {expected.spend} "
        f"(breakdown {row.breakdown.model_dump() if row.breakdown is not None else None})"
    )
    breakdown: CostBreakdown | None = row.breakdown
    if expected.breakdown_persisted:
        assert breakdown is not None, f"{case_name}: no cost_breakdown persisted"
    if breakdown is not None:
        assert_breakdown(case_name, response_content_type, expected, breakdown, response)
    assert row.prompt_tokens == expected.prompt_tokens, (
        f"{case_name}: prompt_tokens {row.prompt_tokens} != expected {expected.prompt_tokens}"
    )
    assert row.completion_tokens == expected.completion_tokens, (
        f"{case_name}: completion_tokens {row.completion_tokens} != expected {expected.completion_tokens}"
    )
    if breakdown is not None:
        assert_total_is_sum_of_components(row, breakdown, case_name)


def assert_recount(case_name: str, expected: RecountExpected, row: CostRow) -> None:
    assert row.prompt_tokens is not None and row.prompt_tokens > 0, (
        f"{case_name}: recount case counted no input tokens: prompt_tokens={row.prompt_tokens}"
    )
    assert row.completion_tokens is not None and row.completion_tokens > 0, (
        f"{case_name}: recount case counted no output tokens: completion_tokens={row.completion_tokens}"
    )
    if expected.prompt_tokens is not None:
        assert row.prompt_tokens == expected.prompt_tokens, (
            f"{case_name}: prompt_tokens {row.prompt_tokens} != pinned {expected.prompt_tokens}"
        )
    if expected.completion_tokens is not None:
        assert row.completion_tokens == expected.completion_tokens, (
            f"{case_name}: completion_tokens {row.completion_tokens} != pinned {expected.completion_tokens}"
        )
    if expected.min_completion_tokens is not None:
        assert row.completion_tokens >= expected.min_completion_tokens, (
            f"{case_name}: completion_tokens {row.completion_tokens} < minimum {expected.min_completion_tokens}"
        )
    if expected.max_completion_tokens is not None:
        assert row.completion_tokens <= expected.max_completion_tokens, (
            f"{case_name}: completion_tokens {row.completion_tokens} > maximum {expected.max_completion_tokens}"
        )
    recount: float = row.prompt_tokens * expected.recount.input_cost_per_token + (
        row.completion_tokens * expected.recount.output_cost_per_token
    )
    assert row.spend is not None and approx_equal(row.spend, recount), (
        f"{case_name}: spend {row.spend} != recount {recount} at map rates"
    )
    assert row.breakdown is not None, f"{case_name}: no cost_breakdown persisted"
    assert_total_is_sum_of_components(row, row.breakdown, case_name)
