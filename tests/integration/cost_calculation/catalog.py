"""Every cost tracking case, gathered from the per-provider literal modules under cases/."""

from __future__ import annotations

import json
from typing import Final

from integration.cost_calculation.cases import (
    anthropic,
    azure,
    batch_realtime,
    bedrock_converse,
    fireworks_ai,
    gemini,
    misc,
    openai,
    together_ai,
    vertex_ai,
)
from integration.cost_calculation.cost_tracking_case import (
    COST_MAP,
    PRIOR_RESPONSE_ID_MARKER,
    BatchCostCase,
    CostTrackingTestCase,
    ExactExpected,
    FailureExpected,
    JsonResponse,
    RealtimeCostCase,
    RecountExpected,
    SseResponse,
)
from integration.cost_calculation.stream_parity.case import StreamParityTestCase

_PROVIDER_MODULES: Final = (
    openai,
    anthropic,
    gemini,
    bedrock_converse,
    vertex_ai,
    azure,
    fireworks_ai,
    together_ai,
    misc,
)
CASES: Final[tuple[CostTrackingTestCase, ...]] = tuple(case for module in _PROVIDER_MODULES for case in module.CASES)
PARITY_CASES: Final[tuple[StreamParityTestCase, ...]] = tuple(
    pair for module in _PROVIDER_MODULES for pair in module.PARITY
)
BATCH_CASES: Final[tuple[BatchCostCase, ...]] = batch_realtime.BATCH_CASES
REALTIME_CASES: Final[tuple[RealtimeCostCase, ...]] = batch_realtime.REALTIME_CASES
_PARITY_LEGS: Final = frozenset(leg.name for pair in PARITY_CASES for leg in (pair.plain, pair.streamed))
STANDALONE_CASES: Final[tuple[CostTrackingTestCase, ...]] = tuple(
    case for case in CASES if case.name not in _PARITY_LEGS
)
_ALL_CASES: Final = CASES + BATCH_CASES + REALTIME_CASES
_LITELLM_MODELS: Final = tuple(case.litellm_model for case in _ALL_CASES)


def data_errors() -> tuple[str, ...]:
    case_models: Final = frozenset(case.model for case in _ALL_CASES) | frozenset(
        case.session_model for case in REALTIME_CASES if case.session_model is not None
    )
    unknown_models: Final = sorted(model for model in case_models if model not in COST_MAP)
    missing_cases: Final = sorted(model for model in COST_MAP if model not in case_models)
    duplicate_names: Final = sorted(
        name for name in {case.name for case in _ALL_CASES} if sum(case.name == name for case in _ALL_CASES) > 1
    )
    input_rates: Final = tuple(
        (entry.litellm_provider, entry.input_cost_per_token, model)
        for model, entry in COST_MAP.items()
        if entry.mode != "realtime"
    )
    shared_input_rate_details: Final = tuple(
        (
            provider,
            rate,
            tuple(
                model
                for candidate_provider, value, model in input_rates
                if candidate_provider == provider and value == rate
            ),
        )
        for provider, rate in frozenset((provider, rate) for provider, rate, _ in input_rates if rate is not None)
    )
    shared_input_rates: Final = sorted(
        f"{rate}: {models}" for _, rate, models in shared_input_rate_details if len(models) > 1
    )
    recount_mismatches: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, RecountExpected)
        and case.model in COST_MAP
        and (
            case.expected.recount.input_cost_per_token != (COST_MAP[case.model].input_cost_per_token or 0.0)
            or case.expected.recount.output_cost_per_token != (COST_MAP[case.model].output_cost_per_token or 0.0)
        )
    )
    component_mismatches: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, ExactExpected)
        and any(
            component is not None
            for component in (
                case.expected.cache_read_cost,
                case.expected.cache_creation_cost,
                case.expected.reasoning_cost,
                case.expected.tool_usage_cost,
            )
        )
        and (
            (case.expected.cache_read_cost or 0.0) + (case.expected.cache_creation_cost or 0.0)
            > case.expected.input_cost
            or (case.expected.reasoning_cost or 0.0) > case.expected.output_cost
            or not _approx_equal(
                case.expected.input_cost + case.expected.output_cost + (case.expected.tool_usage_cost or 0.0),
                case.expected.spend,
            )
        )
    )
    failure_response_mismatches: Final = sorted(
        case.name
        for case in CASES
        if (
            isinstance(case.expected, FailureExpected)
            and (
                not isinstance(case.response, JsonResponse)
                or not 400 <= case.response.status <= 599
                or not 400 <= case.expected.failure.status <= 599
            )
        )
        or (
            not isinstance(case.expected, FailureExpected)
            and isinstance(case.response, JsonResponse)
            and case.response.status != 200
        )
    )
    invalid_opt_outs: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, ExactExpected)
        and (
            (
                not case.expected.breakdown_persisted
                and case.passthrough_provider is None
                and case.rates.mode != "image_generation"
                and not case.reports_provider_cost
            )
            or (
                not case.expected.cost_header
                and case.passthrough_provider is None
                and not isinstance(case.response, SseResponse)
                and case.expected.spend != 0.0
            )
        )
    )
    invalid_fallbacks: Final = sorted(
        case.name
        for case in CASES
        if case.fallback_from is not None
        and (not isinstance(case.fallback_from, JsonResponse) or not 400 <= case.fallback_from.status <= 599)
    )
    invalid_disconnects: Final = sorted(
        case.name
        for case in CASES
        if case.disconnect_after_frames is not None
        and (
            not isinstance(case.response, SseResponse)
            or case.response.frame_delay_ms <= 0
            or not isinstance(case.expected, RecountExpected)
        )
    )
    invalid_rollup_ids: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, ExactExpected)
        and case.expected.rollups
        and "$UNIQUE_ID" not in case.response.model_dump_json()
    )
    invalid_pinned_tool_ids: Final = sorted(
        case.name
        for case in CASES
        if isinstance(case.expected, RecountExpected)
        and (case.expected.prompt_tokens is not None or case.expected.completion_tokens is not None)
        and any(
            marker in case.response.model_dump_json()
            for marker in ('"id": "call_$REQUEST_ID"', '"id": "toolu_$REQUEST_ID"')
        )
    )
    invalid_prior_response_chains: Final = sorted(
        case.name
        for case in CASES
        if PRIOR_RESPONSE_ID_MARKER in json.dumps(case.request) and not case.can_chain_prior_response
    )
    return tuple(
        message
        for message in (
            f"case models absent from cost_map: {unknown_models}" if unknown_models else None,
            f"cost-map entries without cases: {missing_cases}" if missing_cases else None,
            f"duplicate case names: {duplicate_names}" if duplicate_names else None,
            f"cost-map entries share input_cost_per_token: {shared_input_rates}" if shared_input_rates else None,
            f"recount rates differ from cost-map rates: {recount_mismatches}" if recount_mismatches else None,
            f"breakdown components are inconsistent: {component_mismatches}" if component_mismatches else None,
            f"failure response statuses are inconsistent: {failure_response_mismatches}"
            if failure_response_mismatches
            else None,
            f"invalid passthrough opt-outs: {invalid_opt_outs}" if invalid_opt_outs else None,
            f"invalid fallback responses: {invalid_fallbacks}" if invalid_fallbacks else None,
            f"invalid disconnect cases: {invalid_disconnects}" if invalid_disconnects else None,
            f"rollup responses lack $UNIQUE_ID: {invalid_rollup_ids}" if invalid_rollup_ids else None,
            f"pinned tool IDs contain $REQUEST_ID: {invalid_pinned_tool_ids}" if invalid_pinned_tool_ids else None,
            f"{PRIOR_RESPONSE_ID_MARKER} needs a non-rollup, non-failure /v1/responses JSON response with a string id"
            f" as previous_response_id: {invalid_prior_response_chains}"
            if invalid_prior_response_chains
            else None,
        )
        if message is not None
    )


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)
