"""Batch and realtime cost cases (moved from cost_tracking_cases.json)."""

from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    BatchCostCase,
    BatchOutputLine,
    ExactExpected,
    RealtimeCostCase,
    RealtimeTurn,
)

GPT_5_6_BATCH_HALVED_RATES_WHEN_MAP_HAS_NO_BATCH_KEYS: Final = BatchCostCase(
    name="gpt-5.6-batch-halved_rates_when_map_has_no_batch_keys",
    covers="quota_management.spend_tracking.batch_costs.fallback_rates",
    model="gpt-5.6",
    litellm_model="openai/gpt-5.6",
    output_lines=(
        BatchOutputLine(status_code=200, prompt_tokens=100, completion_tokens=50),
        BatchOutputLine(status_code=200, prompt_tokens=120, completion_tokens=30),
        BatchOutputLine(status_code=400),
    ),
    expected=ExactExpected(
        spend=0.0007525,
        input_cost=0.0001925,
        output_cost=0.00056,
        prompt_tokens=220,
        completion_tokens=80,
        cost_header=False,
    ),
)

GPT_5_6_BATCH_CACHED_INPUT_HALVED: Final = BatchCostCase(
    name="gpt-5.6-batch-cached_input_halved",
    covers="quota_management.spend_tracking.batch_costs.cached_input",
    model="gpt-5.6",
    litellm_model="openai/gpt-5.6",
    output_lines=(BatchOutputLine(status_code=200, prompt_tokens=100, completion_tokens=10, cached_tokens=40),),
    expected=ExactExpected(
        spend=0.000126,
        input_cost=5.6e-05,
        output_cost=7e-05,
        prompt_tokens=100,
        completion_tokens=10,
        cost_header=False,
    ),
)

GPT_5_4_BATCH_EXPLICIT_BATCH_RATES_BILL_CACHED_AT_BATCH_INPUT_RATE: Final = BatchCostCase(
    name="gpt-5.4-batch-explicit_batch_rates_bill_cached_at_batch_input_rate",
    covers="quota_management.spend_tracking.batch_costs.explicit_rates",
    model="gpt-5.4",
    litellm_model="openai/gpt-5.4",
    output_lines=(
        BatchOutputLine(status_code=200, prompt_tokens=100, completion_tokens=50, cached_tokens=40),
        BatchOutputLine(status_code=200, prompt_tokens=120, completion_tokens=30),
    ),
    expected=ExactExpected(
        spend=0.000875,
        input_cost=0.000275,
        output_cost=0.0006,
        prompt_tokens=220,
        completion_tokens=80,
        cost_header=False,
    ),
)

GPT_5_6_BATCH_ALL_REQUESTS_FAILED_ZERO_SPEND: Final = BatchCostCase(
    name="gpt-5.6-batch-all_requests_failed_zero_spend",
    covers="quota_management.spend_tracking.batch_costs.failed_requests",
    model="gpt-5.6",
    litellm_model="openai/gpt-5.6",
    output_lines=(
        BatchOutputLine(status_code=400),
        BatchOutputLine(status_code=400),
    ),
    expected=ExactExpected(
        spend=0.0, input_cost=0.0, output_cost=0.0, prompt_tokens=0, completion_tokens=0, cost_header=False
    ),
)

GPT_REALTIME_MINI_2025_12_15_REALTIME_SINGLE_TURN_TEXT_AUDIO_CACHED: Final = RealtimeCostCase(
    name="gpt-realtime-mini-2025-12-15-realtime-single_turn_text_audio_cached",
    covers="quota_management.spend_tracking.realtime_costs.single_turn",
    model="gpt-realtime-mini-2025-12-15",
    litellm_model="openai/gpt-realtime-mini-2025-12-15",
    turns=(
        RealtimeTurn(
            input_tokens=150,
            output_tokens=100,
            input_text_tokens=70,
            input_audio_tokens=80,
            input_cached_tokens=20,
            output_text_tokens=40,
            output_audio_tokens=60,
        ),
    ),
    expected=ExactExpected(
        spend=0.0021272,
        input_cost=0.0008312,
        output_cost=0.001296,
        prompt_tokens=150,
        completion_tokens=100,
        cost_header=False,
    ),
)

GPT_REALTIME_MINI_2025_12_15_REALTIME_TWO_TURNS_SUMMED_INTO_ONE_ROW: Final = RealtimeCostCase(
    name="gpt-realtime-mini-2025-12-15-realtime-two_turns_summed_into_one_row",
    covers="quota_management.spend_tracking.realtime_costs.multiple_turns",
    model="gpt-realtime-mini-2025-12-15",
    litellm_model="openai/gpt-realtime-mini-2025-12-15",
    turns=(
        RealtimeTurn(
            input_tokens=150,
            output_tokens=100,
            input_text_tokens=70,
            input_audio_tokens=80,
            input_cached_tokens=20,
            output_text_tokens=40,
            output_audio_tokens=60,
        ),
        RealtimeTurn(
            input_tokens=100,
            output_tokens=50,
            input_text_tokens=100,
            input_audio_tokens=0,
            input_cached_tokens=0,
            output_text_tokens=50,
            output_audio_tokens=0,
        ),
    ),
    expected=ExactExpected(
        spend=0.0023072,
        input_cost=0.0008912,
        output_cost=0.001416,
        prompt_tokens=250,
        completion_tokens=150,
        cost_header=False,
    ),
)

GPT_REALTIME_MINI_2025_12_15_REALTIME_PRICED_FROM_SESSION_CREATED_MODEL: Final = RealtimeCostCase(
    name="gpt-realtime-mini-2025-12-15-realtime-priced_from_session_created_model",
    covers="quota_management.spend_tracking.realtime_costs.session_model",
    model="gpt-realtime-mini-2025-12-15",
    litellm_model="openai/gpt-realtime-mini-2025-12-15",
    turns=(
        RealtimeTurn(
            input_tokens=150,
            output_tokens=100,
            input_text_tokens=70,
            input_audio_tokens=80,
            input_cached_tokens=20,
            output_text_tokens=40,
            output_audio_tokens=60,
        ),
    ),
    session_model="gpt-realtime-2.1",
    expected=ExactExpected(
        spend=0.007568,
        input_cost=0.002768,
        output_cost=0.0048,
        prompt_tokens=150,
        completion_tokens=100,
        cost_header=False,
    ),
)

GPT_REALTIME_MINI_2025_12_15_REALTIME_SESSION_WITHOUT_TURNS_ZERO_SPEND: Final = RealtimeCostCase(
    name="gpt-realtime-mini-2025-12-15-realtime-session_without_turns_zero_spend",
    covers="quota_management.spend_tracking.realtime_costs.session_without_turns",
    model="gpt-realtime-mini-2025-12-15",
    litellm_model="openai/gpt-realtime-mini-2025-12-15",
    turns=(),
    expected=ExactExpected(
        spend=0.0,
        input_cost=0.0,
        output_cost=0.0,
        prompt_tokens=0,
        completion_tokens=0,
        breakdown_persisted=False,
        cost_header=False,
    ),
)


BATCH_CASES: Final[tuple[BatchCostCase, ...]] = (
    GPT_5_6_BATCH_HALVED_RATES_WHEN_MAP_HAS_NO_BATCH_KEYS,
    GPT_5_6_BATCH_CACHED_INPUT_HALVED,
    GPT_5_4_BATCH_EXPLICIT_BATCH_RATES_BILL_CACHED_AT_BATCH_INPUT_RATE,
    GPT_5_6_BATCH_ALL_REQUESTS_FAILED_ZERO_SPEND,
)

REALTIME_CASES: Final[tuple[RealtimeCostCase, ...]] = (
    GPT_REALTIME_MINI_2025_12_15_REALTIME_SINGLE_TURN_TEXT_AUDIO_CACHED,
    GPT_REALTIME_MINI_2025_12_15_REALTIME_TWO_TURNS_SUMMED_INTO_ONE_ROW,
    GPT_REALTIME_MINI_2025_12_15_REALTIME_PRICED_FROM_SESSION_CREATED_MODEL,
    GPT_REALTIME_MINI_2025_12_15_REALTIME_SESSION_WITHOUT_TURNS_ZERO_SPEND,
)
