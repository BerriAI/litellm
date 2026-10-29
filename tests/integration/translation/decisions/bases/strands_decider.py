from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from https://github.com/strands-agents/decider (POST /v1/systemone, no auth). Mock reply captured live on 2026-10-06.
"""
STRANDS_DECIDER_2B_HOBSON_V19_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "strands_decider/strands-decider-2B-hobson-v19",
        "input": "Help! My payouts have been failing for 3 days!",
        "questions": [{"type": "predicate", "name": "is_urgent", "instructions": "Does this convey urgency?"}],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/systemone",
    expected_provider_headers={"content-type": "application/json"},
    expected_provider_request={
        "model": "strands-decider-2B-hobson-v19",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    },
    mock_provider_response={
        "model": "strands-decider-2B-hobson-v19",
        "answers": {"is_urgent": {"type": "noul", "noul": 0.8277}},
        "usage": {"input_tokens": 86, "output_tokens": 1},
        "latency_ms": 140.03,
    },
    expected_litellm_response={
        "model": "strands-decider-2B-hobson-v19",
        "answers": [{"type": "predicate", "name": "is_urgent", "probability": 0.8277}],
        "usage": {
            "input_tokens": 86,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 87,
        },
    },
)
STRANDS_DECIDER_2B_HOBSON_V19_SYSTEMONE_TEST_CASE: Final = TranslationTestCase(
    scenario="systemone",
    litellm_endpoint="/v1/systemone",
    litellm_request={
        "model": "strands_decider/strands-decider-2B-hobson-v19",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/systemone",
    expected_provider_headers={"content-type": "application/json"},
    expected_provider_request={
        "model": "strands-decider-2B-hobson-v19",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    },
    mock_provider_response={
        "model": "strands-decider-2B-hobson-v19",
        "answers": {"is_urgent": {"type": "noul", "noul": 0.8277}},
        "usage": {"input_tokens": 86, "output_tokens": 1},
        "latency_ms": 140.03,
    },
    expected_litellm_response={
        "model": "strands-decider-2B-hobson-v19",
        "answers": {"is_urgent": {"type": "noul", "noul": 0.8277}},
        "usage": {"input_tokens": 86, "output_tokens": 1},
        "latency_ms": 140.03,
    },
)
