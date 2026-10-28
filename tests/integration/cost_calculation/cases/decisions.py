"""Decisions cost cases: every Decisions provider called through /v1/decisions and /v1/systemone.

Request and response bodies are verbatim captures of what litellm sent to and got back from the
real provider on 2026-10-10 (recorded through a forwarding proxy in front of each API), so each
case replays the provider's own usage shape against the scripted upstream. Spend is the recorded
usage at the suite cost map's rates; OpenRouter has no per-token rate and bills the `usage.cost`
it reports.
"""

from __future__ import annotations

from typing import Final

from integration.cost_calculation.cost_tracking_case import CostTrackingTestCase, ExactExpected, JsonResponse
from integration.cost_calculation.stream_parity.case import StreamParityTestCase


# gpt-6-luna
OPENAI_GPT_6_LUNA_DECISIONS: Final = CostTrackingTestCase(
    name="gpt-6-luna-decisions",
    covers="quota_management.spend_tracking.decisions.logs_cost",
    model="gpt-6-luna",
    endpoint="/v1/decisions",
    request={
        "model": "$MODEL",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    upstream_body={
        "model": "gpt-6-luna",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": "is_refund", "probability": 1.0}],
            "usage": {
                "input_tokens": 167,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 167,
            },
        },
    ),
    expected=ExactExpected(
        spend=1.67e-05,
        input_cost=1.67e-05,
        output_cost=0.0,
        prompt_tokens=167,
        completion_tokens=0,
    ),
)
OPENAI_GPT_6_LUNA_SYSTEMONE: Final = CostTrackingTestCase(
    name="gpt-6-luna-systemone",
    covers="quota_management.spend_tracking.systemone.logs_cost",
    model="gpt-6-luna",
    endpoint="/v1/systemone",
    request={
        "model": "$MODEL",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    upstream_body={
        "model": "gpt-6-luna",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": "is_refund", "probability": 1.0}],
            "usage": {
                "input_tokens": 167,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 0,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 167,
            },
        },
    ),
    expected=ExactExpected(
        spend=1.67e-05,
        input_cost=1.67e-05,
        output_cost=0.0,
        prompt_tokens=167,
        completion_tokens=0,
    ),
)

# typesafe/jev-latest
TYPESAFE_JEV_LATEST_DECISIONS: Final = CostTrackingTestCase(
    name="typesafe/jev-latest-decisions",
    covers="quota_management.spend_tracking.decisions.logs_cost",
    model="typesafe/jev-latest",
    endpoint="/v1/decisions",
    request={
        "model": "$MODEL",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    upstream_body={
        "model": "jev-latest",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "jev-1.13.0",
            "answers": {"is_refund": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 285, "output_tokens": 22},
        },
    ),
    expected=ExactExpected(
        spend=1.197e-05,
        input_cost=1.197e-05,
        output_cost=0.0,
        prompt_tokens=285,
        completion_tokens=22,
    ),
)
TYPESAFE_JEV_LATEST_SYSTEMONE: Final = CostTrackingTestCase(
    name="typesafe/jev-latest-systemone",
    covers="quota_management.spend_tracking.systemone.logs_cost",
    model="typesafe/jev-latest",
    endpoint="/v1/systemone",
    request={
        "model": "$MODEL",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    upstream_body={
        "model": "jev-latest",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "jev-1.13.0",
            "answers": {"is_refund": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 285, "output_tokens": 22},
        },
    ),
    expected=ExactExpected(
        spend=1.197e-05,
        input_cost=1.197e-05,
        output_cost=0.0,
        prompt_tokens=285,
        completion_tokens=22,
    ),
)

# perplexity/pplx-decider-v1-27b
PERPLEXITY_PPLX_DECIDER_V1_27B_DECISIONS: Final = CostTrackingTestCase(
    name="perplexity/pplx-decider-v1-27b-decisions",
    covers="quota_management.spend_tracking.decisions.logs_cost",
    model="perplexity/pplx-decider-v1-27b",
    endpoint="/v1/decisions",
    request={
        "model": "$MODEL",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    upstream_body={
        "model": "pplx-decider-v1-27b",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "pplx-decider-v1-27b",
            "answers": {"is_refund": {"type": "noul", "noul": 0.9997319426472651}},
            "usage": {"input_tokens": 98, "output_tokens": 1},
        },
    ),
    expected=ExactExpected(
        spend=1.96e-06,
        input_cost=1.96e-06,
        output_cost=0.0,
        prompt_tokens=98,
        completion_tokens=1,
    ),
)
PERPLEXITY_PPLX_DECIDER_V1_27B_SYSTEMONE: Final = CostTrackingTestCase(
    name="perplexity/pplx-decider-v1-27b-systemone",
    covers="quota_management.spend_tracking.systemone.logs_cost",
    model="perplexity/pplx-decider-v1-27b",
    endpoint="/v1/systemone",
    request={
        "model": "$MODEL",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    upstream_body={
        "model": "pplx-decider-v1-27b",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "pplx-decider-v1-27b",
            "answers": {"is_refund": {"type": "noul", "noul": 0.9997319426472651}},
            "usage": {"input_tokens": 98, "output_tokens": 1},
        },
    ),
    expected=ExactExpected(
        spend=1.96e-06,
        input_cost=1.96e-06,
        output_cost=0.0,
        prompt_tokens=98,
        completion_tokens=1,
    ),
)

# cloudflare/@cf/cloudflare/clef-flash
CLOUDFLARE_CLEF_FLASH_DECISIONS: Final = CostTrackingTestCase(
    name="cloudflare/@cf/cloudflare/clef-flash-decisions",
    covers="quota_management.spend_tracking.decisions.logs_cost",
    model="cloudflare/@cf/cloudflare/clef-flash",
    endpoint="/v1/decisions",
    request={
        "model": "$MODEL",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    upstream_body={
        "model": "clef-flash",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "result": {
                "model": "clef-flash",
                "answers": {"is_refund": {"type": "noul", "noul": 0.9693}},
                "usage": {"input_tokens": 157, "output_tokens": 0},
            },
            "success": True,
            "errors": [],
            "messages": [],
        },
    ),
    expected=ExactExpected(
        spend=1.413e-05,
        input_cost=1.413e-05,
        output_cost=0.0,
        prompt_tokens=157,
        completion_tokens=0,
    ),
)
CLOUDFLARE_CLEF_FLASH_SYSTEMONE: Final = CostTrackingTestCase(
    name="cloudflare/@cf/cloudflare/clef-flash-systemone",
    covers="quota_management.spend_tracking.systemone.logs_cost",
    model="cloudflare/@cf/cloudflare/clef-flash",
    endpoint="/v1/systemone",
    request={
        "model": "$MODEL",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    upstream_body={
        "model": "clef-flash",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "result": {
                "model": "clef-flash",
                "answers": {"is_refund": {"type": "noul", "noul": 0.9693}},
                "usage": {"input_tokens": 157, "output_tokens": 0},
            },
            "success": True,
            "errors": [],
            "messages": [],
        },
    ),
    expected=ExactExpected(
        spend=1.413e-05,
        input_cost=1.413e-05,
        output_cost=0.0,
        prompt_tokens=157,
        completion_tokens=0,
    ),
)

# databricks/databricks-openjev-qwen35-4b
DATABRICKS_OPENJEV_QWEN35_4B_DECISIONS: Final = CostTrackingTestCase(
    name="databricks/databricks-openjev-qwen35-4b-decisions",
    covers="quota_management.spend_tracking.decisions.logs_cost",
    model="databricks/databricks-openjev-qwen35-4b",
    endpoint="/v1/decisions",
    request={
        "model": "$MODEL",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    upstream_body={
        "model": "databricks-openjev-qwen35-4b",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "databricks-openjev-qwen35-4b",
            "answers": {"is_refund": {"type": "noul", "noul": 0.9046505248181417}},
            "usage": {"input_tokens": 45, "output_tokens": 0},
        },
    ),
    expected=ExactExpected(
        spend=1.89e-06,
        input_cost=1.89e-06,
        output_cost=0.0,
        prompt_tokens=45,
        completion_tokens=0,
    ),
)
DATABRICKS_OPENJEV_QWEN35_4B_SYSTEMONE: Final = CostTrackingTestCase(
    name="databricks/databricks-openjev-qwen35-4b-systemone",
    covers="quota_management.spend_tracking.systemone.logs_cost",
    model="databricks/databricks-openjev-qwen35-4b",
    endpoint="/v1/systemone",
    request={
        "model": "$MODEL",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    upstream_body={
        "model": "databricks-openjev-qwen35-4b",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "databricks-openjev-qwen35-4b",
            "answers": {"is_refund": {"type": "noul", "noul": 0.9046505248181417}},
            "usage": {"input_tokens": 45, "output_tokens": 0},
        },
    ),
    expected=ExactExpected(
        spend=1.89e-06,
        input_cost=1.89e-06,
        output_cost=0.0,
        prompt_tokens=45,
        completion_tokens=0,
    ),
)

# openrouter/typesafe/jev-1.13
OPENROUTER_TYPESAFE_JEV_1_13_DECISIONS: Final = CostTrackingTestCase(
    name="openrouter/typesafe/jev-1.13-decisions",
    covers="quota_management.spend_tracking.decisions.logs_cost",
    model="openrouter/typesafe/jev-1.13",
    endpoint="/v1/decisions",
    request={
        "model": "$MODEL",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    upstream_body={
        "model": "typesafe/jev-1.13",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "typesafe/jev-1.13-20260917",
            "answers": {"is_refund": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 285, "output_tokens": 22, "cost": 1.197e-05},
            "id": "gen-dec-$REQUEST_ID",
            "provider": "TypeSafe",
        },
    ),
    expected=ExactExpected(
        spend=1.197e-05,
        input_cost=1.197e-05,
        output_cost=0.0,
        prompt_tokens=285,
        completion_tokens=22,
        breakdown_persisted=False,
    ),
)
OPENROUTER_TYPESAFE_JEV_1_13_SYSTEMONE: Final = CostTrackingTestCase(
    name="openrouter/typesafe/jev-1.13-systemone",
    covers="quota_management.spend_tracking.systemone.logs_cost",
    model="openrouter/typesafe/jev-1.13",
    endpoint="/v1/systemone",
    request={
        "model": "$MODEL",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    upstream_body={
        "model": "typesafe/jev-1.13",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "typesafe/jev-1.13-20260917",
            "answers": {"is_refund": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 285, "output_tokens": 22, "cost": 1.197e-05},
            "id": "gen-dec-$REQUEST_ID",
            "provider": "TypeSafe",
        },
    ),
    expected=ExactExpected(
        spend=1.197e-05,
        input_cost=1.197e-05,
        output_cost=0.0,
        prompt_tokens=285,
        completion_tokens=22,
        breakdown_persisted=False,
    ),
)

# fireworks_ai/accounts/fireworks/routers/rune
FIREWORKS_AI_RUNE_DECISIONS: Final = CostTrackingTestCase(
    name="fireworks_ai/accounts/fireworks/routers/rune-decisions",
    covers="quota_management.spend_tracking.decisions.logs_cost",
    model="fireworks_ai/accounts/fireworks/routers/rune",
    endpoint="/v1/decisions",
    request={
        "model": "$MODEL",
        "input": "My order arrived broken and I want a refund today.",
        "questions": [
            {"type": "predicate", "name": "is_refund", "instructions": "Is the customer asking for a refund?"}
        ],
    },
    upstream_body={
        "model": "accounts/fireworks/routers/rune",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "accounts/fireworks/routers/rune",
            "answers": {"is_refund": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 300, "cached_input_tokens": 200, "output_tokens": 0},
        },
    ),
    expected=ExactExpected(
        spend=5.88e-06,
        input_cost=5.88e-06,
        output_cost=0.0,
        prompt_tokens=300,
        completion_tokens=0,
    ),
)
FIREWORKS_AI_RUNE_SYSTEMONE: Final = CostTrackingTestCase(
    name="fireworks_ai/accounts/fireworks/routers/rune-systemone",
    covers="quota_management.spend_tracking.systemone.logs_cost",
    model="fireworks_ai/accounts/fireworks/routers/rune",
    endpoint="/v1/systemone",
    request={
        "model": "$MODEL",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    upstream_body={
        "model": "accounts/fireworks/routers/rune",
        "state": "My order arrived broken and I want a refund today.",
        "questions": {"is_refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}},
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "model": "accounts/fireworks/routers/rune",
            "answers": {"is_refund": {"type": "noul", "noul": 0.99}},
            "usage": {"input_tokens": 300, "cached_input_tokens": 200, "output_tokens": 0},
        },
    ),
    expected=ExactExpected(
        spend=5.88e-06,
        input_cost=5.88e-06,
        output_cost=0.0,
        prompt_tokens=300,
        completion_tokens=0,
    ),
)

CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    OPENAI_GPT_6_LUNA_DECISIONS,
    OPENAI_GPT_6_LUNA_SYSTEMONE,
    TYPESAFE_JEV_LATEST_DECISIONS,
    TYPESAFE_JEV_LATEST_SYSTEMONE,
    PERPLEXITY_PPLX_DECIDER_V1_27B_DECISIONS,
    PERPLEXITY_PPLX_DECIDER_V1_27B_SYSTEMONE,
    CLOUDFLARE_CLEF_FLASH_DECISIONS,
    CLOUDFLARE_CLEF_FLASH_SYSTEMONE,
    DATABRICKS_OPENJEV_QWEN35_4B_DECISIONS,
    DATABRICKS_OPENJEV_QWEN35_4B_SYSTEMONE,
    OPENROUTER_TYPESAFE_JEV_1_13_DECISIONS,
    OPENROUTER_TYPESAFE_JEV_1_13_SYSTEMONE,
    FIREWORKS_AI_RUNE_DECISIONS,
    FIREWORKS_AI_RUNE_SYSTEMONE,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = ()
