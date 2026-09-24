"""
Tests for the Lakera AI v1 guardrail hook (pre-call moderation via /moderations).
"""

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.lakera_ai import lakeraAI_Moderation

THRESHOLDS = {"jailbreak": 0.9, "prompt_injection": 0.9}


def flagged_response(flagged: bool, category_scores: dict | None = None) -> dict:
    result = {"flagged": flagged}
    if category_scores is not None:
        result["category_scores"] = category_scores
    return {"results": [result]}


def test_flagged_true_blocks_when_category_thresholds_configured():
    guardrail = lakeraAI_Moderation(category_thresholds=THRESHOLDS)
    response = flagged_response(
        flagged=True,
        category_scores={"jailbreak": 0.1, "prompt_injection": 0.1, "hate": 0.99},
    )
    with pytest.raises(HTTPException) as exc_info:
        guardrail._check_response_flagged(response)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["error"] == "Violated content safety policy"


def test_threshold_violation_blocks_when_not_flagged():
    guardrail = lakeraAI_Moderation(category_thresholds=THRESHOLDS)
    response = flagged_response(
        flagged=False,
        category_scores={"jailbreak": 0.95, "prompt_injection": 0.1},
    )
    with pytest.raises(HTTPException) as exc_info:
        guardrail._check_response_flagged(response)
    assert exc_info.value.detail["error"] == "Violated jailbreak threshold"


def test_clean_response_passes_when_thresholds_configured():
    guardrail = lakeraAI_Moderation(category_thresholds=THRESHOLDS)
    response = flagged_response(
        flagged=False,
        category_scores={"jailbreak": 0.1, "prompt_injection": 0.1},
    )
    assert guardrail._check_response_flagged(response) is None


def test_flagged_true_blocks_without_thresholds():
    guardrail = lakeraAI_Moderation()
    with pytest.raises(HTTPException):
        guardrail._check_response_flagged(flagged_response(flagged=True))
