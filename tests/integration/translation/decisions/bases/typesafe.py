from typing import Final

from integration.translation.case import TranslationTestCase

"""TypeSafe: LiteLLM speaks the OpenAI Decisions shape; the provider speaks Jev / System One.

Provider side from https://docs.typesafe.ai (POST /v1/systemone). Mock reply captured live on 2026-10-06.
"""
JEV_1_13_0_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "typesafe/jev-1.13.0",
        "input": "Ticket (billing): The export job hangs at 99% and never finishes",
        "questions": [
            {"type": "predicate", "name": "defect", "instructions": "Is this a defect?"},
            {
                "type": "choice",
                "name": "severity",
                "instructions": "How severe is it?",
                "choices": [
                    {"value": "low", "description": "cosmetic"},
                    {"value": "high", "description": "blocks users"},
                ],
            },
            {
                "type": "score",
                "name": "confidence",
                "instructions": "How sure are you?",
                "levels": [{"label": "unsure"}, {"label": "sure"}],
            },
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/systemone",
    expected_provider_headers={"authorization": "Bearer synthetic-typesafe-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "jev-1.13.0",
        "state": "Ticket (billing): The export job hangs at 99% and never finishes",
        "questions": {
            "defect": {"type": "noul", "instructions": "Is this a defect?"},
            "severity": {
                "type": "choice",
                "instructions": "How severe is it?",
                "criteria": {"low": "cosmetic", "high": "blocks users"},
            },
            "confidence": {"type": "score", "instructions": "How sure are you?", "criteria": ["unsure", "sure"]},
        },
    },
    mock_provider_response={
        "model": "jev-1.13.0",
        "answers": {
            "defect": {"type": "noul", "noul": 0.78},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.99,
                "probabilities": {"low": 0.01, "high": 0.99},
            },
            "confidence": {
                "type": "score",
                "score": 0.51,
                "confidence": 0.03,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.49, "1": 0.51},
            },
        },
        "usage": {"input_tokens": 377, "output_tokens": 62},
    },
    expected_litellm_response={
        "model": "jev-1.13.0",
        "answers": [
            {"type": "predicate", "name": "defect", "probability": 0.78},
            {
                "type": "choice",
                "name": "severity",
                "choice": "high",
                "probabilities": [{"value": "low", "probability": 0.01}, {"value": "high", "probability": 0.99}],
                "confidence": 0.99,
            },
            {
                "type": "score",
                "name": "confidence",
                "score": 0.51,
                "probabilities": [
                    {"value": 0, "label": "unsure", "probability": 0.49},
                    {"value": 1, "label": "sure", "probability": 0.51},
                ],
                "confidence": 0.03,
            },
        ],
        "usage": {
            "input_tokens": 377,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 62,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 439,
        },
    },
)
