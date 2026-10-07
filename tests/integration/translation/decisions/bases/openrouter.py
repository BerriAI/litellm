from typing import Final

from integration.translation.case import TranslationTestCase

"""OpenRouter: LiteLLM speaks the OpenAI Decisions shape; the provider speaks Jev / System One.

Provider side from https://openrouter.ai/docs (POST /api/alpha/decisions). Mock reply captured live on 2026-10-06.
"""
TYPESAFE_JEV_1_13_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "openrouter/typesafe/jev-1.13",
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
    expected_provider_endpoint="/alpha/decisions",
    expected_provider_headers={"authorization": "Bearer synthetic-openrouter-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "typesafe/jev-1.13",
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
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "defect": {"type": "noul", "noul": 0.81},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.99,
                "probabilities": {"low": 0.01, "high": 0.99},
            },
            "confidence": {
                "type": "score",
                "score": 0.5,
                "confidence": 0,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.5, "1": 0.5},
            },
        },
        "usage": {"input_tokens": 377, "output_tokens": 62, "cost": 1.5834e-05},
        "id": "gen-dec-1791323839-GA15kY0nt34oiJ7srfki",
        "provider": "TypeSafe",
    },
    expected_litellm_response={
        "model": "typesafe/jev-1.13-20260917",
        "answers": [
            {"type": "predicate", "name": "defect", "probability": 0.81},
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
                "score": 0.5,
                "probabilities": [
                    {"value": 0, "label": "unsure", "probability": 0.5},
                    {"value": 1, "label": "sure", "probability": 0.5},
                ],
                "confidence": 0,
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
