from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from https://openrouter.ai/docs (POST /api/alpha/decisions). Mock reply captured live on 2026-10-06.
"""
TYPESAFE_JEV_1_13_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/systemone",
    litellm_request={
        "model": "openrouter/typesafe/jev-1.13",
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
)
