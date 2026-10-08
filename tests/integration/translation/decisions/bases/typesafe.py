from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from https://docs.typesafe.ai (POST /v1/systemone). Mock reply captured live on 2026-10-06.
"""
JEV_1_13_0_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "typesafe/jev-1.13.0",
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
)
