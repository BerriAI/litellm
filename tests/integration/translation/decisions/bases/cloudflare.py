from typing import Final

from integration.translation.case import TranslationTestCase

"""Cloudflare Clef: LiteLLM's /v1/decisions and the provider both speak Jev / System One, so the body passes through.

Provider side from https://developers.cloudflare.com/workers-ai (POST /ai/run/@cf/cloudflare/clef, reply wrapped in result). Mock reply captured live on 2026-10-06.
"""
CLEF_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "cloudflare/@cf/cloudflare/clef",
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
    expected_provider_endpoint="/ai/run/@cf/cloudflare/clef",
    expected_provider_headers={"authorization": "Bearer synthetic-cloudflare-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "clef",
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
        "result": {
            "model": "clef",
            "answers": {
                "defect": {"type": "noul", "noul": 0.9345},
                "severity": {
                    "type": "choice",
                    "choice": "high",
                    "confidence": 0.8067,
                    "probabilities": {"low": 0.0509, "high": 0.9491},
                },
                "confidence": {
                    "type": "score",
                    "score": 0.9036,
                    "confidence": 0.6515,
                    "legend": {"0": "unsure", "1": "sure"},
                    "probabilities": {"0": 0.0964, "1": 0.9036},
                },
            },
            "usage": {"input_tokens": 290, "output_tokens": 0},
        },
        "success": True,
        "errors": [],
        "messages": [],
    },
    expected_litellm_response={
        "model": "clef",
        "answers": {
            "defect": {"type": "noul", "noul": 0.9345},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.8067,
                "probabilities": {"low": 0.0509, "high": 0.9491},
            },
            "confidence": {
                "type": "score",
                "score": 0.9036,
                "confidence": 0.6515,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.0964, "1": 0.9036},
            },
        },
        "usage": {"input_tokens": 290, "output_tokens": 0},
    },
)
