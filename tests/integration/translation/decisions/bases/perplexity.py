from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from https://docs.perplexity.ai (POST /v1/decisions). Mock reply captured live on 2026-10-06.
"""
PPLX_DECIDER_V1_27B_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "perplexity/pplx-decider-v1-27b",
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
    expected_provider_endpoint="/v1/decisions",
    expected_provider_headers={"authorization": "Bearer synthetic-perplexity-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "pplx-decider-v1-27b",
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
        "model": "pplx-decider-v1-27b",
        "answers": {
            "defect": {"type": "noul", "noul": 0.9989100737587077},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.9964631215356778,
                "probabilities": {"low": 0.0017684392321610232, "high": 0.9982315607678389},
            },
            "confidence": {
                "type": "score",
                "score": 0.07367392327139817,
                "confidence": 0.8526521534572037,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.9263260767286018, "1": 0.07367392327139817},
            },
        },
        "usage": {"input_tokens": 318, "output_tokens": 3},
    },
    expected_litellm_response={
        "model": "pplx-decider-v1-27b",
        "answers": {
            "defect": {"type": "noul", "noul": 0.9989100737587077},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.9964631215356778,
                "probabilities": {"low": 0.0017684392321610232, "high": 0.9982315607678389},
            },
            "confidence": {
                "type": "score",
                "score": 0.07367392327139817,
                "confidence": 0.8526521534572037,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.9263260767286018, "1": 0.07367392327139817},
            },
        },
        "usage": {"input_tokens": 318, "output_tokens": 3},
    },
)
