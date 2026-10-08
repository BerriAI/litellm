from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from https://docs.perplexity.ai (POST /v1/decisions). Mock reply captured live on 2026-10-06.
"""
PPLX_DECIDER_V1_27B_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "perplexity/pplx-decider-v1-27b",
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
        "answers": [
            {"type": "predicate", "name": "defect", "probability": 0.9989100737587077},
            {
                "type": "choice",
                "name": "severity",
                "choice": "high",
                "probabilities": [
                    {"value": "low", "probability": 0.0017684392321610232},
                    {"value": "high", "probability": 0.9982315607678389},
                ],
                "confidence": 0.9964631215356778,
            },
            {
                "type": "score",
                "name": "confidence",
                "score": 0.07367392327139817,
                "probabilities": [
                    {"value": 0, "label": "unsure", "probability": 0.9263260767286018},
                    {"value": 1, "label": "sure", "probability": 0.07367392327139817},
                ],
                "confidence": 0.8526521534572037,
            },
        ],
        "usage": {
            "input_tokens": 318,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 321,
        },
    },
)
PPLX_DECIDER_V1_27B_SYSTEMONE_TEST_CASE: Final = TranslationTestCase(
    scenario="systemone",
    litellm_endpoint="/v1/systemone",
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
