from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from Fireworks System One (POST /v1/systemone, private preview). Mock reply captured live on 2026-10-12.
"""
RUNE_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "fireworks_ai/accounts/fireworks/routers/rune",
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
    expected_provider_headers={"authorization": "Bearer synthetic-fireworks-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "accounts/fireworks/routers/rune",
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
        "model": "accounts/fireworks/routers/rune",
        "answers": {
            "defect": {"type": "noul", "noul": 0.9902915235185259},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.6709670742068738,
                "probabilities": {"high": 0.8354835371034369, "low": 0.1645164628965632},
            },
            "confidence": {
                "type": "score",
                "score": 0.7549149868676283,
                "confidence": 0.5098299737352565,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.24508501313237172, "1": 0.7549149868676283},
            },
        },
        "usage": {"input_tokens": 384, "cached_input_tokens": 0, "output_tokens": 0},
    },
    expected_litellm_response={
        "model": "accounts/fireworks/routers/rune",
        "answers": [
            {"type": "predicate", "name": "defect", "probability": 0.9902915235185259},
            {
                "type": "choice",
                "name": "severity",
                "choice": "high",
                "probabilities": [
                    {"value": "low", "probability": 0.1645164628965632},
                    {"value": "high", "probability": 0.8354835371034369},
                ],
                "confidence": 0.6709670742068738,
            },
            {
                "type": "score",
                "name": "confidence",
                "score": 0.7549149868676283,
                "probabilities": [
                    {"value": 0, "label": "unsure", "probability": 0.24508501313237172},
                    {"value": 1, "label": "sure", "probability": 0.7549149868676283},
                ],
                "confidence": 0.5098299737352565,
            },
        ],
        "usage": {
            "input_tokens": 384,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 384,
        },
    },
)
RUNE_SYSTEMONE_TEST_CASE: Final = TranslationTestCase(
    scenario="systemone",
    litellm_endpoint="/v1/systemone",
    litellm_request={
        "model": "fireworks_ai/accounts/fireworks/routers/rune",
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
    expected_provider_headers={"authorization": "Bearer synthetic-fireworks-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "accounts/fireworks/routers/rune",
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
        "model": "accounts/fireworks/routers/rune",
        "answers": {
            "defect": {"type": "noul", "noul": 0.9902915235185259},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.6709670742068738,
                "probabilities": {"high": 0.8354835371034369, "low": 0.1645164628965632},
            },
            "confidence": {
                "type": "score",
                "score": 0.7549149868676283,
                "confidence": 0.5098299737352565,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.24508501313237172, "1": 0.7549149868676283},
            },
        },
        "usage": {"input_tokens": 384, "cached_input_tokens": 0, "output_tokens": 0},
    },
    expected_litellm_response={
        "model": "accounts/fireworks/routers/rune",
        "answers": {
            "defect": {"type": "noul", "noul": 0.9902915235185259},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.6709670742068738,
                "probabilities": {"high": 0.8354835371034369, "low": 0.1645164628965632},
            },
            "confidence": {
                "type": "score",
                "score": 0.7549149868676283,
                "confidence": 0.5098299737352565,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.24508501313237172, "1": 0.7549149868676283},
            },
        },
        "usage": {"input_tokens": 384, "cached_input_tokens": 0, "output_tokens": 0},
    },
)
