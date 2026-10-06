from typing import Final

from integration.translation.case import TranslationTestCase

"""OpenAI Decisions: LiteLLM forwards the request as is.

Spec: https://developers.openai.com/api/docs/guides/decisions. Mock reply captured live on 2026-10-06.
"""
GPT_6_LUNA_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "openai/gpt-6-luna",
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
    expected_provider_headers={"authorization": "Bearer synthetic-openai-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "gpt-6-luna",
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
    },
    mock_provider_response={
        "model": "gpt-6-luna",
        "answers": [
            {"type": "predicate", "name": "defect", "probability": 0.7},
            {
                "type": "choice",
                "name": "severity",
                "choice": "high",
                "probabilities": [{"value": "low", "probability": 0.08}, {"value": "high", "probability": 0.92}],
                "confidence": 0.84,
            },
            {
                "type": "score",
                "name": "confidence",
                "score": 0.07,
                "probabilities": [
                    {"value": 0, "label": "unsure", "probability": 0.93},
                    {"value": 1, "label": "sure", "probability": 0.07},
                ],
                "confidence": 0.86,
            },
        ],
        "usage": {
            "input_tokens": 373,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 373,
        },
    },
    expected_litellm_response={
        "model": "gpt-6-luna",
        "answers": [
            {"type": "predicate", "name": "defect", "probability": 0.7},
            {
                "type": "choice",
                "name": "severity",
                "choice": "high",
                "probabilities": [{"value": "low", "probability": 0.08}, {"value": "high", "probability": 0.92}],
                "confidence": 0.84,
            },
            {
                "type": "score",
                "name": "confidence",
                "score": 0.07,
                "probabilities": [
                    {"value": 0, "label": "unsure", "probability": 0.93},
                    {"value": 1, "label": "sure", "probability": 0.07},
                ],
                "confidence": 0.86,
            },
        ],
        "usage": {
            "input_tokens": 373,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 373,
        },
    },
)
