from dataclasses import replace
from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from https://developers.cloudflare.com/workers-ai (POST /ai/run/@cf/cloudflare/clef, reply wrapped in result). Mock reply captured live on 2026-10-06.
"""
CLEF_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "cloudflare/@cf/cloudflare/clef",
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
        "answers": [
            {"type": "predicate", "name": "defect", "probability": 0.9345},
            {
                "type": "choice",
                "name": "severity",
                "choice": "high",
                "probabilities": [{"value": "low", "probability": 0.0509}, {"value": "high", "probability": 0.9491}],
                "confidence": 0.8067,
            },
            {
                "type": "score",
                "name": "confidence",
                "score": 0.9036,
                "probabilities": [
                    {"value": 0, "label": "unsure", "probability": 0.0964},
                    {"value": 1, "label": "sure", "probability": 0.9036},
                ],
                "confidence": 0.6515,
            },
        ],
        "usage": {
            "input_tokens": 290,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 290,
        },
    },
)
CLEF_SYSTEMONE_TEST_CASE: Final = TranslationTestCase(
    scenario="systemone",
    litellm_endpoint="/v1/systemone",
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
CLEF_IMAGES_TEST_CASE: Final = replace(
    CLEF_TEST_CASE,
    scenario="images",
    litellm_request={
        **CLEF_TEST_CASE.litellm_request,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Ticket (billing): The export job hangs at 99% and never finishes",
                    },
                    {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
                ],
            }
        ],
    },
    expected_provider_request={
        **CLEF_TEST_CASE.expected_provider_request,
        "images": ["data:image/png;base64,iVBORw0KGgo="],
    },
)
CLEF_IMAGES_SYSTEMONE_TEST_CASE: Final = replace(
    CLEF_SYSTEMONE_TEST_CASE,
    scenario="images_systemone",
    litellm_request={
        **CLEF_SYSTEMONE_TEST_CASE.litellm_request,
        "images": [
            "data:image/png;base64,iVBORw0KGgo=",
            {"content_type": "image/jpeg", "base64": "BB=="},
        ],
    },
    expected_provider_request={
        **CLEF_SYSTEMONE_TEST_CASE.expected_provider_request,
        "images": ["data:image/png;base64,iVBORw0KGgo=", "data:image/jpeg;base64,BB=="],
    },
)
