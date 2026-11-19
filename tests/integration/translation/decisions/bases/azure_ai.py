from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from the Microsoft Foundry Decision playground (POST /providers/microsoft/v1/systemone, model is the deployment name). Mock reply captured live from a Microsoft-Decision-1 deployment on 2026-10-09.
"""
MICROSOFT_DECISION_1_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/systemone",
    litellm_request={
        "model": "azure_ai/decision-1",
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
    expected_provider_endpoint="/providers/microsoft/v1/systemone",
    expected_provider_headers={"authorization": "Bearer synthetic-azure-ai-key", "content-type": "application/json"},
    expected_provider_request={
        "model": "decision-1",
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
        "model": "microsoft-decision-1",
        "answers": {
            "confidence": {
                "confidence": 0.6351489346076444,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.1824255326961778, "1": 0.8175744673038222},
                "score": 0.8175744673038222,
                "type": "score",
            },
            "defect": {"noul": 0.982013764002278, "type": "noul"},
            "severity": {
                "choice": "high",
                "confidence": 0.9866142978108987,
                "probabilities": {"high": 0.9933071489054494, "low": 0.006692851094550702},
                "type": "choice",
            },
        },
        "usage": {"input_tokens": 87, "output_tokens": 3},
    },
    expected_litellm_response={
        "model": "microsoft-decision-1",
        "answers": {
            "defect": {"type": "noul", "noul": 0.982013764002278},
            "severity": {
                "type": "choice",
                "choice": "high",
                "confidence": 0.9866142978108987,
                "probabilities": {"high": 0.9933071489054494, "low": 0.006692851094550702},
            },
            "confidence": {
                "type": "score",
                "score": 0.8175744673038222,
                "confidence": 0.6351489346076444,
                "legend": {"0": "unsure", "1": "sure"},
                "probabilities": {"0": 0.1824255326961778, "1": 0.8175744673038222},
            },
        },
        "usage": {"input_tokens": 87, "output_tokens": 3},
    },
)
