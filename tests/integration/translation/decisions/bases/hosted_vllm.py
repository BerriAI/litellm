from typing import Final

from integration.translation.case import TranslationTestCase

"""Provider request and reply shape from a real vLLM serve (POST /v1/systemone, no auth). Mock reply captured live on 2026-10-09 from a vLLM CPU build of vllm main 702ce313f serving Qwen/Qwen3-0.6B.
"""
QWEN3_0_6B_TEST_CASE: Final = TranslationTestCase(
    scenario="basic",
    litellm_endpoint="/v1/systemone",
    litellm_request={
        "model": "hosted_vllm/Qwen/Qwen3-0.6B",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {
            "is_urgent": {
                "type": "choice",
                "instructions": "Does this convey urgency?",
                "criteria": {"yes": "the message conveys urgency", "no": "the message does not convey urgency"},
            }
        },
        "cache": {"no-cache": True},
    },
    expected_provider_endpoint="/v1/systemone",
    expected_provider_headers={"content-type": "application/json"},
    expected_provider_request={
        "model": "Qwen/Qwen3-0.6B",
        "state": "Help! My payouts have been failing for 3 days!",
        "questions": {
            "is_urgent": {
                "type": "choice",
                "instructions": "Does this convey urgency?",
                "criteria": {"yes": "the message conveys urgency", "no": "the message does not convey urgency"},
            }
        },
    },
    mock_provider_response={
        "id": "decision-9c26c028d7f4457e",
        "object": "structured_decision",
        "created": 1791517765,
        "model": "Qwen/Qwen3-0.6B",
        "answers": {
            "is_urgent": {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 0.9669140238756562, "no": 0.03308597612434374},
                "confidence": 0.9649145337527952,
            }
        },
        "usage": {"input_tokens": 61, "output_tokens": 1},
        "diagnostics": {"is_urgent": {"label_mass": 0.9979320910923947, "argmax_is_label": True}},
    },
    expected_litellm_response={
        "id": "decision-9c26c028d7f4457e",
        "object": "structured_decision",
        "created": 1791517765,
        "model": "Qwen/Qwen3-0.6B",
        "answers": {
            "is_urgent": {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 0.9669140238756562, "no": 0.03308597612434374},
                "confidence": 0.9649145337527952,
            }
        },
        "usage": {"input_tokens": 61, "output_tokens": 1},
        "diagnostics": {"is_urgent": {"label_mass": 0.9979320910923947, "argmax_is_label": True}},
    },
)
