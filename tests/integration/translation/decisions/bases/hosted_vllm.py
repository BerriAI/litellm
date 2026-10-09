import json
from dataclasses import replace
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

"""OpenAI Decisions client shape in, vLLM /v1/systemone body out. Mock reply captured live on 2026-10-09 from the same
vLLM build.
"""
QWEN3_0_6B_DECISIONS_TEST_CASE: Final = replace(
    QWEN3_0_6B_TEST_CASE,
    scenario="decisions",
    litellm_endpoint="/v1/decisions",
    litellm_request={
        "model": "hosted_vllm/Qwen/Qwen3-0.6B",
        "input": "The checkout page crashed twice today",
        "questions": [
            {
                "type": "choice",
                "name": "is_defect",
                "instructions": "Is this a defect?",
                "choices": [{"value": "yes"}, {"value": "no"}],
            }
        ],
        "cache": {"no-cache": True},
    },
    expected_provider_request={
        "model": "Qwen/Qwen3-0.6B",
        "state": "The checkout page crashed twice today",
        "questions": {
            "is_defect": {"type": "choice", "instructions": "Is this a defect?", "criteria": {"yes": None, "no": None}}
        },
    },
    mock_provider_response={
        "id": "decision-91ce41627dbf016f",
        "object": "structured_decision",
        "created": 1791577835,
        "model": "Qwen/Qwen3-0.6B",
        "answers": {
            "is_defect": {
                "type": "choice",
                "choice": "yes",
                "probabilities": {"yes": 0.8670357477770336, "no": 0.13296425222296632},
                "confidence": 0.8669148648509282,
            }
        },
        "usage": {"input_tokens": 43, "output_tokens": 1},
        "diagnostics": {"is_defect": {"label_mass": 0.9998605790748358, "argmax_is_label": True}},
    },
    expected_litellm_response={
        "model": "Qwen/Qwen3-0.6B",
        "answers": [
            {
                "type": "choice",
                "name": "is_defect",
                "choice": "yes",
                "probabilities": [
                    {"value": "yes", "probability": 0.8670357477770336},
                    {"value": "no", "probability": 0.13296425222296632},
                ],
                "confidence": 0.8669148648509282,
            }
        ],
        "usage": {
            "input_tokens": 43,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 44,
        },
    },
)

"""vLLM only scores `choice` questions, so an OpenAI `predicate` goes out as `noul` and comes back as vLLM's 400,
which LiteLLM passes through as a 400.
"""
VLLM_NOUL_REJECTED: Final = {
    "error": {
        "message": "unknown question type 'noul'; supported: ['choice']",
        "type": "BadRequestError",
        "param": None,
        "code": 400,
    }
}
QWEN3_0_6B_PREDICATE_REJECTED_TEST_CASE: Final = replace(
    QWEN3_0_6B_DECISIONS_TEST_CASE,
    scenario="predicate_rejected",
    litellm_request={
        "model": "hosted_vllm/Qwen/Qwen3-0.6B",
        "input": "The checkout page crashed twice today",
        "questions": [{"type": "predicate", "name": "is_bug", "instructions": "Is this a bug?"}],
        "cache": {"no-cache": True},
    },
    expected_provider_request={
        "model": "Qwen/Qwen3-0.6B",
        "state": "The checkout page crashed twice today",
        "questions": {"is_bug": {"type": "noul", "instructions": "Is this a bug?"}},
    },
    mock_provider_status_code=400,
    mock_provider_response=VLLM_NOUL_REJECTED,
    expected_litellm_status_code=400,
    expected_litellm_response={
        "error": {
            "message": f"litellm.BadRequestError: Hosted_vllmException - {json.dumps(VLLM_NOUL_REJECTED)}\n\n"
            "LiteLLM: model group 'hosted_vllm/Qwen/Qwen3-0.6B' failed with the error above. No fallback was attempted.",
            "type": "invalid_request_error",
            "param": None,
            "code": "400",
        }
    },
)
