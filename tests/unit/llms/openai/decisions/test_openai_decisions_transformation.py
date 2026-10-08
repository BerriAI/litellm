from typing import Final

from pydantic import TypeAdapter

from litellm.llms.openai.decisions.transformation import to_openai_request, to_systemone_response
from litellm.types.decisions import DecisionsRequestBody

_SYSTEMONE_BODY: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)


def test_a_systemone_request_becomes_an_openai_request_with_questions_named_by_their_keys() -> None:
    request: Final = _SYSTEMONE_BODY.validate_python(
        {
            "state": {"ticket": 1234, "text": "Screen cracked"},
            "questions": {
                "damaged": {
                    "type": "noul",
                    "instructions": "Is the item damaged?",
                    "criteria": {"true": "Visible damage", "false": None},
                    "provider_field": "dropped",
                },
                "rubric_only": {"type": "noul", "criteria": {"true": {"signal": "refund"}}},
                "action": {
                    "type": "choice",
                    "instructions": {"policy": "refund-v2"},
                    "criteria": {"refund": "Within 30 days", "escalate": None},
                },
                "severity": {"type": "score", "criteria": ["minor", {"label": "major"}]},
            },
        }
    )

    assert to_openai_request("gpt-6-luna", request) == {
        "model": "gpt-6-luna",
        "input": '{"ticket": 1234, "text": "Screen cracked"}',
        "questions": [
            {
                "type": "predicate",
                "name": "damaged",
                "instructions": "Is the item damaged?\n\nAnswer true when: Visible damage",
            },
            {"type": "predicate", "name": "rubric_only", "instructions": 'Answer true when: {"signal": "refund"}'},
            {
                "type": "choice",
                "name": "action",
                "instructions": '{"policy": "refund-v2"}',
                "choices": [{"value": "refund", "description": "Within 30 days"}, {"value": "escalate"}],
            },
            {"type": "score", "name": "severity", "levels": [{"label": "minor"}, {"label": '{"label": "major"}'}]},
        ],
    }


def test_an_openai_response_becomes_systemone_answers_keyed_by_question_name_without_refusals() -> None:
    response: Final = to_systemone_response(
        {
            "model": "gpt-6-luna",
            "answers": [
                {"type": "predicate", "name": "damaged", "probability": 0.95},
                {
                    "type": "choice",
                    "name": "action",
                    "choice": True,
                    "probabilities": [{"value": True, "probability": 0.9}, {"value": "escalate", "probability": 0.1}],
                    "confidence": 0.8,
                },
                {
                    "type": "score",
                    "name": "severity",
                    "score": 1.72,
                    "probabilities": [
                        {"value": 0, "label": "minor", "probability": 0.0},
                        {"value": 1, "label": "major", "probability": 0.28},
                        {"value": 2, "label": "critical", "probability": 0.72},
                    ],
                    "confidence": 0.58,
                },
                {"type": "refusal", "name": "fraud"},
            ],
            "usage": {
                "input_tokens": 383,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 385,
            },
        }
    )

    assert response.model_dump(mode="json") == {
        "model": "gpt-6-luna",
        "answers": {
            "damaged": {"type": "noul", "noul": 0.95},
            "action": {
                "type": "choice",
                "choice": "true",
                "confidence": 0.8,
                "probabilities": {"true": 0.9, "escalate": 0.1},
            },
            "severity": {
                "type": "score",
                "score": 1.72,
                "confidence": 0.58,
                "legend": {"0": "minor", "1": "major", "2": "critical"},
                "probabilities": {"0": 0.0, "1": 0.28, "2": 0.72},
            },
        },
        "usage": {"input_tokens": 383, "output_tokens": 2},
    }
