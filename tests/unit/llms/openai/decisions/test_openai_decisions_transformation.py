from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.llms.base_llm.decisions.transformation import ir_to_systemone_response, systemone_request_to_ir
from litellm.llms.openai.decisions.transformation import (
    OpenAIDecisionsConfig,
    ir_to_openai_request,
    ir_to_openai_response,
    openai_request_to_ir,
)
from litellm.types.decisions import (
    DecisionsIRRequest,
    DecisionsRequestBody,
    OpenAIDecisionRequestBody,
    UnsupportedDecisionsRequest,
)

_SYSTEMONE_BODY: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)
_OPENAI_BODY: Final[TypeAdapter[OpenAIDecisionRequestBody]] = TypeAdapter(OpenAIDecisionRequestBody)

_OPENAI_REQUEST: Final[Mapping[str, object]] = {
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "The screen is cracked."},
                {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "high"},
            ],
        },
        {"type": "message", "role": "user", "content": "Order 1234."},
    ],
    "questions": [
        {"type": "predicate", "name": "damaged", "instructions": "Is the item damaged?"},
        {
            "type": "choice",
            "instructions": "Should we refund?",
            "choices": [{"value": True, "description": "Refund now"}, {"value": "escalate"}],
        },
        {
            "type": "score",
            "name": "severity",
            "instructions": "How severe is it?",
            "levels": [{"label": "minor"}, {"label": "major", "description": "Product unusable"}],
        },
        {"type": "predicate", "name": "fraud", "instructions": "Is this fraud?"},
    ],
    "safety_identifier": "end-user-1",
}
_PREDICATE_ANSWER: Final[Mapping[str, object]] = {"type": "predicate", "name": "damaged", "probability": 0.95}
_CHOICE_ANSWER: Final[Mapping[str, object]] = {
    "type": "choice",
    "name": None,
    "choice": True,
    "probabilities": [{"value": True, "probability": 0.9}, {"value": "escalate", "probability": 0.1}],
    "confidence": 0.8,
}
_REFUSAL_ANSWER: Final[Mapping[str, object]] = {"type": "refusal", "name": "fraud"}
_USAGE: Final[Mapping[str, object]] = {
    "input_tokens": 383,
    "input_tokens_details": {"cached_tokens": 256, "cache_write_tokens": 64},
    "output_tokens": 2,
    "output_tokens_details": {"reasoning_tokens": 1},
    "total_tokens": 385,
}
_OPENAI_RESPONSE: Final[Mapping[str, object]] = {
    "model": "gpt-6-luna",
    "answers": [
        _PREDICATE_ANSWER,
        _CHOICE_ANSWER,
        {
            "type": "score",
            "name": "severity",
            "score": 0.7,
            "probabilities": [
                {"value": 0, "label": "minor", "probability": 0.3},
                {"value": 1, "label": "major", "probability": 0.7},
            ],
            "confidence": 0.6,
        },
        _REFUSAL_ANSWER,
    ],
    "usage": _USAGE,
}


def _openai_ir(raw: Mapping[str, object]) -> DecisionsIRRequest:
    return openai_request_to_ir(_OPENAI_BODY.validate_python(raw))


def test_an_openai_request_reaches_openai_unchanged() -> None:
    assert ir_to_openai_request("gpt-6-luna", _openai_ir(_OPENAI_REQUEST)) == {
        "model": "gpt-6-luna",
        **_OPENAI_REQUEST,
    }


def test_an_openai_response_reaches_the_caller_unchanged() -> None:
    ir: Final = _openai_ir(_OPENAI_REQUEST)

    parsed: Final = OpenAIDecisionsConfig().parse_response(_OPENAI_RESPONSE, ir)

    assert ir_to_openai_response(parsed, ir, "requested").model_dump(mode="json") == _OPENAI_RESPONSE


def test_answers_openai_did_not_return_are_refusals() -> None:
    ir: Final = _openai_ir(_OPENAI_REQUEST)
    payload: Final = {**_OPENAI_RESPONSE, "answers": [_PREDICATE_ANSWER]}

    response: Final = ir_to_openai_response(OpenAIDecisionsConfig().parse_response(payload, ir), ir, "requested")

    assert [answer.type for answer in response.answers] == ["predicate", "refusal", "refusal", "refusal"]
    assert [answer.name for answer in response.answers] == ["damaged", None, "severity", "fraud"]


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

    assert ir_to_openai_request("gpt-6-luna", systemone_request_to_ir(request)) == {
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
            {
                "type": "score",
                "name": "severity",
                "instructions": "Which level best fits the input?",
                "levels": [{"label": "minor"}, {"label": '{"label": "major"}'}],
            },
        ],
    }


def test_systemone_questions_without_instructions_become_valid_openai_questions() -> None:
    request: Final = _SYSTEMONE_BODY.validate_python(
        {
            "state": "Screen cracked",
            "questions": {
                "damaged": {"type": "noul", "criteria": {"false": None}},
                "action": {"type": "choice", "criteria": {"refund": None, "escalate": None}},
                "severity": {"type": "score", "criteria": ["minor", "major"]},
            },
        }
    )

    body: Final = ir_to_openai_request("gpt-6-luna", systemone_request_to_ir(request))

    assert all(question.instructions for question in _OPENAI_BODY.validate_python(body).questions)


def test_systemone_images_reach_openai_as_input_images_ahead_of_the_state() -> None:
    request: Final = _SYSTEMONE_BODY.validate_python(
        {
            "state": {"ticket": 1234},
            "questions": {"damaged": {"type": "noul", "instructions": "Is the item damaged?"}},
            "images": ["data:image/png;base64,AA==", {"content_type": "image/webp", "base64": "BB=="}],
        }
    )

    assert ir_to_openai_request("gpt-6-luna", systemone_request_to_ir(request)) == {
        "model": "gpt-6-luna",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": "data:image/png;base64,AA=="},
                    {"type": "input_image", "image_url": "data:image/webp;base64,BB=="},
                    {"type": "input_text", "text": '{"ticket": 1234}'},
                ],
            }
        ],
        "questions": [{"type": "predicate", "name": "damaged", "instructions": "Is the item damaged?"}],
    }


@pytest.mark.parametrize(
    "question",
    ({"type": "choice", "criteria": {"refund": None}}, {"type": "score", "criteria": ["minor"]}),
    ids=("choice", "score"),
)
def test_a_systemone_question_with_one_option_is_unsupported_by_openai(question: Mapping[str, object]) -> None:
    request: Final = _SYSTEMONE_BODY.validate_python(
        {
            "state": "Screen cracked",
            "questions": {"damaged": {"type": "noul", "instructions": "Damaged?"}, "q": question},
        }
    )

    assert isinstance(ir_to_openai_request("gpt-6-luna", systemone_request_to_ir(request)), UnsupportedDecisionsRequest)


def test_an_openai_response_becomes_systemone_answers_with_the_callers_score_labels() -> None:
    ir: Final = systemone_request_to_ir(
        _SYSTEMONE_BODY.validate_python(
            {
                "state": "Screen cracked",
                "questions": {
                    "damaged": {"type": "noul", "instructions": "Damaged?"},
                    "action": {"type": "choice", "criteria": {"true": None, "escalate": None}},
                    "severity": {"type": "score", "criteria": ["minor", {"label": "major"}]},
                    "fraud": {"type": "noul", "instructions": "Fraud?"},
                },
            }
        )
    )
    payload: Final = {
        **_OPENAI_RESPONSE,
        "answers": [
            _PREDICATE_ANSWER,
            _CHOICE_ANSWER,
            {
                "type": "score",
                "name": "severity",
                "score": 0.7,
                "probabilities": [
                    {"value": 0, "label": "minor", "probability": 0.3},
                    {"value": 1, "label": '{"label": "major"}', "probability": 0.7},
                ],
                "confidence": 0.6,
            },
            _REFUSAL_ANSWER,
        ],
    }

    response: Final = ir_to_systemone_response(OpenAIDecisionsConfig().parse_response(payload, ir), ir)

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
                "score": 0.7,
                "confidence": 0.6,
                "legend": {"0": "minor", "1": {"label": "major"}},
                "probabilities": {"0": 0.3, "1": 0.7},
            },
        },
        "usage": {"input_tokens": 383, "output_tokens": 2},
    }
    assert response.usage is not None
    assert (response.usage.cached_tokens, response.usage.cache_write_tokens) == (256, 64)
