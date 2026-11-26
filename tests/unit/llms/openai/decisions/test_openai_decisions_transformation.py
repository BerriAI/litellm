import time
from collections.abc import Mapping
from dataclasses import replace
from typing import Final

import httpx
import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.llms.base_llm.decisions.transformation import (
    DecisionsResponseValidationError,
    ir_to_systemone_response,
    systemone_request_to_ir,
)
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
_USAGE_WITH_WRITE_LIFETIMES: Final[Mapping[str, object]] = {
    **_USAGE,
    "input_tokens_details": {
        "cached_tokens": 256,
        "cache_write_tokens": 64,
        "cache_write_tokens_details": {"ttl_30m_tokens": 16, "ttl_12h_tokens": 48},
    },
}
_USAGE_WITH_AN_UNKNOWN_WRITE_LIFETIME: Final[Mapping[str, object]] = {
    **_USAGE,
    "input_tokens_details": {
        "cached_tokens": 256,
        "cache_write_tokens": 64,
        "cache_write_tokens_details": {"ttl_30m_tokens": 16, "ttl_12h_tokens": 40, "ttl_7d_tokens": 8},
    },
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


@pytest.mark.parametrize(
    "usage",
    (_USAGE, _USAGE_WITH_WRITE_LIFETIMES, _USAGE_WITH_AN_UNKNOWN_WRITE_LIFETIME),
    ids=("no_write_lifetimes", "write_lifetimes", "unknown_write_lifetime"),
)
def test_an_openai_response_reaches_the_caller_unchanged(usage: Mapping[str, object]) -> None:
    ir: Final = _openai_ir(_OPENAI_REQUEST)
    payload: Final = {**_OPENAI_RESPONSE, "usage": usage}

    parsed: Final = OpenAIDecisionsConfig().parse_response(payload, ir)

    assert ir_to_openai_response(parsed, ir, "requested").model_dump(mode="json") == payload


@pytest.mark.parametrize("answer_type", ["choice", "score"])
def test_a_choice_or_score_answer_without_confidence_stays_a_provider_error(answer_type: str) -> None:
    answers: Final = [
        {key: value for key, value in answer.items() if answer["type"] != answer_type or key != "confidence"}
        for answer in _OPENAI_RESPONSE["answers"]
    ]
    with pytest.raises(DecisionsResponseValidationError) as raised:
        OpenAIDecisionsConfig().transform_decisions_response(
            "gpt-6-luna",
            "openai",
            httpx.Response(200, json={**_OPENAI_RESPONSE, "answers": answers}),
            _openai_ir(_OPENAI_REQUEST),
        )
    assert raised.value.status_code == 500


def test_provider_extras_stay_off_the_openai_response_unless_the_provider_names_them() -> None:
    ir: Final = _openai_ir(_OPENAI_REQUEST)
    parsed: Final = replace(
        OpenAIDecisionsConfig().parse_response(_OPENAI_RESPONSE, ir), extra={"provider": "upstream", "citations": []}
    )

    assert ir_to_openai_response(parsed, ir, "requested").model_dump(mode="json") == _OPENAI_RESPONSE
    assert ir_to_openai_response(parsed, ir, "requested", frozenset({"citations"})).model_dump(mode="json") == {
        **_OPENAI_RESPONSE,
        "citations": [],
    }


def test_answers_openai_did_not_return_are_refusals() -> None:
    ir: Final = _openai_ir(_OPENAI_REQUEST)
    payload: Final = {**_OPENAI_RESPONSE, "answers": [_PREDICATE_ANSWER]}

    response: Final = ir_to_openai_response(OpenAIDecisionsConfig().parse_response(payload, ir), ir, "requested")

    assert [answer.type for answer in response.answers] == ["predicate", "refusal", "refusal", "refusal"]
    assert [answer.name for answer in response.answers] == ["damaged", None, "severity", "fraud"]


@pytest.mark.parametrize("order", [(3, 2, 1, 0), (1, 2, 3, 0), (0, 2, 1, 3)], ids=("reversed", "rotated", "swapped"))
def test_answers_openai_returned_out_of_order_keep_their_own_values(order: tuple[int, ...]) -> None:
    ir: Final = _openai_ir(_OPENAI_REQUEST)
    payload: Final = {**_OPENAI_RESPONSE, "answers": [_OPENAI_RESPONSE["answers"][index] for index in order]}

    parsed: Final = OpenAIDecisionsConfig().parse_response(payload, ir)

    assert ir_to_openai_response(parsed, ir, "requested").model_dump(mode="json") == _OPENAI_RESPONSE


def test_an_out_of_order_answer_stays_with_its_question_when_another_is_missing() -> None:
    ir: Final = _openai_ir(_OPENAI_REQUEST)
    payload: Final = {**_OPENAI_RESPONSE, "answers": [_REFUSAL_ANSWER, _PREDICATE_ANSWER]}

    response: Final = ir_to_openai_response(OpenAIDecisionsConfig().parse_response(payload, ir), ir, "requested")

    assert [answer.model_dump(mode="json") for answer in response.answers] == [
        _PREDICATE_ANSWER,
        {"type": "refusal", "name": None},
        {"type": "refusal", "name": "severity"},
        _REFUSAL_ANSWER,
    ]


def test_forty_thousand_answers_returned_in_reverse_are_put_back_in_question_order_quickly() -> None:
    question_count: Final = 40_000
    ir: Final = _openai_ir(
        {
            "input": "The screen is cracked.",
            "questions": [
                {"type": "predicate", "name": f"q{index}", "instructions": "Is the item damaged?"}
                for index in range(question_count)
            ],
        }
    )
    answers: Final = tuple(
        {"type": "predicate", "name": f"q{index}", "probability": index / question_count}
        for index in range(question_count)
    )

    started: Final = time.perf_counter()
    parsed: Final = OpenAIDecisionsConfig().parse_response({**_OPENAI_RESPONSE, "answers": answers[::-1]}, ir)
    elapsed: Final = time.perf_counter() - started

    response: Final = ir_to_openai_response(parsed, ir, "requested")
    assert [answer.model_dump(mode="json") for answer in response.answers] == list(answers)
    assert elapsed < 5.0


_TWO_UNNAMED_QUESTIONS: Final[Mapping[str, object]] = {
    "input": "The screen is cracked.",
    "questions": [
        {"type": "predicate", "name": "damaged", "instructions": "Is the item damaged?"},
        {"type": "predicate", "instructions": "Is this fraud?"},
        {"type": "predicate", "instructions": "Is this urgent?"},
    ],
}
_UNNAMED_ANSWER: Final[Mapping[str, object]] = {"type": "predicate", "name": None, "probability": 0.2}


@pytest.mark.parametrize(
    ("request_body", "answers"),
    [
        (_OPENAI_REQUEST, [{**_PREDICATE_ANSWER, "name": "broken"}]),
        (_OPENAI_REQUEST, [_REFUSAL_ANSWER, _PREDICATE_ANSWER, _PREDICATE_ANSWER]),
        (_OPENAI_REQUEST, [*_OPENAI_RESPONSE["answers"], {**_PREDICATE_ANSWER, "name": "extra"}]),
        (_OPENAI_REQUEST, [*_OPENAI_RESPONSE["answers"], _PREDICATE_ANSWER]),
        (_TWO_UNNAMED_QUESTIONS, [_UNNAMED_ANSWER, _PREDICATE_ANSWER, _UNNAMED_ANSWER]),
        (_TWO_UNNAMED_QUESTIONS, [_UNNAMED_ANSWER, _PREDICATE_ANSWER]),
    ],
    ids=(
        "name_no_question_has",
        "repeated_name_out_of_order",
        "extra_answer",
        "extra_answer_repeating_a_name",
        "unnamed_answers_out_of_order",
        "one_unnamed_answer_for_two_unnamed_questions",
    ),
)
def test_answers_that_cannot_be_matched_to_a_question_are_a_provider_error(
    request_body: Mapping[str, object], answers: list[Mapping[str, object]]
) -> None:
    with pytest.raises(DecisionsResponseValidationError) as raised:
        OpenAIDecisionsConfig().transform_decisions_response(
            "gpt-6-luna",
            "openai",
            httpx.Response(200, json={**_OPENAI_RESPONSE, "answers": answers}),
            _openai_ir(request_body),
        )
    assert raised.value.status_code == 500


def test_unnamed_answers_in_question_order_keep_their_positions() -> None:
    ir: Final = _openai_ir(_TWO_UNNAMED_QUESTIONS)
    answers: Final = [_PREDICATE_ANSWER, _UNNAMED_ANSWER, {**_UNNAMED_ANSWER, "probability": 0.6}]

    response: Final = ir_to_openai_response(
        OpenAIDecisionsConfig().parse_response({**_OPENAI_RESPONSE, "answers": answers}, ir), ir, "requested"
    )

    assert [answer.model_dump(mode="json") for answer in response.answers] == answers


def test_a_box_answer_from_openai_keeps_its_box_instead_of_becoming_a_refusal() -> None:
    request: Final = {
        "input": _OPENAI_REQUEST["input"],
        "questions": [{"type": "box", "name": "crack", "instructions": "Where is the crack?"}],
    }
    box: Final = {"top": 0.1, "bottom": 0.4, "left": 0.2, "right": 0.6}
    payload: Final = {**_OPENAI_RESPONSE, "answers": [{"type": "box", "name": "crack", "box": box, "confidence": 0.7}]}
    ir: Final = _openai_ir(request)

    response: Final = ir_to_openai_response(OpenAIDecisionsConfig().parse_response(payload, ir), ir, "requested")

    assert response.answers[0].model_dump(mode="json") == {
        "type": "box",
        "name": "crack",
        "box": box,
        "confidence": 0.7,
    }


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
            {**_CHOICE_ANSWER, "name": "action"},
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


@pytest.mark.parametrize(
    ("choice_probabilities", "score_probabilities"),
    (
        ((), ()),
        (
            ({"value": True, "probability": 0.8},),
            ({"value": 1, "label": "major", "probability": 0.8},),
        ),
        (
            (
                {"value": True, "probability": 0.0},
                {"value": "escalate", "probability": 0.8},
            ),
            (
                {"value": 0, "label": "minor", "probability": 0.0},
                {"value": 1, "label": "major", "probability": 0.8},
            ),
        ),
        (
            ({"value": True, "probability": 0.0},),
            ({"value": 1, "label": "major", "probability": 0.0},),
        ),
    ),
    ids=("empty", "missing", "explicit_zero", "only_zero"),
)
def test_openai_probabilities_preserve_missing_entries_and_explicit_zeros(
    choice_probabilities: tuple[Mapping[str, object], ...],
    score_probabilities: tuple[Mapping[str, object], ...],
) -> None:
    ir: Final = _openai_ir(_OPENAI_REQUEST)
    payload: Final = {
        **_OPENAI_RESPONSE,
        "answers": [
            _PREDICATE_ANSWER,
            {**_CHOICE_ANSWER, "probabilities": choice_probabilities},
            {
                "type": "score",
                "name": "severity",
                "score": 1,
                "confidence": 0.8,
                "probabilities": score_probabilities,
            },
            _REFUSAL_ANSWER,
        ],
    }

    response: Final = ir_to_openai_response(OpenAIDecisionsConfig().parse_response(payload, ir), ir, "requested")

    assert response.answers[1].model_dump(mode="json")["probabilities"] == list(choice_probabilities)
    assert response.answers[2].model_dump(mode="json")["probabilities"] == list(score_probabilities)


_POLICY_TEXT: Final = "Refunds need an order number. Escalate anything about fraud."
_CACHE_REQUEST: Final[Mapping[str, object]] = {
    "input": [
        {
            "type": "message",
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": _POLICY_TEXT,
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "https://example.com/receipt.png",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {"type": "message", "role": "system", "content": "Answer for the support team."},
        {"type": "message", "role": "user", "content": "Please refund order 4411."},
    ],
    "questions": [{"type": "predicate", "name": "is_refund_request", "instructions": "Is this a refund request?"}],
    "prompt_cache_key": "billing-triage",
    "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
}


def test_prompt_cache_controls_reach_openai_unchanged() -> None:
    assert ir_to_openai_request("gpt-6-luna", _openai_ir(_CACHE_REQUEST)) == {
        "model": "gpt-6-luna",
        **_CACHE_REQUEST,
    }


def test_empty_prompt_cache_options_are_not_sent() -> None:
    request: Final = {
        "input": "Please refund order 4411.",
        "questions": _CACHE_REQUEST["questions"],
        "prompt_cache_options": {},
    }

    assert ir_to_openai_request("gpt-6-luna", _openai_ir(request)) == {
        "model": "gpt-6-luna",
        "input": "Please refund order 4411.",
        "questions": _CACHE_REQUEST["questions"],
    }


def test_prompt_cache_ttls_reach_openai_unchanged() -> None:
    request: Final = {
        "input": [
            {
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": _POLICY_TEXT,
                        "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "12h"},
                    }
                ],
            }
        ],
        "questions": _CACHE_REQUEST["questions"],
        "prompt_cache_options": {"mode": "explicit", "ttl": "12h"},
    }

    assert ir_to_openai_request("gpt-6-luna", _openai_ir(request)) == {"model": "gpt-6-luna", **request}


@pytest.mark.parametrize("role", ("developer", "system"))
def test_an_instruction_message_rejects_an_image(role: str) -> None:
    with pytest.raises(ValidationError, match=f"{role} messages only accept input text"):
        _OPENAI_BODY.validate_python(
            {
                "input": [
                    {
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_image", "image_url": "https://example.com/receipt.png"}],
                    }
                ],
                "questions": _CACHE_REQUEST["questions"],
            }
        )
