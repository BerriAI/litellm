from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.systemone import (
    SYSTEM_ONE_RESPONSE_ADAPTER,
    decisions_response,
    question_keys,
    system_one_request,
)
from litellm.types.decisions import ChoiceAnswer, DecisionsRequest, PredicateAnswer, ScoreAnswer

_INPUT: Final = "The export job hangs at 99% and never finishes"
_QUESTIONS: Final[Sequence[Mapping[str, object]]] = (
    {"type": "predicate", "name": "is_defect", "instructions": "Is this a defect?"},
    {
        "type": "choice",
        "name": "sentiment",
        "instructions": "How does the customer feel?",
        "choices": [{"value": "positive"}, {"value": "negative", "description": "unhappy"}],
    },
    {
        "type": "score",
        "name": "severity",
        "instructions": "How severe is it?",
        "levels": [{"label": "none"}, {"label": "low"}, {"label": "high", "description": "blocks users"}],
    },
)
_SYSTEM_ONE_QUESTIONS: Final[Mapping[str, object]] = {
    "is_defect": {"type": "noul", "instructions": "Is this a defect?"},
    "sentiment": {
        "type": "choice",
        "instructions": "How does the customer feel?",
        "criteria": {"positive": None, "negative": "unhappy"},
    },
    "severity": {"type": "score", "instructions": "How severe is it?", "criteria": ["none", "low", "blocks users"]},
}
_SYSTEM_ONE_RESPONSE: Final[Mapping[str, object]] = {
    "model": "jev-1.13",
    "answers": {
        "is_defect": {"type": "noul", "noul": 0.9},
        "sentiment": {
            "type": "choice",
            "choice": "positive",
            "confidence": 0.8,
            "probabilities": {"positive": 0.8, "negative": 0.2},
        },
        "severity": {
            "type": "score",
            "score": 1,
            "confidence": 0.7,
            "legend": {"0": "none", "1": "low", "2": "high"},
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
        },
    },
    "usage": {"input_tokens": 367, "output_tokens": 3},
}
_EXPECTED_ANSWERS: Final[Sequence[Mapping[str, object]]] = (
    {"type": "predicate", "name": "is_defect", "probability": 0.9},
    {
        "type": "choice",
        "name": "sentiment",
        "choice": "positive",
        "probabilities": [{"value": "positive", "probability": 0.8}, {"value": "negative", "probability": 0.2}],
        "confidence": 0.8,
    },
    {
        "type": "score",
        "name": "severity",
        "score": 1.0,
        "probabilities": [
            {"value": 0, "label": "none", "probability": 0.1},
            {"value": 1, "label": "low", "probability": 0.8},
            {"value": 2, "label": "high", "probability": 0.1},
        ],
        "confidence": 0.7,
    },
)
_EXPECTED_USAGE: Final[Mapping[str, object]] = {
    "input_tokens": 367,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 3,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 370,
}
_REQUEST_ADAPTER: Final[TypeAdapter[DecisionsRequest]] = TypeAdapter(DecisionsRequest)


def _request(
    input_value: object = _INPUT,
    questions: Sequence[Mapping[str, object]] = _QUESTIONS,
) -> DecisionsRequest:
    return _REQUEST_ADAPTER.validate_python({"model": "jev-1.13", "input": input_value, "questions": questions})


def _predicate(name: str | None = "is_defect") -> tuple[Mapping[str, object]]:
    return ({"type": "predicate", "name": name, "instructions": "Is this a defect?"},)


def test_openai_request_becomes_the_system_one_body() -> None:
    assert system_one_request("jev-1.13", _request(), "typesafe") == {
        "model": "jev-1.13",
        "state": _INPUT,
        "questions": _SYSTEM_ONE_QUESTIONS,
    }


def test_system_one_answers_become_openai_answers_in_question_order() -> None:
    response: Final = decisions_response(
        SYSTEM_ONE_RESPONSE_ADAPTER.validate_python(_SYSTEM_ONE_RESPONSE), _request(), "typesafe"
    )

    assert response.model_dump(mode="json") == {
        "model": "jev-1.13",
        "answers": list(_EXPECTED_ANSWERS),
        "usage": _EXPECTED_USAGE,
    }
    assert isinstance(response.answers[0], PredicateAnswer)
    assert isinstance(response.answers[1], ChoiceAnswer)
    assert isinstance(response.answers[2], ScoreAnswer)


def test_user_messages_are_joined_into_one_system_one_state() -> None:
    messages: Final = [
        {"role": "user", "content": "first"},
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "second"}, {"type": "input_text", "text": "third"}],
        },
    ]

    body: Final = system_one_request("jev-1.13", _request(messages, _predicate()), "typesafe")

    assert body["state"] == "first\nsecond\nthird"


@pytest.mark.parametrize(
    ("label", "input_value", "questions"),
    (
        (
            "input_image",
            [{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AA=="}]}],
            _predicate(),
        ),
        ("boolean choice", _INPUT, [{"type": "choice", "instructions": "Refund?", "choices": [{"value": True}]}]),
        ("unique name", _INPUT, [*_predicate(), *_predicate()]),
        (
            "repeated choice",
            _INPUT,
            [{"type": "choice", "instructions": "Refund?", "choices": [{"value": "yes"}, {"value": "yes"}]}],
        ),
    ),
)
def test_what_system_one_cannot_express_is_a_400(
    label: str,
    input_value: object,
    questions: Sequence[Mapping[str, object]],
) -> None:
    with pytest.raises(BaseLLMException, match=label) as error:
        system_one_request("jev-1.13", _request(input_value, questions), "perplexity")

    assert error.value.status_code == 400
    assert "perplexity" in error.value.message


def test_unnamed_questions_get_positional_keys_that_never_shadow_a_supplied_name() -> None:
    request: Final = _request(questions=[*_predicate(None), *_predicate("q0"), *_predicate(None)])

    assert question_keys(request.questions, "typesafe") == ("_q0", "q0", "q2")
    noul: Final = {"type": "noul", "instructions": "Is this a defect?"}
    assert system_one_request("jev-1.13", request, "typesafe")["questions"] == {"_q0": noul, "q0": noul, "q2": noul}


def test_positional_answers_come_back_in_question_order_without_a_name() -> None:
    request: Final = _request(questions=[*_predicate(None), *_predicate("q0")])
    system_one: Final = SYSTEM_ONE_RESPONSE_ADAPTER.validate_python(
        {"answers": {"_q0": {"type": "noul", "noul": 0.25}, "q0": {"type": "noul", "noul": 0.75}}}
    )

    response: Final = decisions_response(system_one, request, "typesafe")

    assert [answer.model_dump(mode="json") for answer in response.answers] == [
        {"type": "predicate", "name": None, "probability": 0.25},
        {"type": "predicate", "name": "q0", "probability": 0.75},
    ]
    assert response.usage.model_dump(mode="json") == {
        **_EXPECTED_USAGE,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_a_reply_without_a_model_reports_the_requested_model() -> None:
    system_one: Final = SYSTEM_ONE_RESPONSE_ADAPTER.validate_python(
        {k: v for k, v in _SYSTEM_ONE_RESPONSE.items() if k != "model"}
    )

    request: Final = _REQUEST_ADAPTER.validate_python(
        {"model": "typesafe/jev-1.13.0", "input": _INPUT, "questions": _QUESTIONS}
    )

    response: Final = decisions_response(system_one, request, "typesafe")

    assert response.model == "typesafe/jev-1.13.0"


def test_a_choice_the_provider_left_out_of_probabilities_is_reported_at_zero() -> None:
    system_one: Final = SYSTEM_ONE_RESPONSE_ADAPTER.validate_python(
        {
            "answers": {
                "sentiment": {
                    "type": "choice",
                    "choice": "positive",
                    "confidence": 1.0,
                    "probabilities": {"positive": 1.0},
                }
            }
        }
    )
    request: Final = _request(questions=_QUESTIONS[1:2])

    response: Final = decisions_response(system_one, request, "typesafe")

    assert response.answers[0].model_dump(mode="json") == {
        "type": "choice",
        "name": "sentiment",
        "choice": "positive",
        "probabilities": [{"value": "positive", "probability": 1.0}, {"value": "negative", "probability": 0.0}],
        "confidence": 1.0,
    }


@pytest.mark.parametrize(
    "answers",
    (
        {},
        {"is_defect": {"type": "choice", "choice": "yes", "confidence": 1.0, "probabilities": {"yes": 1.0}}},
    ),
)
def test_a_reply_without_a_matching_answer_is_a_server_error(answers: Mapping[str, object]) -> None:
    system_one: Final = SYSTEM_ONE_RESPONSE_ADAPTER.validate_python({"answers": answers})

    with pytest.raises(BaseLLMException, match="no predicate answer for question 'is_defect'") as error:
        decisions_response(system_one, _request(questions=_predicate()), "typesafe")

    assert error.value.status_code == 500
