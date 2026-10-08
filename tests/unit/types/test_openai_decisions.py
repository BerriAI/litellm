from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.types.openai_decisions import (
    ChoiceAnswer,
    DecisionsRequestBody,
    DecisionsResponse,
    RefusalAnswer,
    ScoreQuestion,
)

_REQUEST: Final[Mapping[str, object]] = {
    "input": [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Is this receipt a valid business expense?"},
                {"type": "input_image", "image_url": "https://example.com/receipt.png", "detail": "high"},
            ],
        }
    ],
    "questions": [
        {"type": "predicate", "name": "is_expense", "instructions": "Is this a business expense?"},
        {
            "type": "choice",
            "name": "approve",
            "instructions": "Should this be approved?",
            "choices": [{"value": True, "description": "approve"}, {"value": False, "description": "reject"}],
        },
        {
            "type": "score",
            "name": "risk",
            "instructions": "How risky is this expense?",
            "levels": [{"label": "low"}, {"label": "high", "description": "needs a manager"}],
        },
    ],
    "safety_identifier": "user-123",
}
_RESPONSE: Final[Mapping[str, object]] = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "is_expense", "probability": 0.92},
        {
            "type": "choice",
            "name": "approve",
            "choice": True,
            "probabilities": [{"value": True, "probability": 0.7}, {"value": False, "probability": 0.3}],
            "confidence": 0.7,
        },
        {"type": "refusal", "name": "risk"},
    ],
    "usage": {
        "input_tokens": 120,
        "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 0},
        "output_tokens": 12,
        "output_tokens_details": {"reasoning_tokens": 4},
        "total_tokens": 132,
    },
}
_REQUEST_ADAPTER: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)
_RESPONSE_ADAPTER: Final[TypeAdapter[DecisionsResponse]] = TypeAdapter(DecisionsResponse)


def test_the_documented_request_round_trips_with_its_boolean_choices_and_image_part() -> None:
    request: Final = _REQUEST_ADAPTER.validate_python(_REQUEST)

    assert request.model_dump(mode="json", exclude_none=True) == _REQUEST
    assert isinstance(request.questions[2], ScoreQuestion)


def test_the_documented_response_keeps_answer_order_refusals_and_token_details() -> None:
    response: Final = _RESPONSE_ADAPTER.validate_python(_RESPONSE)

    assert response.model_dump(mode="json") == _RESPONSE
    assert isinstance(response.answers[1], ChoiceAnswer)
    assert isinstance(response.answers[2], RefusalAnswer)


_OFF_SPEC: Final[tuple[tuple[str, object], ...]] = (
    ("questions", [{"type": "noul", "name": "q", "instructions": "x"}]),
    ("questions", [{"type": "choice", "name": "q", "instructions": "x", "choices": [{"value": 1}]}]),
    ("questions", [{"type": "score", "name": "q", "levels": [{"label": "low"}]}]),
    ("input", {"state": "not an OpenAI input"}),
)


@pytest.mark.parametrize(("field", "value"), _OFF_SPEC)
def test_requests_off_the_spec_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        _REQUEST_ADAPTER.validate_python({**_REQUEST, field: value})


_EMPTY_COLLECTIONS: Final[tuple[tuple[str, list[object]], ...]] = (
    ("questions", []),
    ("questions", [{"type": "choice", "name": "q", "instructions": "x", "choices": []}]),
    ("questions", [{"type": "score", "name": "q", "instructions": "x", "levels": []}]),
)


@pytest.mark.parametrize(("field", "value"), _EMPTY_COLLECTIONS)
def test_empty_collections_are_left_for_the_provider_to_judge(field: str, value: list[object]) -> None:
    request: Final = _REQUEST_ADAPTER.validate_python({**_REQUEST, field: value})

    assert request.model_dump(mode="json", exclude_none=True)[field] == value


def test_choice_values_keep_their_type_so_a_string_true_and_a_boolean_true_stay_distinct() -> None:
    answer: Final = {
        "type": "choice",
        "name": "approve",
        "choice": "true",
        "probabilities": [{"value": "true", "probability": 0.6}, {"value": True, "probability": 0.4}],
        "confidence": 0.6,
    }

    response: Final = _RESPONSE_ADAPTER.validate_python({**_RESPONSE, "answers": [answer]})

    assert response.model_dump(mode="json")["answers"] == [answer]
    with pytest.raises(ValidationError):
        _RESPONSE_ADAPTER.validate_python({**_RESPONSE, "answers": [{**answer, "choice": 1}]})


_OFF_SPEC_RESPONSE: Final[tuple[str, ...]] = ("model", "usage")


@pytest.mark.parametrize("field", _OFF_SPEC_RESPONSE)
def test_responses_missing_a_required_field_are_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        _RESPONSE_ADAPTER.validate_python({k: v for k, v in _RESPONSE.items() if k != field})


def test_usage_without_token_details_is_rejected() -> None:
    usage: Final = {"input_tokens": 120, "output_tokens": 12, "total_tokens": 132}

    with pytest.raises(ValidationError):
        _RESPONSE_ADAPTER.validate_python({**_RESPONSE, "usage": usage})


def test_hidden_params_live_outside_the_wire_body() -> None:
    response: Final = _RESPONSE_ADAPTER.validate_python(_RESPONSE)

    response.set_hidden_params({"custom_llm_provider": "openai"})

    assert response.hidden_params == {"custom_llm_provider": "openai"}
    assert "_hidden_params" not in response.model_dump(mode="json")
