from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm.decisions.openai_transformation import to_systemone_request
from litellm.types.decisions import MAX_DECISION_QUESTIONS, DecisionsRequestBody, OpenAIDecisionRequestBody

_OPENAI_BODY: Final[TypeAdapter[OpenAIDecisionRequestBody]] = TypeAdapter(OpenAIDecisionRequestBody)
_SYSTEMONE_BODY: Final[TypeAdapter[DecisionsRequestBody]] = TypeAdapter(DecisionsRequestBody)


def _openai_request(question_count: int) -> Mapping[str, object]:
    return {
        "model": "decider",
        "input": "The package arrived with a broken screen.",
        "questions": [{"type": "predicate", "instructions": f"Question {index}?"} for index in range(question_count)],
    }


def test_the_largest_openai_request_accepted_translates_to_a_valid_systemone_request() -> None:
    raw: Final = _openai_request(MAX_DECISION_QUESTIONS)

    translated: Final = _SYSTEMONE_BODY.validate_python(to_systemone_request(raw, _OPENAI_BODY.validate_python(raw)))

    assert len(translated.questions) == MAX_DECISION_QUESTIONS


def test_an_openai_request_with_more_questions_than_systemone_takes_is_rejected_before_translation() -> None:
    with pytest.raises(ValidationError, match="questions"):
        _OPENAI_BODY.validate_python(_openai_request(MAX_DECISION_QUESTIONS + 1))
