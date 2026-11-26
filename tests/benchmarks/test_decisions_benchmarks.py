from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.llms.openai.decisions.transformation import (
    OpenAIDecisionsConfig,
    ir_to_openai_response,
    openai_request_to_ir,
)
from litellm.types.decisions import OpenAIDecisionRequestBody

_QUESTION_COUNT: Final = 2000
_IR: Final = openai_request_to_ir(
    TypeAdapter(OpenAIDecisionRequestBody).validate_python(
        {
            "input": "The screen is cracked.",
            "questions": [
                {"type": "predicate", "name": f"q{index}", "instructions": "Is the item damaged?"}
                for index in range(_QUESTION_COUNT)
            ],
        }
    )
)
_ANSWERS: Final = tuple(
    {"type": "predicate", "name": f"q{index}", "probability": index / _QUESTION_COUNT}
    for index in range(_QUESTION_COUNT)
)
_REVERSED_RESPONSE: Final = {
    "model": "gpt-6-luna",
    "answers": _ANSWERS[::-1],
    "usage": {
        "input_tokens": 383,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 1},
        "total_tokens": 385,
    },
}


@pytest.mark.benchmark
def test_openai_decisions_two_thousand_answers_returned_in_reverse() -> None:
    parsed: Final = OpenAIDecisionsConfig().parse_response(_REVERSED_RESPONSE, _IR)

    response: Final = ir_to_openai_response(parsed, _IR, "requested")
    assert [answer.model_dump(mode="json") for answer in response.answers] == list(_ANSWERS)
