from typing import Final

import pytest
import respx

import litellm
from litellm.types.decisions import OpenAIDecisionResponse

MANY_QUESTIONS: Final = 2_000
QUESTION_NAMES: Final = tuple(f"q{index}" for index in range(MANY_QUESTIONS))


@pytest.mark.benchmark
def test_decisions_over_many_questions() -> None:
    with respx.mock(assert_all_called=True) as router:
        router.post("https://api.perplexity.ai/v1/decisions").respond(
            json={
                "model": "pplx-decider-v1-27b",
                "answers": {name: {"type": "noul", "noul": 0.5} for name in QUESTION_NAMES},
                "usage": {"input_tokens": 12, "output_tokens": MANY_QUESTIONS},
            }
        )
        response: Final = litellm.decisions(
            model="perplexity/pplx-decider-v1-27b",
            api_key="benchmark-key",
            input="review",
            questions=[
                {"type": "predicate", "name": name, "instructions": f"Is {name} true?"} for name in QUESTION_NAMES
            ],
        )

    assert isinstance(response, OpenAIDecisionResponse)
    assert tuple(answer.name for answer in response.answers) == QUESTION_NAMES
