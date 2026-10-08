from collections.abc import Mapping, Sequence
from typing import Final

from typing_extensions import assert_never

from litellm.types.decisions import (
    ChoiceAnswer,
    DecisionAnswer,
    DecisionsResponse,
    DecisionsUsage,
    NoulAnswer,
    OpenAIChoiceAnswer,
    OpenAIChoiceProbability,
    OpenAIChoiceQuestion,
    OpenAIDecisionAnswer,
    OpenAIDecisionInputMessage,
    OpenAIDecisionQuestion,
    OpenAIDecisionRequestBody,
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
    OpenAIPredicateQuestion,
    OpenAIRefusalAnswer,
    OpenAIScoreAnswer,
    OpenAIScoreLevel,
    OpenAIScoreProbability,
    OpenAIScoreQuestion,
    ScoreAnswer,
    systemone_choice_key,
)

_OPENAI_ONLY_FIELDS: Final = frozenset({"input", "questions", "safety_identifier"})


def _message_text(message: OpenAIDecisionInputMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return "\n\n".join(part.text for part in message.content)


def _state(decision_input: str | Sequence[OpenAIDecisionInputMessage]) -> str:
    if isinstance(decision_input, str):
        return decision_input
    return "\n\n".join(_message_text(message) for message in decision_input)


def _level_criterion(level: OpenAIScoreLevel) -> str:
    return level.label if level.description is None else f"{level.label}: {level.description}"


def _systemone_question(question: OpenAIDecisionQuestion) -> Mapping[str, object]:
    match question:
        case OpenAIPredicateQuestion():
            return {"type": "noul", "instructions": question.instructions}
        case OpenAIChoiceQuestion():
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": {systemone_choice_key(option.value): option.description for option in question.choices},
            }
        case OpenAIScoreQuestion():
            return {
                "type": "score",
                "instructions": question.instructions,
                "criteria": [_level_criterion(level) for level in question.levels],
            }
        case _:
            assert_never(question)


def to_systemone_request(request_data: Mapping[str, object], body: OpenAIDecisionRequestBody) -> Mapping[str, object]:
    return {
        **{key: value for key, value in request_data.items() if key not in _OPENAI_ONLY_FIELDS},
        "state": _state(body.input),
        "questions": {str(index): _systemone_question(question) for index, question in enumerate(body.questions)},
    }


def _openai_answer(question: OpenAIDecisionQuestion, answer: DecisionAnswer | None) -> OpenAIDecisionAnswer:
    match question, answer:
        case OpenAIPredicateQuestion(), NoulAnswer():
            return OpenAIPredicateAnswer(name=question.name, probability=answer.noul)
        case OpenAIChoiceQuestion(), ChoiceAnswer():
            typed_values: Final = {systemone_choice_key(option.value): option.value for option in question.choices}
            return OpenAIChoiceAnswer(
                name=question.name,
                choice=typed_values.get(answer.choice, answer.choice),
                probabilities=tuple(
                    OpenAIChoiceProbability(
                        value=option.value,
                        probability=answer.probabilities.get(systemone_choice_key(option.value), 0.0),
                    )
                    for option in question.choices
                ),
                confidence=answer.confidence,
            )
        case OpenAIScoreQuestion(), ScoreAnswer():
            return OpenAIScoreAnswer(
                name=question.name,
                score=answer.score,
                probabilities=tuple(
                    OpenAIScoreProbability(
                        value=index, label=level.label, probability=answer.probabilities.get(str(index), 0.0)
                    )
                    for index, level in enumerate(question.levels)
                ),
                confidence=answer.confidence,
            )
        case _:
            return OpenAIRefusalAnswer(name=question.name)


def to_openai_response(
    response: DecisionsResponse, questions: Sequence[OpenAIDecisionQuestion], requested_model: str
) -> OpenAIDecisionResponse:
    usage: Final = response.usage or DecisionsUsage()
    return OpenAIDecisionResponse(
        model=response.model or requested_model,
        answers=tuple(
            _openai_answer(question, response.answers.get(str(index))) for index, question in enumerate(questions)
        ),
        usage=OpenAIDecisionUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        ),
    )
