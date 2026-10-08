import json
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import assert_never

from litellm.types.decisions import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionsJSON,
    DecisionsRequestBody,
    DecisionsResponse,
    DecisionsUsage,
    NoulAnswer,
    NoulQuestion,
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
    ScoreQuestion,
    systemone_choice_key,
)

_OPENAI_ONLY_FIELDS: Final = frozenset({"input", "questions", "safety_identifier"})
_OPENAI_RESPONSE_ADAPTER: Final[TypeAdapter[OpenAIDecisionResponse]] = TypeAdapter(OpenAIDecisionResponse)


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


def _text(value: DecisionsJSON) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _instructions(instructions: DecisionsJSON | None) -> Mapping[str, str]:
    return {} if instructions is None else {"instructions": _text(instructions)}


def _predicate_instructions(question: NoulQuestion) -> str:
    instructions: Final = () if question.instructions is None else (_text(question.instructions),)
    criteria: Final = tuple(
        f"Answer {answer} when: {_text(rule)}" for answer, rule in (question.criteria or {}).items() if rule is not None
    )
    return "\n\n".join((*instructions, *criteria))


def _choice(value: str, description: DecisionsJSON | None) -> Mapping[str, str]:
    return {"value": value} if description is None else {"value": value, "description": _text(description)}


def _openai_question(name: str, question: DecisionQuestion) -> Mapping[str, object]:
    match question:
        case NoulQuestion():
            return {"type": "predicate", "name": name, "instructions": _predicate_instructions(question)}
        case ChoiceQuestion():
            return {
                "type": "choice",
                "name": name,
                **_instructions(question.instructions),
                "choices": [_choice(value, description) for value, description in question.criteria.items()],
            }
        case ScoreQuestion():
            return {
                "type": "score",
                "name": name,
                **_instructions(question.instructions),
                "levels": [{"label": _text(criterion)} for criterion in question.criteria],
            }
        case _:
            assert_never(question)


def to_openai_request(model: str, request: DecisionsRequestBody) -> Mapping[str, object]:
    return {
        "model": model,
        "input": _text(request.state),
        "questions": [_openai_question(name, question) for name, question in request.questions.items()],
    }


def _systemone_answer(answer: OpenAIDecisionAnswer) -> DecisionAnswer | None:
    match answer:
        case OpenAIPredicateAnswer():
            return NoulAnswer(type="noul", noul=answer.probability)
        case OpenAIChoiceAnswer():
            return ChoiceAnswer(
                type="choice",
                choice=systemone_choice_key(answer.choice),
                confidence=answer.confidence,
                probabilities={systemone_choice_key(item.value): item.probability for item in answer.probabilities},
            )
        case OpenAIScoreAnswer():
            return ScoreAnswer(
                type="score",
                score=answer.score,
                confidence=answer.confidence,
                legend={str(item.value): item.label for item in answer.probabilities},
                probabilities={str(item.value): item.probability for item in answer.probabilities},
            )
        case OpenAIRefusalAnswer():
            return None
        case _:
            assert_never(answer)


def _named_answers(answers: Sequence[OpenAIDecisionAnswer]) -> Mapping[str, DecisionAnswer]:
    translated: Final = ((answer.name, _systemone_answer(answer)) for answer in answers)
    return {name: answer for name, answer in translated if name is not None and answer is not None}


def to_systemone_response(payload: object) -> DecisionsResponse:
    response: Final = _OPENAI_RESPONSE_ADAPTER.validate_python(payload)
    return DecisionsResponse(
        model=response.model,
        answers=_named_answers(response.answers),
        usage=DecisionsUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens),
    )
