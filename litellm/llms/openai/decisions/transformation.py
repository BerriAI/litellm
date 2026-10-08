import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    OpenAIDecisionAnswer,
    OpenAIDecisionResponse,
    OpenAIPredicateAnswer,
    OpenAIRefusalAnswer,
    OpenAIScoreAnswer,
    ScoreAnswer,
    ScoreQuestion,
    systemone_choice_key,
)

_OPENAI_RESPONSE_ADAPTER: Final[TypeAdapter[OpenAIDecisionResponse]] = TypeAdapter(OpenAIDecisionResponse)


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


@dataclass(frozen=True, slots=True)
class OpenAIDecisionsEndpoint:
    api_key_env: tuple[str, ...] = ("OPENAI_API_KEY",)
    api_base_env: str = "OPENAI_BASE_URL"
    api_key_required: bool = True

    def default_api_base(self) -> str | None:
        return "https://api.openai.com"

    def missing_api_base_message(self, provider: str) -> str:
        return f"api_base is required for Decisions provider '{provider}'"

    def canonical_model(self, model: str) -> str:
        return model

    def request_model(self, model: str) -> str:
        return model

    def endpoint_url(self, api_base: str, model: str) -> str:
        return f"{api_base.rstrip('/').removesuffix('/v1')}/v1/decisions"

    def request_body(self, model: str, request: DecisionsRequestBody) -> Mapping[str, object]:
        return to_openai_request(model, request)

    def unwrap_response(self, payload: object) -> DecisionsResponse:
        return to_systemone_response(payload)


OPENAI_DECISIONS_ENDPOINT: Final[OpenAIDecisionsEndpoint] = OpenAIDecisionsEndpoint()
