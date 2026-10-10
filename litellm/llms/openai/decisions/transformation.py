from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import assert_never

import litellm
from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig, decisions_text
from litellm.llms.openai.workload_identity import resolve_openai_bearer_token
from litellm.secret_managers.main import get_secret_str
from litellm.types.decisions import (
    DecisionsIRAnswer,
    DecisionsIRChoiceAnswer,
    DecisionsIRChoiceOption,
    DecisionsIRChoiceProbability,
    DecisionsIRChoiceQuestion,
    DecisionsIRMessages,
    DecisionsIRPredicateAnswer,
    DecisionsIRPredicateQuestion,
    DecisionsIRQuestion,
    DecisionsIRRefusal,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsIRScoreAnswer,
    DecisionsIRScoreLevel,
    DecisionsIRScoreProbability,
    DecisionsIRScoreQuestion,
    DecisionsIRState,
    DecisionsIRUsage,
    DecisionsJSON,
    OpenAIChoiceAnswer,
    OpenAIChoiceProbability,
    OpenAIChoiceQuestion,
    OpenAIDecisionAnswer,
    OpenAIDecisionInput,
    OpenAIDecisionInputTokensDetails,
    OpenAIDecisionOutputTokensDetails,
    OpenAIDecisionQuestion,
    OpenAIDecisionRequestBody,
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
    OpenAIPredicateQuestion,
    OpenAIRefusalAnswer,
    OpenAIScoreAnswer,
    OpenAIScoreProbability,
    OpenAIScoreQuestion,
    UnsupportedDecisionsRequest,
    systemone_choice_key,
)

_OPENAI_RESPONSE_ADAPTER: Final[TypeAdapter[OpenAIDecisionResponse]] = TypeAdapter(OpenAIDecisionResponse)
_SINGLE_OPTION: Final = UnsupportedDecisionsRequest(
    reason="OpenAI needs at least 2 choices or levels on every choice or score question"
)


def _ir_input(decision_input: OpenAIDecisionInput) -> DecisionsIRState | DecisionsIRMessages:
    if isinstance(decision_input, str):
        return DecisionsIRState(state=decision_input)
    return DecisionsIRMessages(messages=tuple(decision_input))


def _ir_question(question: OpenAIDecisionQuestion) -> DecisionsIRQuestion:
    match question:
        case OpenAIPredicateQuestion():
            return DecisionsIRPredicateQuestion(name=question.name, instructions=question.instructions)
        case OpenAIChoiceQuestion():
            return DecisionsIRChoiceQuestion(
                name=question.name,
                instructions=question.instructions,
                choices=tuple(
                    DecisionsIRChoiceOption(value=option.value, description=option.description)
                    for option in question.choices
                ),
            )
        case OpenAIScoreQuestion():
            return DecisionsIRScoreQuestion(
                name=question.name,
                instructions=question.instructions,
                levels=tuple(
                    DecisionsIRScoreLevel(label=level.label, description=level.description) for level in question.levels
                ),
            )
        case _:
            assert_never(question)


def openai_request_to_ir(request: OpenAIDecisionRequestBody) -> DecisionsIRRequest:
    return DecisionsIRRequest(
        input=_ir_input(request.input),
        questions=tuple(_ir_question(question) for question in request.questions),
        safety_identifier=request.safety_identifier,
    )


def _text_field(key: str, value: DecisionsJSON | None) -> Mapping[str, str]:
    return {} if value is None else {key: decisions_text(value)}


def _instructions(value: DecisionsJSON | None, default: str) -> str:
    return default if value is None else decisions_text(value)


def _predicate_instructions(question: DecisionsIRPredicateQuestion) -> str:
    instructions: Final = () if question.instructions is None else (decisions_text(question.instructions),)
    criteria: Final = tuple(
        f"Answer {answer} when: {decisions_text(rule)}"
        for answer, rule in (question.criteria or {}).items()
        if rule is not None
    )
    return "\n\n".join((*instructions, *criteria))


def _openai_question(question: DecisionsIRQuestion) -> Mapping[str, object]:
    match question:
        case DecisionsIRPredicateQuestion():
            return {
                "type": "predicate",
                **_text_field("name", question.name),
                "instructions": _predicate_instructions(question) or "Is this true of the input?",
            }
        case DecisionsIRChoiceQuestion():
            return {
                "type": "choice",
                **_text_field("name", question.name),
                "instructions": _instructions(question.instructions, "Which choice best fits the input?"),
                "choices": [
                    {"value": option.value, **_text_field("description", option.description)}
                    for option in question.choices
                ],
            }
        case DecisionsIRScoreQuestion():
            return {
                "type": "score",
                **_text_field("name", question.name),
                "instructions": _instructions(question.instructions, "Which level best fits the input?"),
                "levels": [
                    {"label": decisions_text(level.label), **_text_field("description", level.description)}
                    for level in question.levels
                ],
            }
        case _:
            assert_never(question)


def _openai_input(decision_input: DecisionsIRState | DecisionsIRMessages) -> str | Sequence[Mapping[str, object]]:
    match decision_input:
        case DecisionsIRState():
            return decisions_text(decision_input.state)
        case DecisionsIRMessages():
            return [message.model_dump(mode="json", exclude_none=True) for message in decision_input.messages]
        case _:
            assert_never(decision_input)


def _has_one_option(question: DecisionsIRQuestion) -> bool:
    match question:
        case DecisionsIRChoiceQuestion():
            return len(question.choices) < 2
        case DecisionsIRScoreQuestion():
            return len(question.levels) < 2
        case _:
            return False


def ir_to_openai_request(model: str, request: DecisionsIRRequest) -> Mapping[str, object] | UnsupportedDecisionsRequest:
    if any(_has_one_option(question) for question in request.questions):
        return _SINGLE_OPTION
    return {
        "model": model,
        "input": _openai_input(request.input),
        "questions": [_openai_question(question) for question in request.questions],
        **_text_field("safety_identifier", request.safety_identifier),
    }


def _level_label(question: DecisionsIRScoreQuestion, probability: OpenAIScoreProbability) -> DecisionsJSON:
    if 0 <= probability.value < len(question.levels):
        return question.levels[probability.value].label
    return probability.label


def _ir_answer(question: DecisionsIRQuestion, answer: OpenAIDecisionAnswer | None) -> DecisionsIRAnswer:
    match question, answer:
        case DecisionsIRPredicateQuestion(), OpenAIPredicateAnswer():
            return DecisionsIRPredicateAnswer(probability=answer.probability)
        case DecisionsIRChoiceQuestion(), OpenAIChoiceAnswer():
            return DecisionsIRChoiceAnswer(
                choice=answer.choice,
                confidence=answer.confidence,
                probabilities=tuple(
                    DecisionsIRChoiceProbability(value=item.value, probability=item.probability)
                    for item in answer.probabilities
                ),
            )
        case DecisionsIRScoreQuestion(), OpenAIScoreAnswer():
            return DecisionsIRScoreAnswer(
                score=answer.score,
                confidence=answer.confidence,
                probabilities=tuple(
                    DecisionsIRScoreProbability(
                        value=item.value, label=_level_label(question, item), probability=item.probability
                    )
                    for item in answer.probabilities
                ),
            )
        case _:
            return DecisionsIRRefusal()


def openai_response_to_ir(response: OpenAIDecisionResponse, request: DecisionsIRRequest) -> DecisionsIRResponse:
    answers: Final = response.answers
    return DecisionsIRResponse(
        model=response.model,
        answers=tuple(
            _ir_answer(question, answers[index] if index < len(answers) else None)
            for index, question in enumerate(request.questions)
        ),
        usage=DecisionsIRUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cached_tokens=response.usage.input_tokens_details.cached_tokens,
            cache_write_tokens=response.usage.input_tokens_details.cache_write_tokens,
            reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
        ),
    )


def _openai_choice_answer(question: DecisionsIRChoiceQuestion, answer: DecisionsIRChoiceAnswer) -> OpenAIChoiceAnswer:
    probabilities: Final = {systemone_choice_key(item.value): item.probability for item in answer.probabilities}
    return OpenAIChoiceAnswer(
        name=question.name,
        choice=answer.choice,
        probabilities=tuple(
            OpenAIChoiceProbability(
                value=option.value, probability=probabilities.get(systemone_choice_key(option.value), 0.0)
            )
            for option in question.choices
        ),
        confidence=answer.confidence,
    )


def _openai_score_answer(question: DecisionsIRScoreQuestion, answer: DecisionsIRScoreAnswer) -> OpenAIScoreAnswer:
    probabilities: Final = {item.value: item.probability for item in answer.probabilities}
    return OpenAIScoreAnswer(
        name=question.name,
        score=answer.score,
        probabilities=tuple(
            OpenAIScoreProbability(
                value=index, label=decisions_text(level.label), probability=probabilities.get(index, 0.0)
            )
            for index, level in enumerate(question.levels)
        ),
        confidence=answer.confidence,
    )


def _openai_answer(question: DecisionsIRQuestion, answer: DecisionsIRAnswer) -> OpenAIDecisionAnswer:
    match question, answer:
        case DecisionsIRPredicateQuestion(), DecisionsIRPredicateAnswer():
            return OpenAIPredicateAnswer(name=question.name, probability=answer.probability)
        case DecisionsIRChoiceQuestion(), DecisionsIRChoiceAnswer():
            return _openai_choice_answer(question, answer)
        case DecisionsIRScoreQuestion(), DecisionsIRScoreAnswer():
            return _openai_score_answer(question, answer)
        case _:
            return OpenAIRefusalAnswer(name=question.name)


def ir_to_openai_response(
    response: DecisionsIRResponse, request: DecisionsIRRequest, requested_model: str
) -> OpenAIDecisionResponse:
    return OpenAIDecisionResponse(
        model=response.model or requested_model,
        answers=tuple(
            _openai_answer(question, answer)
            for question, answer in zip(request.questions, response.answers, strict=True)
        ),
        usage=OpenAIDecisionUsage(
            input_tokens=response.usage.input_tokens,
            input_tokens_details=OpenAIDecisionInputTokensDetails(
                cached_tokens=response.usage.cached_tokens,
                cache_write_tokens=response.usage.cache_write_tokens,
            ),
            output_tokens=response.usage.output_tokens,
            output_tokens_details=OpenAIDecisionOutputTokensDetails(reasoning_tokens=response.usage.reasoning_tokens),
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        ),
    )


class OpenAIDecisionsConfig(BaseDecisionsConfig):
    path = "/v1/decisions"
    supports_safety_identifier = True

    def get_default_api_base(self) -> str | None:
        return "https://api.openai.com"

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or self.get_default_api_base()
        )

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")

    def resolve_credential(
        self, api_key: str | None, api_base: str, litellm_params: Mapping[str, object]
    ) -> str | None:
        return resolve_openai_bearer_token(api_key=api_key, api_base=api_base, litellm_params=litellm_params)

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        return ir_to_openai_request(self.request_model(model), request)

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        return openai_response_to_ir(_OPENAI_RESPONSE_ADAPTER.validate_python(payload), request)
