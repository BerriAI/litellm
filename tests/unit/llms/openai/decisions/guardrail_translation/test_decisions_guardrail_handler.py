from typing import Final, Literal

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.decisions.guardrail_translation.handler import DecisionsHandler, SystemOneHandler
from litellm.types.decisions import OpenAIDecisionResponse
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs


class _TaggingGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(guardrail_name="tagging")
        self.seen: list[tuple[str, tuple[str, ...]]] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        texts: Final = tuple(inputs.get("texts", []))
        self.seen.append((input_type, texts))
        return GenericGuardrailAPIInputs(texts=[f"{text} [GUARDRAILED]" for text in texts])


class _ReturnsNoTexts(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        return GenericGuardrailAPIInputs()


@pytest.mark.parametrize(
    ("call_type", "handler"),
    (
        (CallTypes.decisions, DecisionsHandler),
        (CallTypes.adecisions, DecisionsHandler),
        (CallTypes.systemone, SystemOneHandler),
        (CallTypes.asystemone, SystemOneHandler),
    ),
)
def test_decision_call_types_resolve_to_their_handler(call_type: CallTypes, handler: type[object]) -> None:
    assert get_guardrail_translation_mapping(call_type) is handler


@pytest.mark.asyncio
async def test_decision_message_text_parts_and_question_instructions_are_guardrailed() -> None:
    guardrail: Final = _TaggingGuardrail()
    image: Final = {"type": "input_image", "image_url": "data:image/png;base64,AA=="}
    data: Final = {
        "model": "judge",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "ticket one"}, image]},
            {"role": "user", "content": "ticket two"},
        ],
        "questions": [
            {"type": "predicate", "name": "urgent", "instructions": "Is it urgent?"},
            {"type": "box", "name": "logo", "instructions": "ticket one"},
        ],
    }

    result: Final = await DecisionsHandler().process_input_messages(data, guardrail)

    assert guardrail.seen == [("request", ("ticket one", "ticket two", "Is it urgent?"))]
    assert {key: value for key, value in result.items() if key != "metadata"} == {
        "model": "judge",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "ticket one [GUARDRAILED]"}, image]},
            {"role": "user", "content": "ticket two [GUARDRAILED]"},
        ],
        "questions": [
            {"type": "predicate", "name": "urgent", "instructions": "Is it urgent? [GUARDRAILED]"},
            {"type": "box", "name": "logo", "instructions": "ticket one [GUARDRAILED]"},
        ],
    }
    assert data["input"][0]["content"][0]["text"] == "ticket one"


@pytest.mark.asyncio
async def test_systemone_state_leaves_and_question_instructions_are_guardrailed() -> None:
    guardrail: Final = _TaggingGuardrail()
    criteria_question: Final = {"type": "noul", "criteria": {"yes": "refund"}}
    data: Final = {
        "model": "judge",
        "state": {"subject": "refund", "lines": ["order 7", "", 3, True], "meta": {"id": None}},
        "questions": {
            "refund": {"type": "noul", "instructions": {"ask": "Does the customer want a refund?"}},
            "routed": criteria_question,
        },
    }

    result: Final = await SystemOneHandler().process_input_messages(data, guardrail)

    assert guardrail.seen == [("request", ("refund", "order 7", "Does the customer want a refund?"))]
    assert {key: value for key, value in result.items() if key != "metadata"} == {
        "model": "judge",
        "state": {
            "subject": "refund [GUARDRAILED]",
            "lines": ["order 7 [GUARDRAILED]", "", 3, True],
            "meta": {"id": None},
        },
        "questions": {
            "refund": {"type": "noul", "instructions": {"ask": "Does the customer want a refund? [GUARDRAILED]"}},
            "routed": criteria_question,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "data"),
    (
        (
            SystemOneHandler,
            {"model": "judge", "state": {"count": 3}, "questions": {"q": {"type": "noul", "criteria": {}}}},
        ),
        (
            DecisionsHandler,
            {
                "model": "judge",
                "input": [{"role": "user", "content": ""}],
                "questions": [{"type": "predicate", "instructions": ""}],
            },
        ),
    ),
)
async def test_request_without_text_is_returned_unscanned(
    handler: type[DecisionsHandler] | type[SystemOneHandler], data: dict[str, object]
) -> None:
    guardrail: Final = _TaggingGuardrail()

    result: Final = await handler().process_input_messages(data, guardrail)

    assert result is data
    assert guardrail.seen == []


@pytest.mark.asyncio
async def test_guardrail_that_returns_no_texts_leaves_the_request_text_as_sent() -> None:
    data: Final = {
        "model": "judge",
        "input": "ticket",
        "questions": [{"type": "predicate", "instructions": "Is it urgent?"}],
    }

    result: Final = await DecisionsHandler().process_input_messages(data, _ReturnsNoTexts(guardrail_name="noop"))

    assert result == data


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", (DecisionsHandler, SystemOneHandler))
async def test_decision_answers_are_returned_as_the_provider_sent_them(
    handler: type[DecisionsHandler] | type[SystemOneHandler],
) -> None:
    guardrail: Final = _TaggingGuardrail()
    response: Final = OpenAIDecisionResponse.model_validate(
        {
            "model": "judge",
            "answers": [{"type": "refusal", "name": "q"}],
            "usage": {"input_tokens": 1, "output_tokens": 0, "total_tokens": 1},
        }
    )

    result: Final = await handler().process_output_response(response, guardrail)

    assert result is response
    assert guardrail.seen == []
