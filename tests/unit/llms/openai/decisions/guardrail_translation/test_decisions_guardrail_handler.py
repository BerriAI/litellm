from functools import reduce
from typing import Final, Literal

import pytest
from pydantic import JsonValue

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.prompt_templates.common_utils import request_contains_image_content
from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.base_llm.guardrail_translation.utils import UnappliableRequestRewrite
from litellm.llms.openai.decisions.guardrail_translation.handler import DecisionsHandler, SystemOneHandler
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.types.guardrails import ContentFilterAction, ContentFilterPattern
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


class _MasksSecrets(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        secrets: Final = (
            ("AKIAIOSFODNN7EXAMPLE", "<AWS_KEY>"),
            ("AKIAI44QH8DHBEXAMPLE", "<AWS_KEY>"),
            ("4111111111111111", "<CARD>"),
        )
        return GenericGuardrailAPIInputs(
            texts=[reduce(lambda text, secret: text.replace(*secret), secrets, text) for text in inputs.get("texts", [])]
        )


class _DropsQuotes(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        return GenericGuardrailAPIInputs(texts=[text.replace('"', "") for text in inputs.get("texts", [])])


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
async def test_decision_message_text_parts_and_question_text_are_guardrailed() -> None:
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
            {
                "type": "choice",
                "name": "team",
                "instructions": "Which team?",
                "choices": [{"value": "billing", "description": "Payment issues"}, {"value": True}],
            },
            {
                "type": "score",
                "name": "mood",
                "instructions": "How upset?",
                "levels": [{"label": "calm"}, {"label": "angry", "description": "Uses caps"}],
            },
        ],
    }

    result: Final = await DecisionsHandler().process_input_messages(data, guardrail)

    assert guardrail.seen == [
        (
            "request",
            ("ticket one", "ticket two", "Is it urgent?", "Which team?", "Payment issues", "How upset?", "Uses caps"),
        )
    ]
    assert {key: value for key, value in result.items() if key != "metadata"} == {
        "model": "judge",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "ticket one [GUARDRAILED]"}, image]},
            {"role": "user", "content": "ticket two [GUARDRAILED]"},
        ],
        "questions": [
            {"type": "predicate", "name": "urgent", "instructions": "Is it urgent? [GUARDRAILED]"},
            {"type": "box", "name": "logo", "instructions": "ticket one [GUARDRAILED]"},
            {
                "type": "choice",
                "name": "team",
                "instructions": "Which team? [GUARDRAILED]",
                "choices": [{"value": "billing", "description": "Payment issues [GUARDRAILED]"}, {"value": True}],
            },
            {
                "type": "score",
                "name": "mood",
                "instructions": "How upset? [GUARDRAILED]",
                "levels": [{"label": "calm"}, {"label": "angry", "description": "Uses caps [GUARDRAILED]"}],
            },
        ],
    }
    assert data["input"][0]["content"][0]["text"] == "ticket one"


@pytest.mark.asyncio
async def test_systemone_state_keys_scalars_and_question_text_are_guardrailed() -> None:
    guardrail: Final = _TaggingGuardrail()
    data: Final = {
        "model": "judge",
        "state": {"subject": "refund", "lines": ["order 7", "", 3, True], "meta": {"id": None}},
        "questions": {
            "refund": {"type": "noul", "instructions": {"ask": "Does the customer want a refund?"}},
            "routed": {"type": "noul", "criteria": {"true": "The customer asks for money back", "false": None}},
            "tier": {"type": "score", "criteria": ["calm", {"level": "angry"}]},
        },
    }

    result: Final = await SystemOneHandler().process_input_messages(data, guardrail)

    assert guardrail.seen == [
        (
            "request",
            (
                '"subject": "refund"',
                "lines",
                "order 7",
                "3",
                "meta",
                '"id": null',
                '"ask": "Does the customer want a refund?"',
                "The customer asks for money back",
                "calm",
                '"level": "angry"',
            ),
        )
    ]
    assert {key: value for key, value in result.items() if key != "metadata"} == {
        "model": "judge",
        "state": {
            "subject": '"refund" [GUARDRAILED]',
            "lines [GUARDRAILED]": ["order 7 [GUARDRAILED]", "", "3 [GUARDRAILED]", True],
            "meta [GUARDRAILED]": {"id": "null [GUARDRAILED]"},
        },
        "questions": {
            "refund": {
                "type": "noul",
                "instructions": {"ask": '"Does the customer want a refund?" [GUARDRAILED]'},
            },
            "routed": {
                "type": "noul",
                "criteria": {"true": "The customer asks for money back [GUARDRAILED]", "false": None},
            },
            "tier": {"type": "score", "criteria": ["calm [GUARDRAILED]", {"level": '"angry" [GUARDRAILED]'}]},
        },
    }


@pytest.mark.asyncio
async def test_systemone_secret_key_and_card_number_are_masked_before_the_provider() -> None:
    data: Final = {
        "model": "judge",
        "state": {"AKIAIOSFODNN7EXAMPLE": "<X>", "card": 4111111111111111, "items": 2},
        "questions": {"q": {"type": "noul", "criteria": {}}},
    }

    result: Final = await SystemOneHandler().process_input_messages(data, _MasksSecrets(guardrail_name="mask"))

    assert result["state"] == {"<AWS_KEY>": "<X>", "card": "<CARD>", "items": 2}


@pytest.mark.asyncio
async def test_systemone_masks_a_card_number_that_json_escaping_splits() -> None:
    guardrail: Final = ContentFilterGuardrail(
        guardrail_name="cards",
        patterns=[ContentFilterPattern(pattern_type="prebuilt", pattern_name="visa", action=ContentFilterAction.MASK)],
    )
    data: Final = {
        "model": "judge",
        "state": {"card": "4111\t1111\t1111\t1111", "4111\n1111\n1111\n1111": "on file", "note": "first\nsecond"},
        "questions": {"q": {"type": "noul", "criteria": {}}},
    }

    result: Final = await SystemOneHandler().process_input_messages(data, guardrail)

    assert result["state"] == {"card": "[VISA_REDACTED]", "[VISA_REDACTED]": "on file", "note": "first\nsecond"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pattern_name", "state", "masked"),
    (
        pytest.param(
            "visa",
            {"4111111111111111": "4111\t1111\t1111\t1111"},
            {"[VISA_REDACTED]": "[VISA_REDACTED]"},
            id="key-and-escaped-value",
        ),
        pytest.param(
            "email",
            {"note": "Please call me back.\ndana.quill@example.com"},
            {"note": "Please call me back.\n[EMAIL_REDACTED]"},
            id="match-after-an-escape",
        ),
    ),
)
async def test_systemone_masks_findings_on_both_sides_of_a_json_escape(
    pattern_name: str, state: dict[str, JsonValue], masked: dict[str, JsonValue]
) -> None:
    guardrail: Final = ContentFilterGuardrail(
        guardrail_name="mask",
        patterns=[
            ContentFilterPattern(pattern_type="prebuilt", pattern_name=pattern_name, action=ContentFilterAction.MASK)
        ],
    )
    data: Final = {"model": "judge", "state": state, "questions": {"q": {"type": "noul", "criteria": {}}}}

    result: Final = await SystemOneHandler().process_input_messages(data, guardrail)

    assert result["state"] == masked


@pytest.mark.asyncio
async def test_systemone_keeps_every_state_entry_when_masked_keys_collide() -> None:
    data: Final = {
        "model": "judge",
        "state": {"AKIAIOSFODNN7EXAMPLE": "first", "items": 2, "AKIAI44QH8DHBEXAMPLE": "second"},
        "questions": {"q": {"type": "noul", "criteria": {}}},
    }

    result: Final = await SystemOneHandler().process_input_messages(data, _MasksSecrets(guardrail_name="mask"))

    assert result["state"] == {"<AWS_KEY>": "first", "items": 2, "<AWS_KEY>_2": "second"}


@pytest.mark.asyncio
async def test_systemone_rejects_a_rewrite_that_loses_the_state_key() -> None:
    data: Final = {
        "model": "judge",
        "state": {"ticket": "refund order 7"},
        "questions": {"q": {"type": "noul", "criteria": {}}},
    }

    with pytest.raises(UnappliableRequestRewrite):
        await SystemOneHandler().process_input_messages(data, _DropsQuotes(guardrail_name="drops"))


@pytest.mark.asyncio
async def test_systemone_rejects_masked_keys_that_stay_ambiguous() -> None:
    data: Final = {
        "model": "judge",
        "state": {"AKIAIOSFODNN7EXAMPLE": "first", "AKIAI44QH8DHBEXAMPLE": "second", "<AWS_KEY>_2": "third"},
        "questions": {"q": {"type": "noul", "criteria": {}}},
    }

    with pytest.raises(UnappliableRequestRewrite):
        await SystemOneHandler().process_input_messages(data, _MasksSecrets(guardrail_name="mask"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "data"),
    (
        (
            SystemOneHandler,
            {"model": "judge", "state": {}, "questions": {"q": {"type": "noul", "criteria": {}}}},
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
@pytest.mark.parametrize(
    ("handler", "data"),
    (
        (
            DecisionsHandler,
            {"model": "judge", "input": "ticket", "questions": [{"type": "predicate", "instructions": "Is it urgent?"}]},
        ),
        (
            SystemOneHandler,
            {"model": "judge", "state": {"total": 3, "paid": 2.5}, "questions": {"q": {"type": "noul", "criteria": {}}}},
        ),
    ),
)
async def test_guardrail_that_returns_no_texts_leaves_the_request_text_as_sent(
    handler: type[DecisionsHandler] | type[SystemOneHandler], data: dict[str, object]
) -> None:
    result: Final = await handler().process_input_messages(data, _ReturnsNoTexts(guardrail_name="noop"))

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


def test_router_sees_the_image_in_a_decision_input_routed_by_its_proxy_route() -> None:
    structured: Final = resolve_structured_messages(
        messages=None,
        request_kwargs={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Is this a cat?"},
                        {"type": "input_image", "image_url": "https://example.com/cat.png"},
                    ],
                }
            ],
            "litellm_metadata": {"user_api_key_request_route": "/v1/decisions"},
        },
    )

    assert structured is not None and request_contains_image_content(structured)
